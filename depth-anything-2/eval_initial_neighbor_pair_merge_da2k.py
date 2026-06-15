from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from eval_da2k import (
    MODEL_CONFIGS,
    SCENE_CHOICES,
    add_pair,
    empty_counts,
    finalize_counts,
    load_cv2,
    load_model,
    point_value,
    require_ready,
    resolve_device,
    scene_from_path,
    selected_annotations,
)
from eval_tome_da2k import _merge_wavg, _prepare_tome_tokens, _proportional_attention, _restore_patch_grid


PAIR_MODES = ("snake", "horizontal", "vertical")


@dataclass(frozen=True)
class InitialPairMergeConfig:
    dataset_root: Path
    checkpoint: Path
    output_json: Path
    output_md: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    pair_modes: tuple[str, ...] = ("snake", "horizontal", "vertical")
    proportional_attention: bool = False
    scene_type: str = ""
    max_images: int = 0
    log_every: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_json", Path(self.output_json))
        object.__setattr__(self, "output_md", Path(self.output_md))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if not self.pair_modes:
            raise ValueError("at least one pair mode is required")
        for mode in self.pair_modes:
            if mode not in PAIR_MODES:
                raise ValueError(f"unknown pair mode: {mode}")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


def _parse_pair_modes(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _snake_order(patch_h: int, patch_w: int) -> list[int]:
    order: list[int] = []
    for row in range(patch_h):
        columns = range(patch_w) if row % 2 == 0 else range(patch_w - 1, -1, -1)
        for col in columns:
            order.append(row * patch_w + col)
    return order


def _initial_neighbor_pairs(*, patch_h: int, patch_w: int, mode: str) -> list[tuple[int, int]]:
    if mode == "snake":
        order = _snake_order(patch_h, patch_w)
        return [(order[index + 1], order[index]) for index in range(0, len(order) - 1, 2)]
    if mode == "horizontal":
        return [
            (row * patch_w + col + 1, row * patch_w + col)
            for row in range(patch_h)
            for col in range(0, patch_w - 1, 2)
        ]
    if mode == "vertical":
        return [
            ((row + 1) * patch_w + col, row * patch_w + col)
            for col in range(patch_w)
            for row in range(0, patch_h - 1, 2)
        ]
    raise ValueError(f"unknown pair mode: {mode}")


def _merge_pairs_once(
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    pairs: list[tuple[int, int]],
    *,
    special_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if not pairs:
        return x, sizes, sources, 0

    patch_count = x.shape[1] - special_count
    source_locals = {int(src) for src, _dst in pairs}
    pairs = [
        (int(src), int(dst))
        for src, dst in pairs
        if 0 <= int(src) < patch_count
        and 0 <= int(dst) < patch_count
        and int(src) != int(dst)
        and int(dst) not in source_locals
    ]
    if not pairs:
        return x, sizes, sources, 0

    source_locals = {src for src, _dst in pairs}
    keep_patch_locals = [index for index in range(patch_count) if index not in source_locals]
    keep_idx = torch.as_tensor(
        [*range(special_count), *[special_count + index for index in keep_patch_locals]],
        dtype=torch.long,
        device=x.device,
    )
    new_x = x.index_select(1, keep_idx).clone()
    new_sizes = sizes.index_select(1, keep_idx).clone()
    new_sources = sources.index_select(1, keep_idx).clone()
    old_to_new = {old: new for new, old in enumerate(keep_patch_locals)}

    merged = 0
    for src_local, dst_local in pairs:
        dst_new_local = old_to_new.get(dst_local)
        if dst_new_local is None:
            continue
        src_idx = special_count + src_local
        dst_new_idx = special_count + dst_new_local
        src_size = sizes[:, src_idx]
        dst_size = new_sizes[:, dst_new_idx]
        new_x[:, dst_new_idx] = _merge_wavg(new_x[:, dst_new_idx], x[:, src_idx], dst_size, src_size)
        new_sizes[:, dst_new_idx] = dst_size + src_size
        new_sources[:, dst_new_idx] = new_sources[:, dst_new_idx] + sources[:, src_idx]
        merged += 1

    return new_x, new_sizes, new_sources, merged


def _run_block(
    block: torch.nn.Module,
    x: torch.Tensor,
    sizes: torch.Tensor,
    *,
    proportional_attention: bool,
) -> torch.Tensor:
    if not proportional_attention:
        return block(x)
    attn_out, _metric = _proportional_attention(block.attn, block.norm1(x), sizes)
    x = x + block.drop_path1(block.ls1(attn_out))
    x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
    return x


def get_initial_pair_merge_intermediate_layers(
    vit: torch.nn.Module,
    x: torch.Tensor,
    layers: list[int],
    *,
    pair_mode: str,
    proportional_attention: bool,
) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], int, int]:
    sequence, sizes, sources, patch_count, special_count = _prepare_tome_tokens(vit, x)
    patch_h = int(x.shape[-2] // vit.patch_size)
    patch_w = int(x.shape[-1] // vit.patch_size)
    pairs = _initial_neighbor_pairs(patch_h=patch_h, patch_w=patch_w, mode=pair_mode)
    sequence, sizes, sources, merged_count = _merge_pairs_once(
        sequence,
        sizes,
        sources,
        pairs,
        special_count=special_count,
    )

    layers_to_take = set(layers)
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for block_index, block in enumerate(vit.blocks):
        sequence = _run_block(
            block,
            sequence,
            sizes,
            proportional_attention=proportional_attention,
        )
        if block_index not in layers_to_take:
            continue
        outputs.append(
            _restore_patch_grid(
                vit.norm(sequence),
                sources,
                patch_count=patch_count,
                special_count=special_count,
            )
        )

    if len(outputs) != len(layers):
        raise RuntimeError(f"only captured {len(outputs)} / {len(layers)} requested layers")
    kept_count = patch_count - merged_count
    return tuple(outputs), kept_count, patch_count


@torch.no_grad()
def infer_depth(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    pair_mode: str,
    proportional_attention: bool,
    raw_height: int,
    raw_width: int,
) -> tuple[torch.Tensor, int, int]:
    patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
    layers = model.intermediate_layer_idx[model.encoder]
    features, kept_count, patch_count = get_initial_pair_merge_intermediate_layers(
        model.pretrained,
        x,
        layers,
        pair_mode=pair_mode,
        proportional_attention=proportional_attention,
    )
    depth = model.depth_head(features, patch_h, patch_w)
    depth = F.relu(depth).squeeze(1)
    depth = F.interpolate(depth[:, None], (raw_height, raw_width), mode="bilinear", align_corners=True)[0, 0]
    return depth.detach().float().cpu(), kept_count, patch_count


def _new_token_stats() -> dict[str, float]:
    return {
        "images": 0,
        "kept_sum": 0.0,
        "kept_min": float("inf"),
        "kept_max": 0.0,
        "ratio_sum": 0.0,
        "ratio_min": float("inf"),
        "ratio_max": 0.0,
    }


def _add_token_stats(stats: dict[str, float], *, kept_count: int, patch_count: int) -> None:
    keep_ratio = kept_count / max(patch_count, 1)
    stats["images"] += 1
    stats["kept_sum"] += kept_count
    stats["kept_min"] = min(stats["kept_min"], kept_count)
    stats["kept_max"] = max(stats["kept_max"], kept_count)
    stats["ratio_sum"] += keep_ratio
    stats["ratio_min"] = min(stats["ratio_min"], keep_ratio)
    stats["ratio_max"] = max(stats["ratio_max"], keep_ratio)


def _finalize_token_stats(stats: dict[str, float]) -> dict[str, float | int]:
    images = max(int(stats["images"]), 1)
    return {
        "images": int(stats["images"]),
        "kept_mean": stats["kept_sum"] / images,
        "kept_min": int(stats["kept_min"]) if stats["kept_min"] != float("inf") else 0,
        "kept_max": int(stats["kept_max"]),
        "keep_ratio_mean": stats["ratio_sum"] / images,
        "keep_ratio_min": stats["ratio_min"] if stats["ratio_min"] != float("inf") else 0.0,
        "keep_ratio_max": stats["ratio_max"],
        "approx_transformer_token_work_ratio": stats["ratio_sum"] / images,
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Initial Neighbor Pair Token Merge",
        "",
        f"- Images: {result['metadata']['images_evaluated']}/{result['metadata']['images_requested']}",
        f"- Proportional attention: `{result['metadata']['proportional_attention']}`",
        "- Merge point: before transformer block 0",
        "- Merged features are expanded back to the full patch grid before the DPT head.",
        "",
        "| pair mode | accuracy | correct | mean kept | keep ratio | approx token-work ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, summary in result["pair_modes"].items():
        overall = summary["overall"]
        token_stats = summary["token_stats"]
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    f"{overall['best_accuracy']:.6f}",
                    f"{overall['larger_correct']}/{overall['pairs']}",
                    f"{token_stats['kept_mean']:.1f}",
                    f"{token_stats['keep_ratio_mean']:.4f}",
                    f"{token_stats['approx_transformer_token_work_ratio']:.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def evaluate(config: InitialPairMergeConfig) -> dict[str, Any]:
    require_ready(config.dataset_root, config.checkpoint, config.encoder)
    cv2 = load_cv2()
    device = resolve_device(config.device)
    model = load_model(config.encoder, config.checkpoint, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    selected = selected_annotations(config.dataset_root, scene_type=config.scene_type, max_images=config.max_images)
    if not selected:
        raise RuntimeError("no DA-2K annotations selected")

    cells = {
        mode: {
            "total": empty_counts(),
            "by_scene": defaultdict(empty_counts),
            "token_stats": _new_token_stats(),
        }
        for mode in config.pair_modes
    }
    missing_images: list[str] = []
    images_evaluated = 0
    started = time.monotonic()

    with torch.inference_mode():
        for index, (relative_path, pairs) in enumerate(selected, start=1):
            image_path = config.dataset_root / relative_path
            image = cv2.imread(str(image_path))
            if image is None:
                missing_images.append(str(image_path))
                continue
            raw_height, raw_width = image.shape[:2]
            x, _ = model.image2tensor(image, config.input_size)
            x = x.to(device)
            scene = scene_from_path(relative_path)

            for mode in config.pair_modes:
                depth, kept_count, patch_count = infer_depth(
                    model,
                    x,
                    pair_mode=mode,
                    proportional_attention=config.proportional_attention,
                    raw_height=raw_height,
                    raw_width=raw_width,
                )
                cell = cells[mode]
                _add_token_stats(cell["token_stats"], kept_count=kept_count, patch_count=patch_count)
                for pair in pairs:
                    if pair.get("closer_point") != "point1":
                        raise ValueError(f"unsupported closer_point in {relative_path}: {pair}")
                    d1 = point_value(depth, pair["point1"])
                    d2 = point_value(depth, pair["point2"])
                    add_pair(cell["total"], d1, d2)
                    add_pair(cell["by_scene"][scene], d1, d2)

            images_evaluated += 1
            if config.log_every > 0 and (index % config.log_every == 0 or index == len(selected)):
                print(f"evaluated {index}/{len(selected)} images in {time.monotonic() - started:.1f}s", flush=True)

    result = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_requested": len(selected),
            "images_evaluated": images_evaluated,
            "missing_images": missing_images,
            "elapsed_seconds": time.monotonic() - started,
            "patch_count_at_square_input": (config.input_size // 14) * (config.input_size // 14),
            "merge_point": "before transformer block 0",
            "rule": "DA-2K labels point1 as closer; Depth Anything V2 vits uses larger predicted values for closer points.",
        },
        "pair_modes": {},
    }
    for mode in config.pair_modes:
        cell = cells[mode]
        result["pair_modes"][mode] = {
            "overall": finalize_counts(cell["total"]),
            "by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(cell["by_scene"].items())},
            "token_stats": _finalize_token_stats(cell["token_stats"]),
        }

    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    _write_markdown(result, config.output_md)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one-shot initial neighboring-pair token merging on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/da2k_vits_initial_neighbor_pair_merge.json"))
    parser.add_argument("--output-md", type=Path, default=Path("eval_outputs/da2k_vits_initial_neighbor_pair_merge.md"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pair-modes", type=_parse_pair_modes, default=("snake", "horizontal", "vertical"))
    parser.add_argument("--proportional-attention", action="store_true")
    parser.add_argument("--scene-type", choices=SCENE_CHOICES, default="")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = InitialPairMergeConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        output_md=args.output_md,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        pair_modes=args.pair_modes,
        proportional_attention=args.proportional_attention,
        scene_type=args.scene_type,
        max_images=args.max_images,
        log_every=args.log_every,
    )
    result = evaluate(config)
    compact = {
        mode: {
            "accuracy": summary["overall"]["best_accuracy"],
            "kept_ratio": summary["token_stats"]["keep_ratio_mean"],
            "work_ratio": summary["token_stats"]["approx_transformer_token_work_ratio"],
        }
        for mode, summary in result["pair_modes"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    print(f"wrote {config.output_json}")


if __name__ == "__main__":
    main()
