from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

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


FillMode = Literal["zero", "block0"]
QueryScope = Literal["patch", "all", "class"]


@dataclass(frozen=True)
class Block0AttentionMassPruneConfig:
    dataset_root: Path
    checkpoint: Path
    output_json: Path
    output_md: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    mass_thresholds: tuple[float, ...] = (0.99, 0.98, 0.95, 0.90, 0.85, 0.80)
    query_scope: QueryScope = "patch"
    fill_mode: FillMode = "zero"
    min_keep_tokens: int = 1
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
        if not self.mass_thresholds:
            raise ValueError("at least one mass threshold is required")
        for threshold in self.mass_thresholds:
            if not 0.0 < threshold <= 1.0:
                raise ValueError(f"mass thresholds must be in (0, 1], got {threshold}")
        if self.query_scope not in {"patch", "all", "class"}:
            raise ValueError("query_scope must be patch, all, or class")
        if self.fill_mode not in {"zero", "block0"}:
            raise ValueError("fill_mode must be zero or block0")
        if self.min_keep_tokens <= 0:
            raise ValueError("min_keep_tokens must be positive")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


def _parse_thresholds(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _threshold_key(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _attention_probs(attn: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    batch, token_count, channels = x.shape
    head_count = int(attn.num_heads)
    head_dim = channels // head_count
    qkv = attn.qkv(x).reshape(batch, token_count, 3, head_count, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q = qkv[0] * attn.scale
    k = qkv[1]
    return (q @ k.transpose(-2, -1)).softmax(dim=-1)


def _incoming_patch_attention_mass(
    probs: torch.Tensor,
    *,
    patch_start: int,
    patch_count: int,
    query_scope: QueryScope,
) -> torch.Tensor:
    if probs.ndim != 4 or probs.shape[0] != 1:
        raise ValueError("attention probabilities must have shape [1, heads, tokens, tokens]")
    patch_slice = slice(patch_start, patch_start + patch_count)
    if query_scope == "patch":
        mass = probs[0, :, patch_slice, patch_slice].mean(dim=(0, 1))
    elif query_scope == "all":
        mass = probs[0, :, :, patch_slice].mean(dim=(0, 1))
    elif query_scope == "class":
        mass = probs[0, :, 0, patch_slice].mean(dim=0)
    else:
        raise ValueError(f"unknown query scope: {query_scope}")
    mass = mass.float().clamp_min(0)
    total = mass.sum()
    if float(total.item()) <= 0:
        return torch.full_like(mass, 1.0 / max(mass.numel(), 1))
    return mass / total


def _kept_indices_for_mass(
    mass: torch.Tensor,
    *,
    threshold: float,
    min_keep_tokens: int,
) -> torch.Tensor:
    if threshold >= 1.0:
        return torch.arange(mass.numel(), dtype=torch.long, device=mass.device)
    order = torch.argsort(-mass, stable=True)
    cumulative = torch.cumsum(mass[order], dim=0)
    keep_count = int(torch.searchsorted(cumulative, torch.tensor(threshold, device=mass.device), right=False).item()) + 1
    keep_count = min(max(keep_count, min_keep_tokens), mass.numel())
    return torch.sort(order[:keep_count].to(dtype=torch.long))[0]


def block0_tokens_and_mass(
    vit: torch.nn.Module,
    x: torch.Tensor,
    *,
    query_scope: QueryScope,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    if x.shape[0] != 1:
        raise ValueError("block0 attention pruning currently supports batch size 1")
    tokens = vit.prepare_tokens_with_masks(x)
    register_count = int(getattr(vit, "num_register_tokens", 0))
    patch_start = 1 + register_count
    patch_count = tokens.shape[1] - patch_start
    block0 = vit.blocks[0]
    attn_input = block0.norm1(tokens)
    probs = _attention_probs(block0.attn, attn_input)
    mass = _incoming_patch_attention_mass(
        probs,
        patch_start=patch_start,
        patch_count=patch_count,
        query_scope=query_scope,
    )
    return block0(tokens), mass, patch_start, patch_count, register_count


def get_block0_attention_pruned_intermediate_layers(
    vit: torch.nn.Module,
    x: torch.Tensor,
    layers: list[int],
    *,
    mass_threshold: float,
    query_scope: QueryScope,
    fill_mode: FillMode,
    min_keep_tokens: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], int, int]:
    block0_output, mass, patch_start, patch_count, _register_count = block0_tokens_and_mass(
        vit,
        x,
        query_scope=query_scope,
    )
    kept_indices = _kept_indices_for_mass(
        mass,
        threshold=mass_threshold,
        min_keep_tokens=min_keep_tokens,
    ).to(device=x.device)
    special_tokens = block0_output[:, :patch_start]
    kept_patch_tokens = block0_output[:, patch_start:][:, kept_indices]
    sequence = torch.cat((special_tokens, kept_patch_tokens), dim=1)

    layers_to_take = set(layers)
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    if 0 in layers_to_take:
        normalized = vit.norm(block0_output)
        outputs.append((normalized[:, patch_start:], normalized[:, 0]))

    for block_index in range(1, len(vit.blocks)):
        sequence = vit.blocks[block_index](sequence)
        if block_index not in layers_to_take:
            continue
        normalized = vit.norm(sequence)
        class_token = normalized[:, 0]
        kept_tokens = normalized[:, patch_start:]
        if fill_mode == "block0":
            restored = vit.norm(block0_output[:, patch_start:]).clone()
        else:
            restored = torch.zeros(
                (1, patch_count, normalized.shape[-1]),
                dtype=normalized.dtype,
                device=normalized.device,
            )
        restored[:, kept_indices] = kept_tokens
        outputs.append((restored, class_token))

    if len(outputs) != len(layers):
        raise RuntimeError(f"only captured {len(outputs)} / {len(layers)} requested layers")
    return tuple(outputs), int(kept_indices.numel()), patch_count


@torch.no_grad()
def infer_depth(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    mass_threshold: float,
    query_scope: QueryScope,
    fill_mode: FillMode,
    min_keep_tokens: int,
    raw_height: int,
    raw_width: int,
) -> tuple[torch.Tensor, int, int]:
    patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
    layers = model.intermediate_layer_idx[model.encoder]
    features, kept_count, patch_count = get_block0_attention_pruned_intermediate_layers(
        model.pretrained,
        x,
        layers,
        mass_threshold=mass_threshold,
        query_scope=query_scope,
        fill_mode=fill_mode,
        min_keep_tokens=min_keep_tokens,
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


def _add_token_stats(stats: dict[str, float], kept_count: int, patch_count: int) -> None:
    ratio = kept_count / max(patch_count, 1)
    stats["images"] += 1
    stats["kept_sum"] += kept_count
    stats["kept_min"] = min(stats["kept_min"], kept_count)
    stats["kept_max"] = max(stats["kept_max"], kept_count)
    stats["ratio_sum"] += ratio
    stats["ratio_min"] = min(stats["ratio_min"], ratio)
    stats["ratio_max"] = max(stats["ratio_max"], ratio)


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
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Block-0 Attention-Mass Token Pruning",
        "",
        f"- Images: {result['metadata']['images_evaluated']}/{result['metadata']['images_requested']}",
        f"- Query scope: `{result['metadata']['query_scope']}`",
        f"- Fill mode: `{result['metadata']['fill_mode']}`",
        "",
        "| mass threshold | accuracy | correct | mean kept tokens | mean keep ratio | min-max kept |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for key, summary in result["thresholds"].items():
        overall = summary["overall"]
        token_stats = summary["token_stats"]
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    f"{overall['best_accuracy']:.6f}",
                    f"{overall['larger_correct']}/{overall['pairs']}",
                    f"{token_stats['kept_mean']:.2f}",
                    f"{token_stats['keep_ratio_mean']:.4f}",
                    f"{token_stats['kept_min']}-{token_stats['kept_max']}",
                ]
            )
            + " |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def evaluate(config: Block0AttentionMassPruneConfig) -> dict[str, Any]:
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

    thresholds = {_threshold_key(threshold): threshold for threshold in config.mass_thresholds}
    totals = {key: empty_counts() for key in thresholds}
    by_scene = {key: defaultdict(empty_counts) for key in thresholds}
    token_stats = {key: _new_token_stats() for key in thresholds}
    missing_images: list[str] = []
    started = time.monotonic()
    images_evaluated = 0

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
            depths: dict[str, torch.Tensor] = {}
            scene = scene_from_path(relative_path)
            for key, threshold in thresholds.items():
                depth, kept_count, patch_count = infer_depth(
                    model,
                    x,
                    mass_threshold=threshold,
                    query_scope=config.query_scope,
                    fill_mode=config.fill_mode,
                    min_keep_tokens=config.min_keep_tokens,
                    raw_height=raw_height,
                    raw_width=raw_width,
                )
                depths[key] = depth
                _add_token_stats(token_stats[key], kept_count, patch_count)

            for pair in pairs:
                if pair.get("closer_point") != "point1":
                    raise ValueError(f"unsupported closer_point in {relative_path}: {pair}")
                for key, depth in depths.items():
                    d1 = point_value(depth, pair["point1"])
                    d2 = point_value(depth, pair["point2"])
                    add_pair(totals[key], d1, d2)
                    add_pair(by_scene[key][scene], d1, d2)
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
            "rule": "DA-2K labels point1 as closer; Depth Anything V2 vits uses larger predicted values for closer points.",
        },
        "thresholds": {
            key: {
                "mass_threshold": threshold,
                "overall": finalize_counts(totals[key]),
                "by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(by_scene[key].items())},
                "token_stats": _finalize_token_stats(token_stats[key]),
            }
            for key, threshold in thresholds.items()
        },
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    _write_markdown(result, config.output_md)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pruning after block 0 using cumulative attention mass.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/da2k_vits_block0_attention_mass_prune.json"))
    parser.add_argument("--output-md", type=Path, default=Path("eval_outputs/da2k_vits_block0_attention_mass_prune.md"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mass-thresholds", type=_parse_thresholds, default=(0.99, 0.98, 0.95, 0.90, 0.85, 0.80))
    parser.add_argument("--query-scope", choices=["patch", "all", "class"], default="patch")
    parser.add_argument("--fill-mode", choices=["zero", "block0"], default="zero")
    parser.add_argument("--min-keep-tokens", type=int, default=1)
    parser.add_argument("--scene-type", choices=SCENE_CHOICES, default="")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = Block0AttentionMassPruneConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        output_md=args.output_md,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        mass_thresholds=args.mass_thresholds,
        query_scope=args.query_scope,
        fill_mode=args.fill_mode,
        min_keep_tokens=args.min_keep_tokens,
        scene_type=args.scene_type,
        max_images=args.max_images,
        log_every=args.log_every,
    )
    result = evaluate(config)
    print(
        json.dumps(
            {
                key: {
                    "overall": summary["overall"],
                    "token_stats": summary["token_stats"],
                }
                for key, summary in result["thresholds"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {config.output_json}")


if __name__ == "__main__":
    main()
