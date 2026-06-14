from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from eval_da2k import (
    MODEL_CONFIGS,
    SCENE_CHOICES,
    SetupError,
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
from eval_tome_da2k import (
    _merge_wavg,
    _prepare_tome_tokens,
    _proportional_attention,
    _restore_patch_grid,
)


PROXY_MODES = ("local-horizontal", "local-vertical", "local-checkerboard")


@dataclass(frozen=True)
class DenseSafeProxyConfig:
    dataset_root: Path
    checkpoint: Path
    output_json: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    merge_r: int = 57
    proxy_mode: str = "local-horizontal"
    scene_type: str = ""
    max_images: int = 0
    log_every: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_json", Path(self.output_json))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.merge_r < 0:
            raise ValueError("merge_r must be non-negative")
        if self.proxy_mode not in PROXY_MODES:
            raise ValueError(f"unknown proxy_mode: {self.proxy_mode}")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


def _center(member: list[int], patch_w: int) -> tuple[float, float]:
    rows = [index // patch_w for index in member]
    cols = [index % patch_w for index in member]
    return (sum(rows) / len(rows), sum(cols) / len(cols))


def _pick_local_pairs(
    members: list[list[int]],
    *,
    patch_w: int,
    merge_r: int,
    proxy_mode: str,
) -> list[tuple[int, int]]:
    if merge_r <= 0 or len(members) < 2:
        return []

    centers = [_center(member, patch_w) for member in members]
    used: set[int] = set()
    selected: list[tuple[int, int]] = []

    if proxy_mode == "local-checkerboard":
        for dst in range(0, len(members) - 1, 2):
            src = dst + 1
            selected.append((src, dst))
            if len(selected) >= merge_r:
                break
        return selected

    if proxy_mode == "local-horizontal":
        order = sorted(range(len(members)), key=lambda idx: (round(centers[idx][0]), centers[idx][1]))
        same_line = lambda a, b: round(centers[a][0]) == round(centers[b][0])
    else:
        order = sorted(range(len(members)), key=lambda idx: (round(centers[idx][1]), centers[idx][0]))
        same_line = lambda a, b: round(centers[a][1]) == round(centers[b][1])

    for left, right in zip(order, order[1:]):
        if left in used or right in used or not same_line(left, right):
            continue
        dst = min(left, right)
        src = max(left, right)
        selected.append((src, dst))
        used.add(left)
        used.add(right)
        if len(selected) >= merge_r:
            break
    return selected


def _merge_proxy_pairs(
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    members: list[list[int]],
    *,
    patch_w: int,
    merge_r: int,
    proxy_mode: str,
    special_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], int]:
    selected = _pick_local_pairs(members, patch_w=patch_w, merge_r=merge_r, proxy_mode=proxy_mode)
    if not selected:
        return x, sizes, sources, members, 0

    source_locals = {src for src, _dst in selected}
    keep_patch_locals = [index for index in range(len(members)) if index not in source_locals]
    device = x.device
    keep_idx = torch.as_tensor(
        [*range(special_count), *[special_count + index for index in keep_patch_locals]],
        dtype=torch.long,
        device=device,
    )
    new_x = x.index_select(1, keep_idx).clone()
    new_sizes = sizes.index_select(1, keep_idx).clone()
    new_sources = sources.index_select(1, keep_idx).clone()
    new_members = [list(members[index]) for index in keep_patch_locals]
    old_to_new = {old: new for new, old in enumerate(keep_patch_locals)}

    for src_local, dst_local in selected:
        dst_new_local = old_to_new[dst_local]
        src_idx = special_count + src_local
        dst_new_idx = special_count + dst_new_local
        src_size = sizes[:, src_idx]
        dst_size = new_sizes[:, dst_new_idx]
        new_x[:, dst_new_idx] = _merge_wavg(new_x[:, dst_new_idx], x[:, src_idx], dst_size, src_size)
        new_sizes[:, dst_new_idx] = dst_size + src_size
        new_sources[:, dst_new_idx] = new_sources[:, dst_new_idx] + sources[:, src_idx]
        new_members[dst_new_local].extend(members[src_local])

    return new_x, new_sizes, new_sources, new_members, len(selected)


def _run_proxy_block(
    block: torch.nn.Module,
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    members: list[list[int]],
    *,
    patch_w: int,
    merge_r: int,
    proxy_mode: str,
    special_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], int]:
    attn_out, _metric = _proportional_attention(block.attn, block.norm1(x), sizes)
    x = x + block.drop_path1(block.ls1(attn_out))
    x, sizes, sources, members, merged = _merge_proxy_pairs(
        x,
        sizes,
        sources,
        members,
        patch_w=patch_w,
        merge_r=merge_r,
        proxy_mode=proxy_mode,
        special_count=special_count,
    )
    x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
    return x, sizes, sources, members, merged


def get_dense_proxy_intermediate_layers(
    vit: torch.nn.Module,
    x: torch.Tensor,
    layers: list[int],
    *,
    merge_r: int,
    proxy_mode: str,
    merge_counter: Counter[int] | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    sequence, sizes, sources, patch_count, special_count = _prepare_tome_tokens(vit, x)
    patch_w = int(x.shape[-1] // vit.patch_size)
    members = [[index] for index in range(patch_count)]
    layers_to_take = set(layers)
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    for block_index, block in enumerate(vit.blocks):
        sequence, sizes, sources, members, merged = _run_proxy_block(
            block,
            sequence,
            sizes,
            sources,
            members,
            patch_w=patch_w,
            merge_r=merge_r,
            proxy_mode=proxy_mode,
            special_count=special_count,
        )
        if merge_counter is not None:
            merge_counter[block_index] += merged
        if block_index not in layers_to_take:
            continue
        normalized = vit.norm(sequence)
        outputs.append(
            _restore_patch_grid(
                normalized,
                sources,
                patch_count=patch_count,
                special_count=special_count,
            )
        )

    if len(outputs) != len(layers):
        raise RuntimeError(f"only captured {len(outputs)} / {len(layers)} requested layers")
    return tuple(outputs)


class DenseSafeProxyDepthAnything(torch.nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        merge_r: int,
        proxy_mode: str,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.merge_r = merge_r
        self.proxy_mode = proxy_mode
        self.merge_counter: Counter[int] = Counter()

    def image2tensor(self, raw_image, input_size: int = 518):
        return self.base_model.image2tensor(raw_image, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        layers = self.base_model.intermediate_layer_idx[self.base_model.encoder]
        features = get_dense_proxy_intermediate_layers(
            self.base_model.pretrained,
            x,
            layers,
            merge_r=self.merge_r,
            proxy_mode=self.proxy_mode,
            merge_counter=self.merge_counter,
        )
        depth = self.base_model.depth_head(features, patch_h, patch_w)
        return F.relu(depth).squeeze(1)


@torch.no_grad()
def infer_depth(model: torch.nn.Module, image, input_size: int, device: torch.device) -> torch.Tensor:
    tensor, (height, width) = model.image2tensor(image, input_size)
    tensor = tensor.to(device)
    depth = model(tensor)
    depth = F.interpolate(depth[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
    return depth.detach().float().cpu()


def evaluate(config: DenseSafeProxyConfig) -> dict[str, Any]:
    require_ready(config.dataset_root, config.checkpoint, config.encoder)
    cv2 = load_cv2()
    device = resolve_device(config.device)
    dense_model = load_model(config.encoder, config.checkpoint, device)
    model = DenseSafeProxyDepthAnything(
        dense_model,
        merge_r=config.merge_r,
        proxy_mode=config.proxy_mode,
    ).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    selected = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
    )
    if not selected:
        raise RuntimeError("no DA-2K annotations selected")

    total = empty_counts()
    by_scene = defaultdict(empty_counts)
    missing_images: list[str] = []
    started = time.monotonic()

    for index, (relative_path, pairs) in enumerate(selected, start=1):
        image_path = config.dataset_root / relative_path
        image = cv2.imread(str(image_path))
        if image is None:
            missing_images.append(str(image_path))
            continue
        depth = infer_depth(model, image, config.input_size, device)
        scene = scene_from_path(relative_path)
        for pair in pairs:
            if pair.get("closer_point") != "point1":
                raise ValueError(f"unsupported closer_point in {relative_path}: {pair}")
            d1 = point_value(depth, pair["point1"])
            d2 = point_value(depth, pair["point2"])
            add_pair(total, d1, d2)
            add_pair(by_scene[scene], d1, d2)
        if config.log_every > 0 and (index % config.log_every == 0 or index == len(selected)):
            print(f"{config.proxy_mode}: evaluated {index}/{len(selected)} images", flush=True)

    result = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "variant": "dense_safe_proxy",
            "images_requested": len(selected),
            "missing_images": missing_images,
            "elapsed_seconds": time.monotonic() - started,
            "requested_merges_per_layer": config.merge_r,
            "actual_merges_by_block": {
                str(block_index): int(model.merge_counter.get(block_index, 0))
                for block_index in range(len(dense_model.pretrained.blocks))
            },
            "notes": [
                "Dense-safe proxy keeps source maps and restores every original patch before the DA-V2 depth head.",
                "Pairs are local fixed-pattern proxy merges, not learned or attention-calibrated merges.",
            ],
        },
        "overall": finalize_counts(total),
        "by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(by_scene.items())},
    }
    if config.output_json:
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate simple dense-safe token-reduction proxies for DA-V2 on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/da2k_vits_proxy_local_horizontal_r57.json"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--merge-r", type=int, default=57)
    parser.add_argument("--proxy-mode", choices=PROXY_MODES, default="local-horizontal")
    parser.add_argument("--scene-type", default="", choices=SCENE_CHOICES)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = DenseSafeProxyConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        merge_r=args.merge_r,
        proxy_mode=args.proxy_mode,
        scene_type=args.scene_type,
        max_images=args.max_images,
        log_every=args.log_every,
    )
    try:
        summary = evaluate(config)
    except SetupError as exc:
        raise SystemExit(str(exc))
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print(f"wrote {config.output_json}")


if __name__ == "__main__":
    main()
