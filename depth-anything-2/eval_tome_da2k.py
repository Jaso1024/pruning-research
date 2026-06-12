from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn.functional as F

from eval_da2k import (
    MODEL_CONFIGS,
    add_pair,
    empty_counts,
    finalize_counts,
    load_model,
    point_value,
    resolve_device,
    scene_from_path,
)


@dataclass(frozen=True)
class ToMeConfig:
    dataset_root: Path
    checkpoint: Path
    output_json: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    merge_r: int = 57
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
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


def _prepare_tome_tokens(
    vit: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    if x.shape[0] != 1:
        raise ValueError("ToMe DA-2K evaluation currently supports batch size 1")

    batch, _channels, height, width = x.shape
    patch_embeddings = vit.patch_embed(x)
    patch_count = patch_embeddings.shape[1]

    cls_token = vit.cls_token.expand(batch, -1, -1)
    full_tokens_for_pos = torch.cat((cls_token, patch_embeddings), dim=1)
    pos = vit.interpolate_pos_encoding(full_tokens_for_pos, height, width)
    cls_token = cls_token + pos[:, :1]
    patch_tokens = patch_embeddings + pos[:, 1:]

    if getattr(vit, "register_tokens", None) is not None:
        register_tokens = vit.register_tokens.expand(batch, -1, -1)
        sequence = torch.cat((cls_token, register_tokens, patch_tokens), dim=1)
    else:
        sequence = torch.cat((cls_token, patch_tokens), dim=1)

    register_count = int(getattr(vit, "num_register_tokens", 0))
    special_count = 1 + register_count
    sizes = torch.ones(
        (batch, sequence.shape[1], 1),
        dtype=sequence.dtype,
        device=sequence.device,
    )
    sources = torch.zeros(
        (batch, sequence.shape[1], patch_count),
        dtype=sequence.dtype,
        device=sequence.device,
    )
    sources[:, special_count:, :] = torch.eye(
        patch_count,
        dtype=sequence.dtype,
        device=sequence.device,
    ).unsqueeze(0)

    return sequence, sizes, sources, patch_count, special_count


def _proportional_attention(
    attn: torch.nn.Module,
    x: torch.Tensor,
    sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, token_count, channels = x.shape
    head_count = int(attn.num_heads)
    head_dim = channels // head_count

    qkv = attn.qkv(x).reshape(batch, token_count, 3, head_count, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q = qkv[0] * attn.scale
    k = qkv[1]
    v = qkv[2]

    logits = q @ k.transpose(-2, -1)
    logits = logits + sizes.clamp_min(1e-6).log().transpose(1, 2).unsqueeze(1)
    attn_probs = logits.softmax(dim=-1)
    attn_probs = attn.attn_drop(attn_probs)

    out = (attn_probs @ v).transpose(1, 2).reshape(batch, token_count, channels)
    out = attn.proj(out)
    out = attn.proj_drop(out)

    metric = F.normalize(k.mean(dim=1).float(), p=2, dim=-1)
    return out, metric


def _merge_wavg(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_size: torch.Tensor,
    src_size: torch.Tensor,
) -> torch.Tensor:
    return (dst * dst_size + src * src_size) / (dst_size + src_size)


def _bipartite_soft_matching_merge(
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    metric: torch.Tensor,
    *,
    merge_r: int,
    special_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if merge_r <= 0:
        return x, sizes, sources, 0
    if x.shape[0] != 1:
        raise ValueError("ToMe merge currently supports batch size 1")

    patch_token_count = x.shape[1] - special_count
    if patch_token_count < 2:
        return x, sizes, sources, 0

    device = x.device
    rel = torch.arange(patch_token_count, device=device)
    a_idx = special_count + rel[0::2]
    b_idx = special_count + rel[1::2]
    if a_idx.numel() == 0 or b_idx.numel() == 0:
        return x, sizes, sources, 0

    scores = metric[0, a_idx] @ metric[0, b_idx].transpose(0, 1)
    best_scores, best_b_local = scores.max(dim=1)
    r = min(int(merge_r), int(best_scores.numel()))
    if r <= 0:
        return x, sizes, sources, 0

    selected_a_local = torch.argsort(best_scores, descending=True)[:r]
    selected_mask = torch.zeros(a_idx.numel(), dtype=torch.bool, device=device)
    selected_mask[selected_a_local] = True
    unmerged_a_idx = a_idx[~selected_mask]

    keep_idx = torch.cat(
        (
            torch.arange(special_count, device=device),
            unmerged_a_idx,
            b_idx,
        )
    )
    new_x = x.index_select(1, keep_idx).clone()
    new_sizes = sizes.index_select(1, keep_idx).clone()
    new_sources = sources.index_select(1, keep_idx).clone()

    b_new_offset = special_count + unmerged_a_idx.numel()
    selected_a_idx = a_idx[selected_a_local]
    selected_b_new_idx = b_new_offset + best_b_local[selected_a_local]

    for src_idx, dst_new_idx in zip(selected_a_idx.tolist(), selected_b_new_idx.tolist()):
        src_size = sizes[:, src_idx]
        dst_size = new_sizes[:, dst_new_idx]
        new_x[:, dst_new_idx] = _merge_wavg(new_x[:, dst_new_idx], x[:, src_idx], dst_size, src_size)
        new_sizes[:, dst_new_idx] = dst_size + src_size
        new_sources[:, dst_new_idx] = new_sources[:, dst_new_idx] + sources[:, src_idx]

    return new_x, new_sizes, new_sources, r


def _run_tome_block(
    block: torch.nn.Module,
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    *,
    merge_r: int,
    special_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    attn_out, metric = _proportional_attention(block.attn, block.norm1(x), sizes)
    x = x + block.drop_path1(block.ls1(attn_out))
    x, sizes, sources, merged = _bipartite_soft_matching_merge(
        x,
        sizes,
        sources,
        metric,
        merge_r=merge_r,
        special_count=special_count,
    )
    x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
    return x, sizes, sources, merged


def _restore_patch_grid(
    normalized: torch.Tensor,
    sources: torch.Tensor,
    *,
    patch_count: int,
    special_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    class_token = normalized[:, 0]
    patch_tokens = normalized[:, special_count:]
    patch_sources = sources[:, special_count:]
    restored = torch.bmm(patch_sources.transpose(1, 2), patch_tokens)
    if restored.shape[1] != patch_count:
        raise RuntimeError(f"restored {restored.shape[1]} patches, expected {patch_count}")
    return restored, class_token


def get_tome_intermediate_layers(
    vit: torch.nn.Module,
    x: torch.Tensor,
    layers: list[int],
    *,
    merge_r: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    sequence, sizes, sources, patch_count, special_count = _prepare_tome_tokens(vit, x)
    layers_to_take = set(layers)
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    for block_index, block in enumerate(vit.blocks):
        sequence, sizes, sources, _merged = _run_tome_block(
            block,
            sequence,
            sizes,
            sources,
            merge_r=merge_r,
            special_count=special_count,
        )
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


class ToMeDepthAnything(torch.nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        merge_r: int,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.merge_r = merge_r

    def image2tensor(self, raw_image, input_size: int = 518):
        return self.base_model.image2tensor(raw_image, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        layers = self.base_model.intermediate_layer_idx[self.base_model.encoder]
        features = get_tome_intermediate_layers(
            self.base_model.pretrained,
            x,
            layers,
            merge_r=self.merge_r,
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


def selected_annotations(
    dataset_root: Path,
    *,
    scene_type: str,
    max_images: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    annotations = json.loads((dataset_root / "annotations.json").read_text())
    selected = [
        (image_path, pairs)
        for image_path, pairs in annotations.items()
        if not scene_type or scene_from_path(image_path) == scene_type
    ]
    if max_images > 0:
        selected = selected[:max_images]
    return selected


def evaluate(config: ToMeConfig) -> dict[str, Any]:
    device = resolve_device(config.device)
    dense_model = load_model(config.encoder, config.checkpoint, device)
    model = ToMeDepthAnything(dense_model, merge_r=config.merge_r).to(device).eval()
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
            print(f"evaluated {index}/{len(selected)} images", flush=True)

    patch_count = (config.input_size // 14) * (config.input_size // 14)
    max_merges = len(dense_model.pretrained.blocks) * config.merge_r
    result = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_requested": len(selected),
            "missing_images": missing_images,
            "elapsed_seconds": time.monotonic() - started,
            "patch_count_at_square_input": patch_count,
            "requested_merges_per_layer": config.merge_r,
            "requested_max_merges_across_blocks": max_merges,
            "rule": "DA-2K labels point1 as closer; Depth Anything V2 vits uses larger predicted values for closer points.",
        },
        "overall": finalize_counts(total),
        "by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(by_scene.items())},
    }
    if config.output_json:
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Token Merging (ToMe) for Depth Anything V2 on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/da2k_vits_tome_r57.json"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--merge-r", type=int, default=57, help="Number of A-partition patch tokens to merge per ViT block.")
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ToMeConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        merge_r=args.merge_r,
        scene_type=args.scene_type,
        max_images=args.max_images,
        log_every=args.log_every,
    )
    summary = evaluate(config)
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
