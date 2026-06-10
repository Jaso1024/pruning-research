from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

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


NormKind = Literal["l1", "l2"]
FillMode = Literal["zero", "input"]


@dataclass(frozen=True)
class PatchNormConfig:
    dataset_root: Path
    checkpoint: Path
    output_json: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    keep_percentage: float = 0.5
    norm: NormKind = "l2"
    keep_high: bool = True
    fill_mode: FillMode = "zero"
    scene_type: str = ""
    max_images: int = 0
    log_every: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_json", Path(self.output_json))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if not 0.0 < self.keep_percentage <= 1.0:
            raise ValueError("keep_percentage must be in (0, 1]")
        if self.norm not in {"l1", "l2"}:
            raise ValueError("norm must be l1 or l2")
        if self.fill_mode not in {"zero", "input"}:
            raise ValueError("fill_mode must be zero or input")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


def _stable_topk_indices(scores: torch.Tensor, keep_count: int, *, keep_high: bool) -> torch.Tensor:
    if keep_count >= scores.numel():
        return torch.arange(scores.numel(), dtype=torch.long, device=scores.device)
    values = torch.topk(scores, k=keep_count, largest=keep_high, sorted=False).values
    cutoff = values.min() if keep_high else values.max()
    better = scores > cutoff if keep_high else scores < cutoff
    selected = torch.nonzero(better, as_tuple=False).flatten()
    needed = keep_count - selected.numel()
    if needed > 0:
        ties = torch.nonzero(scores == cutoff, as_tuple=False).flatten()
        selected = torch.cat((selected, ties[:needed]))
    return torch.sort(selected.to(dtype=torch.long))[0]


def select_kept_patch_indices(
    patch_embeddings: torch.Tensor,
    *,
    keep_percentage: float,
    norm: NormKind,
    keep_high: bool,
) -> torch.Tensor:
    if patch_embeddings.ndim != 3 or patch_embeddings.shape[0] != 1:
        raise ValueError("patch_embeddings must have shape [1, patch_count, hidden_size]")
    patch_values = patch_embeddings.detach()[0].float()
    if norm == "l2":
        scores = torch.linalg.vector_norm(patch_values, ord=2, dim=1)
    elif norm == "l1":
        scores = torch.linalg.vector_norm(patch_values, ord=1, dim=1)
    else:
        raise ValueError(f"unsupported norm: {norm}")
    keep_count = max(1, math.ceil(scores.numel() * keep_percentage))
    return _stable_topk_indices(scores, keep_count, keep_high=keep_high)


def _prepare_pruned_tokens(
    vit: torch.nn.Module,
    x: torch.Tensor,
    *,
    keep_percentage: float,
    norm: NormKind,
    keep_high: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if x.shape[0] != 1:
        raise ValueError("patch-norm pruning currently supports batch size 1")
    batch, _channels, width, height = x.shape
    patch_embeddings = vit.patch_embed(x)
    kept_indices = select_kept_patch_indices(
        patch_embeddings,
        keep_percentage=keep_percentage,
        norm=norm,
        keep_high=keep_high,
    )
    cls_token = vit.cls_token.expand(batch, -1, -1)
    full_tokens_for_pos = torch.cat((cls_token, patch_embeddings), dim=1)
    pos = vit.interpolate_pos_encoding(full_tokens_for_pos, width, height)
    cls_token = cls_token + pos[:, :1]
    patch_tokens = patch_embeddings + pos[:, 1:]

    if getattr(vit, "register_tokens", None) is not None:
        register_tokens = vit.register_tokens.expand(batch, -1, -1)
        sequence = torch.cat((cls_token, register_tokens, patch_tokens[:, kept_indices]), dim=1)
    else:
        sequence = torch.cat((cls_token, patch_tokens[:, kept_indices]), dim=1)
    return sequence, patch_tokens, kept_indices, patch_embeddings.shape[1]


def get_pruned_intermediate_layers(
    vit: torch.nn.Module,
    x: torch.Tensor,
    layers: list[int],
    *,
    keep_percentage: float,
    norm_kind: NormKind,
    keep_high: bool,
    fill_mode: FillMode,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    sequence, input_patch_tokens, kept_indices, patch_count = _prepare_pruned_tokens(
        vit,
        x,
        keep_percentage=keep_percentage,
        norm=norm_kind,
        keep_high=keep_high,
    )
    layers_to_take = set(layers)
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    register_count = int(getattr(vit, "num_register_tokens", 0))
    patch_start = 1 + register_count

    for block_index, block in enumerate(vit.blocks):
        sequence = block(sequence)
        if block_index not in layers_to_take:
            continue
        normalized = vit.norm(sequence)
        class_token = normalized[:, 0]
        kept_patch_tokens = normalized[:, patch_start:]
        if fill_mode == "input":
            restored = vit.norm(input_patch_tokens)
        else:
            restored = torch.zeros(
                (1, patch_count, normalized.shape[-1]),
                dtype=normalized.dtype,
                device=normalized.device,
            )
        restored = restored.clone()
        restored[:, kept_indices] = kept_patch_tokens
        outputs.append((restored, class_token))

    if len(outputs) != len(layers):
        raise RuntimeError(f"only captured {len(outputs)} / {len(layers)} requested layers")
    return tuple(outputs)


class PatchNormPrunedDepthAnything(torch.nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        keep_percentage: float,
        norm: NormKind,
        keep_high: bool,
        fill_mode: FillMode,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.keep_percentage = keep_percentage
        self.norm = norm
        self.keep_high = keep_high
        self.fill_mode = fill_mode

    def image2tensor(self, raw_image, input_size: int = 518):
        return self.base_model.image2tensor(raw_image, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        layers = self.base_model.intermediate_layer_idx[self.base_model.encoder]
        features = get_pruned_intermediate_layers(
            self.base_model.pretrained,
            x,
            layers,
            keep_percentage=self.keep_percentage,
            norm_kind=self.norm,
            keep_high=self.keep_high,
            fill_mode=self.fill_mode,
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


def evaluate(config: PatchNormConfig) -> dict[str, Any]:
    device = resolve_device(config.device)
    dense_model = load_model(config.encoder, config.checkpoint, device)
    model = PatchNormPrunedDepthAnything(
        dense_model,
        keep_percentage=config.keep_percentage,
        norm=config.norm,
        keep_high=config.keep_high,
        fill_mode=config.fill_mode,
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
            print(f"evaluated {index}/{len(selected)} images", flush=True)

    patch_count = (config.input_size // 14) * (config.input_size // 14)
    keep_count = math.ceil(patch_count * config.keep_percentage)
    result = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_requested": len(selected),
            "missing_images": missing_images,
            "elapsed_seconds": time.monotonic() - started,
            "patch_count_at_square_input": patch_count,
            "kept_patch_count_at_square_input": keep_count,
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
    parser = argparse.ArgumentParser(description="Evaluate patch-embedding norm token pruning on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/da2k_vits_patch_norm.json"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keep-percentage", type=float, default=0.5)
    parser.add_argument("--norm", choices=["l1", "l2"], default="l2")
    parser.add_argument("--keep-low", action="store_true", help="Keep low-norm patch tokens instead of high-norm tokens.")
    parser.add_argument("--fill-mode", choices=["zero", "input"], default="zero")
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
    config = PatchNormConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        keep_percentage=args.keep_percentage,
        norm=args.norm,
        keep_high=not args.keep_low,
        fill_mode=args.fill_mode,
        scene_type=args.scene_type,
        max_images=args.max_images,
        log_every=args.log_every,
    )
    summary = evaluate(config)
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
