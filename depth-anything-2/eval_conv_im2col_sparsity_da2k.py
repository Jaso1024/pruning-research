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
import torch.nn as nn
import torch.nn.functional as F

from eval_da2k import (
    MODEL_CONFIGS,
    add_pair,
    empty_counts,
    finalize_counts,
    point_value,
    resolve_device,
    scene_from_path,
)
from eval_gelu_relu_compensation_da2k import infer_depth, selected_annotations, write_summary
from eval_moefication_da2k import load_moefication_base


TARGET_MODES = {"rcu_conv2", "rcu_convs", "head_3x3", "head_convs"}
SCORE_MODES = {"activation", "weighted_activation"}


@dataclass(frozen=True)
class ConvIm2ColConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    activation: str = "relu"
    stage2: str = "none"
    stage2_shift: float = 0.0
    summary_json: Path | None = None
    variant_key: str = ""
    state_dict: Path | None = None
    target: str = "rcu_conv2"
    score: str = "weighted_activation"
    fraction: float = 0.25
    block_size: int = 1
    max_images: int = 16
    max_pairs: int = 0
    scene_type: str = ""
    log_every: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.summary_json is not None:
            object.__setattr__(self, "summary_json", Path(self.summary_json))
        if self.state_dict is not None:
            object.__setattr__(self, "state_dict", Path(self.state_dict))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.target not in TARGET_MODES:
            raise ValueError(f"unknown target: {self.target}")
        if self.score not in SCORE_MODES:
            raise ValueError(f"unknown score: {self.score}")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.max_images < 0 or self.max_pairs < 0:
            raise ValueError("max_images and max_pairs must be non-negative")


class Im2ColSparseConv2d(nn.Module):
    def __init__(
        self,
        source: nn.Conv2d,
        *,
        fraction: float,
        score: str,
        block_size: int,
    ) -> None:
        super().__init__()
        if source.groups != 1:
            raise ValueError("Im2ColSparseConv2d only supports groups=1")
        self.weight = source.weight
        self.bias = source.bias
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.kernel_size = source.kernel_size
        self.fraction = float(fraction)
        self.score = score
        self.block_size = int(block_size)
        self.patch_chunk_size = 2048
        flat_weight = source.weight.detach().float().reshape(source.out_channels, -1)
        col_norm = flat_weight.norm(dim=0).clamp_min(1e-8)
        self.register_buffer("column_weight", col_norm, persistent=False)
        self.reset_sparse_stats()

    @property
    def out_channels(self) -> int:
        return int(self.weight.shape[0])

    def reset_sparse_stats(self) -> None:
        self.patches_seen = 0
        self.features_selected = 0

    def _output_hw(self, h: int, w: int) -> tuple[int, int]:
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        out_h = math.floor((h + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
        out_w = math.floor((w + 2 * pw - dw * (kw - 1) - 1) / sw + 1)
        return out_h, out_w

    def _selection_mask(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: [B, K, L], K = in_channels * kernel_h * kernel_w.
        feature_count = patches.shape[1]
        k = max(1, min(feature_count, round(feature_count * self.fraction)))
        values = patches.float().abs()
        if self.score == "weighted_activation":
            values = values * self.column_weight.to(device=patches.device).view(1, -1, 1)
        indices = values.transpose(1, 2).topk(k, dim=2).indices
        if self.block_size <= 1:
            mask = torch.zeros(
                (patches.shape[0], patches.shape[2], feature_count),
                device=patches.device,
                dtype=torch.bool,
            )
            mask.scatter_(2, indices, True)
            return mask.transpose(1, 2)

        block_count = (feature_count + self.block_size - 1) // self.block_size
        block_mask = torch.zeros(
            (patches.shape[0], patches.shape[2], block_count),
            device=patches.device,
            dtype=torch.bool,
        )
        block_mask.scatter_(2, indices // self.block_size, True)
        feature_ids = torch.arange(feature_count, device=patches.device)
        return block_mask[:, :, feature_ids // self.block_size].transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_h, out_w = self._output_hw(int(x.shape[-2]), int(x.shape[-1]))
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        flat_weight = self.weight.reshape(self.out_channels, -1)
        out = patches.new_empty((x.shape[0], self.out_channels, patches.shape[2]))
        selected_features = 0
        for start in range(0, patches.shape[2], self.patch_chunk_size):
            end = min(start + self.patch_chunk_size, patches.shape[2])
            patch_chunk = patches[:, :, start:end]
            mask = self._selection_mask(patch_chunk)
            masked = patch_chunk * mask.to(dtype=patch_chunk.dtype)
            out[:, :, start:end] = torch.einsum("ok,bkl->bol", flat_weight, masked)
            with torch.no_grad():
                selected_features += int(mask.sum().item())
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)
        with torch.no_grad():
            self.patches_seen += int(patches.shape[0] * patches.shape[2])
            self.features_selected += selected_features
        return out.reshape(x.shape[0], self.out_channels, out_h, out_w)

    def sparse_summary(self) -> dict[str, Any]:
        feature_count = int(self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3])
        denom = max(1, self.patches_seen * feature_count)
        return {
            "fraction": self.fraction,
            "score": self.score,
            "block_size": self.block_size,
            "patches_seen": int(self.patches_seen),
            "feature_count": feature_count,
            "selected_feature_fraction": self.features_selected / float(denom),
        }


def target_conv_names(model: nn.Module, target: str) -> list[str]:
    names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if not name.startswith("depth_head."):
            continue
        if target == "rcu_conv2":
            if ".resConfUnit" in name and name.endswith(".conv2"):
                names.append(name)
        elif target == "rcu_convs":
            if ".resConfUnit" in name and (name.endswith(".conv1") or name.endswith(".conv2")):
                names.append(name)
        elif target == "head_3x3":
            if module.kernel_size == (3, 3):
                names.append(name)
        elif target == "head_convs":
            names.append(name)
    return names


def install_sparse_convs(model: nn.Module, config: ConvIm2ColConfig) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for name in target_conv_names(model, config.target):
        module = model.get_submodule(name)
        if module.groups != 1:
            continue
        wrapped = Im2ColSparseConv2d(
            module,
            fraction=config.fraction,
            score=config.score,
            block_size=config.block_size,
        )
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, wrapped)
        summaries[name] = {
            "weight_shape": list(module.weight.shape),
            "params": module.weight.numel() + (module.bias.numel() if module.bias is not None else 0),
        }
    return summaries


def reset_sparse_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, Im2ColSparseConv2d):
            module.reset_sparse_stats()


def collect_sparse_stats(model: nn.Module) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for name, module in model.named_modules():
        if isinstance(module, Im2ColSparseConv2d):
            stats[name] = module.sparse_summary()
    if stats:
        executed = [row for row in stats.values() if row["patches_seen"] > 0]
        selected = [row["selected_feature_fraction"] for row in executed]
        total_selected = sum(
            row["selected_feature_fraction"] * row["patches_seen"] * row["feature_count"]
            for row in executed
        )
        total_features = sum(row["patches_seen"] * row["feature_count"] for row in executed)
        stats["_mean"] = {
            "executed_modules": len(executed),
            "installed_modules": len(stats),
            "selected_feature_fraction": sum(selected) / len(selected) if selected else 0.0,
            "weighted_selected_feature_fraction": total_selected / total_features if total_features else 0.0,
        }
    return stats


def evaluate_da2k(
    *,
    model: nn.Module,
    dataset_root: Path,
    items: list[tuple[str, list[dict[str, Any]]]],
    input_size: int,
    device: torch.device,
    log_every: int,
) -> dict[str, Any]:
    total = empty_counts()
    by_scene = defaultdict(empty_counts)
    missing_images: list[str] = []
    started = time.monotonic()
    model.eval()
    reset_sparse_stats(model)

    for index, (relative_path, pairs) in enumerate(items, start=1):
        image = cv2.imread(str(dataset_root / relative_path))
        if image is None:
            missing_images.append(str(dataset_root / relative_path))
            continue
        depth = infer_depth(model, image, input_size, device)
        scene = scene_from_path(relative_path)
        for pair in pairs:
            if pair.get("closer_point") != "point1":
                raise ValueError(f"unsupported closer_point in {relative_path}: {pair}")
            d1 = point_value(depth, pair["point1"])
            d2 = point_value(depth, pair["point2"])
            add_pair(total, d1, d2)
            add_pair(by_scene[scene], d1, d2)
        del depth
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if log_every > 0 and (index % log_every == 0 or index == len(items)):
            print(f"evaluated {index}/{len(items)} images", flush=True)

    return {
        "metadata": {
            "images_requested": len(items),
            "missing_images": missing_images,
            "elapsed_seconds": time.monotonic() - started,
        },
        "overall": finalize_counts(total),
        "by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(by_scene.items())},
        "sparse_conv_stats": collect_sparse_stats(model),
    }


def run(config: ConvIm2ColConfig) -> dict[str, Any]:
    torch.manual_seed(89)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model, load_summary = load_moefication_base(config, device)
    install_summary = install_sparse_convs(model, config)
    selected_items = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
        max_pairs=config.max_pairs,
    )
    result: dict[str, Any] = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "loaded_model": load_summary,
            "installed_sparse_convs": install_summary,
            "note": (
                "Conv-as-FFN im2col sparsity prototype. Each selected Conv2d is evaluated as "
                "unfolded patches times flattened weights, with top-k patch features retained per output pixel. "
                "This is an accuracy/shape probe, not an optimized kernel."
            ),
        },
        "variants": {},
    }
    summary_path = config.output_dir / "summary.json"
    write_summary(summary_path, result)
    evaluation = evaluate_da2k(
        model=model,
        dataset_root=config.dataset_root,
        items=selected_items,
        input_size=config.input_size,
        device=device,
        log_every=config.log_every,
    )
    key = f"{config.target}_{config.score}_frac{config.fraction:g}_block{config.block_size}"
    result["variants"][key] = {
        "metadata": {
            "target": config.target,
            "score": config.score,
            "fraction": config.fraction,
            "block_size": config.block_size,
        },
        "evaluation": evaluation,
    }
    write_summary(summary_path, result)
    print(json.dumps({key: evaluation["overall"]}, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DA2K im2col sparse-conv probe.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/conv_im2col_sparsity"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--stage2", choices=["none", "norm2", "norm12"], default="none")
    parser.add_argument("--stage2-shift", type=float, default=0.0)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--variant-key", default="")
    parser.add_argument("--state-dict", type=Path, default=None)
    parser.add_argument("--target", choices=sorted(TARGET_MODES), default="rcu_conv2")
    parser.add_argument("--score", choices=sorted(SCORE_MODES), default="weighted_activation")
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--log-every", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(
        ConvIm2ColConfig(
            dataset_root=args.dataset_root,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            encoder=args.encoder,
            input_size=args.input_size,
            device=args.device,
            activation=args.activation,
            stage2=args.stage2,
            stage2_shift=args.stage2_shift,
            summary_json=args.summary_json,
            variant_key=args.variant_key,
            state_dict=args.state_dict,
            target=args.target,
            score=args.score,
            fraction=args.fraction,
            block_size=args.block_size,
            max_images=args.max_images,
            max_pairs=args.max_pairs,
            scene_type=args.scene_type,
            log_every=args.log_every,
        )
    )


if __name__ == "__main__":
    main()
