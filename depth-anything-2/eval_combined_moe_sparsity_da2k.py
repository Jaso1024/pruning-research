from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import torch

from eval_conv_im2col_sparsity_da2k import collect_sparse_stats, install_sparse_convs, reset_sparse_stats
from eval_da2k import (
    MODEL_CONFIGS,
    add_pair,
    empty_counts,
    finalize_counts,
    point_value,
    resolve_device,
    scene_from_path,
)
from eval_gelu_relu_compensation_da2k import (
    infer_depth,
    load_calibration_tensors,
    load_model,
    selected_annotations,
    transformer_mlp_names,
    write_summary,
)
from eval_moefication_da2k import (
    collect_mlp_calibration,
    collect_moe_stats,
    install_moe_layers,
    load_moefication_base,
    reset_moe_stats,
)


@dataclass(frozen=True)
class CombinedMoEConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    activation: str = "original"
    stage2: str = "none"
    stage2_shift: float = 0.0
    summary_json: Path | None = None
    variant_key: str = ""
    state_dict: Path | None = None
    ffn_fraction: float = 0.25
    ffn_score: str = "weighted_activation"
    ffn_calibration_start_image: int = 0
    ffn_calibration_images: int = 4
    ffn_calibration_tokens: int = 2048
    conv_target: str = "rcu_conv2"
    conv_fraction: float = 0.25
    conv_score: str = "weighted_activation"
    conv_block_size: int = 1
    start_image: int = 0
    max_images: int = 16
    max_pairs: int = 0
    scene_type: str = ""
    seed: int = 89
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
        if self.stage2 not in {"none", "norm2", "norm12"}:
            raise ValueError("stage2 must be one of none, norm2, norm12")
        if self.activation.strip().lower() == "original" and self.stage2 != "none":
            raise ValueError("stage2 requires a relu/shift activation, not activation=original")
        if not 0.0 < self.ffn_fraction <= 1.0:
            raise ValueError("ffn_fraction must be in (0, 1]")
        if not 0.0 < self.conv_fraction <= 1.0:
            raise ValueError("conv_fraction must be in (0, 1]")
        if self.ffn_calibration_start_image < 0:
            raise ValueError("ffn_calibration_start_image must be non-negative")
        if self.ffn_calibration_images <= 0 or self.ffn_calibration_tokens <= 0:
            raise ValueError("FFN calibration settings must be positive")
        if self.conv_block_size <= 0:
            raise ValueError("conv_block_size must be positive")
        if self.start_image < 0:
            raise ValueError("start_image must be non-negative")
        if self.max_images < 0 or self.max_pairs < 0:
            raise ValueError("max_images and max_pairs must be non-negative")


def make_ffn_config(config: CombinedMoEConfig, hidden_features: int) -> SimpleNamespace:
    top_k = max(1, min(hidden_features, round(hidden_features * config.ffn_fraction)))
    return SimpleNamespace(
        partition="contiguous",
        routing="oracle",
        score=config.ffn_score,
        num_experts=hidden_features,
        top_k=top_k,
        router_hidden=0,
        router_steps=0,
        router_lr=1e-3,
        router_batch_tokens=2048,
        kmeans_iters=1,
        seed=config.seed,
    )


def make_conv_config(config: CombinedMoEConfig) -> SimpleNamespace:
    return SimpleNamespace(
        target=config.conv_target,
        fraction=config.conv_fraction,
        score=config.conv_score,
        block_size=config.conv_block_size,
    )


def reset_all_stats(model: torch.nn.Module) -> None:
    reset_moe_stats(model)
    reset_sparse_stats(model)


def collect_all_stats(model: torch.nn.Module) -> dict[str, Any]:
    return {
        "ffn": collect_moe_stats(model),
        "conv": collect_sparse_stats(model),
    }


def limit_pairs(
    items: list[tuple[str, list[dict[str, Any]]]],
    max_pairs: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    if max_pairs <= 0:
        return items
    selected: list[tuple[str, list[dict[str, Any]]]] = []
    pair_count = 0
    for image_path, pairs in items:
        remaining = max_pairs - pair_count
        if remaining <= 0:
            break
        kept_pairs = list(pairs[:remaining])
        if kept_pairs:
            selected.append((image_path, kept_pairs))
            pair_count += len(kept_pairs)
    return selected


def evaluate_da2k(
    *,
    model: torch.nn.Module,
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
    reset_all_stats(model)

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
        "sparsity_stats": collect_all_stats(model),
    }


def count_parameter_coverage(model: torch.nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    ffn_wrapped = 0
    conv_wrapped = 0
    for module in model.modules():
        if module.__class__.__name__ == "MoEifiedMlp":
            ffn_wrapped += module.fc1.weight.numel()
            if module.fc1.bias is not None:
                ffn_wrapped += module.fc1.bias.numel()
            ffn_wrapped += module.fc2.weight.numel()
            if module.fc2.bias is not None:
                ffn_wrapped += module.fc2.bias.numel()
        if module.__class__.__name__ == "Im2ColSparseConv2d":
            conv_wrapped += module.weight.numel()
            if module.bias is not None:
                conv_wrapped += module.bias.numel()
    return {
        "total_parameters": total,
        "wrapped_ffn_parameters": ffn_wrapped,
        "wrapped_conv_parameters": conv_wrapped,
        "wrapped_total_parameters": ffn_wrapped + conv_wrapped,
    }


def load_combined_base(config: CombinedMoEConfig, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if config.activation.strip().lower() != "original" or config.summary_json is not None:
        return load_moefication_base(config, device)

    model = load_model(config.encoder, config.checkpoint, device)
    load_summary: dict[str, Any] = {
        "activation": {"name": "original"},
        "stage2": config.stage2,
        "stage2_shift": config.stage2_shift,
        "changed_mlp_modules": [],
        "changed_stage2_modules": [],
    }
    if config.state_dict is not None:
        state = torch.load(config.state_dict, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"state dict mismatch: missing={missing}, unexpected={unexpected}")
        load_summary["state_dict"] = str(config.state_dict)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model, load_summary


def run(config: CombinedMoEConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    started = time.monotonic()
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.start_image > 0:
        selected_items = selected_annotations(
            config.dataset_root,
            scene_type=config.scene_type,
            max_images=0,
            max_pairs=0,
        )
        selected_items = selected_items[config.start_image :]
        if config.max_images > 0:
            selected_items = selected_items[: config.max_images]
        selected_items = limit_pairs(selected_items, config.max_pairs)
    else:
        selected_items = selected_annotations(
            config.dataset_root,
            scene_type=config.scene_type,
            max_images=config.max_images,
            max_pairs=config.max_pairs,
        )
    if not selected_items:
        raise RuntimeError("no DA-2K annotations selected")
    model, load_summary = load_combined_base(config, device)
    mlp_names = transformer_mlp_names(model)
    if not mlp_names:
        raise RuntimeError("no transformer MLP modules found")
    hidden_features = model.get_submodule(mlp_names[0]).fc1.out_features
    if any(model.get_submodule(name).fc1.out_features != hidden_features for name in mlp_names):
        raise RuntimeError("combined evaluator assumes uniform MLP hidden width")

    calibration_items = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=0,
        max_pairs=0,
    )
    calibration_items = calibration_items[config.ffn_calibration_start_image :]
    if not calibration_items:
        raise RuntimeError("no DA-2K calibration annotations selected")
    calibration_tensors, calibration_paths = load_calibration_tensors(
        model,
        dataset_root=config.dataset_root,
        items=calibration_items,
        input_size=config.input_size,
        device=device,
        limit=config.ffn_calibration_images,
    )
    calibration_tensors = [tensor.detach().cpu() for tensor in calibration_tensors]
    calibration = collect_mlp_calibration(
        model=model,
        mlp_names=mlp_names,
        calibration_tensors=calibration_tensors,
        calibration_tokens=config.ffn_calibration_tokens,
        device=device,
        seed=config.seed,
    )
    del calibration_tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ffn_config = make_ffn_config(config, hidden_features)
    ffn_install = install_moe_layers(
        model=model,
        mlp_names=mlp_names,
        calibration=calibration,
        config=ffn_config,
        device=device,
    )
    del calibration
    conv_install = install_sparse_convs(model, make_conv_config(config))
    parameter_coverage = count_parameter_coverage(model)

    result: dict[str, Any] = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "loaded_model": load_summary,
            "calibration_relative_paths": calibration_paths,
            "ffn_hidden_features": hidden_features,
            "ffn_top_k": ffn_config.top_k,
            "ffn_install": ffn_install,
            "conv_install": conv_install,
            "parameter_coverage": parameter_coverage,
            "note": (
                "Combined sparsity probe: transformer MLPs use post-activation channel/expert masking; "
                "selected convs use im2col top-k patch-feature masking. This is an accuracy/coverage probe, "
                "not an optimized CUDA implementation."
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
    key = (
        f"ffn{config.ffn_fraction:g}_{config.ffn_score}_"
        f"{config.conv_target}_conv{config.conv_fraction:g}_block{config.conv_block_size}"
    )
    result["variants"][key] = {
        "metadata": {
            "ffn_fraction": config.ffn_fraction,
            "ffn_top_k": ffn_config.top_k,
            "conv_target": config.conv_target,
            "conv_fraction": config.conv_fraction,
            "conv_block_size": config.conv_block_size,
        },
        "evaluation": evaluation,
    }
    result["metadata"]["elapsed_seconds"] = time.monotonic() - started
    write_summary(summary_path, result)
    print(json.dumps({key: evaluation["overall"]}, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combined FFN MoEification + conv im2col sparsity DA2K probe.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/combined_moe_sparsity"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--activation", default="original")
    parser.add_argument("--stage2", choices=["none", "norm2", "norm12"], default="none")
    parser.add_argument("--stage2-shift", type=float, default=0.0)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--variant-key", default="")
    parser.add_argument("--state-dict", type=Path, default=None)
    parser.add_argument("--ffn-fraction", type=float, default=0.25)
    parser.add_argument("--ffn-score", choices=["activation", "weighted_activation"], default="weighted_activation")
    parser.add_argument("--ffn-calibration-start-image", type=int, default=0)
    parser.add_argument("--ffn-calibration-images", type=int, default=4)
    parser.add_argument("--ffn-calibration-tokens", type=int, default=2048)
    parser.add_argument("--conv-target", choices=["rcu_conv2", "rcu_convs", "head_3x3", "head_convs"], default="rcu_conv2")
    parser.add_argument("--conv-fraction", type=float, default=0.25)
    parser.add_argument("--conv-score", choices=["activation", "weighted_activation"], default="weighted_activation")
    parser.add_argument("--conv-block-size", type=int, default=1)
    parser.add_argument("--start-image", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--seed", type=int, default=89)
    parser.add_argument("--log-every", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(
        CombinedMoEConfig(
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
            ffn_fraction=args.ffn_fraction,
            ffn_score=args.ffn_score,
            ffn_calibration_start_image=args.ffn_calibration_start_image,
            ffn_calibration_images=args.ffn_calibration_images,
            ffn_calibration_tokens=args.ffn_calibration_tokens,
            conv_target=args.conv_target,
            conv_fraction=args.conv_fraction,
            conv_score=args.conv_score,
            conv_block_size=args.conv_block_size,
            start_image=args.start_image,
            max_images=args.max_images,
            max_pairs=args.max_pairs,
            scene_type=args.scene_type,
            seed=args.seed,
            log_every=args.log_every,
        )
    )


if __name__ == "__main__":
    main()
