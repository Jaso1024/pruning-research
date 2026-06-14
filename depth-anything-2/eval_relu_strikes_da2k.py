from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from eval_da2k import MODEL_CONFIGS, resolve_device
from eval_gelu_relu_adapter_sweep_da2k import ADAPTER_METHODS, fit_adapter_method
from eval_gelu_relu_compensation_da2k import (
    evaluate_da2k_model,
    load_calibration_tensors,
    load_model,
    selected_annotations,
    transformer_mlp_names,
    write_summary,
)


DEFAULT_VARIANTS: tuple[str, ...] = (
    "relu:none",
    "shift0_25:none",
    "shift0_5:none",
    "shift1_0:none",
    "relu:norm2",
    "shift0_25:norm2",
    "relu:norm12",
)

STAGE2_MODES = {"none", "norm2", "norm12"}


@dataclass(frozen=True)
class ActivationSpec:
    name: str
    shift: float = 0.0


@dataclass(frozen=True)
class VariantSpec:
    raw: str
    activation: ActivationSpec
    stage2: str
    method: str
    key: str


@dataclass(frozen=True)
class ReLUStrikeConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    calibration_images: int = 32
    calibration_tokens: int = 8192
    max_images: int = 32
    max_pairs: int = 0
    scene_type: str = ""
    variants: tuple[str, ...] = DEFAULT_VARIANTS
    method: str = "dora"
    rank: int = 32
    alpha: float = 32.0
    steps: int = 100
    lr: float = 3e-3
    batch_tokens: int = 2048
    weight_decay: float = 0.0
    optimizer: str = "radam"
    kronecker_factor: int = 4
    stage2_shift: float = 0.0
    save_models: bool = False
    skip_dense_eval: bool = False
    skip_variant_eval: bool = False
    seed: int = 73
    log_every: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "method", self.method.lower())
        object.__setattr__(self, "optimizer", self.optimizer.lower())
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.method not in ADAPTER_METHODS:
            raise ValueError(f"unknown adapter method: {self.method}")
        if not self.variants:
            raise ValueError("at least one variant is required")
        if self.calibration_images <= 0:
            raise ValueError("calibration_images must be positive")
        if self.calibration_tokens <= 0:
            raise ValueError("calibration_tokens must be positive")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.batch_tokens <= 0:
            raise ValueError("batch_tokens must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.kronecker_factor <= 0:
            raise ValueError("kronecker_factor must be positive")


class ShiftedReLU(nn.Module):
    def __init__(self, spec: ActivationSpec) -> None:
        super().__init__()
        self.spec = spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec.shift == 0.0:
            return F.relu(x)
        return F.relu(x - self.spec.shift)

    def extra_repr(self) -> str:
        return f"name={self.spec.name}, shift={self.spec.shift:g}"


class NormThenActivation(nn.Module):
    def __init__(self, norm: nn.Module, activation: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(x))


def parse_variants(value: str) -> tuple[str, ...]:
    variants = tuple(part.strip() for part in value.split(",") if part.strip())
    return variants or DEFAULT_VARIANTS


def sanitize_name(value: str) -> str:
    out = []
    for ch in value:
        out.append(ch if ch.isalnum() or ch in {"_", "-"} else "_")
    return "".join(out)


def parse_float_token(text: str) -> float:
    return float(text.replace("m", "-").replace("p", "+").replace("_", "."))


def activation_spec(name: str) -> ActivationSpec:
    key = name.strip().lower()
    if key == "relu":
        return ActivationSpec(name="relu", shift=0.0)
    if key.startswith("shift"):
        shift = parse_float_token(key.removeprefix("shift"))
        return ActivationSpec(name=f"shift{shift:g}", shift=shift)
    raise ValueError(f"unknown activation: {name}")


def parse_variant(raw: str, *, default_method: str) -> VariantSpec:
    parts = [part.strip().lower() for part in raw.split(":") if part.strip()]
    if len(parts) == 1:
        parts.append("none")
    if len(parts) > 3:
        raise ValueError(f"variant should be activation[:stage2[:method]], got {raw!r}")
    activation = activation_spec(parts[0])
    stage2 = parts[1]
    method = parts[2] if len(parts) == 3 else default_method
    if stage2 not in STAGE2_MODES:
        raise ValueError(f"unknown stage2 mode {stage2!r}; expected one of {sorted(STAGE2_MODES)}")
    if method not in ADAPTER_METHODS:
        raise ValueError(f"unknown adapter method {method!r}")
    key = sanitize_name(f"{activation.name}_{stage2}_{method}")
    return VariantSpec(raw=raw, activation=activation, stage2=stage2, method=method, key=key)


def install_mlp_activation(model: nn.Module, spec: ActivationSpec) -> list[str]:
    changed: list[str] = []
    for name in transformer_mlp_names(model):
        mlp = model.get_submodule(name)
        mlp.act = ShiftedReLU(spec)
        changed.append(f"{name}.act")
    return changed


def install_stage2(model: nn.Module, *, mode: str, shift: float) -> list[str]:
    if mode == "none":
        return []
    changed: list[str] = []
    spec = ActivationSpec(name=f"stage2_shift{shift:g}", shift=shift)
    for block_index, block in enumerate(model.pretrained.blocks):
        if mode == "norm12":
            block.norm1 = NormThenActivation(block.norm1, ShiftedReLU(spec))
            changed.append(f"pretrained.blocks.{block_index}.norm1")
        if mode in {"norm2", "norm12"}:
            block.norm2 = NormThenActivation(block.norm2, ShiftedReLU(spec))
            changed.append(f"pretrained.blocks.{block_index}.norm2")
    return changed


def maybe_gate_mlp_inputs(
    dense_records: dict[str, dict[str, torch.Tensor]],
    *,
    stage2: str,
    shift: float,
) -> dict[str, dict[str, torch.Tensor]]:
    if stage2 not in {"norm2", "norm12"}:
        return dense_records

    gated: dict[str, dict[str, torch.Tensor]] = {}
    for name, record in dense_records.items():
        inputs = record["inputs"]
        if shift == 0.0:
            transformed = F.relu(inputs)
        else:
            transformed = F.relu(inputs - shift)
        gated[name] = {
            "inputs": transformed,
            "targets": record["targets"],
        }
    return gated


def collect_dense_mlp_calibration_flushed(
    *,
    dense_model: nn.Module,
    mlp_names: list[str],
    calibration_tensors: list[torch.Tensor],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    records: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"inputs": [], "targets": []} for name in mlp_names
    }
    handles = []

    for name in mlp_names:
        module = dense_model.get_submodule(name)

        def make_hook(module_name: str):
            def hook(_module, inputs, output) -> None:
                records[module_name]["inputs"].append(inputs[0].detach().flatten(0, 1).float().cpu())
                records[module_name]["targets"].append(output.detach().flatten(0, 1).float().cpu())

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

    dense_model.eval()
    try:
        with torch.inference_mode():
            for tensor in tqdm(calibration_tensors, desc="collect dense MLP targets", unit="image"):
                _ = dense_model(tensor.to(device=device, non_blocking=True))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
    finally:
        for handle in handles:
            handle.remove()

    packed: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensors in records.items():
        packed[name] = {
            "inputs": torch.cat(tensors["inputs"], dim=0),
            "targets": torch.cat(tensors["targets"], dim=0),
        }
    return packed


def run(config: ReLUStrikeConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    started = time.monotonic()
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    selected_items = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
        max_pairs=config.max_pairs,
    )
    if len(selected_items) < config.calibration_images:
        raise RuntimeError(f"selected {len(selected_items)} images, but calibration_images={config.calibration_images}")

    parsed_variants = tuple(parse_variant(variant, default_method=config.method) for variant in config.variants)
    result: dict[str, Any] = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_selected": len(selected_items),
            "pairs_selected": sum(len(pairs) for _path, pairs in selected_items),
            "parsed_variants": [asdict(variant) for variant in parsed_variants],
            "note": (
                "ReLU Strikes Back pairing for Depth Anything V2: replace GELU with ReLU or shifted ReLU, "
                "optionally insert ReLU after ViT norm2/norm1, then fold a layerwise low-rank adapter repair "
                "into MLP fc1/fc2 weights against dense GELU MLP targets."
            ),
        },
        "variants": {},
    }
    summary_path = config.output_dir / "summary.json"
    write_summary(summary_path, result)

    dense_model = load_model(config.encoder, config.checkpoint, device)
    for param in dense_model.parameters():
        param.requires_grad_(False)
    mlp_names = transformer_mlp_names(dense_model)
    result["metadata"]["transformer_mlp_names"] = mlp_names
    calibration_tensors, calibration_paths = load_calibration_tensors(
        dense_model,
        dataset_root=config.dataset_root,
        items=selected_items,
        input_size=config.input_size,
        device=device,
        limit=config.calibration_images,
    )
    calibration_tensors = [tensor.detach().cpu() for tensor in calibration_tensors]
    result["metadata"]["calibration_relative_paths"] = calibration_paths
    write_summary(summary_path, result)

    if config.skip_dense_eval:
        result["variants"]["dense"] = {
            "metadata": {"activation": "original GELU checkpoint"},
            "evaluation": {
                "metadata": {
                    "images_requested": len(selected_items),
                    "skipped": True,
                    "reason": "skip_dense_eval was set.",
                },
                "overall": {},
                "by_scene": {},
            },
        }
    else:
        result["variants"]["dense"] = {
            "metadata": {"activation": "original GELU checkpoint"},
            "evaluation": evaluate_da2k_model(
                model=dense_model,
                dataset_root=config.dataset_root,
                items=selected_items,
                input_size=config.input_size,
                device=device,
                log_every=config.log_every,
            ),
        }
    write_summary(summary_path, result)

    dense_records = collect_dense_mlp_calibration_flushed(
        dense_model=dense_model,
        mlp_names=mlp_names,
        calibration_tensors=calibration_tensors,
        device=device,
    )
    del calibration_tensors
    del dense_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for variant in parsed_variants:
        model = load_model(config.encoder, config.checkpoint, device)
        for param in model.parameters():
            param.requires_grad_(False)

        changed_mlp = install_mlp_activation(model, variant.activation)
        changed_stage2 = install_stage2(model, mode=variant.stage2, shift=config.stage2_shift)
        repair_records = maybe_gate_mlp_inputs(
            dense_records,
            stage2=variant.stage2,
            shift=config.stage2_shift,
        )
        repair = fit_adapter_method(
            relu_model=model,
            dense_records=repair_records,
            mlp_names=mlp_names,
            method=variant.method,
            calibration_tokens=config.calibration_tokens,
            rank=config.rank,
            alpha=config.alpha,
            steps=config.steps,
            lr=config.lr,
            batch_tokens=config.batch_tokens,
            weight_decay=config.weight_decay,
            optimizer_name=config.optimizer,
            kronecker_factor=config.kronecker_factor,
            seed=config.seed,
            device=device,
        )
        if repair_records is not dense_records:
            del repair_records

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        saved_model_path = None
        if config.save_models:
            saved_model_path = config.output_dir / f"{variant.key}.state_dict.pt"
            torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, saved_model_path)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        if config.skip_variant_eval:
            evaluation = {
                "metadata": {
                    "images_requested": len(selected_items),
                    "skipped": True,
                    "reason": "skip_variant_eval was set; evaluate the saved folded state dict in a fresh process.",
                },
                "overall": {},
                "by_scene": {},
            }
        else:
            evaluation = evaluate_da2k_model(
                model=model,
                dataset_root=config.dataset_root,
                items=selected_items,
                input_size=config.input_size,
                device=device,
                log_every=config.log_every,
            )

        result["variants"][variant.key] = {
            "metadata": {
                "variant": variant.raw,
                "activation": asdict(variant.activation),
                "stage2": variant.stage2,
                "stage2_shift": config.stage2_shift,
                "adapter_method": variant.method,
                "rank": config.rank,
                "alpha": config.alpha,
                "steps": config.steps,
                "lr": config.lr,
                "optimizer": config.optimizer,
                "batch_tokens": config.batch_tokens,
                "changed_mlp_modules": changed_mlp,
                "changed_stage2_modules": changed_stage2,
                "mlp_input_transform": (
                    "dense MLP calibration inputs are passed through the same stage-2 ReLU before adapter fitting"
                    if variant.stage2 in {"norm2", "norm12"}
                    else "dense MLP calibration inputs are unchanged"
                ),
                "saved_model_path": str(saved_model_path) if saved_model_path is not None else None,
                "repair": repair,
            },
            "evaluation": evaluation,
        }
        write_summary(summary_path, result)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result["metadata"]["elapsed_seconds"] = time.monotonic() - started
    write_summary(summary_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Depth Anything V2 ReLU Strikes + low-rank adapter repair sweep.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/relu_strikes"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-images", type=int, default=32)
    parser.add_argument("--calibration-tokens", type=int, default=8192)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--method", default="dora")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-tokens", type=int, default=2048)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", default="radam")
    parser.add_argument("--kronecker-factor", type=int, default=4)
    parser.add_argument("--stage2-shift", type=float, default=0.0)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--skip-dense-eval", action="store_true")
    parser.add_argument("--skip-variant-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--log-every", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ReLUStrikeConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        calibration_images=args.calibration_images,
        calibration_tokens=args.calibration_tokens,
        max_images=args.max_images,
        max_pairs=args.max_pairs,
        scene_type=args.scene_type,
        variants=parse_variants(args.variants),
        method=args.method,
        rank=args.rank,
        alpha=args.alpha,
        steps=args.steps,
        lr=args.lr,
        batch_tokens=args.batch_tokens,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        kronecker_factor=args.kronecker_factor,
        stage2_shift=args.stage2_shift,
        save_models=args.save_models,
        skip_dense_eval=args.skip_dense_eval,
        skip_variant_eval=args.skip_variant_eval,
        seed=args.seed,
        log_every=args.log_every,
    )
    summary = run(config)
    print(json.dumps({name: row["evaluation"]["overall"] for name, row in summary["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
