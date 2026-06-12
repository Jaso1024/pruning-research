from __future__ import annotations

import argparse
import copy
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
from tqdm.auto import tqdm

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
class ExperimentConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    calibration_images: int = 8
    calibration_tokens: int = 4096
    max_images: int = 32
    max_pairs: int = 0
    modes: tuple[str, ...] = ("dense", "relu", "newton", "hadamard")
    scene_type: str = ""
    log_every: int = 8
    ridge_lambda: float = 1e-3
    hadamard_block_size: int = 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.calibration_images <= 0:
            raise ValueError("calibration_images must be positive")
        if self.calibration_tokens <= 0:
            raise ValueError("calibration_tokens must be positive")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        if self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be non-negative")
        if self.hadamard_block_size < 0:
            raise ValueError("hadamard_block_size must be non-negative")
        allowed = {"dense", "relu", "newton", "hadamard"}
        unknown = set(self.modes) - allowed
        if unknown:
            raise ValueError(f"unknown mode(s): {sorted(unknown)}")


def parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in value.split(",") if part.strip())
    return modes or ("dense", "relu", "newton", "hadamard")


def selected_annotations(
    dataset_root: Path,
    *,
    scene_type: str,
    max_images: int,
    max_pairs: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    annotations = json.loads((dataset_root / "annotations.json").read_text())
    selected: list[tuple[str, list[dict[str, Any]]]] = []
    pair_count = 0
    for image_path, pairs in annotations.items():
        if scene_type and scene_from_path(image_path) != scene_type:
            continue
        kept_pairs = list(pairs)
        if max_pairs > 0:
            remaining = max_pairs - pair_count
            if remaining <= 0:
                break
            kept_pairs = kept_pairs[:remaining]
        if kept_pairs:
            selected.append((image_path, kept_pairs))
            pair_count += len(kept_pairs)
        if max_images > 0 and len(selected) >= max_images:
            break
    return selected


def load_calibration_tensors(
    model: torch.nn.Module,
    *,
    dataset_root: Path,
    items: list[tuple[str, list[dict[str, Any]]]],
    input_size: int,
    device: torch.device,
    limit: int,
) -> tuple[list[torch.Tensor], list[str]]:
    tensors: list[torch.Tensor] = []
    paths: list[str] = []
    for relative_path, _pairs in items:
        if len(tensors) >= limit:
            break
        image = cv2.imread(str(dataset_root / relative_path))
        if image is None:
            continue
        tensor, _shape = model.image2tensor(image, input_size)
        tensors.append(tensor.to(device=device, non_blocking=True))
        paths.append(relative_path)
    if len(tensors) < limit:
        raise RuntimeError(f"loaded {len(tensors)} calibration images, requested {limit}")
    return tensors, paths


@torch.no_grad()
def infer_depth(model: torch.nn.Module, image, input_size: int, device: torch.device) -> torch.Tensor:
    tensor, (height, width) = model.image2tensor(image, input_size)
    tensor = tensor.to(device=device, non_blocking=True)
    depth = model(tensor)
    depth = F.interpolate(depth[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
    return depth.detach().float().cpu()


def evaluate_da2k_model(
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
    }


def replace_gelu_with_relu(model: torch.nn.Module) -> list[str]:
    replaced: list[str] = []
    for module_name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.GELU):
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                setattr(module, child_name, nn.ReLU(inplace=False))
                replaced.append(full_name)
    return replaced


def transformer_mlp_names(model: torch.nn.Module) -> list[str]:
    names: list[str] = []
    for name, module in model.named_modules():
        if not name.startswith("pretrained.blocks."):
            continue
        if hasattr(module, "fc1") and hasattr(module, "fc2") and hasattr(module, "act"):
            if isinstance(module.fc1, nn.Linear) and isinstance(module.fc2, nn.Linear):
                names.append(name)
    return names


def collect_dense_mlp_calibration(
    *,
    dense_model: torch.nn.Module,
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


def fit_fc2_least_squares_repair(
    *,
    relu_model: torch.nn.Module,
    dense_records: dict[str, dict[str, torch.Tensor]],
    mlp_names: list[str],
    calibration_tokens: int,
    ridge_lambda: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    summaries: list[dict[str, Any]] = []

    for name in tqdm(mlp_names, desc="fit fc2 repairs", unit="mlp"):
        mlp = relu_model.get_submodule(name)
        inputs = dense_records[name]["inputs"]
        targets = dense_records[name]["targets"]
        token_count = inputs.shape[0]
        if token_count > calibration_tokens:
            indices = torch.randperm(token_count, generator=generator)[:calibration_tokens]
            inputs = inputs.index_select(0, indices)
            targets = targets.index_select(0, indices)

        x_in = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        y = targets.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad():
            x = mlp.act(mlp.fc1(x_in)).float()
            ones = torch.ones((x.shape[0], 1), device=device, dtype=x.dtype)
            x_aug = torch.cat([x, ones], dim=1)

            old_y = mlp.fc2(x)
            old_mse = F.mse_loss(old_y, y).item()

            gram = x_aug.T @ x_aug
            if ridge_lambda > 0.0:
                gram.diagonal().add_(ridge_lambda)
                gram[-1, -1].sub_(ridge_lambda)
            rhs = x_aug.T @ y
            beta = torch.linalg.solve(gram, rhs)
            new_weight = beta[:-1].T.contiguous()
            new_bias = beta[-1].contiguous()
            new_y = x @ new_weight.T + new_bias
            new_mse = F.mse_loss(new_y, y).item()

            mlp.fc2.weight.copy_(new_weight.to(dtype=mlp.fc2.weight.dtype))
            if mlp.fc2.bias is not None:
                mlp.fc2.bias.copy_(new_bias.to(dtype=mlp.fc2.bias.dtype))

        summaries.append(
            {
                "module": name,
                "tokens_available": int(token_count),
                "tokens_used": int(inputs.shape[0]),
                "hidden_features": int(x.shape[1]),
                "out_features": int(y.shape[1]),
                "old_mse": old_mse,
                "new_mse": new_mse,
            }
        )
    return summaries


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def normalized_fwht_rows(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"FWHT row count must be a power of two, got {n}")
    y = x.clone()
    h = 1
    while h < n:
        y = y.reshape(n // (2 * h), 2, h, *y.shape[1:])
        a = y[:, 0].clone()
        b = y[:, 1].clone()
        y[:, 0] = a + b
        y[:, 1] = a - b
        y = y.reshape(n, *x.shape[1:])
        h *= 2
    return y / math.sqrt(n)


def hadamard_rows_blockwise(x: torch.Tensor, *, block_size: int) -> tuple[torch.Tensor, list[dict[str, int]]]:
    dim = x.shape[0]
    out = torch.empty_like(x)
    chunks: list[dict[str, int]] = []
    if block_size <= 0:
        block_size = next_power_of_two(dim)
    start = 0
    while start < dim:
        size = min(block_size, dim - start)
        padded_size = next_power_of_two(size)
        chunk = x[start : start + size]
        if padded_size != size:
            pad_shape = (padded_size - size, *chunk.shape[1:])
            chunk = torch.cat([chunk, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=0)
        transformed = normalized_fwht_rows(chunk)[:size]
        out[start : start + size] = transformed
        chunks.append({"start": start, "size": size, "padded_size": padded_size})
        start += size
    return out, chunks


def apply_hadamard_hidden_basis(
    *,
    model: torch.nn.Module,
    mlp_names: list[str],
    block_size: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    with torch.no_grad():
        for name in mlp_names:
            mlp = model.get_submodule(name)
            fc1_weight, chunks = hadamard_rows_blockwise(mlp.fc1.weight.data.float(), block_size=block_size)
            mlp.fc1.weight.copy_(fc1_weight.to(dtype=mlp.fc1.weight.dtype))
            if mlp.fc1.bias is not None:
                bias, _chunks = hadamard_rows_blockwise(mlp.fc1.bias.data.float()[:, None], block_size=block_size)
                mlp.fc1.bias.copy_(bias[:, 0].to(dtype=mlp.fc1.bias.dtype))

            fc2_columns, _chunks = hadamard_rows_blockwise(mlp.fc2.weight.data.float().T, block_size=block_size)
            mlp.fc2.weight.copy_(fc2_columns.T.contiguous().to(dtype=mlp.fc2.weight.dtype))
            summaries.append(
                {
                    "module": name,
                    "hidden_features": int(mlp.fc1.out_features),
                    "chunks": chunks,
                    "rule": "fc1 rows and bias transformed by H; fc2 hidden columns transformed by H^T. H is normalized blockwise FWHT with padding only for non-power-of-two final blocks.",
                }
            )
    return summaries


def write_summary(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")


def run(config: ExperimentConfig) -> dict[str, Any]:
    torch.manual_seed(17)
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
    if not selected_items:
        raise RuntimeError("no DA-2K annotations selected")
    if len(selected_items) < config.calibration_images:
        raise RuntimeError(
            f"selected {len(selected_items)} images, but calibration_images={config.calibration_images}"
        )

    result: dict[str, Any] = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_selected": len(selected_items),
            "pairs_selected": sum(len(pairs) for _path, pairs in selected_items),
            "rule": "DA-2K labels point1 as closer; Depth Anything V2 vits uses larger predicted values for closer points.",
        },
        "variants": {},
    }
    summary_path = config.output_dir / "summary.json"
    write_summary(summary_path, result)

    dense_model = load_model(config.encoder, config.checkpoint, device)
    for param in dense_model.parameters():
        param.requires_grad_(False)
    mlp_names = transformer_mlp_names(dense_model)
    result["metadata"]["transformer_mlp_count"] = len(mlp_names)
    result["metadata"]["transformer_mlp_names"] = mlp_names

    calibration_tensors, calibration_paths = load_calibration_tensors(
        dense_model,
        dataset_root=config.dataset_root,
        items=selected_items,
        input_size=config.input_size,
        device=device,
        limit=config.calibration_images,
    )
    result["metadata"]["calibration_relative_paths"] = calibration_paths
    write_summary(summary_path, result)

    if "dense" in config.modes:
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

    relu_template = load_model(config.encoder, config.checkpoint, device)
    replaced = replace_gelu_with_relu(relu_template)
    for param in relu_template.parameters():
        param.requires_grad_(False)

    if "relu" in config.modes:
        result["variants"]["relu"] = {
            "metadata": {"activation": "all nn.GELU modules replaced with nn.ReLU", "replaced_modules": replaced},
            "evaluation": evaluate_da2k_model(
                model=relu_template,
                dataset_root=config.dataset_root,
                items=selected_items,
                input_size=config.input_size,
                device=device,
                log_every=config.log_every,
            ),
        }
        write_summary(summary_path, result)

    dense_records = None
    if "newton" in config.modes:
        dense_records = collect_dense_mlp_calibration(
            dense_model=dense_model,
            mlp_names=mlp_names,
            calibration_tensors=calibration_tensors,
            device=device,
        )
        newton_model = copy.deepcopy(relu_template).to(device=device).eval()
        repair = fit_fc2_least_squares_repair(
            relu_model=newton_model,
            dense_records=dense_records,
            mlp_names=mlp_names,
            calibration_tokens=config.calibration_tokens,
            ridge_lambda=config.ridge_lambda,
            device=device,
        )
        result["variants"]["newton"] = {
            "metadata": {
                "activation": "GELU replaced with ReLU",
                "approximation": "Damped Gauss-Newton / ridge least-squares closed-form repair of each transformer MLP fc2 weight and bias, matching original GELU MLP outputs on calibration tokens.",
                "ridge_lambda": config.ridge_lambda,
                "calibration_tokens_per_mlp": config.calibration_tokens,
                "replaced_modules": replaced,
                "repair": repair,
            },
            "evaluation": evaluate_da2k_model(
                model=newton_model,
                dataset_root=config.dataset_root,
                items=selected_items,
                input_size=config.input_size,
                device=device,
                log_every=config.log_every,
            ),
        }
        write_summary(summary_path, result)
        del newton_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "hadamard" in config.modes:
        hadamard_model = copy.deepcopy(relu_template).to(device=device).eval()
        hadamard = apply_hadamard_hidden_basis(
            model=hadamard_model,
            mlp_names=mlp_names,
            block_size=config.hadamard_block_size,
        )
        result["variants"]["hadamard"] = {
            "metadata": {
                "activation": "GELU replaced with ReLU",
                "compensation": "Orthogonal hidden-channel basis transform around each transformer MLP fc1/fc2 pair before evaluation.",
                "hadamard_block_size": config.hadamard_block_size,
                "replaced_modules": replaced,
                "hadamard": hadamard,
            },
            "evaluation": evaluate_da2k_model(
                model=hadamard_model,
                dataset_root=config.dataset_root,
                items=selected_items,
                input_size=config.input_size,
                device=device,
                log_every=config.log_every,
            ),
        }
        write_summary(summary_path, result)

    result["metadata"]["elapsed_seconds"] = time.monotonic() - started
    write_summary(summary_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Depth Anything V2 GELU to ReLU activation-swap compensation on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/da2k_vits_gelu_relu_compensation"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-images", type=int, default=8)
    parser.add_argument("--calibration-tokens", type=int, default=4096)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--modes", default="dense,relu,newton,hadamard")
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--hadamard-block-size", type=int, default=1024)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
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
        modes=parse_modes(args.modes),
        scene_type=args.scene_type,
        log_every=args.log_every,
        ridge_lambda=args.ridge_lambda,
        hadamard_block_size=args.hadamard_block_size,
    )
    summary = run(config)
    print(json.dumps({name: row["evaluation"]["overall"] for name, row in summary["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
