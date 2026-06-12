from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import torch
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


TargetKind = Literal["transformer", "all-linear"]
PruningScope = Literal["per-matrix", "per-column", "global"]
ScoreNormalization = Literal["none", "matrix-mean"]
AffineRepairMode = Literal["none", "after-each-chunk"]
AffineRepairScope = Literal["all-selected", "output-only"]


@dataclass(frozen=True)
class UnstructuredWandaConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    target: TargetKind = "transformer"
    pruning_scope: PruningScope = "per-matrix"
    score_normalization: ScoreNormalization = "none"
    layer_indices: tuple[int, ...] = ()
    prune_fraction: float = 0.5
    prune_chunk_fraction: float = 0.05
    recompute_every_weights: int = 0
    calibration_images: int = 8
    exclude_calibration_from_eval: bool = False
    repair_steps: int = 0
    repair_lr: float = 1e-5
    affine_repair: AffineRepairMode = "none"
    affine_repair_tokens: int = 4096
    affine_repair_scope: AffineRepairScope = "all-selected"
    scene_type: str = ""
    max_images: int = 0
    log_every: int = 50
    eval_baseline: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.target not in {"transformer", "all-linear"}:
            raise ValueError("target must be transformer or all-linear")
        if self.pruning_scope not in {"per-matrix", "per-column", "global"}:
            raise ValueError("pruning_scope must be per-matrix, per-column, or global")
        if self.score_normalization not in {"none", "matrix-mean"}:
            raise ValueError("score_normalization must be none or matrix-mean")
        if not 0.0 < self.prune_fraction < 1.0:
            raise ValueError("prune_fraction must be in (0, 1)")
        if not 0.0 < self.prune_chunk_fraction <= 1.0:
            raise ValueError("prune_chunk_fraction must be in (0, 1]")
        if self.recompute_every_weights < 0:
            raise ValueError("recompute_every_weights must be non-negative")
        if self.calibration_images <= 0:
            raise ValueError("calibration_images must be positive")
        if self.repair_steps < 0:
            raise ValueError("repair_steps must be non-negative")
        if self.repair_lr <= 0.0:
            raise ValueError("repair_lr must be positive")
        if self.affine_repair not in {"none", "after-each-chunk"}:
            raise ValueError("affine_repair must be none or after-each-chunk")
        if self.affine_repair_tokens <= 0:
            raise ValueError("affine_repair_tokens must be positive")
        if self.affine_repair_scope not in {"all-selected", "output-only"}:
            raise ValueError("affine_repair_scope must be all-selected or output-only")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


def parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()


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


def transformer_layer_index(module_name: str) -> int | None:
    prefix = "pretrained.blocks."
    if not module_name.startswith(prefix):
        return None
    parts = module_name.split(".")
    if len(parts) < 3:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def find_prunable_linears(
    model: torch.nn.Module,
    *,
    target: TargetKind,
    layer_indices: tuple[int, ...],
) -> list[str]:
    wanted_layers = set(layer_indices)
    module_names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if module.weight.ndim != 2:
            continue
        layer_index = transformer_layer_index(name)
        if target == "transformer" and layer_index is None:
            continue
        if wanted_layers and layer_index not in wanted_layers:
            continue
        module_names.append(name)
    return module_names


def load_calibration_tensors(
    model: torch.nn.Module,
    *,
    dataset_root: Path,
    items: list[tuple[str, list[dict[str, Any]]]],
    input_size: int,
    device: torch.device,
    limit: int,
) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for relative_path, _pairs in items:
        if len(tensors) >= limit:
            break
        image = cv2.imread(str(dataset_root / relative_path))
        if image is None:
            continue
        tensor, _shape = model.image2tensor(image, input_size)
        tensors.append(tensor.to(device=device, non_blocking=True))
    if not tensors:
        raise RuntimeError("no calibration images could be loaded")
    return tensors


def cache_dense_outputs(
    *,
    model: torch.nn.Module,
    image_tensors: list[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for image_tensor in tqdm(image_tensors, desc="cache dense outputs", unit="image"):
                output = model(image_tensor.to(device=device, non_blocking=True))
                outputs.append(output.detach().cpu().float())
    finally:
        model.train(was_training)
    if len(outputs) != len(image_tensors):
        raise RuntimeError("failed to cache dense outputs for all calibration tensors")
    return outputs


def resolve_prune_chunk_fraction(
    *,
    matrix_weights: int,
    prune_chunk_fraction: float,
    recompute_every_weights: int,
) -> float:
    if recompute_every_weights > 0:
        return min(float(recompute_every_weights) / max(float(matrix_weights), 1.0), 1.0)
    return prune_chunk_fraction


def collect_wanda_scores(
    *,
    model: torch.nn.Module,
    module_names: list[str],
    image_tensors: list[torch.Tensor],
    device: torch.device,
    step: int,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, float | int]]]:
    modules = {name: model.get_submodule(name) for name in module_names}
    sumsq = {
        name: torch.zeros(module.in_features, dtype=torch.float64)
        for name, module in modules.items()
        if isinstance(module, torch.nn.Linear)
    }
    counts = {name: 0 for name in sumsq}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            if not isinstance(module, torch.nn.Linear):
                return
            x = inputs[0].detach()
            if x.shape[-1] != module.in_features:
                raise RuntimeError(f"input feature mismatch for {name}: {x.shape[-1]} != {module.in_features}")
            flat = x.reshape(-1, x.shape[-1]).float()
            sumsq[name] += flat.pow(2).sum(dim=0).double().cpu()
            counts[name] += int(flat.shape[0])

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for image_tensor in tqdm(image_tensors, desc=f"wanda refresh {step}", unit="image"):
                model(image_tensor.to(device=device, non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    scores: dict[str, torch.Tensor] = {}
    stats: dict[str, dict[str, float | int]] = {}
    for name, module in modules.items():
        if not isinstance(module, torch.nn.Linear):
            continue
        token_count = counts[name]
        if token_count <= 0:
            raise RuntimeError(f"no WANDA activations captured for {name}")
        rms = (sumsq[name] / float(token_count)).sqrt().float()
        scores[name] = module.weight.detach().cpu().float().abs() * rms.reshape(1, -1)
        stats[name] = {
            "tokens": token_count,
            "mean_input_rms": float(rms.mean().item()),
            "max_input_rms": float(rms.max().item()),
        }
    return scores, stats


def collect_linear_inputs(
    model: torch.nn.Module,
    module_name: str,
    image_tensors: list[torch.Tensor],
    device: torch.device,
    max_tokens: int,
) -> torch.Tensor:
    module = model.get_submodule(module_name)
    if not isinstance(module, torch.nn.Linear):
        raise TypeError(f"{module_name} is not a Linear module")

    captured: list[torch.Tensor] = []
    token_count = 0

    def hook(linear: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        nonlocal token_count
        if token_count >= max_tokens:
            return
        if not isinstance(linear, torch.nn.Linear):
            return
        x = inputs[0].detach()
        if x.shape[-1] != linear.in_features:
            raise RuntimeError(f"input feature mismatch for {module_name}: {x.shape[-1]} != {linear.in_features}")
        flat = x.reshape(-1, x.shape[-1]).float().cpu()
        remaining = max_tokens - token_count
        if flat.shape[0] > remaining:
            flat = flat[:remaining]
        captured.append(flat)
        token_count += int(flat.shape[0])

    handle = module.register_forward_hook(hook)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for image_tensor in image_tensors:
                if token_count >= max_tokens:
                    break
                model(image_tensor.to(device=device, non_blocking=True))
    finally:
        handle.remove()
        model.train(was_training)

    if not captured:
        raise RuntimeError(f"no inputs captured for module {module_name}")
    return torch.cat(captured, dim=0).contiguous()


def should_affine_repair_module(module_name: str, scope: AffineRepairScope) -> bool:
    if scope == "all-selected":
        return True
    if scope == "output-only":
        return module_name.endswith(".attn.proj") or module_name.endswith(".mlp.fc2")
    raise ValueError(f"unknown affine repair scope: {scope}")


@torch.no_grad()
def fit_affine_output_repair_(
    sparse_module: torch.nn.Linear,
    dense_module: torch.nn.Linear,
    x: torch.Tensor,
    max_tokens: int,
) -> dict[str, float | int]:
    device = sparse_module.weight.device
    sample = x[:max_tokens].to(device=device, dtype=torch.float32, non_blocking=True)
    if sample.numel() == 0:
        raise RuntimeError("affine repair received an empty calibration input tensor")

    dense_weight = dense_module.weight.detach().to(device=device, dtype=torch.float32)
    dense_bias = dense_module.bias.detach().to(device=device, dtype=torch.float32) if dense_module.bias is not None else None
    sparse_weight = sparse_module.weight.detach().to(device=device, dtype=torch.float32)
    sparse_bias = (
        sparse_module.bias.detach().to(device=device, dtype=torch.float32)
        if sparse_module.bias is not None
        else None
    )

    target = F.linear(sample, dense_weight, dense_bias)
    pred = F.linear(sample, sparse_weight, sparse_bias)
    before_mse = F.mse_loss(pred, target)
    target_power = target.pow(2).mean().clamp_min(1e-12)
    before_rel_mse = before_mse / target_power

    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    denom = pred_centered.pow(2).mean(dim=0).clamp_min(1e-12)
    scale = (pred_centered * target_centered).mean(dim=0) / denom
    shift = target_mean - scale * pred_mean
    repaired_pred = pred * scale + shift

    after_mse = F.mse_loss(repaired_pred, target)
    after_rel_mse = after_mse / target_power

    weight_dtype = sparse_module.weight.dtype
    sparse_module.weight.mul_(scale.to(dtype=weight_dtype).view(-1, 1))
    if sparse_module.bias is None:
        sparse_module.bias = torch.nn.Parameter(
            shift.to(device=device, dtype=weight_dtype),
            requires_grad=sparse_module.weight.requires_grad,
        )
    else:
        bias_dtype = sparse_module.bias.dtype
        sparse_module.bias.mul_(scale.to(dtype=bias_dtype))
        sparse_module.bias.add_(shift.to(dtype=bias_dtype))

    return {
        "tokens": int(sample.shape[0]),
        "before_mse": float(before_mse.detach().cpu().item()),
        "after_mse": float(after_mse.detach().cpu().item()),
        "before_rel_mse": float(before_rel_mse.detach().cpu().item()),
        "after_rel_mse": float(after_rel_mse.detach().cpu().item()),
        "scale_mean": float(scale.mean().detach().cpu().item()),
        "scale_min": float(scale.min().detach().cpu().item()),
        "scale_max": float(scale.max().detach().cpu().item()),
        "shift_abs_mean": float(shift.abs().mean().detach().cpu().item()),
    }


def repair_selected_linears_affine_(
    *,
    model: torch.nn.Module,
    dense_model: torch.nn.Module,
    module_names: list[str],
    calibration_tensors: list[torch.Tensor],
    pruned_masks: dict[str, torch.Tensor],
    config: UnstructuredWandaConfig,
    device: torch.device,
    step: int,
) -> dict[str, Any]:
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    candidate_names = [
        name for name in module_names if should_affine_repair_module(name, config.affine_repair_scope)
    ]

    for module_index, module_name in enumerate(
        tqdm(candidate_names, desc=f"affine repair {step}", unit="module"),
        start=1,
    ):
        sparse_module = model.get_submodule(module_name)
        dense_module = dense_model.get_submodule(module_name)
        if not isinstance(sparse_module, torch.nn.Linear) or not isinstance(dense_module, torch.nn.Linear):
            continue

        module_started = time.monotonic()
        x = collect_linear_inputs(
            model,
            module_name,
            calibration_tensors,
            device,
            config.affine_repair_tokens,
        )
        stats = fit_affine_output_repair_(
            sparse_module=sparse_module,
            dense_module=dense_module,
            x=x,
            max_tokens=config.affine_repair_tokens,
        )
        reapply_pruned_masks_(model=model, pruned_masks=pruned_masks)
        record = {
            "step": step,
            "module_index": module_index,
            "module_count": len(candidate_names),
            "module_name": module_name,
            "affine_repair": config.affine_repair,
            "affine_repair_scope": config.affine_repair_scope,
            "affine_repair_tokens": config.affine_repair_tokens,
            "elapsed_seconds": time.monotonic() - module_started,
            **stats,
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True, default=str), flush=True)
        del x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reapply_pruned_masks_(model=model, pruned_masks=pruned_masks)
    module_count = len(records)
    return {
        "step": step,
        "affine_repair": config.affine_repair,
        "affine_repair_scope": config.affine_repair_scope,
        "affine_repair_tokens": config.affine_repair_tokens,
        "candidate_module_count": len(candidate_names),
        "repaired_module_count": module_count,
        "elapsed_seconds": time.monotonic() - started,
        "mean_before_mse": (
            sum(float(row["before_mse"]) for row in records) / module_count if module_count else None
        ),
        "mean_after_mse": (
            sum(float(row["after_mse"]) for row in records) / module_count if module_count else None
        ),
        "mean_before_rel_mse": (
            sum(float(row["before_rel_mse"]) for row in records) / module_count if module_count else None
        ),
        "mean_after_rel_mse": (
            sum(float(row["after_rel_mse"]) for row in records) / module_count if module_count else None
        ),
        "mean_scale": (
            sum(float(row["scale_mean"]) for row in records) / module_count if module_count else None
        ),
        "mean_shift_abs": (
            sum(float(row["shift_abs_mean"]) for row in records) / module_count if module_count else None
        ),
        "module_records": records,
    }


def apply_incremental_per_matrix_pruning_(
    *,
    model: torch.nn.Module,
    module_names: list[str],
    saliency_scores: dict[str, torch.Tensor],
    pruned_masks: dict[str, torch.Tensor],
    target_fraction: float,
    chunk_fraction: float,
    step: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    weights_seen = 0
    weights_zeroed_total = 0
    weights_zeroed_this_step = 0

    with torch.no_grad():
        for name in module_names:
            module = model.get_submodule(name)
            if not isinstance(module, torch.nn.Linear):
                continue
            param = module.weight
            weights_seen += int(param.numel())
            score = saliency_scores.get(name)
            if score is None:
                raise RuntimeError(f"missing saliency for {name}")
            if tuple(score.shape) != tuple(param.shape):
                raise ValueError(f"saliency shape mismatch for {name}: {tuple(score.shape)} != {tuple(param.shape)}")

            cumulative = pruned_masks.get(name)
            if cumulative is None:
                cumulative = torch.zeros(tuple(param.shape), dtype=torch.bool, device="cpu")
                pruned_masks[name] = cumulative

            target_count = int(param.numel() * target_fraction)
            already = int(cumulative.sum().item())
            remaining = max(target_count - already, 0)
            if remaining == 0:
                weights_zeroed_total += already
                rows.append(
                    {
                        "name": name,
                        "shape": list(param.shape),
                        "weights": int(param.numel()),
                        "zeroed_this_step": 0,
                        "zeroed_total": already,
                    }
                )
                continue

            chunk_count = min(max(1, int(param.numel() * chunk_fraction)), remaining)
            flat_score = score.detach().flatten().to(device=param.device, dtype=torch.float32)
            flat_cumulative = cumulative.flatten().to(device=param.device)
            flat_score = flat_score.masked_fill(flat_cumulative, torch.inf)
            indices = torch.topk(flat_score, k=chunk_count, largest=False).indices
            flat_mask = torch.zeros(param.numel(), dtype=torch.bool, device=param.device)
            flat_mask[indices] = True
            add_mask = flat_mask.reshape(param.shape)
            param.masked_fill_(add_mask, 0)

            add_mask_cpu = add_mask.to(device="cpu")
            cumulative |= add_mask_cpu
            zeroed_total = int(cumulative.sum().item())
            weights_zeroed_total += zeroed_total
            zeroed_this_module = int(add_mask_cpu.sum().item())
            weights_zeroed_this_step += zeroed_this_module
            rows.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "weights": int(param.numel()),
                    "zeroed_this_step": zeroed_this_module,
                    "zeroed_total": zeroed_total,
                }
            )

    return {
        "step": step,
        "pruning_scope": "per_matrix_iterative",
        "target_fraction": target_fraction,
        "chunk_fraction": chunk_fraction,
        "matrix_tensors_pruned": len(rows),
        "weights_seen": weights_seen,
        "weights_zeroed_this_step": weights_zeroed_this_step,
        "weights_zeroed_total": weights_zeroed_total,
        "actual_zero_fraction": weights_zeroed_total / max(weights_seen, 1),
        "target_reached": weights_zeroed_total >= int(weights_seen * target_fraction),
        "pruned_tensors": rows,
    }


def apply_incremental_per_column_pruning_(
    *,
    model: torch.nn.Module,
    module_names: list[str],
    saliency_scores: dict[str, torch.Tensor],
    pruned_masks: dict[str, torch.Tensor],
    target_fraction: float,
    chunk_fraction: float,
    step: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    weights_seen = 0
    weights_zeroed_total = 0
    weights_zeroed_this_step = 0
    target_zero_count = 0

    with torch.no_grad():
        for name in module_names:
            module = model.get_submodule(name)
            if not isinstance(module, torch.nn.Linear):
                continue
            param = module.weight
            if param.ndim != 2:
                continue
            out_features, in_features = int(param.shape[0]), int(param.shape[1])
            weights_seen += int(param.numel())
            score = saliency_scores.get(name)
            if score is None:
                raise RuntimeError(f"missing saliency for {name}")
            if tuple(score.shape) != tuple(param.shape):
                raise ValueError(f"saliency shape mismatch for {name}: {tuple(score.shape)} != {tuple(param.shape)}")

            cumulative = pruned_masks.get(name)
            if cumulative is None:
                cumulative = torch.zeros(tuple(param.shape), dtype=torch.bool, device="cpu")
                pruned_masks[name] = cumulative

            per_column_target = int(out_features * target_fraction)
            per_column_chunk = max(1, int(out_features * chunk_fraction))
            target_zero_count += per_column_target * in_features
            add_mask = torch.zeros(tuple(param.shape), dtype=torch.bool, device=param.device)

            for column_index in range(in_features):
                column_cumulative = cumulative[:, column_index]
                already = int(column_cumulative.sum().item())
                remaining = max(per_column_target - already, 0)
                if remaining == 0:
                    continue

                chunk_count = min(per_column_chunk, remaining)
                column_score = score[:, column_index].detach().to(device=param.device, dtype=torch.float32)
                column_cumulative_device = column_cumulative.to(device=param.device)
                column_score = column_score.masked_fill(column_cumulative_device, torch.inf)
                indices = torch.topk(column_score, k=chunk_count, largest=False).indices
                add_mask[:, column_index][indices] = True

            zeroed_this_module = int(add_mask.sum().item())
            if zeroed_this_module:
                param.masked_fill_(add_mask, 0)
                cumulative |= add_mask.to(device="cpu")

            zeroed_total = int(cumulative.sum().item())
            column_zero_counts = cumulative.sum(dim=0)
            columns_at_current_target = (
                int((column_zero_counts == per_column_target).sum().item())
                if per_column_target > 0
                else int((column_zero_counts == 0).sum().item())
            )
            weights_zeroed_total += zeroed_total
            weights_zeroed_this_step += zeroed_this_module
            rows.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "rows": out_features,
                    "columns": in_features,
                    "weights": int(param.numel()),
                    "per_column_target": per_column_target,
                    "per_column_chunk": per_column_chunk,
                    "zeroed_this_step": zeroed_this_module,
                    "zeroed_total": zeroed_total,
                    "actual_zero_fraction": zeroed_total / max(int(param.numel()), 1),
                    "column_zero_min": int(column_zero_counts.min().item()) if in_features else 0,
                    "column_zero_max": int(column_zero_counts.max().item()) if in_features else 0,
                    "column_zero_exact_target": columns_at_current_target,
                    "columns_at_current_target": columns_at_current_target,
                }
            )

    return {
        "step": step,
        "pruning_scope": "per_column_iterative",
        "target_fraction": target_fraction,
        "chunk_fraction": chunk_fraction,
        "matrix_tensors_pruned": len(rows),
        "weights_seen": weights_seen,
        "target_zero_count": target_zero_count,
        "weights_zeroed_this_step": weights_zeroed_this_step,
        "weights_zeroed_total": weights_zeroed_total,
        "actual_zero_fraction": weights_zeroed_total / max(weights_seen, 1),
        "target_reached": weights_zeroed_total >= target_zero_count,
        "pruned_tensors": rows,
    }


def apply_incremental_global_pruning_(
    *,
    model: torch.nn.Module,
    module_names: list[str],
    saliency_scores: dict[str, torch.Tensor],
    pruned_masks: dict[str, torch.Tensor],
    target_fraction: float,
    chunk_fraction: float,
    step: int,
    score_normalization: ScoreNormalization,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    weights_seen = 0
    weights_zeroed_total_before = 0
    score_chunks: list[torch.Tensor] = []
    slices: list[tuple[str, int, int]] = []

    for name in module_names:
        module = model.get_submodule(name)
        if not isinstance(module, torch.nn.Linear):
            continue
        param = module.weight
        score = saliency_scores.get(name)
        if score is None:
            raise RuntimeError(f"missing saliency for {name}")
        if tuple(score.shape) != tuple(param.shape):
            raise ValueError(f"saliency shape mismatch for {name}: {tuple(score.shape)} != {tuple(param.shape)}")

        cumulative = pruned_masks.get(name)
        if cumulative is None:
            cumulative = torch.zeros(tuple(param.shape), dtype=torch.bool, device="cpu")
            pruned_masks[name] = cumulative

        start = weights_seen
        weights_seen += int(param.numel())
        slices.append((name, start, weights_seen))
        weights_zeroed_total_before += int(cumulative.sum().item())

        flat_score = score.detach().flatten().cpu().float()
        flat_cumulative = cumulative.flatten()
        if score_normalization == "matrix-mean":
            available = flat_score.masked_select(~flat_cumulative)
            denom = available.mean().clamp_min(1e-12) if available.numel() else torch.tensor(1.0)
            flat_score = flat_score / denom
        elif score_normalization != "none":
            raise ValueError(f"unsupported score_normalization: {score_normalization}")
        flat_score = flat_score.masked_fill(flat_cumulative, torch.inf)
        score_chunks.append(flat_score)

    target_count = int(weights_seen * target_fraction)
    remaining = max(target_count - weights_zeroed_total_before, 0)
    chunk_count = min(max(1, int(weights_seen * chunk_fraction)), remaining) if remaining else 0
    weights_zeroed_this_step = 0

    if chunk_count > 0:
        flat_scores = torch.cat(score_chunks, dim=0)
        indices = torch.topk(flat_scores, k=chunk_count, largest=False).indices.cpu()
        indices = torch.sort(indices).values
        del flat_scores
    else:
        indices = torch.empty(0, dtype=torch.long)

    with torch.no_grad():
        for name, start, end in slices:
            module = model.get_submodule(name)
            if not isinstance(module, torch.nn.Linear):
                continue
            param = module.weight
            cumulative = pruned_masks[name]
            before = int(cumulative.sum().item())
            left = int(torch.searchsorted(indices, start, right=False).item())
            right = int(torch.searchsorted(indices, end, right=False).item())
            local_indices = indices[left:right] - start
            zeroed_this_module = int(local_indices.numel())
            if zeroed_this_module:
                flat_mask = torch.zeros(param.numel(), dtype=torch.bool, device=param.device)
                flat_mask[local_indices.to(device=param.device)] = True
                add_mask = flat_mask.reshape(param.shape)
                param.masked_fill_(add_mask, 0)
                cumulative |= add_mask.to(device="cpu")
            after = int(cumulative.sum().item())
            weights_zeroed_this_step += after - before
            rows.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "weights": int(param.numel()),
                    "zeroed_this_step": after - before,
                    "zeroed_total": after,
                    "actual_zero_fraction": after / max(int(param.numel()), 1),
                }
            )

    weights_zeroed_total = sum(int(pruned_masks[name].sum().item()) for name, _start, _end in slices)
    return {
        "step": step,
        "pruning_scope": "global_iterative",
        "target_fraction": target_fraction,
        "chunk_fraction": chunk_fraction,
        "matrix_tensors_pruned": len(rows),
        "modules_zeroed_this_step": sum(1 for row in rows if int(row["zeroed_this_step"]) > 0),
        "weights_seen": weights_seen,
        "weights_zeroed_this_step": weights_zeroed_this_step,
        "weights_zeroed_total": weights_zeroed_total,
        "actual_zero_fraction": weights_zeroed_total / max(weights_seen, 1),
        "target_reached": weights_zeroed_total >= int(weights_seen * target_fraction),
        "pruned_tensors": rows,
    }


def reapply_pruned_masks_(
    *,
    model: torch.nn.Module,
    pruned_masks: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, mask in pruned_masks.items():
            module = model.get_submodule(name)
            if not isinstance(module, torch.nn.Linear):
                continue
            module.weight.masked_fill_(mask.to(device=module.weight.device), 0)


def pruning_target_zero_count(
    *,
    model: torch.nn.Module,
    module_names: list[str],
    pruning_scope: PruningScope,
    target_fraction: float,
    total_matrix_weights: int,
) -> int:
    if pruning_scope != "per-column":
        return int(total_matrix_weights * target_fraction)

    target_count = 0
    for name in module_names:
        module = model.get_submodule(name)
        if not isinstance(module, torch.nn.Linear):
            continue
        out_features, in_features = int(module.weight.shape[0]), int(module.weight.shape[1])
        target_count += int(out_features * target_fraction) * in_features
    return target_count


def repair_with_depth_distillation_(
    *,
    model: torch.nn.Module,
    module_names: list[str],
    calibration_tensors: list[torch.Tensor],
    dense_outputs: list[torch.Tensor],
    pruned_masks: dict[str, torch.Tensor],
    repair_steps: int,
    repair_lr: float,
    device: torch.device,
    step: int,
) -> dict[str, Any]:
    if repair_steps <= 0:
        return {
            "step": step,
            "repair_steps": 0,
            "repair_lr": repair_lr,
            "losses": [],
            "elapsed_seconds": 0.0,
        }
    if len(calibration_tensors) != len(dense_outputs):
        raise ValueError("calibration_tensors and dense_outputs must have the same length")

    repair_params: list[torch.nn.Parameter] = []
    previous_requires_grad: dict[torch.nn.Parameter, bool] = {}
    for name in module_names:
        module = model.get_submodule(name)
        if not isinstance(module, torch.nn.Linear):
            continue
        previous_requires_grad[module.weight] = module.weight.requires_grad
        module.weight.requires_grad_(True)
        repair_params.append(module.weight)
    if not repair_params:
        raise RuntimeError("no repair parameters selected")

    started = time.monotonic()
    losses: list[float] = []
    optimizer = torch.optim.SGD(repair_params, lr=repair_lr)
    was_training = model.training
    model.eval()
    try:
        for repair_step in range(1, repair_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for image_tensor, dense_output_cpu in zip(calibration_tensors, dense_outputs):
                sparse_output = model(image_tensor.to(device=device, non_blocking=True))
                dense_output = dense_output_cpu.to(device=device, non_blocking=True)
                loss = F.mse_loss(sparse_output.float(), dense_output.float()) / len(calibration_tensors)
                loss.backward()
                total_loss += float(loss.detach().cpu().item())
            for name, mask in pruned_masks.items():
                module = model.get_submodule(name)
                if not isinstance(module, torch.nn.Linear) or module.weight.grad is None:
                    continue
                module.weight.grad.masked_fill_(mask.to(device=module.weight.device), 0)
            optimizer.step()
            reapply_pruned_masks_(model=model, pruned_masks=pruned_masks)
            losses.append(total_loss)
            print(
                json.dumps(
                    {
                        "step": step,
                        "repair_step": repair_step,
                        "repair_loss": total_loss,
                        "repair_lr": repair_lr,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        optimizer.zero_grad(set_to_none=True)
        for param, requires_grad in previous_requires_grad.items():
            param.requires_grad_(requires_grad)
        model.train(was_training)

    return {
        "step": step,
        "repair_steps": repair_steps,
        "repair_lr": repair_lr,
        "losses": losses,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "elapsed_seconds": time.monotonic() - started,
    }


def prune_with_iterative_wanda(
    *,
    model: torch.nn.Module,
    dense_model: torch.nn.Module | None,
    module_names: list[str],
    calibration_tensors: list[torch.Tensor],
    dense_outputs: list[torch.Tensor] | None,
    config: UnstructuredWandaConfig,
    device: torch.device,
) -> dict[str, Any]:
    total_matrix_weights = sum(int(model.get_submodule(name).weight.numel()) for name in module_names)
    chunk_fraction = resolve_prune_chunk_fraction(
        matrix_weights=total_matrix_weights,
        prune_chunk_fraction=config.prune_chunk_fraction,
        recompute_every_weights=config.recompute_every_weights,
    )
    if chunk_fraction <= 0.0:
        raise ValueError("resolved prune chunk fraction must be positive")
    final_target_zero_count = pruning_target_zero_count(
        model=model,
        module_names=module_names,
        pruning_scope=config.pruning_scope,
        target_fraction=config.prune_fraction,
        total_matrix_weights=total_matrix_weights,
    )

    masks: dict[str, torch.Tensor] = {}
    step_summaries: list[dict[str, Any]] = []
    repair_summaries: list[dict[str, Any]] = []
    affine_repair_summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    step = 0
    while True:
        current_zeroed = sum(int(mask.sum().item()) for mask in masks.values())
        if current_zeroed >= final_target_zero_count:
            break
        step += 1
        scores, activation_stats = collect_wanda_scores(
            model=model,
            module_names=module_names,
            image_tensors=calibration_tensors,
            device=device,
            step=step,
        )
        step_target_fraction = min(config.prune_fraction, step * chunk_fraction)
        if config.pruning_scope == "global":
            step_summary = apply_incremental_global_pruning_(
                model=model,
                module_names=module_names,
                saliency_scores=scores,
                pruned_masks=masks,
                target_fraction=step_target_fraction,
                chunk_fraction=chunk_fraction,
                step=step,
                score_normalization=config.score_normalization,
            )
        elif config.pruning_scope == "per-column":
            step_summary = apply_incremental_per_column_pruning_(
                model=model,
                module_names=module_names,
                saliency_scores=scores,
                pruned_masks=masks,
                target_fraction=step_target_fraction,
                chunk_fraction=chunk_fraction,
                step=step,
            )
        else:
            step_summary = apply_incremental_per_matrix_pruning_(
                model=model,
                module_names=module_names,
                saliency_scores=scores,
                pruned_masks=masks,
                target_fraction=step_target_fraction,
                chunk_fraction=chunk_fraction,
                step=step,
            )
        reapply_pruned_masks_(model=model, pruned_masks=masks)
        final_target_reached = step_summary["weights_zeroed_total"] >= final_target_zero_count
        step_summary["final_target_reached"] = final_target_reached
        step_summary["elapsed_seconds"] = time.monotonic() - started
        step_summary["activation_stats"] = {
            "module_count": len(activation_stats),
            "min_tokens": min(int(row["tokens"]) for row in activation_stats.values()),
            "max_tokens": max(int(row["tokens"]) for row in activation_stats.values()),
            "mean_input_rms": sum(float(row["mean_input_rms"]) for row in activation_stats.values())
            / max(len(activation_stats), 1),
        }
        if config.affine_repair == "after-each-chunk" and step_summary["weights_zeroed_this_step"] > 0:
            if dense_model is None:
                raise RuntimeError("affine repair requested without a dense model")
            affine_summary = repair_selected_linears_affine_(
                model=model,
                dense_model=dense_model,
                module_names=module_names,
                calibration_tensors=calibration_tensors,
                pruned_masks=masks,
                config=config,
                device=device,
                step=step,
            )
            step_summary["affine_repair"] = {
                key: value for key, value in affine_summary.items() if key != "module_records"
            }
            affine_repair_summaries.append(affine_summary)
        if config.repair_steps > 0 and step_summary["weights_zeroed_this_step"] > 0:
            if dense_outputs is None:
                raise RuntimeError("repair requested without dense calibration outputs")
            repair_summary = repair_with_depth_distillation_(
                model=model,
                module_names=module_names,
                calibration_tensors=calibration_tensors,
                dense_outputs=dense_outputs,
                pruned_masks=masks,
                repair_steps=config.repair_steps,
                repair_lr=config.repair_lr,
                device=device,
                step=step,
            )
            step_summary["repair"] = {
                key: value for key, value in repair_summary.items() if key != "losses"
            }
            repair_summaries.append(repair_summary)
        append_jsonl(config.output_dir / "steps.jsonl", step_summary)
        if affine_repair_summaries and affine_repair_summaries[-1]["step"] == step:
            append_jsonl(config.output_dir / "affine_repair_steps.jsonl", affine_repair_summaries[-1])
        if repair_summaries and repair_summaries[-1]["step"] == step:
            append_jsonl(config.output_dir / "repair_steps.jsonl", repair_summaries[-1])
        compact = {key: value for key, value in step_summary.items() if key not in {"pruned_tensors"}}
        print(json.dumps(compact, sort_keys=True, default=str), flush=True)
        step_summaries.append(step_summary)
        del scores
        if device.type == "cuda":
            torch.cuda.empty_cache()
        no_progress_at_reachable_target = step_summary["weights_zeroed_this_step"] == 0 and (
            config.pruning_scope != "per-column" or step_target_fraction >= config.prune_fraction
        )
        if final_target_reached or no_progress_at_reachable_target:
            break

    final_zeroed = sum(int(mask.sum().item()) for mask in masks.values())
    final_pruned_tensors = [
        {
            "name": name,
            "shape": list(mask.shape),
            "weights": int(mask.numel()),
            "zeroed": int(mask.sum().item()),
            "actual_zero_fraction": int(mask.sum().item()) / max(int(mask.numel()), 1),
        }
        for name, mask in sorted(masks.items())
    ]
    (config.output_dir / "pruned_tensors.json").write_text(
        json.dumps(final_pruned_tensors, indent=2, sort_keys=True, default=str) + "\n"
    )
    if config.pruning_scope == "global":
        pruning_scope_name = "global_iterative"
        pruning_rule = (
            "iterative global WANDA: recompute abs(weight) * input_activation_rms on the current pruned "
            "Depth Anything model, then rank all remaining target linear weights together and zero the next "
            "lowest-score unstructured global chunk until target_fraction is reached; prune masks are reapplied "
            f"after each chunk; score_normalization={config.score_normalization}"
        )
    elif config.pruning_scope == "per-column":
        pruning_scope_name = "per_column_iterative"
        pruning_rule = (
            "iterative per-column WANDA: recompute abs(weight) * input_activation_rms on the current pruned "
            "Depth Anything model, then for each selected linear matrix and each input column independently, "
            "zero the next lowest-score unstructured column chunk until each column reaches "
            "int(out_features * target_fraction); prune masks are reapplied after each chunk"
        )
    else:
        pruning_scope_name = "per_matrix_iterative"
        pruning_rule = (
            "iterative per-matrix WANDA: recompute abs(weight) * input_activation_rms on the current pruned "
            "Depth Anything model, then zero the next lowest-score unstructured chunk in each linear matrix "
            "until target_fraction is reached; prune masks are reapplied after each chunk"
        )
    if config.affine_repair != "none":
        pruning_rule += "; after each pruning chunk, fit and fold per-output affine linear repair into selected Linears"
    if config.repair_steps > 0:
        pruning_rule += "; after each pruning chunk, run SGD repair on calibration images to match dense depth outputs"

    return {
        "method": "wanda",
        "pruning_scope": pruning_scope_name,
        "target": config.target,
        "score_normalization": config.score_normalization,
        "repair_steps_per_pruning_step": config.repair_steps,
        "repair_lr": config.repair_lr,
        "repair_objective": "dense_depth_mse_on_calibration_images" if config.repair_steps > 0 else "none",
        "affine_repair": config.affine_repair,
        "affine_repair_tokens": config.affine_repair_tokens,
        "affine_repair_scope": config.affine_repair_scope,
        "affine_repair_objective": (
            "per_linear_output_affine_regression_on_calibration_activations"
            if config.affine_repair != "none"
            else "none"
        ),
        "target_fraction": config.prune_fraction,
        "chunk_fraction": chunk_fraction,
        "recompute_every_weights": int(total_matrix_weights * chunk_fraction),
        "steps": len(step_summaries),
        "weights_seen": total_matrix_weights,
        "target_zero_count": final_target_zero_count,
        "weights_zeroed": final_zeroed,
        "actual_zero_fraction": final_zeroed / max(total_matrix_weights, 1),
        "step_summaries": [
            {key: value for key, value in row.items() if key not in {"pruned_tensors"}}
            for row in step_summaries
        ],
        "repair_summaries": repair_summaries,
        "affine_repair_summaries": affine_repair_summaries,
        "pruned_tensors": final_pruned_tensors,
        "rule": pruning_rule,
    }


@torch.no_grad()
def infer_depth(model: torch.nn.Module, image, input_size: int, device: torch.device) -> torch.Tensor:
    tensor, (height, width) = model.image2tensor(image, input_size)
    tensor = tensor.to(device)
    depth = model(tensor)
    depth = torch.nn.functional.interpolate(
        depth[:, None],
        (height, width),
        mode="bilinear",
        align_corners=True,
    )[0, 0]
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

    for index, (relative_path, pairs) in enumerate(items, start=1):
        image_path = dataset_root / relative_path
        image = cv2.imread(str(image_path))
        if image is None:
            missing_images.append(str(image_path))
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


def run(config: UnstructuredWandaConfig) -> dict[str, Any]:
    torch.manual_seed(17)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True, default=str) + "\n"
    )

    selected_items = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
    )
    if not selected_items:
        raise RuntimeError("no DA-2K annotations selected")
    calibration_items = selected_items[: config.calibration_images]
    if len(calibration_items) < config.calibration_images:
        raise RuntimeError(
            f"requested {config.calibration_images} calibration images, but only {len(calibration_items)} were selected"
        )
    eval_items = selected_items[config.calibration_images :] if config.exclude_calibration_from_eval else selected_items
    if not eval_items:
        raise RuntimeError("no DA-2K evaluation annotations remain after excluding calibration images")

    model = load_model(config.encoder, config.checkpoint, device)
    for param in model.parameters():
        param.requires_grad_(False)
    module_names = find_prunable_linears(
        model,
        target=config.target,
        layer_indices=config.layer_indices,
    )
    if not module_names:
        raise RuntimeError("no prunable linear modules found")
    module_records = [
        {
            "name": name,
            "shape": list(model.get_submodule(name).weight.shape),
            "layer_index": transformer_layer_index(name),
            "weights": int(model.get_submodule(name).weight.numel()),
        }
        for name in module_names
    ]
    (config.output_dir / "target_modules.json").write_text(
        json.dumps(module_records, indent=2, sort_keys=True, default=str) + "\n"
    )

    baseline = None
    if config.eval_baseline:
        baseline = evaluate_da2k_model(
            model=model,
            dataset_root=config.dataset_root,
            items=eval_items,
            input_size=config.input_size,
            device=device,
            log_every=config.log_every,
        )

    calibration_tensors = load_calibration_tensors(
        model,
        dataset_root=config.dataset_root,
        items=calibration_items,
        input_size=config.input_size,
        device=device,
        limit=config.calibration_images,
    )
    dense_outputs = None
    if config.repair_steps > 0:
        dense_outputs = cache_dense_outputs(
            model=model,
            image_tensors=calibration_tensors,
            device=device,
        )
    dense_model = None
    if config.affine_repair != "none":
        dense_model = copy.deepcopy(model).to(device=device)
        dense_model.eval()
        for param in dense_model.parameters():
            param.requires_grad_(False)
    pruning = prune_with_iterative_wanda(
        model=model,
        dense_model=dense_model,
        module_names=module_names,
        calibration_tensors=calibration_tensors,
        dense_outputs=dense_outputs,
        config=config,
        device=device,
    )
    pruned = evaluate_da2k_model(
        model=model,
        dataset_root=config.dataset_root,
        items=eval_items,
        input_size=config.input_size,
        device=device,
        log_every=config.log_every,
    )
    result = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_selected": len(selected_items),
            "eval_images_requested": len(eval_items),
            "calibration_images_loaded": len(calibration_tensors),
            "calibration_relative_paths": [relative_path for relative_path, _pairs in calibration_items],
            "eval_excludes_calibration": config.exclude_calibration_from_eval,
            "target_module_count": len(module_names),
            "target_weight_count": sum(int(row["weights"]) for row in module_records),
            "rule": "DA-2K labels point1 as closer; Depth Anything V2 vits uses larger predicted values for closer points.",
        },
        "baseline": baseline,
        "pruning": pruning,
        "pruned": pruned,
    }
    (config.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate iterative unstructured WANDA pruning on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/da2k_vits_wanda_unstructured"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target", choices=["transformer", "all-linear"], default="transformer")
    parser.add_argument("--pruning-scope", choices=["per-matrix", "per-column", "global"], default="per-matrix")
    parser.add_argument(
        "--score-normalization",
        choices=["none", "matrix-mean"],
        default="none",
        help="For global pruning, optionally divide each matrix's remaining WANDA scores by its mean before global ranking.",
    )
    parser.add_argument("--layer-indices", default="", help="Comma-separated DINOv2 block indices to prune. Empty means all selected target modules.")
    parser.add_argument("--prune-fraction", type=float, default=0.5)
    parser.add_argument("--prune-chunk-fraction", type=float, default=0.05)
    parser.add_argument("--recompute-every-weights", type=int, default=0)
    parser.add_argument("--calibration-images", type=int, default=8)
    parser.add_argument("--exclude-calibration-from-eval", action="store_true")
    parser.add_argument("--repair-steps", type=int, default=0)
    parser.add_argument("--repair-lr", type=float, default=1e-5)
    parser.add_argument("--affine-repair", choices=["none", "after-each-chunk"], default="none")
    parser.add_argument("--affine-repair-tokens", type=int, default=4096)
    parser.add_argument("--affine-repair-scope", choices=["all-selected", "output-only"], default="all-selected")
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--skip-baseline", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = UnstructuredWandaConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        target=args.target,
        pruning_scope=args.pruning_scope,
        score_normalization=args.score_normalization,
        layer_indices=parse_int_tuple(args.layer_indices),
        prune_fraction=args.prune_fraction,
        prune_chunk_fraction=args.prune_chunk_fraction,
        recompute_every_weights=args.recompute_every_weights,
        calibration_images=args.calibration_images,
        exclude_calibration_from_eval=args.exclude_calibration_from_eval,
        repair_steps=args.repair_steps,
        repair_lr=args.repair_lr,
        affine_repair=args.affine_repair,
        affine_repair_tokens=args.affine_repair_tokens,
        affine_repair_scope=args.affine_repair_scope,
        scene_type=args.scene_type,
        max_images=args.max_images,
        log_every=args.log_every,
        eval_baseline=not args.skip_baseline,
    )
    summary = run(config)
    print(
        json.dumps(
            {
                "baseline": None if summary["baseline"] is None else summary["baseline"]["overall"],
                "pruned": summary["pruned"]["overall"],
                "pruning": {
                    key: value
                    for key, value in summary["pruning"].items()
                    if key not in {"step_summaries", "repair_summaries", "affine_repair_summaries", "pruned_tensors"}
                },
                "output_dir": str(config.output_dir),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
