from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
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


def find_repo_root(start: Path) -> Path:
    start_dir = start if start.is_dir() else start.parent
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "layer_composition").is_dir():
            return candidate
    return start_dir


REPO_ROOT = find_repo_root(Path(__file__).resolve())
LAYER_COMPOSITION_ROOT = REPO_ROOT / "layer_composition"
if str(LAYER_COMPOSITION_ROOT) not in sys.path:
    sys.path.insert(0, str(LAYER_COMPOSITION_ROOT))

from layer_distill.sparse24 import sparsify_weight_2_4  # noqa: E402


NEEDS_FP_INPUTS = {
    "gptaq-cae",
    "gptaq-cae-diag",
    "gptaq-cae-gd",
    "qronos",
    "rescomp",
    "rescomp-diag",
    "rescomp-gd",
}


@dataclass(frozen=True)
class LayerGroup:
    layer_name: str
    layer_index: int
    module_names: tuple[str, ...]


@dataclass(frozen=True)
class BeamConfig:
    output_dir: Path
    dataset_root: Path
    checkpoint: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    method: str = "wanda"
    beam_width: int = 2
    max_depth: int = 2
    max_images: int = 25
    scene_type: str = ""
    calibration_images: int = 8
    calibration_tokens: int = 4096
    score_direction: str = "larger"
    search_layers: tuple[int, ...] = ()
    max_search_layers: int = 0
    sparsity_n: int = 2
    sparsity_m: int = 4
    damp: float = 0.01
    blocksize: int = 128
    alpha: float = 0.25
    cae_alpha: float = 0.25
    gd_steps: int = 1
    gd_lr: float = 0.25
    gd_chunk_tokens: int = 8192
    linear_repair: str = "none"
    repair_tokens: int = 4096

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.calibration_images <= 0:
            raise ValueError("calibration_images must be positive")
        if self.calibration_tokens <= 0:
            raise ValueError("calibration_tokens must be positive")
        if self.score_direction not in {"larger", "smaller", "best"}:
            raise ValueError("score_direction must be larger, smaller, or best")
        if self.sparsity_n <= 0 or self.sparsity_m <= 0 or self.sparsity_n >= self.sparsity_m:
            raise ValueError("sparsity_n and sparsity_m must satisfy 0 < n < m")
        if self.blocksize <= 0 or self.blocksize % self.sparsity_m != 0:
            raise ValueError("blocksize must be positive and divisible by sparsity_m")
        if self.linear_repair not in {"none", "output-only", "per-linear"}:
            raise ValueError("linear_repair must be none, output-only, or per-linear")
        if self.repair_tokens <= 0:
            raise ValueError("repair_tokens must be positive")


def state_name(state: tuple[int, ...]) -> str:
    return "baseline" if not state else "layers_" + "_".join(f"{idx:02d}" for idx in state)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def select_annotation_items(
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


def find_depth_transformer_groups(model: torch.nn.Module, *, sparsity_m: int) -> list[LayerGroup]:
    grouped: dict[int, list[str]] = {}
    prefix = "pretrained.blocks."
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not name.startswith(prefix):
            continue
        if module.weight.ndim != 2 or module.weight.shape[1] % sparsity_m != 0:
            continue
        parts = name.split(".")
        if len(parts) < 3:
            continue
        try:
            layer_index = int(parts[2])
        except ValueError:
            continue
        grouped.setdefault(layer_index, []).append(name)
    return [
        LayerGroup(
            layer_name=f"pretrained.blocks.{layer_index}",
            layer_index=layer_index,
            module_names=tuple(names),
        )
        for layer_index, names in sorted(grouped.items())
    ]


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
        tensors.append(tensor.to(device=device))
    if not tensors:
        raise RuntimeError("no calibration images could be loaded")
    return tensors


def collect_linear_inputs(
    *,
    model: torch.nn.Module,
    module_name: str,
    image_tensors: list[torch.Tensor],
    device: torch.device,
    max_tokens: int,
) -> torch.Tensor:
    module = model.get_submodule(module_name)
    captured: list[torch.Tensor] = []
    token_count = 0

    def hook(_module, inputs, _output):
        nonlocal token_count
        if token_count >= max_tokens:
            return
        x = inputs[0].detach()
        flat = x.reshape(-1, x.shape[-1]).float().cpu()
        remaining = max_tokens - token_count
        if flat.shape[0] > remaining:
            flat = flat[:remaining]
        captured.append(flat)
        token_count += flat.shape[0]

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


def should_apply_linear_repair(module_name: str, mode: str) -> bool:
    if mode == "none":
        return False
    if mode == "output-only":
        return module_name.endswith(".attn.proj") or module_name.endswith(".mlp.fc2")
    if mode == "per-linear":
        return True
    raise ValueError(f"unknown linear repair mode: {mode}")


@torch.no_grad()
def fit_affine_output_repair(
    sparse_module: torch.nn.Linear,
    dense_module: torch.nn.Linear,
    x: torch.Tensor,
    max_tokens: int,
) -> dict[str, float]:
    device = sparse_module.weight.device
    sample = x[:max_tokens].to(device=device, dtype=torch.float32)

    dense_weight = dense_module.weight.detach().to(device=device, dtype=torch.float32)
    dense_bias = (
        dense_module.bias.detach().to(device=device, dtype=torch.float32)
        if dense_module.bias is not None
        else None
    )
    sparse_weight = sparse_module.weight.detach().to(device=device, dtype=torch.float32)
    sparse_bias = (
        sparse_module.bias.detach().to(device=device, dtype=torch.float32)
        if sparse_module.bias is not None
        else None
    )

    target = torch.nn.functional.linear(sample, dense_weight, dense_bias)
    pred = torch.nn.functional.linear(sample, sparse_weight, sparse_bias)

    before_mse = torch.nn.functional.mse_loss(pred, target)
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

    after_mse = torch.nn.functional.mse_loss(repaired_pred, target)
    after_rel_mse = after_mse / target_power

    weight_dtype = sparse_module.weight.dtype
    sparse_module.weight.mul_(scale.to(dtype=weight_dtype).view(-1, 1))
    if sparse_module.bias is None:
        sparse_module.bias = torch.nn.Parameter(
            shift.to(device=device, dtype=weight_dtype),
            requires_grad=sparse_module.weight.requires_grad,
        )
    else:
        sparse_module.bias.mul_(scale.to(dtype=sparse_module.bias.dtype))
        sparse_module.bias.add_(shift.to(dtype=sparse_module.bias.dtype))

    return {
        "repair_before_mse": float(before_mse.cpu()),
        "repair_after_mse": float(after_mse.cpu()),
        "repair_before_rel_mse": float(before_rel_mse.cpu()),
        "repair_after_rel_mse": float(after_rel_mse.cpu()),
        "repair_scale_mean": float(scale.mean().cpu()),
        "repair_scale_min": float(scale.min().cpu()),
        "repair_scale_max": float(scale.max().cpu()),
        "repair_shift_abs_mean": float(shift.abs().mean().cpu()),
    }


def reconstruction_stats(
    linear: torch.nn.Linear,
    dense_linear: torch.nn.Linear,
    x: torch.Tensor,
    *,
    max_tokens: int = 4096,
) -> dict[str, float]:
    sample = x[:max_tokens].to(device=linear.weight.device, dtype=torch.float32)
    dense = dense_linear.weight.detach().to(device=linear.weight.device, dtype=torch.float32)
    sparse = linear.weight.detach().to(device=linear.weight.device, dtype=torch.float32)
    dense_bias = (
        dense_linear.bias.detach().to(device=linear.weight.device, dtype=torch.float32)
        if dense_linear.bias is not None
        else None
    )
    target = torch.nn.functional.linear(sample, dense, dense_bias)
    pred = sample @ sparse.t()
    if linear.bias is not None:
        bias = linear.bias.detach().float()
        pred = pred + bias
    mse = torch.nn.functional.mse_loss(pred, target)
    rel = mse / target.pow(2).mean().clamp_min(1e-12)
    return {"recon_mse": float(mse.cpu()), "recon_rel_mse": float(rel.cpu())}


def summarize_module_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_nonzeros = sum(int(row["nonzeros"]) for row in records)
    total_weights = sum(int(row["total"]) for row in records)
    return {
        "sparsified_module_count": len(records),
        "sparsified_modules": [row["module_name"] for row in records],
        "mean_module_recon_rel_mse": sum(float(row["recon_rel_mse"]) for row in records) / len(records)
        if records
        else math.nan,
        "mean_weight_rel_mse": sum(float(row["weight_rel_mse"]) for row in records) / len(records)
        if records
        else math.nan,
        "density": total_nonzeros / total_weights if total_weights else math.nan,
    }


def sparsify_layer_group(
    *,
    fp_model: torch.nn.Module,
    sparse_model: torch.nn.Module,
    group: LayerGroup,
    calibration_tensors: list[torch.Tensor],
    config: BeamConfig,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    modules_path = output_dir / "modules.jsonl"
    for module_index, module_name in enumerate(group.module_names, start=1):
        module = sparse_model.get_submodule(module_name)
        dense_module = fp_model.get_submodule(module_name)
        if not isinstance(module, torch.nn.Linear) or not isinstance(dense_module, torch.nn.Linear):
            continue
        start = time.monotonic()
        x_quant = collect_linear_inputs(
            model=sparse_model,
            module_name=module_name,
            image_tensors=calibration_tensors,
            device=device,
            max_tokens=config.calibration_tokens,
        )
        x_fp = None
        if config.method in NEEDS_FP_INPUTS:
            x_fp = collect_linear_inputs(
                model=fp_model,
                module_name=module_name,
                image_tensors=calibration_tensors,
                device=device,
                max_tokens=config.calibration_tokens,
            )
        sparse_weight, stats = sparsify_weight_2_4(
            dense_module.weight.detach(),
            x_quant=x_quant.to(device=device, non_blocking=True),
            x_fp=x_fp.to(device=device, non_blocking=True) if x_fp is not None else None,
            method=config.method,
            damp=config.damp,
            blocksize=config.blocksize,
            sparsity_n=config.sparsity_n,
            sparsity_m=config.sparsity_m,
            alpha=config.alpha,
            cae_alpha=config.cae_alpha,
            gd_steps=config.gd_steps,
            gd_lr=config.gd_lr,
            gd_chunk_tokens=config.gd_chunk_tokens,
        )
        module.weight.data.copy_(sparse_weight.to(device=module.weight.device, dtype=module.weight.dtype))
        repair_applied = should_apply_linear_repair(module_name, config.linear_repair)
        repair_stats = (
            fit_affine_output_repair(module, dense_module, x_quant, config.repair_tokens)
            if repair_applied
            else {}
        )
        record = {
            "layer_index": group.layer_index,
            "layer_name": group.layer_name,
            "module_index": module_index,
            "module_count": len(group.module_names),
            "module_name": module_name,
            "linear_repair": config.linear_repair,
            "repair_applied": repair_applied,
            "elapsed_sec": time.monotonic() - start,
            **stats,
            **repair_stats,
            **reconstruction_stats(module, dense_module, x_quant),
        }
        append_jsonl(modules_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)
        records.append(record)
        del x_quant, x_fp, sparse_weight
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records


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


def score_record(record: dict[str, Any], direction: str) -> float:
    overall = record["overall"]
    if direction == "best":
        return float(overall["best_accuracy"])
    return float(overall[f"{direction}_is_closer_accuracy"])


def rank_records(records: list[dict[str, Any]], *, direction: str) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: (
            -score_record(row, direction),
            -float(row["overall"]["best_accuracy"]),
            float(row["overall"]["tie_fraction"]),
            list(row["layer_indices"]),
        ),
    )


def read_state_record(output_dir: Path, state: tuple[int, ...]) -> dict[str, Any] | None:
    path = output_dir / "states" / state_name(state) / "summary.json"
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if tuple(int(idx) for idx in record.get("layer_indices", [])) != state:
        return None
    return record


def write_state_record(output_dir: Path, record: dict[str, Any]) -> None:
    state_dir = output_dir / "states" / str(record["state_name"])
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "summary.json").write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")


def evaluate_state(
    *,
    fp_model: torch.nn.Module,
    state: tuple[int, ...],
    groups_by_index: dict[int, LayerGroup],
    calibration_tensors: list[torch.Tensor],
    eval_items: list[tuple[str, list[dict[str, Any]]]],
    config: BeamConfig,
    device: torch.device,
) -> dict[str, Any]:
    existing = read_state_record(config.output_dir, state)
    if existing is not None:
        return existing

    state_dir = config.output_dir / "states" / state_name(state)
    state_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    if not state:
        sparse_model = fp_model
    else:
        sparse_model = copy.deepcopy(fp_model).to(device).eval()
        for param in sparse_model.parameters():
            param.requires_grad_(False)

    module_records: list[dict[str, Any]] = []
    try:
        for order_index, layer_index in enumerate(state, start=1):
            group = groups_by_index[layer_index]
            layer_records = sparsify_layer_group(
                fp_model=fp_model,
                sparse_model=sparse_model,
                group=group,
                calibration_tensors=calibration_tensors,
                config=config,
                device=device,
                output_dir=state_dir / f"apply_{order_index:02d}_layer_{layer_index:02d}",
            )
            for record in layer_records:
                record["layer_apply_order"] = order_index
            module_records.extend(layer_records)
        eval_result = evaluate_da2k_model(
            model=sparse_model,
            dataset_root=config.dataset_root,
            items=eval_items,
            input_size=config.input_size,
            device=device,
            log_every=0,
        )
    finally:
        if state:
            del sparse_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    record = {
        "state_name": state_name(state),
        "layer_indices": list(state),
        "layer_count": len(state),
        "method": config.method,
        "score_direction": config.score_direction,
        "score": score_record(eval_result, config.score_direction),
        "sparsify_eval_elapsed_sec": time.monotonic() - start,
        **eval_result,
        **summarize_module_records(module_records),
    }
    write_state_record(config.output_dir, record)
    print(json.dumps(record, sort_keys=True, default=str), flush=True)
    return record


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    direction = summary["config"]["score_direction"]
    lines = [
        f"Ranking metric: `{direction}` DA-2K accuracy.",
        "",
        "| depth | rank | state | score | larger acc | smaller acc | pairs | density | mean recon rel mse |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline = summary["baseline"]
    rows = [{"depth": 0, "rank": 1, **baseline}]
    for depth_row in summary["beam"]:
        for rank, record in enumerate(depth_row["states"], start=1):
            rows.append({"depth": depth_row["depth"], "rank": rank, **record})
    for row in rows:
        overall = row["overall"]
        density = row.get("density")
        recon = row.get("mean_module_recon_rel_mse")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["depth"]),
                    str(row["rank"]),
                    row["state_name"],
                    f"{float(row['score']):.6f}",
                    f"{float(overall['larger_is_closer_accuracy']):.6f}",
                    f"{float(overall['smaller_is_closer_accuracy']):.6f}",
                    str(overall["pairs"]),
                    "" if density is None or math.isnan(float(density)) else f"{float(density):.3f}",
                    "" if recon is None or math.isnan(float(recon)) else f"{float(recon):.6f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def run_beam_search(config: BeamConfig) -> dict[str, Any]:
    torch.manual_seed(17)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True, default=str) + "\n"
    )

    eval_items = select_annotation_items(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
    )
    if not eval_items:
        raise RuntimeError("no DA-2K annotations selected")

    fp_model = load_model(config.encoder, config.checkpoint, device)
    for param in fp_model.parameters():
        param.requires_grad_(False)

    groups = find_depth_transformer_groups(fp_model, sparsity_m=config.sparsity_m)
    if config.search_layers:
        wanted = set(config.search_layers)
        groups = [group for group in groups if group.layer_index in wanted]
    if config.max_search_layers > 0:
        groups = groups[: config.max_search_layers]
    if not groups:
        raise RuntimeError("no prunable DINOv2 transformer layer groups found")
    layer_indices = tuple(group.layer_index for group in groups)
    groups_by_index = {group.layer_index: group for group in groups}
    (config.output_dir / "layer_groups.json").write_text(
        json.dumps([dataclasses.asdict(group) for group in groups], indent=2, sort_keys=True) + "\n"
    )

    calibration_tensors = load_calibration_tensors(
        fp_model,
        dataset_root=config.dataset_root,
        items=eval_items,
        input_size=config.input_size,
        device=device,
        limit=config.calibration_images,
    )

    baseline = evaluate_state(
        fp_model=fp_model,
        state=(),
        groups_by_index=groups_by_index,
        calibration_tensors=calibration_tensors,
        eval_items=eval_items,
        config=config,
        device=device,
    )
    append_jsonl(config.output_dir / "states.jsonl", baseline)

    beam_states: list[tuple[int, ...]] = [()]
    beam_records: list[dict[str, Any]] = []
    all_records: dict[tuple[int, ...], dict[str, Any]] = {(): baseline}
    for depth in range(1, min(config.max_depth, len(layer_indices)) + 1):
        candidate_states: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for state in beam_states:
            for layer_index in layer_indices:
                if layer_index in state:
                    continue
                candidate = state + (layer_index,)
                if candidate in seen:
                    continue
                seen.add(candidate)
                candidate_states.append(candidate)

        depth_records: list[dict[str, Any]] = []
        for state in tqdm(candidate_states, desc=f"beam depth {depth}"):
            record = evaluate_state(
                fp_model=fp_model,
                state=state,
                groups_by_index=groups_by_index,
                calibration_tensors=calibration_tensors,
                eval_items=eval_items,
                config=config,
                device=device,
            )
            depth_records.append(record)
            all_records[state] = record
            append_jsonl(config.output_dir / "states.jsonl", record)

        ranked = rank_records(depth_records, direction=config.score_direction)
        beam_records_for_depth = ranked[: config.beam_width]
        beam_states = [tuple(int(idx) for idx in record["layer_indices"]) for record in beam_records_for_depth]
        depth_summary = {
            "depth": depth,
            "candidate_count": len(depth_records),
            "states": beam_records_for_depth,
        }
        beam_records.append(depth_summary)
        append_jsonl(config.output_dir / "beam.jsonl", depth_summary)

    records = list(all_records.values())
    best_record = rank_records(records, direction=config.score_direction)[0]
    sparse_records = [record for record in records if int(record["layer_count"]) > 0]
    best_sparse_record = rank_records(sparse_records, direction=config.score_direction)[0] if sparse_records else None
    summary = {
        "config": dataclasses.asdict(config),
        "device": str(device),
        "layer_indices": list(layer_indices),
        "layer_count": len(layer_indices),
        "baseline": baseline,
        "beam": beam_records,
        "best_record": best_record,
        "best_sparse_record": best_sparse_record,
        "records": rank_records(records, direction=config.score_direction),
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    write_markdown(config.output_dir / "summary.md", summary)
    return summary


def run_fixed_state_eval(config: BeamConfig, state: tuple[int, ...]) -> dict[str, Any]:
    torch.manual_seed(17)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps({**dataclasses.asdict(config), "eval_state": list(state)}, indent=2, sort_keys=True, default=str) + "\n"
    )

    eval_items = select_annotation_items(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
    )
    if not eval_items:
        raise RuntimeError("no DA-2K annotations selected")

    fp_model = load_model(config.encoder, config.checkpoint, device)
    for param in fp_model.parameters():
        param.requires_grad_(False)

    groups = find_depth_transformer_groups(fp_model, sparsity_m=config.sparsity_m)
    if config.search_layers:
        wanted = set(config.search_layers)
        groups = [group for group in groups if group.layer_index in wanted]
    if config.max_search_layers > 0:
        groups = groups[: config.max_search_layers]
    groups_by_index = {group.layer_index: group for group in groups}
    missing = [layer_index for layer_index in state if layer_index not in groups_by_index]
    if missing:
        raise ValueError(f"eval state contains layer(s) outside the selected groups: {missing}")
    (config.output_dir / "layer_groups.json").write_text(
        json.dumps([dataclasses.asdict(group) for group in groups], indent=2, sort_keys=True) + "\n"
    )

    calibration_tensors = load_calibration_tensors(
        fp_model,
        dataset_root=config.dataset_root,
        items=eval_items,
        input_size=config.input_size,
        device=device,
        limit=config.calibration_images,
    )
    record = evaluate_state(
        fp_model=fp_model,
        state=state,
        groups_by_index=groups_by_index,
        calibration_tensors=calibration_tensors,
        eval_items=eval_items,
        config=config,
        device=device,
    )
    append_jsonl(config.output_dir / "states.jsonl", record)
    summary = {
        "config": {**dataclasses.asdict(config), "eval_state": list(state)},
        "device": str(device),
        "layer_indices": [group.layer_index for group in groups],
        "record": record,
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Beam search DINOv2 layer subsets for 2:4 sparse Depth Anything V2.")
    parser.add_argument("--output-dir", type=Path, default=Path("beam_outputs/da2k_vits_sparse24_beam"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--method", default="wanda")
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=25)
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--calibration-images", type=int, default=8)
    parser.add_argument("--calibration-tokens", type=int, default=4096)
    parser.add_argument("--score-direction", choices=["larger", "smaller", "best"], default="larger")
    parser.add_argument("--search-layers", default="", help="Comma-separated DINOv2 block indices to search. Empty means all.")
    parser.add_argument("--max-search-layers", type=int, default=0, help="Limit search to the first N selected blocks.")
    parser.add_argument("--eval-state", default="", help="Comma-separated ordered layer state to evaluate without beam search.")
    parser.add_argument("--sparsity-n", type=int, default=2)
    parser.add_argument("--sparsity-m", type=int, default=4)
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--cae-alpha", type=float, default=0.25)
    parser.add_argument("--gd-steps", type=int, default=1)
    parser.add_argument("--gd-lr", type=float, default=0.25)
    parser.add_argument("--gd-chunk-tokens", type=int, default=8192)
    parser.add_argument("--linear-repair", choices=["none", "output-only", "per-linear"], default="none")
    parser.add_argument("--repair-tokens", type=int, default=4096)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = BeamConfig(
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        method=args.method,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        max_images=args.max_images,
        scene_type=args.scene_type,
        calibration_images=args.calibration_images,
        calibration_tokens=args.calibration_tokens,
        score_direction=args.score_direction,
        search_layers=parse_int_tuple(args.search_layers),
        max_search_layers=args.max_search_layers,
        sparsity_n=args.sparsity_n,
        sparsity_m=args.sparsity_m,
        damp=args.damp,
        blocksize=args.blocksize,
        alpha=args.alpha,
        cae_alpha=args.cae_alpha,
        gd_steps=args.gd_steps,
        gd_lr=args.gd_lr,
        gd_chunk_tokens=args.gd_chunk_tokens,
        linear_repair=args.linear_repair,
        repair_tokens=args.repair_tokens,
    )
    eval_state = parse_int_tuple(args.eval_state)
    if args.eval_state.strip():
        summary = run_fixed_state_eval(config, eval_state)
        print(json.dumps({"record": summary["record"], "output_dir": str(config.output_dir)}, indent=2, sort_keys=True))
        return
    summary = run_beam_search(config)
    print(
        json.dumps(
            {
                "best_record": summary["best_record"],
                "best_sparse_record": summary["best_sparse_record"],
                "output_dir": str(config.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
