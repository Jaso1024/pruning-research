from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

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


QuantMethod = Literal[
    "fp32",
    "fp16",
    "rtn-int8",
    "gptq-int8",
    "smoothquant-int8",
    "smoothquant-hessian-int8",
]
TargetKind = Literal["transformer", "all-linear"]
BaseDType = Literal["fp32", "fp16"]
WeightQuant = Literal["per-channel", "per-tensor", "per-group"]
ActivationQuant = Literal["per-token", "per-tensor", "per-token-group"]
RotationKind = Literal["none", "hadamard"]


@dataclass(frozen=True)
class QuantEvalConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    method: QuantMethod = "fp16"
    base_dtype: BaseDType = "fp16"
    target: TargetKind = "transformer"
    layer_indices: tuple[int, ...] = ()
    calibration_images: int = 8
    exclude_calibration_from_eval: bool = True
    scene_type: str = ""
    max_images: int = 0
    score_direction: str = "larger"
    log_every: int = 50
    eval_batch_size: int = 1
    weight_bits: int = 8
    activation_bits: int = 8
    weight_quant: WeightQuant = "per-channel"
    act_quant: ActivationQuant = "per-token"
    weight_group_size: int = 128
    activation_group_size: int = 128
    weight_clip_ratio: float = 1.0
    activation_clip_ratio: float = 1.0
    smooth_alpha: float = 0.5
    rotation: RotationKind = "none"
    rotation_group_size: int = 128
    rotation_seed: int = 0
    quantize_qkv_output: bool = False
    gptq_blocksize: int = 128
    gptq_percdamp: float = 0.01
    gptq_act_order: bool = True
    hessian_tokens_per_image: int = 128

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.method not in {
            "fp32",
            "fp16",
            "rtn-int8",
            "gptq-int8",
            "smoothquant-int8",
            "smoothquant-hessian-int8",
        }:
            raise ValueError(f"unknown quantization method: {self.method}")
        if self.base_dtype not in {"fp32", "fp16"}:
            raise ValueError("base_dtype must be fp32 or fp16")
        if self.target not in {"transformer", "all-linear"}:
            raise ValueError("target must be transformer or all-linear")
        if self.calibration_images < 0:
            raise ValueError("calibration_images must be non-negative")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.score_direction not in {"larger", "smaller", "best"}:
            raise ValueError("score_direction must be larger, smaller, or best")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        if not 2 <= self.weight_bits <= 16:
            raise ValueError("weight_bits must be in [2, 16]")
        if not 2 <= self.activation_bits <= 16:
            raise ValueError("activation_bits must be in [2, 16]")
        if self.weight_quant not in {"per-channel", "per-tensor", "per-group"}:
            raise ValueError("weight_quant must be per-channel, per-tensor, or per-group")
        if self.act_quant not in {"per-token", "per-tensor", "per-token-group"}:
            raise ValueError("act_quant must be per-token, per-tensor, or per-token-group")
        if self.weight_group_size <= 0:
            raise ValueError("weight_group_size must be positive")
        if self.activation_group_size <= 0:
            raise ValueError("activation_group_size must be positive")
        if not 0.0 < self.weight_clip_ratio <= 1.0:
            raise ValueError("weight_clip_ratio must be in (0, 1]")
        if not 0.0 < self.activation_clip_ratio <= 1.0:
            raise ValueError("activation_clip_ratio must be in (0, 1]")
        if not 0.0 <= self.smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha must be in [0, 1]")
        if self.rotation not in {"none", "hadamard"}:
            raise ValueError("rotation must be none or hadamard")
        if self.rotation_group_size <= 0:
            raise ValueError("rotation_group_size must be positive")
        if self.gptq_blocksize <= 0:
            raise ValueError("gptq_blocksize must be positive")
        if self.gptq_percdamp < 0.0:
            raise ValueError("gptq_percdamp must be non-negative")
        if self.hessian_tokens_per_image < 0:
            raise ValueError("hessian_tokens_per_image must be non-negative")


def parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


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


def find_target_linears(
    model: nn.Module,
    *,
    target: TargetKind,
    layer_indices: tuple[int, ...],
) -> list[str]:
    wanted_layers = set(layer_indices)
    module_names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        layer_index = transformer_layer_index(name)
        if target == "transformer" and layer_index is None:
            continue
        if wanted_layers and layer_index not in wanted_layers:
            continue
        module_names.append(name)
    return module_names


def model_compute_dtype(model: nn.Module) -> torch.dtype:
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def cast_model_for_method(model: nn.Module, config: QuantEvalConfig) -> torch.dtype:
    if config.method == "fp32":
        dtype = torch.float32
    elif config.method == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float16 if config.base_dtype == "fp16" else torch.float32
    if dtype == torch.float16:
        model.half()
    else:
        model.float()
    return dtype


def get_submodule_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    if "." not in module_name:
        return model, module_name
    parent_name, child_name = module_name.rsplit(".", 1)
    return model.get_submodule(parent_name), child_name


def set_submodule(model: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parent, child_name = get_submodule_parent(model, module_name)
    setattr(parent, child_name, replacement)


def linear_kind(module_name: str) -> str:
    if module_name.endswith(".attn.qkv"):
        return "qkv"
    if module_name.endswith(".attn.proj"):
        return "attn_proj"
    if module_name.endswith(".mlp.fc1"):
        return "fc1"
    if module_name.endswith(".mlp.fc2"):
        return "fc2"
    return "linear"


def load_calibration_tensors(
    model: nn.Module,
    *,
    dataset_root: Path,
    items: list[tuple[str, list[dict[str, Any]]]],
    input_size: int,
    device: torch.device,
    limit: int,
) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    dtype = model_compute_dtype(model)
    for relative_path, _pairs in items:
        if len(tensors) >= limit:
            break
        image = cv2.imread(str(dataset_root / relative_path))
        if image is None:
            continue
        tensor, _shape = model.image2tensor(image, input_size)
        tensors.append(tensor.to(device=device, dtype=dtype, non_blocking=True))
    if limit > 0 and not tensors:
        raise RuntimeError("no calibration images could be loaded")
    return tensors


def score_overall(overall: dict[str, Any], direction: str) -> float:
    if direction == "larger":
        return float(overall["larger_is_closer_accuracy"])
    if direction == "smaller":
        return float(overall["smaller_is_closer_accuracy"])
    if direction == "best":
        return float(overall["best_accuracy"])
    raise ValueError(f"unsupported score direction: {direction}")


@torch.no_grad()
def infer_depth(model: nn.Module, image, input_size: int, device: torch.device) -> torch.Tensor:
    tensor, (height, width) = model.image2tensor(image, input_size)
    tensor = tensor.to(device=device, dtype=model_compute_dtype(model), non_blocking=True)
    depth = model(tensor)
    depth = F.interpolate(
        depth.float()[:, None],
        (height, width),
        mode="bilinear",
        align_corners=True,
    )[0, 0]
    return depth.detach().float().cpu()


def evaluate_da2k_model(
    *,
    model: nn.Module,
    dataset_root: Path,
    items: list[tuple[str, list[dict[str, Any]]]],
    input_size: int,
    device: torch.device,
    log_every: int,
    batch_size: int,
) -> dict[str, Any]:
    total = empty_counts()
    by_scene = defaultdict(empty_counts)
    missing_images: list[str] = []
    started = time.monotonic()
    dtype = model_compute_dtype(model)
    completed = 0

    def score_depth(relative_path: str, pairs: list[dict[str, Any]], depth: torch.Tensor) -> None:
        scene = scene_from_path(relative_path)
        for pair in pairs:
            if pair.get("closer_point") != "point1":
                raise ValueError(f"unsupported closer_point in {relative_path}: {pair}")
            d1 = point_value(depth, pair["point1"])
            d2 = point_value(depth, pair["point2"])
            add_pair(total, d1, d2)
            add_pair(by_scene[scene], d1, d2)

    def maybe_log() -> None:
        if log_every > 0 and (completed % log_every == 0 or completed == len(items)):
            print(f"evaluated {completed}/{len(items)} images", flush=True)

    pending: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)

    def flush_shape(shape: tuple[int, int, int]) -> None:
        nonlocal completed
        bucket = pending.get(shape, [])
        if not bucket:
            return
        tensors = torch.cat([row["tensor"] for row in bucket], dim=0)
        with torch.inference_mode():
            depths = model(tensors)
        for index, row in enumerate(bucket):
            height, width = row["original_shape"]
            depth = F.interpolate(
                depths[index : index + 1].float()[:, None],
                (height, width),
                mode="bilinear",
                align_corners=True,
            )[0, 0]
            score_depth(str(row["relative_path"]), row["pairs"], depth.detach().float().cpu())
            completed += 1
            maybe_log()
        bucket.clear()

    for index, (relative_path, pairs) in enumerate(items, start=1):
        image_path = dataset_root / relative_path
        image = cv2.imread(str(image_path))
        if image is None:
            missing_images.append(str(image_path))
            continue
        tensor, original_shape = model.image2tensor(image, input_size)
        tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)
        shape = tuple(int(dim) for dim in tensor.shape[1:])
        pending[shape].append(
            {
                "tensor": tensor,
                "original_shape": original_shape,
                "relative_path": relative_path,
                "pairs": pairs,
            }
        )
        if len(pending[shape]) >= batch_size:
            flush_shape(shape)

    for shape in list(pending):
        flush_shape(shape)

    return {
        "metadata": {
            "images_requested": len(items),
            "missing_images": missing_images,
            "eval_batch_size": batch_size,
            "elapsed_seconds": time.monotonic() - started,
        },
        "overall": finalize_counts(total),
        "by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(by_scene.items())},
    }


def choose_hessian_tokens(flat: torch.Tensor, limit: int) -> torch.Tensor:
    if limit <= 0 or flat.shape[0] <= limit:
        return flat
    indices = torch.linspace(0, flat.shape[0] - 1, steps=limit, device=flat.device).long()
    return flat.index_select(0, indices)


def collect_linear_stats(
    *,
    model: nn.Module,
    module_names: list[str],
    image_tensors: list[torch.Tensor],
    device: torch.device,
    collect_hessian: bool,
    hessian_tokens_per_image: int,
) -> dict[str, Any]:
    modules = {name: model.get_submodule(name) for name in module_names}
    act_absmax = {
        name: torch.zeros(module.in_features, dtype=torch.float32)
        for name, module in modules.items()
        if isinstance(module, nn.Linear)
    }
    hessians = {
        name: torch.zeros((module.in_features, module.in_features), dtype=torch.float32)
        for name, module in modules.items()
        if collect_hessian and isinstance(module, nn.Linear)
    }
    hessian_counts = {name: 0 for name in hessians}
    activation_counts = {name: 0 for name in act_absmax}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not isinstance(module, nn.Linear):
                return
            x = inputs[0].detach()
            if x.shape[-1] != module.in_features:
                raise RuntimeError(f"input feature mismatch for {name}: {x.shape[-1]} != {module.in_features}")
            flat = x.reshape(-1, x.shape[-1]).float()
            act_absmax[name] = torch.maximum(act_absmax[name], flat.abs().amax(dim=0).cpu())
            activation_counts[name] += int(flat.shape[0])
            if collect_hessian:
                hflat = choose_hessian_tokens(flat, hessian_tokens_per_image)
                hessians[name] += hflat.transpose(0, 1).matmul(hflat).cpu()
                hessian_counts[name] += int(hflat.shape[0])

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for image_tensor in tqdm(image_tensors, desc="collect quant stats", unit="image"):
                model(image_tensor.to(device=device, dtype=model_compute_dtype(model), non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    stats: dict[str, Any] = {}
    for name in module_names:
        if activation_counts.get(name, 0) <= 0:
            raise RuntimeError(f"no calibration activations captured for {name}")
        if collect_hessian:
            count = hessian_counts[name]
            if count <= 0:
                raise RuntimeError(f"no Hessian activations captured for {name}")
            hessians[name].div_(float(count))
        stats[name] = {
            "activation_tokens": activation_counts[name],
            "activation_absmax_mean": float(act_absmax[name].mean().item()),
            "activation_absmax_max": float(act_absmax[name].max().item()),
            "hessian_tokens": hessian_counts.get(name, 0),
        }

    return {
        "act_absmax": act_absmax,
        "hessians": hessians,
        "module_stats": stats,
    }


def symmetric_qmax(bits: int) -> int:
    return (2 ** (bits - 1)) - 1


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def stable_name_seed(name: str, seed: int) -> int:
    total = int(seed)
    for index, char in enumerate(name):
        total += (index + 1) * ord(char)
    return total % (2**31)


def rotation_signs_for_module(
    *,
    name: str,
    in_features: int,
    rotation: RotationKind,
    group_size: int,
    seed: int,
) -> torch.Tensor | None:
    if rotation == "none":
        return None
    if rotation != "hadamard":
        raise ValueError(f"unsupported rotation: {rotation}")
    if not is_power_of_two(group_size):
        raise ValueError("hadamard rotation_group_size must be a power of two")
    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} is not divisible by rotation_group_size={group_size}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_name_seed(name, seed))
    signs = torch.randint(0, 2, (in_features,), generator=generator, dtype=torch.int8)
    return signs.float().mul_(2.0).sub_(1.0)


def normalized_hadamard_last_dim(x: torch.Tensor) -> torch.Tensor:
    size = x.shape[-1]
    if not is_power_of_two(size):
        raise ValueError("Hadamard size must be a power of two")
    y = x
    step = 1
    while step < size:
        y = y.reshape(*y.shape[:-1], size // (2 * step), 2, step)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-1], size)
        step *= 2
    return y / float(size**0.5)


def apply_block_hadamard(x: torch.Tensor, signs: torch.Tensor, group_size: int) -> torch.Tensor:
    if signs is None:
        return x
    dtype = x.dtype
    if x.shape[-1] != signs.numel():
        raise ValueError(f"rotation sign length mismatch: {signs.numel()} != {x.shape[-1]}")
    work = x.float() * signs.to(device=x.device, dtype=torch.float32).view(*([1] * (x.ndim - 1)), -1)
    work = work.reshape(*work.shape[:-1], work.shape[-1] // group_size, group_size)
    work = normalized_hadamard_last_dim(work)
    return work.reshape_as(x).to(dtype=dtype)


def transform_hessian_for_rotation(hessian: torch.Tensor, signs: torch.Tensor, group_size: int) -> torch.Tensor:
    if hessian.shape[0] != signs.numel() or hessian.shape[1] != signs.numel():
        raise ValueError(f"hessian/sign shape mismatch: {tuple(hessian.shape)} vs {signs.numel()}")
    rotated = apply_block_hadamard(hessian, signs.to(device=hessian.device), group_size)
    rotated = apply_block_hadamard(rotated.transpose(0, 1), signs.to(device=hessian.device), group_size).transpose(0, 1)
    return rotated


def grouped_absmax_scale(
    work: torch.Tensor,
    *,
    n_bits: int,
    group_size: int,
    clip_ratio: float,
) -> torch.Tensor:
    if work.shape[-1] % group_size != 0:
        raise ValueError(f"last dimension {work.shape[-1]} is not divisible by group_size={group_size}")
    qmax = symmetric_qmax(n_bits)
    grouped = work.reshape(*work.shape[:-1], work.shape[-1] // group_size, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True).mul(float(clip_ratio)).clamp_min(1e-8) / float(qmax)
    return scale.expand_as(grouped).reshape_as(work)


def fake_quant_activation_absmax(
    x: torch.Tensor,
    *,
    n_bits: int,
    granularity: ActivationQuant,
    group_size: int,
    clip_ratio: float,
) -> torch.Tensor:
    dtype = x.dtype
    work = x.float()
    qmax = symmetric_qmax(n_bits)
    if granularity == "per-token":
        scale = work.abs().amax(dim=-1, keepdim=True).mul(float(clip_ratio)).clamp_min(1e-8) / float(qmax)
    elif granularity == "per-token-group":
        scale = grouped_absmax_scale(work, n_bits=n_bits, group_size=group_size, clip_ratio=clip_ratio)
    elif granularity == "per-tensor":
        scale = work.abs().amax().mul(float(clip_ratio)).clamp_min(1e-8) / float(qmax)
    else:
        raise ValueError(f"unsupported activation quantization granularity: {granularity}")
    q = torch.clamp(torch.round(torch.clamp(work, -scale * qmax, scale * qmax) / scale), -qmax, qmax)
    return (q * scale).to(dtype=dtype)


def quantize_weight_absmax(
    weight: torch.Tensor,
    *,
    n_bits: int,
    granularity: WeightQuant,
    group_size: int,
    clip_ratio: float,
) -> torch.Tensor:
    dtype = weight.dtype
    work = weight.float()
    qmax = symmetric_qmax(n_bits)
    if granularity == "per-channel":
        scale = work.abs().amax(dim=1, keepdim=True).mul(float(clip_ratio)).clamp_min(1e-8) / float(qmax)
    elif granularity == "per-group":
        scale = grouped_absmax_scale(work, n_bits=n_bits, group_size=group_size, clip_ratio=clip_ratio)
    elif granularity == "per-tensor":
        scale = work.abs().amax().mul(float(clip_ratio)).clamp_min(1e-8) / float(qmax)
    else:
        raise ValueError(f"unsupported weight quantization granularity: {granularity}")
    q = torch.clamp(torch.round(torch.clamp(work, -scale * qmax, scale * qmax) / scale), -qmax, qmax)
    return (q * scale).to(dtype=dtype)


def quantize_weight_columns(
    columns: torch.Tensor,
    scales: torch.Tensor,
    *,
    n_bits: int,
    column_index: int | None = None,
) -> torch.Tensor:
    qmax = symmetric_qmax(n_bits)
    if scales.ndim == 2 and scales.shape[1] != 1:
        if column_index is None:
            scale = scales[:, : columns.shape[1]]
        else:
            scale = scales[:, column_index : column_index + columns.shape[1]]
    else:
        scale = scales
    q = torch.clamp(torch.round(torch.clamp(columns, -scale * qmax, scale * qmax) / scale), -qmax, qmax)
    return q * scale


def quantization_scales(
    weight: torch.Tensor,
    *,
    n_bits: int,
    granularity: WeightQuant,
    group_size: int,
    clip_ratio: float,
) -> torch.Tensor:
    qmax = symmetric_qmax(n_bits)
    if granularity == "per-channel":
        return weight.abs().amax(dim=1, keepdim=True).mul(float(clip_ratio)).clamp_min(1e-8) / float(qmax)
    if granularity == "per-group":
        return grouped_absmax_scale(weight.float(), n_bits=n_bits, group_size=group_size, clip_ratio=clip_ratio)
    if granularity == "per-tensor":
        return weight.abs().amax().mul(float(clip_ratio)).clamp_min(1e-8) / float(qmax)
    raise ValueError(f"unsupported weight quantization granularity: {granularity}")


def stable_cholesky_inverse_factor(
    hessian: torch.Tensor,
    *,
    percdamp: float,
) -> tuple[torch.Tensor, float]:
    hessian = hessian.float()
    diag = torch.diag(hessian)
    mean_diag = torch.mean(diag[diag > 0]) if torch.any(diag > 0) else torch.tensor(1.0, device=hessian.device)
    damp = float(percdamp) * float(mean_diag.item())
    if damp == 0.0:
        damp = 1e-6 * float(mean_diag.item())
    eye = torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)
    for attempt in range(6):
        try:
            h = hessian + eye * (damp * (10**attempt))
            chol = torch.linalg.cholesky(h)
            inv = torch.cholesky_inverse(chol)
            return torch.linalg.cholesky(inv, upper=True), damp * (10**attempt)
        except RuntimeError:
            continue
    h = hessian + eye * (damp * 1_000_000.0 + 1e-3)
    chol = torch.linalg.cholesky(h)
    inv = torch.cholesky_inverse(chol)
    return torch.linalg.cholesky(inv, upper=True), damp * 1_000_000.0 + 1e-3


def gptq_quantize_weight(
    *,
    weight: torch.Tensor,
    hessian: torch.Tensor,
    n_bits: int,
    granularity: WeightQuant,
    group_size: int,
    clip_ratio: float,
    blocksize: int,
    percdamp: float,
    act_order: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = weight.device
    original_dtype = weight.dtype
    weight_work = weight.detach().float().clone()
    hessian_work = hessian.to(device=device, dtype=torch.float32).clone()
    rows, columns = weight_work.shape

    diag = torch.diag(hessian_work)
    dead = diag <= 0
    if torch.any(dead):
        hessian_work[dead, dead] = 1
        weight_work[:, dead] = 0

    if act_order:
        perm = torch.argsort(torch.diag(hessian_work), descending=True)
        weight_work = weight_work[:, perm]
        hessian_work = hessian_work[perm][:, perm]
        invperm = torch.argsort(perm)
    else:
        perm = None
        invperm = None

    scales = quantization_scales(
        weight_work,
        n_bits=n_bits,
        granularity=granularity,
        group_size=group_size,
        clip_ratio=clip_ratio,
    )
    hessian_inv, damp_used = stable_cholesky_inverse_factor(hessian_work, percdamp=percdamp)

    quantized = torch.zeros_like(weight_work)
    losses = torch.zeros_like(weight_work)
    total_error = 0.0
    started = time.monotonic()

    for start in range(0, columns, blocksize):
        end = min(start + blocksize, columns)
        count = end - start
        block_weight = weight_work[:, start:end].clone()
        block_quant = torch.zeros_like(block_weight)
        block_err = torch.zeros_like(block_weight)
        block_losses = torch.zeros_like(block_weight)
        block_hinv = hessian_inv[start:end, start:end]

        for index in range(count):
            w = block_weight[:, index]
            d = block_hinv[index, index].clamp_min(1e-8)
            q = quantize_weight_columns(w.unsqueeze(1), scales, n_bits=n_bits, column_index=start + index).flatten()
            block_quant[:, index] = q
            block_losses[:, index] = (w - q).pow(2) / d.pow(2)
            err = (w - q) / d
            block_weight[:, index:] -= err.unsqueeze(1).matmul(block_hinv[index, index:].unsqueeze(0))
            block_err[:, index] = err

        quantized[:, start:end] = block_quant
        losses[:, start:end] = block_losses / 2
        if end < columns:
            weight_work[:, end:] -= block_err.matmul(hessian_inv[start:end, end:])

    if invperm is not None:
        quantized = quantized[:, invperm]
        losses = losses[:, invperm]
    total_error = float(losses.sum().detach().cpu().item())
    return quantized.to(dtype=original_dtype), {
        "rows": rows,
        "columns": columns,
        "bits": n_bits,
        "weight_quant": granularity,
        "weight_group_size": group_size,
        "weight_clip_ratio": clip_ratio,
        "blocksize": blocksize,
        "percdamp": percdamp,
        "damp_used": damp_used,
        "act_order": act_order,
        "dead_columns": int(dead.sum().item()),
        "gptq_loss": total_error,
        "elapsed_seconds": time.monotonic() - started,
    }


class FakeQuantLinear(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        act_bits: int | None,
        act_quant: ActivationQuant,
        act_group_size: int,
        act_clip_ratio: float,
        input_smooth_scale: torch.Tensor | None = None,
        rotation_signs: torch.Tensor | None = None,
        rotation_group_size: int = 128,
        quantize_output: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_bits = act_bits
        self.act_quant = act_quant
        self.act_group_size = act_group_size
        self.act_clip_ratio = act_clip_ratio
        self.rotation_group_size = rotation_group_size
        self.quantize_output = quantize_output
        self.register_buffer("weight", weight.detach().clone())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone())
        if input_smooth_scale is None:
            self.register_buffer("input_smooth_scale", None)
        else:
            self.register_buffer("input_smooth_scale", input_smooth_scale.detach().clone())
        if rotation_signs is None:
            self.register_buffer("rotation_signs", None)
        else:
            self.register_buffer("rotation_signs", rotation_signs.detach().clone())

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        weight: torch.Tensor | None,
        weight_bits: int,
        weight_quant: WeightQuant,
        weight_group_size: int,
        weight_clip_ratio: float,
        act_bits: int | None,
        act_quant: ActivationQuant,
        act_group_size: int,
        act_clip_ratio: float,
        input_smooth_scale: torch.Tensor | None = None,
        rotation_signs: torch.Tensor | None = None,
        rotation_group_size: int = 128,
        quantize_output: bool = False,
    ) -> "FakeQuantLinear":
        if weight is None:
            smooth_weight = module.weight.detach()
            if input_smooth_scale is not None:
                scale = input_smooth_scale.to(device=smooth_weight.device, dtype=smooth_weight.dtype)
                smooth_weight = smooth_weight * scale.view(1, -1)
            if rotation_signs is not None:
                smooth_weight = apply_block_hadamard(
                    smooth_weight,
                    rotation_signs.to(device=smooth_weight.device),
                    rotation_group_size,
                )
            weight = quantize_weight_absmax(
                smooth_weight,
                n_bits=weight_bits,
                granularity=weight_quant,
                group_size=weight_group_size,
                clip_ratio=weight_clip_ratio,
            )
        return cls(
            in_features=module.in_features,
            out_features=module.out_features,
            weight=weight.to(device=module.weight.device, dtype=module.weight.dtype),
            bias=module.bias.to(device=module.weight.device, dtype=module.weight.dtype) if module.bias is not None else None,
            act_bits=act_bits,
            act_quant=act_quant,
            act_group_size=act_group_size,
            act_clip_ratio=act_clip_ratio,
            input_smooth_scale=input_smooth_scale.to(device=module.weight.device, dtype=module.weight.dtype)
            if input_smooth_scale is not None
            else None,
            rotation_signs=rotation_signs.to(device=module.weight.device, dtype=module.weight.dtype)
            if rotation_signs is not None
            else None,
            rotation_group_size=rotation_group_size,
            quantize_output=quantize_output,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_smooth_scale is not None:
            scale = self.input_smooth_scale.to(device=x.device, dtype=x.dtype).view(*([1] * (x.ndim - 1)), -1)
            x = x / scale
        if self.rotation_signs is not None:
            x = apply_block_hadamard(x, self.rotation_signs.to(device=x.device, dtype=x.dtype), self.rotation_group_size)
        if self.act_bits is not None:
            x = fake_quant_activation_absmax(
                x,
                n_bits=self.act_bits,
                granularity=self.act_quant,
                group_size=self.act_group_size,
                clip_ratio=self.act_clip_ratio,
            )
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        output = F.linear(x, weight, bias)
        if self.quantize_output and self.act_bits is not None:
            output = fake_quant_activation_absmax(
                output,
                n_bits=self.act_bits,
                granularity=self.act_quant,
                group_size=self.act_group_size,
                clip_ratio=self.act_clip_ratio,
            )
        return output


def compute_smooth_scales(
    *,
    model: nn.Module,
    module_names: list[str],
    act_absmax: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    scales: dict[str, torch.Tensor] = {}
    for name in module_names:
        module = model.get_submodule(name)
        if not isinstance(module, nn.Linear):
            continue
        act_scale = act_absmax[name].float().clamp_min(1e-5)
        weight_scale = module.weight.detach().float().abs().amax(dim=0).cpu().clamp_min(1e-5)
        scales[name] = (act_scale.pow(alpha) / weight_scale.pow(1.0 - alpha)).clamp_min(1e-5)
    return scales


def transform_hessian_for_smoothing(hessian: torch.Tensor, scale: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if scale is None:
        return hessian.to(device=device, dtype=torch.float32)
    scale = scale.to(device=device, dtype=torch.float32).clamp_min(1e-5)
    return hessian.to(device=device, dtype=torch.float32) / scale.view(-1, 1) / scale.view(1, -1)


def transformed_input_hessian(
    *,
    hessian: torch.Tensor,
    scale: torch.Tensor | None,
    rotation_signs: torch.Tensor | None,
    rotation_group_size: int,
    device: torch.device,
) -> torch.Tensor:
    transformed = transform_hessian_for_smoothing(hessian, scale, device)
    if rotation_signs is not None:
        transformed = transform_hessian_for_rotation(
            transformed,
            rotation_signs.to(device=device, dtype=torch.float32),
            rotation_group_size,
        )
    return transformed


def apply_rtn_int8(
    *,
    model: nn.Module,
    module_names: list[str],
    config: QuantEvalConfig,
) -> dict[str, Any]:
    rows = []
    started = time.monotonic()
    for name in tqdm(module_names, desc="apply RTN int8", unit="module"):
        module = model.get_submodule(name)
        if not isinstance(module, nn.Linear):
            continue
        quant_module = FakeQuantLinear.from_linear(
            module,
            weight=None,
            weight_bits=config.weight_bits,
            weight_quant=config.weight_quant,
            weight_group_size=config.weight_group_size,
            weight_clip_ratio=config.weight_clip_ratio,
            act_bits=None,
            act_quant=config.act_quant,
            act_group_size=config.activation_group_size,
            act_clip_ratio=config.activation_clip_ratio,
        )
        set_submodule(model, name, quant_module)
        rows.append(
            {
                "name": name,
                "kind": linear_kind(name),
                "shape": list(module.weight.shape),
                "weights": int(module.weight.numel()),
            }
        )
    return {
        "method": config.method,
        "modules_quantized": len(rows),
        "weights_quantized": sum(int(row["weights"]) for row in rows),
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def apply_gptq_int8(
    *,
    model: nn.Module,
    module_names: list[str],
    hessians: dict[str, torch.Tensor],
    config: QuantEvalConfig,
    smooth_scales: dict[str, torch.Tensor] | None = None,
    w8a8: bool = False,
) -> dict[str, Any]:
    rows = []
    started = time.monotonic()
    for name in tqdm(module_names, desc="apply GPTQ int8", unit="module"):
        module = model.get_submodule(name)
        if not isinstance(module, nn.Linear):
            continue
        scale = smooth_scales[name] if smooth_scales is not None and name in smooth_scales else None
        rotation_signs = rotation_signs_for_module(
            name=name,
            in_features=module.in_features,
            rotation=config.rotation if w8a8 else "none",
            group_size=config.rotation_group_size,
            seed=config.rotation_seed,
        )
        weight = module.weight.detach()
        if scale is not None:
            weight = weight * scale.to(device=weight.device, dtype=weight.dtype).view(1, -1)
        if rotation_signs is not None:
            weight = apply_block_hadamard(weight, rotation_signs.to(device=weight.device), config.rotation_group_size)
        transformed_hessian = transformed_input_hessian(
            hessian=hessians[name],
            scale=scale,
            rotation_signs=rotation_signs,
            rotation_group_size=config.rotation_group_size,
            device=module.weight.device,
        )
        quantized_weight, qstats = gptq_quantize_weight(
            weight=weight,
            hessian=transformed_hessian,
            n_bits=config.weight_bits,
            granularity=config.weight_quant,
            group_size=config.weight_group_size,
            clip_ratio=config.weight_clip_ratio,
            blocksize=config.gptq_blocksize,
            percdamp=config.gptq_percdamp,
            act_order=config.gptq_act_order,
        )
        if w8a8:
            quant_module = FakeQuantLinear.from_linear(
                module,
                weight=quantized_weight,
                weight_bits=config.weight_bits,
                weight_quant=config.weight_quant,
                weight_group_size=config.weight_group_size,
                weight_clip_ratio=config.weight_clip_ratio,
                act_bits=config.activation_bits,
                act_quant=config.act_quant,
                act_group_size=config.activation_group_size,
                act_clip_ratio=config.activation_clip_ratio,
                input_smooth_scale=scale,
                rotation_signs=rotation_signs,
                rotation_group_size=config.rotation_group_size,
                quantize_output=config.quantize_qkv_output and linear_kind(name) == "qkv",
            )
            set_submodule(model, name, quant_module)
        else:
            module.weight.data.copy_(quantized_weight.to(dtype=module.weight.dtype))
        rows.append(
            {
                "name": name,
                "kind": linear_kind(name),
                "shape": list(module.weight.shape),
                "weights": int(module.weight.numel()),
                **qstats,
            }
        )
    return {
        "method": config.method,
        "modules_quantized": len(rows),
        "weights_quantized": sum(int(row["weights"]) for row in rows),
        "mean_gptq_loss": sum(float(row["gptq_loss"]) for row in rows) / max(len(rows), 1),
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def apply_smoothquant_int8(
    *,
    model: nn.Module,
    module_names: list[str],
    smooth_scales: dict[str, torch.Tensor],
    config: QuantEvalConfig,
) -> dict[str, Any]:
    rows = []
    started = time.monotonic()
    for name in tqdm(module_names, desc="apply SmoothQuant int8", unit="module"):
        module = model.get_submodule(name)
        if not isinstance(module, nn.Linear):
            continue
        scale = smooth_scales[name]
        rotation_signs = rotation_signs_for_module(
            name=name,
            in_features=module.in_features,
            rotation=config.rotation,
            group_size=config.rotation_group_size,
            seed=config.rotation_seed,
        )
        quant_module = FakeQuantLinear.from_linear(
            module,
            weight=None,
            weight_bits=config.weight_bits,
            weight_quant=config.weight_quant,
            weight_group_size=config.weight_group_size,
            weight_clip_ratio=config.weight_clip_ratio,
            act_bits=config.activation_bits,
            act_quant=config.act_quant,
            act_group_size=config.activation_group_size,
            act_clip_ratio=config.activation_clip_ratio,
            input_smooth_scale=scale,
            rotation_signs=rotation_signs,
            rotation_group_size=config.rotation_group_size,
            quantize_output=config.quantize_qkv_output and linear_kind(name) == "qkv",
        )
        set_submodule(model, name, quant_module)
        rows.append(
            {
                "name": name,
                "kind": linear_kind(name),
                "shape": list(module.weight.shape),
                "weights": int(module.weight.numel()),
                "smooth_scale_mean": float(scale.mean().item()),
                "smooth_scale_max": float(scale.max().item()),
                "rotation": config.rotation,
            }
        )
    return {
        "method": config.method,
        "modules_quantized": len(rows),
        "weights_quantized": sum(int(row["weights"]) for row in rows),
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def apply_quantization(
    *,
    model: nn.Module,
    module_names: list[str],
    calibration_tensors: list[torch.Tensor],
    device: torch.device,
    config: QuantEvalConfig,
) -> dict[str, Any]:
    if config.method in {"fp32", "fp16"}:
        return {
            "method": config.method,
            "modules_quantized": 0,
            "weights_quantized": 0,
            "elapsed_seconds": 0.0,
        }
    if config.method == "rtn-int8":
        return apply_rtn_int8(model=model, module_names=module_names, config=config)

    needs_hessian = config.method in {"gptq-int8", "smoothquant-hessian-int8"}
    needs_smooth = config.method in {"smoothquant-int8", "smoothquant-hessian-int8"}
    stats = collect_linear_stats(
        model=model,
        module_names=module_names,
        image_tensors=calibration_tensors,
        device=device,
        collect_hessian=needs_hessian,
        hessian_tokens_per_image=config.hessian_tokens_per_image,
    )

    smooth_scales: dict[str, torch.Tensor] | None = None
    if needs_smooth:
        smooth_scales = compute_smooth_scales(
            model=model,
            module_names=module_names,
            act_absmax=stats["act_absmax"],
            alpha=config.smooth_alpha,
        )

    if config.method == "rtn-int8":
        quant_summary = apply_rtn_int8(model=model, module_names=module_names, config=config)
    elif config.method == "gptq-int8":
        quant_summary = apply_gptq_int8(
            model=model,
            module_names=module_names,
            hessians=stats["hessians"],
            config=config,
            smooth_scales=None,
            w8a8=False,
        )
    elif config.method == "smoothquant-int8":
        if smooth_scales is None:
            raise RuntimeError("internal error: missing SmoothQuant scales")
        quant_summary = apply_smoothquant_int8(
            model=model,
            module_names=module_names,
            smooth_scales=smooth_scales,
            config=config,
        )
    elif config.method == "smoothquant-hessian-int8":
        if smooth_scales is None:
            raise RuntimeError("internal error: missing SmoothQuant scales")
        quant_summary = apply_gptq_int8(
            model=model,
            module_names=module_names,
            hessians=stats["hessians"],
            config=config,
            smooth_scales=smooth_scales,
            w8a8=True,
        )
    else:
        raise ValueError(f"unsupported method: {config.method}")

    quant_summary["calibration_stats"] = stats["module_stats"]
    if smooth_scales is not None:
        quant_summary["smooth_alpha"] = config.smooth_alpha
        quant_summary["smooth_scale_summary"] = {
            "mean": sum(float(scale.mean().item()) for scale in smooth_scales.values()) / max(len(smooth_scales), 1),
            "max": max((float(scale.max().item()) for scale in smooth_scales.values()), default=0.0),
        }
    quant_summary["quantization_options"] = {
        "weight_bits": config.weight_bits,
        "activation_bits": config.activation_bits,
        "weight_quant": config.weight_quant,
        "act_quant": config.act_quant,
        "weight_group_size": config.weight_group_size,
        "activation_group_size": config.activation_group_size,
        "weight_clip_ratio": config.weight_clip_ratio,
        "activation_clip_ratio": config.activation_clip_ratio,
        "rotation": config.rotation,
        "rotation_group_size": config.rotation_group_size,
        "rotation_seed": config.rotation_seed,
    }
    return quant_summary


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["result"]["overall"]
    quant = summary["quantization"]
    lines = [
        f"# DA-2K Quantization: {summary['config']['method']}",
        "",
        "| field | value |",
        "| --- | ---: |",
        f"| score direction | {summary['config']['score_direction']} |",
        f"| score | {summary['score']} |",
        f"| pairs | {overall['pairs']} |",
        f"| larger acc | {overall['larger_is_closer_accuracy']} |",
        f"| smaller acc | {overall['smaller_is_closer_accuracy']} |",
        f"| best direction | {overall['best_direction']} |",
        f"| best acc | {overall['best_accuracy']} |",
        f"| eval images | {summary['result']['metadata']['images_requested']} |",
        f"| calibration images | {summary['metadata']['calibration_images_used']} |",
        f"| modules quantized | {quant.get('modules_quantized', 0)} |",
        f"| weights quantized | {quant.get('weights_quantized', 0)} |",
        f"| elapsed seconds | {summary['metadata']['elapsed_seconds']} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: QuantEvalConfig) -> dict[str, Any]:
    started = time.monotonic()
    device = resolve_device(config.device)
    model = load_model(config.encoder, config.checkpoint, device)
    dtype = cast_model_for_method(model, config)
    module_names = find_target_linears(model, target=config.target, layer_indices=config.layer_indices)
    if config.method not in {"fp32", "fp16"} and not module_names:
        raise RuntimeError("no linear modules selected for quantization")

    selected = selected_annotations(config.dataset_root, scene_type=config.scene_type, max_images=config.max_images)
    if not selected:
        raise RuntimeError("no DA-2K items selected")
    calibration_items = selected[: config.calibration_images] if config.calibration_images > 0 else []
    if config.exclude_calibration_from_eval and config.calibration_images > 0:
        eval_items = selected[config.calibration_images :]
    else:
        eval_items = selected
    if not eval_items:
        raise RuntimeError("no evaluation items left after calibration exclusion")

    calibration_tensors = load_calibration_tensors(
        model,
        dataset_root=config.dataset_root,
        items=calibration_items,
        input_size=config.input_size,
        device=device,
        limit=config.calibration_images,
    )
    quant_summary = apply_quantization(
        model=model,
        module_names=module_names,
        calibration_tensors=calibration_tensors,
        device=device,
        config=config,
    )

    result = evaluate_da2k_model(
        model=model,
        dataset_root=config.dataset_root,
        items=eval_items,
        input_size=config.input_size,
        device=device,
        log_every=config.log_every,
        batch_size=config.eval_batch_size,
    )
    score = score_overall(result["overall"], config.score_direction)

    summary = {
        "config": asdict(config),
        "metadata": {
            "device": str(device),
            "dtype": str(dtype),
            "target_modules": module_names,
            "target_module_count": len(module_names),
            "selected_images": len(selected),
            "calibration_images_used": len(calibration_tensors),
            "calibration_excluded_from_eval": config.exclude_calibration_from_eval,
            "elapsed_seconds": time.monotonic() - started,
            "rule": "DA-2K labels point1 as closer; for this vits checkpoint larger predicted values are the verified convention.",
            "implementation_note": (
                "INT methods use PyTorch fake quantization/dequantization for benchmark accuracy, not custom int GEMM kernels. "
                "GPTQ is weight-only unless method is smoothquant-hessian-int8; SmoothQuant methods quantize linear inputs dynamically."
            ),
        },
        "quantization": quant_summary,
        "score": score,
        "result": result,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    write_summary_markdown(config.output_dir / "summary.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FP16, GPTQ, and SmoothQuant variants on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--method",
        choices=[
            "fp32",
            "fp16",
            "rtn-int8",
            "gptq-int8",
            "smoothquant-int8",
            "smoothquant-hessian-int8",
        ],
        default="fp16",
    )
    parser.add_argument("--base-dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--target", choices=["transformer", "all-linear"], default="transformer")
    parser.add_argument("--layer-indices", default="")
    parser.add_argument("--calibration-images", type=int, default=8)
    parser.add_argument(
        "--include-calibration-in-eval",
        action="store_true",
        help="Evaluate on calibration images too. By default calibration images are held out from DA-2K scoring.",
    )
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--score-direction", choices=["larger", "smaller", "best"], default="larger")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--weight-quant", choices=["per-channel", "per-tensor", "per-group"], default="per-channel")
    parser.add_argument("--act-quant", choices=["per-token", "per-tensor", "per-token-group"], default="per-token")
    parser.add_argument("--weight-group-size", type=int, default=128)
    parser.add_argument("--activation-group-size", type=int, default=128)
    parser.add_argument("--weight-clip-ratio", type=float, default=1.0)
    parser.add_argument("--activation-clip-ratio", type=float, default=1.0)
    parser.add_argument("--smooth-alpha", type=float, default=0.5)
    parser.add_argument("--rotation", choices=["none", "hadamard"], default="none")
    parser.add_argument("--rotation-group-size", type=int, default=128)
    parser.add_argument("--rotation-seed", type=int, default=0)
    parser.add_argument("--quantize-qkv-output", action="store_true")
    parser.add_argument("--gptq-blocksize", type=int, default=128)
    parser.add_argument("--gptq-percdamp", type=float, default=0.01)
    parser.add_argument("--no-gptq-act-order", action="store_true")
    parser.add_argument("--hessian-tokens-per-image", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = QuantEvalConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        method=args.method,
        base_dtype=args.base_dtype,
        target=args.target,
        layer_indices=parse_int_tuple(args.layer_indices),
        calibration_images=args.calibration_images,
        exclude_calibration_from_eval=not args.include_calibration_in_eval,
        scene_type=args.scene_type,
        max_images=args.max_images,
        score_direction=args.score_direction,
        log_every=args.log_every,
        eval_batch_size=args.eval_batch_size,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        weight_quant=args.weight_quant,
        act_quant=args.act_quant,
        weight_group_size=args.weight_group_size,
        activation_group_size=args.activation_group_size,
        weight_clip_ratio=args.weight_clip_ratio,
        activation_clip_ratio=args.activation_clip_ratio,
        smooth_alpha=args.smooth_alpha,
        rotation=args.rotation,
        rotation_group_size=args.rotation_group_size,
        rotation_seed=args.rotation_seed,
        quantize_qkv_output=args.quantize_qkv_output,
        gptq_blocksize=args.gptq_blocksize,
        gptq_percdamp=args.gptq_percdamp,
        gptq_act_order=not args.no_gptq_act_order,
        hessian_tokens_per_image=args.hessian_tokens_per_image,
    )
    summary = run(config)
    print(json.dumps({"score": summary["score"], "overall": summary["result"]["overall"]}, indent=2))


if __name__ == "__main__":
    main()
