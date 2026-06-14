from __future__ import annotations

import argparse
import gc
import json
import math
import random
import types
import time
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from eval_da2k import MODEL_CONFIGS, add_pair, empty_counts, finalize_counts, point_value, resolve_device, scene_from_path
from eval_gelu_relu_compensation_da2k import load_model, selected_annotations

try:
    from eval_relu_strikes_da2k import activation_spec, install_mlp_activation, install_stage2
    from eval_relu_strikes_state_da2k import metadata_from_summary
except ModuleNotFoundError:
    activation_spec = None
    install_mlp_activation = None
    install_stage2 = None
    metadata_from_summary = None


@dataclass(frozen=True)
class CircuitNodeSpec:
    name: str
    kind: str
    module_name: str
    layer_index: int | None


@dataclass(frozen=True)
class AttributionPatchingConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    node_types: tuple[str, ...] = ("attn", "mlp", "head")
    components: tuple[str, ...] = ()
    corruption: str = "point_mask"
    mask_radius: int = 32
    blur_kernel: int = 41
    scene_type: str = ""
    max_images: int = 16
    max_pairs: int = 0
    image_start: int = 0
    image_count: int = 0
    seed: int = 123
    activation: str = "original"
    stage2: str = "none"
    stage2_shift: float = 0.0
    summary_json: Path | None = None
    variant_key: str = ""
    state_dict: Path | None = None
    node_batch_size: int = 0
    scoring_method: str = "atp"
    relp_rules: tuple[str, ...] = ("ln", "act", "attn")
    log_every: int = 8

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
        allowed_node_types = {"block", "attn", "mlp", "linear", "mlp_linear", "head", "head_conv"}
        unknown_node_types = sorted(set(self.node_types) - allowed_node_types)
        if unknown_node_types:
            raise ValueError(f"unknown node type(s): {unknown_node_types}")
        allowed_corruptions = {"point_mask", "blur", "mean", "gray", "black", "noise", "patch_shuffle"}
        if self.corruption not in allowed_corruptions:
            raise ValueError(f"corruption must be one of {sorted(allowed_corruptions)}")
        if self.mask_radius < 1:
            raise ValueError("mask_radius must be positive")
        if self.blur_kernel < 3:
            raise ValueError("blur_kernel must be >= 3")
        if self.max_images < 0 or self.max_pairs < 0 or self.image_start < 0 or self.image_count < 0:
            raise ValueError("image and pair limits must be non-negative")
        if self.stage2 not in {"none", "norm2", "norm12"}:
            raise ValueError("stage2 must be one of none, norm2, norm12")
        if self.activation.strip().lower() == "original" and self.stage2 != "none":
            raise ValueError("stage2 requires a relu/shift activation, not activation=original")
        if self.node_batch_size < 0:
            raise ValueError("node_batch_size must be non-negative")
        if self.scoring_method not in {"atp", "relp"}:
            raise ValueError("scoring_method must be one of atp or relp")
        allowed_relp_rules = {"ln", "act", "attn"}
        unknown_relp_rules = sorted(set(self.relp_rules) - allowed_relp_rules)
        if unknown_relp_rules:
            raise ValueError(f"unknown relp rule(s): {unknown_relp_rules}")


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


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


def odd_kernel(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def make_corrupted_image(
    image: np.ndarray,
    pairs: list[dict[str, Any]],
    *,
    mode: str,
    mask_radius: int,
    blur_kernel: int,
    rng: random.Random,
) -> np.ndarray:
    if mode == "blur":
        return cv2.GaussianBlur(image, (odd_kernel(blur_kernel), odd_kernel(blur_kernel)), 0)
    if mode == "mean":
        mean = image.reshape(-1, image.shape[-1]).mean(axis=0)
        return np.full_like(image, np.round(mean).astype(np.uint8))
    if mode == "gray":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode == "black":
        return np.zeros_like(image)
    if mode == "noise":
        mean = image.reshape(-1, image.shape[-1]).mean(axis=0)
        std = np.maximum(image.reshape(-1, image.shape[-1]).std(axis=0), 1.0)
        noise = rng.normalvariate
        flat = np.empty_like(image.reshape(-1, image.shape[-1]), dtype=np.float32)
        for channel in range(image.shape[-1]):
            flat[:, channel] = [noise(float(mean[channel]), float(std[channel])) for _ in range(flat.shape[0])]
        return np.clip(flat.reshape(image.shape), 0, 255).astype(np.uint8)
    if mode == "patch_shuffle":
        patch = 28
        h, w = image.shape[:2]
        out = image.copy()
        tiles: list[np.ndarray] = []
        coords: list[tuple[int, int]] = []
        for y in range(0, h - patch + 1, patch):
            for x in range(0, w - patch + 1, patch):
                tiles.append(out[y : y + patch, x : x + patch].copy())
                coords.append((y, x))
        rng.shuffle(tiles)
        for (y, x), tile in zip(coords, tiles):
            out[y : y + patch, x : x + patch] = tile
        return out
    if mode == "point_mask":
        blurred = cv2.GaussianBlur(image, (odd_kernel(blur_kernel), odd_kernel(blur_kernel)), 0)
        out = image.copy()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for pair in pairs:
            for key in ("point1", "point2"):
                row, col = pair[key]
                cv2.circle(mask, (int(col), int(row)), int(mask_radius), 255, thickness=-1)
        out[mask > 0] = blurred[mask > 0]
        return out
    raise ValueError(f"unsupported corruption mode: {mode}")


def image_to_tensor(model: torch.nn.Module, image: np.ndarray, input_size: int, device: torch.device) -> tuple[torch.Tensor, int, int]:
    tensor, (height, width) = model.image2tensor(image, input_size)
    return tensor.to(device), int(height), int(width)


def pair_margin_from_depth(depth: torch.Tensor, pairs: list[dict[str, Any]]) -> torch.Tensor:
    margins: list[torch.Tensor] = []
    height, width = depth.shape[-2:]
    for pair in pairs:
        if pair.get("closer_point") != "point1":
            raise ValueError(f"unsupported closer_point: {pair}")
        row1 = max(0, min(int(pair["point1"][0]), height - 1))
        col1 = max(0, min(int(pair["point1"][1]), width - 1))
        row2 = max(0, min(int(pair["point2"][0]), height - 1))
        col2 = max(0, min(int(pair["point2"][1]), width - 1))
        margins.append(depth[row1, col1] - depth[row2, col2])
    if not margins:
        return depth.new_tensor(0.0)
    return torch.stack(margins).mean()


def differentiable_depth_and_margin(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    height: int,
    width: int,
    pairs: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    depth = model(tensor)
    depth = F.interpolate(depth[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
    return depth, pair_margin_from_depth(depth, pairs)


@torch.no_grad()
def evaluate_depth_pair_counts(
    *,
    model: torch.nn.Module,
    image: np.ndarray,
    pairs: list[dict[str, Any]],
    input_size: int,
    device: torch.device,
) -> dict[str, Any]:
    tensor, height, width = image_to_tensor(model, image, input_size, device)
    depth = model(tensor)
    depth = F.interpolate(depth[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
    counts = empty_counts()
    margins: list[float] = []
    for pair in pairs:
        d1 = point_value(depth.detach().float().cpu(), pair["point1"])
        d2 = point_value(depth.detach().float().cpu(), pair["point2"])
        add_pair(counts, d1, d2)
        margins.append(d1 - d2)
    return {
        "counts": counts,
        "mean_margin": float(sum(margins) / max(len(margins), 1)),
        "margins": margins,
    }


def tensor_from_output(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = tensor_from_output(item)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for item in output.values():
            tensor = tensor_from_output(item)
            if tensor is not None:
                return tensor
    return None


def stabilize_for_lrp(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sign = torch.where(tensor >= 0, torch.ones_like(tensor), -torch.ones_like(tensor))
    return tensor + sign * eps


def patch_gelu_identity_rule(module: torch.nn.GELU) -> None:
    original_forward = module.forward

    def relp_forward(x: torch.Tensor) -> torch.Tensor:
        output = original_forward(x)
        proxy = stabilize_for_lrp(x)
        return proxy * (output / proxy).detach()

    module.forward = relp_forward  # type: ignore[method-assign]


def patch_layernorm_ln_rule(module: torch.nn.LayerNorm) -> None:
    normalized_ndim = len(tuple(module.normalized_shape))

    def relp_forward(x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(x.ndim - normalized_ndim, x.ndim))
        centered = x - x.mean(dim=dims, keepdim=True)
        scale = torch.sqrt(centered.pow(2).mean(dim=dims, keepdim=True) + module.eps).detach()
        output = centered / scale
        if module.elementwise_affine:
            output = output * module.weight + module.bias
        return output

    module.forward = relp_forward  # type: ignore[method-assign]


def is_dinov2_attention(module: torch.nn.Module) -> bool:
    return (
        module.__class__.__name__ in {"Attention", "MemEffAttention"}
        and hasattr(module, "qkv")
        and hasattr(module, "proj")
        and hasattr(module, "num_heads")
        and hasattr(module, "scale")
    )


def patch_attention_ah_rule(module: torch.nn.Module) -> None:
    def relp_forward(self: torch.nn.Module, x: torch.Tensor, attn_bias: Any = None) -> torch.Tensor:
        if attn_bias is not None:
            raise AssertionError("RelP AH-rule fallback does not support xFormers attention bias")
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attention = q @ k.transpose(-2, -1)
        attention = attention.softmax(dim=-1).detach()
        attention = self.attn_drop(attention)
        output = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output

    module.forward = types.MethodType(relp_forward, module)  # type: ignore[method-assign]


def install_relp_rules(model: torch.nn.Module, rules: tuple[str, ...]) -> dict[str, int]:
    wanted = set(rules)
    installed = {"layernorm_ln_rule": 0, "activation_identity_rule": 0, "attention_ah_rule": 0}
    for module in model.modules():
        if "attn" in wanted and is_dinov2_attention(module):
            patch_attention_ah_rule(module)
            installed["attention_ah_rule"] += 1
        elif "ln" in wanted and isinstance(module, torch.nn.LayerNorm):
            patch_layernorm_ln_rule(module)
            installed["layernorm_ln_rule"] += 1
        elif "act" in wanted and isinstance(module, torch.nn.GELU):
            patch_gelu_identity_rule(module)
            installed["activation_identity_rule"] += 1
    return installed


class ActivationPatchingHooks(AbstractContextManager["ActivationPatchingHooks"]):
    def __init__(self, model: torch.nn.Module, specs: list[CircuitNodeSpec]) -> None:
        self.model = model
        self.specs = specs
        self.phase = "off"
        self.clean: dict[str, torch.Tensor] = {}
        self.corrupt: dict[str, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "ActivationPatchingHooks":
        for spec in self.specs:
            module = self.model.get_submodule(spec.module_name)
            self.handles.append(module.register_forward_hook(self._hook(spec.name)))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clean.clear()
        self.corrupt.clear()
        self.phase = "off"

    def _hook(self, name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = tensor_from_output(output)
            if tensor is None:
                return
            if self.phase == "clean":
                self.clean[name] = tensor.detach().float().cpu()
            elif self.phase == "corrupt" and tensor.requires_grad:
                tensor.retain_grad()
                self.corrupt[name] = tensor

        return hook

    def clear(self) -> None:
        self.clean.clear()
        self.corrupt.clear()


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


def add_if_module(model: torch.nn.Module, specs: list[CircuitNodeSpec], name: str, kind: str, module_name: str, layer_index: int | None) -> None:
    try:
        model.get_submodule(module_name)
    except AttributeError:
        return
    specs.append(CircuitNodeSpec(name=name, kind=kind, module_name=module_name, layer_index=layer_index))


def build_circuit_nodes(model: torch.nn.Module, *, node_types: tuple[str, ...]) -> list[CircuitNodeSpec]:
    wanted = set(node_types)
    specs: list[CircuitNodeSpec] = []
    if {"block", "attn", "mlp", "linear", "mlp_linear"} & wanted:
        for layer_index, block in enumerate(model.pretrained.blocks):
            base = f"pretrained.blocks.{layer_index}"
            if "block" in wanted:
                specs.append(CircuitNodeSpec(f"block_{layer_index:02d}", "block", base, layer_index))
            if "attn" in wanted and hasattr(block, "attn"):
                specs.append(CircuitNodeSpec(f"block_{layer_index:02d}_attn", "attn", f"{base}.attn", layer_index))
            if "mlp" in wanted and hasattr(block, "mlp"):
                specs.append(CircuitNodeSpec(f"block_{layer_index:02d}_mlp", "mlp", f"{base}.mlp", layer_index))
            if "mlp_linear" in wanted and hasattr(block, "mlp"):
                add_if_module(model, specs, f"block_{layer_index:02d}_fc1", "mlp_linear", f"{base}.mlp.fc1", layer_index)
                add_if_module(model, specs, f"block_{layer_index:02d}_fc2", "mlp_linear", f"{base}.mlp.fc2", layer_index)
            if "linear" in wanted:
                add_if_module(model, specs, f"block_{layer_index:02d}_qkv", "linear", f"{base}.attn.qkv", layer_index)
                add_if_module(model, specs, f"block_{layer_index:02d}_proj", "linear", f"{base}.attn.proj", layer_index)
                add_if_module(model, specs, f"block_{layer_index:02d}_fc1", "linear", f"{base}.mlp.fc1", layer_index)
                add_if_module(model, specs, f"block_{layer_index:02d}_fc2", "linear", f"{base}.mlp.fc2", layer_index)
    if "head" in wanted:
        for index in range(4):
            add_if_module(model, specs, f"head_project_{index}", "head", f"depth_head.projects.{index}", None)
            add_if_module(model, specs, f"head_resize_{index}", "head", f"depth_head.resize_layers.{index}", None)
        for name in ("layer1_rn", "layer2_rn", "layer3_rn", "layer4_rn"):
            add_if_module(model, specs, f"head_scratch_{name}", "head", f"depth_head.scratch.{name}", None)
        for index in range(1, 5):
            add_if_module(model, specs, f"head_refinenet_{index}", "head", f"depth_head.scratch.refinenet{index}", None)
        add_if_module(model, specs, "head_output_conv1", "head", "depth_head.scratch.output_conv1", None)
        add_if_module(model, specs, "head_output_conv2", "head", "depth_head.scratch.output_conv2", None)
    if "head_conv" in wanted:
        for module_name, module in model.named_modules():
            if not module_name.startswith("depth_head.") or not isinstance(module, torch.nn.Conv2d):
                continue
            specs.append(CircuitNodeSpec(module_name.replace(".", "_"), "head_conv", module_name, None))
    return specs


def load_attribution_model(config: AttributionPatchingConfig, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if config.summary_json is not None:
        if metadata_from_summary is None:
            raise RuntimeError("summary/state reluification helpers are unavailable in this checkout")
        activation, stage2, stage2_shift = metadata_from_summary(config.summary_json, config.variant_key)
    else:
        if config.activation.strip().lower() != "original" and activation_spec is None:
            raise RuntimeError("activation replacement helpers are unavailable in this checkout")
        activation = activation_spec(config.activation) if config.activation.strip().lower() != "original" else None
        stage2 = config.stage2
        stage2_shift = config.stage2_shift

    model = load_model(config.encoder, config.checkpoint, device)
    load_summary: dict[str, Any] = {
        "checkpoint": str(config.checkpoint),
        "encoder": config.encoder,
        "activation": "original" if activation is None else activation.__dict__,
        "stage2": stage2,
        "stage2_shift": stage2_shift,
    }
    if activation is not None:
        if install_mlp_activation is None:
            raise RuntimeError("activation replacement helpers are unavailable in this checkout")
        load_summary["changed_mlp_modules"] = install_mlp_activation(model, activation)
    if stage2 != "none":
        if install_stage2 is None:
            raise RuntimeError("stage2 reluification helpers are unavailable in this checkout")
        load_summary["changed_stage2_modules"] = install_stage2(model, mode=stage2, shift=stage2_shift)
    if config.state_dict is not None:
        state = torch.load(config.state_dict, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"state dict mismatch: missing={missing}, unexpected={unexpected}")
        load_summary["state_dict"] = str(config.state_dict)
    if config.summary_json is not None:
        load_summary["summary_json"] = str(config.summary_json)
        load_summary["variant_key"] = config.variant_key
    if config.scoring_method == "relp":
        load_summary["relp_rules"] = install_relp_rules(model, config.relp_rules)
    for param in model.parameters():
        param.requires_grad_(False)
    return model.to(device=device).eval(), load_summary


def attribution_rows_for_image(
    *,
    model: torch.nn.Module,
    specs_by_name: dict[str, CircuitNodeSpec],
    hooks: ActivationPatchingHooks,
    clean_image: np.ndarray,
    corrupted_image: np.ndarray,
    pairs: list[dict[str, Any]],
    input_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hooks.clear()
    clean_eval = evaluate_depth_pair_counts(
        model=model,
        image=clean_image,
        pairs=pairs,
        input_size=input_size,
        device=device,
    )
    corrupt_eval = evaluate_depth_pair_counts(
        model=model,
        image=corrupted_image,
        pairs=pairs,
        input_size=input_size,
        device=device,
    )

    with torch.no_grad():
        clean_tensor, clean_height, clean_width = image_to_tensor(model, clean_image, input_size, device)
        hooks.phase = "clean"
        _clean_depth, clean_metric = differentiable_depth_and_margin(model, clean_tensor, clean_height, clean_width, pairs)
    hooks.phase = "off"
    del clean_tensor, _clean_depth

    corrupt_tensor, corrupt_height, corrupt_width = image_to_tensor(model, corrupted_image, input_size, device)
    corrupt_tensor.requires_grad_(True)
    hooks.phase = "corrupt"
    corrupt_depth, corrupt_metric = differentiable_depth_and_margin(model, corrupt_tensor, corrupt_height, corrupt_width, pairs)
    hooks.phase = "off"
    model.zero_grad(set_to_none=True)
    corrupt_metric.backward()

    rows: list[dict[str, Any]] = []
    for name, corrupt_act in hooks.corrupt.items():
        clean_act = hooks.clean.get(name)
        grad = corrupt_act.grad
        if clean_act is None or grad is None:
            continue
        if tuple(clean_act.shape) != tuple(corrupt_act.shape):
            continue
        clean_device = clean_act.to(device=corrupt_act.device, dtype=corrupt_act.dtype, non_blocking=True)
        delta = clean_device - corrupt_act.detach()
        grad_detached = grad.detach()
        attr = delta * grad_detached
        attr_sum = float(attr.sum().item())
        attr_abs = float(attr.abs().sum().item())
        attr_pos = float(attr.clamp_min(0).sum().item())
        attr_neg = float(attr.clamp_max(0).sum().item())
        delta_l2 = float(delta.float().pow(2).sum().sqrt().item())
        grad_l2 = float(grad_detached.float().pow(2).sum().sqrt().item())
        numel = int(delta.numel())
        spec = specs_by_name[name]
        rows.append(
            {
                "component": spec.name,
                "kind": spec.kind,
                "module_name": spec.module_name,
                "layer_index": spec.layer_index,
                "attribution": attr_sum,
                "abs_attribution": attr_abs,
                "positive_attribution": attr_pos,
                "negative_attribution": attr_neg,
                "mean_attribution": attr_sum / max(numel, 1),
                "delta_l2": delta_l2,
                "grad_l2": grad_l2,
                "numel": numel,
            }
        )

    image_summary = {
        "clean": {
            "overall": finalize_counts(clean_eval["counts"]),
            "mean_margin": clean_eval["mean_margin"],
            "metric": float(clean_metric.detach().item()),
        },
        "corrupted": {
            "overall": finalize_counts(corrupt_eval["counts"]),
            "mean_margin": corrupt_eval["mean_margin"],
            "metric": float(corrupt_metric.detach().item()),
        },
        "estimated_patch_gain_sum": float(sum(row["attribution"] for row in rows)),
    }
    del corrupt_tensor, corrupt_depth, corrupt_metric
    hooks.clear()
    model.zero_grad(set_to_none=True)
    return image_summary, rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accum: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["component"]
        if key not in accum:
            accum[key] = {
                "component": row["component"],
                "kind": row["kind"],
                "module_name": row["module_name"],
                "layer_index": row["layer_index"],
                "images": 0,
                "attribution": 0.0,
                "abs_attribution": 0.0,
                "positive_attribution": 0.0,
                "negative_attribution": 0.0,
                "delta_l2": 0.0,
                "grad_l2": 0.0,
                "numel": 0,
            }
        item = accum[key]
        item["images"] += 1
        for field in ("attribution", "abs_attribution", "positive_attribution", "negative_attribution", "delta_l2", "grad_l2"):
            item[field] += float(row[field])
        item["numel"] += int(row["numel"])
    aggregated = []
    for item in accum.values():
        images = max(int(item["images"]), 1)
        item["mean_attribution_per_image"] = item["attribution"] / images
        item["mean_abs_attribution_per_image"] = item["abs_attribution"] / images
        item["mean_positive_attribution_per_image"] = item["positive_attribution"] / images
        item["mean_negative_attribution_per_image"] = item["negative_attribution"] / images
        item["mean_attribution_per_value"] = item["attribution"] / max(int(item["numel"]), 1)
        aggregated.append(item)
    return aggregated


def circuit_cover(rows_by_positive: list[dict[str, Any]], fractions: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)) -> dict[str, Any]:
    positive_rows = [row for row in rows_by_positive if float(row["positive_attribution"]) > 0.0]
    total = sum(float(row["positive_attribution"]) for row in positive_rows)
    covers: dict[str, Any] = {}
    for fraction in fractions:
        threshold = total * fraction
        running = 0.0
        kept: list[str] = []
        for row in positive_rows:
            running += float(row["positive_attribution"])
            kept.append(str(row["component"]))
            if running >= threshold:
                break
        covers[f"{int(round(fraction * 100))}pct_positive_mass"] = {
            "node_count": len(kept),
            "positive_mass": running,
            "components": kept,
        }
    return {
        "total_positive_mass": total,
        "covers": covers,
    }


def layer_rollup(aggregated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rollup: dict[str, dict[str, Any]] = {}
    for row in aggregated:
        layer = row["layer_index"]
        key = "head" if layer is None else f"block_{int(layer):02d}"
        item = rollup.setdefault(
            key,
            {
                "layer": key,
                "layer_index": layer,
                "nodes": 0,
                "attribution": 0.0,
                "abs_attribution": 0.0,
                "positive_attribution": 0.0,
                "negative_attribution": 0.0,
            },
        )
        item["nodes"] += 1
        for field in ("attribution", "abs_attribution", "positive_attribution", "negative_attribution"):
            item[field] += float(row[field])
    return sorted(rollup.values(), key=lambda row: (-float(row["positive_attribution"]), str(row["layer"])))


def batched_specs(specs: list[CircuitNodeSpec], batch_size: int) -> list[list[CircuitNodeSpec]]:
    if batch_size <= 0 or batch_size >= len(specs):
        return [specs]
    return [specs[index : index + batch_size] for index in range(0, len(specs), batch_size)]


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows_by_positive"][:40]
    lines = [
        "# Attribution Patching Circuit Search",
        "",
        f"Corruption: `{summary['config']['corruption']}`.",
        f"Clean accuracy: `{summary['clean_overall']['larger_is_closer_accuracy']:.6f}` larger-is-closer.",
        f"Corrupted accuracy: `{summary['corrupted_overall']['larger_is_closer_accuracy']:.6f}` larger-is-closer.",
        "",
        "## Top Positive Circuit Nodes",
        "",
        "| rank | component | kind | layer | positive | attribution | abs | images |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        layer = "" if row["layer_index"] is None else str(row["layer_index"])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    row["component"],
                    row["kind"],
                    layer,
                    f"{float(row['positive_attribution']):.6g}",
                    f"{float(row['attribution']):.6g}",
                    f"{float(row['abs_attribution']):.6g}",
                    str(row["images"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Layer Rollup",
            "",
            "| rank | layer | nodes | positive | attribution | abs |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, row in enumerate(summary["layer_rollup"], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    row["layer"],
                    str(row["nodes"]),
                    f"{float(row['positive_attribution']):.6g}",
                    f"{float(row['attribution']):.6g}",
                    f"{float(row['abs_attribution']):.6g}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def run(config: AttributionPatchingConfig) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    started = time.monotonic()
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True, default=str) + "\n")

    items = selected_annotations(config.dataset_root, scene_type=config.scene_type, max_images=config.max_images, max_pairs=config.max_pairs)
    if config.image_start or config.image_count:
        end = None if config.image_count == 0 else config.image_start + config.image_count
        items = items[config.image_start : end]
    if not items:
        raise RuntimeError("no DA-2K annotations selected")

    model, load_summary = load_attribution_model(config, device)
    specs = build_circuit_nodes(model, node_types=config.node_types)
    if config.components:
        wanted = set(config.components)
        specs = [spec for spec in specs if spec.name in wanted or spec.module_name in wanted]
        missing = sorted(wanted - {spec.name for spec in specs} - {spec.module_name for spec in specs})
        if missing:
            raise ValueError(f"requested component(s) not found: {missing}")
    if not specs:
        raise RuntimeError("no circuit nodes selected")
    specs_by_name = {spec.name: spec for spec in specs}
    (config.output_dir / "nodes.json").write_text(json.dumps([asdict(spec) for spec in specs], indent=2, sort_keys=True) + "\n")

    all_rows: list[dict[str, Any]] = []
    image_rows_path = config.output_dir / "image_rows.jsonl"
    node_rows_path = config.output_dir / "node_rows.jsonl"
    clean_counts = empty_counts()
    corrupt_counts = empty_counts()
    clean_by_scene = defaultdict(empty_counts)
    corrupt_by_scene = defaultdict(empty_counts)
    image_summaries: list[dict[str, Any]] = []
    missing_images: list[str] = []

    spec_batches = batched_specs(specs, config.node_batch_size)

    for index, (relative_path, pairs) in enumerate(tqdm(items, desc="attribution patching", unit="image"), start=1):
        image_path = config.dataset_root / relative_path
        image = cv2.imread(str(image_path))
        if image is None:
            missing_images.append(str(image_path))
            continue
        rng = random.Random(config.seed + index)
        corrupted = make_corrupted_image(
            image,
            pairs,
            mode=config.corruption,
            mask_radius=config.mask_radius,
            blur_kernel=config.blur_kernel,
            rng=rng,
        )
        scene = scene_from_path(relative_path)
        image_summary: dict[str, Any] | None = None
        rows: list[dict[str, Any]] = []
        for batch_index, spec_batch in enumerate(spec_batches):
            specs_by_batch_name = {spec.name: spec for spec in spec_batch}
            with ActivationPatchingHooks(model, spec_batch) as hooks:
                batch_summary, batch_rows = attribution_rows_for_image(
                    model=model,
                    specs_by_name=specs_by_batch_name,
                    hooks=hooks,
                    clean_image=image,
                    corrupted_image=corrupted,
                    pairs=pairs,
                    input_size=config.input_size,
                    device=device,
                )
            if batch_index == 0:
                image_summary = batch_summary
            rows.extend(batch_rows)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
        if image_summary is None:
            continue
        for key, target in (("clean", clean_counts), ("corrupted", corrupt_counts)):
            counts = image_summary[key]["overall"]
            target["pairs"] += int(counts["pairs"])
            target["smaller_correct"] += int(counts["smaller_correct"])
            target["larger_correct"] += int(counts["larger_correct"])
            target["ties"] += int(counts["ties"])
            scene_counts = clean_by_scene[scene] if key == "clean" else corrupt_by_scene[scene]
            scene_counts["pairs"] += int(counts["pairs"])
            scene_counts["smaller_correct"] += int(counts["smaller_correct"])
            scene_counts["larger_correct"] += int(counts["larger_correct"])
            scene_counts["ties"] += int(counts["ties"])
        compact_image = {
            "index": index,
            "relative_path": relative_path,
            "scene": scene,
            "pairs": len(pairs),
            **image_summary,
        }
        image_summaries.append(compact_image)
        with image_rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(compact_image, sort_keys=True, default=str) + "\n")
        with node_rows_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                enriched = {
                    "image_index": index,
                    "relative_path": relative_path,
                    "scene": scene,
                    **row,
                }
                all_rows.append(enriched)
                handle.write(json.dumps(enriched, sort_keys=True, default=str) + "\n")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        if config.log_every > 0 and (index % config.log_every == 0 or index == len(items)):
            clean_overall = finalize_counts(clean_counts)
            corrupt_overall = finalize_counts(corrupt_counts)
            print(
                json.dumps(
                    {
                        "images": index,
                        "clean_larger_accuracy": clean_overall["larger_is_closer_accuracy"],
                        "corrupt_larger_accuracy": corrupt_overall["larger_is_closer_accuracy"],
                        "nodes_recorded": len(rows),
                        "node_batches": len(spec_batches),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    aggregated = aggregate_rows(all_rows)
    rows_by_positive = sorted(
        aggregated,
        key=lambda row: (
            -float(row["positive_attribution"]),
            -float(row["attribution"]),
            math.inf if row["layer_index"] is None else int(row["layer_index"]),
            str(row["component"]),
        ),
    )
    rows_by_signed = sorted(
        aggregated,
        key=lambda row: (
            -float(row["attribution"]),
            math.inf if row["layer_index"] is None else int(row["layer_index"]),
            str(row["component"]),
        ),
    )
    rows_by_abs = sorted(
        aggregated,
        key=lambda row: (
            -float(row["abs_attribution"]),
            math.inf if row["layer_index"] is None else int(row["layer_index"]),
            str(row["component"]),
        ),
    )
    summary = {
        "config": asdict(config),
        "device": str(device),
        "load_summary": load_summary,
        "node_count": len(specs),
        "image_count": len(items),
        "missing_images": missing_images,
        "clean_overall": finalize_counts(clean_counts),
        "corrupted_overall": finalize_counts(corrupt_counts),
        "clean_by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(clean_by_scene.items())},
        "corrupted_by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(corrupt_by_scene.items())},
        "rows_by_positive": rows_by_positive,
        "rows_by_signed": rows_by_signed,
        "rows_by_abs": rows_by_abs,
        "layer_rollup": layer_rollup(aggregated),
        "circuit_candidates": circuit_cover(rows_by_positive),
        "image_summaries": image_summaries,
        "metadata": {
            "elapsed_seconds": time.monotonic() - started,
            "method": (
                (
                    "Attribution patching estimate from arXiv:2310.10348: for each selected module output, "
                    "score = sum((clean_activation - corrupted_activation) * d margin(corrupted) / d corrupted_activation). "
                    "Positive scores estimate nodes whose clean activations would restore DA-2K depth-order margin."
                )
                if config.scoring_method == "atp"
                else (
                    "Relevance patching estimate from arXiv:2508.21258: same clean/corrupt patching setup as attribution "
                    "patching, but the backward pass uses RelP/LRP-inspired propagation rules for selected transformer "
                    "components, so the gradient term acts as a relevance propagation coefficient."
                )
            ),
            "objective": "mean over selected DA-2K pairs of depth(point1) - depth(point2); point1 is labeled closer.",
        },
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    write_markdown(config.output_dir / "summary.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find Depth Anything V2 circuit nodes with attribution patching on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/attribution_patching_da2k"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--node-types", default="attn,mlp,head")
    parser.add_argument("--components", default="", help="Comma-separated component names or module names to score.")
    parser.add_argument("--corruption", choices=["point_mask", "blur", "mean", "gray", "black", "noise", "patch_shuffle"], default="point_mask")
    parser.add_argument("--mask-radius", type=int, default=32)
    parser.add_argument("--blur-kernel", type=int, default=41)
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--image-start", type=int, default=0)
    parser.add_argument("--image-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--activation", default="original")
    parser.add_argument("--stage2", choices=["none", "norm2", "norm12"], default="none")
    parser.add_argument("--stage2-shift", type=float, default=0.0)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--variant-key", default="")
    parser.add_argument("--state-dict", type=Path, default=None)
    parser.add_argument("--node-batch-size", type=int, default=0, help="Score at most this many hooked nodes per backward pass.")
    parser.add_argument("--scoring-method", choices=["atp", "relp"], default="atp")
    parser.add_argument("--relp-rules", default="ln,act,attn", help="Comma-separated RelP rules to install when scoring-method=relp: ln,act,attn.")
    parser.add_argument("--log-every", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = AttributionPatchingConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        node_types=parse_csv(args.node_types),
        components=parse_csv(args.components),
        corruption=args.corruption,
        mask_radius=args.mask_radius,
        blur_kernel=args.blur_kernel,
        scene_type=args.scene_type,
        max_images=args.max_images,
        max_pairs=args.max_pairs,
        image_start=args.image_start,
        image_count=args.image_count,
        seed=args.seed,
        activation=args.activation,
        stage2=args.stage2,
        stage2_shift=args.stage2_shift,
        summary_json=args.summary_json,
        variant_key=args.variant_key,
        state_dict=args.state_dict,
        node_batch_size=args.node_batch_size,
        scoring_method=args.scoring_method,
        relp_rules=parse_csv(args.relp_rules),
        log_every=args.log_every,
    )
    summary = run(config)
    print(
        json.dumps(
            {
                "clean_overall": summary["clean_overall"],
                "corrupted_overall": summary["corrupted_overall"],
                "top_circuit_nodes": [
                    {
                        "component": row["component"],
                        "kind": row["kind"],
                        "layer_index": row["layer_index"],
                        "positive_attribution": row["positive_attribution"],
                        "attribution": row["attribution"],
                    }
                    for row in summary["rows_by_positive"][:12]
                ],
                "layer_rollup": summary["layer_rollup"][:8],
                "output_dir": str(config.output_dir),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
