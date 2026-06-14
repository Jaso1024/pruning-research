from __future__ import annotations

import argparse
import json
import random
import re
import time
import types
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from eval_da2k import MODEL_CONFIGS, add_pair, empty_counts, finalize_counts, point_value, resolve_device, scene_from_path
from eval_gelu_relu_compensation_da2k import load_model, selected_annotations
from eval_attribution_patching_da2k import image_to_tensor


@dataclass(frozen=True)
class SubcircuitSpec:
    name: str
    kind: str
    module_name: str
    layer_index: int | None
    index: int
    start: int
    end: int
    total: int
    parameter_estimate: int


@dataclass(frozen=True)
class SubcircuitAblationConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    circuit_types: tuple[str, ...] = ("attn_head", "mlp_group", "head_channel_group")
    components: tuple[str, ...] = ()
    component_regex: str = ""
    scene_type: str = ""
    max_images: int = 64
    max_pairs: int = 0
    seed: int = 123
    attention_group_size: int = 128
    mlp_group_size: int = 128
    head_channel_group_size: int = 64
    log_every: int = 12

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        allowed = {
            "attn_head",
            "attn_q_head",
            "attn_k_head",
            "attn_v_head",
            "attn_q_group",
            "attn_k_group",
            "attn_v_group",
            "attn_route_head",
            "attn_proj_group",
            "mlp_group",
            "head_channel_group",
            "head_input_channel_group",
        }
        unknown = sorted(set(self.circuit_types) - allowed)
        if unknown:
            raise ValueError(f"unknown circuit type(s): {unknown}")
        if self.max_images < 0 or self.max_pairs < 0:
            raise ValueError("max_images and max_pairs must be non-negative")
        if self.attention_group_size < 1 or self.mlp_group_size < 1 or self.head_channel_group_size < 1:
            raise ValueError("group sizes must be positive")
        if self.component_regex:
            re.compile(self.component_regex)


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def linear_group_params(linear: torch.nn.Linear, start: int, end: int, *, axis: str) -> int:
    width = max(0, end - start)
    if axis == "out":
        return width * linear.in_features + (width if linear.bias is not None else 0)
    if axis == "in":
        return width * linear.out_features
    raise ValueError(f"unknown axis: {axis}")


def qkv_group_params(linear: torch.nn.Linear, start: int, end: int) -> int:
    width = max(0, end - start)
    return width * linear.in_features + (width if linear.bias is not None else 0)


def conv_group_params(module: torch.nn.Module, start: int, end: int) -> int:
    width = max(0, end - start)
    if not hasattr(module, "weight"):
        return 0
    weight = module.weight
    if weight.ndim < 2:
        return 0
    per_out = int(np.prod(tuple(weight.shape[1:])))
    bias = width if getattr(module, "bias", None) is not None else 0
    return width * per_out + bias


def conv_input_group_params(module: torch.nn.Module, start: int, end: int) -> int:
    width = max(0, end - start)
    if not hasattr(module, "weight"):
        return 0
    weight = module.weight
    if weight.ndim < 2:
        return 0
    if isinstance(module, torch.nn.ConvTranspose2d):
        per_in = int(weight.shape[1] * np.prod(tuple(weight.shape[2:])))
    else:
        per_in = int(weight.shape[0] * np.prod(tuple(weight.shape[2:])))
    return width * per_in


def build_subcircuit_specs(model: torch.nn.Module, config: SubcircuitAblationConfig) -> list[SubcircuitSpec]:
    wanted = set(config.circuit_types)
    specs: list[SubcircuitSpec] = []

    attention_kinds = {
        "attn_head",
        "attn_q_head",
        "attn_k_head",
        "attn_v_head",
        "attn_q_group",
        "attn_k_group",
        "attn_v_group",
        "attn_route_head",
        "attn_proj_group",
    }
    if (attention_kinds | {"mlp_group"}) & wanted:
        for layer_index, block in enumerate(model.pretrained.blocks):
            base = f"pretrained.blocks.{layer_index}"
            if attention_kinds & wanted and hasattr(block, "attn"):
                attn = block.attn
                channels = int(attn.qkv.in_features)
                heads = int(attn.num_heads)
                head_dim = channels // heads
                for head in range(heads):
                    start = head * head_dim
                    end = start + head_dim
                    if "attn_head" in wanted:
                        param_est = 3 * linear_group_params(attn.qkv, start, end, axis="out")
                        param_est += linear_group_params(attn.proj, start, end, axis="in")
                        specs.append(
                            SubcircuitSpec(
                                name=f"block_{layer_index:02d}_head_{head:02d}",
                                kind="attn_head",
                                module_name=f"{base}.attn",
                                layer_index=layer_index,
                                index=head,
                                start=start,
                                end=end,
                                total=heads,
                                parameter_estimate=param_est,
                            )
                        )
                    for component, qkv_offset in (("attn_q_head", 0), ("attn_k_head", 1), ("attn_v_head", 2)):
                        if component not in wanted:
                            continue
                        specs.append(
                            SubcircuitSpec(
                                name=f"block_{layer_index:02d}_{component.removeprefix('attn_')}_{head:02d}",
                                kind=component,
                                module_name=f"{base}.attn",
                                layer_index=layer_index,
                                index=head,
                                start=start,
                                end=end,
                                total=heads,
                                parameter_estimate=qkv_group_params(attn.qkv, qkv_offset * channels + start, qkv_offset * channels + end),
                            )
                        )
                    if "attn_route_head" in wanted:
                        param_est = 2 * qkv_group_params(attn.qkv, start, end)
                        specs.append(
                            SubcircuitSpec(
                                name=f"block_{layer_index:02d}_route_head_{head:02d}",
                                kind="attn_route_head",
                                module_name=f"{base}.attn",
                                layer_index=layer_index,
                                index=head,
                                start=start,
                                end=end,
                                total=heads,
                                parameter_estimate=param_est,
                            )
                        )
                for component, qkv_offset, label in (
                    ("attn_q_group", 0, "q"),
                    ("attn_k_group", 1, "k"),
                    ("attn_v_group", 2, "v"),
                ):
                    if component not in wanted:
                        continue
                    group_count = (channels + config.attention_group_size - 1) // config.attention_group_size
                    for group in range(group_count):
                        start = group * config.attention_group_size
                        end = min(channels, start + config.attention_group_size)
                        specs.append(
                            SubcircuitSpec(
                                name=f"block_{layer_index:02d}_{label}_group_{group:03d}_{start}_{end}",
                                kind=component,
                                module_name=f"{base}.attn",
                                layer_index=layer_index,
                                index=group,
                                start=start,
                                end=end,
                                total=group_count,
                                parameter_estimate=qkv_group_params(attn.qkv, qkv_offset * channels + start, qkv_offset * channels + end),
                            )
                        )
                if "attn_proj_group" in wanted:
                    group_count = (channels + config.attention_group_size - 1) // config.attention_group_size
                    for group in range(group_count):
                        start = group * config.attention_group_size
                        end = min(channels, start + config.attention_group_size)
                        specs.append(
                            SubcircuitSpec(
                                name=f"block_{layer_index:02d}_attn_proj_group_{group:02d}_{start}_{end}",
                                kind="attn_proj_group",
                                module_name=f"{base}.attn",
                                layer_index=layer_index,
                                index=group,
                                start=start,
                                end=end,
                                total=group_count,
                                parameter_estimate=linear_group_params(attn.proj, start, end, axis="out"),
                            )
                        )
            if "mlp_group" in wanted and hasattr(block, "mlp"):
                mlp = block.mlp
                hidden = int(mlp.fc1.out_features)
                group_count = (hidden + config.mlp_group_size - 1) // config.mlp_group_size
                for group in range(group_count):
                    start = group * config.mlp_group_size
                    end = min(hidden, start + config.mlp_group_size)
                    param_est = linear_group_params(mlp.fc1, start, end, axis="out")
                    param_est += linear_group_params(mlp.fc2, start, end, axis="in")
                    specs.append(
                        SubcircuitSpec(
                            name=f"block_{layer_index:02d}_mlp_group_{group:02d}_{start}_{end}",
                            kind="mlp_group",
                            module_name=f"{base}.mlp",
                            layer_index=layer_index,
                            index=group,
                            start=start,
                            end=end,
                            total=group_count,
                            parameter_estimate=param_est,
                        )
                    )

    if {"head_channel_group", "head_input_channel_group"} & wanted:
        for module_name, module in model.named_modules():
            if not module_name.startswith("depth_head."):
                continue
            if not isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                continue
            safe_name = module_name.replace(".", "_")
            if "head_channel_group" in wanted:
                out_channels = int(module.out_channels)
                group_count = (out_channels + config.head_channel_group_size - 1) // config.head_channel_group_size
                for group in range(group_count):
                    start = group * config.head_channel_group_size
                    end = min(out_channels, start + config.head_channel_group_size)
                    specs.append(
                        SubcircuitSpec(
                            name=f"{safe_name}_out_group_{group:02d}_{start}_{end}",
                            kind="head_channel_group",
                            module_name=module_name,
                            layer_index=None,
                            index=group,
                            start=start,
                            end=end,
                            total=group_count,
                            parameter_estimate=conv_group_params(module, start, end),
                        )
                    )
            if "head_input_channel_group" in wanted:
                in_channels = int(module.in_channels)
                group_count = (in_channels + config.head_channel_group_size - 1) // config.head_channel_group_size
                for group in range(group_count):
                    start = group * config.head_channel_group_size
                    end = min(in_channels, start + config.head_channel_group_size)
                    specs.append(
                        SubcircuitSpec(
                            name=f"{safe_name}_in_group_{group:02d}_{start}_{end}",
                            kind="head_input_channel_group",
                            module_name=module_name,
                            layer_index=None,
                            index=group,
                            start=start,
                            end=end,
                            total=group_count,
                            parameter_estimate=conv_input_group_params(module, start, end),
                        )
                    )

    if config.components or config.component_regex:
        wanted_components = set(config.components)
        all_names = {spec.name for spec in specs}
        all_modules = {spec.module_name for spec in specs}
        pattern = re.compile(config.component_regex) if config.component_regex else None
        specs = [
            spec
            for spec in specs
            if spec.name in wanted_components
            or spec.module_name in wanted_components
            or (pattern is not None and (pattern.search(spec.name) or pattern.search(spec.module_name)))
        ]
        missing = sorted(wanted_components - all_names - all_modules)
        if missing:
            raise ValueError(f"requested component(s) not found: {missing}")
    return specs


@contextmanager
def ablate_attention_head(module: torch.nn.Module, head_index: int) -> Iterator[None]:
    old_forward = module.forward

    def forward(self: torch.nn.Module, x: torch.Tensor, attn_bias: Any = None) -> torch.Tensor:
        if attn_bias is not None:
            raise AssertionError("attention-head ablation does not support xFormers attention bias")
        batch, tokens, channels = x.shape
        head_dim = channels // self.num_heads
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attention = q @ k.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)
        heads = attention @ v
        heads = heads.clone()
        heads[:, head_index, :, :] = 0
        output = heads.transpose(1, 2).reshape(batch, tokens, channels)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output

    module.forward = types.MethodType(forward, module)  # type: ignore[method-assign]
    try:
        yield
    finally:
        module.forward = old_forward  # type: ignore[method-assign]


@contextmanager
def ablate_attention_component(module: torch.nn.Module, kind: str, head_index: int, start: int, end: int) -> Iterator[None]:
    old_forward = module.forward

    def zero_flat_attention_range(tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
        batch, heads, tokens, head_dim = tensor.shape
        patched = tensor.permute(0, 2, 1, 3).reshape(batch, tokens, heads * head_dim).clone()
        patched[..., start:end] = 0
        return patched.reshape(batch, tokens, heads, head_dim).permute(0, 2, 1, 3)

    def forward(self: torch.nn.Module, x: torch.Tensor, attn_bias: Any = None) -> torch.Tensor:
        if attn_bias is not None:
            raise AssertionError("attention component ablation does not support xFormers attention bias")
        batch, tokens, channels = x.shape
        head_dim = channels // self.num_heads
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        if kind == "attn_q_head":
            q = q.clone()
            q[:, head_index, :, :] = 0
        elif kind == "attn_k_head":
            k = k.clone()
            k[:, head_index, :, :] = 0
        elif kind == "attn_v_head":
            v = v.clone()
            v[:, head_index, :, :] = 0
        elif kind == "attn_q_group":
            q = zero_flat_attention_range(q, start, end)
        elif kind == "attn_k_group":
            k = zero_flat_attention_range(k, start, end)
        elif kind == "attn_v_group":
            v = zero_flat_attention_range(v, start, end)
        attention = q @ k.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)
        if kind == "attn_route_head":
            attention = attention.clone()
            attention[:, head_index, :, :] = 1.0 / float(tokens)
        heads = attention @ v
        output = heads.transpose(1, 2).reshape(batch, tokens, channels)
        output = self.proj(output)
        if kind == "attn_proj_group":
            output = output.clone()
            output[..., start:end] = 0
        output = self.proj_drop(output)
        return output

    module.forward = types.MethodType(forward, module)  # type: ignore[method-assign]
    try:
        yield
    finally:
        module.forward = old_forward  # type: ignore[method-assign]


@contextmanager
def ablate_mlp_hidden_group(module: torch.nn.Module, start: int, end: int) -> Iterator[None]:
    old_forward = module.forward

    def forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        hidden = self.act(hidden)
        hidden = self.drop(hidden)
        hidden = hidden.clone()
        hidden[..., start:end] = 0
        output = self.fc2(hidden)
        output = self.drop(output)
        return output

    module.forward = types.MethodType(forward, module)  # type: ignore[method-assign]
    try:
        yield
    finally:
        module.forward = old_forward  # type: ignore[method-assign]


@contextmanager
def ablate_output_channel_group(module: torch.nn.Module, start: int, end: int) -> Iterator[None]:
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if not torch.is_tensor(output):
            return output
        if output.ndim < 2:
            return output
        patched = output.clone()
        patched[:, start:end, ...] = 0
        return patched

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_input_channel_group(module: torch.nn.Module, start: int, end: int) -> Iterator[None]:
    def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not inputs or not torch.is_tensor(inputs[0]):
            return inputs
        patched = inputs[0].clone()
        patched[:, start:end, ...] = 0
        return (patched, *inputs[1:])

    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_subcircuit(model: torch.nn.Module, spec: SubcircuitSpec) -> Iterator[None]:
    module = model.get_submodule(spec.module_name)
    if spec.kind == "attn_head":
        with ablate_attention_head(module, spec.index):
            yield
        return
    if spec.kind in {
        "attn_q_head",
        "attn_k_head",
        "attn_v_head",
        "attn_q_group",
        "attn_k_group",
        "attn_v_group",
        "attn_route_head",
        "attn_proj_group",
    }:
        with ablate_attention_component(module, spec.kind, spec.index, spec.start, spec.end):
            yield
        return
    if spec.kind == "mlp_group":
        with ablate_mlp_hidden_group(module, spec.start, spec.end):
            yield
        return
    if spec.kind == "head_channel_group":
        with ablate_output_channel_group(module, spec.start, spec.end):
            yield
        return
    if spec.kind == "head_input_channel_group":
        with ablate_input_channel_group(module, spec.start, spec.end):
            yield
        return
    raise ValueError(f"unsupported spec kind: {spec.kind}")


def summarize_pair_deltas(baseline_records: list[dict[str, Any]], ablated_records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(baseline_records) != len(ablated_records):
        raise ValueError("baseline and ablated record counts do not match")
    lost: list[dict[str, Any]] = []
    gained: list[dict[str, Any]] = []
    deltas: list[float] = []
    abs_deltas: list[float] = []
    sign_flips = 0
    for base, ablated in zip(baseline_records, ablated_records):
        if base["pair_id"] != ablated["pair_id"]:
            raise ValueError(f"pair order mismatch: {base['pair_id']} vs {ablated['pair_id']}")
        delta = float(base["margin"]) - float(ablated["margin"])
        deltas.append(delta)
        abs_deltas.append(abs(delta))
        if int(base["direction"]) != int(ablated["direction"]):
            sign_flips += 1
        detail = {
            "pair_id": base["pair_id"],
            "image": base["image"],
            "pair_index": base["pair_index"],
            "scene": base["scene"],
            "baseline_margin": base["margin"],
            "ablated_margin": ablated["margin"],
            "margin_drop": delta,
        }
        if base["larger_correct"] and not ablated["larger_correct"]:
            lost.append(detail)
        elif not base["larger_correct"] and ablated["larger_correct"]:
            gained.append(detail)
    lost.sort(key=lambda item: (-float(item["margin_drop"]), str(item["pair_id"])))
    gained.sort(key=lambda item: (float(item["margin_drop"]), str(item["pair_id"])))
    return {
        "lost_pair_count": len(lost),
        "gained_pair_count": len(gained),
        "sign_flip_count": sign_flips,
        "mean_margin_delta": float(np.mean(deltas)) if deltas else 0.0,
        "mean_abs_margin_delta": float(np.mean(abs_deltas)) if abs_deltas else 0.0,
        "max_abs_margin_delta": float(max(abs_deltas)) if abs_deltas else 0.0,
        "top_lost_pairs": lost[:8],
        "top_gained_pairs": gained[:8],
    }


@torch.no_grad()
def evaluate_items(
    *,
    model: torch.nn.Module,
    items: list[tuple[str, list[dict[str, Any]]]],
    dataset_root: Path,
    input_size: int,
    device: torch.device,
    desc: str,
    collect_records: bool = False,
) -> dict[str, Any]:
    counts = empty_counts()
    by_scene = defaultdict(empty_counts)
    mean_margins: list[float] = []
    records: list[dict[str, Any]] = []
    missing_images: list[str] = []
    evaluated_images = 0
    for relative_path, pairs in tqdm(items, desc=desc, unit="image", leave=False, disable=True):
        image_path = dataset_root / relative_path
        image = cv2.imread(str(image_path))
        if image is None:
            missing_images.append(str(image_path))
            continue
        tensor, height, width = image_to_tensor(model, image, input_size, device)
        depth = model(tensor)
        depth = F.interpolate(depth[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
        scene = scene_from_path(relative_path)
        margins: list[float] = []
        for pair_index, pair in enumerate(pairs):
            d1 = point_value(depth.detach().float().cpu(), pair["point1"])
            d2 = point_value(depth.detach().float().cpu(), pair["point2"])
            add_pair(counts, d1, d2)
            add_pair(by_scene[scene], d1, d2)
            margin = d1 - d2
            margins.append(margin)
            if collect_records:
                direction = 1 if margin > 0 else (-1 if margin < 0 else 0)
                records.append(
                    {
                        "pair_id": f"{relative_path}#{pair_index}",
                        "image": relative_path,
                        "pair_index": pair_index,
                        "scene": scene,
                        "point1": pair["point1"],
                        "point2": pair["point2"],
                        "margin": float(margin),
                        "direction": direction,
                        "larger_correct": direction > 0,
                    }
                )
        if margins:
            mean_margins.append(float(sum(margins) / len(margins)))
        evaluated_images += 1
    return {
        "overall": finalize_counts(counts),
        "by_scene": {scene: finalize_counts(scene_counts) for scene, scene_counts in sorted(by_scene.items())},
        "mean_margin": float(sum(mean_margins) / max(len(mean_margins), 1)),
        "evaluated_images": evaluated_images,
        "missing_images": missing_images,
        "records": records,
    }


def run(config: SubcircuitAblationConfig) -> dict[str, Any]:
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

    items = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
        max_pairs=config.max_pairs,
    )
    if not items:
        raise RuntimeError("no DA-2K annotations selected")

    model = load_model(config.encoder, config.checkpoint, device).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    specs = build_subcircuit_specs(model, config)
    if not specs:
        raise RuntimeError("no subcircuits selected")
    (config.output_dir / "nodes.json").write_text(json.dumps([asdict(spec) for spec in specs], indent=2, sort_keys=True) + "\n")

    baseline = evaluate_items(
        model=model,
        items=items,
        dataset_root=config.dataset_root,
        input_size=config.input_size,
        device=device,
        desc="baseline",
        collect_records=True,
    )
    baseline_acc = float(baseline["overall"]["larger_is_closer_accuracy"])
    baseline_correct = int(baseline["overall"]["larger_correct"])
    baseline_margin = float(baseline["mean_margin"])

    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        with ablate_subcircuit(model, spec):
            result = evaluate_items(
                model=model,
                items=items,
                dataset_root=config.dataset_root,
                input_size=config.input_size,
                device=device,
                desc=f"ablate {spec.name}",
                collect_records=True,
            )
        pair_delta = summarize_pair_deltas(baseline["records"], result["records"])
        row = {
            **asdict(spec),
            "overall": result["overall"],
            "by_scene": result["by_scene"],
            "mean_margin": result["mean_margin"],
            "accuracy_drop": baseline_acc - float(result["overall"]["larger_is_closer_accuracy"]),
            "correct_drop": baseline_correct - int(result["overall"]["larger_correct"]),
            "mean_margin_drop": baseline_margin - float(result["mean_margin"]),
            "pair_delta": pair_delta,
        }
        rows.append(row)
        if config.log_every > 0 and (index % config.log_every == 0 or index == len(specs)):
            print(
                json.dumps(
                    {
                        "nodes_done": index,
                        "nodes_total": len(specs),
                        "last_component": spec.name,
                        "last_accuracy_drop": row["accuracy_drop"],
                        "last_correct_drop": row["correct_drop"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows_by_accuracy_drop = sorted(rows, key=lambda row: (-float(row["accuracy_drop"]), -float(row["mean_margin_drop"]), str(row["name"])))
    rows_by_margin_drop = sorted(rows, key=lambda row: (-float(row["mean_margin_drop"]), -float(row["accuracy_drop"]), str(row["name"])))
    rows_by_safe = sorted(rows, key=lambda row: (float(row["accuracy_drop"]), float(row["mean_margin_drop"]), str(row["name"])))
    summary = {
        "config": asdict(config),
        "device": str(device),
        "baseline": {key: value for key, value in baseline.items() if key != "records"},
        "baseline_records": baseline["records"],
        "node_count": len(specs),
        "image_count": len(items),
        "rows_by_accuracy_drop": rows_by_accuracy_drop,
        "rows_by_margin_drop": rows_by_margin_drop,
        "rows_by_safe": rows_by_safe,
        "metadata": {
            "elapsed_seconds": time.monotonic() - started,
            "method": "Subcircuit zero ablation: attention components, MLP hidden-channel groups, and decoder head input/output-channel groups are individually zeroed.",
        },
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate fine-grained subcircuit zero ablations for Depth Anything V2 on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/subcircuit_ablation_da2k"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--circuit-types", default="attn_head,mlp_group,head_channel_group")
    parser.add_argument("--components", default="")
    parser.add_argument("--component-regex", default="")
    parser.add_argument("--scene-type", default="")
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--attention-group-size", type=int, default=128)
    parser.add_argument("--mlp-group-size", type=int, default=128)
    parser.add_argument("--head-channel-group-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SubcircuitAblationConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        circuit_types=parse_csv(args.circuit_types),
        components=parse_csv(args.components),
        component_regex=args.component_regex,
        scene_type=args.scene_type,
        max_images=args.max_images,
        max_pairs=args.max_pairs,
        seed=args.seed,
        attention_group_size=args.attention_group_size,
        mlp_group_size=args.mlp_group_size,
        head_channel_group_size=args.head_channel_group_size,
        log_every=args.log_every,
    )
    summary = run(config)
    baseline = summary["baseline"]["overall"]
    print(
        json.dumps(
            {
                "output_dir": str(config.output_dir),
                "baseline": {
                    "pairs": baseline["pairs"],
                    "larger_correct": baseline["larger_correct"],
                    "larger_is_closer_accuracy": baseline["larger_is_closer_accuracy"],
                },
                "top_drop": [
                    {
                        "component": row["name"],
                        "kind": row["kind"],
                        "correct_drop": row["correct_drop"],
                        "accuracy_drop": row["accuracy_drop"],
                        "mean_margin_drop": row["mean_margin_drop"],
                    }
                    for row in summary["rows_by_accuracy_drop"][:12]
                ],
                "safest": [
                    {
                        "component": row["name"],
                        "kind": row["kind"],
                        "correct_drop": row["correct_drop"],
                        "accuracy_drop": row["accuracy_drop"],
                        "mean_margin_drop": row["mean_margin_drop"],
                    }
                    for row in summary["rows_by_safe"][:12]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
