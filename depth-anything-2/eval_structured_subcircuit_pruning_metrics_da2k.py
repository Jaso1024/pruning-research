from __future__ import annotations

import argparse
import json
import random
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

from eval_da2k import MODEL_CONFIGS, add_pair, empty_counts, finalize_counts, point_value, resolve_device, scene_from_path
from eval_gelu_relu_compensation_da2k import load_model, selected_annotations
from eval_attribution_patching_da2k import image_to_tensor


DEFAULT_CANDIDATE_KINDS = (
    "attn_q_head",
    "attn_k_head",
    "attn_v_head",
    "attn_q_group",
    "attn_k_group",
    "attn_v_group",
    "attn_proj_group",
    "mlp_group",
    "head_channel_group",
    "head_input_channel_group",
)


@dataclass(frozen=True)
class StructuredMetricConfig:
    dataset_root: Path
    checkpoint: Path
    circuit_summary: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    metrics: tuple[str, ...] = (
        "ablation_correct",
        "ablation_margin",
        "stability",
        "param_eff_correct",
        "magnitude",
        "wanda",
        "safe_wanda",
        "safe_magnitude",
        "stability_wanda",
        "stability_param",
        "wanda_circuit4",
        "wanda_anticircuit0p5",
        "random",
    )
    candidate_kinds: tuple[str, ...] = DEFAULT_CANDIDATE_KINDS
    budget_nodes: tuple[int, ...] = (50, 100, 200, 400)
    calibration_images: int = 8
    scene_type: str = ""
    skip_images: int = 0
    max_images: int = 64
    seed: int = 123
    log_every: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "circuit_summary", Path(self.circuit_summary))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.skip_images < 0:
            raise ValueError("skip_images must be non-negative")
        if self.calibration_images <= 0:
            raise ValueError("calibration_images must be positive")
        if any(budget <= 0 for budget in self.budget_nodes):
            raise ValueError("budget_nodes must be positive")


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def selected_rows(summary: dict[str, Any], candidate_kinds: tuple[str, ...]) -> list[dict[str, Any]]:
    kinds = set(candidate_kinds)
    rows = summary.get("rows_by_accuracy_drop")
    if not isinstance(rows, list):
        raise ValueError("summary missing rows_by_accuracy_drop")
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("kind") not in kinds:
            continue
        # This output scalar is not a useful compression decision; removing it
        # just changes the sign/range of the final depth map.
        if row.get("name") == "depth_head_scratch_output_conv2_2_out_group_00_0_1":
            continue
        candidates.append(row)
    return candidates


def load_images(
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
        tensor, _height, _width = image_to_tensor(model, image, input_size, device)
        tensors.append(tensor.to(device=device, non_blocking=True))
    if not tensors:
        raise RuntimeError("no calibration tensors loaded")
    return tensors


def collect_input_rms(
    *,
    model: torch.nn.Module,
    module_names: set[str],
    image_tensors: list[torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            x = inputs[0].detach().float()
            if isinstance(module, torch.nn.Linear):
                flat = x.reshape(-1, x.shape[-1])
            elif isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                flat = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
            else:
                return
            if name not in sums:
                sums[name] = torch.zeros(flat.shape[-1], dtype=torch.float64)
                counts[name] = 0
            sums[name] += flat.pow(2).sum(dim=0).double().cpu()
            counts[name] += int(flat.shape[0])

        return hook

    for name in sorted(module_names):
        try:
            module = model.get_submodule(name)
        except AttributeError:
            continue
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for tensor in image_tensors:
                model(tensor.to(device=device, non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return {name: (sums[name] / max(counts[name], 1)).sqrt().float() for name in sums}


def linear_wanda_score(module: torch.nn.Linear, input_rms: torch.Tensor | None) -> torch.Tensor:
    weight = module.weight.detach().cpu().float().abs()
    if input_rms is None:
        return weight
    return weight * input_rms.reshape(1, -1)


def conv_wanda_score(module: torch.nn.Module, input_rms: torch.Tensor | None) -> torch.Tensor:
    weight = module.weight.detach().cpu().float().abs()
    if input_rms is None:
        return weight
    if isinstance(module, torch.nn.ConvTranspose2d):
        # ConvTranspose2d weight is [in_channels, out_channels, kh, kw].
        return weight * input_rms.reshape(-1, 1, 1, 1)
    return weight * input_rms.reshape(1, -1, 1, 1)


def module_structural_score(
    model: torch.nn.Module,
    row: dict[str, Any],
    *,
    input_rms: dict[str, torch.Tensor] | None,
) -> float:
    kind = str(row["kind"])
    module_name = str(row["module_name"])
    start = int(row["start"])
    end = int(row["end"])
    index = int(row["index"])
    module = model.get_submodule(module_name)

    if kind in {"attn_q_head", "attn_k_head", "attn_v_head", "attn_q_group", "attn_k_group", "attn_v_group"}:
        attn = module
        qkv = attn.qkv
        if not isinstance(qkv, torch.nn.Linear):
            return 0.0
        channels = int(qkv.in_features)
        offset = {
            "attn_q_head": 0,
            "attn_q_group": 0,
            "attn_k_head": channels,
            "attn_k_group": channels,
            "attn_v_head": 2 * channels,
            "attn_v_group": 2 * channels,
        }[kind]
        score = linear_wanda_score(qkv, None if input_rms is None else input_rms.get(module_name + ".qkv"))
        return float(score[offset + start : offset + end, :].sum().item())

    if kind == "attn_proj_group":
        proj = module.proj
        if not isinstance(proj, torch.nn.Linear):
            return 0.0
        score = linear_wanda_score(proj, None if input_rms is None else input_rms.get(module_name + ".proj"))
        return float(score[start:end, :].sum().item())

    if kind == "mlp_group":
        fc1 = module.fc1
        fc2 = module.fc2
        if not isinstance(fc1, torch.nn.Linear) or not isinstance(fc2, torch.nn.Linear):
            return 0.0
        fc1_score = linear_wanda_score(fc1, None if input_rms is None else input_rms.get(module_name + ".fc1"))
        fc2_score = linear_wanda_score(fc2, None if input_rms is None else input_rms.get(module_name + ".fc2"))
        return float(fc1_score[start:end, :].sum().item() + fc2_score[:, start:end].sum().item())

    if kind == "head_channel_group":
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            return 0.0
        score = conv_wanda_score(module, None if input_rms is None else input_rms.get(module_name))
        if isinstance(module, torch.nn.ConvTranspose2d):
            return float(score[:, start:end, :, :].sum().item())
        return float(score[start:end, :, :, :].sum().item())

    if kind == "head_input_channel_group":
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            return 0.0
        score = conv_wanda_score(module, None if input_rms is None else input_rms.get(module_name))
        if isinstance(module, torch.nn.ConvTranspose2d):
            return float(score[start:end, :, :, :].sum().item())
        return float(score[:, start:end, :, :].sum().item())

    return float(index) * 0.0


def structural_scores(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    input_rms: dict[str, torch.Tensor] | None,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in rows:
        scores[str(row["name"])] = module_structural_score(model, row, input_rms=input_rms)
    return scores


def metric_scores(
    *,
    metric: str,
    rows: list[dict[str, Any]],
    magnitude_scores: dict[str, float],
    wanda_scores: dict[str, float],
    seed: int,
) -> dict[str, float]:
    max_positive = max(max(float(row.get("correct_drop", 0.0)), 0.0) for row in rows) or 1.0
    random_scores: dict[str, float] = {}
    if metric == "random":
        rng = random.Random(seed)
        shuffled = list(rows)
        rng.shuffle(shuffled)
        random_scores = {str(row["name"]): float(rank) for rank, row in enumerate(shuffled)}

    out: dict[str, float] = {}
    for row in rows:
        name = str(row["name"])
        correct_drop = float(row.get("correct_drop", 0.0))
        margin_drop = float(row.get("mean_margin_drop", 0.0))
        pair_delta = row.get("pair_delta", {})
        mean_abs_delta = float(pair_delta.get("mean_abs_margin_delta", abs(margin_drop)))
        params = max(float(row.get("parameter_estimate", 1.0)), 1.0)
        protection = max(correct_drop, 0.0) / max_positive

        if metric == "ablation_correct":
            score = correct_drop + 0.01 * margin_drop
        elif metric == "ablation_margin":
            score = margin_drop
        elif metric == "stability":
            score = mean_abs_delta
        elif metric == "param_eff_correct":
            score = correct_drop / params
        elif metric == "param_eff_margin":
            score = margin_drop / params
        elif metric == "magnitude":
            score = magnitude_scores.get(name, 0.0)
        elif metric == "wanda":
            score = wanda_scores.get(name, 0.0)
        elif metric == "safe_wanda":
            penalty = 1e18 if correct_drop > 0.0 else 0.0
            score = penalty + wanda_scores.get(name, 0.0)
        elif metric == "safe_magnitude":
            penalty = 1e18 if correct_drop > 0.0 else 0.0
            score = penalty + magnitude_scores.get(name, 0.0)
        elif metric == "stability_wanda":
            wanda_scale = max(max(wanda_scores.values()), 1e-12)
            score = mean_abs_delta + 1e-4 * (wanda_scores.get(name, 0.0) / wanda_scale)
        elif metric == "stability_param":
            score = mean_abs_delta / params
        elif metric.startswith("wanda_circuit"):
            lam = float(metric.removeprefix("wanda_circuit").replace("p", ".") or "1")
            score = wanda_scores.get(name, 0.0) * (1.0 + lam * protection)
        elif metric.startswith("wanda_anticircuit"):
            lam = float(metric.removeprefix("wanda_anticircuit").replace("p", ".") or "0.5")
            score = wanda_scores.get(name, 0.0) * max(1.0 - lam * protection, 1e-6)
        elif metric.startswith("magnitude_circuit"):
            lam = float(metric.removeprefix("magnitude_circuit").replace("p", ".") or "1")
            score = magnitude_scores.get(name, 0.0) * (1.0 + lam * protection)
        elif metric == "random":
            score = random_scores[name]
        else:
            raise ValueError(f"unknown metric: {metric}")
        out[name] = float(score)
    return out


def select_nodes(rows: list[dict[str, Any]], scores: dict[str, float], budget: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (scores[str(row["name"])], str(row["name"])))
    return ranked[: min(budget, len(ranked))]


@contextmanager
def apply_structured_ablations(model: torch.nn.Module, rows: list[dict[str, Any]]) -> Iterator[None]:
    by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_module[str(row["module_name"])].append(row)

    old_forwards: list[tuple[torch.nn.Module, Any]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def ranges_for(module_rows: list[dict[str, Any]], kind: str) -> list[tuple[int, int]]:
        return [(int(row["start"]), int(row["end"])) for row in module_rows if row["kind"] == kind]

    try:
        for module_name, module_rows in by_module.items():
            module = model.get_submodule(module_name)
            kinds = {row["kind"] for row in module_rows}
            if kinds & {
                "attn_q_head",
                "attn_k_head",
                "attn_v_head",
                "attn_q_group",
                "attn_k_group",
                "attn_v_group",
                "attn_proj_group",
            }:
                old_forward = module.forward
                old_forwards.append((module, old_forward))
                q_ranges = ranges_for(module_rows, "attn_q_head") + ranges_for(module_rows, "attn_q_group")
                k_ranges = ranges_for(module_rows, "attn_k_head") + ranges_for(module_rows, "attn_k_group")
                v_ranges = ranges_for(module_rows, "attn_v_head") + ranges_for(module_rows, "attn_v_group")
                proj_ranges = ranges_for(module_rows, "attn_proj_group")

                def make_attention_forward(
                    q_ranges: list[tuple[int, int]],
                    k_ranges: list[tuple[int, int]],
                    v_ranges: list[tuple[int, int]],
                    proj_ranges: list[tuple[int, int]],
                ):
                    def zero_flat_ranges(tensor: torch.Tensor, ranges: list[tuple[int, int]]) -> torch.Tensor:
                        if not ranges:
                            return tensor
                        batch, heads, tokens, head_dim = tensor.shape
                        patched = tensor.permute(0, 2, 1, 3).reshape(batch, tokens, heads * head_dim).clone()
                        for start, end in ranges:
                            patched[..., start:end] = 0
                        return patched.reshape(batch, tokens, heads, head_dim).permute(0, 2, 1, 3)

                    def forward(self: torch.nn.Module, x: torch.Tensor, attn_bias: Any = None) -> torch.Tensor:
                        if attn_bias is not None:
                            raise AssertionError("structured ablation does not support xFormers attention bias")
                        batch, tokens, channels = x.shape
                        head_dim = channels // self.num_heads
                        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
                        q = zero_flat_ranges(q, q_ranges)
                        k = zero_flat_ranges(k, k_ranges)
                        v = zero_flat_ranges(v, v_ranges)
                        attention = q @ k.transpose(-2, -1)
                        attention = attention.softmax(dim=-1)
                        attention = self.attn_drop(attention)
                        output = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
                        output = self.proj(output)
                        if proj_ranges:
                            output = output.clone()
                            for start, end in proj_ranges:
                                output[..., start:end] = 0
                        output = self.proj_drop(output)
                        return output

                    return forward

                module.forward = types.MethodType(make_attention_forward(q_ranges, k_ranges, v_ranges, proj_ranges), module)  # type: ignore[method-assign]

            elif "mlp_group" in kinds:
                old_forward = module.forward
                old_forwards.append((module, old_forward))
                mlp_ranges = ranges_for(module_rows, "mlp_group")

                def make_mlp_forward(mlp_ranges: list[tuple[int, int]]):
                    def forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
                        hidden = self.fc1(x)
                        hidden = self.act(hidden)
                        hidden = self.drop(hidden)
                        hidden = hidden.clone()
                        for start, end in mlp_ranges:
                            hidden[..., start:end] = 0
                        output = self.fc2(hidden)
                        output = self.drop(output)
                        return output

                    return forward

                module.forward = types.MethodType(make_mlp_forward(mlp_ranges), module)  # type: ignore[method-assign]

            conv_out_ranges = ranges_for(module_rows, "head_channel_group")
            conv_in_ranges = ranges_for(module_rows, "head_input_channel_group")
            if conv_in_ranges:
                def make_pre_hook(conv_in_ranges: list[tuple[int, int]]):
                    def pre_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
                        if not inputs or not torch.is_tensor(inputs[0]):
                            return inputs
                        patched = inputs[0].clone()
                        for start, end in conv_in_ranges:
                            patched[:, start:end, ...] = 0
                        return (patched, *inputs[1:])

                    return pre_hook

                handles.append(module.register_forward_pre_hook(make_pre_hook(conv_in_ranges)))
            if conv_out_ranges:
                def make_out_hook(conv_out_ranges: list[tuple[int, int]]):
                    def out_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
                        if not torch.is_tensor(output) or output.ndim < 2:
                            return output
                        patched = output.clone()
                        for start, end in conv_out_ranges:
                            patched[:, start:end, ...] = 0
                        return patched

                    return out_hook

                handles.append(module.register_forward_hook(make_out_hook(conv_out_ranges)))

        yield
    finally:
        for module, old_forward in reversed(old_forwards):
            module.forward = old_forward  # type: ignore[method-assign]
        for handle in handles:
            handle.remove()


@torch.no_grad()
def evaluate_items(
    *,
    model: torch.nn.Module,
    items: list[tuple[str, list[dict[str, Any]]]],
    dataset_root: Path,
    input_size: int,
    device: torch.device,
) -> dict[str, Any]:
    counts = empty_counts()
    by_scene = defaultdict(empty_counts)
    mean_margins: list[float] = []
    missing_images: list[str] = []
    evaluated_images = 0
    for relative_path, pairs in items:
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
        for pair in pairs:
            d1 = point_value(depth.detach().float().cpu(), pair["point1"])
            d2 = point_value(depth.detach().float().cpu(), pair["point2"])
            add_pair(counts, d1, d2)
            add_pair(by_scene[scene], d1, d2)
            margins.append(d1 - d2)
        if margins:
            mean_margins.append(float(sum(margins) / len(margins)))
        evaluated_images += 1
    return {
        "overall": finalize_counts(counts),
        "by_scene": {scene: finalize_counts(scene_counts) for scene, scene_counts in sorted(by_scene.items())},
        "mean_margin": float(sum(mean_margins) / max(len(mean_margins), 1)),
        "evaluated_images": evaluated_images,
        "missing_images": missing_images,
    }


def run(config: StructuredMetricConfig) -> dict[str, Any]:
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

    summary = json.loads(config.circuit_summary.read_text())
    rows = selected_rows(summary, config.candidate_kinds)
    if not rows:
        raise RuntimeError("no candidate rows selected")

    all_items = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=0,
        max_pairs=0,
    )
    if config.max_images > 0:
        items = all_items[config.skip_images : config.skip_images + config.max_images]
    else:
        items = all_items[config.skip_images :]
    if not items:
        raise RuntimeError("no DA-2K annotations selected")

    model = load_model(config.encoder, config.checkpoint, device).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    calibration_tensors = load_images(
        model,
        dataset_root=config.dataset_root,
        items=items,
        input_size=config.input_size,
        device=device,
        limit=config.calibration_images,
    )
    module_names = {str(row["module_name"]) for row in rows}
    module_names |= {
        str(row["module_name"]) + suffix
        for row in rows
        if row["kind"] in {"attn_q_head", "attn_k_head", "attn_v_head"}
        for suffix in (".qkv",)
    }
    module_names |= {
        str(row["module_name"]) + suffix
        for row in rows
        if row["kind"] == "attn_proj_group"
        for suffix in (".proj",)
    }
    module_names |= {
        str(row["module_name"]) + suffix
        for row in rows
        if row["kind"] == "mlp_group"
        for suffix in (".fc1", ".fc2")
    }
    input_rms = collect_input_rms(
        model=model,
        module_names=module_names,
        image_tensors=calibration_tensors,
        device=device,
    )
    magnitude = structural_scores(model, rows, input_rms=None)
    wanda = structural_scores(model, rows, input_rms=input_rms)

    baseline = evaluate_items(
        model=model,
        items=items,
        dataset_root=config.dataset_root,
        input_size=config.input_size,
        device=device,
    )
    baseline_acc = float(baseline["overall"]["larger_is_closer_accuracy"])
    baseline_correct = int(baseline["overall"]["larger_correct"])

    results: list[dict[str, Any]] = []
    total_runs = len(config.metrics) * len(config.budget_nodes)
    run_index = 0
    for metric in config.metrics:
        scores = metric_scores(
            metric=metric,
            rows=rows,
            magnitude_scores=magnitude,
            wanda_scores=wanda,
            seed=config.seed,
        )
        for budget in config.budget_nodes:
            run_index += 1
            selected = select_nodes(rows, scores, budget)
            with apply_structured_ablations(model, selected):
                result = evaluate_items(
                    model=model,
                    items=items,
                    dataset_root=config.dataset_root,
                    input_size=config.input_size,
                    device=device,
                )
            row = {
                "metric": metric,
                "budget_nodes": budget,
                "selected_nodes": len(selected),
                "selected_parameter_estimate": int(sum(int(node.get("parameter_estimate", 0)) for node in selected)),
                "overall": result["overall"],
                "by_scene": result["by_scene"],
                "mean_margin": result["mean_margin"],
                "accuracy_drop": baseline_acc - float(result["overall"]["larger_is_closer_accuracy"]),
                "correct_drop": baseline_correct - int(result["overall"]["larger_correct"]),
                "top_selected": [
                    {
                        "name": node["name"],
                        "kind": node["kind"],
                        "score": scores[str(node["name"])],
                        "correct_drop": node.get("correct_drop"),
                        "mean_margin_drop": node.get("mean_margin_drop"),
                        "parameter_estimate": node.get("parameter_estimate"),
                    }
                    for node in selected[:20]
                ],
            }
            results.append(row)
            if config.log_every > 0 and (run_index % config.log_every == 0 or run_index == total_runs):
                print(
                    json.dumps(
                        {
                            "runs_done": run_index,
                            "runs_total": total_runs,
                            "metric": metric,
                            "budget_nodes": budget,
                            "accuracy": row["overall"]["larger_is_closer_accuracy"],
                            "correct_drop": row["correct_drop"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    results_by_accuracy = sorted(results, key=lambda row: (int(row["correct_drop"]), float(row["accuracy_drop"]), row["metric"], row["budget_nodes"]))
    summary_out = {
        "config": asdict(config),
        "device": str(device),
        "baseline": baseline,
        "candidate_count": len(rows),
        "calibration_rms_modules": len(input_rms),
        "results": results,
        "results_by_accuracy": results_by_accuracy,
        "metadata": {
            "elapsed_seconds": time.monotonic() - started,
            "method": "Composite structured subcircuit ablation using rankings from causal, magnitude, Wanda, and circuit-weighted metrics.",
        },
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary_out, indent=2, sort_keys=True, default=str) + "\n")
    return summary_out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pruning metrics over fine DAV2 subcircuit candidates.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--circuit-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/structured_subcircuit_metric_sweep"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metrics", default="ablation_correct,ablation_margin,stability,param_eff_correct,magnitude,wanda,safe_wanda,safe_magnitude,stability_wanda,stability_param,wanda_circuit4,wanda_anticircuit0p5,random")
    parser.add_argument("--candidate-kinds", default=",".join(DEFAULT_CANDIDATE_KINDS))
    parser.add_argument("--budget-nodes", default="50,100,200,400")
    parser.add_argument("--calibration-images", type=int, default=8)
    parser.add_argument("--scene-type", default="")
    parser.add_argument("--skip-images", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = StructuredMetricConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        circuit_summary=args.circuit_summary,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        metrics=parse_csv(args.metrics),
        candidate_kinds=parse_csv(args.candidate_kinds),
        budget_nodes=parse_int_csv(args.budget_nodes),
        calibration_images=args.calibration_images,
        scene_type=args.scene_type,
        skip_images=args.skip_images,
        max_images=args.max_images,
        seed=args.seed,
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
                "best_results": [
                    {
                        "metric": row["metric"],
                        "budget_nodes": row["budget_nodes"],
                        "correct_drop": row["correct_drop"],
                        "accuracy": row["overall"]["larger_is_closer_accuracy"],
                        "selected_parameter_estimate": row["selected_parameter_estimate"],
                    }
                    for row in summary["results_by_accuracy"][:20]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
