from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from eval_da2k import MODEL_CONFIGS, add_pair, empty_counts, finalize_counts, point_value, resolve_device, scene_from_path
from eval_gelu_relu_compensation_da2k import load_model
from eval_wanda_unstructured_da2k import evaluate_da2k_model, find_prunable_linears, selected_annotations, transformer_layer_index


TargetKind = Literal["transformer", "all-linear"]
ScalarScoreName = Literal[
    "abs_wgrad",
    "positive_wgrad",
    "magnitude",
    "hybrid_abs_mag",
    "hybrid_protect_mag",
    "taylor1_abs",
    "taylor2_abs",
    "taylor3_abs",
    "taylor1_damage",
    "taylor2_damage",
    "taylor3_damage",
]


@dataclass(frozen=True)
class ScalarCircuitConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    target: TargetKind = "transformer"
    layer_indices: tuple[int, ...] = ()
    scene_type: str = ""
    max_images: int = 0
    score_images: int = 32
    max_pairs: int = 0
    prune_budgets: tuple[int, ...] = ()
    prune_score: ScalarScoreName = "abs_wgrad"
    hybrid_alpha: float = 1.0
    loss_tau: float = 1.0
    save_scores: bool = False
    eval_baseline: bool = True
    log_every: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.target not in {"transformer", "all-linear"}:
            raise ValueError("target must be transformer or all-linear")
        if self.max_images < 0 or self.score_images < 0 or self.max_pairs < 0:
            raise ValueError("image and pair limits must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")
        if self.prune_score not in {
            "abs_wgrad",
            "positive_wgrad",
            "magnitude",
            "hybrid_abs_mag",
            "hybrid_protect_mag",
            "taylor1_abs",
            "taylor2_abs",
            "taylor3_abs",
            "taylor1_damage",
            "taylor2_damage",
            "taylor3_damage",
        }:
            raise ValueError(f"unknown prune score: {self.prune_score}")
        if self.hybrid_alpha < 0.0:
            raise ValueError("hybrid_alpha must be non-negative")
        if self.loss_tau <= 0.0:
            raise ValueError("loss_tau must be positive")
        if any(budget < 0 for budget in self.prune_budgets):
            raise ValueError("prune budgets must be non-negative")


def parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_budget_tuple(value: str) -> tuple[int, ...]:
    budgets = parse_int_tuple(value)
    return tuple(sorted(set(budgets)))


def limit_total_pairs(
    items: list[tuple[str, list[dict[str, Any]]]],
    max_pairs: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    if max_pairs <= 0:
        return items
    kept: list[tuple[str, list[dict[str, Any]]]] = []
    total = 0
    for relative_path, pairs in items:
        remaining = max_pairs - total
        if remaining <= 0:
            break
        limited = list(pairs[:remaining])
        if limited:
            kept.append((relative_path, limited))
            total += len(limited)
    return kept


def image_to_tensor(model: torch.nn.Module, image, input_size: int, device: torch.device) -> tuple[torch.Tensor, int, int]:
    tensor, (height, width) = model.image2tensor(image, input_size)
    return tensor.to(device=device, non_blocking=True), int(height), int(width)


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


def differentiable_margin(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    height: int,
    width: int,
    pairs: list[dict[str, Any]],
) -> torch.Tensor:
    depth = model(tensor)
    depth = F.interpolate(depth[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
    return pair_margin_from_depth(depth, pairs)


def evaluate_margin_counts(
    *,
    model: torch.nn.Module,
    image,
    pairs: list[dict[str, Any]],
    input_size: int,
    device: torch.device,
) -> dict[str, Any]:
    with torch.no_grad():
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
        "counts": finalize_counts(counts),
        "mean_margin": float(sum(margins) / max(len(margins), 1)),
    }


def enable_only_target_weight_grads(model: torch.nn.Module, module_names: list[str]) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for name in module_names:
        module = model.get_submodule(name)
        if not isinstance(module, torch.nn.Linear):
            continue
        module.weight.requires_grad_(True)


def init_accumulators(model: torch.nn.Module, module_names: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    accum: dict[str, dict[str, torch.Tensor]] = {}
    for name in module_names:
        module = model.get_submodule(name)
        if not isinstance(module, torch.nn.Linear):
            continue
        shape = tuple(module.weight.shape)
        accum[name] = {
            "signed_wgrad": torch.zeros(shape, dtype=torch.float32),
            "abs_wgrad": torch.zeros(shape, dtype=torch.float32),
            "positive_wgrad": torch.zeros(shape, dtype=torch.float32),
            "negative_wgrad": torch.zeros(shape, dtype=torch.float32),
            "grad_abs": torch.zeros(shape, dtype=torch.float32),
            "magnitude": module.weight.detach().cpu().float().abs().clone(),
            "taylor1_delta_loss": torch.zeros(shape, dtype=torch.float32),
            "taylor2_delta_loss": torch.zeros(shape, dtype=torch.float32),
            "taylor3_delta_loss": torch.zeros(shape, dtype=torch.float32),
            "taylor1_abs": torch.zeros(shape, dtype=torch.float32),
            "taylor2_abs": torch.zeros(shape, dtype=torch.float32),
            "taylor3_abs": torch.zeros(shape, dtype=torch.float32),
            "taylor1_damage": torch.zeros(shape, dtype=torch.float32),
            "taylor2_damage": torch.zeros(shape, dtype=torch.float32),
            "taylor3_damage": torch.zeros(shape, dtype=torch.float32),
        }
    return accum


def logistic_loss_derivatives(margin: torch.Tensor, tau: float) -> tuple[float, float, float]:
    """Derivatives of softplus(-margin / tau) with respect to margin."""
    z = -float(margin.detach().cpu().item()) / float(tau)
    if z >= 0.0:
        ez = math.exp(-z)
        s = 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        s = ez / (1.0 + ez)
    first = -s / tau
    second = s * (1.0 - s) / (tau * tau)
    third = -s * (1.0 - s) * (1.0 - 2.0 * s) / (tau * tau * tau)
    return first, second, third


def collect_scalar_scores(
    *,
    model: torch.nn.Module,
    module_names: list[str],
    items: list[tuple[str, list[dict[str, Any]]]],
    dataset_root: Path,
    input_size: int,
    device: torch.device,
    log_every: int,
    loss_tau: float,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    enable_only_target_weight_grads(model, module_names)
    accum = init_accumulators(model, module_names)
    total_counts = empty_counts()
    image_rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    started = time.monotonic()
    total_pairs = 0

    for index, (relative_path, pairs) in enumerate(tqdm(items, desc="scalar circuit", unit="image"), start=1):
        image_path = dataset_root / relative_path
        image = cv2.imread(str(image_path))
        if image is None:
            missing_images.append(str(image_path))
            continue
        eval_row = evaluate_margin_counts(
            model=model,
            image=image,
            pairs=pairs,
            input_size=input_size,
            device=device,
        )
        for key in ("pairs", "smaller_correct", "larger_correct", "ties"):
            total_counts[key] += int(eval_row["counts"][key])
        total_pairs += len(pairs)

        tensor, height, width = image_to_tensor(model, image, input_size, device)
        model.zero_grad(set_to_none=True)
        margin = differentiable_margin(model, tensor, height, width, pairs)
        margin.backward()
        loss_first, loss_second, loss_third = logistic_loss_derivatives(margin, loss_tau)

        nonzero_grad_modules = 0
        with torch.no_grad():
            for name in module_names:
                module = model.get_submodule(name)
                if not isinstance(module, torch.nn.Linear):
                    continue
                grad = module.weight.grad
                if grad is None:
                    continue
                nonzero_grad_modules += 1
                wgrad = (module.weight.detach() * grad.detach()).cpu().float()
                delta_margin = -wgrad
                taylor1 = loss_first * delta_margin
                taylor2 = taylor1 + 0.5 * loss_second * delta_margin.pow(2)
                taylor3 = taylor2 + (1.0 / 6.0) * loss_third * delta_margin.pow(3)
                accum[name]["signed_wgrad"].add_(wgrad)
                accum[name]["abs_wgrad"].add_(wgrad.abs())
                accum[name]["positive_wgrad"].add_(wgrad.clamp_min(0.0))
                accum[name]["negative_wgrad"].add_((-wgrad).clamp_min(0.0))
                accum[name]["grad_abs"].add_(grad.detach().cpu().float().abs())
                accum[name]["taylor1_delta_loss"].add_(taylor1)
                accum[name]["taylor2_delta_loss"].add_(taylor2)
                accum[name]["taylor3_delta_loss"].add_(taylor3)
                accum[name]["taylor1_abs"].add_(taylor1.abs())
                accum[name]["taylor2_abs"].add_(taylor2.abs())
                accum[name]["taylor3_abs"].add_(taylor3.abs())
                accum[name]["taylor1_damage"].add_(taylor1.clamp_min(0.0))
                accum[name]["taylor2_damage"].add_(taylor2.clamp_min(0.0))
                accum[name]["taylor3_damage"].add_(taylor3.clamp_min(0.0))

        image_rows.append(
            {
                "index": index,
                "relative_path": relative_path,
                "pairs": len(pairs),
                "mean_margin": float(margin.detach().cpu().item()),
                "larger_is_closer_accuracy": eval_row["counts"]["larger_is_closer_accuracy"],
                "nonzero_grad_modules": nonzero_grad_modules,
            }
        )
        del tensor, margin
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if log_every > 0 and (index % log_every == 0 or index == len(items)):
            print(
                json.dumps(
                    {
                        "scored_images": index,
                        "pairs": total_pairs,
                        "larger_is_closer_accuracy": finalize_counts(total_counts)["larger_is_closer_accuracy"],
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    metadata = {
        "images_scored": len(image_rows),
        "pairs_scored": total_pairs,
        "missing_images": missing_images,
        "score_counts": finalize_counts(total_counts),
        "elapsed_seconds": time.monotonic() - started,
        "image_rows": image_rows,
        "score_rule": (
            "For DA2K margin m = depth(point1)-depth(point2), zeroing scalar w has first-order "
            "effect delta_m ~= -w * grad_m(w). positive_wgrad protects scalars whose removal lowers m; "
            "abs_wgrad is first-order scalar saliency magnitude. Taylor loss scores approximate the change in "
            "softplus(-m/tau) under w->0 using a local-linear margin model and derivatives through third order."
        ),
        "loss": "softplus(-margin / tau)",
        "loss_tau": loss_tau,
    }
    return accum, metadata


def module_score_summaries(
    *,
    model: torch.nn.Module,
    module_names: list[str],
    accum: dict[str, dict[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in module_names:
        module = model.get_submodule(name)
        if not isinstance(module, torch.nn.Linear) or name not in accum:
            continue
        weights = int(module.weight.numel())
        abs_sum = float(accum[name]["abs_wgrad"].sum().item())
        positive_sum = float(accum[name]["positive_wgrad"].sum().item())
        negative_sum = float(accum[name]["negative_wgrad"].sum().item())
        signed_sum = float(accum[name]["signed_wgrad"].sum().item())
        grad_abs_sum = float(accum[name]["grad_abs"].sum().item())
        magnitude_sum = float(accum[name]["magnitude"].sum().item())
        taylor1_abs_sum = float(accum[name]["taylor1_abs"].sum().item())
        taylor2_abs_sum = float(accum[name]["taylor2_abs"].sum().item())
        taylor3_abs_sum = float(accum[name]["taylor3_abs"].sum().item())
        rows.append(
            {
                "module_name": name,
                "layer_index": transformer_layer_index(name),
                "shape": list(module.weight.shape),
                "weights": weights,
                "signed_wgrad": signed_sum,
                "abs_wgrad": abs_sum,
                "positive_wgrad": positive_sum,
                "negative_wgrad": negative_sum,
                "grad_abs": grad_abs_sum,
                "magnitude": magnitude_sum,
                "taylor1_abs": taylor1_abs_sum,
                "taylor2_abs": taylor2_abs_sum,
                "taylor3_abs": taylor3_abs_sum,
                "abs_wgrad_per_weight": abs_sum / max(weights, 1),
                "positive_wgrad_per_weight": positive_sum / max(weights, 1),
                "negative_wgrad_per_weight": negative_sum / max(weights, 1),
                "taylor1_abs_per_weight": taylor1_abs_sum / max(weights, 1),
                "taylor2_abs_per_weight": taylor2_abs_sum / max(weights, 1),
                "taylor3_abs_per_weight": taylor3_abs_sum / max(weights, 1),
            }
        )
    rows.sort(key=lambda row: float(row["positive_wgrad"]), reverse=True)
    return rows


def qkv_head_group_rows(accum: dict[str, dict[str, torch.Tensor]], *, top_k: int = 200) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, scores in accum.items():
        if not name.endswith(".attn.qkv"):
            continue
        layer_index = transformer_layer_index(name)
        score = scores["positive_wgrad"]
        abs_score = scores["abs_wgrad"]
        out_features = int(score.shape[0])
        if out_features % 3 != 0:
            continue
        embed_dim = out_features // 3
        head_dim = 64 if embed_dim % 64 == 0 else max(1, math.gcd(embed_dim, 64))
        heads = max(1, embed_dim // head_dim)
        for proj_index, projection in enumerate(("q", "k", "v")):
            start = proj_index * embed_dim
            for head in range(heads):
                h0 = start + head * head_dim
                h1 = start + min((head + 1) * head_dim, embed_dim)
                if h0 >= h1:
                    continue
                group = score[h0:h1]
                abs_group = abs_score[h0:h1]
                rows.append(
                    {
                        "group": f"block{layer_index}.{projection}.head{head}",
                        "kind": "attn_qkv_head",
                        "module_name": name,
                        "layer_index": layer_index,
                        "projection": projection,
                        "head": head,
                        "weights": int(group.numel()),
                        "positive_wgrad": float(group.sum().item()),
                        "abs_wgrad": float(abs_group.sum().item()),
                        "positive_wgrad_per_weight": float(group.sum().item()) / max(int(group.numel()), 1),
                        "abs_wgrad_per_weight": float(abs_group.sum().item()) / max(int(group.numel()), 1),
                    }
                )
    rows.sort(key=lambda row: float(row["positive_wgrad"]), reverse=True)
    return rows[:top_k]


def mlp_hidden_group_rows(accum: dict[str, dict[str, torch.Tensor]], *, top_k: int = 200) -> list[dict[str, Any]]:
    groups: dict[tuple[int | None, int], dict[str, Any]] = {}
    for name, scores in accum.items():
        layer_index = transformer_layer_index(name)
        if ".mlp.fc1" in name:
            pos = scores["positive_wgrad"].sum(dim=1)
            abs_score = scores["abs_wgrad"].sum(dim=1)
            part = "fc1_row"
        elif ".mlp.fc2" in name:
            pos = scores["positive_wgrad"].sum(dim=0)
            abs_score = scores["abs_wgrad"].sum(dim=0)
            part = "fc2_col"
        else:
            continue
        for hidden_index in range(int(pos.numel())):
            key = (layer_index, hidden_index)
            item = groups.setdefault(
                key,
                {
                    "group": f"block{layer_index}.mlp.hidden{hidden_index}",
                    "kind": "mlp_hidden_channel",
                    "layer_index": layer_index,
                    "hidden_index": hidden_index,
                    "weights": 0,
                    "positive_wgrad": 0.0,
                    "abs_wgrad": 0.0,
                    "parts": [],
                },
            )
            item["weights"] += int(scores["positive_wgrad"].shape[1] if part == "fc1_row" else scores["positive_wgrad"].shape[0])
            item["positive_wgrad"] += float(pos[hidden_index].item())
            item["abs_wgrad"] += float(abs_score[hidden_index].item())
            item["parts"].append(part)
    rows = list(groups.values())
    for row in rows:
        row["positive_wgrad_per_weight"] = float(row["positive_wgrad"]) / max(int(row["weights"]), 1)
        row["abs_wgrad_per_weight"] = float(row["abs_wgrad"]) / max(int(row["weights"]), 1)
    rows.sort(key=lambda row: float(row["positive_wgrad"]), reverse=True)
    return rows[:top_k]


def top_scalar_rows(
    *,
    model: torch.nn.Module,
    accum: dict[str, dict[str, torch.Tensor]],
    field: str,
    top_k: int = 200,
) -> list[dict[str, Any]]:
    per_module: list[tuple[torch.Tensor, str, torch.Tensor]] = []
    for name, scores in accum.items():
        score = scores[field].flatten()
        if score.numel() == 0:
            continue
        k = min(top_k, int(score.numel()))
        values, indices = torch.topk(score, k=k, largest=True)
        per_module.append((values.cpu(), name, indices.cpu()))
    if not per_module:
        return []
    all_values = torch.cat([chunk[0] for chunk in per_module], dim=0)
    k = min(top_k, int(all_values.numel()))
    top_values, top_indices = torch.topk(all_values, k=k, largest=True)
    offsets: list[tuple[int, int, str, torch.Tensor]] = []
    cursor = 0
    for values, name, indices in per_module:
        next_cursor = cursor + int(values.numel())
        offsets.append((cursor, next_cursor, name, indices))
        cursor = next_cursor
    rows: list[dict[str, Any]] = []
    for value, global_index in zip(top_values.tolist(), top_indices.tolist()):
        for start, end, name, indices in offsets:
            if start <= global_index < end:
                local_flat = int(indices[global_index - start].item())
                module = model.get_submodule(name)
                if not isinstance(module, torch.nn.Linear):
                    continue
                out_features, in_features = int(module.weight.shape[0]), int(module.weight.shape[1])
                row = local_flat // in_features
                col = local_flat % in_features
                signed = float(accum[name]["signed_wgrad"].flatten()[local_flat].item())
                rows.append(
                    {
                        "module_name": name,
                        "layer_index": transformer_layer_index(name),
                        "flat_index": local_flat,
                        "row": row,
                        "column": col,
                        "out_features": out_features,
                        "in_features": in_features,
                        field: float(value),
                        "signed_wgrad": signed,
                        "positive_wgrad": float(accum[name]["positive_wgrad"].flatten()[local_flat].item()),
                        "abs_wgrad": float(accum[name]["abs_wgrad"].flatten()[local_flat].item()),
                        "magnitude": float(accum[name]["magnitude"].flatten()[local_flat].item()),
                    }
                )
                break
    return rows


def normalized_like(score: torch.Tensor) -> torch.Tensor:
    finite = score[torch.isfinite(score)]
    if finite.numel() == 0:
        return torch.zeros_like(score)
    denom = finite.mean().clamp_min(1e-12)
    return score / denom


def pruning_scores(accum: dict[str, dict[str, torch.Tensor]], score_name: ScalarScoreName, alpha: float) -> dict[str, torch.Tensor]:
    scores: dict[str, torch.Tensor] = {}
    for name, fields in accum.items():
        if score_name == "abs_wgrad":
            score = fields["abs_wgrad"]
        elif score_name == "positive_wgrad":
            score = fields["positive_wgrad"]
        elif score_name == "magnitude":
            score = fields["magnitude"]
        elif score_name == "hybrid_abs_mag":
            score = fields["magnitude"] * (1.0 + alpha * normalized_like(fields["abs_wgrad"]))
        elif score_name == "hybrid_protect_mag":
            score = fields["magnitude"] * (1.0 + alpha * normalized_like(fields["positive_wgrad"]))
        elif score_name in {
            "taylor1_abs",
            "taylor2_abs",
            "taylor3_abs",
            "taylor1_damage",
            "taylor2_damage",
            "taylor3_damage",
        }:
            score = fields[score_name]
        else:
            raise ValueError(f"unknown score: {score_name}")
        scores[name] = score.detach().cpu().float()
    return scores


def global_pruning_order(scores: dict[str, torch.Tensor], module_names: list[str], max_budget: int) -> tuple[torch.Tensor, list[tuple[str, int, int]]]:
    chunks: list[torch.Tensor] = []
    slices: list[tuple[str, int, int]] = []
    cursor = 0
    for name in module_names:
        score = scores.get(name)
        if score is None:
            continue
        flat = score.flatten().float()
        chunks.append(flat)
        start = cursor
        cursor += int(flat.numel())
        slices.append((name, start, cursor))
    if not chunks:
        raise RuntimeError("no pruning scores available")
    flat_scores = torch.cat(chunks, dim=0)
    budget = min(max_budget, int(flat_scores.numel()))
    if budget <= 0:
        return torch.empty(0, dtype=torch.long), slices
    indices = torch.topk(flat_scores, k=budget, largest=False).indices.cpu()
    indices = torch.sort(indices).values
    return indices, slices


def apply_global_prefix_mask_(
    *,
    model: torch.nn.Module,
    ordered_indices: torch.Tensor,
    slices: list[tuple[str, int, int]],
    previous_budget: int,
    budget: int,
) -> dict[str, Any]:
    if budget <= previous_budget:
        return {"masked_this_step": 0, "masked_total": previous_budget, "modules_touched": 0, "tensors": []}
    add_indices = ordered_indices[previous_budget:budget]
    rows: list[dict[str, Any]] = []
    modules_touched = 0
    with torch.no_grad():
        for name, start, end in slices:
            left = int(torch.searchsorted(add_indices, start, right=False).item())
            right = int(torch.searchsorted(add_indices, end, right=False).item())
            local = add_indices[left:right] - start
            if local.numel() == 0:
                continue
            module = model.get_submodule(name)
            if not isinstance(module, torch.nn.Linear):
                continue
            flat = module.weight.flatten()
            flat[local.to(device=flat.device)] = 0
            modules_touched += 1
            rows.append(
                {
                    "module_name": name,
                    "shape": list(module.weight.shape),
                    "masked_this_step": int(local.numel()),
                }
            )
    return {
        "masked_this_step": int(add_indices.numel()),
        "masked_total": int(budget),
        "modules_touched": modules_touched,
        "tensors": rows,
    }


def run_pruning_budgets(
    *,
    model: torch.nn.Module,
    dataset_root: Path,
    eval_items: list[tuple[str, list[dict[str, Any]]]],
    module_names: list[str],
    scores: dict[str, torch.Tensor],
    budgets: tuple[int, ...],
    input_size: int,
    device: torch.device,
    log_every: int,
) -> dict[str, Any]:
    if not budgets:
        return {"enabled": False, "results": []}
    total_weights = sum(int(scores[name].numel()) for name in module_names if name in scores)
    clean_budgets = tuple(sorted(set(min(max(0, budget), total_weights) for budget in budgets)))
    order, slices = global_pruning_order(scores, module_names, max(clean_budgets))
    previous_budget = 0
    results: list[dict[str, Any]] = []
    for budget in clean_budgets:
        mask_summary = apply_global_prefix_mask_(
            model=model,
            ordered_indices=order,
            slices=slices,
            previous_budget=previous_budget,
            budget=budget,
        )
        previous_budget = budget
        eval_result = evaluate_da2k_model(
            model=model,
            dataset_root=dataset_root,
            items=eval_items,
            input_size=input_size,
            device=device,
            log_every=log_every,
        )
        results.append(
            {
                "budget": budget,
                "target_weight_count": total_weights,
                "zero_fraction": budget / max(total_weights, 1),
                "mask_summary": mask_summary,
                "overall": eval_result["overall"],
                "by_scene": eval_result["by_scene"],
            }
        )
    return {
        "enabled": True,
        "target_weight_count": total_weights,
        "results": results,
    }


def run(config: ScalarCircuitConfig) -> dict[str, Any]:
    torch.manual_seed(17)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True, default=str) + "\n")

    selected = selected_annotations(config.dataset_root, scene_type=config.scene_type, max_images=config.max_images)
    selected = limit_total_pairs(selected, config.max_pairs)
    if not selected:
        raise RuntimeError("no DA-2K annotations selected")
    score_items = selected[: config.score_images] if config.score_images > 0 else selected
    if not score_items:
        raise RuntimeError("no scoring images selected")

    model = load_model(config.encoder, config.checkpoint, device)
    module_names = find_prunable_linears(model, target=config.target, layer_indices=config.layer_indices)
    if not module_names:
        raise RuntimeError("no target linear modules found")
    module_records = [
        {
            "name": name,
            "shape": list(model.get_submodule(name).weight.shape),
            "layer_index": transformer_layer_index(name),
            "weights": int(model.get_submodule(name).weight.numel()),
        }
        for name in module_names
    ]
    (config.output_dir / "target_modules.json").write_text(json.dumps(module_records, indent=2, sort_keys=True) + "\n")

    baseline = None
    if config.eval_baseline:
        baseline = evaluate_da2k_model(
            model=model,
            dataset_root=config.dataset_root,
            items=selected,
            input_size=config.input_size,
            device=device,
            log_every=config.log_every,
        )

    accum, score_metadata = collect_scalar_scores(
        model=model,
        module_names=module_names,
        items=score_items,
        dataset_root=config.dataset_root,
        input_size=config.input_size,
        device=device,
        log_every=config.log_every,
        loss_tau=config.loss_tau,
    )

    modules_by_positive = module_score_summaries(model=model, module_names=module_names, accum=accum)
    modules_by_abs = sorted(modules_by_positive, key=lambda row: float(row["abs_wgrad"]), reverse=True)
    modules_by_density = sorted(modules_by_positive, key=lambda row: float(row["positive_wgrad_per_weight"]), reverse=True)
    qkv_groups = qkv_head_group_rows(accum)
    mlp_groups = mlp_hidden_group_rows(accum)
    top_positive_scalars = top_scalar_rows(model=model, accum=accum, field="positive_wgrad")
    top_abs_scalars = top_scalar_rows(model=model, accum=accum, field="abs_wgrad")

    score_tensors = pruning_scores(accum, config.prune_score, config.hybrid_alpha)
    pruning = run_pruning_budgets(
        model=model,
        dataset_root=config.dataset_root,
        eval_items=selected,
        module_names=module_names,
        scores=score_tensors,
        budgets=config.prune_budgets,
        input_size=config.input_size,
        device=device,
        log_every=config.log_every,
    )

    score_path = None
    if config.save_scores:
        score_path = config.output_dir / "scalar_scores.pt"
        torch.save(
            {
                "accumulators": accum,
                "pruning_scores": score_tensors,
                "module_names": module_names,
                "config": asdict(config),
                "score_metadata": score_metadata,
            },
            score_path,
        )

    result = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_selected": len(selected),
            "images_scored": len(score_items),
            "target_module_count": len(module_names),
            "target_weight_count": sum(int(row["weights"]) for row in module_records),
            "score_path": str(score_path) if score_path is not None else None,
        },
        "baseline": baseline,
        "score_metadata": {key: value for key, value in score_metadata.items() if key != "image_rows"},
        "score_image_rows": score_metadata["image_rows"],
        "modules_by_positive": modules_by_positive,
        "modules_by_abs": modules_by_abs,
        "modules_by_positive_density": modules_by_density,
        "qkv_head_groups_by_positive": qkv_groups,
        "mlp_hidden_groups_by_positive": mlp_groups,
        "top_positive_scalars": top_positive_scalars,
        "top_abs_scalars": top_abs_scalars,
        "pruning": pruning,
    }
    (config.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scalar-level weight circuit attribution and pruning for Depth Anything V2 on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/scalar_weight_circuit_da2k"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target", choices=["transformer", "all-linear"], default="transformer")
    parser.add_argument("--layer-indices", default="", help="Comma-separated transformer block indices. Empty means all selected target modules.")
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--score-images", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--prune-budgets", default="", help="Comma-separated exact global zero budgets to evaluate.")
    parser.add_argument(
        "--prune-score",
        choices=[
            "abs_wgrad",
            "positive_wgrad",
            "magnitude",
            "hybrid_abs_mag",
            "hybrid_protect_mag",
            "taylor1_abs",
            "taylor2_abs",
            "taylor3_abs",
            "taylor1_damage",
            "taylor2_damage",
            "taylor3_damage",
        ],
        default="abs_wgrad",
    )
    parser.add_argument("--hybrid-alpha", type=float, default=1.0)
    parser.add_argument("--loss-tau", type=float, default=1.0, help="Temperature for Taylor loss softplus(-margin / tau).")
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ScalarCircuitConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        target=args.target,
        layer_indices=parse_int_tuple(args.layer_indices),
        scene_type=args.scene_type,
        max_images=args.max_images,
        score_images=args.score_images,
        max_pairs=args.max_pairs,
        prune_budgets=parse_budget_tuple(args.prune_budgets),
        prune_score=args.prune_score,
        hybrid_alpha=args.hybrid_alpha,
        loss_tau=args.loss_tau,
        save_scores=args.save_scores,
        eval_baseline=not args.skip_baseline,
        log_every=args.log_every,
    )
    result = run(config)
    print(json.dumps(result["score_metadata"], indent=2, sort_keys=True, default=str))
    if result["pruning"]["enabled"]:
        print(json.dumps(result["pruning"]["results"], indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
