from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from saliency.prune_eval import lowest_saliency_mask, lowest_saliency_mask_per_output_row

_MAX_DIAGNOSTIC_VALUES = 1_000_000


def _safe_float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def _sample_flat(values: torch.Tensor, max_values: int = _MAX_DIAGNOSTIC_VALUES) -> torch.Tensor:
    flat = values.detach().flatten().float().cpu()
    if flat.numel() <= max_values:
        return flat
    stride = math.ceil(flat.numel() / max_values)
    return flat[::stride][:max_values]


def _sample_pair(a: torch.Tensor, b: torch.Tensor, max_values: int = _MAX_DIAGNOSTIC_VALUES) -> tuple[torch.Tensor, torch.Tensor]:
    if a.numel() != b.numel():
        raise ValueError(f"paired diagnostic inputs must have same numel: {a.numel()} != {b.numel()}")
    af = a.detach().flatten().float().cpu()
    bf = b.detach().flatten().float().cpu()
    if af.numel() <= max_values:
        return af, bf
    stride = math.ceil(af.numel() / max_values)
    return af[::stride][:max_values], bf[::stride][:max_values]


def _quantiles(values: torch.Tensor, qs: tuple[float, ...]) -> dict[str, float]:
    if values.numel() == 0:
        return {f"q{int(q * 100):02d}": 0.0 for q in qs}
    v = _sample_flat(values)
    return {f"q{int(q * 100):02d}": _safe_float(torch.quantile(v, q)) for q in qs}


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    flat = values.detach().flatten().float()
    order = torch.argsort(flat, stable=True)
    ranks = torch.empty(flat.numel(), dtype=torch.float32, device=flat.device)
    ranks[order] = torch.arange(flat.numel(), dtype=torch.float32, device=flat.device)
    return ranks


def spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() != b.numel():
        raise ValueError(f"spearman inputs must have same numel: {a.numel()} != {b.numel()}")
    if a.numel() < 2:
        return 0.0
    af, bf = _sample_pair(a, b)
    ar = _rankdata(af)
    br = _rankdata(bf)
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = ar.norm() * br.norm()
    if float(denom.item()) == 0.0:
        return 0.0
    return _safe_float(ar.dot(br) / denom)


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() != b.numel():
        raise ValueError(f"pearson inputs must have same numel: {a.numel()} != {b.numel()}")
    if a.numel() < 2:
        return 0.0
    af, bf = _sample_pair(a, b)
    af = af - af.mean()
    bf = bf - bf.mean()
    denom = af.norm() * bf.norm()
    if float(denom.item()) == 0.0:
        return 0.0
    return _safe_float(af.dot(bf) / denom)


def pruning_mask(score: torch.Tensor, *, fraction: float, pruning_scope: str) -> torch.Tensor:
    if pruning_scope == "per_matrix":
        return lowest_saliency_mask(score, fraction=fraction)
    if pruning_scope == "per_output_row":
        return lowest_saliency_mask_per_output_row(score, fraction=fraction)
    raise ValueError(f"unknown pruning_scope: {pruning_scope}")


def _lowest_flat_mask(score: torch.Tensor, *, fraction: float) -> torch.Tensor:
    count = int(score.numel() * float(fraction))
    if count <= 0:
        return torch.zeros_like(score, dtype=torch.bool)
    order = torch.argsort(score, stable=True)
    mask = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    mask[order[:count]] = True
    return mask


def tensor_weight_summary(name: str, tensor: torch.Tensor, *, bottom_fraction: float = 0.25) -> dict[str, Any]:
    flat = tensor.detach().flatten().float().cpu()
    abs_flat = flat.abs()
    numel = int(flat.numel())
    if numel == 0:
        return {"name": name, "numel": 0}
    sample_abs = _sample_flat(abs_flat)
    bottom_count = max(1, int(sample_abs.numel() * float(bottom_fraction)))
    top_count = max(1, int(sample_abs.numel() * float(bottom_fraction)))
    sorted_abs = torch.sort(sample_abs).values
    l1 = sample_abs.sum().clamp_min(1e-30)
    l2 = sample_abs.square().sum().clamp_min(1e-30)
    q = _quantiles(abs_flat, (0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
    return {
        "name": name,
        "shape": list(tensor.shape),
        "numel": numel,
        "sample_numel": int(sample_abs.numel()),
        "mean": _safe_float(flat.mean()),
        "std": _safe_float(flat.std(unbiased=False)),
        "abs_mean": _safe_float(abs_flat.mean()),
        "abs_std": _safe_float(abs_flat.std(unbiased=False)),
        "abs_max": _safe_float(abs_flat.max()),
        "zero_fraction": _safe_float((abs_flat == 0).float().mean()),
        "bottom_fraction": float(bottom_fraction),
        "bottom_abs_l1_fraction": _safe_float(sorted_abs[:bottom_count].sum() / l1),
        "bottom_abs_l2_fraction": _safe_float(sorted_abs[:bottom_count].square().sum() / l2),
        "top_abs_l1_fraction": _safe_float(sorted_abs[-top_count:].sum() / l1),
        "top_abs_l2_fraction": _safe_float(sorted_abs[-top_count:].square().sum() / l2),
        **{f"abs_{key}": value for key, value in q.items()},
    }


def score_tensor_summary(
    name: str,
    score: torch.Tensor,
    weight: torch.Tensor,
    *,
    prune_fraction: float = 0.25,
    pruning_scope: str = "per_matrix",
) -> dict[str, Any]:
    if tuple(score.shape) != tuple(weight.shape):
        raise ValueError(f"score and weight shape mismatch for {name}: {tuple(score.shape)} != {tuple(weight.shape)}")
    score_flat = score.detach().flatten().float().cpu()
    if score_flat.numel() <= _MAX_DIAGNOSTIC_VALUES:
        score_for_mask = score.detach().float().cpu()
        abs_weight = weight.detach().abs().flatten().float().cpu()
        mask = pruning_mask(score_for_mask, fraction=prune_fraction, pruning_scope=pruning_scope).flatten()
        sampled = False
    else:
        score_flat, abs_weight = _sample_pair(score, weight.detach().abs())
        mask = _lowest_flat_mask(score_flat, fraction=prune_fraction)
        sampled = True
    pruned_weight = abs_weight[mask]
    kept_weight = abs_weight[~mask]
    weight_l2 = abs_weight.square().sum().clamp_min(1e-30)
    q = _quantiles(score_flat, (0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
    return {
        "name": name,
        "shape": list(score.shape),
        "numel": int(score.numel()),
        "sample_numel": int(score_flat.numel()),
        "sampled": sampled,
        "score_mean": _safe_float(score_flat.mean()),
        "score_std": _safe_float(score_flat.std(unbiased=False)),
        "score_max": _safe_float(score_flat.max()) if score_flat.numel() else 0.0,
        "score_zero_fraction": _safe_float((score_flat == 0).float().mean()) if score_flat.numel() else 0.0,
        "spearman_score_abs_weight": spearman_corr(score_flat, abs_weight),
        "pearson_score_abs_weight": pearson_corr(score_flat, abs_weight),
        "score_pruned_mean_abs_weight": _safe_float(pruned_weight.mean()) if pruned_weight.numel() else 0.0,
        "score_kept_mean_abs_weight": _safe_float(kept_weight.mean()) if kept_weight.numel() else 0.0,
        "score_pruned_l2_weight_fraction": _safe_float(pruned_weight.square().sum() / weight_l2) if pruned_weight.numel() else 0.0,
        **{f"score_{key}": value for key, value in q.items()},
    }


def score_pair_diagnostics(
    left_name: str,
    right_name: str,
    left_score: torch.Tensor,
    right_score: torch.Tensor,
    *,
    prune_fraction: float = 0.25,
    pruning_scope: str = "per_matrix",
) -> dict[str, Any]:
    if tuple(left_score.shape) != tuple(right_score.shape):
        raise ValueError(
            f"score shape mismatch for {left_name}/{right_name}: "
            f"{tuple(left_score.shape)} != {tuple(right_score.shape)}"
        )
    if left_score.numel() <= _MAX_DIAGNOSTIC_VALUES:
        left = left_score.detach().flatten().float().cpu()
        right = right_score.detach().flatten().float().cpu()
        left_mask = pruning_mask(left_score.detach().float().cpu(), fraction=prune_fraction, pruning_scope=pruning_scope).flatten()
        right_mask = pruning_mask(right_score.detach().float().cpu(), fraction=prune_fraction, pruning_scope=pruning_scope).flatten()
        sampled = False
    else:
        left, right = _sample_pair(left_score, right_score)
        left_mask = _lowest_flat_mask(left, fraction=prune_fraction)
        right_mask = _lowest_flat_mask(right, fraction=prune_fraction)
        sampled = True
    intersection = int((left_mask & right_mask).sum().item())
    union = int((left_mask | right_mask).sum().item())
    left_count = int(left_mask.sum().item())
    right_count = int(right_mask.sum().item())
    return {
        "left": left_name,
        "right": right_name,
        "shape": list(left_score.shape),
        "numel": int(left_score.numel()),
        "sample_numel": int(left.numel()),
        "sampled": sampled,
        "spearman": spearman_corr(left, right),
        "pearson_log1p": pearson_corr(torch.log1p(left.clamp_min(0.0)), torch.log1p(right.clamp_min(0.0))),
        "left_mask_count": left_count,
        "right_mask_count": right_count,
        "mask_intersection": intersection,
        "mask_union": union,
        "mask_jaccard": intersection / max(union, 1),
        "mask_overlap_fraction": intersection / max(min(left_count, right_count), 1),
    }


def aggregate_score_pair_diagnostics(
    left_name: str,
    right_name: str,
    left_scores: Mapping[str, torch.Tensor],
    right_scores: Mapping[str, torch.Tensor],
    *,
    prune_fraction: float = 0.25,
    pruning_scope: str = "per_matrix",
) -> dict[str, Any]:
    rows = []
    weights = []
    for name, left_score in left_scores.items():
        right_score = right_scores.get(name)
        if right_score is None or left_score.ndim != 2 or tuple(left_score.shape) != tuple(right_score.shape):
            continue
        row = score_pair_diagnostics(
            left_name,
            right_name,
            left_score,
            right_score,
            prune_fraction=prune_fraction,
            pruning_scope=pruning_scope,
        )
        row["name"] = name
        rows.append(row)
        weights.append(int(row["numel"]))
    total = sum(weights)
    if not rows:
        return {"left": left_name, "right": right_name, "matrix_count": 0, "rows": []}
    return {
        "left": left_name,
        "right": right_name,
        "matrix_count": len(rows),
        "weights": total,
        "mean_spearman": sum(float(r["spearman"]) for r in rows) / len(rows),
        "weighted_spearman": sum(float(r["spearman"]) * int(r["numel"]) for r in rows) / max(total, 1),
        "mean_mask_jaccard": sum(float(r["mask_jaccard"]) for r in rows) / len(rows),
        "weighted_mask_jaccard": sum(float(r["mask_jaccard"]) * int(r["numel"]) for r in rows) / max(total, 1),
        "mean_overlap_fraction": sum(float(r["mask_overlap_fraction"]) for r in rows) / len(rows),
        "weighted_overlap_fraction": sum(float(r["mask_overlap_fraction"]) * int(r["numel"]) for r in rows) / max(total, 1),
        "rows": rows,
    }
