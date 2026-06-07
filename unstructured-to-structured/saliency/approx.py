from __future__ import annotations

import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
import json
from math import prod
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

from saliency.calibration import batched, build_causal_lm_batch
from saliency.experiment import resolve_device, resolve_torch_dtype
from saliency.gptq_eval import linear_modules
from saliency.prune_eval import (
    evaluate_perplexity,
    lowest_saliency_mask,
    lowest_saliency_mask_per_output_row,
    summarize_ppl_change,
)
from saliency.saliency import save_saliency_artifacts


@dataclass(slots=True)
class ApproxSaliencyConfig:
    output_dir: str | Path
    model_name: str = "EleutherAI/pythia-31m"
    method: str = "wanda"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    eval_split: str = ""
    max_examples: int = 128
    max_eval_examples: int = 0
    batch_size: int = 8
    max_length: int = 512
    dtype: str = "bf16"
    device: str = "auto"
    answer_only_loss: bool = True
    top_k: int = 50
    revision: str | None = None
    angular_hybrid_lambda: float = 0.5
    feature_cosine_alpha: float = 0.05
    feature_cosine_clip: float = 10.0
    graph_num_probes: int = 4
    graph_seed: int = 17
    local_forward_eps: float = 1e-3
    superset_gain_power: float = 1.0
    superset_gain_clip_quantile: float = 0.0


@dataclass(slots=True)
class IterativeApproxPruneConfig:
    output_dir: str | Path
    model_name: str = "EleutherAI/pythia-31m"
    method: str = "wanda"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    eval_split: str = ""
    max_examples: int = 128
    max_eval_examples: int = 0
    batch_size: int = 8
    max_length: int = 512
    dtype: str = "bf16"
    device: str = "auto"
    answer_only_loss: bool = True
    prune_fraction: float = 0.25
    prune_chunk_fraction: float = 0.05
    recompute_every_weights: int = 0
    pruning_structure: str = "unstructured"
    structured_n: int = 2
    structured_m: int = 4
    structured_group_dim: int = 1
    matrix_limit: int = 0
    repair_with_gptq_gd: bool = False
    repair_with_loss_gd: bool = False
    repair_learning_rate: float = 1e-5
    revision: str | None = None


@dataclass(slots=True)
class WandaAblationPruneConfig:
    output_dir: str | Path
    model_name: str = "EleutherAI/pythia-31m"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    eval_split: str = ""
    max_examples: int = 128
    max_eval_examples: int = 0
    batch_size: int = 8
    max_length: int = 512
    dtype: str = "bf16"
    device: str = "auto"
    answer_only_loss: bool = True
    prune_fraction: float = 0.25
    pruning_scope: str = "per_matrix"
    wanda_activation: str = "masked"
    wanda_schedule: str = "one_shot"
    wanda_method: str = "wanda"
    superset_gain_power: float = 1.0
    superset_gain_clip_quantile: float = 0.0
    revision: str | None = None


def weight_magnitude_scores(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    matrices_only: bool = True,
) -> dict[str, torch.Tensor]:
    scores: dict[str, torch.Tensor] = {}
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if matrices_only and param.ndim != 2:
            continue
        scores[name] = param.detach().abs().to(device="cpu", dtype=torch.float32)
    return scores


def relative_importance_scores(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    activation_rms: dict[str, torch.Tensor] | None = None,
    activation_exponent: float = 0.5,
    matrices_only: bool = True,
) -> dict[str, torch.Tensor]:
    scores: dict[str, torch.Tensor] = {}
    activation_rms = activation_rms or {}
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if matrices_only and param.ndim != 2:
            continue
        weight_abs = param.detach().abs().to(device="cpu", dtype=torch.float32)
        if weight_abs.ndim != 2:
            continue
        row_l1 = weight_abs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        col_l1 = weight_abs.sum(dim=0, keepdim=True).clamp_min(1e-12)
        score = weight_abs.div(col_l1).add_(weight_abs.div(row_l1))
        input_rms = activation_rms.get(name)
        if input_rms is not None:
            input_rms = input_rms.to(device="cpu", dtype=torch.float32)
            if input_rms.numel() != weight_abs.shape[1]:
                raise ValueError(f"RIA input dimension mismatch for {name}: {input_rms.numel()} != {weight_abs.shape[1]}")
            score.mul_(input_rms.clamp_min(0.0).pow(float(activation_exponent)).unsqueeze(0))
        scores[name] = score
    return scores


def input_activation_stat_scores(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    activation_stats: dict[str, dict[str, torch.Tensor]],
    method: str,
    *,
    matrices_only: bool = True,
) -> dict[str, torch.Tensor]:
    method = method.lower()
    squared_methods = {"output_l2", "output_damage", "local_reconstruction", "squared_wanda", "var_output", "variance_output"}
    scores: dict[str, torch.Tensor] = {}
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if matrices_only and param.ndim != 2:
            continue
        weight_abs = param.detach().abs().to(device="cpu", dtype=torch.float32)
        stats = activation_stats.get(name)
        if stats is None:
            scores[name] = weight_abs.square() if method in squared_methods else weight_abs
            continue

        if method in {"output_l2", "output_damage", "local_reconstruction", "squared_wanda"}:
            scale = stats["sumsq"]
            weight_term = weight_abs.square()
        elif method in {"mean_abs_wanda", "wanda_mean_abs", "mean_abs"}:
            scale = stats["mean_abs"]
            weight_term = weight_abs
        elif method in {"var_output", "variance_output"}:
            scale = stats["variance"]
            weight_term = weight_abs.square()
        elif method in {"q95_wanda", "wanda_q95", "outlier_q95"}:
            scale = stats["q95_abs"]
            weight_term = weight_abs
        elif method in {"max_wanda", "wanda_max", "outlier_max"}:
            scale = stats["max_abs"]
            weight_term = weight_abs
        else:
            raise ValueError(f"unknown input activation stat method: {method}")

        scale = scale.to(device="cpu", dtype=torch.float32)
        if scale.numel() != weight_abs.shape[1]:
            raise ValueError(f"activation statistic dimension mismatch for {name}: {scale.numel()} != {weight_abs.shape[1]}")
        scores[name] = weight_term.mul(scale.unsqueeze(0))
    return scores


def angular_saliency_scores(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    angular_stats: dict[str, dict[str, torch.Tensor]],
    method: str,
    *,
    hybrid_lambda: float = 0.5,
    matrices_only: bool = True,
) -> dict[str, torch.Tensor]:
    method = method.lower()
    exact_methods = {"angular", "angular_exact", "pure_angular"}
    approx_methods = {"angular_approx", "approx_angular"}
    hybrid_methods = {"angular_hybrid", "hybrid_angular", "angular_energy_hybrid"}
    if method not in exact_methods | approx_methods | hybrid_methods:
        raise ValueError(f"unknown angular saliency method: {method}")

    scores: dict[str, torch.Tensor] = {}
    eps = 1e-12
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if matrices_only and param.ndim != 2:
            continue
        stats = angular_stats.get(name)
        if stats is None:
            scores[name] = param.detach().to(device="cpu", dtype=torch.float32).square()
            continue

        device = stats["yx_dot"].device
        weight = param.detach().to(device=device, dtype=torch.float32)
        x_sumsq = stats["x_sumsq"].to(device=device, dtype=torch.float32).unsqueeze(0)
        y_sumsq = stats["y_sumsq"].to(device=device, dtype=torch.float32).unsqueeze(1).clamp_min(eps)
        yx_dot = stats["yx_dot"].to(device=device, dtype=torch.float32)
        if tuple(yx_dot.shape) != tuple(weight.shape):
            raise ValueError(f"angular yx_dot shape mismatch for {name}: {tuple(yx_dot.shape)} != {tuple(weight.shape)}")
        if x_sumsq.shape[1] != weight.shape[1]:
            raise ValueError(f"angular input dimension mismatch for {name}: {x_sumsq.shape[1]} != {weight.shape[1]}")
        if y_sumsq.shape[0] != weight.shape[0]:
            raise ValueError(f"angular output dimension mismatch for {name}: {y_sumsq.shape[0]} != {weight.shape[0]}")

        weight_sq = weight.square()
        if method in exact_methods:
            dot_after = y_sumsq - weight * yx_dot
            norm_after = (y_sumsq - 2.0 * weight * yx_dot + weight_sq * x_sumsq).clamp_min(eps).sqrt()
            cos = dot_after.div(y_sumsq.sqrt() * norm_after).clamp_(-1.0, 1.0)
            score = 1.0 - cos
        else:
            projection = yx_dot.square().div(y_sumsq)
            lam = 1.0 if method in approx_methods else float(hybrid_lambda)
            score = weight_sq.div(y_sumsq).mul((x_sumsq - lam * projection).clamp_min_(0.0))
        scores[name] = score.clamp_min_(0.0).to(device="cpu", dtype=torch.float32)
    return scores


def feature_wanda_cosine_scores(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    stats_by_name: dict[str, dict[str, torch.Tensor]],
    *,
    alpha: float = 0.05,
    cosine_clip: float = 10.0,
    matrices_only: bool = True,
) -> dict[str, torch.Tensor]:
    scores: dict[str, torch.Tensor] = {}
    eps = 1e-12
    alpha = float(alpha)
    cosine_clip = float(cosine_clip)
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if matrices_only and param.ndim != 2:
            continue
        weight_abs_cpu = param.detach().abs().to(device="cpu", dtype=torch.float32)
        stats = stats_by_name.get(name)
        if stats is None:
            scores[name] = weight_abs_cpu
            continue

        device = stats["weighted_abs_input"].device
        weight = param.detach().to(device=device, dtype=torch.float32)
        weight_abs = weight.abs()
        weighted_abs_input = stats["weighted_abs_input"].to(device=device, dtype=torch.float32)
        row_weight_sums = stats["row_weight_sums"].to(device=device, dtype=torch.float32)
        if tuple(weighted_abs_input.shape) != tuple(weight.shape):
            raise ValueError(
                f"feature-WANDA cosine stat shape mismatch for {name}: "
                f"{tuple(weighted_abs_input.shape)} != {tuple(weight.shape)}"
            )
        feature_activation = weighted_abs_input.div(row_weight_sums.clamp_min(eps).unsqueeze(1))
        base = weight_abs.mul(feature_activation)

        if alpha != 0.0:
            x_sumsq = stats["x_sumsq"].to(device=device, dtype=torch.float32).unsqueeze(0)
            y_sumsq = stats["y_sumsq"].to(device=device, dtype=torch.float32).unsqueeze(1).clamp_min(eps)
            yx_dot = stats["yx_dot"].to(device=device, dtype=torch.float32)
            if x_sumsq.shape[1] != weight.shape[1]:
                raise ValueError(f"feature-WANDA cosine input dimension mismatch for {name}: {x_sumsq.shape[1]} != {weight.shape[1]}")
            if y_sumsq.shape[0] != weight.shape[0]:
                raise ValueError(f"feature-WANDA cosine output dimension mismatch for {name}: {y_sumsq.shape[0]} != {weight.shape[0]}")
            if tuple(yx_dot.shape) != tuple(weight.shape):
                raise ValueError(f"feature-WANDA cosine yx_dot shape mismatch for {name}: {tuple(yx_dot.shape)} != {tuple(weight.shape)}")

            weight_sq = weight.square()
            dot_after = y_sumsq - weight * yx_dot
            norm_after = (y_sumsq - 2.0 * weight * yx_dot + weight_sq * x_sumsq).clamp_min(eps).sqrt()
            cos = dot_after.div(y_sumsq.sqrt() * norm_after).clamp_(-1.0, 1.0)
            cosine_damage = (1.0 - cos).clamp_min_(0.0)
            cosine_norm = cosine_damage.div(cosine_damage.mean().clamp_min(eps))
            if cosine_clip > 0.0:
                cosine_norm.clamp_(max=cosine_clip)
            base.mul_(1.0 + alpha * cosine_norm)

        scores[name] = base.to(device="cpu", dtype=torch.float32)
    return scores


def _layernorm_affine_weight(norm: torch.nn.Module, *, device: torch.device, width: int) -> torch.Tensor:
    weight = getattr(norm, "weight", None)
    if torch.is_tensor(weight):
        return weight.detach().to(device=device, dtype=torch.float32)
    return torch.ones(width, device=device, dtype=torch.float32)


def _layernorm_centered(inputs: torch.Tensor, norm: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = inputs.detach().reshape(-1, inputs.shape[-1]).to(dtype=torch.float32)
    eps = float(getattr(norm, "eps", 1e-5))
    centered = flat - flat.mean(dim=-1, keepdim=True)
    inv_std = centered.square().mean(dim=-1, keepdim=True).add(eps).rsqrt()
    xhat = centered * inv_std
    return flat, xhat, inv_std


def layernorm_input_jacobian_colnorm_squares(inputs: torch.Tensor, norm: torch.nn.Module) -> torch.Tensor:
    flat, xhat, inv_std = _layernorm_centered(inputs, norm)
    width = flat.shape[-1]
    gamma = _layernorm_affine_weight(norm, device=flat.device, width=width)
    g2 = gamma.square().unsqueeze(0).mul(inv_std.square())
    d = float(width)
    g0 = g2.sum(dim=-1, keepdim=True)
    g1 = g2.mul(xhat).sum(dim=-1, keepdim=True)
    g2x = g2.mul(xhat.square()).sum(dim=-1, keepdim=True)
    base = (g0 + 2.0 * xhat * g1 + xhat.square() * g2x).div(d * d)
    diagonal_old = ((1.0 + xhat.square()) / d).square()
    diagonal_new = (1.0 - (1.0 + xhat.square()) / d).square()
    return base.add(g2.mul(diagonal_new - diagonal_old)).clamp_min_(0.0)


def layernorm_input_projected_gain_from_vectors(
    inputs: torch.Tensor,
    norm: torch.nn.Module,
    projected_vectors: torch.Tensor,
) -> torch.Tensor:
    flat, xhat, inv_std = _layernorm_centered(inputs, norm)
    width = flat.shape[-1]
    if projected_vectors.ndim != 2 or projected_vectors.shape[1] != width:
        raise ValueError(f"projected vectors must have shape [probes, {width}]")
    if projected_vectors.shape[0] <= 0:
        raise ValueError("at least one projected vector is required")
    gamma = _layernorm_affine_weight(norm, device=flat.device, width=width)
    vectors = projected_vectors.to(device=flat.device, dtype=torch.float32).mul(gamma.unsqueeze(0))
    mean_vectors = vectors.mean(dim=-1)
    mean_vectors_xhat = xhat.matmul(vectors.transpose(0, 1)).div(float(width))
    vjp = (
        vectors.unsqueeze(0)
        .sub(mean_vectors.view(1, -1, 1))
        .sub(xhat.unsqueeze(1).mul(mean_vectors_xhat.unsqueeze(-1)))
        .mul(inv_std.unsqueeze(1))
    )
    return vjp.square().mean(dim=1)


def layernorm_input_downstream_colnorm_squares(
    inputs: torch.Tensor,
    norm: torch.nn.Module,
    downstream_weight: torch.Tensor,
) -> torch.Tensor:
    flat, xhat, inv_std = _layernorm_centered(inputs, norm)
    width = flat.shape[-1]
    if downstream_weight.ndim != 2 or downstream_weight.shape[1] != width:
        raise ValueError(f"downstream_weight must have shape [out_features, {width}]")
    gamma = _layernorm_affine_weight(norm, device=flat.device, width=width)
    weight = downstream_weight.detach().to(device=flat.device, dtype=torch.float32)
    gram = weight.transpose(0, 1).matmul(weight)
    metric = gram.mul(gamma.unsqueeze(1)).mul(gamma.unsqueeze(0))
    d = float(width)
    colsum = metric.sum(dim=0)
    diag = metric.diag()
    total1 = colsum.sum()
    metric_x = xhat.matmul(metric)
    total1x = xhat.matmul(colsum)
    totalxx = metric_x.mul(xhat).sum(dim=-1)
    gain = (
        diag.unsqueeze(0)
        .sub(colsum.mul(2.0 / d).unsqueeze(0))
        .sub(xhat.mul(metric_x).mul(2.0 / d))
        .add(total1 / (d * d))
        .add(xhat.mul(total1x.unsqueeze(1)).mul(2.0 / (d * d)))
        .add(xhat.square().mul(totalxx.unsqueeze(1)).div(d * d))
    )
    return gain.mul(inv_std.square()).clamp_min_(0.0)


def _rmsnorm_affine_weight(norm: torch.nn.Module, *, device: torch.device, width: int) -> torch.Tensor:
    weight = getattr(norm, "weight", None)
    if torch.is_tensor(weight):
        return weight.detach().to(device=device, dtype=torch.float32)
    return torch.ones(width, device=device, dtype=torch.float32)


def _rmsnorm_eps(norm: torch.nn.Module) -> float:
    return float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))


def _rmsnorm_metric_diag_from_products(
    inputs: torch.Tensor,
    norm: torch.nn.Module,
    metric_diag: torch.Tensor,
    metric_gamma_x: torch.Tensor,
    gamma_x_metric_gamma_x: torch.Tensor,
) -> torch.Tensor:
    x = inputs.detach().to(dtype=torch.float32)
    width = x.shape[-1]
    gamma = _rmsnorm_affine_weight(norm, device=x.device, width=width)
    inv = x.square().mean(dim=-1, keepdim=True).add(_rmsnorm_eps(norm)).rsqrt()
    coeff = inv.pow(3).div(float(width))
    return (
        inv.square().mul(gamma.square()).mul(metric_diag)
        .sub(2.0 * inv * coeff * x * gamma * metric_gamma_x)
        .add(coeff.square() * x.square() * gamma_x_metric_gamma_x.unsqueeze(-1))
    ).clamp_min_(0.0)


def rmsnorm_input_downstream_colnorm_squares(
    inputs: torch.Tensor,
    norm: torch.nn.Module,
    downstream_weight: torch.Tensor,
) -> torch.Tensor:
    flat = inputs.detach().reshape(-1, inputs.shape[-1]).to(dtype=torch.float32)
    width = flat.shape[-1]
    if downstream_weight.ndim != 2 or downstream_weight.shape[1] != width:
        raise ValueError(f"downstream_weight must have shape [out_features, {width}]")
    gamma = _rmsnorm_affine_weight(norm, device=flat.device, width=width)
    weight = downstream_weight.detach().to(device=flat.device, dtype=torch.float32)
    gram = weight.transpose(0, 1).matmul(weight)
    z = flat.mul(gamma.unsqueeze(0))
    metric_diag = gram.diag().unsqueeze(0).expand_as(flat)
    metric_gamma_x = z.matmul(gram)
    gamma_x_metric_gamma_x = metric_gamma_x.mul(z).sum(dim=-1)
    return _rmsnorm_metric_diag_from_products(flat, norm, metric_diag, metric_gamma_x, gamma_x_metric_gamma_x)


def layernorm_forward_diff_colnorm_squares(inputs: torch.Tensor, norm: torch.nn.Module, *, eps: float = 1e-3) -> torch.Tensor:
    flat, _, _ = _layernorm_centered(inputs, norm)
    width = flat.shape[-1]
    if width <= 1:
        return torch.zeros_like(flat)
    eps_value = float(eps)
    if eps_value <= 0.0:
        raise ValueError("eps must be positive")
    gamma = _layernorm_affine_weight(norm, device=flat.device, width=width)
    gamma_sq = gamma.square().unsqueeze(0)
    norm_eps = float(getattr(norm, "eps", 1e-5))
    centered = flat - flat.mean(dim=-1, keepdim=True)
    var = centered.square().mean(dim=-1, keepdim=True)
    inv_std = var.add(norm_eps).rsqrt()
    d = float(width)

    shifted_inv_std = (
        var
        .add(centered.mul(2.0 * eps_value / d))
        .add((eps_value * eps_value) * (d - 1.0) / (d * d))
        .add(norm_eps)
        .rsqrt()
    )
    delta_inv = shifted_inv_std - inv_std
    gamma0 = gamma_sq.sum(dim=-1, keepdim=True)
    gamma_c = gamma_sq.mul(centered).sum(dim=-1, keepdim=True)
    gamma_c2 = gamma_sq.mul(centered.square()).sum(dim=-1, keepdim=True)
    common_shift = eps_value / d
    common = (
        delta_inv.square().mul(gamma_c2)
        .sub(2.0 * common_shift * shifted_inv_std * delta_inv * gamma_c)
        .add((common_shift * common_shift) * shifted_inv_std.square() * gamma0)
    )
    base_delta = centered.mul(delta_inv).sub(common_shift * shifted_inv_std)
    target_delta = centered.mul(delta_inv).add(eps_value * (1.0 - 1.0 / d) * shifted_inv_std)
    gain = common.sub(gamma_sq.mul(base_delta.square())).add(gamma_sq.mul(target_delta.square()))
    return gain.div(eps_value * eps_value).clamp_min_(0.0)


def activation_forward_unit_gain(
    preactivation: torch.Tensor,
    activation: torch.nn.Module,
    *,
    eps: float = 1e-3,
) -> torch.Tensor:
    eps_value = float(eps)
    if eps_value <= 0.0:
        raise ValueError("eps must be positive")
    y = preactivation.detach().to(dtype=torch.float32)
    baseline = activation(y)
    plus = activation(y + eps_value).sub(baseline).square()
    minus = activation(y - eps_value).sub(baseline).square()
    return plus.add_(minus).mul_(0.5 / (eps_value * eps_value))


def activation_local_jacobian_square(preactivation: torch.Tensor, activation: torch.nn.Module) -> torch.Tensor:
    y = preactivation.detach().to(dtype=torch.float32)
    name = activation.__class__.__name__.lower()
    if isinstance(activation, torch.nn.GELU) or name in {"geluactivation", "gelu"}:
        approximate = str(getattr(activation, "approximate", "none"))
        if approximate == "tanh":
            c = (2.0 / torch.pi) ** 0.5
            k = 0.044715
            u = c * (y + k * y.pow(3))
            tanh_u = torch.tanh(u)
            du = c * (1.0 + 3.0 * k * y.square())
            grad = 0.5 * (1.0 + tanh_u) + 0.5 * y * (1.0 - tanh_u.square()) * du
        else:
            normal_pdf = torch.exp(-0.5 * y.square()) * 0.3989422804014327
            grad = torch.special.ndtr(y) + y * normal_pdf
        return grad.square()
    if "newgelu" in name:
        c = (2.0 / torch.pi) ** 0.5
        k = 0.044715
        u = c * (y + k * y.pow(3))
        tanh_u = torch.tanh(u)
        du = c * (1.0 + 3.0 * k * y.square())
        grad = 0.5 * (1.0 + tanh_u) + 0.5 * y * (1.0 - tanh_u.square()) * du
        return grad.square()
    if "fastgelu" in name:
        sig = torch.sigmoid(1.702 * y)
        grad = sig + 1.702 * y * sig * (1.0 - sig)
        return grad.square()
    return activation_forward_unit_gain(y, activation)


def linear_superset_wanda_scores(
    weight: torch.Tensor,
    inputs: torch.Tensor,
    output_gain: torch.Tensor,
) -> torch.Tensor:
    flat_x = inputs.detach().reshape(-1, inputs.shape[-1]).to(dtype=torch.float32)
    gain = output_gain.detach().to(device=flat_x.device, dtype=torch.float32)
    if gain.ndim == 1:
        if gain.numel() != weight.shape[0]:
            raise ValueError(f"output_gain has {gain.numel()} entries, expected {weight.shape[0]}")
        stat = gain.unsqueeze(1).mul(flat_x.square().sum(dim=0).unsqueeze(0))
    elif gain.ndim == 2:
        if gain.shape != (flat_x.shape[0], weight.shape[0]):
            raise ValueError(f"output_gain shape {tuple(gain.shape)} != {(flat_x.shape[0], weight.shape[0])}")
        stat = gain.transpose(0, 1).matmul(flat_x.square())
    else:
        raise ValueError("output_gain must be 1D [out_features] or 2D [tokens, out_features]")
    return weight.detach().to(device=flat_x.device, dtype=torch.float32).square().mul(stat)


def transform_superset_gain(
    gain: torch.Tensor,
    *,
    power: float = 1.0,
    clip_quantile: float = 0.0,
) -> torch.Tensor:
    out = gain.detach().to(dtype=torch.float32).clamp_min(0.0)
    if clip_quantile > 0.0:
        if not 0.0 < float(clip_quantile) < 1.0:
            raise ValueError("clip_quantile must be in (0, 1) when enabled")
        if out.numel() > 0:
            flat = out.flatten()
            if flat.numel() > 1_000_000:
                stride = (flat.numel() + 999_999) // 1_000_000
                flat = flat[::stride][:1_000_000]
            out = out.clamp_max(torch.quantile(flat, float(clip_quantile), interpolation="lower"))
    if float(power) != 1.0:
        if float(power) <= 0.0:
            raise ValueError("power must be positive")
        out = out.pow(float(power))
    return out


def qwen_mlp_superset_output_gains(
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate = gate_output.detach().to(dtype=torch.float32)
    up = up_output.detach().to(device=gate.device, dtype=torch.float32)
    if gate.shape != up.shape:
        raise ValueError(f"gate_output shape {tuple(gate.shape)} != up_output shape {tuple(up.shape)}")
    if down_weight.ndim != 2 or down_weight.shape[1] != gate.shape[-1]:
        raise ValueError(f"down_weight must have shape [out_features, {gate.shape[-1]}]")
    colnorm = down_weight.detach().to(device=gate.device, dtype=torch.float32).square().sum(dim=0)
    sigmoid = torch.sigmoid(gate)
    silu = gate.mul(sigmoid)
    silu_grad = sigmoid.mul(1.0 + gate.mul(1.0 - sigmoid))
    gate_gain = silu_grad.square().mul(up.square()).mul(colnorm.view(*([1] * (gate.ndim - 1)), -1))
    up_gain = silu.square().mul(colnorm.view(*([1] * (gate.ndim - 1)), -1))
    return gate_gain, up_gain


def _qwen_rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _qwen_apply_rotary_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x.mul(cos.unsqueeze(1)).add(_qwen_rotate_half(x).mul(sin.unsqueeze(1)))


def _apply_rotary_gain_diagonal(
    gain: torch.Tensor,
    cross: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    rotary_dim = int(cos.shape[-1])
    if rotary_dim <= 0:
        return gain
    half = rotary_dim // 2
    if half <= 0:
        return gain
    out = gain.clone()
    first = gain[..., :half]
    second = gain[..., half:rotary_dim]
    c1 = cos[..., :half].to(device=gain.device, dtype=gain.dtype)
    s1 = sin[..., :half].to(device=gain.device, dtype=gain.dtype)
    c2 = cos[..., half:rotary_dim].to(device=gain.device, dtype=gain.dtype)
    s2 = sin[..., half:rotary_dim].to(device=gain.device, dtype=gain.dtype)
    pair_cross = cross[..., :half].to(device=gain.device, dtype=gain.dtype)
    out[..., :half] = c1.square().mul(first).add_(s2.square().mul(second)).sub_(2.0 * c1 * s2 * pair_cross)
    out[..., half:rotary_dim] = s1.square().mul(first).add_(c2.square().mul(second)).add_(2.0 * s1 * c2 * pair_cross)
    return out


def qwen_attention_qkv_superset_output_gains(
    q_output: torch.Tensor,
    k_output: torch.Tensor,
    v_output: torch.Tensor,
    dense_weight: torch.Tensor,
    q_norm: torch.nn.Module | None,
    k_norm: torch.nn.Module | None,
    *,
    num_heads: int,
    num_key_value_heads: int,
    head_size: int,
    attention_mask: torch.Tensor | None,
    scaling: float,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q_output.ndim != 3 or k_output.ndim != 3 or v_output.ndim != 3:
        raise ValueError("q_output, k_output, and v_output must have shape [batch, sequence, width]")
    batch, seq_len, q_width = q_output.shape
    if k_output.shape[:2] != (batch, seq_len) or v_output.shape[:2] != (batch, seq_len):
        raise ValueError("Q/K/V outputs must share batch and sequence dimensions")
    if q_width != int(num_heads) * int(head_size):
        raise ValueError(f"q_output width {q_width} != {int(num_heads) * int(head_size)}")
    if k_output.shape[-1] != int(num_key_value_heads) * int(head_size):
        raise ValueError(f"k_output width {k_output.shape[-1]} != {int(num_key_value_heads) * int(head_size)}")
    if v_output.shape[-1] != int(num_key_value_heads) * int(head_size):
        raise ValueError(f"v_output width {v_output.shape[-1]} != {int(num_key_value_heads) * int(head_size)}")
    hidden = int(num_heads) * int(head_size)
    if dense_weight.ndim != 2 or dense_weight.shape[1] != hidden:
        raise ValueError(f"dense_weight must have shape [out_features, {hidden}]")
    if int(num_heads) % int(num_key_value_heads) != 0:
        raise ValueError("num_heads must be divisible by num_key_value_heads")

    heads = int(num_heads)
    kv_heads = int(num_key_value_heads)
    head_dim = int(head_size)
    groups = heads // kv_heads
    q_raw = q_output.detach().to(dtype=torch.float32).view(batch, seq_len, heads, head_dim).transpose(1, 2)
    k_raw = k_output.detach().to(dtype=torch.float32).view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
    v = v_output.detach().to(dtype=torch.float32).view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
    q_normed = (
        q_norm(q_raw.reshape(-1, head_dim)).reshape(batch, heads, seq_len, head_dim).to(dtype=torch.float32)
        if q_norm is not None
        else q_raw
    )
    k_normed = (
        k_norm(k_raw.reshape(-1, head_dim)).reshape(batch, kv_heads, seq_len, head_dim).to(dtype=torch.float32)
        if k_norm is not None
        else k_raw
    )
    if position_embeddings is not None:
        cos, sin = position_embeddings
        cos = cos.detach().to(device=q_raw.device, dtype=torch.float32)
        sin = sin.detach().to(device=q_raw.device, dtype=torch.float32)
        q = _qwen_apply_rotary_single(q_normed, cos, sin)
        k = _qwen_apply_rotary_single(k_normed, cos, sin)
    else:
        cos = None
        sin = None
        q = q_normed
        k = k_normed
    k_rep = k[:, :, None].expand(batch, kv_heads, groups, seq_len, head_dim).reshape(batch, heads, seq_len, head_dim)
    v_rep = v[:, :, None].expand(batch, kv_heads, groups, seq_len, head_dim).reshape(batch, heads, seq_len, head_dim)

    mask = attention_mask.detach().to(device=q_raw.device, dtype=torch.float32) if attention_mask is not None else None
    dense = dense_weight.detach().to(device=q_raw.device, dtype=torch.float32)
    q_gain = torch.empty_like(q_raw)
    k_gain = torch.empty_like(k_raw)
    v_gain = torch.zeros_like(v)
    scale = float(scaling)
    block = 16
    attn_by_head: list[torch.Tensor] = []
    context_by_head: list[torch.Tensor] = []
    gram_by_head: list[torch.Tensor] = []

    for head_idx in range(heads):
        kv_idx = head_idx // groups
        qh = q[:, head_idx]
        kh = k_rep[:, head_idx]
        vh = v_rep[:, head_idx]
        columns = dense[:, head_idx * head_dim : (head_idx + 1) * head_dim]
        gram = columns.transpose(0, 1).matmul(columns)
        logits = qh.matmul(kh.transpose(1, 2)).mul(scale)
        if mask is not None:
            logits = logits + (mask[:, 0] if mask.ndim == 4 else mask)
        attn = torch.softmax(logits, dim=-1, dtype=torch.float32)
        context = attn.matmul(vh)
        attn_by_head.append(attn)
        context_by_head.append(context)
        gram_by_head.append(gram)
        context_g = context.matmul(gram)
        context_g_context = context_g.mul(context).sum(dim=-1)
        value_g = vh.matmul(gram)
        value_g_value = value_g.mul(vh).sum(dim=-1)
        context_g_value = context_g.matmul(vh.transpose(1, 2))
        residual_metric = context_g_context.unsqueeze(2).add(value_g_value.unsqueeze(1)).sub_(2.0 * context_g_value)
        attn_sq = attn.square()
        if groups == 1:
            v_gain[:, kv_idx].add_(attn_sq.sum(dim=1).unsqueeze(-1).mul(gram.diag().view(1, 1, -1)))
        k_weights = attn_sq.mul(residual_metric).mul(scale * scale)

        if q_norm is not None:
            q_z = q_raw[:, head_idx].mul(_rmsnorm_affine_weight(q_norm, device=q_raw.device, width=head_dim).view(1, 1, -1))
            q_rot_z = _qwen_apply_rotary_single(q_z.unsqueeze(1), cos, sin).squeeze(1) if cos is not None and sin is not None else q_z
            q_dot_z = q_rot_z.matmul(kh.transpose(1, 2))
            q_dot_z = q_dot_z.sub(attn.mul(q_dot_z).sum(dim=-1, keepdim=True))
            q_u_z = torch.einsum("bas,bas,bsc->bac", attn, q_dot_z, vh)
            q_u_z_g = q_u_z.matmul(gram)
            q_z_m_z = q_u_z_g.mul(q_u_z).sum(dim=-1).mul(scale * scale)
        else:
            q_u_z_g = None
            q_z_m_z = None
        q_diag = torch.empty_like(qh)
        q_m_z = torch.empty_like(qh) if q_norm is not None else None
        for start in range(0, head_dim, block):
            end = min(start + block, head_dim)
            indices = torch.arange(start, end, device=q_raw.device)
            q_eff = []
            for idx_value in indices.tolist():
                basis = torch.zeros((batch, seq_len, head_dim), device=q_raw.device, dtype=torch.float32)
                basis[:, :, idx_value] = 1.0
                rot_basis = _qwen_apply_rotary_single(basis.unsqueeze(1), cos, sin).squeeze(1) if cos is not None and sin is not None else basis
                q_eff.append(rot_basis.matmul(kh.transpose(1, 2)))
            q_eff_block = torch.stack(q_eff, dim=-1)
            q_eff_block = q_eff_block.sub(attn.unsqueeze(-1).mul(q_eff_block).sum(dim=2, keepdim=True))
            weighted = attn.unsqueeze(-1).mul(q_eff_block)
            weighted_value = torch.einsum("basd,bsc->badc", weighted, vh)
            weighted_value_g = torch.matmul(weighted_value, gram)
            q_diag[:, :, start:end] = weighted_value_g.mul(weighted_value).sum(dim=-1).mul(scale * scale)
            if q_m_z is not None and q_u_z_g is not None:
                q_m_z[:, :, start:end] = weighted_value.mul(q_u_z_g.unsqueeze(2)).sum(dim=-1).mul(scale * scale)
        q_gain[:, head_idx] = (
            _rmsnorm_metric_diag_from_products(q_raw[:, head_idx], q_norm, q_diag, q_m_z, q_z_m_z)
            if q_norm is not None and q_m_z is not None and q_z_m_z is not None
            else q_diag
        )

        if groups == 1:
            if k_norm is not None:
                k_z = k_raw[:, kv_idx].mul(_rmsnorm_affine_weight(k_norm, device=k_raw.device, width=head_dim).view(1, 1, -1))
                k_rot_z = _qwen_apply_rotary_single(k_z.unsqueeze(1), cos, sin).squeeze(1) if cos is not None and sin is not None else k_z
                k_dot_z = qh.matmul(k_rot_z.transpose(1, 2))
                k_z_m_z = k_weights.mul(k_dot_z.square()).sum(dim=1)
            else:
                k_dot_z = None
                k_z_m_z = None
            k_diag_head = torch.empty_like(kh)
            k_m_z_head = torch.empty_like(kh) if k_norm is not None else None
            for start in range(0, head_dim, block):
                end = min(start + block, head_dim)
                indices = torch.arange(start, end, device=k_raw.device)
                k_eff = []
                for idx_value in indices.tolist():
                    basis = torch.zeros((batch, seq_len, head_dim), device=k_raw.device, dtype=torch.float32)
                    basis[:, :, idx_value] = 1.0
                    rot_basis = _qwen_apply_rotary_single(basis.unsqueeze(1), cos, sin).squeeze(1) if cos is not None and sin is not None else basis
                    k_eff.append(qh.matmul(rot_basis.transpose(1, 2)))
                k_eff_block = torch.stack(k_eff, dim=-1)
                k_diag_head[:, :, start:end] = k_weights.unsqueeze(-1).mul(k_eff_block.square()).sum(dim=1)
                if k_m_z_head is not None and k_dot_z is not None:
                    k_m_z_head[:, :, start:end] = k_weights.unsqueeze(-1).mul(k_eff_block).mul(k_dot_z.unsqueeze(-1)).sum(dim=1)
            k_gain[:, kv_idx].add_(
                _rmsnorm_metric_diag_from_products(k_raw[:, kv_idx], k_norm, k_diag_head, k_m_z_head, k_z_m_z)
                if k_norm is not None and k_m_z_head is not None and k_z_m_z is not None
                else k_diag_head
            )

    if groups > 1:
        attn_all = torch.stack(attn_by_head, dim=1)
        context_all = torch.stack(context_by_head, dim=1)
        for kv_idx in range(kv_heads):
            head_start = kv_idx * groups
            head_end = head_start + groups
            attn_group = attn_all[:, head_start:head_end]
            context_group = context_all[:, head_start:head_end]
            columns = dense[:, head_start * head_dim : head_end * head_dim].reshape(dense.shape[0], groups, head_dim)
            gram_group = torch.einsum("ogd,ohe->gdhe", columns, columns)
            residual = v[:, kv_idx].unsqueeze(1).unsqueeze(2) - context_group.unsqueeze(3)
            residual_metric_cross = torch.einsum("bgasd,gdhe,bhase->basgh", residual, gram_group, residual)
            for dim_idx in range(head_dim):
                gram_same = gram_group[:, dim_idx, :, dim_idx]
                v_gain[:, kv_idx, :, dim_idx] = torch.einsum("bgas,bhas,gh->bs", attn_group, attn_group, gram_same)

            if k_norm is not None:
                k_z = k_raw[:, kv_idx].mul(_rmsnorm_affine_weight(k_norm, device=k_raw.device, width=head_dim).view(1, 1, -1))
                k_rot_z = _qwen_apply_rotary_single(k_z.unsqueeze(1), cos, sin).squeeze(1) if cos is not None and sin is not None else k_z
                k_dot_z = torch.stack(
                    [q[:, head_start + group_idx].matmul(k_rot_z.transpose(1, 2)) for group_idx in range(groups)],
                    dim=1,
                )
                coeff_z = attn_group.mul(k_dot_z).mul(scale)
                k_z_m_z = torch.einsum("bgas,bhas,basgh->bs", coeff_z, coeff_z, residual_metric_cross)
            else:
                coeff_z = None
                k_z_m_z = None
            k_diag_total = torch.empty((batch, seq_len, head_dim), device=k_raw.device, dtype=torch.float32)
            k_m_z_total = torch.empty_like(k_diag_total) if k_norm is not None else None
            for start in range(0, head_dim, block):
                end = min(start + block, head_dim)
                coeff_parts = []
                for idx_value in range(start, end):
                    basis = torch.zeros((batch, seq_len, head_dim), device=k_raw.device, dtype=torch.float32)
                    basis[:, :, idx_value] = 1.0
                    rot_basis = _qwen_apply_rotary_single(basis.unsqueeze(1), cos, sin).squeeze(1) if cos is not None and sin is not None else basis
                    coeff_parts.append(
                        torch.stack(
                            [q[:, head_start + group_idx].matmul(rot_basis.transpose(1, 2)) for group_idx in range(groups)],
                            dim=1,
                        )
                    )
                coeff = attn_group.unsqueeze(-1).mul(torch.stack(coeff_parts, dim=-1)).mul(scale)
                k_diag_total[:, :, start:end] = torch.einsum("bgasj,bhasj,basgh->bsj", coeff, coeff, residual_metric_cross)
                if k_m_z_total is not None and coeff_z is not None:
                    k_m_z_total[:, :, start:end] = torch.einsum("bgasj,bhas,basgh->bsj", coeff, coeff_z, residual_metric_cross)
            k_gain[:, kv_idx] = (
                _rmsnorm_metric_diag_from_products(k_raw[:, kv_idx], k_norm, k_diag_total, k_m_z_total, k_z_m_z)
                if k_norm is not None and k_m_z_total is not None and k_z_m_z is not None
                else k_diag_total
            )

    return (
        q_gain.transpose(1, 2).reshape(batch, seq_len, heads * head_dim).clamp_min_(0.0),
        k_gain.transpose(1, 2).reshape(batch, seq_len, kv_heads * head_dim).clamp_min_(0.0),
        v_gain.transpose(1, 2).reshape(batch, seq_len, kv_heads * head_dim).clamp_min_(0.0),
    )


def attention_qkv_superset_output_gains(
    qkv_output: torch.Tensor,
    dense_weight: torch.Tensor,
    *,
    num_heads: int,
    head_size: int,
    attention_mask: torch.Tensor | None,
    scaling: float,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if qkv_output.ndim != 3:
        raise ValueError("qkv_output must have shape [batch, sequence, hidden * 3]")
    batch, seq_len, width = qkv_output.shape
    expected_width = int(num_heads) * 3 * int(head_size)
    if width != expected_width:
        raise ValueError(f"qkv_output width {width} != expected {expected_width}")
    hidden = int(num_heads) * int(head_size)
    if dense_weight.ndim != 2 or dense_weight.shape[1] != hidden:
        raise ValueError(f"dense_weight must have shape [out_features, {hidden}]")

    qkv = qkv_output.detach().to(dtype=torch.float32).view(batch, seq_len, int(num_heads), 3 * int(head_size)).transpose(1, 2)
    query, key, value = qkv.chunk(3, dim=-1)
    cos: torch.Tensor | None = None
    sin: torch.Tensor | None = None
    if position_embeddings is not None:
        cos, sin = position_embeddings
        cos = cos.detach().to(device=query.device, dtype=torch.float32)
        sin = sin.detach().to(device=query.device, dtype=torch.float32)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

    mask = attention_mask.detach().to(device=query.device, dtype=torch.float32) if attention_mask is not None else None
    dense = dense_weight.detach().to(device=query.device, dtype=torch.float32)
    gains_by_head: list[torch.Tensor] = []
    scale = float(scaling)
    for head_idx in range(int(num_heads)):
        q = query[:, head_idx].to(dtype=torch.float32)
        k = key[:, head_idx].to(dtype=torch.float32)
        v = value[:, head_idx].to(dtype=torch.float32)
        columns = dense[:, head_idx * int(head_size) : (head_idx + 1) * int(head_size)]
        gram = columns.transpose(0, 1).matmul(columns)

        logits = q.matmul(k.transpose(1, 2)).mul(scale)
        if mask is not None:
            logits = logits + mask[:, 0]
        attn = torch.softmax(logits, dim=-1, dtype=torch.float32)
        context = attn.matmul(v)
        context_g = context.matmul(gram)
        context_g_context = context_g.mul(context).sum(dim=-1)

        value_g = v.matmul(gram)
        value_g_value = value_g.mul(v).sum(dim=-1)
        context_g_value = context_g.matmul(v.transpose(1, 2))
        residual_metric = context_g_context.unsqueeze(2).add(value_g_value.unsqueeze(1)).sub_(2.0 * context_g_value)

        attn_sq = attn.square()
        v_gain = attn_sq.sum(dim=1).unsqueeze(-1).mul(gram.diag().view(1, 1, -1))
        k_weights = attn_sq.mul(residual_metric).mul(scale * scale)
        rotary_dim = int(cos.shape[-1]) if cos is not None else 0
        q_gain = torch.empty_like(q)
        k_gain = torch.empty_like(k)
        block = 16 if rotary_dim > 1 else 32
        if cos is not None and sin is not None and rotary_dim > 1:
            half = rotary_dim // 2
            for start in range(0, int(head_size), block):
                end = min(start + block, int(head_size))
                indices = torch.arange(start, end, device=q.device)
                q_eff_parts: list[torch.Tensor] = []
                k_eff_parts: list[torch.Tensor] = []
                for idx_value in indices.tolist():
                    if idx_value < half:
                        q_eff = cos[:, :, idx_value].unsqueeze(2) * k[:, :, idx_value].unsqueeze(1)
                        q_eff = q_eff + sin[:, :, idx_value + half].unsqueeze(2) * k[:, :, idx_value + half].unsqueeze(1)
                        k_eff = cos[:, :, idx_value].unsqueeze(1) * q[:, :, idx_value].unsqueeze(2)
                        k_eff = k_eff + sin[:, :, idx_value + half].unsqueeze(1) * q[:, :, idx_value + half].unsqueeze(2)
                    elif idx_value < rotary_dim:
                        pair_idx = idx_value - half
                        q_eff = cos[:, :, idx_value].unsqueeze(2) * k[:, :, idx_value].unsqueeze(1)
                        q_eff = q_eff - sin[:, :, pair_idx].unsqueeze(2) * k[:, :, pair_idx].unsqueeze(1)
                        k_eff = cos[:, :, idx_value].unsqueeze(1) * q[:, :, idx_value].unsqueeze(2)
                        k_eff = k_eff - sin[:, :, pair_idx].unsqueeze(1) * q[:, :, pair_idx].unsqueeze(2)
                    else:
                        q_eff = k[:, :, idx_value].unsqueeze(1).expand(-1, seq_len, -1)
                        k_eff = q[:, :, idx_value].unsqueeze(2).expand(-1, -1, seq_len)
                    q_eff_parts.append(q_eff)
                    k_eff_parts.append(k_eff)
                q_eff_block = torch.stack(q_eff_parts, dim=-1)
                k_eff_block = torch.stack(k_eff_parts, dim=-1)
                weighted = attn.unsqueeze(-1).mul(q_eff_block)
                weighted_value = torch.einsum("basd,bsc->badc", weighted, v)
                weighted_value_g = torch.matmul(weighted_value, gram)
                u_g_u = weighted_value_g.mul(weighted_value).sum(dim=-1)
                u_g_context = weighted_value.mul(context_g.unsqueeze(2)).sum(dim=-1)
                mean_block = weighted.sum(dim=2)
                q_gain[:, :, start:end] = (
                    u_g_u.sub(2.0 * mean_block * u_g_context).add_(mean_block.square() * context_g_context.unsqueeze(-1)).mul_(scale * scale)
                )
                k_gain[:, :, start:end] = k_weights.unsqueeze(-1).mul(k_eff_block.square()).sum(dim=1)
        else:
            k_gain = torch.einsum("bas,bad->bsd", k_weights, q.square())
            mean_k = attn.matmul(k)
            context_g_context_expanded = context_g_context.unsqueeze(-1)
            for start in range(0, int(head_size), block):
                end = min(start + block, int(head_size))
                weighted_value = torch.einsum("bas,bsd,bsc->badc", attn, k[:, :, start:end], v)
                weighted_value_g = torch.matmul(weighted_value, gram)
                u_g_u = weighted_value_g.mul(weighted_value).sum(dim=-1)
                u_g_context = weighted_value.mul(context_g.unsqueeze(2)).sum(dim=-1)
                mean_block = mean_k[:, :, start:end]
                q_gain[:, :, start:end] = (
                    u_g_u.sub(2.0 * mean_block * u_g_context).add_(mean_block.square() * context_g_context_expanded).mul_(scale * scale)
                )

        gains_by_head.append(torch.cat([q_gain, k_gain, v_gain], dim=-1))

    return torch.stack(gains_by_head, dim=2).reshape(batch, seq_len, expected_width).clamp_min_(0.0)


def activation_forward_diff_scores(
    weight: torch.Tensor,
    inputs: torch.Tensor,
    preactivation: torch.Tensor,
    activation: torch.nn.Module,
    *,
    eps: float = 1e-3,
    row_chunk: int = 64,
    col_chunk: int = 256,
) -> torch.Tensor:
    del eps
    weight_f = weight.detach().to(dtype=torch.float32)
    x = inputs.detach().reshape(-1, inputs.shape[-1]).to(dtype=torch.float32)
    y = preactivation.detach().reshape(-1, preactivation.shape[-1]).to(dtype=torch.float32)
    if x.shape[0] != y.shape[0]:
        raise ValueError("inputs and preactivation must have the same flattened token count")
    if tuple(weight_f.shape) != (y.shape[1], x.shape[1]):
        raise ValueError(f"weight shape {tuple(weight_f.shape)} is incompatible with inputs {tuple(x.shape)} and preactivation {tuple(y.shape)}")
    baseline = activation(y)
    scores = torch.empty_like(weight_f)
    for row_start in range(0, weight_f.shape[0], int(row_chunk)):
        row_end = min(row_start + int(row_chunk), weight_f.shape[0])
        y_rows = y[:, row_start:row_end].transpose(0, 1).contiguous()
        baseline_rows = baseline[:, row_start:row_end].transpose(0, 1).contiguous()
        for col_start in range(0, weight_f.shape[1], int(col_chunk)):
            col_end = min(col_start + int(col_chunk), weight_f.shape[1])
            delta = weight_f[row_start:row_end, col_start:col_end].unsqueeze(-1).mul(
                x[:, col_start:col_end].transpose(0, 1).unsqueeze(0)
            )
            perturbed = y_rows.unsqueeze(1).sub(delta)
            diff = activation(perturbed).sub(baseline_rows.unsqueeze(1))
            scores[row_start:row_end, col_start:col_end] = diff.square().sum(dim=-1)
    return scores


def graph_propagated_scores(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    propagated_input_sums: dict[str, torch.Tensor],
    fallback_input_sumsq: dict[str, torch.Tensor],
    *,
    matrices_only: bool = True,
) -> dict[str, torch.Tensor]:
    scores: dict[str, torch.Tensor] = {}
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if matrices_only and param.ndim != 2:
            continue
        weight_sq = param.detach().to(device="cpu", dtype=torch.float32).square()
        propagated = propagated_input_sums.get(name)
        if propagated is not None:
            propagated = propagated.to(device="cpu", dtype=torch.float32)
            if tuple(propagated.shape) != tuple(weight_sq.shape):
                raise ValueError(f"graph-propagated stat shape mismatch for {name}: {tuple(propagated.shape)} != {tuple(weight_sq.shape)}")
            scores[name] = weight_sq.mul(propagated)
            continue
        sumsq = fallback_input_sumsq.get(name)
        if sumsq is not None:
            sumsq = sumsq.to(device="cpu", dtype=torch.float32)
            if sumsq.numel() != weight_sq.shape[1]:
                raise ValueError(f"graph fallback input dimension mismatch for {name}: {sumsq.numel()} != {weight_sq.shape[1]}")
            scores[name] = weight_sq.mul(sumsq.unsqueeze(0))
        else:
            scores[name] = weight_sq
    return scores


def accumulate_vjp_parameter_scores_(
    scores: dict[str, torch.Tensor],
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    only_names: set[str] | None = None,
) -> None:
    for name, param in named_parameters:
        if only_names is not None and name not in only_names:
            continue
        if not param.requires_grad or param.ndim != 2 or param.grad is None:
            continue
        contribution = param.detach().to(dtype=torch.float32).mul(param.grad.detach().to(dtype=torch.float32)).square()
        contribution_cpu = contribution.to(device="cpu")
        existing = scores.get(name)
        if existing is None:
            scores[name] = contribution_cpu
        else:
            existing.add_(contribution_cpu)


def accumulate_vjp_gradient_scores_(
    scores: dict[str, torch.Tensor],
    normalizers: dict[str, int],
    named_parameters_and_grads: Iterable[tuple[str, torch.nn.Parameter, torch.Tensor | None]],
    *,
    count: int,
) -> None:
    for name, param, grad in named_parameters_and_grads:
        if grad is None or not param.requires_grad or param.ndim != 2:
            continue
        contribution = param.detach().to(dtype=torch.float32).mul(grad.detach().to(dtype=torch.float32)).square()
        scores[name].add_(contribution.to(device="cpu"))
        normalizers[name] = normalizers.get(name, 0) + int(count)


def build_local_subgraph_endpoint_groups(
    *,
    num_layers: int,
    parameter_names: set[str],
) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for layer_idx in range(num_layers):
        previous_targets = (
            ["gpt_neox.embed_in.weight"]
            if layer_idx == 0
            else [
                f"gpt_neox.layers.{layer_idx - 1}.attention.dense.weight",
                f"gpt_neox.layers.{layer_idx - 1}.mlp.dense_4h_to_h.weight",
            ]
        )
        groups.append(
            {
                "endpoint": f"attn_context_{layer_idx}",
                "targets": [
                    name
                    for name in [
                        f"gpt_neox.layers.{layer_idx}.attention.query_key_value.weight",
                        *previous_targets,
                    ]
                    if name in parameter_names
                ],
            }
        )
        groups.append(
            {
                "endpoint": f"mlp_activation_{layer_idx}",
                "targets": [
                    name
                    for name in [
                        f"gpt_neox.layers.{layer_idx}.mlp.dense_h_to_4h.weight",
                        *previous_targets,
                    ]
                    if name in parameter_names
                ],
            }
        )

    if num_layers > 0:
        groups.append(
            {
                "endpoint": "final_norm",
                "targets": [
                    name
                    for name in [
                        f"gpt_neox.layers.{num_layers - 1}.attention.dense.weight",
                        f"gpt_neox.layers.{num_layers - 1}.mlp.dense_4h_to_h.weight",
                    ]
                    if name in parameter_names
                ],
            }
        )
    if "embed_out.weight" in parameter_names:
        groups.append({"endpoint": "logits", "targets": ["embed_out.weight"]})
    return [group for group in groups if group["targets"]]


def matrix_weight_count(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad and param.ndim == 2))


def resolve_prune_chunk_fraction(
    *,
    matrix_weights: int,
    prune_chunk_fraction: float,
    recompute_every_weights: int,
) -> float:
    if recompute_every_weights > 0:
        return min(float(recompute_every_weights) / max(float(matrix_weights), 1.0), 1.0)
    return prune_chunk_fraction


def normalize_pruning_structure(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"", "unstructured", "per-matrix", "per-matrix-iterative"}:
        return "unstructured"
    if normalized in {"2:4", "2to4", "2-4", "nm", "n:m", "semi-structured", "semistructured"}:
        return "nm"
    raise ValueError(f"unknown pruning_structure: {value}")


def apply_incremental_per_matrix_pruning_(
    model: torch.nn.Module,
    saliency_scores: dict[str, torch.Tensor],
    *,
    pruned_masks: dict[str, torch.Tensor],
    target_fraction: float,
    chunk_fraction: float,
    step: int,
) -> dict[str, object]:
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must be between 0 and 1")
    if not 0.0 < chunk_fraction <= 1.0:
        raise ValueError("chunk_fraction must be between 0 and 1")

    rows: list[dict[str, object]] = []
    weights_seen = 0
    weights_zeroed_total = 0
    weights_zeroed_this_step = 0
    missing_saliency: list[str] = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad or param.ndim != 2:
                continue
            weights_seen += param.numel()
            score = saliency_scores.get(name)
            if score is None:
                missing_saliency.append(name)
                continue
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
                        "weights": param.numel(),
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
            weights_zeroed_this_step += int(add_mask_cpu.sum().item())
            rows.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "weights": param.numel(),
                    "zeroed_this_step": int(add_mask_cpu.sum().item()),
                    "zeroed_total": zeroed_total,
                }
            )

    return {
        "step": step,
        "pruning_scope": "per_matrix_iterative",
        "target_fraction": target_fraction,
        "chunk_fraction": chunk_fraction,
        "matrix_tensors_seen": len(rows) + len(missing_saliency),
        "matrix_tensors_pruned": len(rows),
        "weights_seen": weights_seen,
        "weights_zeroed_this_step": weights_zeroed_this_step,
        "weights_zeroed_total": weights_zeroed_total,
        "actual_zero_fraction": weights_zeroed_total / max(weights_seen, 1),
        "target_reached": weights_zeroed_total >= int(weights_seen * target_fraction),
        "missing_saliency": missing_saliency,
        "pruned_tensors": rows,
    }


def apply_incremental_nm_pruning_(
    model: torch.nn.Module,
    saliency_scores: dict[str, torch.Tensor],
    *,
    pruned_masks: dict[str, torch.Tensor],
    n: int,
    m: int,
    target_zeros_per_group: int,
    group_dim: int,
    step: int,
) -> dict[str, object]:
    if group_dim != 1:
        raise ValueError("N:M pruning currently supports group_dim=1 for native row-wise input quartets")
    if not 0 < n < m:
        raise ValueError("N:M pruning requires 0 < n < m")
    if not 0 <= target_zeros_per_group <= n:
        raise ValueError("target_zeros_per_group must be between 0 and n")

    rows: list[dict[str, object]] = []
    weights_seen = 0
    weights_zeroed_total = 0
    weights_zeroed_this_step = 0
    missing_saliency: list[str] = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad or param.ndim != 2:
                continue
            weights_seen += param.numel()
            if param.shape[group_dim] % m != 0:
                raise ValueError(f"{name} dimension {group_dim}={param.shape[group_dim]} must be divisible by {m}")
            score = saliency_scores.get(name)
            if score is None:
                missing_saliency.append(name)
                continue
            if tuple(score.shape) != tuple(param.shape):
                raise ValueError(f"saliency shape mismatch for {name}: {tuple(score.shape)} != {tuple(param.shape)}")

            cumulative = pruned_masks.get(name)
            if cumulative is None:
                cumulative = torch.zeros(tuple(param.shape), dtype=torch.bool, device="cpu")
                pruned_masks[name] = cumulative

            rows_count = param.shape[0]
            groups_count = param.shape[1] // m
            group_mask = cumulative.reshape(rows_count, groups_count, m).to(device=param.device)
            zero_counts = group_mask.sum(dim=-1)
            need_counts = (int(target_zeros_per_group) - zero_counts).clamp_min(0)
            max_needed = int(need_counts.max().item()) if need_counts.numel() else 0
            add_group_mask = torch.zeros_like(group_mask)
            if max_needed > 0:
                group_scores = score.detach().reshape(rows_count, groups_count, m).to(device=param.device, dtype=torch.float32)
                group_scores = group_scores.masked_fill(group_mask, torch.inf)
                for _ in range(max_needed):
                    active = need_counts > 0
                    if not bool(active.any().item()):
                        break
                    indices = group_scores.argmin(dim=-1)
                    selected = torch.nn.functional.one_hot(indices, num_classes=m).to(dtype=torch.bool, device=param.device)
                    selected &= active.unsqueeze(-1)
                    add_group_mask |= selected
                    group_scores = group_scores.masked_fill(selected, torch.inf)
                    need_counts = need_counts - active.to(dtype=need_counts.dtype)

            add_mask = add_group_mask.reshape(param.shape)
            if bool(add_mask.any().item()):
                param.masked_fill_(add_mask, 0)

            add_mask_cpu = add_mask.to(device="cpu")
            cumulative |= add_mask_cpu
            zeroed_total = int(cumulative.sum().item())
            zeroed_now = int(add_mask_cpu.sum().item())
            weights_zeroed_total += zeroed_total
            weights_zeroed_this_step += zeroed_now
            rows.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "weights": param.numel(),
                    "groups": rows_count * groups_count,
                    "zeroed_this_step": zeroed_now,
                    "zeroed_total": zeroed_total,
                }
            )

    target_total = int(weights_seen * (n / m) * (target_zeros_per_group / max(n, 1)))
    final_target_total = int(weights_seen * (n / m))
    return {
        "step": step,
        "pruning_scope": f"{n}:{m}_semi_structured",
        "target_zeros_per_group": int(target_zeros_per_group),
        "n": int(n),
        "m": int(m),
        "group_dim": int(group_dim),
        "matrix_tensors_seen": len(rows) + len(missing_saliency),
        "matrix_tensors_pruned": len(rows),
        "weights_seen": weights_seen,
        "weights_zeroed_this_step": weights_zeroed_this_step,
        "weights_zeroed_total": weights_zeroed_total,
        "actual_zero_fraction": weights_zeroed_total / max(weights_seen, 1),
        "target_reached": weights_zeroed_total >= target_total and target_zeros_per_group == n,
        "final_target_zero_fraction": final_target_total / max(weights_seen, 1),
        "missing_saliency": missing_saliency,
        "pruned_tensors": rows,
    }


def apply_incremental_nm_pruning_to_parameter_(
    model: torch.nn.Module,
    saliency_scores: dict[str, torch.Tensor],
    *,
    pruned_masks: dict[str, torch.Tensor],
    parameter_name: str,
    n: int,
    m: int,
    target_zeros_per_group: int,
    group_dim: int,
    step: int,
) -> dict[str, object]:
    if group_dim != 1:
        raise ValueError("N:M pruning currently supports group_dim=1 for native row-wise input quartets")
    if not 0 < n < m:
        raise ValueError("N:M pruning requires 0 < n < m")
    if not 0 <= target_zeros_per_group <= n:
        raise ValueError("target_zeros_per_group must be between 0 and n")

    params = dict(model.named_parameters())
    param = params.get(parameter_name)
    if param is None:
        raise ValueError(f"parameter not found: {parameter_name}")
    if not param.requires_grad or param.ndim != 2:
        raise ValueError(f"parameter is not a trainable 2D matrix: {parameter_name}")
    if param.shape[group_dim] % m != 0:
        raise ValueError(f"{parameter_name} dimension {group_dim}={param.shape[group_dim]} must be divisible by {m}")

    score = saliency_scores.get(parameter_name)
    if score is None:
        raise ValueError(f"missing saliency for {parameter_name}")
    if tuple(score.shape) != tuple(param.shape):
        raise ValueError(f"saliency shape mismatch for {parameter_name}: {tuple(score.shape)} != {tuple(param.shape)}")

    cumulative = pruned_masks.get(parameter_name)
    if cumulative is None:
        cumulative = torch.zeros(tuple(param.shape), dtype=torch.bool, device="cpu")
        pruned_masks[parameter_name] = cumulative

    with torch.no_grad():
        rows_count = param.shape[0]
        groups_count = param.shape[1] // m
        group_mask = cumulative.reshape(rows_count, groups_count, m).to(device=param.device)
        zero_counts = group_mask.sum(dim=-1)
        need_counts = (int(target_zeros_per_group) - zero_counts).clamp_min(0)
        max_needed = int(need_counts.max().item()) if need_counts.numel() else 0
        add_group_mask = torch.zeros_like(group_mask)
        if max_needed > 0:
            group_scores = score.detach().reshape(rows_count, groups_count, m).to(device=param.device, dtype=torch.float32)
            group_scores = group_scores.masked_fill(group_mask, torch.inf)
            for _ in range(max_needed):
                active = need_counts > 0
                if not bool(active.any().item()):
                    break
                indices = group_scores.argmin(dim=-1)
                selected = torch.nn.functional.one_hot(indices, num_classes=m).to(dtype=torch.bool, device=param.device)
                selected &= active.unsqueeze(-1)
                add_group_mask |= selected
                group_scores = group_scores.masked_fill(selected, torch.inf)
                need_counts = need_counts - active.to(dtype=need_counts.dtype)

        add_mask = add_group_mask.reshape(param.shape)
        if bool(add_mask.any().item()):
            param.masked_fill_(add_mask, 0)

        add_mask_cpu = add_mask.to(device="cpu")
        cumulative |= add_mask_cpu

    zeroed_total = int(cumulative.sum().item())
    zeroed_now = int(add_mask_cpu.sum().item())
    target_count = int(param.numel() * (n / m) * (target_zeros_per_group / max(n, 1)))
    final_target_count = int(param.numel() * (n / m))
    return {
        "step": step,
        "name": parameter_name,
        "shape": list(param.shape),
        "weights": param.numel(),
        "groups": rows_count * groups_count,
        "pruning_scope": f"{n}:{m}_semi_structured_single_matrix",
        "target_zeros_per_group": int(target_zeros_per_group),
        "n": int(n),
        "m": int(m),
        "group_dim": int(group_dim),
        "zeroed_this_step": zeroed_now,
        "zeroed_total": zeroed_total,
        "actual_zero_fraction": zeroed_total / max(param.numel(), 1),
        "target_reached": zeroed_total >= target_count and target_zeros_per_group == n,
        "final_target_zero_fraction": final_target_count / max(param.numel(), 1),
    }


def reapply_pruned_masks_(model: torch.nn.Module, pruned_masks: dict[str, torch.Tensor]) -> int:
    reapplied = 0
    with torch.no_grad():
        for name, module in linear_modules(model):
            mask = pruned_masks.get(f"{name}.weight")
            if mask is None:
                continue
            module.weight.masked_fill_(mask.to(device=module.weight.device), 0)
            reapplied += int(mask.sum().item())
    return reapplied


def reapply_all_pruned_parameter_masks_(model: torch.nn.Module, pruned_masks: dict[str, torch.Tensor]) -> int:
    reapplied = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            mask = pruned_masks.get(name)
            if mask is None:
                continue
            param.masked_fill_(mask.to(device=param.device), 0)
            reapplied += int(mask.sum().item())
    return reapplied


def apply_masked_gradient_step_(
    model: torch.nn.Module,
    *,
    pruned_masks: dict[str, torch.Tensor],
    learning_rate: float,
    step: int,
) -> dict[str, object]:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    updated_tensors = 0
    updated_weights = 0
    total_abs_step_delta = 0.0
    max_abs_step_delta = 0.0
    squared_grad_norm = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad or param.ndim != 2 or param.grad is None:
                continue
            grad = param.grad.detach().to(device=param.device, dtype=torch.float32)
            mask = pruned_masks.get(name)
            if mask is not None:
                grad = grad.masked_fill(mask.to(device=param.device), 0)
            before = param.detach().to(dtype=torch.float32).clone()
            updated = before - float(learning_rate) * grad
            if mask is not None:
                updated = updated.masked_fill(mask.to(device=param.device), 0)
            param.copy_(updated.to(dtype=param.dtype))
            step_delta = updated - before
            weights = param.numel()
            updated_tensors += 1
            updated_weights += weights
            total_abs_step_delta += float(step_delta.abs().sum().item())
            max_abs_step_delta = max(max_abs_step_delta, float(step_delta.abs().max().item()))
            squared_grad_norm += float(grad.square().sum().item())

    reapplied_zero_weights = reapply_all_pruned_parameter_masks_(model, pruned_masks)
    return {
        "step": step,
        "format": "loss_gradient_descent",
        "learning_rate": float(learning_rate),
        "updated_tensors": updated_tensors,
        "updated_weights": updated_weights,
        "reapplied_zero_weights": reapplied_zero_weights,
        "mean_abs_step_delta": total_abs_step_delta / max(updated_weights, 1),
        "max_abs_step_delta": max_abs_step_delta,
        "grad_l2_norm": squared_grad_norm**0.5,
    }


def _weight_name(module_name: str) -> str:
    return f"{module_name}.weight" if module_name else "weight"


def _nearest_rank_tail_k(quantile: float, count: int) -> int:
    if count <= 0:
        return 0
    rank = max(1, min(count, int(float(quantile) * float(count) + 0.999999)))
    return count - rank + 1


class WandaActivationAccumulator:
    def __init__(self, model: torch.nn.Module, *, target_names: Iterable[str] | None = None):
        self._model = model
        self._target_names = set(target_names) if target_names is not None else None
        self._sumsq: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._attention_mask: torch.Tensor | None = None

    def __enter__(self) -> WandaActivationAccumulator:
        for module_name, module in self._model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight = module.weight
            if weight is None or not weight.requires_grad or weight.ndim != 2:
                continue
            weight_name = _weight_name(module_name)
            if self._target_names is not None and weight_name not in self._target_names:
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(weight_name)))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear_attention_mask()

    def set_attention_mask(self, attention_mask: torch.Tensor | None) -> None:
        self._attention_mask = attention_mask.detach().bool() if torch.is_tensor(attention_mask) else None

    def clear_attention_mask(self) -> None:
        self._attention_mask = None

    def _flatten_activations(self, activations: torch.Tensor) -> torch.Tensor:
        if (
            activations.ndim == 3
            and self._attention_mask is not None
            and tuple(activations.shape[:2]) == tuple(self._attention_mask.shape)
        ):
            return activations.detach()[self._attention_mask].float()
        return activations.detach().reshape(-1, activations.shape[-1]).float()

    def _make_hook(self, weight_name: str):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            del module, output
            if not inputs:
                return
            activations = inputs[0]
            if not torch.is_tensor(activations) or activations.numel() == 0:
                return
            flat = self._flatten_activations(activations)
            if flat.numel() == 0:
                return
            sumsq = flat.square().sum(dim=0).to(device="cpu")
            if weight_name in self._sumsq:
                self._sumsq[weight_name].add_(sumsq)
            else:
                self._sumsq[weight_name] = sumsq
            self._counts[weight_name] = self._counts.get(weight_name, 0) + flat.shape[0]

        return hook

    def finalize(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        fallback_to_magnitude: bool = True,
    ) -> dict[str, torch.Tensor]:
        scores: dict[str, torch.Tensor] = {}
        for name, param in named_parameters:
            if not param.requires_grad or param.ndim != 2:
                continue
            weight_abs = param.detach().abs().to(device="cpu", dtype=torch.float32)
            sumsq = self._sumsq.get(name)
            count = self._counts.get(name, 0)
            if sumsq is not None and count > 0:
                input_rms = torch.sqrt(sumsq.div(float(count))).to(dtype=torch.float32)
                if input_rms.numel() != weight_abs.shape[1]:
                    raise ValueError(f"WANDA input dimension mismatch for {name}: {input_rms.numel()} != {weight_abs.shape[1]}")
                scores[name] = weight_abs.mul(input_rms.unsqueeze(0))
            elif fallback_to_magnitude:
                scores[name] = weight_abs
        return scores

    def hessian_diagonals(self) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        hessians: dict[str, torch.Tensor] = {}
        tokens: dict[str, int] = {}
        for weight_name, sumsq in self._sumsq.items():
            if not weight_name.endswith(".weight"):
                continue
            module_name = weight_name[: -len(".weight")]
            hessians[module_name] = sumsq.to(dtype=torch.float64).cpu()
            tokens[module_name] = self._counts.get(weight_name, 0)
        return hessians, tokens

    def activation_rms(self) -> dict[str, torch.Tensor]:
        return {
            weight_name: torch.sqrt(sumsq.div(float(self._counts[weight_name]))).to(device="cpu", dtype=torch.float32)
            for weight_name, sumsq in self._sumsq.items()
            if self._counts.get(weight_name, 0) > 0
        }


class InputActivationStatsAccumulator:
    def __init__(self, model: torch.nn.Module, *, quantile: float = 0.95, max_rows: int = 0):
        self._model = model
        self._quantile = float(quantile)
        self._max_rows = int(max_rows)
        self._sum: dict[str, torch.Tensor] = {}
        self._sumsq: dict[str, torch.Tensor] = {}
        self._sum_abs: dict[str, torch.Tensor] = {}
        self._max_abs: dict[str, torch.Tensor] = {}
        self._top_abs: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> InputActivationStatsAccumulator:
        for module_name, module in self._model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight = module.weight
            if weight is None or not weight.requires_grad or weight.ndim != 2:
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(_weight_name(module_name))))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, weight_name: str):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            del module, output
            if not inputs:
                return
            activations = inputs[0]
            if not torch.is_tensor(activations) or activations.numel() == 0:
                return
            flat = activations.detach().reshape(-1, activations.shape[-1]).float()
            flat_abs = flat.abs()
            batch_sum = flat.sum(dim=0).to(device="cpu")
            batch_sumsq = flat.square().sum(dim=0).to(device="cpu")
            batch_sum_abs = flat_abs.sum(dim=0).to(device="cpu")
            batch_max_abs = flat_abs.max(dim=0).values.to(device="cpu")
            if weight_name in self._sum:
                self._sum[weight_name].add_(batch_sum)
                self._sumsq[weight_name].add_(batch_sumsq)
                self._sum_abs[weight_name].add_(batch_sum_abs)
                self._max_abs[weight_name] = torch.maximum(self._max_abs[weight_name], batch_max_abs)
            else:
                self._sum[weight_name] = batch_sum
                self._sumsq[weight_name] = batch_sumsq
                self._sum_abs[weight_name] = batch_sum_abs
                self._max_abs[weight_name] = batch_max_abs
            self._counts[weight_name] = self._counts.get(weight_name, 0) + flat.shape[0]

            if self._max_rows > 0:
                tail_k = max(1, _nearest_rank_tail_k(self._quantile, self._max_rows))
                batch_k = min(tail_k, flat_abs.shape[0])
                batch_top = torch.topk(flat_abs, k=batch_k, dim=0, sorted=False).values.to(device="cpu")
                previous = self._top_abs.get(weight_name)
                combined = batch_top if previous is None else torch.cat((previous, batch_top), dim=0)
                keep_k = min(tail_k, combined.shape[0])
                self._top_abs[weight_name] = torch.topk(combined, k=keep_k, dim=0, sorted=False).values

        return hook

    def finalize_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        stats: dict[str, dict[str, torch.Tensor]] = {}
        for weight_name, count in self._counts.items():
            if count <= 0:
                continue
            mean = self._sum[weight_name].div(float(count))
            sumsq = self._sumsq[weight_name].to(dtype=torch.float32)
            variance = sumsq.div(float(count)).sub(mean.square()).clamp_min_(0.0)
            row = {
                "sumsq": sumsq,
                "mean_abs": self._sum_abs[weight_name].div(float(count)).to(dtype=torch.float32),
                "variance": variance.to(dtype=torch.float32),
                "max_abs": self._max_abs[weight_name].to(dtype=torch.float32),
            }
            top_abs = self._top_abs.get(weight_name)
            if top_abs is not None:
                tail_k = max(1, _nearest_rank_tail_k(self._quantile, count))
                tail_k = min(tail_k, top_abs.shape[0])
                key = f"q{int(round(self._quantile * 100))}_abs"
                row[key] = torch.topk(top_abs, k=tail_k, dim=0, sorted=False).values.min(dim=0).values.to(dtype=torch.float32)
            stats[weight_name] = row
        return stats


class AngularActivationAccumulator:
    def __init__(self, model: torch.nn.Module):
        self._model = model
        self._x_sumsq: dict[str, torch.Tensor] = {}
        self._y_sumsq: dict[str, torch.Tensor] = {}
        self._yx_dot: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> AngularActivationAccumulator:
        for module_name, module in self._model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight = module.weight
            if weight is None or not weight.requires_grad or weight.ndim != 2:
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(_weight_name(module_name))))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, weight_name: str):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            del module
            if not inputs or not torch.is_tensor(output):
                return
            activations = inputs[0]
            if not torch.is_tensor(activations) or activations.numel() == 0 or output.numel() == 0:
                return
            flat_x = activations.detach().reshape(-1, activations.shape[-1]).to(dtype=torch.float32)
            flat_y = output.detach().reshape(-1, output.shape[-1]).to(dtype=torch.float32)
            if flat_x.shape[0] != flat_y.shape[0]:
                return
            batch_x_sumsq = flat_x.square().sum(dim=0)
            batch_y_sumsq = flat_y.square().sum(dim=0)
            batch_yx_dot = flat_y.transpose(0, 1).matmul(flat_x)
            if weight_name in self._yx_dot:
                self._x_sumsq[weight_name].add_(batch_x_sumsq)
                self._y_sumsq[weight_name].add_(batch_y_sumsq)
                self._yx_dot[weight_name].add_(batch_yx_dot)
            else:
                self._x_sumsq[weight_name] = batch_x_sumsq
                self._y_sumsq[weight_name] = batch_y_sumsq
                self._yx_dot[weight_name] = batch_yx_dot
            self._counts[weight_name] = self._counts.get(weight_name, 0) + flat_x.shape[0]

        return hook

    def finalize_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            weight_name: {
                "x_sumsq": self._x_sumsq[weight_name],
                "y_sumsq": self._y_sumsq[weight_name],
                "yx_dot": self._yx_dot[weight_name],
            }
            for weight_name, count in self._counts.items()
            if count > 0
        }


class FeatureCosineWandaAccumulator:
    def __init__(self, model: torch.nn.Module):
        self._model = model
        self._weighted_abs_input: dict[str, torch.Tensor] = {}
        self._row_weight_sums: dict[str, torch.Tensor] = {}
        self._x_sumsq: dict[str, torch.Tensor] = {}
        self._y_sumsq: dict[str, torch.Tensor] = {}
        self._yx_dot: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> FeatureCosineWandaAccumulator:
        for module_name, module in self._model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight = module.weight
            if weight is None or not weight.requires_grad or weight.ndim != 2:
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(_weight_name(module_name))))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, weight_name: str):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            del module
            if not inputs or not torch.is_tensor(output):
                return
            activations = inputs[0]
            if not torch.is_tensor(activations) or activations.numel() == 0 or output.numel() == 0:
                return
            flat_x = activations.detach().reshape(-1, activations.shape[-1]).to(dtype=torch.float32)
            flat_y = output.detach().reshape(-1, output.shape[-1]).to(dtype=torch.float32)
            if flat_x.shape[0] != flat_y.shape[0]:
                return

            flat_x_abs = flat_x.abs()
            flat_y_abs = flat_y.abs()
            weighted_abs_input = flat_y_abs.transpose(0, 1).matmul(flat_x_abs)
            row_weight_sums = flat_y_abs.sum(dim=0)
            x_sumsq = flat_x.square().sum(dim=0)
            y_sumsq = flat_y.square().sum(dim=0)
            yx_dot = flat_y.transpose(0, 1).matmul(flat_x)

            if weight_name in self._weighted_abs_input:
                self._weighted_abs_input[weight_name].add_(weighted_abs_input)
                self._row_weight_sums[weight_name].add_(row_weight_sums)
                self._x_sumsq[weight_name].add_(x_sumsq)
                self._y_sumsq[weight_name].add_(y_sumsq)
                self._yx_dot[weight_name].add_(yx_dot)
            else:
                self._weighted_abs_input[weight_name] = weighted_abs_input
                self._row_weight_sums[weight_name] = row_weight_sums
                self._x_sumsq[weight_name] = x_sumsq
                self._y_sumsq[weight_name] = y_sumsq
                self._yx_dot[weight_name] = yx_dot
            self._counts[weight_name] = self._counts.get(weight_name, 0) + flat_x.shape[0]

        return hook

    def finalize_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            weight_name: {
                "weighted_abs_input": self._weighted_abs_input[weight_name],
                "row_weight_sums": self._row_weight_sums[weight_name],
                "x_sumsq": self._x_sumsq[weight_name],
                "y_sumsq": self._y_sumsq[weight_name],
                "yx_dot": self._yx_dot[weight_name],
            }
            for weight_name, count in self._counts.items()
            if count > 0
        }


class GPTNeoXGraphPropagatedAccumulator:
    _METHODS = {
        "graph_norm",
        "subgraph_norm",
        "residual_norm",
        "graph_qkv",
        "subgraph_qkv",
        "residual_norm_qkv",
        "graph_mlp",
        "subgraph_mlp",
        "residual_norm_mlp",
        "graph_qkv_mlp",
        "graph_next_projections",
        "subgraph_qkv_mlp",
    }

    def __init__(self, model: torch.nn.Module, *, method: str, num_probes: int = 4, seed: int = 17):
        self._model = model
        self._method = method.lower()
        if self._method not in self._METHODS:
            raise ValueError(f"unknown graph-propagated method: {method}")
        self._num_probes = int(num_probes)
        self._seed = int(seed)
        self._fallback_input_sumsq: dict[str, torch.Tensor] = {}
        self._propagated_input_sums: dict[str, torch.Tensor] = {}
        self._pending_x2: dict[int, dict[str, torch.Tensor]] = {}
        self._target_to_layer: dict[str, int] = {}
        self._projected_vectors: dict[tuple[int, str], torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        gpt_neox = getattr(model, "gpt_neox", None)
        self._layers = list(getattr(gpt_neox, "layers", [])) if gpt_neox is not None else []
        self._final_layer_norm = getattr(gpt_neox, "final_layer_norm", None) if gpt_neox is not None else None

    def __enter__(self) -> GPTNeoXGraphPropagatedAccumulator:
        for layer_idx, layer in enumerate(self._layers):
            attention = getattr(layer, "attention", None)
            mlp = getattr(layer, "mlp", None)
            if attention is not None and isinstance(getattr(attention, "dense", None), torch.nn.Linear):
                self._target_to_layer[f"gpt_neox.layers.{layer_idx}.attention.dense.weight"] = layer_idx
            if mlp is not None and isinstance(getattr(mlp, "dense_4h_to_h", None), torch.nn.Linear):
                self._target_to_layer[f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h.weight"] = layer_idx
            self._handles.append(layer.register_forward_hook(self._make_layer_hook(layer_idx)))

        if self._method not in {"graph_norm", "subgraph_norm", "residual_norm"}:
            self._prepare_projection_vectors()

        for module_name, module in self._model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight = module.weight
            if weight is None or not weight.requires_grad or weight.ndim != 2:
                continue
            self._handles.append(module.register_forward_hook(self._make_linear_hook(_weight_name(module_name))))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending_x2.clear()

    def _prepare_projection_vectors(self) -> None:
        if self._num_probes <= 0:
            raise ValueError("graph_num_probes must be positive for projection-based graph saliency")
        wants_qkv = self._method in {"graph_qkv", "subgraph_qkv", "residual_norm_qkv", "graph_qkv_mlp", "graph_next_projections", "subgraph_qkv_mlp"}
        wants_mlp = self._method in {"graph_mlp", "subgraph_mlp", "residual_norm_mlp", "graph_qkv_mlp", "graph_next_projections", "subgraph_qkv_mlp"}
        for layer_idx in range(max(len(self._layers) - 1, 0)):
            next_layer = self._layers[layer_idx + 1]
            if wants_qkv:
                qkv = getattr(getattr(next_layer, "attention", None), "query_key_value", None)
                if isinstance(qkv, torch.nn.Linear):
                    self._projected_vectors[(layer_idx, "qkv")] = self._make_projected_vectors(
                        qkv.weight,
                        seed_offset=1009 * (layer_idx + 1) + 17,
                    )
            if wants_mlp:
                up = getattr(getattr(next_layer, "mlp", None), "dense_h_to_4h", None)
                if isinstance(up, torch.nn.Linear):
                    self._projected_vectors[(layer_idx, "mlp")] = self._make_projected_vectors(
                        up.weight,
                        seed_offset=1009 * (layer_idx + 1) + 53,
                    )

    def _make_projected_vectors(self, weight: torch.Tensor, *, seed_offset: int) -> torch.Tensor:
        weight_f = weight.detach().to(dtype=torch.float32)
        generator = torch.Generator(device=weight_f.device)
        generator.manual_seed(self._seed + seed_offset + 104729 * int(weight_f.shape[0]))
        signs = torch.randint(
            0,
            2,
            (self._num_probes, weight_f.shape[0]),
            device=weight_f.device,
            generator=generator,
            dtype=torch.int8,
        )
        random_vectors = signs.to(dtype=torch.float32).mul_(2.0).sub_(1.0)
        return random_vectors.matmul(weight_f)

    def _make_linear_hook(self, weight_name: str):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            del module, output
            if not inputs:
                return
            activations = inputs[0]
            if not torch.is_tensor(activations) or activations.numel() == 0:
                return
            flat = activations.detach().reshape(-1, activations.shape[-1]).to(dtype=torch.float32)
            x2 = flat.square()
            sumsq = x2.sum(dim=0).to(device="cpu")
            if weight_name in self._fallback_input_sumsq:
                self._fallback_input_sumsq[weight_name].add_(sumsq)
            else:
                self._fallback_input_sumsq[weight_name] = sumsq

            layer_idx = self._target_to_layer.get(weight_name)
            if layer_idx is not None:
                self._pending_x2.setdefault(layer_idx, {})[weight_name] = x2

        return hook

    def _make_layer_hook(self, layer_idx: int):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            del module, inputs
            hidden = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(hidden):
                self._flush_layer(layer_idx, hidden)

        return hook

    def _flush_layer(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        pending = self._pending_x2.pop(layer_idx, None)
        if not pending:
            return
        gain = self._downstream_gain(layer_idx, hidden_states)
        if gain is None:
            return
        gain = gain.detach().reshape(-1, gain.shape[-1]).to(dtype=torch.float32)
        for weight_name, x2 in pending.items():
            if x2.shape[0] != gain.shape[0]:
                continue
            stat = gain.transpose(0, 1).matmul(x2).to(device="cpu")
            if weight_name in self._propagated_input_sums:
                self._propagated_input_sums[weight_name].add_(stat)
            else:
                self._propagated_input_sums[weight_name] = stat

    def _downstream_gain(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if layer_idx + 1 >= len(self._layers):
            if self._final_layer_norm is None:
                return None
            return layernorm_input_jacobian_colnorm_squares(hidden_states, self._final_layer_norm)

        next_layer = self._layers[layer_idx + 1]
        if self._method in {"graph_norm", "subgraph_norm", "residual_norm"}:
            gain = layernorm_input_jacobian_colnorm_squares(hidden_states, next_layer.input_layernorm)
            post_norm = getattr(next_layer, "post_attention_layernorm", None)
            if post_norm is not None:
                gain = gain.add(layernorm_input_jacobian_colnorm_squares(hidden_states, post_norm))
            return gain

        gain: torch.Tensor | None = None
        if self._method in {"graph_qkv", "subgraph_qkv", "residual_norm_qkv", "graph_qkv_mlp", "graph_next_projections", "subgraph_qkv_mlp"}:
            vectors = self._projected_vectors.get((layer_idx, "qkv"))
            if vectors is not None:
                gain = layernorm_input_projected_gain_from_vectors(hidden_states, next_layer.input_layernorm, vectors)
        if self._method in {"graph_mlp", "subgraph_mlp", "residual_norm_mlp", "graph_qkv_mlp", "graph_next_projections", "subgraph_qkv_mlp"}:
            vectors = self._projected_vectors.get((layer_idx, "mlp"))
            post_norm = getattr(next_layer, "post_attention_layernorm", None)
            if vectors is not None and post_norm is not None:
                mlp_gain = layernorm_input_projected_gain_from_vectors(hidden_states, post_norm, vectors)
                gain = mlp_gain if gain is None else gain.add(mlp_gain)
        return gain

    def finalize(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
        return graph_propagated_scores(
            named_parameters,
            self._propagated_input_sums,
            self._fallback_input_sumsq,
        )


class LocalSubgraphEndpointCollector:
    def __init__(self, model: torch.nn.Module):
        self._model = model
        self.endpoints: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        gpt_neox = getattr(model, "gpt_neox", None)
        self.layers = list(getattr(gpt_neox, "layers", [])) if gpt_neox is not None else []
        self.final_layer_norm = getattr(gpt_neox, "final_layer_norm", None) if gpt_neox is not None else None

    def __enter__(self) -> LocalSubgraphEndpointCollector:
        for layer_idx, layer in enumerate(self.layers):
            attention_dense = getattr(getattr(layer, "attention", None), "dense", None)
            if isinstance(attention_dense, torch.nn.Linear):
                self._handles.append(attention_dense.register_forward_hook(self._capture_input(f"attn_context_{layer_idx}")))
            mlp_down = getattr(getattr(layer, "mlp", None), "dense_4h_to_h", None)
            if isinstance(mlp_down, torch.nn.Linear):
                self._handles.append(mlp_down.register_forward_hook(self._capture_input(f"mlp_activation_{layer_idx}")))
        if self.final_layer_norm is not None:
            self._handles.append(self.final_layer_norm.register_forward_hook(self._capture_output("final_norm")))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()

    def clear(self) -> None:
        self.endpoints.clear()

    def _capture_input(self, name: str):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            del module, output
            if inputs and torch.is_tensor(inputs[0]):
                self.endpoints[name] = inputs[0]

        return hook

    def _capture_output(self, name: str):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            del module, inputs
            if torch.is_tensor(output):
                self.endpoints[name] = output

        return hook


class LocalForwardWandaAccumulator:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        eps: float = 1e-3,
        use_attention_mask: bool = True,
        target_names: Iterable[str] | None = None,
        closed_form: bool = False,
        superset_gain_power: float = 1.0,
        superset_gain_clip_quantile: float = 0.0,
    ):
        self._model = model
        self._eps = float(eps)
        self._use_attention_mask = bool(use_attention_mask)
        self._target_names = set(target_names) if target_names is not None else None
        self._closed_form = bool(closed_form)
        self._superset_gain_power = float(superset_gain_power)
        self._superset_gain_clip_quantile = float(superset_gain_clip_quantile)
        self._stats: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}
        self._pending_x2: dict[int, dict[str, torch.Tensor]] = {}
        self._pending_qkv: dict[int, tuple[str, torch.Tensor, torch.Tensor]] = {}
        self._pending_qwen_qkv: dict[int, dict[str, tuple[str, torch.Tensor, torch.Tensor]]] = {}
        self._pending_qwen_mlp: dict[int, dict[str, tuple[str, torch.Tensor, torch.Tensor]]] = {}
        self._pending_qwen_x2: dict[int, dict[str, torch.Tensor]] = {}
        self._pending_opt_qkv: dict[int, dict[str, tuple[str, torch.Tensor, torch.Tensor]]] = {}
        self._pending_opt_x2: dict[int, dict[str, torch.Tensor]] = {}
        self._qkv_to_layer: dict[str, int] = {}
        self._qwen_qkv_to_layer: dict[str, tuple[int, str]] = {}
        self._qwen_mlp_to_layer: dict[str, tuple[int, str]] = {}
        self._opt_qkv_to_layer: dict[str, tuple[int, str]] = {}
        self._residual_output_to_layer: dict[str, int] = {}
        self._qwen_residual_output_to_layer: dict[str, int] = {}
        self._opt_residual_output_to_layer: dict[str, int] = {}
        self._mlp_up_activations: dict[str, torch.nn.Module] = {}
        self._mlp_up_downstream_colnorms: dict[str, torch.Tensor] = {}
        self._linear_hook_dependencies: set[str] = set()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._current_input_ids: torch.Tensor | None = None
        self._current_attention_mask: torch.Tensor | None = None
        gpt_neox = getattr(model, "gpt_neox", None)
        self._layers = list(getattr(gpt_neox, "layers", [])) if gpt_neox is not None else []
        self._final_layer_norm = getattr(gpt_neox, "final_layer_norm", None) if gpt_neox is not None else None
        qwen_model = getattr(model, "model", None)
        self._qwen_layers = list(getattr(qwen_model, "layers", [])) if qwen_model is not None else []
        self._qwen_final_norm = getattr(qwen_model, "norm", None) if qwen_model is not None else None
        opt_decoder = getattr(qwen_model, "decoder", None) if qwen_model is not None else None
        self._opt_layers = list(getattr(opt_decoder, "layers", [])) if opt_decoder is not None else []
        self._opt_final_norm = getattr(opt_decoder, "final_layer_norm", None) if opt_decoder is not None else None
        self._opt_project_out = getattr(opt_decoder, "project_out", None) if opt_decoder is not None else None
        self._opt_embed_tokens = getattr(opt_decoder, "embed_tokens", None) if opt_decoder is not None else None
        self._opt_embed_positions = getattr(opt_decoder, "embed_positions", None) if opt_decoder is not None else None

    def _wants(self, weight_name: str) -> bool:
        return self._target_names is None or weight_name in self._target_names

    def __enter__(self) -> LocalForwardWandaAccumulator:
        for layer_idx, layer in enumerate(self._layers):
            attention = getattr(layer, "attention", None)
            mlp = getattr(layer, "mlp", None)
            qkv_name = f"gpt_neox.layers.{layer_idx}.attention.query_key_value.weight"
            if attention is not None and isinstance(getattr(attention, "query_key_value", None), torch.nn.Linear) and self._wants(qkv_name):
                self._qkv_to_layer[qkv_name] = layer_idx
                self._handles.append(attention.register_forward_hook(self._make_attention_hook(layer_idx), with_kwargs=True))
            attention_dense_name = f"gpt_neox.layers.{layer_idx}.attention.dense.weight"
            if attention is not None and isinstance(getattr(attention, "dense", None), torch.nn.Linear) and self._wants(attention_dense_name):
                self._residual_output_to_layer[attention_dense_name] = layer_idx
            if mlp is not None:
                up_name = f"gpt_neox.layers.{layer_idx}.mlp.dense_h_to_4h.weight"
                down_name = f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h.weight"
                if isinstance(getattr(mlp, "dense_h_to_4h", None), torch.nn.Linear) and self._wants(up_name):
                    self._mlp_up_activations[up_name] = mlp.act
                    down = getattr(mlp, "dense_4h_to_h", None)
                    if isinstance(down, torch.nn.Linear):
                        self._mlp_up_downstream_colnorms[up_name] = down.weight.detach().to(dtype=torch.float32).square().sum(dim=0)
                if isinstance(getattr(mlp, "dense_4h_to_h", None), torch.nn.Linear) and self._wants(down_name):
                    self._residual_output_to_layer[down_name] = layer_idx
            self._handles.append(layer.register_forward_hook(self._make_layer_hook(layer_idx)))

        for layer_idx, layer in enumerate(self._qwen_layers):
            attention = getattr(layer, "self_attn", None)
            mlp = getattr(layer, "mlp", None)
            qkv_names = {
                "q": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
                "k": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
                "v": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            }
            if attention is not None and any(self._wants(name) for name in qkv_names.values()):
                if all(isinstance(getattr(attention, attr, None), torch.nn.Linear) for attr in ("q_proj", "k_proj", "v_proj", "o_proj")):
                    self._handles.append(attention.register_forward_hook(self._make_qwen_attention_hook(layer_idx), with_kwargs=True))
                    for kind, name in qkv_names.items():
                        self._qwen_qkv_to_layer[name] = (layer_idx, kind)
                        self._linear_hook_dependencies.add(name)
            o_name = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            if attention is not None and isinstance(getattr(attention, "o_proj", None), torch.nn.Linear) and self._wants(o_name):
                self._qwen_residual_output_to_layer[o_name] = layer_idx
            if mlp is not None:
                gate_name = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
                up_name = f"model.layers.{layer_idx}.mlp.up_proj.weight"
                down_name = f"model.layers.{layer_idx}.mlp.down_proj.weight"
                if any(self._wants(name) for name in (gate_name, up_name)):
                    if all(isinstance(getattr(mlp, attr, None), torch.nn.Linear) for attr in ("gate_proj", "up_proj", "down_proj")):
                        self._handles.append(mlp.register_forward_hook(self._make_qwen_mlp_hook(layer_idx)))
                        self._qwen_mlp_to_layer[gate_name] = (layer_idx, "gate")
                        self._qwen_mlp_to_layer[up_name] = (layer_idx, "up")
                        self._linear_hook_dependencies.update({gate_name, up_name})
                if isinstance(getattr(mlp, "down_proj", None), torch.nn.Linear) and self._wants(down_name):
                    self._qwen_residual_output_to_layer[down_name] = layer_idx
            self._handles.append(layer.register_forward_hook(self._make_qwen_layer_hook(layer_idx)))

        for layer_idx, layer in enumerate(self._opt_layers):
            attention = getattr(layer, "self_attn", None)
            qkv_names = {
                "q": f"model.decoder.layers.{layer_idx}.self_attn.q_proj.weight",
                "k": f"model.decoder.layers.{layer_idx}.self_attn.k_proj.weight",
                "v": f"model.decoder.layers.{layer_idx}.self_attn.v_proj.weight",
            }
            if attention is not None and any(self._wants(name) for name in qkv_names.values()):
                if all(isinstance(getattr(attention, attr, None), torch.nn.Linear) for attr in ("q_proj", "k_proj", "v_proj", "out_proj")):
                    self._handles.append(attention.register_forward_hook(self._make_opt_attention_hook(layer_idx), with_kwargs=True))
                    for kind, name in qkv_names.items():
                        self._opt_qkv_to_layer[name] = (layer_idx, kind)
                        self._linear_hook_dependencies.add(name)
            out_name = f"model.decoder.layers.{layer_idx}.self_attn.out_proj.weight"
            if attention is not None and isinstance(getattr(attention, "out_proj", None), torch.nn.Linear) and self._wants(out_name):
                self._opt_residual_output_to_layer[out_name] = layer_idx
            fc1_name = f"model.decoder.layers.{layer_idx}.fc1.weight"
            fc2_name = f"model.decoder.layers.{layer_idx}.fc2.weight"
            fc1 = getattr(layer, "fc1", None)
            fc2 = getattr(layer, "fc2", None)
            if isinstance(fc1, torch.nn.Linear) and self._wants(fc1_name):
                activation = getattr(layer, "activation_fn", None)
                if callable(activation):
                    self._mlp_up_activations[fc1_name] = activation
                    if isinstance(fc2, torch.nn.Linear):
                        self._mlp_up_downstream_colnorms[fc1_name] = fc2.weight.detach().to(dtype=torch.float32).square().sum(dim=0)
            if isinstance(fc2, torch.nn.Linear) and self._wants(fc2_name):
                self._opt_residual_output_to_layer[fc2_name] = layer_idx
            self._handles.append(layer.register_forward_hook(self._make_opt_layer_hook(layer_idx)))

        for module_name, module in self._model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight = module.weight
            if weight is None or not weight.requires_grad or weight.ndim != 2:
                continue
            weight_name = _weight_name(module_name)
            if self._wants(weight_name) or weight_name in self._linear_hook_dependencies:
                self._handles.append(module.register_forward_hook(self._make_linear_hook(weight_name)))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending_x2.clear()
        self._pending_qkv.clear()
        self._pending_qwen_qkv.clear()
        self._pending_qwen_mlp.clear()
        self._pending_qwen_x2.clear()
        self._pending_opt_qkv.clear()
        self._pending_opt_x2.clear()
        self.clear_batch()

    def set_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self._current_input_ids = input_ids.detach()
        self._current_attention_mask = attention_mask.detach().bool()

    def clear_batch(self) -> None:
        self._current_input_ids = None
        self._current_attention_mask = None

    def _selected_flat(self, tensor: torch.Tensor) -> torch.Tensor:
        if (
            self._use_attention_mask
            and tensor.ndim == 3
            and self._current_attention_mask is not None
            and tuple(tensor.shape[:2]) == tuple(self._current_attention_mask.shape)
        ):
            return tensor.detach()[self._current_attention_mask].to(dtype=torch.float32)
        return tensor.detach().reshape(-1, tensor.shape[-1]).to(dtype=torch.float32)

    def _selected_token_ids(self) -> torch.Tensor | None:
        if self._current_input_ids is None:
            return None
        if (
            self._use_attention_mask
            and self._current_attention_mask is not None
            and tuple(self._current_input_ids.shape) == tuple(self._current_attention_mask.shape)
        ):
            return self._current_input_ids[self._current_attention_mask].detach()
        return self._current_input_ids.reshape(-1).detach()

    def _add_stat(self, weight_name: str, stat: torch.Tensor, *, count: int) -> None:
        stat_cpu = stat.detach().to(device="cpu", dtype=torch.float32)
        existing = self._stats.get(weight_name)
        if existing is None:
            self._stats[weight_name] = stat_cpu
        else:
            existing.add_(stat_cpu)
        self._counts[weight_name] = self._counts.get(weight_name, 0) + int(count)

    def _transform_gain(self, gain: torch.Tensor) -> torch.Tensor:
        return transform_superset_gain(
            gain,
            power=self._superset_gain_power,
            clip_quantile=self._superset_gain_clip_quantile,
        )

    def _make_linear_hook(self, weight_name: str):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                return
            flat_x = self._selected_flat(inputs[0])
            if flat_x.numel() == 0:
                return
            x2 = flat_x.square()

            qkv_layer_idx = self._qkv_to_layer.get(weight_name)
            if qkv_layer_idx is not None:
                self._pending_qkv[qkv_layer_idx] = (weight_name, x2, output.detach())
                return

            qwen_qkv = self._qwen_qkv_to_layer.get(weight_name)
            if qwen_qkv is not None:
                layer_idx, kind = qwen_qkv
                self._pending_qwen_qkv.setdefault(layer_idx, {})[kind] = (weight_name, x2, output.detach())
                return

            qwen_mlp = self._qwen_mlp_to_layer.get(weight_name)
            if qwen_mlp is not None:
                layer_idx, kind = qwen_mlp
                self._pending_qwen_mlp.setdefault(layer_idx, {})[kind] = (weight_name, x2, output.detach())
                return

            opt_qkv = self._opt_qkv_to_layer.get(weight_name)
            if opt_qkv is not None:
                layer_idx, kind = opt_qkv
                self._pending_opt_qkv.setdefault(layer_idx, {})[kind] = (weight_name, x2, output.detach())
                return

            activation = self._mlp_up_activations.get(weight_name)
            if activation is not None:
                flat_y = self._selected_flat(output)
                if flat_y.shape[0] != flat_x.shape[0]:
                    return
                gain = (
                    activation_local_jacobian_square(flat_y, activation)
                    if self._closed_form
                    else activation_forward_unit_gain(flat_y, activation, eps=self._eps)
                )
                downstream_colnorm = self._mlp_up_downstream_colnorms.get(weight_name)
                if self._closed_form and downstream_colnorm is not None:
                    gain = gain.mul(downstream_colnorm.to(device=gain.device, dtype=gain.dtype).unsqueeze(0))
                gain = self._transform_gain(gain)
                self._add_stat(weight_name, gain.transpose(0, 1).matmul(x2), count=flat_x.shape[0])
                return

            layer_idx = self._residual_output_to_layer.get(weight_name)
            if layer_idx is not None:
                self._pending_x2.setdefault(layer_idx, {})[weight_name] = x2
                return

            qwen_layer_idx = self._qwen_residual_output_to_layer.get(weight_name)
            if qwen_layer_idx is not None:
                self._pending_qwen_x2.setdefault(qwen_layer_idx, {})[weight_name] = x2
                return

            opt_layer_idx = self._opt_residual_output_to_layer.get(weight_name)
            if opt_layer_idx is not None:
                self._pending_opt_x2.setdefault(opt_layer_idx, {})[weight_name] = x2
                return

            if not self._wants(weight_name):
                return
            self._add_stat(weight_name, x2.sum(dim=0), count=flat_x.shape[0])

        return hook

    def _make_attention_hook(self, layer_idx: int):
        def hook(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            kwargs: dict[str, object],
            output: object,
        ) -> None:
            del output
            pending = self._pending_qkv.pop(layer_idx, None)
            if pending is None:
                return
            weight_name, x2, qkv_output = pending
            dense = getattr(module, "dense", None)
            if not isinstance(dense, torch.nn.Linear):
                return
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None and len(inputs) >= 2:
                attention_mask = inputs[1]
            if attention_mask is not None and not torch.is_tensor(attention_mask):
                attention_mask = None
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(inputs) >= 4:
                position_embeddings = inputs[3]
            if not (
                isinstance(position_embeddings, tuple)
                and len(position_embeddings) == 2
                and torch.is_tensor(position_embeddings[0])
                and torch.is_tensor(position_embeddings[1])
            ):
                position_embeddings = None
            num_heads = int(getattr(module, "num_attention_heads", 0) or getattr(getattr(module, "config", None), "num_attention_heads", 0))
            head_size = int(getattr(module, "head_size", 0))
            if num_heads <= 0 or head_size <= 0:
                return
            gain = attention_qkv_superset_output_gains(
                qkv_output,
                dense.weight,
                num_heads=num_heads,
                head_size=head_size,
                attention_mask=attention_mask,
                scaling=float(getattr(module, "scaling", head_size ** -0.5)),
                position_embeddings=position_embeddings,
            )
            flat_gain = self._selected_flat(gain)
            if flat_gain.shape[0] != x2.shape[0]:
                return
            flat_gain = self._transform_gain(flat_gain)
            self._add_stat(weight_name, flat_gain.transpose(0, 1).matmul(x2), count=x2.shape[0])

        return hook

    def _make_qwen_attention_hook(self, layer_idx: int):
        def hook(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            kwargs: dict[str, object],
            output: object,
        ) -> None:
            del output
            pending = self._pending_qwen_qkv.pop(layer_idx, None)
            if not pending or not {"q", "k", "v"}.issubset(pending):
                return
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None and len(inputs) >= 3:
                attention_mask = inputs[2]
            if attention_mask is not None and not torch.is_tensor(attention_mask):
                attention_mask = None
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(inputs) >= 5:
                position_embeddings = inputs[4]
            if not (
                isinstance(position_embeddings, tuple)
                and len(position_embeddings) == 2
                and torch.is_tensor(position_embeddings[0])
                and torch.is_tensor(position_embeddings[1])
            ):
                position_embeddings = None
            q_name, q_x2, q_out = pending["q"]
            k_name, k_x2, k_out = pending["k"]
            v_name, v_x2, v_out = pending["v"]
            o_proj = getattr(module, "o_proj", None)
            q_norm = getattr(module, "q_norm", None)
            k_norm = getattr(module, "k_norm", None)
            if q_norm is not None and not isinstance(q_norm, torch.nn.Module):
                q_norm = None
            if k_norm is not None and not isinstance(k_norm, torch.nn.Module):
                k_norm = None
            if not isinstance(o_proj, torch.nn.Linear):
                return
            num_heads = int(getattr(getattr(module, "config", None), "num_attention_heads", getattr(module, "num_heads", 0)))
            num_kv_heads = int(getattr(getattr(module, "config", None), "num_key_value_heads", getattr(module, "num_key_value_heads", 0)))
            head_size = int(getattr(module, "head_dim", 0))
            if num_heads <= 0 or num_kv_heads <= 0 or head_size <= 0:
                return
            q_gain, k_gain, v_gain = qwen_attention_qkv_superset_output_gains(
                q_out,
                k_out,
                v_out,
                o_proj.weight,
                q_norm,
                k_norm,
                num_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                head_size=head_size,
                attention_mask=attention_mask,
                scaling=float(getattr(module, "scaling", head_size ** -0.5)),
                position_embeddings=position_embeddings,
            )
            for weight_name, x2, gain in ((q_name, q_x2, q_gain), (k_name, k_x2, k_gain), (v_name, v_x2, v_gain)):
                if not self._wants(weight_name):
                    continue
                flat_gain = self._selected_flat(gain)
                if flat_gain.shape[0] == x2.shape[0]:
                    flat_gain = self._transform_gain(flat_gain)
                    self._add_stat(weight_name, flat_gain.transpose(0, 1).matmul(x2), count=x2.shape[0])

        return hook

    def _make_opt_attention_hook(self, layer_idx: int):
        def hook(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            kwargs: dict[str, object],
            output: object,
        ) -> None:
            del output
            pending = self._pending_opt_qkv.pop(layer_idx, None)
            if not pending or not {"q", "k", "v"}.issubset(pending):
                return
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None and len(inputs) >= 3:
                attention_mask = inputs[2]
            if attention_mask is not None and not torch.is_tensor(attention_mask):
                attention_mask = None
            q_name, q_x2, q_out = pending["q"]
            k_name, k_x2, k_out = pending["k"]
            v_name, v_x2, v_out = pending["v"]
            out_proj = getattr(module, "out_proj", None)
            if not isinstance(out_proj, torch.nn.Linear):
                return
            num_heads = int(getattr(module, "num_heads", 0))
            head_size = int(getattr(module, "head_dim", 0))
            if num_heads <= 0 or head_size <= 0:
                return
            q_gain, k_gain, v_gain = qwen_attention_qkv_superset_output_gains(
                q_out,
                k_out,
                v_out,
                out_proj.weight,
                None,
                None,
                num_heads=num_heads,
                num_key_value_heads=num_heads,
                head_size=head_size,
                attention_mask=attention_mask,
                scaling=float(getattr(module, "scaling", head_size ** -0.5)),
                position_embeddings=None,
            )
            for weight_name, x2, gain in ((q_name, q_x2, q_gain), (k_name, k_x2, k_gain), (v_name, v_x2, v_gain)):
                if not self._wants(weight_name):
                    continue
                flat_gain = self._selected_flat(gain)
                if flat_gain.shape[0] == x2.shape[0]:
                    flat_gain = self._transform_gain(flat_gain)
                    self._add_stat(weight_name, flat_gain.transpose(0, 1).matmul(x2), count=x2.shape[0])

        return hook

    def _make_qwen_mlp_hook(self, layer_idx: int):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            del inputs, output
            pending = self._pending_qwen_mlp.pop(layer_idx, None)
            if not pending or not {"gate", "up"}.issubset(pending):
                return
            down = getattr(module, "down_proj", None)
            if not isinstance(down, torch.nn.Linear):
                return
            gate_name, gate_x2, gate_out = pending["gate"]
            up_name, up_x2, up_out = pending["up"]
            gate_gain, up_gain = qwen_mlp_superset_output_gains(gate_out, up_out, down.weight)
            for weight_name, x2, gain in ((gate_name, gate_x2, gate_gain), (up_name, up_x2, up_gain)):
                if not self._wants(weight_name):
                    continue
                flat_gain = self._selected_flat(gain)
                if flat_gain.shape[0] == x2.shape[0]:
                    flat_gain = self._transform_gain(flat_gain)
                    self._add_stat(weight_name, flat_gain.transpose(0, 1).matmul(x2), count=x2.shape[0])

        return hook

    def _make_layer_hook(self, layer_idx: int):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if layer_idx == 0 and inputs and torch.is_tensor(inputs[0]):
                self._flush_embed_in(module, inputs[0])
            hidden = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(hidden):
                self._flush_residual_outputs(layer_idx, hidden)

        return hook

    def _make_qwen_layer_hook(self, layer_idx: int):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if layer_idx == 0 and inputs and torch.is_tensor(inputs[0]):
                self._flush_qwen_embed_tokens(module, inputs[0])
            hidden = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(hidden):
                self._flush_qwen_residual_outputs(layer_idx, hidden)

        return hook

    def _make_opt_layer_hook(self, layer_idx: int):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if layer_idx == 0 and inputs and torch.is_tensor(inputs[0]):
                self._flush_opt_embeddings(module, inputs[0])
            hidden = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(hidden):
                self._flush_opt_residual_outputs(layer_idx, hidden)

        return hook

    def _layer_input_gain(self, layer: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._closed_form:
            gain: torch.Tensor | None = None
            attention = getattr(layer, "attention", None)
            qkv = getattr(attention, "query_key_value", None)
            if isinstance(qkv, torch.nn.Linear):
                gain = layernorm_input_downstream_colnorm_squares(hidden_states, layer.input_layernorm, qkv.weight)
            mlp = getattr(layer, "mlp", None)
            up = getattr(mlp, "dense_h_to_4h", None)
            post_norm = getattr(layer, "post_attention_layernorm", None)
            if isinstance(up, torch.nn.Linear) and post_norm is not None:
                mlp_gain = layernorm_input_downstream_colnorm_squares(hidden_states, post_norm, up.weight)
                gain = mlp_gain if gain is None else gain.add(mlp_gain)
            if gain is not None:
                return gain
        norm_gain = layernorm_input_jacobian_colnorm_squares if self._closed_form else layernorm_forward_diff_colnorm_squares
        kwargs = {} if self._closed_form else {"eps": self._eps}
        gain = norm_gain(hidden_states, layer.input_layernorm, **kwargs)
        post_norm = getattr(layer, "post_attention_layernorm", None)
        if post_norm is not None:
            gain = gain.add(norm_gain(hidden_states, post_norm, **kwargs))
        return gain

    def _qwen_layer_input_gain(self, layer: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        attention = getattr(layer, "self_attn", None)
        input_norm = getattr(layer, "input_layernorm", None)
        gain: torch.Tensor | None = None
        if self._closed_form and input_norm is not None and attention is not None:
            for attr in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(attention, attr, None)
                if isinstance(proj, torch.nn.Linear):
                    piece = rmsnorm_input_downstream_colnorm_squares(hidden_states, input_norm, proj.weight)
                    gain = piece if gain is None else gain.add(piece)
        mlp = getattr(layer, "mlp", None)
        post_norm = getattr(layer, "post_attention_layernorm", None)
        if self._closed_form and post_norm is not None and mlp is not None:
            for attr in ("gate_proj", "up_proj"):
                proj = getattr(mlp, attr, None)
                if isinstance(proj, torch.nn.Linear):
                    piece = rmsnorm_input_downstream_colnorm_squares(hidden_states, post_norm, proj.weight)
                    gain = piece if gain is None else gain.add(piece)
        if gain is not None:
            return gain
        raise ValueError("Qwen superset WANDA requires closed_form=True")

    def _opt_layer_input_gain(self, layer: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        attention = getattr(layer, "self_attn", None)
        attn_norm = getattr(layer, "self_attn_layer_norm", None)
        mlp_norm = getattr(layer, "final_layer_norm", None)
        gain: torch.Tensor | None = None
        if bool(getattr(layer, "do_layer_norm_before", True)) and self._closed_form:
            if attention is not None and attn_norm is not None:
                for attr in ("q_proj", "k_proj", "v_proj"):
                    proj = getattr(attention, attr, None)
                    if isinstance(proj, torch.nn.Linear):
                        piece = layernorm_input_downstream_colnorm_squares(hidden_states, attn_norm, proj.weight)
                        gain = piece if gain is None else gain.add(piece)
            fc1 = getattr(layer, "fc1", None)
            if mlp_norm is not None and isinstance(fc1, torch.nn.Linear):
                piece = layernorm_input_downstream_colnorm_squares(hidden_states, mlp_norm, fc1.weight)
                gain = piece if gain is None else gain.add(piece)
        elif attention is not None:
            for attr in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(attention, attr, None)
                if isinstance(proj, torch.nn.Linear):
                    colnorm = proj.weight.detach().to(device=hidden_states.device, dtype=torch.float32).square().sum(dim=0)
                    piece = colnorm.unsqueeze(0).expand(hidden_states.shape[0], -1)
                    gain = piece if gain is None else gain.add(piece)
        if gain is not None:
            return gain
        raise ValueError("OPT superset WANDA requires closed_form=True")

    def _qwen_downstream_residual_gain(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor | None:
        flat_hidden = self._selected_flat(hidden_states)
        if flat_hidden.numel() == 0:
            return None
        if layer_idx + 1 < len(self._qwen_layers):
            return self._qwen_layer_input_gain(self._qwen_layers[layer_idx + 1], flat_hidden)
        if self._qwen_final_norm is not None:
            lm_head = getattr(self._model, "lm_head", None)
            if self._closed_form and isinstance(lm_head, torch.nn.Linear):
                return rmsnorm_input_downstream_colnorm_squares(flat_hidden, self._qwen_final_norm, lm_head.weight)
        return None

    def _opt_downstream_residual_gain(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor | None:
        flat_hidden = self._selected_flat(hidden_states)
        if flat_hidden.numel() == 0:
            return None
        if layer_idx + 1 < len(self._opt_layers):
            return self._opt_layer_input_gain(self._opt_layers[layer_idx + 1], flat_hidden)
        lm_head = getattr(self._model, "lm_head", None)
        if isinstance(lm_head, torch.nn.Linear):
            if self._opt_project_out is not None and isinstance(self._opt_project_out, torch.nn.Linear):
                downstream = lm_head.weight.detach().to(device=flat_hidden.device, dtype=torch.float32).matmul(
                    self._opt_project_out.weight.detach().to(device=flat_hidden.device, dtype=torch.float32)
                )
            else:
                downstream = lm_head.weight
            if self._opt_final_norm is not None:
                return layernorm_input_downstream_colnorm_squares(flat_hidden, self._opt_final_norm, downstream)
            colnorm = downstream.detach().to(device=flat_hidden.device, dtype=torch.float32).square().sum(dim=0)
            return colnorm.unsqueeze(0).expand(flat_hidden.shape[0], -1)
        return None

    def _downstream_residual_gain(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor | None:
        flat_hidden = self._selected_flat(hidden_states)
        if flat_hidden.numel() == 0:
            return None
        if layer_idx + 1 < len(self._layers):
            return self._layer_input_gain(self._layers[layer_idx + 1], flat_hidden)
        if self._final_layer_norm is not None:
            embed_out = getattr(self._model, "embed_out", None)
            if self._closed_form and isinstance(embed_out, torch.nn.Linear):
                return layernorm_input_downstream_colnorm_squares(flat_hidden, self._final_layer_norm, embed_out.weight)
            return layernorm_forward_diff_colnorm_squares(flat_hidden, self._final_layer_norm, eps=self._eps)
        return None

    def _flush_embed_in(self, layer: torch.nn.Module, hidden_states: torch.Tensor) -> None:
        weight_name = "gpt_neox.embed_in.weight"
        if not self._wants(weight_name):
            return
        token_ids = self._selected_token_ids()
        if token_ids is None or token_ids.numel() == 0:
            return
        flat_hidden = self._selected_flat(hidden_states)
        if flat_hidden.shape[0] != token_ids.numel():
            return
        gain = self._transform_gain(self._layer_input_gain(layer, flat_hidden)).to(device="cpu", dtype=torch.float32)
        embed = getattr(getattr(self._model, "gpt_neox", None), "embed_in", None)
        if not isinstance(embed, torch.nn.Embedding):
            return
        stat = self._stats.get(weight_name)
        if stat is None:
            stat = torch.zeros_like(embed.weight.detach(), dtype=torch.float32, device="cpu")
            self._stats[weight_name] = stat
        stat.index_add_(0, token_ids.to(device="cpu", dtype=torch.long), gain)
        self._counts[weight_name] = self._counts.get(weight_name, 0) + int(token_ids.numel())

    def _flush_qwen_embed_tokens(self, layer: torch.nn.Module, hidden_states: torch.Tensor) -> None:
        weight_name = "model.embed_tokens.weight"
        if not self._wants(weight_name):
            return
        token_ids = self._selected_token_ids()
        if token_ids is None or token_ids.numel() == 0:
            return
        flat_hidden = self._selected_flat(hidden_states)
        if flat_hidden.shape[0] != token_ids.numel():
            return
        gain = self._transform_gain(self._qwen_layer_input_gain(layer, flat_hidden)).to(device="cpu", dtype=torch.float32)
        embed = getattr(getattr(self._model, "model", None), "embed_tokens", None)
        if not isinstance(embed, torch.nn.Embedding):
            return
        stat = self._stats.get(weight_name)
        if stat is None:
            stat = torch.zeros_like(embed.weight.detach(), dtype=torch.float32, device="cpu")
            self._stats[weight_name] = stat
        stat.index_add_(0, token_ids.to(device="cpu", dtype=torch.long), gain)
        self._counts[weight_name] = self._counts.get(weight_name, 0) + int(token_ids.numel())

    def _flush_opt_embeddings(self, layer: torch.nn.Module, hidden_states: torch.Tensor) -> None:
        token_ids = self._selected_token_ids()
        if token_ids is None or token_ids.numel() == 0:
            return
        flat_hidden = self._selected_flat(hidden_states)
        if flat_hidden.shape[0] != token_ids.numel():
            return
        gain = self._transform_gain(self._opt_layer_input_gain(layer, flat_hidden)).to(device="cpu", dtype=torch.float32)
        token_name = "model.decoder.embed_tokens.weight"
        if self._wants(token_name) and isinstance(self._opt_embed_tokens, torch.nn.Embedding):
            stat = self._stats.get(token_name)
            if stat is None:
                stat = torch.zeros_like(self._opt_embed_tokens.weight.detach(), dtype=torch.float32, device="cpu")
                self._stats[token_name] = stat
            stat.index_add_(0, token_ids.to(device="cpu", dtype=torch.long), gain)
            self._counts[token_name] = self._counts.get(token_name, 0) + int(token_ids.numel())

        position_name = "model.decoder.embed_positions.weight"
        if self._wants(position_name) and isinstance(self._opt_embed_positions, torch.nn.Embedding):
            input_ids = self._current_input_ids
            if input_ids is None:
                return
            if self._current_attention_mask is not None:
                positions = self._current_attention_mask.to(dtype=torch.long).cumsum(dim=1).sub_(1)
                positions.masked_fill_(~self._current_attention_mask, 0)
            else:
                positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            positions = positions.add(int(getattr(self._opt_embed_positions, "offset", 0)))
            if self._use_attention_mask and self._current_attention_mask is not None:
                position_ids = positions[self._current_attention_mask]
            else:
                position_ids = positions.reshape(-1)
            if position_ids.numel() != gain.shape[0]:
                return
            stat = self._stats.get(position_name)
            if stat is None:
                stat = torch.zeros_like(self._opt_embed_positions.weight.detach(), dtype=torch.float32, device="cpu")
                self._stats[position_name] = stat
            stat.index_add_(0, position_ids.to(device="cpu", dtype=torch.long), gain)
            self._counts[position_name] = self._counts.get(position_name, 0) + int(position_ids.numel())

    def _flush_residual_outputs(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        pending = self._pending_x2.pop(layer_idx, None)
        if not pending:
            return
        gain = self._downstream_residual_gain(layer_idx, hidden_states)
        if gain is None:
            return
        gain = self._transform_gain(gain)
        for weight_name, x2 in pending.items():
            if x2.shape[0] != gain.shape[0]:
                continue
            self._add_stat(weight_name, gain.transpose(0, 1).matmul(x2), count=x2.shape[0])

    def _flush_opt_residual_outputs(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        pending = self._pending_opt_x2.pop(layer_idx, None)
        if not pending:
            return
        gain = self._opt_downstream_residual_gain(layer_idx, hidden_states)
        if gain is None:
            return
        gain = self._transform_gain(gain)
        for weight_name, x2 in pending.items():
            if x2.shape[0] != gain.shape[0]:
                continue
            self._add_stat(weight_name, gain.transpose(0, 1).matmul(x2), count=x2.shape[0])

    def _flush_qwen_residual_outputs(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        pending = self._pending_qwen_x2.pop(layer_idx, None)
        if not pending:
            return
        gain = self._qwen_downstream_residual_gain(layer_idx, hidden_states)
        if gain is None:
            return
        gain = self._transform_gain(gain)
        for weight_name, x2 in pending.items():
            if x2.shape[0] != gain.shape[0]:
                continue
            self._add_stat(weight_name, gain.transpose(0, 1).matmul(x2), count=x2.shape[0])

    def finalize(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
        scores: dict[str, torch.Tensor] = {}
        for name, param in named_parameters:
            if not param.requires_grad or param.ndim != 2:
                continue
            if self._target_names is not None and name not in self._target_names:
                continue
            stat = self._stats.get(name)
            if stat is None:
                raise ValueError(f"local forward WANDA produced no statistic for {name}")
            weight_sq = param.detach().to(device="cpu", dtype=torch.float32).square()
            stat = stat.to(device="cpu", dtype=torch.float32)
            if tuple(stat.shape) == tuple(weight_sq.shape):
                scores[name] = weight_sq.mul(stat)
            elif stat.ndim == 1 and stat.numel() == weight_sq.shape[1]:
                scores[name] = weight_sq.mul(stat.unsqueeze(0))
            else:
                raise ValueError(f"local forward WANDA statistic shape mismatch for {name}: {tuple(stat.shape)} != {tuple(weight_sq.shape)}")
        return scores


class RowConditionedWandaAccumulator:
    def __init__(self, model: torch.nn.Module):
        self._model = model
        self._weighted_abs_input: dict[str, torch.Tensor] = {}
        self._row_weight_sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> RowConditionedWandaAccumulator:
        for module_name, module in self._model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight = module.weight
            if weight is None or not weight.requires_grad or weight.ndim != 2:
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(_weight_name(module_name))))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, weight_name: str):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            del module
            if not inputs or not torch.is_tensor(output):
                return
            activations = inputs[0]
            if not torch.is_tensor(activations) or activations.numel() == 0 or output.numel() == 0:
                return
            flat_x = activations.detach().reshape(-1, activations.shape[-1]).abs().float()
            flat_y = output.detach().reshape(-1, output.shape[-1]).abs().float()
            if flat_x.shape[0] != flat_y.shape[0]:
                return
            weighted_abs_input = flat_y.transpose(0, 1).matmul(flat_x)
            row_weight_sums = flat_y.sum(dim=0)
            if weight_name in self._weighted_abs_input:
                self._weighted_abs_input[weight_name].add_(weighted_abs_input)
                self._row_weight_sums[weight_name].add_(row_weight_sums)
            else:
                self._weighted_abs_input[weight_name] = weighted_abs_input
                self._row_weight_sums[weight_name] = row_weight_sums
            self._counts[weight_name] = self._counts.get(weight_name, 0) + flat_x.shape[0]

        return hook

    def finalize(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        fallback_to_magnitude: bool = True,
    ) -> dict[str, torch.Tensor]:
        scores: dict[str, torch.Tensor] = {}
        for name, param in named_parameters:
            if not param.requires_grad or param.ndim != 2:
                continue
            weight_abs = param.detach().abs().to(device="cpu", dtype=torch.float32)
            weighted_abs_input = self._weighted_abs_input.get(name)
            row_weight_sums = self._row_weight_sums.get(name)
            if weighted_abs_input is not None and row_weight_sums is not None:
                activation = weighted_abs_input.div(row_weight_sums.clamp_min(1e-12).unsqueeze(1)).to(
                    device="cpu",
                    dtype=torch.float32,
                )
                if tuple(activation.shape) != tuple(weight_abs.shape):
                    raise ValueError(
                        f"row-conditioned WANDA activation shape mismatch for {name}: "
                        f"{tuple(activation.shape)} != {tuple(weight_abs.shape)}"
                    )
                scores[name] = weight_abs.mul(activation)
            elif fallback_to_magnitude:
                scores[name] = weight_abs
        return scores


class DfaActivationCollector:
    def __init__(self, model: torch.nn.Module):
        self._model = model
        self.inputs: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> DfaActivationCollector:
        self.inputs.clear()
        for module_name, module in self._model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight = module.weight
            if weight is None or not weight.requires_grad or weight.ndim != 2:
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(_weight_name(module_name))))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, weight_name: str):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            del module, output
            if not inputs:
                return
            activations = inputs[0]
            if torch.is_tensor(activations):
                self.inputs[weight_name] = activations.detach().to(device="cpu")

        return hook


def cross_entropy_residual(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    if not bool(mask.any().item()):
        raise ValueError("batch has no supervised target tokens after causal shift")
    flat_logits = shift_logits[mask]
    flat_labels = shift_labels[mask]
    residual = torch.softmax(flat_logits, dim=-1)
    residual[torch.arange(flat_labels.numel(), device=residual.device), flat_labels] -= 1.0
    return residual, mask


def hash_project_residual(residual: torch.Tensor, *, out_features: int, seed: int) -> torch.Tensor:
    if residual.ndim != 2:
        raise ValueError("residual must be a 2D [tokens, vocab] tensor")
    vocab_size = residual.shape[1]
    generator = torch.Generator(device=residual.device)
    generator.manual_seed(int(seed) + 104729 * int(out_features) + 1009 * int(vocab_size))
    buckets = torch.randint(out_features, (vocab_size,), generator=generator, device=residual.device)
    signs = torch.randint(2, (vocab_size,), generator=generator, device=residual.device)
    signed_residual = residual.float() * signs.float().mul_(2.0).sub_(1.0).unsqueeze(0)
    projected = torch.zeros((residual.shape[0], out_features), dtype=torch.float32, device=residual.device)
    projected.scatter_add_(1, buckets.unsqueeze(0).expand(residual.shape[0], -1), signed_residual)
    return projected


def _prepare_tokenizer(model_name: str, revision: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _prepare_model(model_name: str, revision: str | None, dtype: torch.dtype, device: torch.device):
    load_dtype = dtype if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=load_dtype)
    model.config.use_cache = False
    model.to(device)
    model.eval()
    return model


def _load_records(config: ApproxSaliencyConfig) -> list[dict[str, Any]]:
    dataset = load_dataset(config.dataset_name, config.dataset_config, split=config.split)
    limit = min(config.max_examples, len(dataset)) if config.max_examples > 0 else len(dataset)
    return [dict(row) for row in dataset.select(range(limit))]


_GPT_NEOX_LAYER_PARAM_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.")


def sequential_wanda_parameter_groups(model: torch.nn.Module) -> list[tuple[str, list[str]]]:
    groups: dict[tuple[int, int, str], tuple[str, list[str]]] = {}
    for order, (name, param) in enumerate(model.named_parameters()):
        if not param.requires_grad or param.ndim != 2:
            continue
        match = _GPT_NEOX_LAYER_PARAM_RE.match(name)
        if match is not None:
            layer_idx = int(match.group(1))
            key = (1, layer_idx, f"gpt_neox.layers.{layer_idx}")
            label = f"gpt_neox.layers.{layer_idx}"
        elif "embed_out" in name or "lm_head" in name:
            key = (2, order, name)
            label = name
        elif "embed_in" in name or "wte" in name or "embed_tokens" in name:
            key = (0, order, name)
            label = name
        else:
            key = (3, order, name)
            label = name
        if key not in groups:
            groups[key] = (label, [])
        groups[key][1].append(name)
    return [groups[key] for key in sorted(groups)]


def sequential_wanda_matrix_parameter_groups(model: torch.nn.Module) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    for _, target_names in sequential_wanda_parameter_groups(model):
        for name in target_names:
            groups.append((name, [name]))
    return groups


def _wanda_pruning_mask(score: torch.Tensor, *, pruning_scope: str, fraction: float) -> torch.Tensor:
    if pruning_scope == "per_matrix":
        return lowest_saliency_mask(score, fraction=fraction)
    if pruning_scope == "per_output_row":
        return lowest_saliency_mask_per_output_row(score, fraction=fraction)
    raise ValueError(f"unknown WANDA ablation pruning_scope: {pruning_scope}")


def apply_wanda_group_pruning_(
    model: torch.nn.Module,
    saliency_scores: dict[str, torch.Tensor],
    *,
    target_names: Iterable[str],
    pruning_scope: str,
    fraction: float,
    group_name: str,
    step: int,
) -> dict[str, object]:
    target = set(target_names)
    rows: list[dict[str, object]] = []
    weights_seen = 0
    weights_zeroed = 0
    missing_saliency: list[str] = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in target or not param.requires_grad or param.ndim != 2:
                continue
            weights_seen += param.numel()
            score = saliency_scores.get(name)
            if score is None:
                missing_saliency.append(name)
                continue
            if tuple(score.shape) != tuple(param.shape):
                raise ValueError(f"saliency shape mismatch for {name}: {tuple(score.shape)} != {tuple(param.shape)}")

            mask = _wanda_pruning_mask(score.to(device=param.device), pruning_scope=pruning_scope, fraction=fraction)
            param.masked_fill_(mask, 0)
            zeroed = int(mask.sum().item())
            weights_zeroed += zeroed
            rows.append(
                {
                    "step": step,
                    "group": group_name,
                    "name": name,
                    "shape": list(param.shape),
                    "weights": param.numel(),
                    "zeroed": zeroed,
                }
            )

    return {
        "step": step,
        "group": group_name,
        "pruning_scope": pruning_scope,
        "prune_fraction": fraction,
        "matrix_tensors_seen": len(rows) + len(missing_saliency),
        "matrix_tensors_pruned": len(rows),
        "weights_seen": weights_seen,
        "weights_zeroed": weights_zeroed,
        "actual_zero_fraction": weights_zeroed / max(weights_seen, 1),
        "missing_saliency": missing_saliency,
        "pruned_tensors": rows,
    }


def _wanda_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
    *,
    use_attention_mask: bool = True,
) -> dict[str, torch.Tensor]:
    scores, _, _ = _wanda_scores_and_hessian_diagonal(
        model,
        tokenizer,
        records,
        config,
        device,
        use_attention_mask=use_attention_mask,
    )
    return scores


def _wanda_scores_and_hessian_diagonal(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
    *,
    use_attention_mask: bool = True,
    target_names: Iterable[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, int]]:
    batches = list(batched(records, config.batch_size))
    target_names_tuple = tuple(target_names) if target_names is not None else None
    with torch.inference_mode(), WandaActivationAccumulator(model, target_names=target_names_tuple) as accumulator:
        for record_batch in tqdm(batches, desc="wanda_activations", unit="batch"):
            batch = build_causal_lm_batch(
                tokenizer,
                record_batch,
                config.max_length,
                answer_only_loss=config.answer_only_loss,
                device=device,
            )
            if use_attention_mask:
                accumulator.set_attention_mask(batch["attention_mask"])
            try:
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
            finally:
                accumulator.clear_attention_mask()
    hessians, tokens = accumulator.hessian_diagonals()
    if target_names_tuple is None:
        named_parameters = list(model.named_parameters())
    else:
        target = set(target_names_tuple)
        named_parameters = [(name, param) for name, param in model.named_parameters() if name in target]
    return accumulator.finalize(named_parameters), hessians, tokens


def _ria_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batches = list(batched(records, config.batch_size))
    with torch.inference_mode(), WandaActivationAccumulator(model) as accumulator:
        for record_batch in tqdm(batches, desc="ria_activations", unit="batch"):
            batch = build_causal_lm_batch(
                tokenizer,
                record_batch,
                config.max_length,
                answer_only_loss=config.answer_only_loss,
                device=device,
            )
            accumulator.set_attention_mask(batch["attention_mask"])
            try:
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
            finally:
                accumulator.clear_attention_mask()
    return relative_importance_scores(
        model.named_parameters(),
        activation_rms=accumulator.activation_rms(),
        activation_exponent=0.5,
    )


def _input_activation_stat_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
    method: str,
) -> dict[str, torch.Tensor]:
    batches = list(batched(records, config.batch_size))
    max_rows = max(1, int(config.max_examples) * int(config.max_length)) if method in {"q95_wanda", "wanda_q95", "outlier_q95"} else 0
    with torch.inference_mode(), InputActivationStatsAccumulator(model, quantile=0.95, max_rows=max_rows) as accumulator:
        for record_batch in tqdm(batches, desc=f"{method}_activations", unit="batch"):
            batch = build_causal_lm_batch(
                tokenizer,
                record_batch,
                config.max_length,
                answer_only_loss=config.answer_only_loss,
                device=device,
            )
            model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
    return input_activation_stat_scores(model.named_parameters(), accumulator.finalize_stats(), method)


def _angular_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
    method: str,
) -> dict[str, torch.Tensor]:
    batches = list(batched(records, config.batch_size))
    with torch.inference_mode(), AngularActivationAccumulator(model) as accumulator:
        for record_batch in tqdm(batches, desc=f"{method}_activations", unit="batch"):
            batch = build_causal_lm_batch(
                tokenizer,
                record_batch,
                config.max_length,
                answer_only_loss=config.answer_only_loss,
                device=device,
            )
            model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
    return angular_saliency_scores(
        model.named_parameters(),
        accumulator.finalize_stats(),
        method,
        hybrid_lambda=config.angular_hybrid_lambda,
    )


def _row_conditioned_wanda_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batches = list(batched(records, config.batch_size))
    with torch.inference_mode(), RowConditionedWandaAccumulator(model) as accumulator:
        for record_batch in tqdm(batches, desc="row_wanda_activations", unit="batch"):
            batch = build_causal_lm_batch(
                tokenizer,
                record_batch,
                config.max_length,
                answer_only_loss=config.answer_only_loss,
                device=device,
            )
            model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
    return accumulator.finalize(model.named_parameters())


def _feature_wanda_cosine_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batches = list(batched(records, config.batch_size))
    with torch.inference_mode(), FeatureCosineWandaAccumulator(model) as accumulator:
        for record_batch in tqdm(batches, desc="feature_cosine_wanda_activations", unit="batch"):
            batch = build_causal_lm_batch(
                tokenizer,
                record_batch,
                config.max_length,
                answer_only_loss=config.answer_only_loss,
                device=device,
            )
            model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
    return feature_wanda_cosine_scores(
        model.named_parameters(),
        accumulator.finalize_stats(),
        alpha=config.feature_cosine_alpha,
        cosine_clip=config.feature_cosine_clip,
    )


def _graph_propagated_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
    method: str,
) -> dict[str, torch.Tensor]:
    batches = list(batched(records, config.batch_size))
    with torch.inference_mode(), GPTNeoXGraphPropagatedAccumulator(
        model,
        method=method,
        num_probes=config.graph_num_probes,
        seed=config.graph_seed,
    ) as accumulator:
        for record_batch in tqdm(batches, desc=f"{method}_activations", unit="batch"):
            batch = build_causal_lm_batch(
                tokenizer,
                record_batch,
                config.max_length,
                answer_only_loss=config.answer_only_loss,
                device=device,
            )
            model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
    return accumulator.finalize(model.named_parameters())


def _local_forward_wanda_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
    method: str,
) -> dict[str, torch.Tensor]:
    del method
    batches = list(batched(records, config.batch_size))
    with torch.inference_mode(), LocalForwardWandaAccumulator(model, eps=config.local_forward_eps) as accumulator:
        for record_batch in tqdm(batches, desc="local_forward_wanda", unit="batch"):
            batch = build_causal_lm_batch(
                tokenizer,
                record_batch,
                config.max_length,
                answer_only_loss=config.answer_only_loss,
                device=device,
            )
            accumulator.set_batch(batch["input_ids"], batch["attention_mask"])
            model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            accumulator.clear_batch()
    return accumulator.finalize(model.named_parameters())


SUPERSET_WANDA_METHODS = {
    "superset_wanda",
    "closed_form_superset_wanda",
    "superset_subgraph_wanda",
    "closed_form_subgraph_wanda",
}


def _superset_wanda_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
    *,
    use_attention_mask: bool,
    target_names: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    batches = list(batched(records, config.batch_size))
    with torch.inference_mode(), LocalForwardWandaAccumulator(
        model,
        eps=config.local_forward_eps,
        use_attention_mask=use_attention_mask,
        target_names=target_names,
        closed_form=True,
        superset_gain_power=config.superset_gain_power,
        superset_gain_clip_quantile=config.superset_gain_clip_quantile,
    ) as accumulator:
        for record_batch in tqdm(batches, desc="superset_wanda", unit="batch"):
            batch = build_causal_lm_batch(
                tokenizer,
                record_batch,
                config.max_length,
                answer_only_loss=config.answer_only_loss,
                device=device,
            )
            accumulator.set_batch(batch["input_ids"], batch["attention_mask"])
            model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            accumulator.clear_batch()
    return accumulator.finalize(model.named_parameters())


def _wanda_ablation_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
    *,
    use_attention_mask: bool,
    target_names: Iterable[str],
    score_method: str,
) -> dict[str, torch.Tensor]:
    normalized = score_method.strip().lower().replace("-", "_")
    target_names_tuple = tuple(target_names)
    if normalized in SUPERSET_WANDA_METHODS:
        return _superset_wanda_scores(
            model,
            tokenizer,
            records,
            config,
            device,
            use_attention_mask=use_attention_mask,
            target_names=target_names_tuple,
        )
    if normalized in {"magnitude", "weight_magnitude"}:
        target = set(target_names_tuple)
        return weight_magnitude_scores(
            ((name, param) for name, param in model.named_parameters() if name in target),
        )
    if normalized not in {"wanda", "wanda_masked", "proper_wanda", "wanda_unmasked", "original_wanda", "legacy_wanda"}:
        raise ValueError(f"unknown WANDA ablation score_method: {score_method}")
    scores, _, _ = _wanda_scores_and_hessian_diagonal(
        model,
        tokenizer,
        records,
        config,
        device,
        use_attention_mask=use_attention_mask,
        target_names=target_names_tuple,
    )
    return scores


def _empty_matrix_scores(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(param.detach(), dtype=torch.float32, device="cpu")
        for name, param in model.named_parameters()
        if param.requires_grad and param.ndim == 2
    }


def _graph_vjp_logits_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    if config.graph_num_probes <= 0:
        raise ValueError("graph_num_probes must be positive for graph_vjp_logits")
    scores = _empty_matrix_scores(model)
    total_supervised_tokens = 0
    batches = list(batched(records, config.batch_size))
    generator = torch.Generator(device=device)
    progress = tqdm(batches, desc="graph_vjp_logits", unit="batch")
    for batch_idx, record_batch in enumerate(progress):
        batch = build_causal_lm_batch(
            tokenizer,
            record_batch,
            config.max_length,
            answer_only_loss=config.answer_only_loss,
            device=device,
        )
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        shift_logits = outputs.logits[:, :-1, :]
        if config.answer_only_loss:
            endpoint_mask = batch["labels"][:, 1:] != -100
        else:
            endpoint_mask = batch["attention_mask"][:, 1:].bool()
        selected_logits = shift_logits[endpoint_mask]
        supervised_tokens = int(selected_logits.shape[0])
        if supervised_tokens == 0:
            continue
        total_supervised_tokens += supervised_tokens

        for probe_idx in range(int(config.graph_num_probes)):
            model.zero_grad(set_to_none=True)
            generator.manual_seed(int(config.graph_seed) + 1_000_003 * batch_idx + 10_007 * probe_idx)
            signs = torch.randint(
                0,
                2,
                selected_logits.shape,
                device=device,
                generator=generator,
                dtype=torch.int8,
            )
            random_projection = signs.to(dtype=torch.float32).mul_(2.0).sub_(1.0)
            scalar = selected_logits.float().mul(random_projection).sum()
            scalar.backward(retain_graph=probe_idx + 1 < int(config.graph_num_probes))
            accumulate_vjp_parameter_scores_(scores, model.named_parameters())

        progress.set_postfix(tokens=total_supervised_tokens)
        model.zero_grad(set_to_none=True)
        del outputs, shift_logits, selected_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    normalizer = max(total_supervised_tokens * int(config.graph_num_probes), 1)
    return {name: score.div(normalizer) for name, score in scores.items()}, total_supervised_tokens


def _select_prediction_positions(
    tensor: torch.Tensor,
    batch: dict[str, torch.Tensor | int],
    *,
    answer_aligned: bool,
) -> torch.Tensor:
    if tensor.ndim != 3:
        return tensor.reshape(-1, tensor.shape[-1])
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    if not torch.is_tensor(labels) or not torch.is_tensor(attention_mask):
        raise TypeError("batch labels and attention_mask must be tensors")
    width = min(tensor.shape[1] - 1, labels.shape[1] - 1)
    if width <= 0:
        return tensor.reshape(0, tensor.shape[-1])
    positions = tensor[:, :width, :]
    if answer_aligned:
        mask = labels[:, 1 : width + 1] != -100
    else:
        mask = attention_mask[:, 1 : width + 1].bool()
    return positions[mask]


def _accumulate_local_endpoint_vjp_(
    scores: dict[str, torch.Tensor],
    normalizers: dict[str, int],
    params_by_name: dict[str, torch.nn.Parameter],
    endpoint: torch.Tensor,
    batch: dict[str, torch.Tensor | int],
    target_names: list[str],
    *,
    answer_aligned: bool,
    num_probes: int,
    generator: torch.Generator,
    seed: int,
) -> int:
    selected = _select_prediction_positions(endpoint, batch, answer_aligned=answer_aligned)
    if selected.numel() == 0 or selected.shape[0] == 0:
        return 0
    target_items = [(name, params_by_name[name]) for name in target_names if name in params_by_name]
    if not target_items:
        return int(selected.shape[0])
    target_params = [param for _, param in target_items]
    for probe_idx in range(num_probes):
        generator.manual_seed(int(seed) + 10_007 * probe_idx)
        signs = torch.randint(
            0,
            2,
            selected.shape,
            device=selected.device,
            generator=generator,
            dtype=torch.int8,
        )
        random_projection = signs.to(dtype=torch.float32).mul_(2.0).sub_(1.0)
        scalar = selected.float().mul(random_projection).sum()
        grads = torch.autograd.grad(
            scalar,
            target_params,
            retain_graph=True,
            allow_unused=True,
        )
        accumulate_vjp_gradient_scores_(
            scores,
            normalizers,
            ((name, param, grad) for (name, param), grad in zip(target_items, grads, strict=True)),
            count=int(selected.shape[0]),
        )
    return int(selected.shape[0])


def _local_subgraph_vjp_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
    method: str,
) -> tuple[dict[str, torch.Tensor], int, int]:
    if config.graph_num_probes <= 0:
        raise ValueError("graph_num_probes must be positive for local subgraph VJP saliency")
    answer_aligned = "all_tokens" not in method.lower()
    scores = _empty_matrix_scores(model)
    normalizers: dict[str, int] = {}
    params_by_name = {name: param for name, param in model.named_parameters() if param.requires_grad and param.ndim == 2}
    total_endpoint_tokens = 0
    total_supervised_tokens = 0
    batches = list(batched(records, config.batch_size))
    generator = torch.Generator(device=device)

    with LocalSubgraphEndpointCollector(model) as collector:
        groups = build_local_subgraph_endpoint_groups(
            num_layers=len(collector.layers),
            parameter_names=set(params_by_name),
        )
        covered = {target for group in groups for target in group["targets"]}
        missing_targets = sorted(set(params_by_name) - covered)
        if missing_targets:
            raise ValueError(f"local subgraph VJP has no endpoint for parameters: {missing_targets}")

        progress = tqdm(batches, desc=method, unit="batch")
        for batch_idx, record_batch in enumerate(progress):
            collector.clear()
            batch = build_causal_lm_batch(
                tokenizer,
                record_batch,
                config.max_length,
                answer_only_loss=config.answer_only_loss,
                device=device,
            )
            labels = batch["labels"]
            if torch.is_tensor(labels):
                total_supervised_tokens += int((labels[:, 1:] != -100).sum().item())
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            endpoints = dict(collector.endpoints)
            endpoints["logits"] = outputs.logits
            for group_idx, group in enumerate(groups):
                endpoint_name = str(group["endpoint"])
                endpoint = endpoints.get(endpoint_name)
                if endpoint is None:
                    raise ValueError(f"missing local subgraph endpoint: {endpoint_name}")
                total_endpoint_tokens += _accumulate_local_endpoint_vjp_(
                    scores,
                    normalizers,
                    params_by_name,
                    endpoint,
                    batch,
                    list(group["targets"]),
                    answer_aligned=answer_aligned,
                    num_probes=int(config.graph_num_probes),
                    generator=generator,
                    seed=int(config.graph_seed) + 1_000_003 * batch_idx + 10_000_019 * group_idx,
                )
            progress.set_postfix(endpoint_tokens=total_endpoint_tokens)
            model.zero_grad(set_to_none=True)
            collector.clear()
            del outputs, endpoints
            if device.type == "cuda":
                torch.cuda.empty_cache()

    missing_scores = sorted(name for name in params_by_name if normalizers.get(name, 0) == 0)
    if missing_scores:
        raise ValueError(f"local subgraph VJP produced no score for parameters: {missing_scores}")
    normalized_scores = {
        name: score.div(float(max(normalizers.get(name, 0), 1)))
        for name, score in scores.items()
    }
    return normalized_scores, total_supervised_tokens, len(groups), total_endpoint_tokens


def _linear_parameter_shapes(model: torch.nn.Module) -> dict[str, tuple[int, int]]:
    shapes: dict[str, tuple[int, int]] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.weight is not None and module.weight.requires_grad:
            shapes[_weight_name(module_name)] = tuple(module.weight.shape)
    return shapes


def matrix_parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad and param.ndim == 2]


def _score_bytes(scores: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in scores.values())


def _loss_gradient_repair_step_(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: IterativeApproxPruneConfig,
    device: torch.device,
    *,
    pruned_masks: dict[str, torch.Tensor],
    step: int,
) -> dict[str, object]:
    batches = list(batched(records, config.batch_size))
    model.zero_grad(set_to_none=True)
    total_loss_sum = 0.0
    supervised_tokens = 0
    for record_batch in tqdm(batches, desc=f"loss_gd_repair_{step}", unit="batch"):
        batch = build_causal_lm_batch(
            tokenizer,
            record_batch,
            config.max_length,
            answer_only_loss=config.answer_only_loss,
            device=device,
        )
        token_count = int((batch["labels"][:, 1:] != -100).sum().item())
        if token_count == 0:
            continue
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        loss = outputs.loss
        weighted_loss = loss * token_count
        weighted_loss.backward()
        total_loss_sum += float(weighted_loss.detach().float().item())
        supervised_tokens += token_count

    if supervised_tokens == 0:
        model.zero_grad(set_to_none=True)
        raise ValueError("loss GD repair saw no supervised tokens")

    for param in model.parameters():
        if param.grad is not None:
            param.grad.div_(float(supervised_tokens))

    update_summary = apply_masked_gradient_step_(
        model,
        pruned_masks=pruned_masks,
        learning_rate=config.repair_learning_rate,
        step=step,
    )
    model.zero_grad(set_to_none=True)
    return {
        **update_summary,
        "loss_sum": total_loss_sum,
        "supervised_tokens": supervised_tokens,
        "loss_per_token": total_loss_sum / max(supervised_tokens, 1),
        "num_batches": len(batches),
    }


def _accumulate_dfa_embedding_score_(
    score: torch.Tensor,
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    projected_error: torch.Tensor,
) -> None:
    grad = torch.zeros_like(score, device=projected_error.device)
    grad.index_add_(0, token_ids.to(projected_error.device), projected_error)
    score.add_((weight.detach().to(device=projected_error.device, dtype=torch.float32) * grad).abs().cpu())


def _dfa_gradcam_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    scores = _empty_matrix_scores(model)
    linear_shapes = _linear_parameter_shapes(model)
    matrix_params = {name: param for name, param in model.named_parameters() if param.requires_grad and param.ndim == 2}
    hidden_widths = sorted({shape[0] for name, shape in linear_shapes.items() if name != "embed_out.weight"})
    embed_in = matrix_params.get("gpt_neox.embed_in.weight")
    if embed_in is not None:
        hidden_widths.append(embed_in.shape[1])
        hidden_widths = sorted(set(hidden_widths))

    total_supervised_tokens = 0
    batches = list(batched(records, config.batch_size))
    progress = tqdm(batches, desc="dfa_gradcam", unit="batch")
    for record_batch in progress:
        batch = build_causal_lm_batch(
            tokenizer,
            record_batch,
            config.max_length,
            answer_only_loss=config.answer_only_loss,
            device=device,
        )
        with torch.inference_mode(), DfaActivationCollector(model) as collector:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
        residual, mask = cross_entropy_residual(outputs.logits, batch["labels"])
        supervised_tokens = int(mask.sum().item())
        total_supervised_tokens += supervised_tokens
        flat_mask = mask.reshape(-1)
        flat_mask_cpu = flat_mask.cpu()
        projections = {
            width: hash_project_residual(residual, out_features=width, seed=17)
            for width in hidden_widths
        }

        input_ids = batch["input_ids"][:, :-1].reshape(-1)[flat_mask]
        if embed_in is not None and "gpt_neox.embed_in.weight" in scores:
            _accumulate_dfa_embedding_score_(
                scores["gpt_neox.embed_in.weight"],
                embed_in,
                input_ids,
                projections[embed_in.shape[1]],
            )

        for name, shape in linear_shapes.items():
            param = matrix_params[name]
            captured = collector.inputs.get(name)
            if captured is None:
                continue
            if captured.ndim == 3 and captured.shape[:2] == batch["input_ids"].shape:
                activation_positions = captured[:, :-1, :]
            elif captured.ndim >= 2 and prod(captured.shape[:-1]) == mask.numel():
                activation_positions = captured
            else:
                continue
            if prod(activation_positions.shape[:-1]) != mask.numel():
                continue
            activation = activation_positions.reshape(-1, activation_positions.shape[-1])[flat_mask_cpu].to(
                device=device,
                dtype=torch.float32,
            )
            if name == "embed_out.weight":
                direct_error = residual
            else:
                direct_error = projections[shape[0]]
            grad = direct_error.transpose(0, 1).matmul(activation)
            scores[name].add_((param.detach().to(device=device, dtype=torch.float32) * grad).abs().cpu())

        del outputs, residual, projections, collector
        if device.type == "cuda":
            torch.cuda.empty_cache()
        progress.set_postfix(tokens=total_supervised_tokens)

    return {name: score.div(max(total_supervised_tokens, 1)) for name, score in scores.items()}, total_supervised_tokens


def _wanda_ablation_records_config(config: WandaAblationPruneConfig, *, split: str, max_examples: int) -> ApproxSaliencyConfig:
    return ApproxSaliencyConfig(
        output_dir=config.output_dir,
        model_name=config.model_name,
        method=config.wanda_method,
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=split,
        max_examples=max_examples,
        batch_size=config.batch_size,
        max_length=config.max_length,
        dtype=config.dtype,
        device=config.device,
        answer_only_loss=config.answer_only_loss,
        superset_gain_power=config.superset_gain_power,
        superset_gain_clip_quantile=config.superset_gain_clip_quantile,
        revision=config.revision,
    )


def _normalize_wanda_activation(value: str) -> bool:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"masked", "proper", "proper_wanda", "mask", "nonpadding", "non_padding"}:
        return True
    if normalized in {"unmasked", "original", "original_wanda", "legacy", "legacy_wanda", "include_padding"}:
        return False
    raise ValueError(f"unknown wanda_activation: {value}")


def _matrix_parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad and param.ndim == 2]


def _compact_pruning_summary(
    *,
    pruning_scope: str,
    prune_fraction: float,
    pruning_schedule: str,
    steps: list[dict[str, object]],
) -> dict[str, object]:
    tensors_seen = sum(int(step["matrix_tensors_seen"]) for step in steps)
    tensors_pruned = sum(int(step["matrix_tensors_pruned"]) for step in steps)
    weights_seen = sum(int(step["weights_seen"]) for step in steps)
    weights_zeroed = sum(int(step["weights_zeroed"]) for step in steps)
    missing: list[str] = []
    rows: list[dict[str, object]] = []
    for step in steps:
        missing.extend(str(name) for name in step["missing_saliency"])
        rows.extend(dict(row) for row in step["pruned_tensors"])
    return {
        "pruning_scope": pruning_scope,
        "pruning_schedule": pruning_schedule,
        "prune_fraction": prune_fraction,
        "matrix_tensors_seen": tensors_seen,
        "matrix_tensors_pruned": tensors_pruned,
        "weights_seen": weights_seen,
        "weights_zeroed": weights_zeroed,
        "actual_zero_fraction": weights_zeroed / max(weights_seen, 1),
        "missing_saliency": missing,
        "steps": steps,
        "pruned_tensors": rows,
    }


def apply_one_shot_wanda_pruning_(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
    *,
    use_attention_mask: bool,
    pruning_scope: str,
    fraction: float,
    score_method: str = "wanda",
) -> dict[str, object]:
    target_names = _matrix_parameter_names(model)
    scores = _wanda_ablation_scores(
        model,
        tokenizer,
        records,
        config,
        device,
        use_attention_mask=use_attention_mask,
        target_names=target_names,
        score_method=score_method,
    )
    step = apply_wanda_group_pruning_(
        model,
        scores,
        target_names=target_names,
        pruning_scope=pruning_scope,
        fraction=fraction,
        group_name="all_matrices",
        step=1,
    )
    summary = _compact_pruning_summary(
        pruning_scope=pruning_scope,
        prune_fraction=fraction,
        pruning_schedule="one_shot",
        steps=[step],
    )
    summary["score_method"] = score_method
    return summary


def apply_sequential_wanda_pruning_(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
    *,
    use_attention_mask: bool,
    pruning_scope: str,
    fraction: float,
    score_method: str = "wanda",
) -> dict[str, object]:
    steps: list[dict[str, object]] = []
    for step, (group_name, target_names) in enumerate(sequential_wanda_parameter_groups(model), start=1):
        scores = _wanda_ablation_scores(
            model,
            tokenizer,
            records,
            config,
            device,
            use_attention_mask=use_attention_mask,
            target_names=target_names,
            score_method=score_method,
        )
        steps.append(
            apply_wanda_group_pruning_(
                model,
                scores,
                target_names=target_names,
                pruning_scope=pruning_scope,
                fraction=fraction,
                group_name=group_name,
                step=step,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = _compact_pruning_summary(
        pruning_scope=pruning_scope,
        prune_fraction=fraction,
        pruning_schedule="sequential",
        steps=steps,
    )
    summary["score_method"] = score_method
    return summary


def apply_matrix_sequential_wanda_pruning_(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: ApproxSaliencyConfig,
    device: torch.device,
    *,
    use_attention_mask: bool,
    pruning_scope: str,
    fraction: float,
    score_method: str = "wanda",
) -> dict[str, object]:
    steps: list[dict[str, object]] = []
    for step, (group_name, target_names) in enumerate(sequential_wanda_matrix_parameter_groups(model), start=1):
        scores = _wanda_ablation_scores(
            model,
            tokenizer,
            records,
            config,
            device,
            use_attention_mask=use_attention_mask,
            target_names=target_names,
            score_method=score_method,
        )
        steps.append(
            apply_wanda_group_pruning_(
                model,
                scores,
                target_names=target_names,
                pruning_scope=pruning_scope,
                fraction=fraction,
                group_name=group_name,
                step=step,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = _compact_pruning_summary(
        pruning_scope=pruning_scope,
        prune_fraction=fraction,
        pruning_schedule="matrix_sequential",
        steps=steps,
    )
    summary["score_method"] = score_method
    return summary


def run_wanda_ablation_prune_ppl_experiment(config: WandaAblationPruneConfig) -> dict[str, object]:
    started = time.time()
    device = resolve_device(config.device)
    dtype = resolve_torch_dtype(config.dtype)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    use_attention_mask = _normalize_wanda_activation(config.wanda_activation)
    schedule = config.wanda_schedule.strip().lower().replace("-", "_")
    if schedule not in {
        "one_shot",
        "oneshot",
        "sequential",
        "layerwise",
        "layer_wise",
        "matrix_sequential",
        "matrixwise",
        "matrix_wise",
    }:
        raise ValueError(f"unknown wanda_schedule: {config.wanda_schedule}")
    if schedule in {"matrix_sequential", "matrixwise", "matrix_wise"}:
        normalized_schedule = "matrix_sequential"
    elif schedule in {"sequential", "layerwise", "layer_wise"}:
        normalized_schedule = "sequential"
    else:
        normalized_schedule = "one_shot"

    tokenizer = _prepare_tokenizer(config.model_name, config.revision)
    train_records = _load_records(
        _wanda_ablation_records_config(config, split=config.split, max_examples=config.max_examples)
    )
    if config.eval_split or config.max_eval_examples > 0:
        eval_records = _load_records(
            _wanda_ablation_records_config(
                config,
                split=config.eval_split or config.split,
                max_examples=config.max_eval_examples if config.max_eval_examples > 0 else config.max_examples,
            )
        )
    else:
        eval_records = train_records

    baseline_model = _prepare_model(config.model_name, config.revision, dtype, device)
    baseline = evaluate_perplexity(baseline_model, tokenizer, eval_records, config, device, desc="baseline_ppl")
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pruned_model = _prepare_model(config.model_name, config.revision, dtype, device)
    score_config = _wanda_ablation_records_config(config, split=config.split, max_examples=config.max_examples)
    if normalized_schedule == "matrix_sequential":
        pruning = apply_matrix_sequential_wanda_pruning_(
            pruned_model,
            tokenizer,
            train_records,
            score_config,
            device,
            use_attention_mask=use_attention_mask,
            pruning_scope=config.pruning_scope,
            fraction=config.prune_fraction,
            score_method=config.wanda_method,
        )
    elif normalized_schedule == "sequential":
        pruning = apply_sequential_wanda_pruning_(
            pruned_model,
            tokenizer,
            train_records,
            score_config,
            device,
            use_attention_mask=use_attention_mask,
            pruning_scope=config.pruning_scope,
            fraction=config.prune_fraction,
            score_method=config.wanda_method,
        )
    else:
        pruning = apply_one_shot_wanda_pruning_(
            pruned_model,
            tokenizer,
            train_records,
            score_config,
            device,
            use_attention_mask=use_attention_mask,
            pruning_scope=config.pruning_scope,
            fraction=config.prune_fraction,
            score_method=config.wanda_method,
        )
    pruned = evaluate_perplexity(pruned_model, tokenizer, eval_records, config, device, desc="pruned_ppl")

    result = {
        "metadata": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "device": str(device),
            "torch_dtype": str(dtype),
            "elapsed_seconds": time.time() - started,
            "calibration_examples": len(train_records),
            "eval_examples": len(eval_records),
            "wanda_activation_rule": "masked_non_padding" if use_attention_mask else "original_unmasked_including_padding",
            "wanda_method": config.wanda_method,
            "wanda_schedule_normalized": normalized_schedule,
            "wanda_ablation_rule": (
                "one_shot computes dense-model WANDA scores once before pruning; sequential recomputes WANDA on the "
                "current pruned model before each embedding/layer/head group and prunes that group immediately; "
                "matrix_sequential recomputes WANDA before each single weight matrix"
            ),
        },
        "pruning": pruning,
        "ppl_change": summarize_ppl_change(baseline, pruned),
    }

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "wanda_ablation_prune_ppl_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "pruned_tensors.jsonl").open("w") as handle:
        for row in pruning["pruned_tensors"]:
            handle.write(json.dumps(row) + "\n")
    return result


def run_approx_saliency_experiment(config: ApproxSaliencyConfig) -> dict[str, object]:
    started = time.time()
    device = resolve_device(config.device)
    dtype = resolve_torch_dtype(config.dtype)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model = _prepare_model(config.model_name, config.revision, dtype, device)
    method = config.method.lower()
    num_examples = 0
    num_batches = 0

    if method in {"magnitude", "weight_magnitude"}:
        scores = weight_magnitude_scores(model.named_parameters())
    elif method in {"ri", "ria_no_activation", "relative_importance_only"}:
        scores = relative_importance_scores(model.named_parameters())
    elif method in {"ria", "relative_importance", "relative_importance_activation"}:
        tokenizer = _prepare_tokenizer(config.model_name, config.revision)
        records = _load_records(config)
        num_examples = len(records)
        num_batches = len(list(batched(records, config.batch_size)))
        scores = _ria_scores(model, tokenizer, records, config, device)
    elif method in {
        "output_l2",
        "output_damage",
        "local_reconstruction",
        "squared_wanda",
        "mean_abs_wanda",
        "wanda_mean_abs",
        "mean_abs",
        "var_output",
        "variance_output",
        "q95_wanda",
        "wanda_q95",
        "outlier_q95",
        "max_wanda",
        "wanda_max",
        "outlier_max",
    }:
        tokenizer = _prepare_tokenizer(config.model_name, config.revision)
        records = _load_records(config)
        num_examples = len(records)
        num_batches = len(list(batched(records, config.batch_size)))
        scores = _input_activation_stat_scores(model, tokenizer, records, config, device, method)
    elif method in {
        "angular",
        "angular_exact",
        "pure_angular",
        "angular_approx",
        "approx_angular",
        "angular_hybrid",
        "hybrid_angular",
        "angular_energy_hybrid",
    }:
        tokenizer = _prepare_tokenizer(config.model_name, config.revision)
        records = _load_records(config)
        num_examples = len(records)
        num_batches = len(list(batched(records, config.batch_size)))
        scores = _angular_scores(model, tokenizer, records, config, device, method)
    elif method in {"wanda", "wanda_masked", "proper_wanda"}:
        tokenizer = _prepare_tokenizer(config.model_name, config.revision)
        records = _load_records(config)
        num_examples = len(records)
        num_batches = len(list(batched(records, config.batch_size)))
        scores = _wanda_scores(model, tokenizer, records, config, device)
    elif method in {"wanda_unmasked", "original_wanda", "legacy_wanda"}:
        tokenizer = _prepare_tokenizer(config.model_name, config.revision)
        records = _load_records(config)
        num_examples = len(records)
        num_batches = len(list(batched(records, config.batch_size)))
        scores = _wanda_scores(model, tokenizer, records, config, device, use_attention_mask=False)
    elif method in {"row_wanda", "token_wanda", "row_conditioned_wanda"}:
        tokenizer = _prepare_tokenizer(config.model_name, config.revision)
        records = _load_records(config)
        num_examples = len(records)
        num_batches = len(list(batched(records, config.batch_size)))
        scores = _row_conditioned_wanda_scores(model, tokenizer, records, config, device)
    elif method in {"feature_cosine_wanda", "feature_wanda_cosine", "row_wanda_cosine", "cosine_feature_wanda"}:
        tokenizer = _prepare_tokenizer(config.model_name, config.revision)
        records = _load_records(config)
        num_examples = len(records)
        num_batches = len(list(batched(records, config.batch_size)))
        scores = _feature_wanda_cosine_scores(model, tokenizer, records, config, device)
    elif method in {
        "graph_norm",
        "subgraph_norm",
        "residual_norm",
        "graph_qkv",
        "subgraph_qkv",
        "residual_norm_qkv",
        "graph_mlp",
        "subgraph_mlp",
        "residual_norm_mlp",
        "graph_qkv_mlp",
        "graph_next_projections",
        "subgraph_qkv_mlp",
    }:
        tokenizer = _prepare_tokenizer(config.model_name, config.revision)
        records = _load_records(config)
        num_examples = len(records)
        num_batches = len(list(batched(records, config.batch_size)))
        scores = _graph_propagated_scores(model, tokenizer, records, config, device, method)
    elif method in {"local_forward_wanda", "forward_subgraph_wanda", "local_wanda_diff", "subgraph_wanda_diff"}:
        tokenizer = _prepare_tokenizer(config.model_name, config.revision)
        records = _load_records(config)
        num_examples = len(records)
        num_batches = len(list(batched(records, config.batch_size)))
        scores = _local_forward_wanda_scores(model, tokenizer, records, config, device, method)
    elif method in SUPERSET_WANDA_METHODS:
        tokenizer = _prepare_tokenizer(config.model_name, config.revision)
        records = _load_records(config)
        num_examples = len(records)
        num_batches = len(list(batched(records, config.batch_size)))
        scores = _superset_wanda_scores(
            model,
            tokenizer,
            records,
            config,
            device,
            use_attention_mask=False,
        )
    elif method in {"graph_vjp_logits", "graph_logits", "subgraph_logits", "hutchinson_logits"}:
        tokenizer = _prepare_tokenizer(config.model_name, config.revision)
        records = _load_records(config)
        num_examples = len(records)
        num_batches = len(list(batched(records, config.batch_size)))
        scores, supervised_tokens = _graph_vjp_logits_scores(model, tokenizer, records, config, device)
    elif method in {"local_subgraph_vjp", "local_graph_vjp", "local_vjp", "local_subgraph_vjp_all_tokens", "local_graph_vjp_all_tokens", "local_vjp_all_tokens"}:
        tokenizer = _prepare_tokenizer(config.model_name, config.revision)
        records = _load_records(config)
        num_examples = len(records)
        num_batches = len(list(batched(records, config.batch_size)))
        scores, supervised_tokens, local_endpoint_groups, local_endpoint_tokens = _local_subgraph_vjp_scores(
            model,
            tokenizer,
            records,
            config,
            device,
            method,
        )
    elif method in {"dfa", "dfa_gradcam"}:
        tokenizer = _prepare_tokenizer(config.model_name, config.revision)
        records = _load_records(config)
        num_examples = len(records)
        num_batches = len(list(batched(records, config.batch_size)))
        scores, supervised_tokens = _dfa_gradcam_scores(model, tokenizer, records, config, device)
    else:
        raise ValueError(f"unknown approximation method: {config.method}")

    if method in {"magnitude", "weight_magnitude"}:
        saliency_method = "weight_magnitude: abs(weight) for trainable 2D parameters"
    elif method in {"ri", "ria_no_activation", "relative_importance_only"}:
        saliency_method = (
            "ri: abs(weight) / input-channel L1 + abs(weight) / output-channel L1 for trainable 2D parameters; "
            "no activation term"
        )
    elif method in {"ria", "relative_importance", "relative_importance_activation"}:
        saliency_method = (
            "ria: (abs(weight) / input-channel L1 + abs(weight) / output-channel L1) times "
            "per-input-column activation RMS^0.5 from calibration forward passes; trainable 2D "
            "non-Linear weights use relative importance without activation scaling"
        )
    elif method in {"output_l2", "output_damage", "local_reconstruction", "squared_wanda"}:
        saliency_method = "output_l2: weight^2 times per-input-column activation sumsq; non-Linear 2D weights fall back to weight^2"
    elif method in {"mean_abs_wanda", "wanda_mean_abs", "mean_abs"}:
        saliency_method = "mean_abs_wanda: abs(weight) times per-input-column mean(abs(activation)); non-Linear 2D weights fall back to abs(weight)"
    elif method in {"var_output", "variance_output"}:
        saliency_method = "var_output: weight^2 times per-input-column activation variance; non-Linear 2D weights fall back to weight^2"
    elif method in {"q95_wanda", "wanda_q95", "outlier_q95"}:
        saliency_method = "q95_wanda: abs(weight) times per-input-column nearest-rank q95(abs(activation)); non-Linear 2D weights fall back to abs(weight)"
    elif method in {"max_wanda", "wanda_max", "outlier_max"}:
        saliency_method = "max_wanda: abs(weight) times per-input-column max(abs(activation)); non-Linear 2D weights fall back to abs(weight)"
    elif method in {"angular", "angular_exact", "pure_angular"}:
        saliency_method = (
            "angular_exact: 1 - cos(y_i, y_i - w_ij x_j) using calibration Linear inputs x_j and outputs y_i; "
            "non-Linear 2D weights fall back to weight^2"
        )
    elif method in {"angular_approx", "approx_angular"}:
        saliency_method = (
            "angular_approx: weight^2 / ||y_i||^2 * (||x_j||^2 - (y_i^T x_j)^2 / ||y_i||^2); "
            "non-Linear 2D weights fall back to weight^2"
        )
    elif method in {"angular_hybrid", "hybrid_angular", "angular_energy_hybrid"}:
        lam = float(config.angular_hybrid_lambda)
        saliency_method = (
            f"angular_hybrid_lambda_{lam:g}: weight^2 / ||y_i||^2 * "
            f"(||x_j||^2 - {lam:g} * (y_i^T x_j)^2 / ||y_i||^2); non-Linear 2D weights fall back to weight^2"
        )
    elif method in {"dfa", "dfa_gradcam"}:
        saliency_method = (
            "dfa_gradcam: abs(weight * direct-feedback-alignment gradient); output CE residual is hash-projected "
            "to each linear output width and combined with captured forward activations; no backprop"
        )
    elif method in {"row_wanda", "token_wanda", "row_conditioned_wanda"}:
        saliency_method = (
            "row_wanda: abs(weight) times row-conditioned mean absolute input activation, "
            "computed as |linear_output|^T @ |linear_input| / sum(|linear_output|); "
            "trainable 2D non-Linear weights fall back to abs(weight)"
        )
    elif method in {"feature_cosine_wanda", "feature_wanda_cosine", "row_wanda_cosine", "cosine_feature_wanda"}:
        alpha = float(config.feature_cosine_alpha)
        clip = float(config.feature_cosine_clip)
        saliency_method = (
            f"feature_cosine_wanda_alpha_{alpha:g}_clip_{clip:g}: feature-WANDA base "
            "abs(weight) * (|linear_output|^T @ |linear_input| / sum(|linear_output|)) "
            "multiplied by 1 + alpha * normalized exact cosine damage; trainable 2D "
            "non-Linear weights fall back to abs(weight)"
        )
    elif method in {"graph_norm", "subgraph_norm", "residual_norm"}:
        saliency_method = (
            "graph_norm: for GPT-NeoX residual-output matrices attention.dense and dense_4h_to_h, score "
            "weight^2 times the exact LayerNorm-Jacobian propagated input energy through the next block's "
            "LayerNorm branches; all other trainable 2D weights use output_l2 fallback"
        )
    elif method in {"graph_qkv", "subgraph_qkv", "residual_norm_qkv"}:
        saliency_method = (
            f"graph_qkv_probes_{config.graph_num_probes}: for GPT-NeoX residual-output matrices attention.dense "
            "and dense_4h_to_h, score weight^2 times a Hutchinson estimate of propagated input energy through "
            "next input LayerNorm plus next QKV projection; final layer and other trainable 2D weights use "
            "output_l2/LayerNorm fallback"
        )
    elif method in {"graph_mlp", "subgraph_mlp", "residual_norm_mlp"}:
        saliency_method = (
            f"graph_mlp_probes_{config.graph_num_probes}: for GPT-NeoX residual-output matrices attention.dense "
            "and dense_4h_to_h, score weight^2 times a Hutchinson estimate of propagated input energy through "
            "next post-attention LayerNorm plus next MLP input projection; final layer and other trainable 2D "
            "weights use output_l2/LayerNorm fallback"
        )
    elif method in {"graph_qkv_mlp", "graph_next_projections", "subgraph_qkv_mlp"}:
        saliency_method = (
            f"graph_qkv_mlp_probes_{config.graph_num_probes}: for GPT-NeoX residual-output matrices attention.dense "
            "and dense_4h_to_h, score weight^2 times the summed Hutchinson propagated input energy through both "
            "next QKV and next MLP input projection branches; final layer and other trainable 2D weights use "
            "output_l2/LayerNorm fallback"
        )
    elif method in {"local_forward_wanda", "forward_subgraph_wanda", "local_wanda_diff", "subgraph_wanda_diff"}:
        saliency_method = (
            f"local_forward_wanda_eps_{config.local_forward_eps:g}: no-autograd local forward-diff WANDA. "
            "MLP-up weights use finite forward differences through GELU, residual-output and embed_in weights use "
            "finite forward differences through the next local LayerNorm endpoint, and QKV/embed_out use immediate "
            "linear endpoint damage; all trainable 2D parameters are covered with no output_l2 fallback"
        )
    elif method in SUPERSET_WANDA_METHODS:
        saliency_method = (
            "superset_wanda: closed-form local subgraph WANDA damage score "
            "W_ij^2 * sum_t x_tj^2 * ||J_F e_i||^2. MLP-up weights use activation Jacobian gains multiplied "
            "by downstream down-projection column norms; residual-output and embed_in weights use exact "
            "LayerNorm-to-next-linear column-norm gains; QKV uses the softmax attention Jacobian pushed through "
            "the attention output projection; embed_out uses immediate linear endpoint damage; "
            "default artifact scoring includes padding positions to match the best unmasked WANDA variant"
        )
    elif method in {"graph_vjp_logits", "graph_logits", "subgraph_logits", "hutchinson_logits"}:
        mask_desc = "answer-token" if config.answer_only_loss else "non-padding"
        saliency_method = (
            f"graph_vjp_logits_probes_{config.graph_num_probes}: all trainable 2D parameters scored as "
            "mean Hutchinson estimate of full downstream logits-endpoint damage, using "
            f"(parameter * d(random_logits_projection)/dparameter)^2 on {mask_desc} positions; no Linear-weight "
            "output_l2 fallback"
        )
    elif method in {"local_subgraph_vjp", "local_graph_vjp", "local_vjp", "local_subgraph_vjp_all_tokens", "local_graph_vjp_all_tokens", "local_vjp_all_tokens"}:
        mask_desc = "all non-padding next-token positions" if "all_tokens" in method else "answer-token-aligned positions"
        saliency_method = (
            f"local_subgraph_vjp_probes_{config.graph_num_probes}: all trainable 2D parameters scored by local "
            "one-block Hutchinson VJPs. Attention-context endpoints score same-layer QKV and previous residual "
            "outputs, MLP-activation endpoints score same-layer MLP-up and previous residual outputs, final norm "
            f"scores final residual outputs, and logits score only embed_out; endpoint mask uses {mask_desc}; "
            "no output_l2 fallback"
        )
    elif method in {"wanda_unmasked", "original_wanda", "legacy_wanda"}:
        saliency_method = (
            "wanda_unmasked: legacy abs(weight) times per-input-column activation RMS over all calibration "
            "positions including padding; trainable 2D non-Linear weights fall back to abs(weight); saliency "
            "artifacts are dense-model scores and later pruning is not layerwise sequential WANDA"
        )
    else:
        saliency_method = (
            "wanda: abs(weight) times per-input-column activation RMS over non-padding calibration positions; "
            "trainable 2D non-Linear weights fall back to abs(weight); saliency artifacts are dense-model scores "
            "and later pruning is not layerwise sequential WANDA"
        )

    metadata = {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "device": str(device),
        "torch_dtype": str(dtype),
        "num_examples": num_examples,
        "num_batches": num_batches,
        "supervised_tokens": locals().get("supervised_tokens", 0),
        "local_endpoint_groups": locals().get("local_endpoint_groups", 0),
        "local_endpoint_tokens": locals().get("local_endpoint_tokens", 0),
        "score_bytes": _score_bytes(scores),
        "elapsed_seconds": time.time() - started,
        "saliency_method": saliency_method,
    }
    return save_saliency_artifacts(config.output_dir, scores, metadata, top_k=config.top_k)


def run_iterative_approx_prune_ppl_experiment(config: IterativeApproxPruneConfig) -> dict[str, object]:
    started = time.time()
    method = config.method.lower()
    if method != "wanda":
        raise ValueError("iterative approximation pruning currently supports method='wanda'")
    if config.repair_with_gptq_gd:
        raise ValueError("repair_with_gptq_gd used the wrong objective; use repair_with_loss_gd for pruning repair")
    pruning_structure = normalize_pruning_structure(config.pruning_structure)

    device = resolve_device(config.device)
    dtype = resolve_torch_dtype(config.dtype)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    tokenizer = _prepare_tokenizer(config.model_name, config.revision)
    records = _load_records(
        ApproxSaliencyConfig(
            output_dir=config.output_dir,
            model_name=config.model_name,
            method=config.method,
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            split=config.split,
            max_examples=config.max_examples,
            batch_size=config.batch_size,
            max_length=config.max_length,
            dtype=config.dtype,
            device=config.device,
            answer_only_loss=config.answer_only_loss,
            revision=config.revision,
        )
    )
    if config.eval_split or config.max_eval_examples > 0:
        eval_records = _load_records(
            ApproxSaliencyConfig(
                output_dir=config.output_dir,
                model_name=config.model_name,
                method=config.method,
                dataset_name=config.dataset_name,
                dataset_config=config.dataset_config,
                split=config.eval_split or config.split,
                max_examples=config.max_eval_examples if config.max_eval_examples > 0 else config.max_examples,
                batch_size=config.batch_size,
                max_length=config.max_length,
                dtype=config.dtype,
                device=config.device,
                answer_only_loss=config.answer_only_loss,
                revision=config.revision,
            )
        )
    else:
        eval_records = records

    baseline_model = _prepare_model(config.model_name, config.revision, dtype, device)
    baseline = evaluate_perplexity(baseline_model, tokenizer, eval_records, config, device, desc="baseline_ppl")
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pruned_model = _prepare_model(config.model_name, config.revision, dtype, device)
    total_matrix_weights = matrix_weight_count(pruned_model)
    if pruning_structure == "nm":
        if config.structured_group_dim != 1:
            raise ValueError("2:4 structured pruning requires structured_group_dim=1")
        if not 0 < config.structured_n < config.structured_m:
            raise ValueError("structured_n and structured_m must satisfy 0 < n < m")
        effective_target_fraction = config.structured_n / config.structured_m
        chunk_fraction = effective_target_fraction / config.structured_n
        max_structured_steps = config.structured_n
    else:
        effective_target_fraction = config.prune_fraction
        chunk_fraction = resolve_prune_chunk_fraction(
            matrix_weights=total_matrix_weights,
            prune_chunk_fraction=config.prune_chunk_fraction,
            recompute_every_weights=config.recompute_every_weights,
        )
        if chunk_fraction <= 0.0:
            raise ValueError("resolved prune chunk fraction must be positive")
        max_structured_steps = 0

    pruned_masks: dict[str, torch.Tensor] = {}
    step_summaries: list[dict[str, object]] = []
    repair_summaries: list[dict[str, object]] = []
    step = 0
    while True:
        current_zeroed = sum(int(mask.sum().item()) for mask in pruned_masks.values())
        if current_zeroed >= int(total_matrix_weights * effective_target_fraction):
            break
        step += 1
        if pruning_structure == "nm" and step > max_structured_steps:
            break
        scores, _, _ = _wanda_scores_and_hessian_diagonal(
            pruned_model,
            tokenizer,
            records,
            config,
            device,
        )
        if pruning_structure == "nm":
            step_summary = apply_incremental_nm_pruning_(
                pruned_model,
                scores,
                pruned_masks=pruned_masks,
                n=config.structured_n,
                m=config.structured_m,
                target_zeros_per_group=min(config.structured_n, step),
                group_dim=config.structured_group_dim,
                step=step,
            )
        else:
            step_target_fraction = min(config.prune_fraction, step * chunk_fraction)
            step_summary = apply_incremental_per_matrix_pruning_(
                pruned_model,
                scores,
                pruned_masks=pruned_masks,
                target_fraction=step_target_fraction,
                chunk_fraction=chunk_fraction,
                step=step,
            )
        step_summaries.append(step_summary)
        if config.repair_with_loss_gd and step_summary["weights_zeroed_this_step"] > 0:
            repair_summaries.append(
                _loss_gradient_repair_step_(
                    pruned_model,
                    tokenizer,
                    records,
                    config,
                    device,
                    pruned_masks=pruned_masks,
                    step=step,
                )
            )
        del scores
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if step_summary["target_reached"] or step_summary["weights_zeroed_this_step"] == 0:
            break

    pruned = evaluate_perplexity(pruned_model, tokenizer, eval_records, config, device, desc="pruned_ppl")
    final_zeroed = sum(int(mask.sum().item()) for mask in pruned_masks.values())
    final_pruned_tensors = [
        {
            "name": name,
            "shape": list(mask.shape),
            "weights": mask.numel(),
            "zeroed": int(mask.sum().item()),
        }
        for name, mask in sorted(pruned_masks.items())
    ]
    compact_step_summaries = [
        {key: value for key, value in row.items() if key not in {"pruned_tensors", "missing_saliency"}}
        for row in step_summaries
    ]
    compact_rows = [
        {key: value for key, value in row.items() if key not in {"pruning_steps", "repair_steps"}}
        for row in rows
    ]
    result = {
        "metadata": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "device": str(device),
            "torch_dtype": str(dtype),
            "elapsed_seconds": time.time() - started,
            "calibration_examples": len(records),
            "eval_examples": len(eval_records),
            "eval_split_resolved": config.eval_split or config.split,
            "resolved_prune_chunk_fraction": chunk_fraction,
            "resolved_recompute_every_weights": int(total_matrix_weights * chunk_fraction),
            "repair_with_loss_gd": config.repair_with_loss_gd,
            "repair_learning_rate": config.repair_learning_rate,
            "pruning_rule": (
                (
                    f"native row-wise {config.structured_n}:{config.structured_m} WANDA: for every row and contiguous input-dimension "
                    f"group of {config.structured_m}, recompute abs(weight) * input_activation_rms on the current pruned model, "
                    "zero the next lowest-score remaining entry per group until the hardware quartet target is reached, "
                    "use no column permutation, optionally apply one standard loss-gradient descent step on calibration LM loss "
                    "after each pruning chunk, and reapply prune masks"
                )
                if pruning_structure == "nm"
                else (
                    "iterative per-matrix WANDA: recompute abs(weight) * input_activation_rms on the current pruned model, "
                    "then zero the next lowest-score chunk in each 2D matrix until target_fraction is reached; "
                    "optionally apply one standard loss-gradient descent step on calibration LM loss after each pruning chunk "
                    "and reapply prune masks"
                )
            ),
        },
        "pruning": {
            "pruning_scope": f"{config.structured_n}:{config.structured_m}_semi_structured"
            if pruning_structure == "nm"
            else "per_matrix_iterative",
            "method": method,
            "target_fraction": effective_target_fraction,
            "requested_prune_fraction": config.prune_fraction,
            "chunk_fraction": chunk_fraction,
            "pruning_structure": pruning_structure,
            "structured_n": config.structured_n if pruning_structure == "nm" else None,
            "structured_m": config.structured_m if pruning_structure == "nm" else None,
            "structured_group_dim": config.structured_group_dim if pruning_structure == "nm" else None,
            "steps": len(step_summaries),
            "weights_seen": total_matrix_weights,
            "weights_zeroed": final_zeroed,
            "actual_zero_fraction": final_zeroed / max(total_matrix_weights, 1),
            "pruned_tensors": final_pruned_tensors,
            "step_summaries": compact_step_summaries,
        },
        "repair": {
            "enabled": config.repair_with_loss_gd,
            "method": "one_step_loss_gradient_descent" if config.repair_with_loss_gd else "none",
            "learning_rate": config.repair_learning_rate,
            "step_summaries": repair_summaries,
        },
        "ppl_change": summarize_ppl_change(baseline, pruned),
    }

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "iterative_prune_ppl_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "iterative_pruning_steps.jsonl").open("w") as handle:
        for row in step_summaries:
            handle.write(json.dumps(row) + "\n")
    with (out / "repair_steps.jsonl").open("w") as handle:
        for row in repair_summaries:
            handle.write(json.dumps(row) + "\n")
    with (out / "pruned_tensors.jsonl").open("w") as handle:
        for row in final_pruned_tensors:
            handle.write(json.dumps(row) + "\n")
    return result


def run_nm_matrix_attribution_experiment(config: IterativeApproxPruneConfig) -> dict[str, object]:
    started = time.time()
    method = config.method.lower()
    if method != "wanda":
        raise ValueError("N:M matrix attribution currently supports method='wanda'")
    if config.repair_with_gptq_gd:
        raise ValueError("repair_with_gptq_gd used the wrong objective; use repair_with_loss_gd for pruning repair")
    if normalize_pruning_structure(config.pruning_structure) != "nm":
        raise ValueError("N:M matrix attribution requires pruning_structure='2:4' or another N:M alias")
    if config.structured_group_dim != 1:
        raise ValueError("N:M matrix attribution requires structured_group_dim=1")
    if not 0 < config.structured_n < config.structured_m:
        raise ValueError("structured_n and structured_m must satisfy 0 < n < m")

    device = resolve_device(config.device)
    dtype = resolve_torch_dtype(config.dtype)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    tokenizer = _prepare_tokenizer(config.model_name, config.revision)
    records = _load_records(
        ApproxSaliencyConfig(
            output_dir=config.output_dir,
            model_name=config.model_name,
            method=config.method,
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            split=config.split,
            max_examples=config.max_examples,
            batch_size=config.batch_size,
            max_length=config.max_length,
            dtype=config.dtype,
            device=config.device,
            answer_only_loss=config.answer_only_loss,
            revision=config.revision,
        )
    )
    eval_records = _load_records(
        ApproxSaliencyConfig(
            output_dir=config.output_dir,
            model_name=config.model_name,
            method=config.method,
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            split=config.eval_split or config.split,
            max_examples=config.max_eval_examples if config.max_eval_examples > 0 else config.max_examples,
            batch_size=config.batch_size,
            max_length=config.max_length,
            dtype=config.dtype,
            device=config.device,
            answer_only_loss=config.answer_only_loss,
            revision=config.revision,
        )
    )

    baseline_model = _prepare_model(config.model_name, config.revision, dtype, device)
    baseline = evaluate_perplexity(baseline_model, tokenizer, eval_records, config, device, desc="baseline_ppl")
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = _prepare_model(config.model_name, config.revision, dtype, device)
    names = matrix_parameter_names(model)
    if config.matrix_limit > 0:
        names = names[: config.matrix_limit]
    total_matrix_weights = matrix_weight_count(model)
    pruned_masks: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    previous_ppl = baseline.perplexity
    final_ppl = baseline

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "matrix_attribution.jsonl"
    with rows_path.open("w") as rows_handle:
        for matrix_index, name in enumerate(tqdm(names, desc="matrix_attribution", unit="matrix"), start=1):
            matrix_step_summaries = []
            matrix_repair_summaries = []
            for target_zeros_per_group in range(1, config.structured_n + 1):
                scores, _, _ = _wanda_scores_and_hessian_diagonal(model, tokenizer, records, config, device)
                step = (matrix_index - 1) * config.structured_n + target_zeros_per_group
                step_summary = apply_incremental_nm_pruning_to_parameter_(
                    model,
                    scores,
                    pruned_masks=pruned_masks,
                    parameter_name=name,
                    n=config.structured_n,
                    m=config.structured_m,
                    target_zeros_per_group=target_zeros_per_group,
                    group_dim=config.structured_group_dim,
                    step=step,
                )
                matrix_step_summaries.append(step_summary)
                if config.repair_with_loss_gd and step_summary["zeroed_this_step"] > 0:
                    matrix_repair_summaries.append(
                        _loss_gradient_repair_step_(
                            model,
                            tokenizer,
                            records,
                            config,
                            device,
                            pruned_masks=pruned_masks,
                            step=step,
                        )
                    )
                del scores
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            ppl = evaluate_perplexity(model, tokenizer, eval_records, config, device, desc=f"ppl_after_matrix_{matrix_index}")
            final_ppl = ppl
            ppl_dict = ppl.to_dict()
            cumulative_zeroed = sum(int(mask.sum().item()) for mask in pruned_masks.values())
            row = {
                "matrix_index": matrix_index,
                "name": name,
                "shape": matrix_step_summaries[-1]["shape"],
                "weights": matrix_step_summaries[-1]["weights"],
                "matrix_zeroed": matrix_step_summaries[-1]["zeroed_total"],
                "matrix_zero_fraction": matrix_step_summaries[-1]["actual_zero_fraction"],
                "cumulative_zeroed": cumulative_zeroed,
                "cumulative_zero_fraction": cumulative_zeroed / max(total_matrix_weights, 1),
                "ppl": ppl_dict,
                "delta_perplexity_from_baseline": ppl.perplexity - baseline.perplexity,
                "marginal_delta_perplexity": ppl.perplexity - previous_ppl,
                "delta_loss_per_token_from_baseline": ppl.loss_per_token - baseline.loss_per_token,
                "pruning_steps": matrix_step_summaries,
                "repair_steps": matrix_repair_summaries,
            }
            previous_ppl = ppl.perplexity
            rows.append(row)
            rows_handle.write(json.dumps(row) + "\n")
            rows_handle.flush()

    compact_rows = [
        {key: value for key, value in row.items() if key not in {"pruning_steps", "repair_steps"}}
        for row in rows
    ]
    result = {
        "metadata": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "device": str(device),
            "torch_dtype": str(dtype),
            "elapsed_seconds": time.time() - started,
            "calibration_examples": len(records),
            "eval_examples": len(eval_records),
            "eval_split_resolved": config.eval_split or config.split,
            "matrix_count": len(names),
            "total_matrix_weights": total_matrix_weights,
            "pruning_rule": (
                f"cumulative matrix attribution for native row-wise {config.structured_n}:{config.structured_m} WANDA with no column "
                "permutation: for each trainable 2D matrix in model parameter order, recompute WANDA, prune one lowest-saliency "
                "entry per native quartet, run masked loss-GD repair if enabled, recompute WANDA, prune the second entry per "
                "quartet, repair again, then evaluate heldout PPL and record the marginal PPL change"
            ),
        },
        "baseline": baseline.to_dict(),
        "final": final_ppl.to_dict(),
        "ppl_change": summarize_ppl_change(baseline, final_ppl),
        "top_marginal_perplexity_increases": sorted(
            compact_rows,
            key=lambda row: float(row["marginal_delta_perplexity"]),
            reverse=True,
        )[:20],
        "rows_path": str(rows_path),
        "rows": compact_rows,
    }
    (out / "matrix_attribution_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_nm_global_pass_matrix_attribution_experiment(config: IterativeApproxPruneConfig) -> dict[str, object]:
    started = time.time()
    method = config.method.lower()
    if method != "wanda":
        raise ValueError("N:M global-pass matrix attribution currently supports method='wanda'")
    if config.repair_with_gptq_gd:
        raise ValueError("repair_with_gptq_gd used the wrong objective; use repair_with_loss_gd for pruning repair")
    if normalize_pruning_structure(config.pruning_structure) != "nm":
        raise ValueError("N:M global-pass matrix attribution requires pruning_structure='2:4' or another N:M alias")
    if config.structured_group_dim != 1:
        raise ValueError("N:M global-pass matrix attribution requires structured_group_dim=1")
    if not 0 < config.structured_n < config.structured_m:
        raise ValueError("structured_n and structured_m must satisfy 0 < n < m")

    device = resolve_device(config.device)
    dtype = resolve_torch_dtype(config.dtype)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    tokenizer = _prepare_tokenizer(config.model_name, config.revision)
    records = _load_records(
        ApproxSaliencyConfig(
            output_dir=config.output_dir,
            model_name=config.model_name,
            method=config.method,
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            split=config.split,
            max_examples=config.max_examples,
            batch_size=config.batch_size,
            max_length=config.max_length,
            dtype=config.dtype,
            device=config.device,
            answer_only_loss=config.answer_only_loss,
            revision=config.revision,
        )
    )
    eval_records = _load_records(
        ApproxSaliencyConfig(
            output_dir=config.output_dir,
            model_name=config.model_name,
            method=config.method,
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            split=config.eval_split or config.split,
            max_examples=config.max_eval_examples if config.max_eval_examples > 0 else config.max_examples,
            batch_size=config.batch_size,
            max_length=config.max_length,
            dtype=config.dtype,
            device=config.device,
            answer_only_loss=config.answer_only_loss,
            revision=config.revision,
        )
    )

    model = _prepare_model(config.model_name, config.revision, dtype, device)
    names = matrix_parameter_names(model)
    if config.matrix_limit > 0:
        names = names[: config.matrix_limit]
    total_matrix_weights = matrix_weight_count(model)
    pruned_masks: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    repair_rows: list[dict[str, object]] = []

    baseline = evaluate_perplexity(model, tokenizer, eval_records, config, device, desc="baseline_ppl")
    previous_ppl = baseline.perplexity
    final_ppl = baseline

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "global_pass_matrix_attribution.jsonl"
    repair_path = out / "global_pass_repair_checkpoints.jsonl"
    with rows_path.open("w") as rows_handle, repair_path.open("w") as repair_handle:
        for pass_index in range(1, config.structured_n + 1):
            scores, _, _ = _wanda_scores_and_hessian_diagonal(model, tokenizer, records, config, device)
            for matrix_index, name in enumerate(
                tqdm(names, desc=f"global_pass_{pass_index}_matrix_attribution", unit="matrix"),
                start=1,
            ):
                step = (pass_index - 1) * max(len(names), 1) + matrix_index
                step_summary = apply_incremental_nm_pruning_to_parameter_(
                    model,
                    scores,
                    pruned_masks=pruned_masks,
                    parameter_name=name,
                    n=config.structured_n,
                    m=config.structured_m,
                    target_zeros_per_group=pass_index,
                    group_dim=config.structured_group_dim,
                    step=step,
                )
                ppl = evaluate_perplexity(
                    model,
                    tokenizer,
                    eval_records,
                    config,
                    device,
                    desc=f"ppl_after_pass_{pass_index}_matrix_{matrix_index}",
                )
                final_ppl = ppl
                cumulative_zeroed = sum(int(mask.sum().item()) for mask in pruned_masks.values())
                row = {
                    "pass_index": pass_index,
                    "matrix_index": matrix_index,
                    "global_matrix_step": step,
                    "name": name,
                    "shape": step_summary["shape"],
                    "weights": step_summary["weights"],
                    "target_zeros_per_group": pass_index,
                    "matrix_zeroed": step_summary["zeroed_total"],
                    "matrix_zero_fraction": step_summary["actual_zero_fraction"],
                    "cumulative_zeroed": cumulative_zeroed,
                    "cumulative_zero_fraction": cumulative_zeroed / max(total_matrix_weights, 1),
                    "ppl": ppl.to_dict(),
                    "delta_perplexity_from_baseline": ppl.perplexity - baseline.perplexity,
                    "marginal_delta_perplexity": ppl.perplexity - previous_ppl,
                    "delta_loss_per_token_from_baseline": ppl.loss_per_token - baseline.loss_per_token,
                    "pruning_step": step_summary,
                }
                previous_ppl = ppl.perplexity
                rows.append(row)
                rows_handle.write(json.dumps(row) + "\n")
                rows_handle.flush()

            del scores
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if config.repair_with_loss_gd:
                repair_summary = _loss_gradient_repair_step_(
                    model,
                    tokenizer,
                    records,
                    config,
                    device,
                    pruned_masks=pruned_masks,
                    step=pass_index,
                )
                ppl = evaluate_perplexity(
                    model,
                    tokenizer,
                    eval_records,
                    config,
                    device,
                    desc=f"ppl_after_pass_{pass_index}_repair",
                )
                final_ppl = ppl
                cumulative_zeroed = sum(int(mask.sum().item()) for mask in pruned_masks.values())
                repair_row = {
                    "pass_index": pass_index,
                    "cumulative_zeroed": cumulative_zeroed,
                    "cumulative_zero_fraction": cumulative_zeroed / max(total_matrix_weights, 1),
                    "ppl": ppl.to_dict(),
                    "delta_perplexity_from_baseline": ppl.perplexity - baseline.perplexity,
                    "marginal_delta_perplexity": ppl.perplexity - previous_ppl,
                    "delta_loss_per_token_from_baseline": ppl.loss_per_token - baseline.loss_per_token,
                    "repair_step": repair_summary,
                }
                previous_ppl = ppl.perplexity
                repair_rows.append(repair_row)
                repair_handle.write(json.dumps(repair_row) + "\n")
                repair_handle.flush()

    compact_rows = [{key: value for key, value in row.items() if key != "pruning_step"} for row in rows]
    compact_repairs = [{key: value for key, value in row.items() if key != "repair_step"} for row in repair_rows]
    result = {
        "metadata": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "device": str(device),
            "torch_dtype": str(dtype),
            "elapsed_seconds": time.time() - started,
            "calibration_examples": len(records),
            "eval_examples": len(eval_records),
            "eval_split_resolved": config.eval_split or config.split,
            "matrix_count": len(names),
            "total_matrix_weights": total_matrix_weights,
            "pruning_rule": (
                f"original-cadence native row-wise {config.structured_n}:{config.structured_m} WANDA matrix attribution with no "
                "column permutation: recompute WANDA once per global structured pass, prune one additional lowest-saliency "
                "entry per quartet one matrix at a time, evaluate heldout PPL after each matrix, then run the single masked "
                "loss-GD repair step at the end of the pass if enabled"
            ),
        },
        "baseline": baseline.to_dict(),
        "final": final_ppl.to_dict(),
        "ppl_change": summarize_ppl_change(baseline, final_ppl),
        "top_marginal_perplexity_increases": sorted(
            compact_rows,
            key=lambda row: float(row["marginal_delta_perplexity"]),
            reverse=True,
        )[:20],
        "repair_checkpoints": compact_repairs,
        "rows_path": str(rows_path),
        "repair_path": str(repair_path),
        "rows": compact_rows,
    }
    (out / "global_pass_matrix_attribution_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
