from __future__ import annotations

import argparse
import gc
import dataclasses
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .experiment import _torch_dtype


@dataclass(frozen=True)
class AttentionAnalysisConfig:
    output_dir: Path
    small_model: str = "EleutherAI/pythia-31m"
    big_model: str = "EleutherAI/pythia-70m"
    prompts: tuple[str, ...] = (
        "In a small gridworld, the agent starts at the red square and must reach the blue square while avoiding walls.",
        "Question: If Alice has three keys and gives Bob one key, how many keys does Alice still have? Answer:",
        "A Python function receives a list of integers and returns the largest even number in the list.",
    )
    max_length: int = 96
    dtype: str = "fp32"
    device: str = "auto"
    local_window: int = 8
    save_tensors: bool = False

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "prompts", tuple(str(prompt) for prompt in self.prompts))
        if not self.prompts:
            raise ValueError("prompts must be non-empty")
        if self.max_length <= 1:
            raise ValueError("max_length must be greater than 1")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be bf16, fp16, or fp32")
        if self.local_window <= 0:
            raise ValueError("local_window must be positive")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")


def _normalize(attn: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    values = attn.float().clamp_min(0.0)
    denom = values.sum(dim=-1, keepdim=True)
    return values / denom.clamp_min(eps)


def attention_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"attention shapes must match: {tuple(a.shape)} != {tuple(b.shape)}")
    p = _normalize(a, eps)
    q = _normalize(b, eps)
    m = 0.5 * (p + q)
    kl_pm = (p * ((p + eps).log() - (m + eps).log())).sum(dim=-1)
    kl_qm = (q * ((q + eps).log() - (m + eps).log())).sum(dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm)
    tv = 0.5 * (p - q).abs().sum(dim=-1)
    p_flat = p.reshape(-1)
    q_flat = q.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(p_flat, q_flat, dim=0, eps=eps)
    return {
        "jsd": float(jsd.mean().item()),
        "tv": float(tv.mean().item()),
        "cosine_distance": float((1.0 - cosine).item()),
    }


def head_summary(attn: torch.Tensor, *, local_window: int = 8, eps: float = 1e-12) -> dict[str, float]:
    probs = _normalize(attn, eps)
    entropy_per_query = -(probs * (probs + eps).log()).sum(dim=-1)
    nonzero = (probs > eps).sum(dim=-1).float()
    norm_denom = nonzero.clamp_min(2.0).log()
    normalized_entropy = torch.where(nonzero > 1, entropy_per_query / norm_denom, torch.zeros_like(entropy_per_query))

    seq_len = probs.shape[-1]
    diag = probs.diagonal(dim1=-2, dim2=-1)
    query_idx = torch.arange(seq_len, device=probs.device).view(1, seq_len, 1)
    key_idx = torch.arange(seq_len, device=probs.device).view(1, 1, seq_len)
    local_mask = (key_idx <= query_idx) & (key_idx >= query_idx - local_window + 1)
    local_mass = probs.masked_fill(~local_mask, 0.0).sum(dim=-1)

    return {
        "entropy": float(entropy_per_query.mean().item()),
        "normalized_entropy": float(normalized_entropy.mean().item()),
        "max_prob": float(probs.max(dim=-1).values.mean().item()),
        "diagonal_mass": float(diag.mean().item()),
        f"local_{local_window}_mass": float(local_mass.mean().item()),
    }


def compare_within_model(model_name: str, attentions: list[torch.Tensor], *, prompt_index: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_idx, layer_attn in enumerate(attentions):
        heads = layer_attn.shape[1]
        for head_a in range(heads):
            for head_b in range(head_a + 1, heads):
                metrics = attention_distance(layer_attn[:, head_a], layer_attn[:, head_b])
                rows.append(
                    {
                        "prompt_index": prompt_index,
                        "model": model_name,
                        "layer": layer_idx,
                        "head_a": head_a,
                        "head_b": head_b,
                        **metrics,
                    }
                )
    return rows


def compare_within_model_head_matching(model_name: str, attentions: list[torch.Tensor], *, prompt_index: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_idx, layer_attn in enumerate(attentions):
        heads = layer_attn.shape[1]
        pair_metrics: dict[tuple[int, int], dict[str, float]] = {}
        for head_a in range(heads):
            for head_b in range(head_a + 1, heads):
                metrics = attention_distance(layer_attn[:, head_a], layer_attn[:, head_b])
                pair_metrics[(head_a, head_b)] = metrics
                rows.append(
                    {
                        "prompt_index": prompt_index,
                        "model": model_name,
                        "layer": layer_idx,
                        "head_a": head_a,
                        "head_b": head_b,
                        "match_type": "all_pairs",
                        **metrics,
                    }
                )
        if heads < 2:
            continue

        cost_matrix = [[0.0] * heads for _ in range(heads)]
        for head_a in range(heads):
            for head_b in range(heads):
                if head_a == head_b:
                    cost_matrix[head_a][head_b] = 1e12
                else:
                    pair = (head_a, head_b) if head_a < head_b else (head_b, head_a)
                    cost_matrix[head_a][head_b] = pair_metrics[pair]["jsd"]

        for head, costs in enumerate(cost_matrix):
            matched_head = min((idx for idx in range(heads) if idx != head), key=lambda idx: costs[idx])
            pair = (head, matched_head) if head < matched_head else (matched_head, head)
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "model": model_name,
                    "layer": layer_idx,
                    "head": head,
                    "matched_head": matched_head,
                    "match_type": "head_to_best_head",
                    **pair_metrics[pair],
                }
            )

        if heads <= 64:
            for head, matched_head in _rectangular_assignment_pairs(cost_matrix):
                pair = (head, matched_head) if head < matched_head else (matched_head, head)
                rows.append(
                    {
                        "prompt_index": prompt_index,
                        "model": model_name,
                        "layer": layer_idx,
                        "head": head,
                        "matched_head": matched_head,
                        "match_type": "one_to_one",
                        **pair_metrics[pair],
                    }
                )
    return rows


def compare_cross_model(
    small_name: str,
    small_attentions: list[torch.Tensor],
    big_name: str,
    big_attentions: list[torch.Tensor],
    *,
    prompt_index: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layer_count = min(len(small_attentions), len(big_attentions))
    for layer_idx in range(layer_count):
        small_layer = small_attentions[layer_idx]
        big_layer = big_attentions[layer_idx]
        head_count = min(small_layer.shape[1], big_layer.shape[1])
        for head in range(head_count):
            metrics = attention_distance(small_layer[:, head], big_layer[:, head])
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "small_model": small_name,
                    "big_model": big_name,
                    "layer": layer_idx,
                    "head": head,
                    **metrics,
                }
            )
    return rows


def compare_cross_model_head_matching(
    small_name: str,
    small_attentions: list[torch.Tensor],
    big_name: str,
    big_attentions: list[torch.Tensor],
    *,
    prompt_index: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layer_count = min(len(small_attentions), len(big_attentions))
    for layer_idx in range(layer_count):
        small_layer = small_attentions[layer_idx]
        big_layer = big_attentions[layer_idx]
        small_heads = small_layer.shape[1]
        big_heads = big_layer.shape[1]
        matrix: list[list[dict[str, float]]] = []
        for small_head in range(small_heads):
            row_metrics = []
            for big_head in range(big_heads):
                metrics = attention_distance(small_layer[:, small_head], big_layer[:, big_head])
                row_metrics.append(metrics)
                rows.append(
                    {
                        "prompt_index": prompt_index,
                        "small_model": small_name,
                        "big_model": big_name,
                        "layer": layer_idx,
                        "small_head": small_head,
                        "big_head": big_head,
                        "match_type": "all_pairs",
                        **metrics,
                    }
                )
            matrix.append(row_metrics)

        for small_head, row_metrics in enumerate(matrix):
            big_head = min(range(big_heads), key=lambda idx: row_metrics[idx]["jsd"])
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "small_model": small_name,
                    "big_model": big_name,
                    "layer": layer_idx,
                    "small_head": small_head,
                    "big_head": big_head,
                    "match_type": "small_to_best_big",
                    **row_metrics[big_head],
                }
            )

        for big_head in range(big_heads):
            small_head = min(range(small_heads), key=lambda idx: matrix[idx][big_head]["jsd"])
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "small_model": small_name,
                    "big_model": big_name,
                    "layer": layer_idx,
                    "small_head": small_head,
                    "big_head": big_head,
                    "match_type": "big_to_best_small",
                    **matrix[small_head][big_head],
                }
            )

        if min(small_heads, big_heads) <= max(small_heads, big_heads) <= 64:
            assignment_pairs = _rectangular_assignment_pairs([[cell["jsd"] for cell in row] for row in matrix])
            for small_head, big_head in assignment_pairs:
                rows.append(
                    {
                        "prompt_index": prompt_index,
                        "small_model": small_name,
                        "big_model": big_name,
                        "layer": layer_idx,
                        "small_head": small_head,
                        "big_head": big_head,
                        "match_type": "one_to_one",
                        **matrix[small_head][big_head],
                    }
                )
    return rows


def fit_head_linear_combinations(
    small_layer: torch.Tensor,
    big_layer: torch.Tensor,
    *,
    steps: int = 300,
    lr: float = 0.2,
    eps: float = 1e-12,
) -> dict[str, Any]:
    if small_layer.ndim != 4 or big_layer.ndim != 4:
        raise ValueError("attention layers must have shape batch x heads x query x key")
    if small_layer.shape[0] != big_layer.shape[0] or small_layer.shape[-2:] != big_layer.shape[-2:]:
        raise ValueError("small and big attention layers must have matching batch/query/key dimensions")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")

    small = _normalize(small_layer).float()
    target = _normalize(big_layer).float()
    small_heads = small.shape[1]
    big_heads = target.shape[1]
    with torch.no_grad():
        pair_jsd = torch.empty(big_heads, small_heads, dtype=torch.float32)
        for big_head in range(big_heads):
            for small_head in range(small_heads):
                pair_jsd[big_head, small_head] = attention_distance(small[:, small_head], target[:, big_head])["jsd"]
        logits_init = (-4.0 * pair_jsd).clamp(-8.0, 8.0)
    logits = torch.nn.Parameter(logits_init)
    optimizer = torch.optim.Adam([logits], lr=lr)
    best_loss = math.inf
    best_weights = torch.softmax(logits.detach(), dim=-1)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(logits, dim=-1)
        combo = torch.einsum("hs,bsqk->bhqk", weights, small)
        loss = _jsd_tensor(combo, target, eps=eps).mean()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_weights = torch.softmax(logits.detach(), dim=-1)
    combo = torch.einsum("hs,bsqk->bhqk", best_weights, small)
    metrics = [attention_distance(combo[:, big_head], target[:, big_head], eps=eps) for big_head in range(big_heads)]
    mse = ((combo - target) ** 2).mean(dim=(0, 2, 3))
    return {
        "weights": best_weights.cpu().tolist(),
        "metrics": metrics,
        "mse": [float(value.item()) for value in mse.cpu()],
        "steps": steps,
        "lr": lr,
    }


def fit_head_exponential_combinations(
    small_layer: torch.Tensor,
    big_layer: torch.Tensor,
    *,
    steps: int = 300,
    lr: float = 0.2,
    eps: float = 1e-12,
) -> dict[str, Any]:
    if small_layer.ndim != 4 or big_layer.ndim != 4:
        raise ValueError("attention layers must have shape batch x heads x query x key")
    if small_layer.shape[0] != big_layer.shape[0] or small_layer.shape[-2:] != big_layer.shape[-2:]:
        raise ValueError("small and big attention layers must have matching batch/query/key dimensions")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")

    small = _normalize(small_layer, eps).float()
    target = _normalize(big_layer, eps).float()
    small_heads = small.shape[1]
    big_heads = target.shape[1]
    support = small.sum(dim=1, keepdim=True) > eps
    small_log = small.clamp_min(eps).log()
    with torch.no_grad():
        pair_jsd = torch.empty(big_heads, small_heads, dtype=torch.float32)
        for big_head in range(big_heads):
            for small_head in range(small_heads):
                pair_jsd[big_head, small_head] = attention_distance(small[:, small_head], target[:, big_head])["jsd"]
        logits_init = (-4.0 * pair_jsd).clamp(-8.0, 8.0)
    logits = torch.nn.Parameter(logits_init)
    optimizer = torch.optim.Adam([logits], lr=lr)
    best_loss = math.inf
    best_weights = torch.softmax(logits.detach(), dim=-1)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(logits, dim=-1)
        combo = _exponential_attention_combo(weights, small_log, support, eps=eps)
        loss = _jsd_tensor(combo, target, eps=eps).mean()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_weights = torch.softmax(logits.detach(), dim=-1)
    combo = _exponential_attention_combo(best_weights, small_log, support, eps=eps)
    metrics = [attention_distance(combo[:, big_head], target[:, big_head], eps=eps) for big_head in range(big_heads)]
    mse = ((combo - target) ** 2).mean(dim=(0, 2, 3))
    return {
        "weights": best_weights.cpu().tolist(),
        "metrics": metrics,
        "mse": [float(value.item()) for value in mse.cpu()],
        "steps": steps,
        "lr": lr,
    }


def fit_head_wasserstein_combinations(
    small_layer: torch.Tensor,
    big_layer: torch.Tensor,
    *,
    steps: int = 300,
    lr: float = 0.2,
    eps: float = 1e-12,
    quantile_count: int | None = None,
) -> dict[str, Any]:
    if small_layer.ndim != 4 or big_layer.ndim != 4:
        raise ValueError("attention layers must have shape batch x heads x query x key")
    if small_layer.shape[0] != big_layer.shape[0] or small_layer.shape[-2:] != big_layer.shape[-2:]:
        raise ValueError("small and big attention layers must have matching batch/query/key dimensions")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")

    small = _normalize(small_layer, eps).float()
    target = _normalize(big_layer, eps).float()
    key_count = small.shape[-1]
    quantile_count = key_count if quantile_count is None else quantile_count
    if quantile_count <= 0:
        raise ValueError("quantile_count must be positive")
    small_heads = small.shape[1]
    big_heads = target.shape[1]
    quantile_positions = _attention_quantile_positions(small, quantile_count=quantile_count)
    with torch.no_grad():
        pair_jsd = torch.empty(big_heads, small_heads, dtype=torch.float32)
        for big_head in range(big_heads):
            for small_head in range(small_heads):
                pair_jsd[big_head, small_head] = attention_distance(small[:, small_head], target[:, big_head])["jsd"]
        logits_init = (-4.0 * pair_jsd).clamp(-8.0, 8.0)
    logits = torch.nn.Parameter(logits_init)
    optimizer = torch.optim.Adam([logits], lr=lr)
    best_loss = math.inf
    best_weights = torch.softmax(logits.detach(), dim=-1)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(logits, dim=-1)
        combo = _wasserstein_attention_combo(weights, quantile_positions, key_count)
        loss = _jsd_tensor(combo, target, eps=eps).mean()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_weights = torch.softmax(logits.detach(), dim=-1)
    combo = _wasserstein_attention_combo(best_weights, quantile_positions, key_count)
    metrics = [attention_distance(combo[:, big_head], target[:, big_head], eps=eps) for big_head in range(big_heads)]
    mse = ((combo - target) ** 2).mean(dim=(0, 2, 3))
    return {
        "weights": best_weights.cpu().tolist(),
        "metrics": metrics,
        "mse": [float(value.item()) for value in mse.cpu()],
        "steps": steps,
        "lr": lr,
        "quantile_count": quantile_count,
    }


def compare_cross_model_linear_combinations(
    small_name: str,
    small_attentions: list[torch.Tensor],
    big_name: str,
    big_attentions: list[torch.Tensor],
    *,
    prompt_index: int = 0,
    steps: int = 300,
    lr: float = 0.2,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layer_count = min(len(small_attentions), len(big_attentions))
    for layer_idx in range(layer_count):
        fit = fit_head_linear_combinations(small_attentions[layer_idx], big_attentions[layer_idx], steps=steps, lr=lr)
        for big_head, metrics in enumerate(fit["metrics"]):
            weights = fit["weights"][big_head]
            top_head = max(range(len(weights)), key=lambda idx: weights[idx])
            entropy = -sum(weight * math.log(max(weight, 1e-12)) for weight in weights)
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "small_model": small_name,
                    "big_model": big_name,
                    "layer": layer_idx,
                    "big_head": big_head,
                    "small_heads": len(weights),
                    "top_small_head": top_head,
                    "top_weight": weights[top_head],
                    "weight_entropy": entropy,
                    "effective_heads": math.exp(entropy),
                    "mse": fit["mse"][big_head],
                    "fit_steps": fit["steps"],
                    "fit_lr": fit["lr"],
                    "weights": weights,
                    **metrics,
                }
            )
    return rows


def compare_cross_model_wasserstein_combinations(
    small_name: str,
    small_attentions: list[torch.Tensor],
    big_name: str,
    big_attentions: list[torch.Tensor],
    *,
    prompt_index: int = 0,
    steps: int = 300,
    lr: float = 0.2,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layer_count = min(len(small_attentions), len(big_attentions))
    for layer_idx in range(layer_count):
        fit = fit_head_wasserstein_combinations(small_attentions[layer_idx], big_attentions[layer_idx], steps=steps, lr=lr)
        for big_head, metrics in enumerate(fit["metrics"]):
            weights = fit["weights"][big_head]
            top_head = max(range(len(weights)), key=lambda idx: weights[idx])
            entropy = -sum(weight * math.log(max(weight, 1e-12)) for weight in weights)
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "small_model": small_name,
                    "big_model": big_name,
                    "layer": layer_idx,
                    "big_head": big_head,
                    "small_heads": len(weights),
                    "top_small_head": top_head,
                    "top_weight": weights[top_head],
                    "weight_entropy": entropy,
                    "effective_heads": math.exp(entropy),
                    "mse": fit["mse"][big_head],
                    "fit_steps": fit["steps"],
                    "fit_lr": fit["lr"],
                    "quantile_count": fit["quantile_count"],
                    "method": "wasserstein_distribution_combo",
                    "weights": weights,
                    **metrics,
                }
            )
    return rows


def compare_cross_model_exponential_combinations(
    small_name: str,
    small_attentions: list[torch.Tensor],
    big_name: str,
    big_attentions: list[torch.Tensor],
    *,
    prompt_index: int = 0,
    steps: int = 300,
    lr: float = 0.2,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layer_count = min(len(small_attentions), len(big_attentions))
    for layer_idx in range(layer_count):
        fit = fit_head_exponential_combinations(small_attentions[layer_idx], big_attentions[layer_idx], steps=steps, lr=lr)
        for big_head, metrics in enumerate(fit["metrics"]):
            weights = fit["weights"][big_head]
            top_head = max(range(len(weights)), key=lambda idx: weights[idx])
            entropy = -sum(weight * math.log(max(weight, 1e-12)) for weight in weights)
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "small_model": small_name,
                    "big_model": big_name,
                    "layer": layer_idx,
                    "big_head": big_head,
                    "small_heads": len(weights),
                    "top_small_head": top_head,
                    "top_weight": weights[top_head],
                    "weight_entropy": entropy,
                    "effective_heads": math.exp(entropy),
                    "mse": fit["mse"][big_head],
                    "fit_steps": fit["steps"],
                    "fit_lr": fit["lr"],
                    "method": "exponential_distribution_combo",
                    "weights": weights,
                    **metrics,
                }
            )
    return rows


def _flatten_attention_heads(layer_attentions: list[torch.Tensor], eps: float = 1e-12) -> tuple[torch.Tensor, list[tuple[int, int]], list[torch.Tensor]]:
    if not layer_attentions:
        raise ValueError("layer_attentions must be non-empty")
    head_count = int(layer_attentions[0].shape[1])
    parts = []
    segments: list[tuple[int, int]] = []
    masks: list[torch.Tensor] = []
    offset = 0
    for layer in layer_attentions:
        if layer.ndim != 4:
            raise ValueError("attention layers must have shape batch x heads x query x key")
        if int(layer.shape[1]) != head_count:
            raise ValueError("all attention layers must have the same number of heads")
        probs = _normalize(layer, eps).float()
        batch, _, queries, keys = probs.shape
        for batch_idx in range(batch):
            for query_idx in range(queries):
                row = probs[batch_idx, :, query_idx, :]
                parts.append(row)
                segments.append((offset, offset + keys))
                masks.append(row.sum(dim=0) > eps)
                offset += keys
    return torch.cat(parts, dim=1), segments, masks


def _segmented_jsd(p_raw: torch.Tensor, q_raw: torch.Tensor, segments: list[tuple[int, int]], eps: float = 1e-12) -> torch.Tensor:
    losses = []
    for start, end in segments:
        p = _normalize(p_raw[:, start:end], eps)
        q = _normalize(q_raw[:, start:end], eps)
        m = 0.5 * (p + q)
        kl_pm = (p * ((p + eps).log() - (m + eps).log())).sum(dim=-1)
        kl_qm = (q * ((q + eps).log() - (m + eps).log())).sum(dim=-1)
        losses.append(0.5 * (kl_pm + kl_qm))
    return torch.stack(losses, dim=1)


def fit_attention_basis(
    layer_attentions: list[torch.Tensor],
    *,
    basis_size: int,
    steps: int = 400,
    lr: float = 0.2,
    seed: int = 0,
    eps: float = 1e-12,
) -> dict[str, Any]:
    if basis_size <= 0:
        raise ValueError("basis_size must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    target_cpu, segments, masks = _flatten_attention_heads(layer_attentions, eps)
    heads, features = target_cpu.shape
    if basis_size >= heads:
        raise ValueError("basis_size must be smaller than the number of heads")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = target_cpu.to(device)
    segment_masks = [mask.to(device) for mask in masks]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    source_indices = torch.linspace(0, heads - 1, steps=basis_size, device=device).round().long().unique(sorted=True)
    if len(source_indices) < basis_size:
        extra = torch.randperm(heads, generator=generator, device=device)
        source_indices = torch.cat([source_indices, extra])[:basis_size].unique(sorted=True)
    source_indices = source_indices[:basis_size]

    basis_init = target[source_indices].clamp_min(eps).log()
    basis_logits = torch.nn.Parameter(basis_init + 0.01 * torch.randn((basis_size, features), generator=generator, device=device))
    with torch.no_grad():
        init_basis = _attention_basis_from_logits(basis_logits, segments, segment_masks)
        init_distances = _segmented_jsd(target.repeat_interleave(basis_size, dim=0), init_basis.repeat(heads, 1), segments, eps)
        init_distances = init_distances.mean(dim=1).view(heads, basis_size)
        coeff_init = (-8.0 * init_distances).clamp(-8.0, 8.0)
    coeff_logits = torch.nn.Parameter(coeff_init)
    optimizer = torch.optim.Adam([basis_logits, coeff_logits], lr=lr)
    best_loss = math.inf
    best_basis = None
    best_coeff = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        basis = _attention_basis_from_logits(basis_logits, segments, segment_masks)
        coeff = torch.softmax(coeff_logits, dim=-1)
        recon = coeff @ basis
        loss = _segmented_jsd(recon, target, segments, eps).mean()
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_basis = basis.detach()
            best_coeff = coeff.detach()
    if best_basis is None or best_coeff is None:
        raise RuntimeError("basis fit failed")
    recon = best_coeff @ best_basis
    per_head_jsd = _segmented_jsd(recon, target, segments, eps).mean(dim=1)
    per_head_tv = _segmented_tv(recon, target, segments, eps).mean(dim=1)
    entropy = -(best_coeff * best_coeff.clamp_min(eps).log()).sum(dim=-1)
    return {
        "basis_size": basis_size,
        "heads": heads,
        "features": features,
        "segments": len(segments),
        "steps": steps,
        "lr": lr,
        "mean_jsd": float(per_head_jsd.mean().cpu().item()),
        "max_jsd": float(per_head_jsd.max().cpu().item()),
        "mean_tv": float(per_head_tv.mean().cpu().item()),
        "max_tv": float(per_head_tv.max().cpu().item()),
        "mean_effective_basis": float(entropy.exp().mean().cpu().item()),
        "per_head_jsd": [float(value.item()) for value in per_head_jsd.cpu()],
        "per_head_tv": [float(value.item()) for value in per_head_tv.cpu()],
        "coefficients": best_coeff.cpu().tolist(),
    }


def fit_attention_basis_nmf(
    layer_attentions: list[torch.Tensor],
    *,
    basis_size: int,
    steps: int = 500,
    seed: int = 0,
    eps: float = 1e-12,
) -> dict[str, Any]:
    if basis_size <= 0:
        raise ValueError("basis_size must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    target, segments, masks = _flatten_attention_heads(layer_attentions, eps)
    heads, features = target.shape
    if basis_size >= heads:
        raise ValueError("basis_size must be smaller than the number of heads")

    from sklearn.decomposition import NMF

    target_np = target.numpy()
    nmf = NMF(
        n_components=basis_size,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=steps,
        random_state=seed,
        tol=1e-5,
    )
    init_coeff = nmf.fit_transform(target_np)
    basis = torch.from_numpy(nmf.components_).float().clamp_min(0.0)
    basis = _normalize_basis_segments(basis, segments, masks, eps)
    coeff_logits = torch.nn.Parameter(torch.from_numpy(init_coeff).float().clamp_min(eps).log())
    optimizer = torch.optim.Adam([coeff_logits], lr=0.1)
    coeff_steps = min(80, max(20, steps // 4))
    for _ in range(coeff_steps):
        optimizer.zero_grad(set_to_none=True)
        coeffs = torch.nn.functional.softplus(coeff_logits)
        recon = coeffs @ basis
        loss = _segmented_jsd(recon, target, segments, eps).mean()
        loss.backward()
        optimizer.step()
    coeffs = torch.nn.functional.softplus(coeff_logits.detach())
    recon = coeffs @ basis
    per_head_jsd = _segmented_jsd(recon, target, segments, eps).mean(dim=1)
    per_head_tv = _segmented_tv(recon, target, segments, eps).mean(dim=1)
    coeff_sum = coeffs.sum(dim=-1, keepdim=True).clamp_min(eps)
    coeff_probs = coeffs / coeff_sum
    entropy = -(coeff_probs * coeff_probs.clamp_min(eps).log()).sum(dim=-1)
    return {
        "basis_size": basis_size,
        "heads": heads,
        "features": features,
        "segments": len(segments),
        "steps": steps,
        "mean_jsd": float(per_head_jsd.mean().item()),
        "max_jsd": float(per_head_jsd.max().item()),
        "mean_tv": float(per_head_tv.mean().item()),
        "max_tv": float(per_head_tv.max().item()),
        "mean_effective_basis": float(entropy.exp().mean().item()),
        "per_head_jsd": [float(value.item()) for value in per_head_jsd],
        "per_head_tv": [float(value.item()) for value in per_head_tv],
        "coefficient_sums": [float(value.item()) for value in coeff_sum.squeeze(-1)],
    }


def _normalize_basis_segments(
    basis: torch.Tensor,
    segments: list[tuple[int, int]],
    masks: list[torch.Tensor],
    eps: float = 1e-12,
) -> torch.Tensor:
    parts = []
    for (start, end), mask in zip(segments, masks, strict=True):
        part = basis[:, start:end].masked_fill(~mask.view(1, -1), 0.0)
        parts.append(_normalize(part, eps))
    return torch.cat(parts, dim=-1)


def _attention_basis_from_logits(basis_logits: torch.Tensor, segments: list[tuple[int, int]], masks: list[torch.Tensor]) -> torch.Tensor:
    parts = []
    for (start, end), mask in zip(segments, masks, strict=True):
        logits = basis_logits[:, start:end].masked_fill(~mask.view(1, -1), -80.0)
        parts.append(torch.softmax(logits, dim=-1))
    return torch.cat(parts, dim=-1)


def _segmented_tv(p_raw: torch.Tensor, q_raw: torch.Tensor, segments: list[tuple[int, int]], eps: float = 1e-12) -> torch.Tensor:
    values = []
    for start, end in segments:
        p = _normalize(p_raw[:, start:end], eps)
        q = _normalize(q_raw[:, start:end], eps)
        values.append(0.5 * (p - q).abs().sum(dim=-1))
    return torch.stack(values, dim=1)


def _jsd_tensor(p_raw: torch.Tensor, q_raw: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    p = _normalize(p_raw, eps)
    q = _normalize(q_raw, eps)
    m = 0.5 * (p + q)
    kl_pm = (p * ((p + eps).log() - (m + eps).log())).sum(dim=-1)
    kl_qm = (q * ((q + eps).log() - (m + eps).log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def _exponential_attention_combo(weights: torch.Tensor, small_log: torch.Tensor, support: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    combo_logits = torch.einsum("hs,bsqk->bhqk", weights, small_log)
    combo_logits = combo_logits.masked_fill(~support, -80.0)
    combo = torch.softmax(combo_logits, dim=-1)
    combo = combo.masked_fill(~support, 0.0)
    return _normalize(combo, eps)


def _attention_quantile_positions(probs: torch.Tensor, *, quantile_count: int) -> torch.Tensor:
    cdf = probs.cumsum(dim=-1)
    cdf[..., -1] = 1.0
    quantiles = (torch.arange(quantile_count, device=probs.device, dtype=probs.dtype) + 0.5) / quantile_count
    positions = (cdf.unsqueeze(-2) < quantiles.view(1, 1, 1, quantile_count, 1)).sum(dim=-1)
    return positions.clamp_max(probs.shape[-1] - 1).to(probs.dtype)


def _wasserstein_attention_combo(weights: torch.Tensor, quantile_positions: torch.Tensor, key_count: int) -> torch.Tensor:
    positions = torch.einsum("hs,bsqr->bhqr", weights, quantile_positions)
    lower = positions.floor().long().clamp(0, key_count - 1)
    upper = (lower + 1).clamp(0, key_count - 1)
    upper_weight = positions - lower.to(positions.dtype)
    lower_weight = 1.0 - upper_weight
    mass_scale = 1.0 / positions.shape[-1]
    output = positions.new_zeros((*positions.shape[:-1], key_count))
    output.scatter_add_(-1, lower, lower_weight * mass_scale)
    output.scatter_add_(-1, upper, upper_weight * mass_scale)
    return output


def summarize_attention_run(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    cross_rows = _read_jsonl(output_dir / "cross_model.jsonl")
    match_rows = _read_jsonl(output_dir / "cross_head_matching.jsonl")
    within_rows = _read_jsonl(output_dir / "within_model.jsonl")
    head_rows = _read_jsonl(output_dir / "head_summary.jsonl")
    best_match_rows = [row for row in match_rows if row.get("match_type") == "small_to_best_big"]
    within_pair_rows = [row for row in within_rows if row.get("match_type", "all_pairs") == "all_pairs"]
    within_best_rows = [row for row in within_rows if row.get("match_type") == "head_to_best_head"]
    within_assignment_rows = [row for row in within_rows if row.get("match_type") == "one_to_one"]
    summary = {
        "cross_model_count": len(cross_rows),
        "cross_head_matching_count": len(match_rows),
        "within_model_count": len(within_rows),
        "within_model_pair_count": len(within_pair_rows),
        "within_head_best_match_count": len(within_best_rows),
        "within_head_one_to_one_count": len(within_assignment_rows),
        "head_summary_count": len(head_rows),
        "best_cross_model": min(cross_rows, key=lambda row: row["jsd"]) if cross_rows else None,
        "worst_cross_model": max(cross_rows, key=lambda row: row["jsd"]) if cross_rows else None,
        "best_cross_head_match": min(best_match_rows, key=lambda row: row["jsd"]) if best_match_rows else None,
        "worst_cross_head_best_match": max(best_match_rows, key=lambda row: row["jsd"]) if best_match_rows else None,
        "closest_within_model_pair": min(within_pair_rows, key=lambda row: row["jsd"]) if within_pair_rows else None,
        "furthest_within_model_pair": max(within_pair_rows, key=lambda row: row["jsd"]) if within_pair_rows else None,
        "best_within_head_match": min(within_best_rows, key=lambda row: row["jsd"]) if within_best_rows else None,
        "worst_within_head_best_match": max(within_best_rows, key=lambda row: row["jsd"]) if within_best_rows else None,
        "best_within_one_to_one_match": min(within_assignment_rows, key=lambda row: row["jsd"]) if within_assignment_rows else None,
        "worst_within_one_to_one_match": max(within_assignment_rows, key=lambda row: row["jsd"]) if within_assignment_rows else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_attention_analysis(config: AttentionAnalysisConfig) -> dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True, default=str) + "\n")
    device = _select_device(config.device)
    dtype = _torch_dtype(config.dtype)
    model_dtype = dtype if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(config.big_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    models = {}
    for label, model_name in (("small", config.small_model), ("big", config.big_model)):
        model = _load_attention_model(AutoModel, model_name, model_dtype, device)
        models[label] = model

    metadata = {
        "small_model": config.small_model,
        "big_model": config.big_model,
        "device": str(device),
        "dtype": config.dtype,
        "prompts": list(config.prompts),
        "started_at_unix": time.time(),
    }
    (config.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    with (
        (config.output_dir / "head_summary.jsonl").open("w", encoding="utf-8") as head_file,
        (config.output_dir / "within_model.jsonl").open("w", encoding="utf-8") as within_file,
        (config.output_dir / "cross_model.jsonl").open("w", encoding="utf-8") as cross_file,
        (config.output_dir / "cross_head_matching.jsonl").open("w", encoding="utf-8") as match_file,
    ):
        for prompt_index, prompt in enumerate(config.prompts):
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_length,
                add_special_tokens=False,
            )
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            prompt_attentions = {}
            for label, model in models.items():
                attentions = _capture_attentions(model, input_ids, attention_mask, device, dtype)
                prompt_attentions[label] = attentions
                if config.save_tensors:
                    torch.save(attentions, config.output_dir / f"{label}_prompt_{prompt_index}_attentions.pt")
                for layer_idx, layer_attn in enumerate(attentions):
                    for head_idx in range(layer_attn.shape[1]):
                        row = {
                            "prompt_index": prompt_index,
                            "model": config.small_model if label == "small" else config.big_model,
                            "layer": layer_idx,
                            "head": head_idx,
                            "seq_len": int(layer_attn.shape[-1]),
                            **head_summary(layer_attn[:, head_idx], local_window=config.local_window),
                        }
                        _write_jsonl(head_file, row)
                for row in compare_within_model_head_matching(
                    config.small_model if label == "small" else config.big_model,
                    attentions,
                    prompt_index=prompt_index,
                ):
                    _write_jsonl(within_file, row)
            for row in compare_cross_model(
                config.small_model,
                prompt_attentions["small"],
                config.big_model,
                prompt_attentions["big"],
                prompt_index=prompt_index,
            ):
                _write_jsonl(cross_file, row)
            for row in compare_cross_model_head_matching(
                config.small_model,
                prompt_attentions["small"],
                config.big_model,
                prompt_attentions["big"],
                prompt_index=prompt_index,
            ):
                _write_jsonl(match_file, row)

    summary = summarize_attention_run(config.output_dir)
    metadata["completed_at_unix"] = time.time()
    (config.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return summary


def run_attention_linear_combo_analysis(config: AttentionAnalysisConfig, *, fit_steps: int = 300, fit_lr: float = 0.2) -> dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    if fit_steps <= 0:
        raise ValueError("fit_steps must be positive")
    if fit_lr <= 0:
        raise ValueError("fit_lr must be positive")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True, default=str) + "\n")
    device = _select_device(config.device)
    dtype = _torch_dtype(config.dtype)
    model_dtype = dtype if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(config.big_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    models = {}
    for label, model_name in (("small", config.small_model), ("big", config.big_model)):
        models[label] = _load_attention_model(AutoModel, model_name, model_dtype, device)

    metadata = {
        "small_model": config.small_model,
        "big_model": config.big_model,
        "device": str(device),
        "dtype": config.dtype,
        "prompts": list(config.prompts),
        "fit_steps": fit_steps,
        "fit_lr": fit_lr,
        "started_at_unix": time.time(),
    }
    (config.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    with (config.output_dir / "linear_combo.jsonl").open("w", encoding="utf-8") as combo_file:
        for prompt_index, prompt in enumerate(config.prompts):
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_length,
                add_special_tokens=False,
            )
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            small_attentions = _capture_attentions(models["small"], input_ids, attention_mask, device, dtype)
            big_attentions = _capture_attentions(models["big"], input_ids, attention_mask, device, dtype)
            if config.save_tensors:
                torch.save(small_attentions, config.output_dir / f"small_prompt_{prompt_index}_attentions.pt")
                torch.save(big_attentions, config.output_dir / f"big_prompt_{prompt_index}_attentions.pt")
            for row in compare_cross_model_linear_combinations(
                config.small_model,
                small_attentions,
                config.big_model,
                big_attentions,
                prompt_index=prompt_index,
                steps=fit_steps,
                lr=fit_lr,
            ):
                _write_jsonl(combo_file, row)
            gc.collect()
    summary = summarize_attention_linear_combo_run(config.output_dir)
    metadata["completed_at_unix"] = time.time()
    (config.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return summary


def run_attention_exponential_combo_analysis(config: AttentionAnalysisConfig, *, fit_steps: int = 300, fit_lr: float = 0.2) -> dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    if fit_steps <= 0:
        raise ValueError("fit_steps must be positive")
    if fit_lr <= 0:
        raise ValueError("fit_lr must be positive")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True, default=str) + "\n")
    device = _select_device(config.device)
    dtype = _torch_dtype(config.dtype)
    model_dtype = dtype if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(config.big_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    models = {}
    for label, model_name in (("small", config.small_model), ("big", config.big_model)):
        models[label] = _load_attention_model(AutoModel, model_name, model_dtype, device)

    metadata = {
        "small_model": config.small_model,
        "big_model": config.big_model,
        "device": str(device),
        "dtype": config.dtype,
        "prompts": list(config.prompts),
        "fit_steps": fit_steps,
        "fit_lr": fit_lr,
        "method": "exponential_distribution_combo",
        "started_at_unix": time.time(),
    }
    (config.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    with (config.output_dir / "exponential_combo.jsonl").open("w", encoding="utf-8") as combo_file:
        for prompt_index, prompt in enumerate(config.prompts):
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_length,
                add_special_tokens=False,
            )
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            small_attentions = _capture_attentions(models["small"], input_ids, attention_mask, device, dtype)
            big_attentions = _capture_attentions(models["big"], input_ids, attention_mask, device, dtype)
            if config.save_tensors:
                torch.save(small_attentions, config.output_dir / f"small_prompt_{prompt_index}_attentions.pt")
                torch.save(big_attentions, config.output_dir / f"big_prompt_{prompt_index}_attentions.pt")
            for row in compare_cross_model_exponential_combinations(
                config.small_model,
                small_attentions,
                config.big_model,
                big_attentions,
                prompt_index=prompt_index,
                steps=fit_steps,
                lr=fit_lr,
            ):
                _write_jsonl(combo_file, row)
            gc.collect()
    summary = summarize_attention_exponential_combo_run(config.output_dir)
    metadata["completed_at_unix"] = time.time()
    (config.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return summary


def run_attention_wasserstein_combo_analysis(config: AttentionAnalysisConfig, *, fit_steps: int = 300, fit_lr: float = 0.2) -> dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    if fit_steps <= 0:
        raise ValueError("fit_steps must be positive")
    if fit_lr <= 0:
        raise ValueError("fit_lr must be positive")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True, default=str) + "\n")
    device = _select_device(config.device)
    dtype = _torch_dtype(config.dtype)
    model_dtype = dtype if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(config.big_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    models = {}
    for label, model_name in (("small", config.small_model), ("big", config.big_model)):
        models[label] = _load_attention_model(AutoModel, model_name, model_dtype, device)

    metadata = {
        "small_model": config.small_model,
        "big_model": config.big_model,
        "device": str(device),
        "dtype": config.dtype,
        "prompts": list(config.prompts),
        "fit_steps": fit_steps,
        "fit_lr": fit_lr,
        "method": "wasserstein_distribution_combo",
        "started_at_unix": time.time(),
    }
    (config.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    with (config.output_dir / "wasserstein_combo.jsonl").open("w", encoding="utf-8") as combo_file:
        for prompt_index, prompt in enumerate(config.prompts):
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_length,
                add_special_tokens=False,
            )
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            small_attentions = _capture_attentions(models["small"], input_ids, attention_mask, device, dtype)
            big_attentions = _capture_attentions(models["big"], input_ids, attention_mask, device, dtype)
            if config.save_tensors:
                torch.save(small_attentions, config.output_dir / f"small_prompt_{prompt_index}_attentions.pt")
                torch.save(big_attentions, config.output_dir / f"big_prompt_{prompt_index}_attentions.pt")
            for row in compare_cross_model_wasserstein_combinations(
                config.small_model,
                small_attentions,
                config.big_model,
                big_attentions,
                prompt_index=prompt_index,
                steps=fit_steps,
                lr=fit_lr,
            ):
                _write_jsonl(combo_file, row)
            gc.collect()
    summary = summarize_attention_wasserstein_combo_run(config.output_dir)
    metadata["completed_at_unix"] = time.time()
    (config.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return summary


def run_attention_combo_analysis(
    config: AttentionAnalysisConfig,
    *,
    combo_method: str,
    fit_steps: int = 300,
    fit_lr: float = 0.2,
) -> dict[str, Any]:
    if combo_method == "linear":
        return run_attention_linear_combo_analysis(config, fit_steps=fit_steps, fit_lr=fit_lr)
    if combo_method == "exponential":
        return run_attention_exponential_combo_analysis(config, fit_steps=fit_steps, fit_lr=fit_lr)
    if combo_method == "wasserstein":
        return run_attention_wasserstein_combo_analysis(config, fit_steps=fit_steps, fit_lr=fit_lr)
    raise ValueError("combo_method must be linear, exponential, or wasserstein")


def summarize_attention_linear_combo_run(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    rows = _read_jsonl(output_dir / "linear_combo.jsonl")
    if not rows:
        summary = {"linear_combo_count": 0, "best_linear_combo": None, "worst_linear_combo": None, "mean_jsd_by_layer": []}
    else:
        layers = sorted({row["layer"] for row in rows})
        summary = {
            "linear_combo_count": len(rows),
            "best_linear_combo": min(rows, key=lambda row: row["jsd"]),
            "worst_linear_combo": max(rows, key=lambda row: row["jsd"]),
            "mean_jsd": sum(row["jsd"] for row in rows) / len(rows),
            "mean_effective_heads": sum(row["effective_heads"] for row in rows) / len(rows),
            "mean_top_weight": sum(row["top_weight"] for row in rows) / len(rows),
            "mean_jsd_by_layer": [
                {
                    "layer": layer,
                    "mean_jsd": sum(row["jsd"] for row in rows if row["layer"] == layer)
                    / sum(1 for row in rows if row["layer"] == layer),
                    "mean_effective_heads": sum(row["effective_heads"] for row in rows if row["layer"] == layer)
                    / sum(1 for row in rows if row["layer"] == layer),
                    "mean_top_weight": sum(row["top_weight"] for row in rows if row["layer"] == layer)
                    / sum(1 for row in rows if row["layer"] == layer),
                }
                for layer in layers
            ],
        }
    (output_dir / "linear_combo_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def summarize_attention_exponential_combo_run(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    rows = _read_jsonl(output_dir / "exponential_combo.jsonl")
    if not rows:
        summary = {"exponential_combo_count": 0, "best_exponential_combo": None, "worst_exponential_combo": None, "mean_jsd_by_layer": []}
    else:
        layers = sorted({row["layer"] for row in rows})
        summary = {
            "exponential_combo_count": len(rows),
            "best_exponential_combo": min(rows, key=lambda row: row["jsd"]),
            "worst_exponential_combo": max(rows, key=lambda row: row["jsd"]),
            "mean_jsd": sum(row["jsd"] for row in rows) / len(rows),
            "mean_effective_heads": sum(row["effective_heads"] for row in rows) / len(rows),
            "mean_top_weight": sum(row["top_weight"] for row in rows) / len(rows),
            "mean_jsd_by_layer": [
                {
                    "layer": layer,
                    "mean_jsd": sum(row["jsd"] for row in rows if row["layer"] == layer)
                    / sum(1 for row in rows if row["layer"] == layer),
                    "mean_effective_heads": sum(row["effective_heads"] for row in rows if row["layer"] == layer)
                    / sum(1 for row in rows if row["layer"] == layer),
                    "mean_top_weight": sum(row["top_weight"] for row in rows if row["layer"] == layer)
                    / sum(1 for row in rows if row["layer"] == layer),
                }
                for layer in layers
            ],
        }
    (output_dir / "exponential_combo_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def summarize_attention_wasserstein_combo_run(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    rows = _read_jsonl(output_dir / "wasserstein_combo.jsonl")
    if not rows:
        summary = {"wasserstein_combo_count": 0, "best_wasserstein_combo": None, "worst_wasserstein_combo": None, "mean_jsd_by_layer": []}
    else:
        layers = sorted({row["layer"] for row in rows})
        summary = {
            "wasserstein_combo_count": len(rows),
            "best_wasserstein_combo": min(rows, key=lambda row: row["jsd"]),
            "worst_wasserstein_combo": max(rows, key=lambda row: row["jsd"]),
            "mean_jsd": sum(row["jsd"] for row in rows) / len(rows),
            "mean_effective_heads": sum(row["effective_heads"] for row in rows) / len(rows),
            "mean_top_weight": sum(row["top_weight"] for row in rows) / len(rows),
            "mean_jsd_by_layer": [
                {
                    "layer": layer,
                    "mean_jsd": sum(row["jsd"] for row in rows if row["layer"] == layer)
                    / sum(1 for row in rows if row["layer"] == layer),
                    "mean_effective_heads": sum(row["effective_heads"] for row in rows if row["layer"] == layer)
                    / sum(1 for row in rows if row["layer"] == layer),
                    "mean_top_weight": sum(row["top_weight"] for row in rows if row["layer"] == layer)
                    / sum(1 for row in rows if row["layer"] == layer),
                }
                for layer in layers
            ],
        }
    (output_dir / "wasserstein_combo_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_attention_pair_sweep(
    *,
    output_dir: Path,
    model_pairs: tuple[tuple[str, str], ...],
    prompts: tuple[str, ...] = AttentionAnalysisConfig.prompts,
    max_length: int = 96,
    dtype: str = "fp32",
    device: str = "auto",
    local_window: int = 8,
    save_tensors: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for small_model, big_model in model_pairs:
        pair_dir = output_dir / _model_pair_slug(small_model, big_model)
        config = AttentionAnalysisConfig(
            output_dir=pair_dir,
            small_model=small_model,
            big_model=big_model,
            prompts=prompts,
            max_length=max_length,
            dtype=dtype,
            device=device,
            local_window=local_window,
            save_tensors=save_tensors,
        )
        summary = run_attention_analysis(config)
        runs.append(
            {
                "small_model": small_model,
                "big_model": big_model,
                "run_dir": str(pair_dir),
                "summary": summary,
            }
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    sweep_summary = {"runs": runs}
    (output_dir / "sweep_summary.json").write_text(json.dumps(sweep_summary, indent=2, sort_keys=True) + "\n")
    return sweep_summary


def run_attention_basis_sweep(
    *,
    output_dir: Path,
    model_names: tuple[str, ...],
    prompts: tuple[str, ...] = AttentionAnalysisConfig.prompts,
    max_length: int = 96,
    dtype: str = "fp32",
    device: str = "auto",
    basis_sizes: tuple[int, ...] = (),
    fit_steps: int = 400,
    fit_lr: float = 0.2,
) -> dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    if not model_names:
        raise ValueError("model_names must be non-empty")
    if fit_steps <= 0:
        raise ValueError("fit_steps must be positive")
    if fit_lr <= 0:
        raise ValueError("fit_lr must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = _select_device(device)
    dtype_obj = _torch_dtype(dtype)
    model_dtype = dtype_obj if device_obj.type == "cuda" else torch.float32
    runs = []
    for model_name in model_names:
        model_dir = output_dir / _model_slug(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model": model_name,
                    "prompts": list(prompts),
                    "max_length": max_length,
                    "dtype": dtype,
                    "device": str(device_obj),
                    "basis_sizes": list(basis_sizes),
                    "fit_steps": fit_steps,
                    "fit_lr": fit_lr,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = _load_attention_model(AutoModel, model_name, model_dtype, device_obj)
        prompt_attentions = []
        for prompt in prompts:
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            prompt_attentions.append(
                _capture_attentions(
                    model,
                    encoded.input_ids.to(device_obj),
                    encoded.attention_mask.to(device_obj),
                    device_obj,
                    dtype_obj,
                )
            )
        with (model_dir / "attention_basis.jsonl").open("w", encoding="utf-8") as basis_file:
            layer_count = len(prompt_attentions[0])
            for layer_idx in range(layer_count):
                layer_attentions = [attentions[layer_idx] for attentions in prompt_attentions]
                heads = int(layer_attentions[0].shape[1])
                candidate_sizes = basis_sizes if basis_sizes else _default_basis_sizes(heads)
                for basis_size in candidate_sizes:
                    if basis_size >= heads:
                        continue
                    fit = fit_attention_basis_nmf(
                        layer_attentions,
                        basis_size=basis_size,
                        steps=fit_steps,
                        seed=17_000 + layer_idx * 101 + basis_size,
                    )
                    fit.pop("coefficients", None)
                    row = {
                        "model": model_name,
                        "layer": layer_idx,
                        "heads": heads,
                        "basis_size": basis_size,
                        "compression": heads / basis_size,
                        "prompt_count": len(prompts),
                        "method": "nmf_nonnegative_basis",
                        **fit,
                    }
                    _write_jsonl(basis_file, row)
        summary = summarize_attention_basis_run(model_dir)
        runs.append({"model": model_name, "run_dir": str(model_dir), "summary": summary})
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    sweep_summary = {"runs": runs}
    (output_dir / "basis_sweep_summary.json").write_text(json.dumps(sweep_summary, indent=2, sort_keys=True) + "\n")
    return sweep_summary


def summarize_attention_basis_run(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    rows = _read_jsonl(output_dir / "attention_basis.jsonl")
    if not rows:
        summary = {"attention_basis_count": 0, "mean_by_basis_size": []}
    else:
        basis_sizes = sorted({row["basis_size"] for row in rows})
        layers = sorted({row["layer"] for row in rows})
        summary = {
            "attention_basis_count": len(rows),
            "layers": len(layers),
            "heads": rows[0]["heads"],
            "best_row": min(rows, key=lambda row: row["mean_jsd"]),
            "worst_row": max(rows, key=lambda row: row["mean_jsd"]),
            "mean_by_basis_size": [
                {
                    "basis_size": basis_size,
                    "mean_jsd": sum(row["mean_jsd"] for row in rows if row["basis_size"] == basis_size)
                    / sum(1 for row in rows if row["basis_size"] == basis_size),
                    "max_jsd": max(row["max_jsd"] for row in rows if row["basis_size"] == basis_size),
                    "mean_tv": sum(row["mean_tv"] for row in rows if row["basis_size"] == basis_size)
                    / sum(1 for row in rows if row["basis_size"] == basis_size),
                    "mean_effective_basis": sum(row["mean_effective_basis"] for row in rows if row["basis_size"] == basis_size)
                    / sum(1 for row in rows if row["basis_size"] == basis_size),
                }
                for basis_size in basis_sizes
            ],
        }
    (output_dir / "attention_basis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _load_attention_model(auto_model, model_name: str, dtype: torch.dtype, device: torch.device) -> torch.nn.Module:
    kwargs = {"attn_implementation": "eager"}
    try:
        model = auto_model.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        model = auto_model.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _capture_attentions(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=True,
            )
    if outputs.attentions is None:
        raise RuntimeError("model did not return attentions")
    return [attn.detach().float().cpu().contiguous() for attn in outputs.attentions]


def _select_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    if selected.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("mps requested but unavailable")
    return selected


def _min_cost_assignment(costs: list[list[float]]) -> tuple[int, ...]:
    if not costs:
        return ()
    width = len(costs[0])
    if any(len(row) != width for row in costs):
        raise ValueError("cost matrix must be rectangular")
    height = len(costs)
    if height > width:
        raise ValueError("cost matrix must have no more rows than columns")
    u = [0.0] * (height + 1)
    v = [0.0] * (width + 1)
    p = [0] * (width + 1)
    way = [0] * (width + 1)
    for row in range(1, height + 1):
        p[0] = row
        col0 = 0
        minv = [math.inf] * (width + 1)
        used = [False] * (width + 1)
        while True:
            used[col0] = True
            row0 = p[col0]
            delta = math.inf
            col1 = 0
            for col in range(1, width + 1):
                if used[col]:
                    continue
                cur = costs[row0 - 1][col - 1] - u[row0] - v[col]
                if cur < minv[col]:
                    minv[col] = cur
                    way[col] = col0
                if minv[col] < delta:
                    delta = minv[col]
                    col1 = col
            for col in range(0, width + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    minv[col] -= delta
            col0 = col1
            if p[col0] == 0:
                break
        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break
    assignment = [-1] * height
    for col in range(1, width + 1):
        if p[col] != 0:
            assignment[p[col] - 1] = col - 1
    if any(col < 0 for col in assignment):
        raise RuntimeError("failed to compute assignment")
    return tuple(assignment)


def _rectangular_assignment_pairs(costs: list[list[float]]) -> tuple[tuple[int, int], ...]:
    if not costs:
        return ()
    height = len(costs)
    width = len(costs[0])
    if height <= width:
        assignment = _min_cost_assignment(costs)
        return tuple((row_idx, col_idx) for row_idx, col_idx in enumerate(assignment))
    transposed = [[costs[row_idx][col_idx] for row_idx in range(height)] for col_idx in range(width)]
    assignment = _min_cost_assignment(transposed)
    return tuple((row_idx, col_idx) for col_idx, row_idx in enumerate(assignment))


def _write_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parse_prompts(value: str) -> tuple[str, ...]:
    path = Path(value)
    if path.exists():
        return tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    return tuple(part.strip() for part in value.split("||") if part.strip())


def _parse_model_pairs(value: str) -> tuple[tuple[str, str], ...]:
    pairs = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if ">" not in item:
            raise ValueError("model pairs must use small>big syntax")
        small, big = (side.strip() for side in item.split(">", 1))
        if not small or not big:
            raise ValueError("model pairs must use non-empty small>big values")
        pairs.append((small, big))
    if not pairs:
        raise ValueError("model_pairs must be non-empty")
    return tuple(pairs)


def _parse_model_names(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise ValueError("model names must be non-empty")
    return names


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if any(item <= 0 for item in values):
        raise ValueError("integer values must be positive")
    return values


def _default_basis_sizes(heads: int) -> tuple[int, ...]:
    values = {1, 2, 4, heads // 4, heads // 2, (3 * heads) // 4, heads - 1}
    return tuple(sorted(value for value in values if 0 < value < heads))


def _model_pair_slug(small_model: str, big_model: str) -> str:
    def clean(value: str) -> str:
        return value.rsplit("/", 1)[-1].replace(".", "_")

    return f"{clean(small_model)}_to_{clean(big_model)}"


def _model_slug(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1].replace(".", "_")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/attention_analysis"))
    parser.add_argument("--small-model", default="EleutherAI/pythia-31m")
    parser.add_argument("--big-model", default="EleutherAI/pythia-70m")
    parser.add_argument("--model-pairs", default=None, help="Comma-separated small>big model pairs. Overrides --small-model/--big-model.")
    parser.add_argument("--model-names", default=None, help="Comma-separated model names for attention-basis analysis.")
    parser.add_argument("--prompts", default=None, help="Either a file path with one prompt per line, or prompts separated by ||.")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="fp32")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--local-window", type=int, default=8)
    parser.add_argument("--save-tensors", action="store_true")
    parser.add_argument("--linear-combos", action="store_true")
    parser.add_argument("--exponential-combos", action="store_true")
    parser.add_argument("--wasserstein-combos", action="store_true")
    parser.add_argument("--attention-basis", action="store_true")
    parser.add_argument("--basis-sizes", default="")
    parser.add_argument("--fit-steps", type=int, default=300)
    parser.add_argument("--fit-lr", type=float, default=0.2)
    args = parser.parse_args(argv)

    prompts = _parse_prompts(args.prompts) if args.prompts else AttentionAnalysisConfig.prompts
    if args.model_pairs:
        print(
            json.dumps(
                run_attention_pair_sweep(
                    output_dir=args.output_dir,
                    model_pairs=_parse_model_pairs(args.model_pairs),
                    prompts=prompts,
                    max_length=args.max_length,
                    dtype=args.dtype,
                    device=args.device,
                    local_window=args.local_window,
                    save_tensors=args.save_tensors,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.attention_basis:
        model_names = _parse_model_names(args.model_names) if args.model_names else (args.small_model, args.big_model)
        print(
            json.dumps(
                run_attention_basis_sweep(
                    output_dir=args.output_dir,
                    model_names=model_names,
                    prompts=prompts,
                    max_length=args.max_length,
                    dtype=args.dtype,
                    device=args.device,
                    basis_sizes=_parse_int_tuple(args.basis_sizes) if args.basis_sizes else (),
                    fit_steps=args.fit_steps,
                    fit_lr=args.fit_lr,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    config = AttentionAnalysisConfig(
        output_dir=args.output_dir,
        small_model=args.small_model,
        big_model=args.big_model,
        prompts=prompts,
        max_length=args.max_length,
        dtype=args.dtype,
        device=args.device,
        local_window=args.local_window,
        save_tensors=args.save_tensors,
    )
    if args.linear_combos:
        print(json.dumps(run_attention_linear_combo_analysis(config, fit_steps=args.fit_steps, fit_lr=args.fit_lr), indent=2, sort_keys=True))
        return
    if args.exponential_combos:
        print(json.dumps(run_attention_exponential_combo_analysis(config, fit_steps=args.fit_steps, fit_lr=args.fit_lr), indent=2, sort_keys=True))
        return
    if args.wasserstein_combos:
        print(json.dumps(run_attention_wasserstein_combo_analysis(config, fit_steps=args.fit_steps, fit_lr=args.fit_lr), indent=2, sort_keys=True))
        return
    print(json.dumps(run_attention_analysis(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
