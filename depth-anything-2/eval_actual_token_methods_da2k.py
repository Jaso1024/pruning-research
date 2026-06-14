from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from eval_da2k import (
    MODEL_CONFIGS,
    SCENE_CHOICES,
    SetupError,
    add_pair,
    empty_counts,
    finalize_counts,
    load_cv2,
    load_model,
    point_value,
    require_ready,
    resolve_device,
    scene_from_path,
    selected_annotations,
)
from eval_tome_da2k import _merge_wavg, _prepare_tome_tokens, _restore_patch_grid


METHODS = (
    "tome_actual",
    "evit_actual",
    "ats_actual",
    "token_pooling_actual",
    "pitome_actual",
    "adamerge_actual",
    "ppt_actual",
)


METHOD_NOTES = {
    "tome_actual": (
        "Faithful ToMe-style bipartite soft matching with proportional attention and "
        "size-weighted averaging. Dense-task adaptation: source maps restore the full "
        "patch grid before the Depth Anything depth head."
    ),
    "evit_actual": (
        "Near-faithful EViT inference adaptation: class-token attention after MHSA keeps "
        "attentive patch tokens and fuses inattentive patch tokens between attention and "
        "MLP. Dense-task adaptation keeps source maps instead of permanently dropping grid cells."
    ),
    "ats_actual": (
        "ATS proper-ish deterministic adaptation: significance is derived from class-token "
        "attention and value/token magnitude, with variable token counts from cumulative "
        "attention mass. Original ATS samples tokens; this evaluator uses deterministic top-mass "
        "selection and fuses unselected tokens into selected anchors for repeatable dense eval."
    ),
    "token_pooling_actual": (
        "Token Pooling proper-ish adaptation: after attention, patch tokens are clustered to "
        "minimize weighted reconstruction error, then cluster centroids replace the original "
        "tokens. A deterministic greedy agglomerative clustering path is used for practical "
        "DA-2K evaluation and source maps restore the dense patch grid."
    ),
    "pitome_actual": (
        "PiToMe near-faithful adaptation: computes a graph/similarity energy score so isolated "
        "low-energy tokens are protected and high-energy redundant tokens are candidates for "
        "bipartite merging. Exact repository-specific ordering details may differ."
    ),
    "adamerge_actual": (
        "AdaMerge near-faithful adaptation: salience is estimated by column-wise feature-affinity "
        "centrality and used in the merge score/weighted average; per-layer merge intensity is "
        "adapted from input redundancy against cached calibration statistics when available."
    ),
    "ppt_actual": (
        "PPT-style training-free hybrid: combines class-attention pruning/fusion for inattentive "
        "redundancy with Token-Pooling clustering for duplicative redundancy. This is feasible "
        "without learned thresholds; LTMP-style learned threshold modules are not implemented."
    ),
}

UNSUPPORTED_METHODS = {
    "dynamicvit": "requires trained token prediction modules",
    "a-vit": "requires learned adaptive halting behavior",
    "patchmerger": "requires trained merge modules",
    "diffrate": "requires learned/differentiated rate allocation components",
    "dtem": "requires method-specific trained components",
    "ltmp": "requires learned threshold masking modules",
    "dtop": "task-specific/trained dense-prediction token policy is not available",
}


@dataclass(frozen=True)
class ActualTokenConfig:
    dataset_root: Path
    checkpoint: Path
    output_json: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    method: str = "tome_actual"
    merge_r: int = 57
    target_ratio: float = 0.0
    ats_mass_threshold: float = 0.90
    pooling_iters: int = 4
    salience_lambda: float = 1.0
    size_lambda: float = 0.0
    calib_images: int = 0
    calib_cache: Path = Path("eval_outputs/da2k_adamerge_calib.json")
    force_calib: bool = False
    scene_type: str = ""
    max_images: int = 0
    log_every: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_json", Path(self.output_json))
        object.__setattr__(self, "calib_cache", Path(self.calib_cache))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.method not in METHODS:
            raise ValueError(f"unknown method: {self.method}")
        if self.merge_r < 0:
            raise ValueError("merge_r must be non-negative")
        if self.target_ratio and not 0.0 < self.target_ratio < 1.0:
            raise ValueError("target_ratio must be 0 or in (0, 1)")
        if not 0.0 < self.ats_mass_threshold <= 1.0:
            raise ValueError("ats_mass_threshold must be in (0, 1]")
        if self.pooling_iters < 1:
            raise ValueError("pooling_iters must be positive")
        if self.salience_lambda < 0:
            raise ValueError("salience_lambda must be non-negative")
        if self.size_lambda < 0:
            raise ValueError("size_lambda must be non-negative")
        if self.calib_images < 0:
            raise ValueError("calib_images must be non-negative")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


@dataclass
class ReductionStats:
    similarity: torch.Tensor
    class_attn: torch.Tensor
    salience: torch.Tensor
    energy: torch.Tensor
    significance: torch.Tensor
    size: torch.Tensor
    value_norm: torch.Tensor
    nearest_similarity_mean: float


def _normalize01(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    if torch.isclose(hi, lo):
        return torch.zeros_like(values)
    return (values - lo) / (hi - lo).clamp_min(1e-6)


def _proportional_attention_with_stats(
    attn: torch.nn.Module,
    x: torch.Tensor,
    sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, token_count, channels = x.shape
    head_count = int(attn.num_heads)
    head_dim = channels // head_count

    qkv = attn.qkv(x).reshape(batch, token_count, 3, head_count, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q = qkv[0] * attn.scale
    k = qkv[1]
    v = qkv[2]

    logits = q @ k.transpose(-2, -1)
    logits = logits + sizes.clamp_min(1e-6).log().transpose(1, 2).unsqueeze(1)
    attn_probs = logits.softmax(dim=-1)
    out = (attn.attn_drop(attn_probs) @ v).transpose(1, 2).reshape(batch, token_count, channels)
    out = attn.proj(out)
    out = attn.proj_drop(out)

    metric = F.normalize(k.mean(dim=1).float(), p=2, dim=-1)
    value_norm = v.float().norm(dim=-1).mean(dim=1)
    return out, metric, attn_probs.float(), value_norm


def _patch_stats(
    block_tokens: torch.Tensor,
    metric: torch.Tensor,
    attn_probs: torch.Tensor,
    value_norm: torch.Tensor,
    sizes: torch.Tensor,
    *,
    special_count: int,
) -> ReductionStats:
    patch_slice = slice(special_count, block_tokens.shape[1])
    patch_metric = metric[0, patch_slice]
    patch_count = patch_metric.shape[0]
    if patch_count == 0:
        empty = torch.empty(0, device=block_tokens.device)
        return ReductionStats(empty, empty, empty, empty, empty, empty, empty, 0.0)
    if patch_count == 1:
        one = torch.ones(1, device=block_tokens.device)
        zero = torch.zeros(1, device=block_tokens.device)
        similarity = torch.full((1, 1), -float("inf"), device=block_tokens.device)
        patch_size = sizes[0, patch_slice, 0].float()
        return ReductionStats(
            similarity=similarity,
            class_attn=one,
            salience=one,
            energy=zero,
            significance=one,
            size=patch_size / patch_size.max().clamp_min(1.0),
            value_norm=zero,
            nearest_similarity_mean=0.0,
        )

    similarity = patch_metric @ patch_metric.transpose(0, 1)
    similarity.fill_diagonal_(-float("inf"))
    finite_similarity = similarity.masked_fill(~torch.isfinite(similarity), -1.0)
    nearest_similarity = finite_similarity.max(dim=1).values

    class_attn = attn_probs[0, :, 0, patch_slice].mean(dim=0)
    incoming = attn_probs[0, :, :, patch_slice].mean(dim=(0, 1))
    outgoing = attn_probs[0, :, patch_slice, :].mean(dim=(0, 2))

    feature_affinity = torch.softmax(finite_similarity / 0.07, dim=0)
    centrality = feature_affinity.sum(dim=1)
    salience = _normalize01(0.55 * centrality + 0.45 * class_attn)
    energy = _normalize01(nearest_similarity.clamp(-1.0, 1.0))
    patch_value_norm = _normalize01(value_norm[0, patch_slice])
    significance = _normalize01(class_attn * (patch_value_norm + 1e-6))
    patch_size = sizes[0, patch_slice, 0].float()
    size = patch_size / patch_size.max().clamp_min(1.0)

    return ReductionStats(
        similarity=similarity,
        class_attn=class_attn,
        salience=salience,
        energy=energy,
        significance=significance,
        size=size,
        value_norm=patch_value_norm,
        nearest_similarity_mean=float(nearest_similarity.mean().item()),
    )


def _stats_from_current_tokens(
    x: torch.Tensor,
    sizes: torch.Tensor,
    *,
    special_count: int,
) -> ReductionStats:
    patch_x = x[:, special_count:, :].squeeze(0)
    patch_count = patch_x.shape[0]
    if patch_count == 0:
        empty = torch.empty(0, device=x.device)
        return ReductionStats(empty, empty, empty, empty, empty, empty, empty, 0.0)
    metric = F.normalize(patch_x.float(), p=2, dim=-1)
    similarity = metric @ metric.transpose(0, 1)
    similarity.fill_diagonal_(-float("inf"))
    finite_similarity = similarity.masked_fill(~torch.isfinite(similarity), -1.0)
    nearest_similarity = finite_similarity.max(dim=1).values
    token_size = sizes[:, special_count:, 0].squeeze(0).float()
    norm = _normalize01(patch_x.float().norm(dim=-1))
    size = token_size / token_size.max().clamp_min(1.0)
    return ReductionStats(
        similarity=similarity,
        class_attn=norm,
        salience=norm,
        energy=_normalize01(nearest_similarity.clamp(-1.0, 1.0)),
        significance=norm,
        size=size,
        value_norm=norm,
        nearest_similarity_mean=float(nearest_similarity.mean().item()),
    )


def _requested_r(config: ActualTokenConfig, current_patch_count: int, original_patch_count: int) -> int:
    if current_patch_count < 2:
        return 0
    if config.target_ratio > 0:
        keep = max(1, int(math.ceil(current_patch_count * config.target_ratio)))
        return min(current_patch_count - 1, max(0, current_patch_count - keep))
    return min(config.merge_r, current_patch_count - 1)


def _topk_candidate_indices(score: torch.Tensor, r: int) -> list[tuple[int, int, float]]:
    finite = torch.isfinite(score)
    valid_count = int(finite.sum().item())
    if r <= 0 or valid_count <= 0:
        return []
    candidate_count = min(valid_count, max(1024, r * 128))
    flat_score = score.flatten()
    values, indices = torch.topk(flat_score, k=candidate_count, largest=True, sorted=True)
    width = score.shape[1]
    candidates: list[tuple[int, int, float]] = []
    for value, flat_index in zip(values.tolist(), indices.tolist()):
        if math.isfinite(value):
            candidates.append((flat_index // width, flat_index % width, float(value)))
    return candidates


def _merge_local_pairs(
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    pairs: list[tuple[int, int]],
    *,
    special_count: int,
    salience: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if not pairs:
        return x, sizes, sources, 0

    patch_count = x.shape[1] - special_count
    source_locals = {int(src) for src, _dst in pairs}
    pairs = [
        (int(src), int(dst))
        for src, dst in pairs
        if 0 <= int(src) < patch_count
        and 0 <= int(dst) < patch_count
        and int(src) != int(dst)
        and int(dst) not in source_locals
    ]
    if not pairs:
        return x, sizes, sources, 0

    source_locals = {src for src, _dst in pairs}
    keep_patch_locals = [index for index in range(patch_count) if index not in source_locals]
    keep_idx = torch.as_tensor(
        [*range(special_count), *[special_count + index for index in keep_patch_locals]],
        dtype=torch.long,
        device=x.device,
    )
    new_x = x.index_select(1, keep_idx).clone()
    new_sizes = sizes.index_select(1, keep_idx).clone()
    new_sources = sources.index_select(1, keep_idx).clone()
    old_to_new = {old: new for new, old in enumerate(keep_patch_locals)}

    merged = 0
    for src_local, dst_local in pairs:
        dst_new_local = old_to_new.get(dst_local)
        if dst_new_local is None:
            continue
        src_idx = special_count + src_local
        dst_new_idx = special_count + dst_new_local
        src_size = sizes[:, src_idx]
        dst_size = new_sizes[:, dst_new_idx]
        if salience is None:
            src_weight = src_size
            dst_weight = dst_size
        else:
            src_weight = src_size * (1.0 + salience[src_local]).to(src_size.dtype)
            dst_weight = dst_size * (1.0 + salience[dst_local]).to(dst_size.dtype)
        new_x[:, dst_new_idx] = (new_x[:, dst_new_idx] * dst_weight + x[:, src_idx] * src_weight) / (
            dst_weight + src_weight
        ).clamp_min(1e-6)
        new_sizes[:, dst_new_idx] = dst_size + src_size
        new_sources[:, dst_new_idx] = new_sources[:, dst_new_idx] + sources[:, src_idx]
        merged += 1

    return new_x, new_sizes, new_sources, merged


def _bipartite_soft_matching_merge(
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    metric: torch.Tensor,
    *,
    merge_r: int,
    special_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if merge_r <= 0 or x.shape[1] - special_count < 2:
        return x, sizes, sources, 0
    if x.shape[0] != 1:
        raise ValueError("actual token evaluator currently supports batch size 1")

    patch_token_count = x.shape[1] - special_count
    device = x.device
    rel = torch.arange(patch_token_count, device=device)
    a_idx = special_count + rel[0::2]
    b_idx = special_count + rel[1::2]
    if a_idx.numel() == 0 or b_idx.numel() == 0:
        return x, sizes, sources, 0

    scores = metric[0, a_idx] @ metric[0, b_idx].transpose(0, 1)
    best_scores, best_b_local = scores.max(dim=1)
    r = min(int(merge_r), int(best_scores.numel()))
    if r <= 0:
        return x, sizes, sources, 0

    selected_a_local = torch.argsort(best_scores, descending=True)[:r]
    selected_mask = torch.zeros(a_idx.numel(), dtype=torch.bool, device=device)
    selected_mask[selected_a_local] = True
    unmerged_a_idx = a_idx[~selected_mask]

    keep_idx = torch.cat((torch.arange(special_count, device=device), unmerged_a_idx, b_idx))
    new_x = x.index_select(1, keep_idx).clone()
    new_sizes = sizes.index_select(1, keep_idx).clone()
    new_sources = sources.index_select(1, keep_idx).clone()

    b_new_offset = special_count + unmerged_a_idx.numel()
    selected_a_idx = a_idx[selected_a_local]
    selected_b_new_idx = b_new_offset + best_b_local[selected_a_local]

    for src_idx, dst_new_idx in zip(selected_a_idx.tolist(), selected_b_new_idx.tolist()):
        src_size = sizes[:, src_idx]
        dst_size = new_sizes[:, dst_new_idx]
        new_x[:, dst_new_idx] = _merge_wavg(new_x[:, dst_new_idx], x[:, src_idx], dst_size, src_size)
        new_sizes[:, dst_new_idx] = dst_size + src_size
        new_sources[:, dst_new_idx] = new_sources[:, dst_new_idx] + sources[:, src_idx]

    return new_x, new_sizes, new_sources, r


def _greedy_pairs(
    score: torch.Tensor,
    *,
    merge_r: int,
    allowed_source: torch.Tensor,
    allowed_dst: torch.Tensor,
    source_preference: torch.Tensor,
) -> list[tuple[int, int]]:
    if merge_r <= 0:
        return []
    pair_score = score.clone()
    pair_score = pair_score.masked_fill(~allowed_source[:, None], -float("inf"))
    pair_score = pair_score.masked_fill(~allowed_dst[None, :], -float("inf"))
    pair_score.fill_diagonal_(-float("inf"))

    touched: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for left, right, _value in _topk_candidate_indices(pair_score, merge_r):
        if left in touched or right in touched:
            continue
        if source_preference[left] <= source_preference[right]:
            src, dst = left, right
        else:
            src, dst = right, left
        if not bool(allowed_source[src].item()) or not bool(allowed_dst[dst].item()):
            src, dst = dst, src
        if not bool(allowed_source[src].item()) or not bool(allowed_dst[dst].item()):
            continue
        pairs.append((src, dst))
        touched.add(left)
        touched.add(right)
        if len(pairs) >= merge_r:
            break
    return pairs


def _evit_fuse(
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    stats: ReductionStats,
    *,
    merge_r: int,
    special_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    patch_count = x.shape[1] - special_count
    if merge_r <= 0 or patch_count < 3:
        return x, sizes, sources, 0
    inattentive_count = min(patch_count - 1, merge_r + 1)
    keep_count = patch_count - inattentive_count
    if keep_count <= 0:
        return x, sizes, sources, 0

    keep_local = torch.sort(torch.topk(stats.class_attn, k=keep_count, largest=True).indices).values
    keep_mask = torch.zeros(patch_count, dtype=torch.bool, device=x.device)
    keep_mask[keep_local] = True
    drop_local = torch.nonzero(~keep_mask, as_tuple=False).flatten()
    if drop_local.numel() <= 1:
        return x, sizes, sources, 0

    keep_idx = torch.as_tensor(
        [*range(special_count), *[special_count + int(index) for index in keep_local.tolist()]],
        dtype=torch.long,
        device=x.device,
    )
    new_x = x.index_select(1, keep_idx).clone()
    new_sizes = sizes.index_select(1, keep_idx).clone()
    new_sources = sources.index_select(1, keep_idx).clone()

    drop_idx = special_count + drop_local
    weights = (stats.class_attn[drop_local].to(x.dtype) * sizes[:, drop_idx, 0].flatten()).clamp_min(1e-6)
    fused = (x[:, drop_idx, :] * weights.view(1, -1, 1)).sum(dim=1, keepdim=True) / weights.sum().clamp_min(1e-6)
    fused_size = sizes[:, drop_idx, :].sum(dim=1, keepdim=True)
    fused_sources = sources[:, drop_idx, :].sum(dim=1, keepdim=True)

    new_x = torch.cat((new_x, fused), dim=1)
    new_sizes = torch.cat((new_sizes, fused_size), dim=1)
    new_sources = torch.cat((new_sources, fused_sources), dim=1)
    return new_x, new_sizes, new_sources, int(drop_local.numel() - 1)


def _merge_unselected_to_anchors(
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    keep_mask: torch.Tensor,
    stats: ReductionStats,
    *,
    special_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    patch_count = x.shape[1] - special_count
    keep_local = torch.nonzero(keep_mask, as_tuple=False).flatten()
    drop_local = torch.nonzero(~keep_mask, as_tuple=False).flatten()
    if keep_local.numel() == 0 or drop_local.numel() == 0:
        return x, sizes, sources, 0

    keep_idx = torch.as_tensor(
        [*range(special_count), *[special_count + int(index) for index in keep_local.tolist()]],
        dtype=torch.long,
        device=x.device,
    )
    new_x = x.index_select(1, keep_idx).clone()
    new_sizes = sizes.index_select(1, keep_idx).clone()
    new_sources = sources.index_select(1, keep_idx).clone()

    anchor_lookup = {int(old): new for new, old in enumerate(keep_local.tolist())}
    for src_local in drop_local.tolist():
        anchor_scores = stats.similarity[src_local, keep_local] + stats.significance[keep_local]
        dst_old = int(keep_local[int(torch.argmax(anchor_scores).item())].item())
        dst_new_idx = special_count + anchor_lookup[dst_old]
        src_idx = special_count + int(src_local)
        src_size = sizes[:, src_idx]
        dst_size = new_sizes[:, dst_new_idx]
        new_x[:, dst_new_idx] = _merge_wavg(new_x[:, dst_new_idx], x[:, src_idx], dst_size, src_size)
        new_sizes[:, dst_new_idx] = dst_size + src_size
        new_sources[:, dst_new_idx] = new_sources[:, dst_new_idx] + sources[:, src_idx]

    return new_x, new_sizes, new_sources, int(patch_count - keep_local.numel())


def _ats_reduce(
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    stats: ReductionStats,
    *,
    max_merge_r: int,
    special_count: int,
    mass_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    patch_count = x.shape[1] - special_count
    if max_merge_r <= 0 or patch_count < 2:
        return x, sizes, sources, 0, 0

    score = (stats.significance + 1e-6).float()
    order = torch.argsort(score, descending=True)
    probs = score[order] / score.sum().clamp_min(1e-6)
    cumulative = torch.cumsum(probs, dim=0)
    mass_keep = int((cumulative < mass_threshold).sum().item()) + 1
    requested_keep = max(1, patch_count - max_merge_r)
    keep_count = max(requested_keep, min(patch_count, mass_keep))
    if keep_count >= patch_count:
        return x, sizes, sources, 0, 0

    keep_mask = torch.zeros(patch_count, dtype=torch.bool, device=x.device)
    keep_mask[order[:keep_count]] = True
    new_x, new_sizes, new_sources, merged = _merge_unselected_to_anchors(
        x,
        sizes,
        sources,
        keep_mask,
        stats,
        special_count=special_count,
    )
    requested = patch_count - keep_count
    return new_x, new_sizes, new_sources, requested, merged


def _token_pooling_reduce(
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    stats: ReductionStats,
    *,
    merge_r: int,
    special_count: int,
    iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    patch_count = x.shape[1] - special_count
    if merge_r <= 0 or patch_count < 2:
        return x, sizes, sources, 0
    usable_r = min(merge_r, patch_count - 1)
    if stats.salience.numel() != patch_count:
        stats = _stats_from_current_tokens(x, sizes, special_count=special_count)
    source_mask = torch.ones(patch_count, dtype=torch.bool, device=x.device)
    score = stats.similarity - 0.10 * (stats.size[:, None] + stats.size[None, :])
    score = score - 0.05 * (stats.salience[:, None] + stats.salience[None, :])
    pairs = _greedy_pairs(
        score,
        merge_r=usable_r,
        allowed_source=source_mask,
        allowed_dst=source_mask,
        source_preference=stats.salience + 0.5 * stats.size,
    )
    return _merge_local_pairs(x, sizes, sources, pairs, special_count=special_count)


def _pitome_reduce(
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    stats: ReductionStats,
    *,
    merge_r: int,
    special_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    patch_count = x.shape[1] - special_count
    if merge_r <= 0 or patch_count < 2:
        return x, sizes, sources, 0
    high_energy_cutoff = torch.quantile(stats.energy, 0.40)
    source_mask = stats.energy >= high_energy_cutoff
    if int(source_mask.sum().item()) < 2:
        source_mask = torch.ones_like(source_mask, dtype=torch.bool)
    score = stats.similarity + 0.35 * (stats.energy[:, None] + stats.energy[None, :])
    score = score - 0.15 * (1.0 - stats.salience[None, :])
    pairs = _greedy_pairs(
        score,
        merge_r=merge_r,
        allowed_source=source_mask,
        allowed_dst=torch.ones(patch_count, dtype=torch.bool, device=x.device),
        source_preference=stats.salience - stats.energy,
    )
    return _merge_local_pairs(x, sizes, sources, pairs, special_count=special_count)


def _adamerge_adaptive_r(
    base_r: int,
    stats: ReductionStats,
    calib_stats: dict[str, Any] | None,
    block_index: int,
) -> int:
    patch_count = stats.salience.numel()
    if base_r <= 0 or patch_count < 2:
        return 0
    base_r = min(base_r, patch_count - 1)
    if not calib_stats:
        redundancy_scale = 0.45 + 0.75 * max(0.0, min(1.0, (stats.nearest_similarity_mean + 1.0) * 0.5))
        return min(patch_count - 1, max(1, int(round(base_r * redundancy_scale))))

    per_block = calib_stats.get("per_block", {})
    record = per_block.get(str(block_index), {})
    mean = float(record.get("nearest_similarity_mean", stats.nearest_similarity_mean))
    std = max(float(record.get("nearest_similarity_std", 0.05)), 1e-3)
    z = (stats.nearest_similarity_mean - mean) / std
    scale = 0.45 + 0.9 * (1.0 / (1.0 + math.exp(-z)))
    return min(patch_count - 1, max(1, int(round(base_r * scale))))


def _adamerge_reduce(
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    stats: ReductionStats,
    *,
    merge_r: int,
    special_count: int,
    config: ActualTokenConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    patch_count = x.shape[1] - special_count
    if merge_r <= 0 or patch_count < 2:
        return x, sizes, sources, 0
    source_mask = torch.ones(patch_count, dtype=torch.bool, device=x.device)
    dst_mask = torch.ones(patch_count, dtype=torch.bool, device=x.device)
    score = stats.similarity * (1.0 + config.salience_lambda * stats.salience[None, :])
    score = score - config.salience_lambda * stats.salience[:, None]
    score = score - config.size_lambda * stats.size[:, None]
    pairs = _greedy_pairs(
        score,
        merge_r=merge_r,
        allowed_source=source_mask,
        allowed_dst=dst_mask,
        source_preference=stats.salience + 0.25 * stats.size,
    )
    return _merge_local_pairs(x, sizes, sources, pairs, special_count=special_count, salience=stats.salience)


def _ppt_reduce(
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    stats: ReductionStats,
    *,
    merge_r: int,
    special_count: int,
    config: ActualTokenConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if merge_r <= 0:
        return x, sizes, sources, 0
    patch_count = x.shape[1] - special_count
    prune_budget = min(merge_r, max(0, merge_r // 2))
    merged_total = 0
    if prune_budget > 0 and patch_count - prune_budget >= 2:
        keep_count = patch_count - prune_budget
        keep_mask = torch.zeros(patch_count, dtype=torch.bool, device=x.device)
        keep_mask[torch.topk(stats.class_attn, k=keep_count, largest=True).indices] = True
        x, sizes, sources, merged = _merge_unselected_to_anchors(
            x,
            sizes,
            sources,
            keep_mask,
            stats,
            special_count=special_count,
        )
        merged_total += merged

    remaining = merge_r - merged_total
    if remaining <= 0:
        return x, sizes, sources, merged_total

    refreshed_stats = _stats_from_current_tokens(x, sizes, special_count=special_count)
    x, sizes, sources, pooled = _token_pooling_reduce(
        x,
        sizes,
        sources,
        refreshed_stats,
        merge_r=min(remaining, x.shape[1] - special_count - 1),
        special_count=special_count,
        iters=config.pooling_iters,
    )
    return x, sizes, sources, merged_total + pooled


def _run_actual_block(
    block: torch.nn.Module,
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    *,
    config: ActualTokenConfig,
    special_count: int,
    original_patch_count: int,
    block_index: int,
    calib_stats: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    normed = block.norm1(x)
    attn_out, metric, attn_probs, value_norm = _proportional_attention_with_stats(block.attn, normed, sizes)
    x = x + block.drop_path1(block.ls1(attn_out))

    stats = _patch_stats(normed, metric, attn_probs, value_norm, sizes, special_count=special_count)
    requested = _requested_r(config, x.shape[1] - special_count, original_patch_count)

    if config.method == "tome_actual":
        x, sizes, sources, merged = _bipartite_soft_matching_merge(
            x,
            sizes,
            sources,
            metric,
            merge_r=requested,
            special_count=special_count,
        )
    elif config.method == "evit_actual":
        x, sizes, sources, merged = _evit_fuse(
            x,
            sizes,
            sources,
            stats,
            merge_r=requested,
            special_count=special_count,
        )
    elif config.method == "ats_actual":
        x, sizes, sources, requested, merged = _ats_reduce(
            x,
            sizes,
            sources,
            stats,
            max_merge_r=requested,
            special_count=special_count,
            mass_threshold=config.ats_mass_threshold,
        )
    elif config.method == "token_pooling_actual":
        x, sizes, sources, merged = _token_pooling_reduce(
            x,
            sizes,
            sources,
            stats,
            merge_r=requested,
            special_count=special_count,
            iters=config.pooling_iters,
        )
    elif config.method == "pitome_actual":
        x, sizes, sources, merged = _pitome_reduce(
            x,
            sizes,
            sources,
            stats,
            merge_r=requested,
            special_count=special_count,
        )
    elif config.method == "adamerge_actual":
        requested = _adamerge_adaptive_r(requested, stats, calib_stats, block_index)
        x, sizes, sources, merged = _adamerge_reduce(
            x,
            sizes,
            sources,
            stats,
            merge_r=requested,
            special_count=special_count,
            config=config,
        )
    elif config.method == "ppt_actual":
        x, sizes, sources, merged = _ppt_reduce(
            x,
            sizes,
            sources,
            stats,
            merge_r=requested,
            special_count=special_count,
            config=config,
        )
    else:
        raise ValueError(f"unknown method: {config.method}")

    x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
    return x, sizes, sources, requested, merged


def get_actual_intermediate_layers(
    vit: torch.nn.Module,
    x: torch.Tensor,
    layers: list[int],
    *,
    config: ActualTokenConfig,
    calib_stats: dict[str, Any] | None = None,
    merge_counter: Counter[int] | None = None,
    requested_counter: Counter[int] | None = None,
    token_counter: Counter[int] | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    sequence, sizes, sources, patch_count, special_count = _prepare_tome_tokens(vit, x)
    layers_to_take = set(layers)
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    for block_index, block in enumerate(vit.blocks):
        sequence, sizes, sources, requested_r, merged = _run_actual_block(
            block,
            sequence,
            sizes,
            sources,
            config=config,
            special_count=special_count,
            original_patch_count=patch_count,
            block_index=block_index,
            calib_stats=calib_stats,
        )
        if merge_counter is not None:
            merge_counter[block_index] += merged
        if requested_counter is not None:
            requested_counter[block_index] += requested_r
        if token_counter is not None:
            token_counter[block_index] += sequence.shape[1] - special_count
        if block_index not in layers_to_take:
            continue
        normalized = vit.norm(sequence)
        outputs.append(_restore_patch_grid(normalized, sources, patch_count=patch_count, special_count=special_count))

    if len(outputs) != len(layers):
        raise RuntimeError(f"only captured {len(outputs)} / {len(layers)} requested layers")
    return tuple(outputs)


class ActualTokenDepthAnything(torch.nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        config: ActualTokenConfig,
        calib_stats: dict[str, Any] | None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.calib_stats = calib_stats
        self.merge_counter: Counter[int] = Counter()
        self.requested_counter: Counter[int] = Counter()
        self.token_counter: Counter[int] = Counter()

    def image2tensor(self, raw_image, input_size: int = 518):
        return self.base_model.image2tensor(raw_image, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        layers = self.base_model.intermediate_layer_idx[self.base_model.encoder]
        features = get_actual_intermediate_layers(
            self.base_model.pretrained,
            x,
            layers,
            config=self.config,
            calib_stats=self.calib_stats,
            merge_counter=self.merge_counter,
            requested_counter=self.requested_counter,
            token_counter=self.token_counter,
        )
        depth = self.base_model.depth_head(features, patch_h, patch_w)
        return F.relu(depth).squeeze(1)


@torch.no_grad()
def infer_depth(model: torch.nn.Module, image, input_size: int, device: torch.device) -> torch.Tensor:
    tensor, (height, width) = model.image2tensor(image, input_size)
    tensor = tensor.to(device)
    depth = model(tensor)
    depth = F.interpolate(depth[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
    return depth.detach().float().cpu()


@torch.no_grad()
def _calibration_stats_for_tensor(vit: torch.nn.Module, tensor: torch.Tensor) -> list[float]:
    sequence, sizes, _sources, _patch_count, special_count = _prepare_tome_tokens(vit, tensor)
    means: list[float] = []
    for block in vit.blocks:
        normed = block.norm1(sequence)
        attn_out, metric, attn_probs, value_norm = _proportional_attention_with_stats(block.attn, normed, sizes)
        stats = _patch_stats(normed, metric, attn_probs, value_norm, sizes, special_count=special_count)
        means.append(stats.nearest_similarity_mean)
        sequence = sequence + block.drop_path1(block.ls1(attn_out))
        sequence = sequence + block.drop_path2(block.ls2(block.mlp(block.norm2(sequence))))
    return means


def _load_or_build_adamerge_calibration(
    dense_model: torch.nn.Module,
    config: ActualTokenConfig,
    selected: list[tuple[str, list[dict[str, Any]]]],
    cv2_module: Any,
    device: torch.device,
) -> tuple[dict[str, Any] | None, str]:
    if config.method != "adamerge_actual":
        return None, "not_used"
    if config.calib_cache.is_file() and not config.force_calib:
        try:
            cached = json.loads(config.calib_cache.read_text())
            if (
                cached.get("encoder") == config.encoder
                and int(cached.get("input_size", -1)) == config.input_size
                and int(cached.get("block_count", -1)) == len(dense_model.pretrained.blocks)
            ):
                return cached, "loaded"
        except json.JSONDecodeError:
            pass
    if config.calib_images <= 0:
        return None, "online_fallback_no_cache"

    per_image: list[list[float]] = []
    for relative_path, _pairs in selected[: config.calib_images]:
        image = cv2_module.imread(str(config.dataset_root / relative_path))
        if image is None:
            continue
        tensor, _shape = dense_model.image2tensor(image, config.input_size)
        tensor = tensor.to(device)
        per_image.append(_calibration_stats_for_tensor(dense_model.pretrained, tensor))
    if not per_image:
        return None, "online_fallback_no_calibration_images"

    values = torch.tensor(per_image, dtype=torch.float32)
    per_block = {}
    for block_index in range(values.shape[1]):
        column = values[:, block_index]
        per_block[str(block_index)] = {
            "nearest_similarity_mean": float(column.mean().item()),
            "nearest_similarity_std": float(column.std(unbiased=False).item()),
        }
    stats = {
        "method": "adamerge_actual",
        "encoder": config.encoder,
        "input_size": config.input_size,
        "checkpoint": str(config.checkpoint),
        "block_count": len(dense_model.pretrained.blocks),
        "calibration_images_used": len(per_image),
        "created_at_unix": time.time(),
        "per_block": per_block,
    }
    config.calib_cache.parent.mkdir(parents=True, exist_ok=True)
    config.calib_cache.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return stats, "built"


def evaluate(config: ActualTokenConfig) -> dict[str, Any]:
    require_ready(config.dataset_root, config.checkpoint, config.encoder)
    cv2 = load_cv2()
    device = resolve_device(config.device)
    dense_model = load_model(config.encoder, config.checkpoint, device)
    for param in dense_model.parameters():
        param.requires_grad_(False)

    selected = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
    )
    if not selected:
        raise RuntimeError("no DA-2K annotations selected")

    calib_stats, calib_status = _load_or_build_adamerge_calibration(dense_model, config, selected, cv2, device)
    model = ActualTokenDepthAnything(dense_model, config=config, calib_stats=calib_stats).to(device).eval()

    total = empty_counts()
    by_scene = defaultdict(empty_counts)
    missing_images: list[str] = []
    started = time.monotonic()

    for index, (relative_path, pairs) in enumerate(selected, start=1):
        image_path = config.dataset_root / relative_path
        image = cv2.imread(str(image_path))
        if image is None:
            missing_images.append(str(image_path))
            continue
        depth = infer_depth(model, image, config.input_size, device)
        scene = scene_from_path(relative_path)
        for pair in pairs:
            if pair.get("closer_point") != "point1":
                raise ValueError(f"unsupported closer_point in {relative_path}: {pair}")
            d1 = point_value(depth, pair["point1"])
            d2 = point_value(depth, pair["point2"])
            add_pair(total, d1, d2)
            add_pair(by_scene[scene], d1, d2)
        if config.log_every > 0 and (index % config.log_every == 0 or index == len(selected)):
            print(f"{config.method}: evaluated {index}/{len(selected)} images", flush=True)

    block_count = len(dense_model.pretrained.blocks)
    evaluated_images = max(1, len(selected) - len(missing_images))
    patch_count = (config.input_size // 14) * (config.input_size // 14)
    result = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "variant": config.method,
            "method_note": METHOD_NOTES[config.method],
            "images_requested": len(selected),
            "missing_images": missing_images,
            "elapsed_seconds": time.monotonic() - started,
            "patch_count_at_square_input": patch_count,
            "requested_merges_per_layer": config.merge_r,
            "target_ratio_effective": config.target_ratio if config.target_ratio > 0 else None,
            "adamerge_calibration_status": calib_status,
            "restores_full_patch_grid_before_depth_head": True,
            "actual_merges_by_block": {
                str(block_index): int(model.merge_counter.get(block_index, 0))
                for block_index in range(block_count)
            },
            "requested_merges_by_block": {
                str(block_index): int(model.requested_counter.get(block_index, 0))
                for block_index in range(block_count)
            },
            "mean_patch_tokens_after_block": {
                str(block_index): float(model.token_counter.get(block_index, 0)) / evaluated_images
                for block_index in range(block_count)
            },
            "unsupported_learned_methods": UNSUPPORTED_METHODS,
            "notes": [
                "All implemented methods are training-free. Methods needing trained predictors or learned threshold modules are explicitly unsupported.",
                "Reduction runs inside the ViT blocks, but source maps expand captured intermediate features back to the original patch count before DA-V2 depth_head.",
                "The evaluator supports batch size 1, matching the existing DA-2K scripts.",
                "Primary references consulted: ToMe arXiv:2210.09461, EViT arXiv:2202.07800, ATS arXiv:2111.15667, Token Pooling arXiv:2110.03860, PiToMe arXiv:2405.16148, AdaMerge arXiv:2605.27465, PPT arXiv:2310.01812.",
            ],
            "rule": "DA-2K labels point1 as closer; Depth Anything V2 relative depth normally uses larger predicted values for closer points.",
        },
        "overall": finalize_counts(total),
        "by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(by_scene.items())},
    }
    if config.output_json:
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate paper-derived training-free token reduction methods for Depth Anything V2 on DA-2K."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/da2k_vits_tome_actual_r57.json"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--method", choices=METHODS, default="tome_actual")
    parser.add_argument("--merge-r", type=int, default=57)
    parser.add_argument("--target-ratio", type=float, default=0.0, help="If >0, keep this patch-token ratio per block.")
    parser.add_argument("--ats-mass-threshold", type=float, default=0.90)
    parser.add_argument("--pooling-iters", type=int, default=4)
    parser.add_argument("--salience-lambda", type=float, default=1.0)
    parser.add_argument("--size-lambda", type=float, default=0.0)
    parser.add_argument("--calib-images", type=int, default=0)
    parser.add_argument("--calib-cache", type=Path, default=Path("eval_outputs/da2k_adamerge_calib.json"))
    parser.add_argument("--force-calib", action="store_true")
    parser.add_argument("--scene-type", default="", choices=SCENE_CHOICES)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ActualTokenConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        method=args.method,
        merge_r=args.merge_r,
        target_ratio=args.target_ratio,
        ats_mass_threshold=args.ats_mass_threshold,
        pooling_iters=args.pooling_iters,
        salience_lambda=args.salience_lambda,
        size_lambda=args.size_lambda,
        calib_images=args.calib_images,
        calib_cache=args.calib_cache,
        force_calib=args.force_calib,
        scene_type=args.scene_type,
        max_images=args.max_images,
        log_every=args.log_every,
    )
    try:
        summary = evaluate(config)
    except SetupError as exc:
        raise SystemExit(str(exc))
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print(f"wrote {config.output_json}")


if __name__ == "__main__":
    main()
