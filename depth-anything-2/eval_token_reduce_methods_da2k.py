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
from eval_tome_da2k import (
    _merge_wavg,
    _prepare_tome_tokens,
    _restore_patch_grid,
)


METHODS = ("pitome", "adamerge", "mctf", "evit", "ats", "ppt")

METHOD_NOTES = {
    "pitome": (
        "PiToMe-style dense-safe proxy: protects high class-attention/activation-energy tokens "
        "and merges only low-informativeness similar tokens. This is not a faithful reproduction "
        "of PiToMe internals."
    ),
    "adamerge": (
        "AdaMerge-style dense-safe proxy: salience-weighted similarity with adaptive per-layer "
        "merge counts. This uses no learned merge controller."
    ),
    "mctf": (
        "MCTF-style dense-safe proxy: multi-criteria merge score combining similarity, "
        "informativeness/salience penalties, and token-size penalties."
    ),
    "evit": (
        "EViT-style dense-safe proxy: keeps class-attended tokens and fuses inattentive tokens "
        "into retained anchors instead of deleting them. This is an inference-only adaptation."
    ),
    "ats": (
        "ATS-style dense-safe proxy: deterministic attention-significance fusion proxy using "
        "class/incoming/outgoing attention and activation energy, with no stochastic sampler."
    ),
    "ppt": (
        "PPT-style dense-safe hybrid proxy: fuses low-salience trash tokens and merges redundant "
        "medium-salience tokens, while restoring the full patch grid for the DA-V2 depth head."
    ),
}


@dataclass(frozen=True)
class TokenReduceConfig:
    dataset_root: Path
    checkpoint: Path
    output_json: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    method: str = "pitome"
    merge_r: int = 57
    adaptive: bool = False
    salience_lambda: float = 1.0
    size_lambda: float = 0.25
    protect_fraction: float = 0.15
    scene_type: str = ""
    max_images: int = 0
    log_every: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_json", Path(self.output_json))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.method not in METHODS:
            raise ValueError(f"unknown method: {self.method}")
        if self.merge_r < 0:
            raise ValueError("merge_r must be non-negative")
        if self.salience_lambda < 0:
            raise ValueError("salience_lambda must be non-negative")
        if self.size_lambda < 0:
            raise ValueError("size_lambda must be non-negative")
        if not 0.0 <= self.protect_fraction < 1.0:
            raise ValueError("protect_fraction must be in [0, 1)")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return out, metric, attn_probs.float()


@dataclass
class PatchStats:
    similarity: torch.Tensor
    salience: torch.Tensor
    energy: torch.Tensor
    informativeness: torch.Tensor
    significance: torch.Tensor
    size: torch.Tensor
    protect_mask: torch.Tensor


def _patch_stats(
    block_input: torch.Tensor,
    metric: torch.Tensor,
    attn_probs: torch.Tensor,
    sizes: torch.Tensor,
    *,
    special_count: int,
    protect_fraction: float,
) -> PatchStats:
    patch_slice = slice(special_count, block_input.shape[1])
    patch_metric = metric[0, patch_slice]
    similarity = patch_metric @ patch_metric.transpose(0, 1)
    similarity.fill_diagonal_(-float("inf"))

    patch_count = patch_metric.shape[0]
    if patch_count == 0:
        empty = torch.empty(0, device=block_input.device)
        return PatchStats(similarity, empty, empty, empty, empty, empty, torch.empty(0, dtype=torch.bool, device=block_input.device))

    class_attn = attn_probs[0, :, 0, patch_slice].mean(dim=0)
    incoming = attn_probs[0, :, :, patch_slice].mean(dim=(0, 1))
    outgoing = attn_probs[0, :, patch_slice, :].mean(dim=(0, 2))
    salience = _normalize01(class_attn + 0.5 * incoming + 0.25 * outgoing)

    energy = _normalize01(block_input[0, patch_slice].float().norm(dim=-1))
    informativeness = _normalize01(0.6 * salience + 0.4 * energy)
    significance = _normalize01(0.5 * salience + 0.3 * incoming + 0.2 * energy)
    patch_size = sizes[0, patch_slice, 0].float()
    size = patch_size / patch_size.max().clamp_min(1.0)

    protect_count = min(patch_count - 1, int(round(patch_count * protect_fraction)))
    protect_mask = torch.zeros(patch_count, dtype=torch.bool, device=block_input.device)
    if protect_count > 0:
        protected = torch.topk(informativeness, k=protect_count, largest=True).indices
        protect_mask[protected] = True

    return PatchStats(
        similarity=similarity,
        salience=salience,
        energy=energy,
        informativeness=informativeness,
        significance=significance,
        size=size,
        protect_mask=protect_mask,
    )


def _adaptive_merge_r(base_r: int, stats: PatchStats, *, enabled: bool) -> int:
    patch_count = stats.informativeness.numel()
    if base_r <= 0 or patch_count < 2:
        return 0
    if not enabled:
        return min(base_r, patch_count - 1)

    finite_similarity = stats.similarity.masked_fill(~torch.isfinite(stats.similarity), -1.0)
    nearest = finite_similarity.max(dim=1).values
    redundancy = ((nearest.clamp(-1.0, 1.0) + 1.0) * 0.5).mean()
    low_salience = 1.0 - stats.informativeness.mean()
    scale = (0.35 + 0.65 * (0.7 * redundancy + 0.3 * low_salience)).clamp(0.15, 1.25)
    return min(max(1, int(round(base_r * float(scale.item())))), patch_count - 1)


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
        if not math.isfinite(value):
            continue
        candidates.append((flat_index // width, flat_index % width, float(value)))
    return candidates


def _greedy_redundant_pairs(
    score: torch.Tensor,
    *,
    merge_r: int,
    allowed_mask: torch.Tensor,
    source_preference: torch.Tensor,
) -> list[tuple[int, int]]:
    if merge_r <= 0:
        return []
    pair_score = score.clone()
    pair_score = pair_score.masked_fill(~allowed_mask[:, None], -float("inf"))
    pair_score = pair_score.masked_fill(~allowed_mask[None, :], -float("inf"))
    pair_score = torch.triu(pair_score, diagonal=1)
    pair_score = pair_score.masked_fill(pair_score == 0, -float("inf"))

    touched: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for left, right, _value in _topk_candidate_indices(pair_score, merge_r):
        if left in touched or right in touched:
            continue
        if source_preference[left] <= source_preference[right]:
            src, dst = left, right
        else:
            src, dst = right, left
        pairs.append((src, dst))
        touched.add(left)
        touched.add(right)
        if len(pairs) >= merge_r:
            break
    return pairs


def _fuse_low_to_anchors(
    similarity: torch.Tensor,
    *,
    merge_r: int,
    source_score: torch.Tensor,
    anchor_score: torch.Tensor,
    source_mask: torch.Tensor,
    anchor_mask: torch.Tensor,
    salience_lambda: float,
) -> list[tuple[int, int]]:
    if merge_r <= 0:
        return []
    source_candidates = torch.argsort(source_score, descending=False).tolist()
    anchor_bonus = salience_lambda * anchor_score
    pairs: list[tuple[int, int]] = []
    used_sources: set[int] = set()
    for src in source_candidates:
        if len(pairs) >= merge_r:
            break
        if src in used_sources or not bool(source_mask[src].item()):
            continue
        dst_mask = anchor_mask.clone()
        dst_mask[src] = False
        if not bool(dst_mask.any().item()):
            continue
        dst_score = similarity[src].clone() + anchor_bonus
        dst_score = dst_score.masked_fill(~dst_mask, -float("inf"))
        dst = int(torch.argmax(dst_score).item())
        if not math.isfinite(float(dst_score[dst].item())):
            continue
        pairs.append((src, dst))
        used_sources.add(src)
    return pairs


def _merge_local_pairs(
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    pairs: list[tuple[int, int]],
    *,
    special_count: int,
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
        new_x[:, dst_new_idx] = _merge_wavg(new_x[:, dst_new_idx], x[:, src_idx], dst_size, src_size)
        new_sizes[:, dst_new_idx] = dst_size + src_size
        new_sources[:, dst_new_idx] = new_sources[:, dst_new_idx] + sources[:, src_idx]
        merged += 1

    return new_x, new_sizes, new_sources, merged


def _pitome_pairs(stats: PatchStats, merge_r: int, config: TokenReduceConfig) -> list[tuple[int, int]]:
    low_energy_limit = torch.quantile(stats.informativeness, 0.55)
    allowed = (stats.informativeness <= low_energy_limit) & ~stats.protect_mask
    score = stats.similarity - config.salience_lambda * (
        stats.informativeness[:, None] + stats.informativeness[None, :]
    ) * 0.5
    score = score - config.size_lambda * (stats.size[:, None] + stats.size[None, :]) * 0.5
    return _greedy_redundant_pairs(
        score,
        merge_r=merge_r,
        allowed_mask=allowed,
        source_preference=stats.informativeness + 0.25 * stats.energy,
    )


def _adamerge_pairs(stats: PatchStats, merge_r: int, config: TokenReduceConfig) -> list[tuple[int, int]]:
    allowed = ~stats.protect_mask
    salience_weight = 1.0 - config.salience_lambda * (
        stats.informativeness[:, None] + stats.informativeness[None, :]
    ) * 0.5
    score = stats.similarity * salience_weight - config.size_lambda * stats.size[:, None]
    return _greedy_redundant_pairs(
        score,
        merge_r=merge_r,
        allowed_mask=allowed,
        source_preference=stats.informativeness,
    )


def _mctf_pairs(stats: PatchStats, merge_r: int, config: TokenReduceConfig) -> list[tuple[int, int]]:
    allowed = ~stats.protect_mask
    info_penalty = (stats.informativeness[:, None] + stats.informativeness[None, :]) * 0.5
    size_penalty = (stats.size[:, None] + stats.size[None, :]) * 0.5
    score = stats.similarity - config.salience_lambda * info_penalty - config.size_lambda * size_penalty
    return _greedy_redundant_pairs(
        score,
        merge_r=merge_r,
        allowed_mask=allowed,
        source_preference=stats.informativeness + config.size_lambda * stats.size,
    )


def _evit_pairs(stats: PatchStats, merge_r: int, config: TokenReduceConfig) -> list[tuple[int, int]]:
    source_mask = ~stats.protect_mask
    anchor_mask = stats.protect_mask.clone()
    if not bool(anchor_mask.any().item()):
        protect_count = max(1, min(stats.salience.numel() - 1, int(round(stats.salience.numel() * 0.1))))
        anchor_mask[torch.topk(stats.salience, k=protect_count, largest=True).indices] = True
        source_mask = ~anchor_mask
    return _fuse_low_to_anchors(
        stats.similarity,
        merge_r=merge_r,
        source_score=stats.salience,
        anchor_score=stats.salience,
        source_mask=source_mask,
        anchor_mask=anchor_mask,
        salience_lambda=config.salience_lambda,
    )


def _ats_pairs(stats: PatchStats, merge_r: int, config: TokenReduceConfig) -> list[tuple[int, int]]:
    significance = _normalize01(0.7 * stats.significance + 0.3 * (1.0 - stats.size))
    source_cutoff = torch.quantile(significance, 0.65)
    source_mask = (significance <= source_cutoff) & ~stats.protect_mask
    anchor_mask = significance > source_cutoff
    if not bool(anchor_mask.any().item()):
        anchor_mask = ~source_mask
    return _fuse_low_to_anchors(
        stats.similarity,
        merge_r=merge_r,
        source_score=significance,
        anchor_score=significance,
        source_mask=source_mask,
        anchor_mask=anchor_mask,
        salience_lambda=config.salience_lambda,
    )


def _ppt_pairs(stats: PatchStats, merge_r: int, config: TokenReduceConfig) -> list[tuple[int, int]]:
    if merge_r <= 0:
        return []
    trash_budget = max(1, merge_r // 2)
    low_cutoff = torch.quantile(stats.salience, 0.35)
    trash_mask = (stats.salience <= low_cutoff) & ~stats.protect_mask
    anchor_mask = stats.protect_mask | (stats.salience >= torch.quantile(stats.salience, 0.65))
    trash_pairs = _fuse_low_to_anchors(
        stats.similarity,
        merge_r=trash_budget,
        source_score=stats.salience,
        anchor_score=stats.informativeness,
        source_mask=trash_mask,
        anchor_mask=anchor_mask,
        salience_lambda=config.salience_lambda,
    )

    used_sources = {src for src, _dst in trash_pairs}
    remaining = merge_r - len(trash_pairs)
    if remaining <= 0:
        return trash_pairs

    medium_mask = ~stats.protect_mask
    if used_sources:
        used = torch.as_tensor(sorted(used_sources), dtype=torch.long, device=stats.salience.device)
        medium_mask[used] = False
    medium_mask = medium_mask & (stats.salience > low_cutoff)
    medium_score = stats.similarity - config.salience_lambda * torch.abs(
        stats.informativeness[:, None] - stats.informativeness[None, :]
    )
    medium_score = medium_score - config.size_lambda * (stats.size[:, None] + stats.size[None, :]) * 0.5
    medium_pairs = _greedy_redundant_pairs(
        medium_score,
        merge_r=remaining,
        allowed_mask=medium_mask,
        source_preference=stats.informativeness,
    )
    return trash_pairs + medium_pairs


def _select_pairs(stats: PatchStats, merge_r: int, config: TokenReduceConfig) -> list[tuple[int, int]]:
    if config.method == "pitome":
        return _pitome_pairs(stats, merge_r, config)
    if config.method == "adamerge":
        return _adamerge_pairs(stats, merge_r, config)
    if config.method == "mctf":
        return _mctf_pairs(stats, merge_r, config)
    if config.method == "evit":
        return _evit_pairs(stats, merge_r, config)
    if config.method == "ats":
        return _ats_pairs(stats, merge_r, config)
    if config.method == "ppt":
        return _ppt_pairs(stats, merge_r, config)
    raise ValueError(f"unknown method: {config.method}")


def _run_token_reduce_block(
    block: torch.nn.Module,
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    *,
    config: TokenReduceConfig,
    special_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    normed = block.norm1(x)
    attn_out, metric, attn_probs = _proportional_attention_with_stats(block.attn, normed, sizes)
    x = x + block.drop_path1(block.ls1(attn_out))

    stats = _patch_stats(
        normed,
        metric,
        attn_probs,
        sizes,
        special_count=special_count,
        protect_fraction=config.protect_fraction,
    )
    adaptive = config.adaptive or config.method == "adamerge"
    requested_r = _adaptive_merge_r(config.merge_r, stats, enabled=adaptive)
    pairs = _select_pairs(stats, requested_r, config)
    x, sizes, sources, merged = _merge_local_pairs(x, sizes, sources, pairs, special_count=special_count)

    x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
    return x, sizes, sources, requested_r, merged


def get_token_reduce_intermediate_layers(
    vit: torch.nn.Module,
    x: torch.Tensor,
    layers: list[int],
    *,
    config: TokenReduceConfig,
    merge_counter: Counter[int] | None = None,
    requested_counter: Counter[int] | None = None,
    token_counter: Counter[int] | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    sequence, sizes, sources, patch_count, special_count = _prepare_tome_tokens(vit, x)
    layers_to_take = set(layers)
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    for block_index, block in enumerate(vit.blocks):
        sequence, sizes, sources, requested_r, merged = _run_token_reduce_block(
            block,
            sequence,
            sizes,
            sources,
            config=config,
            special_count=special_count,
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
        outputs.append(
            _restore_patch_grid(
                normalized,
                sources,
                patch_count=patch_count,
                special_count=special_count,
            )
        )

    if len(outputs) != len(layers):
        raise RuntimeError(f"only captured {len(outputs)} / {len(layers)} requested layers")
    return tuple(outputs)


class TokenReduceDepthAnything(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, *, config: TokenReduceConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.merge_counter: Counter[int] = Counter()
        self.requested_counter: Counter[int] = Counter()
        self.token_counter: Counter[int] = Counter()

    def image2tensor(self, raw_image, input_size: int = 518):
        return self.base_model.image2tensor(raw_image, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        layers = self.base_model.intermediate_layer_idx[self.base_model.encoder]
        features = get_token_reduce_intermediate_layers(
            self.base_model.pretrained,
            x,
            layers,
            config=self.config,
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


def evaluate(config: TokenReduceConfig) -> dict[str, Any]:
    require_ready(config.dataset_root, config.checkpoint, config.encoder)
    cv2 = load_cv2()
    device = resolve_device(config.device)
    dense_model = load_model(config.encoder, config.checkpoint, device)
    model = TokenReduceDepthAnything(dense_model, config=config).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    selected = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
    )
    if not selected:
        raise RuntimeError("no DA-2K annotations selected")

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
    result = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "variant": f"{config.method}_dense_safe_proxy",
            "method_note": METHOD_NOTES[config.method],
            "images_requested": len(selected),
            "missing_images": missing_images,
            "elapsed_seconds": time.monotonic() - started,
            "requested_merges_per_layer": config.merge_r,
            "adaptive_effective": bool(config.adaptive or config.method == "adamerge"),
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
            "notes": [
                "All methods are practical training-free proxy/adaptation variants inspired by the named papers.",
                "Merged/fused tokens keep source maps; intermediate features are expanded to the original patch count before the DA-V2 depth head.",
                "The evaluator supports batch size 1, matching the existing DA-2K scripts.",
            ],
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
        description="Evaluate training-free dense-depth-safe token reduction method proxies on DA-2K."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/da2k_vits_token_reduce_pitome_r57.json"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--method", choices=METHODS, default="pitome")
    parser.add_argument("--merge-r", type=int, default=57)
    parser.add_argument("--adaptive", action="store_true", help="Adapt merge_r per layer from current redundancy/salience.")
    parser.add_argument("--salience-lambda", type=float, default=1.0)
    parser.add_argument("--size-lambda", type=float, default=0.25)
    parser.add_argument("--protect-fraction", type=float, default=0.15)
    parser.add_argument("--scene-type", default="", choices=SCENE_CHOICES)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = TokenReduceConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        method=args.method,
        merge_r=args.merge_r,
        adaptive=args.adaptive,
        salience_lambda=args.salience_lambda,
        size_lambda=args.size_lambda,
        protect_fraction=args.protect_fraction,
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
