from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

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
from eval_tome_da2k import (
    _merge_wavg,
    _prepare_tome_tokens,
    _proportional_attention,
    _restore_patch_grid,
    selected_annotations,
)


SCENE_CHOICES = [
    "",
    "indoor",
    "outdoor",
    "non_real",
    "transparent_reflective",
    "adverse_style",
    "aerial",
    "underwater",
    "object",
]


@dataclass(frozen=True)
class AttentionCalibMergeConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    output_json: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    calib_images: int = 64
    merge_r: int = 57
    score_mode: str = "mutual_minus_external"
    external_lambda: float = 1.0
    scene_type: str = ""
    max_images: int = 0
    log_every: int = 25
    top_pairs_per_layer: int = 50_000
    candidate_multiplier: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "output_json", Path(self.output_json))
        if self.encoder != "vits":
            raise ValueError("minimal attention calibration merge evaluator only supports --encoder vits")
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.calib_images <= 0:
            raise ValueError("calib_images must be positive")
        if self.merge_r < 0:
            raise ValueError("merge_r must be non-negative")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.top_pairs_per_layer <= 0:
            raise ValueError("top_pairs_per_layer must be positive")
        if self.candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")
        if self.score_mode not in {"high_mutual", "mutual_minus_external"}:
            raise ValueError(f"unknown score mode: {self.score_mode}")


@dataclass
class LayerPlan:
    score: torch.Tensor
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_score: np.ndarray


@dataclass
class ShapePlan:
    patch_h: int
    patch_w: int
    layers: list[LayerPlan]
    calibration_images: int
    source_shape: tuple[int, int] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.patch_h, self.patch_w)


def _shape_key(shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}"


def _manual_attention_with_probs(attn: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, token_count, channels = x.shape
    head_count = int(attn.num_heads)
    head_dim = channels // head_count

    qkv = attn.qkv(x).reshape(batch, token_count, 3, head_count, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q = qkv[0] * attn.scale
    k = qkv[1]
    v = qkv[2]

    attn_probs = (q @ k.transpose(-2, -1)).softmax(dim=-1)
    out = (attn.attn_drop(attn_probs) @ v).transpose(1, 2).reshape(batch, token_count, channels)
    out = attn.proj(out)
    out = attn.proj_drop(out)
    return out, attn_probs


def _score_from_attention(avg_attn: torch.Tensor, score_mode: str, external_lambda: float) -> torch.Tensor:
    mutual = 0.5 * (avg_attn + avg_attn.transpose(0, 1))
    if score_mode == "high_mutual":
        return mutual

    patch_count = avg_attn.shape[0]
    if patch_count <= 2:
        return mutual
    row_sum = avg_attn.sum(dim=1)
    self_attn = avg_attn.diagonal()
    external_from_i = row_sum[:, None] - avg_attn - self_attn[:, None]
    external_from_j = row_sum[None, :] - avg_attn.transpose(0, 1) - self_attn[None, :]
    external_penalty = 0.5 * (external_from_i + external_from_j) / float(patch_count - 2)
    return mutual - float(external_lambda) * external_penalty


def _build_layer_plan(
    score: torch.Tensor,
    *,
    top_pairs_per_layer: int,
) -> LayerPlan:
    patch_count = score.shape[0]
    rows, cols = torch.triu_indices(patch_count, patch_count, offset=1, device=score.device)
    values = score[rows, cols]
    top_k = min(int(top_pairs_per_layer), int(values.numel()))
    top_values, top_indices = torch.topk(values, k=top_k, largest=True, sorted=True)
    pair_i = rows[top_indices].detach().cpu().numpy().astype(np.int32, copy=False)
    pair_j = cols[top_indices].detach().cpu().numpy().astype(np.int32, copy=False)
    pair_score = top_values.detach().float().cpu().numpy().astype(np.float32, copy=False)
    return LayerPlan(
        score=score.detach().to(device="cpu", dtype=torch.float16).contiguous(),
        pair_i=pair_i,
        pair_j=pair_j,
        pair_score=pair_score,
    )


def _resize_pair_score(
    score: torch.Tensor,
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> torch.Tensor:
    if source_shape == target_shape:
        return score
    src_h, src_w = source_shape
    dst_h, dst_w = target_shape
    src_count = src_h * src_w
    dst_count = dst_h * dst_w
    key_resized = F.interpolate(
        score.float().reshape(src_count, 1, src_h, src_w),
        size=(dst_h, dst_w),
        mode="bilinear",
        align_corners=False,
    ).reshape(src_h, src_w, dst_h, dst_w)
    query_resized = F.interpolate(
        key_resized.permute(2, 3, 0, 1).reshape(dst_count, 1, src_h, src_w),
        size=(dst_h, dst_w),
        mode="bilinear",
        align_corners=False,
    )
    return query_resized.reshape(dst_h, dst_w, dst_h, dst_w).permute(2, 3, 0, 1).reshape(dst_count, dst_count)


class CalibratedScoreBank:
    def __init__(
        self,
        plans: dict[tuple[int, int], ShapePlan],
        *,
        top_pairs_per_layer: int,
        build_device: torch.device,
    ) -> None:
        self.plans = plans
        self.top_pairs_per_layer = top_pairs_per_layer
        self.build_device = build_device
        self.fallback_shapes: dict[tuple[int, int], tuple[int, int]] = {}

    def get(self, patch_h: int, patch_w: int) -> ShapePlan:
        shape = (int(patch_h), int(patch_w))
        if shape in self.plans:
            return self.plans[shape]
        if not self.plans:
            raise RuntimeError("no calibrated score plans are available")

        source_shape = min(
            self.plans,
            key=lambda item: abs(item[0] - shape[0]) + abs(item[1] - shape[1]),
        )
        source_plan = self.plans[source_shape]
        print(f"building resized calibrated score fallback {source_shape} -> {shape}", flush=True)
        layers: list[LayerPlan] = []
        for layer in source_plan.layers:
            score = _resize_pair_score(
                layer.score,
                source_shape=source_shape,
                target_shape=shape,
            ).to(self.build_device)
            layers.append(_build_layer_plan(score, top_pairs_per_layer=self.top_pairs_per_layer))
            del score
            if self.build_device.type == "cuda":
                torch.cuda.empty_cache()
        plan = ShapePlan(
            patch_h=shape[0],
            patch_w=shape[1],
            layers=layers,
            calibration_images=0,
            source_shape=source_shape,
        )
        self.plans[shape] = plan
        self.fallback_shapes[shape] = source_shape
        return plan

    def summary(self) -> dict[str, Any]:
        exact = {
            _shape_key(shape): plan.calibration_images
            for shape, plan in sorted(self.plans.items())
            if plan.source_shape is None
        }
        fallbacks = {
            _shape_key(shape): _shape_key(source)
            for shape, source in sorted(self.fallback_shapes.items())
        }
        return {
            "exact_shape_counts": exact,
            "fallback_shapes": fallbacks,
            "top_pairs_per_layer": self.top_pairs_per_layer,
        }


def _read_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return image


def _image_patch_shape(model: torch.nn.Module, image_path: Path, input_size: int) -> tuple[int, int]:
    image = _read_image(image_path)
    tensor, _ = model.image2tensor(image, input_size)
    return (int(tensor.shape[-2] // 14), int(tensor.shape[-1] // 14))


@torch.no_grad()
def _collect_shape_attention_sums(
    model: torch.nn.Module,
    image_paths: list[Path],
    *,
    input_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    vit = model.pretrained
    sums: list[torch.Tensor] | None = None
    patch_count = 0

    for image_path in image_paths:
        image = _read_image(image_path)
        tensor, _ = model.image2tensor(image, input_size)
        tensor = tensor.to(device)
        sequence, _sizes, _sources, current_patch_count, special_count = _prepare_tome_tokens(vit, tensor)
        if sums is None:
            patch_count = current_patch_count
            sums = [
                torch.zeros((patch_count, patch_count), dtype=torch.float32, device=device)
                for _ in vit.blocks
            ]
        elif current_patch_count != patch_count:
            raise RuntimeError(f"mixed patch counts inside one calibration shape for {image_path}")

        for block_index, block in enumerate(vit.blocks):
            attn_out, attn_probs = _manual_attention_with_probs(block.attn, block.norm1(sequence))
            patch_probs = attn_probs[:, :, special_count:, special_count:].mean(dim=(0, 1))
            sums[block_index].add_(patch_probs.float())
            sequence = sequence + block.drop_path1(block.ls1(attn_out))
            sequence = sequence + block.drop_path2(block.ls2(block.mlp(block.norm2(sequence))))

    if sums is None:
        raise RuntimeError("no images were supplied for calibration")
    return sums


def collect_calibrated_scores(
    model: torch.nn.Module,
    calibration_paths: list[Path],
    *,
    input_size: int,
    device: torch.device,
    score_mode: str,
    external_lambda: float,
    top_pairs_per_layer: int,
    log_every: int,
) -> CalibratedScoreBank:
    shape_by_path = {}
    shape_counts: Counter[tuple[int, int]] = Counter()
    for index, image_path in enumerate(calibration_paths, start=1):
        shape = _image_patch_shape(model, image_path, input_size)
        shape_by_path[image_path] = shape
        shape_counts[shape] += 1
        if log_every > 0 and (index % log_every == 0 or index == len(calibration_paths)):
            print(f"scanned calibration image shapes {index}/{len(calibration_paths)}", flush=True)

    grouped: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for image_path in calibration_paths:
        grouped[shape_by_path[image_path]].append(image_path)

    plans: dict[tuple[int, int], ShapePlan] = {}
    for shape_index, (shape, paths) in enumerate(sorted(grouped.items()), start=1):
        print(
            f"calibrating shape {_shape_key(shape)} ({len(paths)} images, {shape_index}/{len(grouped)})",
            flush=True,
        )
        sums = _collect_shape_attention_sums(model, paths, input_size=input_size, device=device)
        layers: list[LayerPlan] = []
        for block_index, attn_sum in enumerate(sums):
            avg_attn = attn_sum / float(len(paths))
            score = _score_from_attention(avg_attn, score_mode, external_lambda)
            layers.append(_build_layer_plan(score, top_pairs_per_layer=top_pairs_per_layer))
            print(f"  built layer {block_index:02d} top-pair plan", flush=True)
            del avg_attn, score
        plans[shape] = ShapePlan(
            patch_h=shape[0],
            patch_w=shape[1],
            layers=layers,
            calibration_images=len(paths),
        )
        del sums
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return CalibratedScoreBank(plans, top_pairs_per_layer=top_pairs_per_layer, build_device=device)


def _group_pair_score(layer: LayerPlan, members_a: list[int], members_b: list[int], base_score: float) -> float:
    if len(members_a) == 1 and len(members_b) == 1:
        return base_score
    index_a = torch.as_tensor(members_a, dtype=torch.long)
    index_b = torch.as_tensor(members_b, dtype=torch.long)
    values = layer.score.index_select(0, index_a).index_select(1, index_b)
    return float(values.float().mean().item())


def _pick_calibrated_pairs(
    layer: LayerPlan,
    members: list[list[int]],
    owner: list[int],
    *,
    merge_r: int,
    candidate_multiplier: int,
) -> list[tuple[int, int]]:
    if merge_r <= 0 or len(members) < 2:
        return []

    target_candidates = max(int(merge_r) * int(candidate_multiplier), int(merge_r))
    candidates: dict[tuple[int, int], float] = {}

    def select_from_candidates() -> list[tuple[int, int]]:
        selected_pairs: list[tuple[int, int]] = []
        used_groups: set[int] = set()
        for (candidate_i, candidate_j), _score in sorted(
            candidates.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            if candidate_i in used_groups or candidate_j in used_groups:
                continue
            dst = min(candidate_i, candidate_j)
            src = max(candidate_i, candidate_j)
            selected_pairs.append((src, dst))
            used_groups.add(candidate_i)
            used_groups.add(candidate_j)
            if len(selected_pairs) >= merge_r:
                break
        return selected_pairs

    next_check = target_candidates
    for original_i, original_j, base_score in zip(layer.pair_i, layer.pair_j, layer.pair_score):
        group_i = owner[int(original_i)]
        group_j = owner[int(original_j)]
        if group_i == group_j:
            continue
        if group_j < group_i:
            group_i, group_j = group_j, group_i
        key = (group_i, group_j)
        if key in candidates:
            continue
        candidates[key] = _group_pair_score(
            layer,
            members[group_i],
            members[group_j],
            float(base_score),
        )
        if len(candidates) >= next_check:
            selected = select_from_candidates()
            if len(selected) >= merge_r:
                return selected
            next_check *= 2

    return select_from_candidates()


def _merge_calibrated_pairs(
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    members: list[list[int]],
    owner: list[int],
    *,
    layer: LayerPlan,
    merge_r: int,
    special_count: int,
    candidate_multiplier: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], list[int], int]:
    selected = _pick_calibrated_pairs(
        layer,
        members,
        owner,
        merge_r=merge_r,
        candidate_multiplier=candidate_multiplier,
    )
    if not selected:
        return x, sizes, sources, members, owner, 0

    source_locals = {src for src, _dst in selected}
    keep_patch_locals = [index for index in range(len(members)) if index not in source_locals]
    device = x.device
    keep_idx = torch.as_tensor(
        [*range(special_count), *[special_count + index for index in keep_patch_locals]],
        dtype=torch.long,
        device=device,
    )
    new_x = x.index_select(1, keep_idx).clone()
    new_sizes = sizes.index_select(1, keep_idx).clone()
    new_sources = sources.index_select(1, keep_idx).clone()
    new_members = [list(members[index]) for index in keep_patch_locals]
    old_to_new = {old: new for new, old in enumerate(keep_patch_locals)}

    for src_local, dst_local in selected:
        dst_new_local = old_to_new[dst_local]
        src_idx = special_count + src_local
        dst_new_idx = special_count + dst_new_local
        src_size = sizes[:, src_idx]
        dst_size = new_sizes[:, dst_new_idx]
        new_x[:, dst_new_idx] = _merge_wavg(new_x[:, dst_new_idx], x[:, src_idx], dst_size, src_size)
        new_sizes[:, dst_new_idx] = dst_size + src_size
        new_sources[:, dst_new_idx] = new_sources[:, dst_new_idx] + sources[:, src_idx]
        new_members[dst_new_local].extend(members[src_local])

    new_owner = [-1] * len(owner)
    for group_index, group_members in enumerate(new_members):
        for original_index in group_members:
            new_owner[original_index] = group_index
    if any(value < 0 for value in new_owner):
        raise RuntimeError("lost source ownership during calibrated merge")

    return new_x, new_sizes, new_sources, new_members, new_owner, len(selected)


def _run_calibrated_block(
    block: torch.nn.Module,
    x: torch.Tensor,
    sizes: torch.Tensor,
    sources: torch.Tensor,
    members: list[list[int]],
    owner: list[int],
    *,
    layer: LayerPlan,
    merge_r: int,
    special_count: int,
    candidate_multiplier: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], list[int], int]:
    attn_out, _metric = _proportional_attention(block.attn, block.norm1(x), sizes)
    x = x + block.drop_path1(block.ls1(attn_out))
    x, sizes, sources, members, owner, merged = _merge_calibrated_pairs(
        x,
        sizes,
        sources,
        members,
        owner,
        layer=layer,
        merge_r=merge_r,
        special_count=special_count,
        candidate_multiplier=candidate_multiplier,
    )
    x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
    return x, sizes, sources, members, owner, merged


def get_attention_calib_intermediate_layers(
    vit: torch.nn.Module,
    x: torch.Tensor,
    layers: list[int],
    *,
    score_bank: CalibratedScoreBank,
    merge_r: int,
    candidate_multiplier: int,
    merge_counter: Counter[int] | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    sequence, sizes, sources, patch_count, special_count = _prepare_tome_tokens(vit, x)
    patch_h = int(x.shape[-2] // vit.patch_size)
    patch_w = int(x.shape[-1] // vit.patch_size)
    if patch_h * patch_w != patch_count:
        raise RuntimeError(f"patch shape {patch_h}x{patch_w} does not match patch count {patch_count}")
    shape_plan = score_bank.get(patch_h, patch_w)
    if len(shape_plan.layers) != len(vit.blocks):
        raise RuntimeError("calibrated layer plan count does not match ViT block count")

    members = [[index] for index in range(patch_count)]
    owner = list(range(patch_count))
    layers_to_take = set(layers)
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    for block_index, block in enumerate(vit.blocks):
        sequence, sizes, sources, members, owner, merged = _run_calibrated_block(
            block,
            sequence,
            sizes,
            sources,
            members,
            owner,
            layer=shape_plan.layers[block_index],
            merge_r=merge_r,
            special_count=special_count,
            candidate_multiplier=candidate_multiplier,
        )
        if merge_counter is not None:
            merge_counter[block_index] += merged
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


class AttentionCalibMergeDepthAnything(torch.nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        score_bank: CalibratedScoreBank,
        merge_r: int,
        candidate_multiplier: int,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.score_bank = score_bank
        self.merge_r = merge_r
        self.candidate_multiplier = candidate_multiplier
        self.merge_counter: Counter[int] = Counter()

    def image2tensor(self, raw_image, input_size: int = 518):
        return self.base_model.image2tensor(raw_image, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        layers = self.base_model.intermediate_layer_idx[self.base_model.encoder]
        features = get_attention_calib_intermediate_layers(
            self.base_model.pretrained,
            x,
            layers,
            score_bank=self.score_bank,
            merge_r=self.merge_r,
            candidate_multiplier=self.candidate_multiplier,
            merge_counter=self.merge_counter,
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


def _calibration_paths(config: AttentionCalibMergeConfig) -> list[Path]:
    selected = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.calib_images,
    )
    paths = [config.dataset_root / relative_path for relative_path, _pairs in selected]
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise RuntimeError("no calibration images found")
    return existing


def evaluate(config: AttentionCalibMergeConfig) -> dict[str, Any]:
    started = time.monotonic()
    device = resolve_device(config.device)
    dense_model = load_model(config.encoder, config.checkpoint, device)
    for param in dense_model.parameters():
        param.requires_grad_(False)

    calibration_paths = _calibration_paths(config)
    score_bank = collect_calibrated_scores(
        dense_model,
        calibration_paths,
        input_size=config.input_size,
        device=device,
        score_mode=config.score_mode,
        external_lambda=config.external_lambda,
        top_pairs_per_layer=config.top_pairs_per_layer,
        log_every=config.log_every,
    )

    model = AttentionCalibMergeDepthAnything(
        dense_model,
        score_bank=score_bank,
        merge_r=config.merge_r,
        candidate_multiplier=config.candidate_multiplier,
    ).to(device).eval()

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

    eval_started = time.monotonic()
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
            print(f"evaluated {index}/{len(selected)} images", flush=True)

    result = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "mode": "gradual",
            "calibration_images_found": len(calibration_paths),
            "score_bank": score_bank.summary(),
            "images_requested": len(selected),
            "missing_images": missing_images,
            "elapsed_seconds": time.monotonic() - started,
            "eval_elapsed_seconds": time.monotonic() - eval_started,
            "requested_merges_per_layer": config.merge_r,
            "actual_merges_by_block": {
                str(block_index): int(model.merge_counter.get(block_index, 0))
                for block_index in range(len(dense_model.pretrained.blocks))
            },
            "rule": "DA-2K labels point1 as closer; Depth Anything V2 vits uses larger predicted values for closer points.",
            "notes": [
                "Minimal evaluator: vits, batch size 1, gradual per-block calibrated merge only.",
                "Calibration attention is patch-token to patch-token post-softmax attention, averaged over heads and images sharing the same patch-grid shape.",
                "Pair selection scans each layer's top original-position calibrated pairs, rejects pairs already in the same source group, and uses the exact average score over current source groups for candidate ranking.",
            ],
        },
        "overall": finalize_counts(total),
        "by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(by_scene.items())},
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return result


def _default_output_json(args: argparse.Namespace) -> Path:
    max_part = f"max{args.max_images}" if args.max_images > 0 else "full"
    lambda_part = f"{args.external_lambda:g}".replace(".", "p")
    filename = (
        f"gradual_r{args.merge_r}_calib{args.calib_images}_"
        f"{args.score_mode}_lambda{lambda_part}_{max_part}.json"
    )
    return args.output_dir / filename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate calibrated attention-guided gradual token merging for Depth Anything V2 on DA-2K."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/da2k_vits_attention_calib_merge"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--encoder", choices=["vits"], default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calib-images", type=int, default=64)
    parser.add_argument("--merge-r", type=int, default=57)
    parser.add_argument("--score-mode", choices=["high_mutual", "mutual_minus_external"], default="mutual_minus_external")
    parser.add_argument("--external-lambda", type=float, default=1.0)
    parser.add_argument("--scene-type", default="", choices=SCENE_CHOICES)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--top-pairs-per-layer", type=int, default=50_000)
    parser.add_argument("--candidate-multiplier", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_json = args.output_json or _default_output_json(args)
    config = AttentionCalibMergeConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        output_json=output_json,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        calib_images=args.calib_images,
        merge_r=args.merge_r,
        score_mode=args.score_mode,
        external_lambda=args.external_lambda,
        scene_type=args.scene_type,
        max_images=args.max_images,
        log_every=args.log_every,
        top_pairs_per_layer=args.top_pairs_per_layer,
        candidate_multiplier=args.candidate_multiplier,
    )
    summary = evaluate(config)
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print(f"wrote {config.output_json}")


if __name__ == "__main__":
    main()
