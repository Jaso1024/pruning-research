from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from eval_da2k import (
    MODEL_CONFIGS,
    SCENE_CHOICES,
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
from eval_patch_norm_da2k import get_pruned_intermediate_layers, select_kept_patch_indices


@dataclass(frozen=True)
class PatchNormFailureConfig:
    dataset_root: Path
    checkpoint: Path
    output_json: Path
    output_md: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    keep_percentages: tuple[float, ...] = (0.99, 0.98, 0.95, 0.90, 0.85, 0.80)
    norm: str = "l2"
    fill_mode: str = "zero"
    scene_type: str = ""
    max_images: int = 0
    log_every: int = 50
    top_examples: int = 25

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_json", Path(self.output_json))
        object.__setattr__(self, "output_md", Path(self.output_md))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if not self.keep_percentages:
            raise ValueError("at least one keep percentage is required")
        for keep in self.keep_percentages:
            if not 0.0 < keep <= 1.0:
                raise ValueError(f"keep percentage must be in (0, 1], got {keep}")
        if self.norm not in {"l1", "l2"}:
            raise ValueError("norm must be l1 or l2")
        if self.fill_mode not in {"zero", "input"}:
            raise ValueError("fill_mode must be zero or input")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")
        if self.top_examples < 0:
            raise ValueError("top_examples must be non-negative")


def _threshold_key(keep: float) -> str:
    return f"{keep:.4f}".rstrip("0").rstrip(".")


def _parse_keep_percentages(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _bool_rate(value: bool) -> float:
    return 1.0 if value else 0.0


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _normalize_dist(values: torch.Tensor) -> torch.Tensor:
    values = values.float().clamp_min(0)
    total = values.sum()
    if float(total.item()) <= 0:
        return torch.full_like(values, 1.0 / max(values.numel(), 1))
    return values / total


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= 0:
        return 0.0
    return float(torch.dot(a, b).div(denom).item())


def _jsd(a: torch.Tensor, b: torch.Tensor) -> float:
    eps = 1e-12
    p = _normalize_dist(a)
    q = _normalize_dist(b)
    m = 0.5 * (p + q)
    kl_pm = (p * ((p + eps).log() - (m + eps).log())).sum()
    kl_qm = (q * ((q + eps).log() - (m + eps).log())).sum()
    return float((0.5 * (kl_pm + kl_qm)).item())


def _attention_probs(attn: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    batch, token_count, channels = x.shape
    head_count = int(attn.num_heads)
    head_dim = channels // head_count
    qkv = attn.qkv(x).reshape(batch, token_count, 3, head_count, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q = qkv[0] * attn.scale
    k = qkv[1]
    return (q @ k.transpose(-2, -1)).softmax(dim=-1)


def _point_to_patch_index(
    point: list[int],
    *,
    raw_height: int,
    raw_width: int,
    input_height: int,
    input_width: int,
    patch_h: int,
    patch_w: int,
) -> int:
    row = max(0, min(int(point[0]), raw_height - 1))
    col = max(0, min(int(point[1]), raw_width - 1))
    input_row = min(input_height - 1, max(0, int(row * input_height / raw_height)))
    input_col = min(input_width - 1, max(0, int(col * input_width / raw_width)))
    patch_row = min(patch_h - 1, input_row // 14)
    patch_col = min(patch_w - 1, input_col // 14)
    return int(patch_row * patch_w + patch_col)


def _new_group(layer_count: int) -> dict[str, Any]:
    return {
        "count": 0,
        "endpoint_deleted_any_sum": 0.0,
        "endpoint_deleted_p1_sum": 0.0,
        "endpoint_deleted_p2_sum": 0.0,
        "top_any_deleted_readout_sum": 0.0,
        "top_nonself_deleted_readout_sum": 0.0,
        "deleted_mass_abs_readout_sum": 0.0,
        "deleted_mass_rel_readout_sum": 0.0,
        "deleted_mass_abs_last_sum": 0.0,
        "deleted_mass_rel_last_sum": 0.0,
        "deleted_mass_abs_by_layer_sum": [0.0] * layer_count,
        "deleted_mass_rel_by_layer_sum": [0.0] * layer_count,
    }


def _add_group_row(group: dict[str, Any], row: dict[str, Any]) -> None:
    group["count"] += 1
    for key in (
        "endpoint_deleted_any",
        "endpoint_deleted_p1",
        "endpoint_deleted_p2",
        "top_any_deleted_readout",
        "top_nonself_deleted_readout",
        "deleted_mass_abs_readout",
        "deleted_mass_rel_readout",
        "deleted_mass_abs_last",
        "deleted_mass_rel_last",
    ):
        group[f"{key}_sum"] += float(row[key])
    for index, value in enumerate(row["deleted_mass_abs_by_layer"]):
        group["deleted_mass_abs_by_layer_sum"][index] += float(value)
    for index, value in enumerate(row["deleted_mass_rel_by_layer"]):
        group["deleted_mass_rel_by_layer_sum"][index] += float(value)


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    count = group["count"]
    return {
        "count": count,
        "endpoint_deleted_any_rate": _safe_div(group["endpoint_deleted_any_sum"], count),
        "endpoint_deleted_p1_rate": _safe_div(group["endpoint_deleted_p1_sum"], count),
        "endpoint_deleted_p2_rate": _safe_div(group["endpoint_deleted_p2_sum"], count),
        "top_any_deleted_readout_rate": _safe_div(group["top_any_deleted_readout_sum"], count),
        "top_nonself_deleted_readout_rate": _safe_div(group["top_nonself_deleted_readout_sum"], count),
        "deleted_mass_abs_readout_mean": _safe_div(group["deleted_mass_abs_readout_sum"], count),
        "deleted_mass_rel_readout_mean": _safe_div(group["deleted_mass_rel_readout_sum"], count),
        "deleted_mass_abs_last_mean": _safe_div(group["deleted_mass_abs_last_sum"], count),
        "deleted_mass_rel_last_mean": _safe_div(group["deleted_mass_rel_last_sum"], count),
        "deleted_mass_abs_by_layer_mean": [_safe_div(value, count) for value in group["deleted_mass_abs_by_layer_sum"]],
        "deleted_mass_rel_by_layer_mean": [_safe_div(value, count) for value in group["deleted_mass_rel_by_layer_sum"]],
    }


def _new_threshold_state(layer_count: int) -> dict[str, Any]:
    return {
        "overall": empty_counts(),
        "by_scene": defaultdict(empty_counts),
        "transition_counts": defaultdict(int),
        "groups": defaultdict(lambda: _new_group(layer_count)),
        "global_deleted_abs_by_layer_sum": [0.0] * layer_count,
        "global_deleted_rel_by_layer_sum": [0.0] * layer_count,
        "global_deleted_image_count": 0,
        "examples": [],
    }


def _transition_name(dense_correct: bool, pruned_correct: bool) -> str:
    if dense_correct and pruned_correct:
        return "stable_correct"
    if dense_correct and not pruned_correct:
        return "regression"
    if not dense_correct and pruned_correct:
        return "fix"
    return "stable_wrong"


def _collect_dense_attention(
    vit: torch.nn.Module,
    x: torch.Tensor,
    query_patch_indices: set[int],
) -> dict[str, Any]:
    register_count = int(getattr(vit, "num_register_tokens", 0))
    patch_start = 1 + register_count
    patch_count = vit.patch_embed(x).shape[1]
    patch_slice = slice(patch_start, patch_start + patch_count)
    layer_count = len(vit.blocks)

    point_raw: dict[int, list[torch.Tensor]] = {patch_index: [] for patch_index in sorted(query_patch_indices)}
    point_norm: dict[int, list[torch.Tensor]] = {patch_index: [] for patch_index in sorted(query_patch_indices)}
    point_top_any: dict[int, list[int]] = {patch_index: [] for patch_index in sorted(query_patch_indices)}
    point_top_nonself: dict[int, list[int]] = {patch_index: [] for patch_index in sorted(query_patch_indices)}
    global_raw: list[torch.Tensor] = []
    global_norm: list[torch.Tensor] = []

    tokens = vit.prepare_tokens_with_masks(x)
    for layer_index, block in enumerate(vit.blocks):
        attn_input = block.norm1(tokens)
        probs = _attention_probs(block.attn, attn_input)[0].float()
        patch_to_patch = probs[:, patch_slice, patch_slice]
        global_dist = patch_to_patch.mean(dim=(0, 1)).detach().cpu()
        global_raw.append(global_dist)
        global_norm.append(_normalize_dist(global_dist))

        for patch_index in query_patch_indices:
            token_index = patch_start + patch_index
            dist = probs[:, token_index, patch_slice].mean(dim=0).detach().cpu()
            point_raw[patch_index].append(dist)
            point_norm[patch_index].append(_normalize_dist(dist))
            point_top_any[patch_index].append(int(torch.argmax(dist).item()))
            if dist.numel() > 1:
                nonself = dist.clone()
                nonself[patch_index] = -float("inf")
                point_top_nonself[patch_index].append(int(torch.argmax(nonself).item()))
            else:
                point_top_nonself[patch_index].append(int(torch.argmax(dist).item()))

        tokens = block(tokens)
        if x.is_cuda:
            torch.cuda.empty_cache()

    if len(global_raw) != layer_count:
        raise RuntimeError("failed to collect all attention layers")
    return {
        "global_raw": global_raw,
        "global_norm": global_norm,
        "point_raw": point_raw,
        "point_norm": point_norm,
        "point_top_any": point_top_any,
        "point_top_nonself": point_top_nonself,
    }


def _update_similarity(
    similarity_state: dict[str, Any],
    attention: dict[str, Any],
    query_patch_indices: set[int],
) -> None:
    global_norm = attention["global_norm"]
    layer_count = len(global_norm)
    similarity_state["image_count"] += 1
    for i in range(layer_count):
        for j in range(layer_count):
            similarity_state["global_cos_sum"][i][j] += _cosine(global_norm[i], global_norm[j])
            similarity_state["global_jsd_sum"][i][j] += _jsd(global_norm[i], global_norm[j])
    for i in range(layer_count - 1):
        similarity_state["global_adjacent_cos_sum"][i] += _cosine(global_norm[i], global_norm[i + 1])
        similarity_state["global_adjacent_jsd_sum"][i] += _jsd(global_norm[i], global_norm[i + 1])

    point_norm = attention["point_norm"]
    for patch_index in query_patch_indices:
        dists = point_norm[patch_index]
        similarity_state["point_count"] += 1
        for i in range(layer_count - 1):
            similarity_state["point_adjacent_cos_sum"][i] += _cosine(dists[i], dists[i + 1])
            similarity_state["point_adjacent_jsd_sum"][i] += _jsd(dists[i], dists[i + 1])


def _finalize_similarity(similarity_state: dict[str, Any]) -> dict[str, Any]:
    image_count = similarity_state["image_count"]
    point_count = similarity_state["point_count"]
    layer_count = len(similarity_state["global_adjacent_cos_sum"]) + 1
    return {
        "image_count": image_count,
        "point_query_count": point_count,
        "global_adjacent_cosine": [_safe_div(value, image_count) for value in similarity_state["global_adjacent_cos_sum"]],
        "global_adjacent_jsd": [_safe_div(value, image_count) for value in similarity_state["global_adjacent_jsd_sum"]],
        "point_adjacent_cosine": [_safe_div(value, point_count) for value in similarity_state["point_adjacent_cos_sum"]],
        "point_adjacent_jsd": [_safe_div(value, point_count) for value in similarity_state["point_adjacent_jsd_sum"]],
        "global_cosine_matrix": [
            [_safe_div(similarity_state["global_cos_sum"][i][j], image_count) for j in range(layer_count)]
            for i in range(layer_count)
        ],
        "global_jsd_matrix": [
            [_safe_div(similarity_state["global_jsd_sum"][i][j], image_count) for j in range(layer_count)]
            for i in range(layer_count)
        ],
    }


def _infer_dense_depth(model: torch.nn.Module, x: torch.Tensor, raw_height: int, raw_width: int) -> torch.Tensor:
    depth = model(x)
    depth = F.interpolate(depth[:, None], (raw_height, raw_width), mode="bilinear", align_corners=True)[0, 0]
    return depth.detach().float().cpu()


def _infer_patch_norm_depth(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    keep_percentage: float,
    norm: str,
    fill_mode: str,
    raw_height: int,
    raw_width: int,
) -> torch.Tensor:
    patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
    layers = model.intermediate_layer_idx[model.encoder]
    features = get_pruned_intermediate_layers(
        model.pretrained,
        x,
        layers,
        keep_percentage=keep_percentage,
        norm_kind=norm,
        keep_high=True,
        fill_mode=fill_mode,
    )
    depth = model.depth_head(features, patch_h, patch_w)
    depth = F.relu(depth).squeeze(1)
    depth = F.interpolate(depth[:, None], (raw_height, raw_width), mode="bilinear", align_corners=True)[0, 0]
    return depth.detach().float().cpu()


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Patch-Norm Input Pruning Failure Analysis")
    lines.append("")
    meta = result["metadata"]
    dense = result["dense_overall"]
    lines.append(f"- Images analyzed: {meta['images_evaluated']}/{meta['images_requested']}")
    lines.append(f"- Dense accuracy: {_format_float(dense['best_accuracy'])} ({dense['larger_correct']}/{dense['pairs']})")
    lines.append(f"- Readout layers: {meta['readout_layers']}")
    lines.append("")
    lines.append("## Accuracy and Transitions")
    lines.append("")
    lines.append("| keep | kept patches | accuracy | regressions | fixes | stable correct | stable wrong |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for key, summary in result["thresholds"].items():
        transitions = summary["transition_counts"]
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(summary["kept_patch_count_at_square_input"]),
                    _format_float(summary["overall"]["best_accuracy"]),
                    str(transitions.get("regression", 0)),
                    str(transitions.get("fix", 0)),
                    str(transitions.get("stable_correct", 0)),
                    str(transitions.get("stable_wrong", 0)),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Regression Attribution")
    lines.append("")
    lines.append(
        "| keep | group | n | endpoint deleted | top nonself readout deleted | readout deleted attention | last-layer deleted attention |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for key, summary in result["thresholds"].items():
        for group_name in ("regression", "stable_correct", "fix", "stable_wrong"):
            group = summary["groups"].get(group_name)
            if not group or group["count"] == 0:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        key,
                        group_name,
                        str(group["count"]),
                        _format_float(group["endpoint_deleted_any_rate"]),
                        _format_float(group["top_nonself_deleted_readout_rate"]),
                        _format_float(group["deleted_mass_rel_readout_mean"]),
                        _format_float(group["deleted_mass_rel_last_mean"]),
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("## Global Deleted Attention")
    lines.append("")
    lines.append("Mean dense patch-to-patch attention mass landing on tokens deleted by the threshold.")
    lines.append("")
    lines.append("| keep | layer 0 | layer 2 | layer 5 | layer 8 | layer 11 |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for key, summary in result["thresholds"].items():
        rel = summary["global_deleted_rel_by_layer_mean"]
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    _format_float(rel[0]),
                    _format_float(rel[2]),
                    _format_float(rel[5]),
                    _format_float(rel[8]),
                    _format_float(rel[11]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Adjacent-Layer Attention Similarity")
    lines.append("")
    lines.append("| layer pair | global cosine | global JSD | point-query cosine | point-query JSD |")
    lines.append("|---|---:|---:|---:|---:|")
    sim = result["attention_similarity"]
    for i, (gcos, gjsd, pcos, pjsd) in enumerate(
        zip(
            sim["global_adjacent_cosine"],
            sim["global_adjacent_jsd"],
            sim["point_adjacent_cosine"],
            sim["point_adjacent_jsd"],
            strict=True,
        )
    ):
        lines.append(
            f"| {i}-{i + 1} | {_format_float(gcos)} | {_format_float(gjsd)} | {_format_float(pcos)} | {_format_float(pjsd)} |"
        )
    lines.append("")
    lines.append("## Top Regression Examples")
    lines.append("")
    lines.append("Sorted by readout-layer deleted attention mass.")
    lines.append("")
    lines.append("| keep | image | pair | scene | dense margin | pruned margin | endpoint deleted | top nonself deleted | readout deleted attention |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|---:|")
    for example in result["top_regression_examples"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    example["keep"],
                    example["image_path"],
                    str(example["pair_index"]),
                    example["scene"],
                    _format_float(example["dense_margin"]),
                    _format_float(example["pruned_margin"]),
                    _format_float(float(example["endpoint_deleted_any"])),
                    _format_float(float(example["top_nonself_deleted_readout"])),
                    _format_float(example["deleted_mass_rel_readout"]),
                ]
            )
            + " |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def analyze(config: PatchNormFailureConfig) -> dict[str, Any]:
    require_ready(config.dataset_root, config.checkpoint, config.encoder)
    cv2 = load_cv2()
    device = resolve_device(config.device)
    model = load_model(config.encoder, config.checkpoint, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    layer_count = len(model.pretrained.blocks)
    readout_layers = list(model.intermediate_layer_idx[model.encoder])
    last_readout = readout_layers[-1]
    thresholds = {_threshold_key(keep): keep for keep in config.keep_percentages}
    threshold_state = {key: _new_threshold_state(layer_count) for key in thresholds}
    similarity_state = {
        "image_count": 0,
        "point_count": 0,
        "global_adjacent_cos_sum": [0.0] * (layer_count - 1),
        "global_adjacent_jsd_sum": [0.0] * (layer_count - 1),
        "point_adjacent_cos_sum": [0.0] * (layer_count - 1),
        "point_adjacent_jsd_sum": [0.0] * (layer_count - 1),
        "global_cos_sum": [[0.0] * layer_count for _ in range(layer_count)],
        "global_jsd_sum": [[0.0] * layer_count for _ in range(layer_count)],
    }

    selected = selected_annotations(config.dataset_root, scene_type=config.scene_type, max_images=config.max_images)
    if not selected:
        raise RuntimeError("no DA-2K annotations selected")

    dense_total = empty_counts()
    dense_by_scene = defaultdict(empty_counts)
    missing_images: list[str] = []
    started = time.monotonic()
    images_evaluated = 0

    with torch.inference_mode():
        for image_index, (relative_path, pairs) in enumerate(selected, start=1):
            image_path = config.dataset_root / relative_path
            raw_image = cv2.imread(str(image_path))
            if raw_image is None:
                missing_images.append(str(image_path))
                continue

            raw_height, raw_width = raw_image.shape[:2]
            x, _ = model.image2tensor(raw_image, config.input_size)
            x = x.to(device)
            input_height, input_width = int(x.shape[-2]), int(x.shape[-1])
            patch_h, patch_w = input_height // 14, input_width // 14
            scene = scene_from_path(relative_path)
            images_evaluated += 1

            pair_records: list[dict[str, Any]] = []
            query_patch_indices: set[int] = set()
            for pair_index, pair in enumerate(pairs):
                if pair.get("closer_point") != "point1":
                    raise ValueError(f"unsupported closer_point in {relative_path}: {pair}")
                p1_patch = _point_to_patch_index(
                    pair["point1"],
                    raw_height=raw_height,
                    raw_width=raw_width,
                    input_height=input_height,
                    input_width=input_width,
                    patch_h=patch_h,
                    patch_w=patch_w,
                )
                p2_patch = _point_to_patch_index(
                    pair["point2"],
                    raw_height=raw_height,
                    raw_width=raw_width,
                    input_height=input_height,
                    input_width=input_width,
                    patch_h=patch_h,
                    patch_w=patch_w,
                )
                query_patch_indices.add(p1_patch)
                query_patch_indices.add(p2_patch)
                pair_records.append(
                    {
                        "pair_index": pair_index,
                        "pair": pair,
                        "p1_patch": p1_patch,
                        "p2_patch": p2_patch,
                    }
                )

            patch_embeddings = model.pretrained.patch_embed(x)
            patch_count = int(patch_embeddings.shape[1])
            keep_masks: dict[str, torch.Tensor] = {}
            for key, keep in thresholds.items():
                kept_indices = select_kept_patch_indices(
                    patch_embeddings,
                    keep_percentage=keep,
                    norm=config.norm,  # type: ignore[arg-type]
                    keep_high=True,
                ).cpu()
                mask = torch.zeros(patch_count, dtype=torch.bool)
                mask[kept_indices] = True
                keep_masks[key] = mask

            dense_depth = _infer_dense_depth(model, x, raw_height, raw_width)
            pruned_depths = {
                key: _infer_patch_norm_depth(
                    model,
                    x,
                    keep_percentage=keep,
                    norm=config.norm,
                    fill_mode=config.fill_mode,
                    raw_height=raw_height,
                    raw_width=raw_width,
                )
                for key, keep in thresholds.items()
            }
            attention = _collect_dense_attention(model.pretrained, x, query_patch_indices)
            _update_similarity(similarity_state, attention, query_patch_indices)

            for key, mask in keep_masks.items():
                state = threshold_state[key]
                for layer_index in range(layer_count):
                    deleted = ~mask
                    global_raw = attention["global_raw"][layer_index]
                    global_norm = attention["global_norm"][layer_index]
                    state["global_deleted_abs_by_layer_sum"][layer_index] += float(global_raw[deleted].sum().item())
                    state["global_deleted_rel_by_layer_sum"][layer_index] += float(global_norm[deleted].sum().item())
                state["global_deleted_image_count"] += 1

            for record in pair_records:
                pair = record["pair"]
                pair_index = record["pair_index"]
                dense_d1 = point_value(dense_depth, pair["point1"])
                dense_d2 = point_value(dense_depth, pair["point2"])
                dense_margin = dense_d1 - dense_d2
                dense_correct = dense_margin > 0
                add_pair(dense_total, dense_d1, dense_d2)
                add_pair(dense_by_scene[scene], dense_d1, dense_d2)

                p1_patch = record["p1_patch"]
                p2_patch = record["p2_patch"]
                for key, pruned_depth in pruned_depths.items():
                    pruned_d1 = point_value(pruned_depth, pair["point1"])
                    pruned_d2 = point_value(pruned_depth, pair["point2"])
                    pruned_margin = pruned_d1 - pruned_d2
                    pruned_correct = pruned_margin > 0
                    state = threshold_state[key]
                    add_pair(state["overall"], pruned_d1, pruned_d2)
                    add_pair(state["by_scene"][scene], pruned_d1, pruned_d2)

                    mask = keep_masks[key]
                    endpoint_deleted_p1 = not bool(mask[p1_patch].item())
                    endpoint_deleted_p2 = not bool(mask[p2_patch].item())
                    deleted_mass_abs_by_layer: list[float] = []
                    deleted_mass_rel_by_layer: list[float] = []
                    top_any_deleted_readout = False
                    top_nonself_deleted_readout = False
                    deleted = ~mask
                    for layer_index in range(layer_count):
                        p1_raw = attention["point_raw"][p1_patch][layer_index]
                        p2_raw = attention["point_raw"][p2_patch][layer_index]
                        p1_norm = attention["point_norm"][p1_patch][layer_index]
                        p2_norm = attention["point_norm"][p2_patch][layer_index]
                        abs_mass = 0.5 * float((p1_raw[deleted].sum() + p2_raw[deleted].sum()).item())
                        rel_mass = 0.5 * float((p1_norm[deleted].sum() + p2_norm[deleted].sum()).item())
                        deleted_mass_abs_by_layer.append(abs_mass)
                        deleted_mass_rel_by_layer.append(rel_mass)
                        if layer_index in readout_layers:
                            p1_top_any = attention["point_top_any"][p1_patch][layer_index]
                            p2_top_any = attention["point_top_any"][p2_patch][layer_index]
                            p1_top_nonself = attention["point_top_nonself"][p1_patch][layer_index]
                            p2_top_nonself = attention["point_top_nonself"][p2_patch][layer_index]
                            top_any_deleted_readout = top_any_deleted_readout or not bool(mask[p1_top_any].item())
                            top_any_deleted_readout = top_any_deleted_readout or not bool(mask[p2_top_any].item())
                            top_nonself_deleted_readout = top_nonself_deleted_readout or not bool(mask[p1_top_nonself].item())
                            top_nonself_deleted_readout = top_nonself_deleted_readout or not bool(mask[p2_top_nonself].item())

                    readout_abs = sum(deleted_mass_abs_by_layer[layer] for layer in readout_layers) / len(readout_layers)
                    readout_rel = sum(deleted_mass_rel_by_layer[layer] for layer in readout_layers) / len(readout_layers)
                    transition = _transition_name(dense_correct, pruned_correct)
                    state["transition_counts"][transition] += 1
                    row = {
                        "endpoint_deleted_any": _bool_rate(endpoint_deleted_p1 or endpoint_deleted_p2),
                        "endpoint_deleted_p1": _bool_rate(endpoint_deleted_p1),
                        "endpoint_deleted_p2": _bool_rate(endpoint_deleted_p2),
                        "top_any_deleted_readout": _bool_rate(top_any_deleted_readout),
                        "top_nonself_deleted_readout": _bool_rate(top_nonself_deleted_readout),
                        "deleted_mass_abs_readout": readout_abs,
                        "deleted_mass_rel_readout": readout_rel,
                        "deleted_mass_abs_last": deleted_mass_abs_by_layer[last_readout],
                        "deleted_mass_rel_last": deleted_mass_rel_by_layer[last_readout],
                        "deleted_mass_abs_by_layer": deleted_mass_abs_by_layer,
                        "deleted_mass_rel_by_layer": deleted_mass_rel_by_layer,
                    }
                    _add_group_row(state["groups"][transition], row)
                    if transition == "regression":
                        state["examples"].append(
                            {
                                "keep": key,
                                "image_path": relative_path,
                                "pair_index": pair_index,
                                "scene": scene,
                                "dense_margin": dense_margin,
                                "pruned_margin": pruned_margin,
                                "endpoint_deleted_any": bool(endpoint_deleted_p1 or endpoint_deleted_p2),
                                "endpoint_deleted_p1": endpoint_deleted_p1,
                                "endpoint_deleted_p2": endpoint_deleted_p2,
                                "top_any_deleted_readout": top_any_deleted_readout,
                                "top_nonself_deleted_readout": top_nonself_deleted_readout,
                                "deleted_mass_rel_readout": readout_rel,
                                "deleted_mass_rel_last": deleted_mass_rel_by_layer[last_readout],
                                "p1_patch": p1_patch,
                                "p2_patch": p2_patch,
                            }
                        )

            if config.log_every > 0 and (image_index % config.log_every == 0 or image_index == len(selected)):
                elapsed = time.monotonic() - started
                print(f"analyzed {image_index}/{len(selected)} images in {elapsed:.1f}s", flush=True)

    dense_overall = finalize_counts(dense_total)
    all_examples: list[dict[str, Any]] = []
    threshold_summaries: dict[str, Any] = {}
    square_patch_count = (config.input_size // 14) * (config.input_size // 14)
    for key, keep in thresholds.items():
        state = threshold_state[key]
        image_count = state["global_deleted_image_count"]
        groups = {name: _finalize_group(group) for name, group in sorted(state["groups"].items())}
        examples = sorted(
            state["examples"],
            key=lambda item: (item["deleted_mass_rel_readout"], abs(item["pruned_margin"])),
            reverse=True,
        )
        all_examples.extend(examples)
        threshold_summaries[key] = {
            "keep_percentage": keep,
            "kept_patch_count_at_square_input": math.ceil(square_patch_count * keep),
            "overall": finalize_counts(state["overall"]),
            "by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(state["by_scene"].items())},
            "transition_counts": dict(sorted(state["transition_counts"].items())),
            "groups": groups,
            "global_deleted_abs_by_layer_mean": [_safe_div(value, image_count) for value in state["global_deleted_abs_by_layer_sum"]],
            "global_deleted_rel_by_layer_mean": [_safe_div(value, image_count) for value in state["global_deleted_rel_by_layer_sum"]],
            "top_regression_examples": examples[: config.top_examples],
        }

    all_examples = sorted(
        all_examples,
        key=lambda item: (item["deleted_mass_rel_readout"], abs(item["pruned_margin"])),
        reverse=True,
    )[: config.top_examples]
    result = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_requested": len(selected),
            "images_evaluated": images_evaluated,
            "missing_images": missing_images,
            "elapsed_seconds": time.monotonic() - started,
            "layer_count": layer_count,
            "readout_layers": readout_layers,
            "patch_count_at_square_input": square_patch_count,
            "rule": "DA-2K labels point1 as closer; Depth Anything V2 vits uses larger predicted values for closer points.",
        },
        "dense_overall": dense_overall,
        "dense_by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(dense_by_scene.items())},
        "thresholds": threshold_summaries,
        "attention_similarity": _finalize_similarity(similarity_state),
        "top_regression_examples": all_examples,
    }

    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    _write_markdown(result, config.output_md)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze why patch-embedding-norm input token pruning fails on Depth Anything V2 / DA-2K."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/patch_norm_failure_analysis.json"))
    parser.add_argument("--output-md", type=Path, default=Path("eval_outputs/patch_norm_failure_analysis.md"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keep-percentages", type=_parse_keep_percentages, default=(0.99, 0.98, 0.95, 0.90, 0.85, 0.80))
    parser.add_argument("--norm", choices=["l1", "l2"], default="l2")
    parser.add_argument("--fill-mode", choices=["zero", "input"], default="zero")
    parser.add_argument("--scene-type", choices=SCENE_CHOICES, default="")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--top-examples", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = PatchNormFailureConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        output_md=args.output_md,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        keep_percentages=args.keep_percentages,
        norm=args.norm,
        fill_mode=args.fill_mode,
        scene_type=args.scene_type,
        max_images=args.max_images,
        log_every=args.log_every,
        top_examples=args.top_examples,
    )
    result = analyze(config)
    print(json.dumps({
        "dense": result["dense_overall"],
        "thresholds": {
            key: {
                "overall": summary["overall"],
                "transition_counts": summary["transition_counts"],
            }
            for key, summary in result["thresholds"].items()
        },
        "output_json": str(config.output_json),
        "output_md": str(config.output_md),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
