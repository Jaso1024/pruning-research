from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from eval_da2k import (
    MODEL_CONFIGS,
    SCENE_CHOICES,
    load_cv2,
    load_model,
    require_ready,
    resolve_device,
    scene_from_path,
    selected_annotations,
)


@dataclass(frozen=True)
class HeadSimilarityConfig:
    dataset_root: Path
    checkpoint: Path
    output_json: Path
    output_md: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    scene_type: str = ""
    max_images: int = 0
    log_every: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_json", Path(self.output_json))
        object.__setattr__(self, "output_md", Path(self.output_md))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


def _normalize_rows(values: torch.Tensor) -> torch.Tensor:
    values = values.float().clamp_min(0)
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _cosine_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_norm = left / torch.linalg.vector_norm(left, dim=-1, keepdim=True).clamp_min(1e-12)
    right_norm = right / torch.linalg.vector_norm(right, dim=-1, keepdim=True).clamp_min(1e-12)
    return left_norm @ right_norm.transpose(0, 1)


def _jsd_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    left = _normalize_rows(left)
    right = _normalize_rows(right)
    rows: list[torch.Tensor] = []
    for row in left:
        row_expand = row.unsqueeze(0).expand_as(right)
        midpoint = 0.5 * (row_expand + right)
        kl_left = (row_expand * ((row_expand + eps).log() - (midpoint + eps).log())).sum(dim=-1)
        kl_right = (right * ((right + eps).log() - (midpoint + eps).log())).sum(dim=-1)
        rows.append(0.5 * (kl_left + kl_right))
    return torch.stack(rows, dim=0)


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


def _best_perm(matrix: list[list[float]], *, maximize: bool) -> tuple[list[int], float]:
    head_count = len(matrix)
    best_score: float | None = None
    best_perm: tuple[int, ...] | None = None
    for perm in itertools.permutations(range(head_count)):
        score = sum(matrix[i][perm[i]] for i in range(head_count)) / head_count
        if best_score is None or (score > best_score if maximize else score < best_score):
            best_score = score
            best_perm = perm
    if best_score is None or best_perm is None:
        raise RuntimeError("no head permutation found")
    return list(best_perm), best_score


def _matrix_to_lists(matrix: torch.Tensor) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix.tolist()]


def _mean_diag(matrix: list[list[float]]) -> float:
    return sum(matrix[i][i] for i in range(len(matrix))) / len(matrix)


def _transpose_perm(perm: list[int]) -> list[int]:
    inverse = [0] * len(perm)
    for left, right in enumerate(perm):
        inverse[right] = left
    return inverse


def _propagated_groups(assignments: list[list[int]]) -> list[list[int]]:
    if not assignments:
        return []
    head_count = len(assignments[0])
    groups: list[list[int]] = [[head] for head in range(head_count)]
    for perm in assignments:
        for group in groups:
            group.append(perm[group[-1]])
    return groups


def _head_dists_for_image(
    vit: torch.nn.Module,
    x: torch.Tensor,
    query_patch_indices: list[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    register_count = int(getattr(vit, "num_register_tokens", 0))
    patch_start = 1 + register_count
    patch_count = vit.patch_embed(x).shape[1]
    patch_slice = slice(patch_start, patch_start + patch_count)

    global_dists: list[torch.Tensor] = []
    point_dists: list[torch.Tensor] = []

    tokens = vit.prepare_tokens_with_masks(x)
    for block in vit.blocks:
        attn_input = block.norm1(tokens)
        probs = _attention_probs(block.attn, attn_input)[0].float()
        global_dist = probs[:, patch_slice, patch_slice].mean(dim=1)
        global_dists.append(_normalize_rows(global_dist.detach().cpu()))

        if query_patch_indices:
            query_token_indices = [patch_start + patch_index for patch_index in query_patch_indices]
            point_dist = probs[:, query_token_indices, patch_slice].permute(1, 0, 2)
            point_dists.append(_normalize_rows(point_dist.detach().cpu()))
        else:
            head_count = int(block.attn.num_heads)
            point_dists.append(torch.empty(0, head_count, patch_count))

        tokens = block(tokens)
        if x.is_cuda:
            torch.cuda.empty_cache()

    return global_dists, point_dists


def _new_pair_state(head_count: int) -> dict[str, Any]:
    zeros = [[0.0 for _ in range(head_count)] for _ in range(head_count)]
    return {
        "global_count": 0,
        "point_count": 0,
        "global_cos_sum": [row[:] for row in zeros],
        "global_jsd_sum": [row[:] for row in zeros],
        "point_cos_sum": [row[:] for row in zeros],
        "point_jsd_sum": [row[:] for row in zeros],
    }


def _add_matrix_sum(target: list[list[float]], matrix: torch.Tensor, weight: float = 1.0) -> None:
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            target[row_index][col_index] += float(matrix[row_index, col_index].item()) * weight


def _divide_matrix(matrix: list[list[float]], count: int) -> list[list[float]]:
    denom = max(count, 1)
    return [[value / denom for value in row] for row in matrix]


def _finalize_pair_state(state: dict[str, Any]) -> dict[str, Any]:
    global_cos = _divide_matrix(state["global_cos_sum"], state["global_count"])
    global_jsd = _divide_matrix(state["global_jsd_sum"], state["global_count"])
    point_cos = _divide_matrix(state["point_cos_sum"], state["point_count"])
    point_jsd = _divide_matrix(state["point_jsd_sum"], state["point_count"])

    global_best_cos_perm, global_best_cos = _best_perm(global_cos, maximize=True)
    global_best_jsd_perm, global_best_jsd = _best_perm(global_jsd, maximize=False)
    point_best_cos_perm, point_best_cos = _best_perm(point_cos, maximize=True)
    point_best_jsd_perm, point_best_jsd = _best_perm(point_jsd, maximize=False)
    return {
        "global_count": state["global_count"],
        "point_count": state["point_count"],
        "global_same_head_cosine": _mean_diag(global_cos),
        "global_same_head_jsd": _mean_diag(global_jsd),
        "global_best_cosine": global_best_cos,
        "global_best_cosine_perm": global_best_cos_perm,
        "global_best_jsd": global_best_jsd,
        "global_best_jsd_perm": global_best_jsd_perm,
        "point_same_head_cosine": _mean_diag(point_cos),
        "point_same_head_jsd": _mean_diag(point_jsd),
        "point_best_cosine": point_best_cos,
        "point_best_cosine_perm": point_best_cos_perm,
        "point_best_jsd": point_best_jsd,
        "point_best_jsd_perm": point_best_jsd_perm,
        "global_cosine_matrix": global_cos,
        "global_jsd_matrix": global_jsd,
        "point_cosine_matrix": point_cos,
        "point_jsd_matrix": point_jsd,
    }


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    meta = result["metadata"]
    lines: list[str] = [
        "# Attention Head Similarity",
        "",
        f"- Images analyzed: {meta['images_evaluated']}/{meta['images_requested']}",
        f"- Point-query distributions: {result['point_query_count']} endpoint patch queries",
        f"- Layers: {meta['layer_count']}; heads/layer: {meta['head_count']}",
        "",
        "## Adjacent Layers",
        "",
        "| layers | global same cos | global grouped cos | global same JSD | global grouped JSD | point same cos | point grouped cos | point same JSD | point grouped JSD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    adjacent = result["adjacent_pairs"]
    for key, row in adjacent.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    key.replace("-", " -> "),
                    _format_float(row["global_same_head_cosine"]),
                    _format_float(row["global_best_cosine"]),
                    _format_float(row["global_same_head_jsd"]),
                    _format_float(row["global_best_jsd"]),
                    _format_float(row["point_same_head_cosine"]),
                    _format_float(row["point_best_cosine"]),
                    _format_float(row["point_same_head_jsd"]),
                    _format_float(row["point_best_jsd"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Readout-Layer Pairs", ""])
    lines.append("| layers | global same cos | global grouped cos | global same JSD | global grouped JSD | point same cos | point grouped cos | point same JSD | point grouped JSD |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, row in result["readout_pairs"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    key.replace("-", " -> "),
                    _format_float(row["global_same_head_cosine"]),
                    _format_float(row["global_best_cosine"]),
                    _format_float(row["global_same_head_jsd"]),
                    _format_float(row["global_best_jsd"]),
                    _format_float(row["point_same_head_cosine"]),
                    _format_float(row["point_best_cosine"]),
                    _format_float(row["point_same_head_jsd"]),
                    _format_float(row["point_best_jsd"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Propagated Head Groups", ""])
    lines.append("Groups are propagated through adjacent-layer assignments optimized by JSD.")
    lines.extend(["", "### Global Patch Attention", ""])
    for index, group in enumerate(result["global_jsd_groups"]):
        lines.append(f"- group {index}: " + " -> ".join(f"L{layer}:H{head}" for layer, head in enumerate(group)))
    lines.extend(["", "### Point-Query Attention", ""])
    for index, group in enumerate(result["point_jsd_groups"]):
        lines.append(f"- group {index}: " + " -> ".join(f"L{layer}:H{head}" for layer, head in enumerate(group)))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def analyze(config: HeadSimilarityConfig) -> dict[str, Any]:
    require_ready(config.dataset_root, config.checkpoint, config.encoder)
    cv2 = load_cv2()
    device = resolve_device(config.device)
    model = load_model(config.encoder, config.checkpoint, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    layer_count = len(model.pretrained.blocks)
    head_count = int(model.pretrained.blocks[0].attn.num_heads)
    readout_layers = list(model.intermediate_layer_idx[model.encoder])
    pair_keys: list[tuple[int, int]] = []
    pair_keys.extend((i, i + 1) for i in range(layer_count - 1))
    for i, left in enumerate(readout_layers):
        for right in readout_layers[i + 1 :]:
            pair_keys.append((left, right))
    pair_state = {pair: _new_pair_state(head_count) for pair in pair_keys}

    selected = selected_annotations(config.dataset_root, scene_type=config.scene_type, max_images=config.max_images)
    if not selected:
        raise RuntimeError("no DA-2K annotations selected")

    missing_images: list[str] = []
    images_evaluated = 0
    point_query_count = 0
    by_scene = defaultdict(int)
    started = time.monotonic()

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

            query_patch_indices: set[int] = set()
            for pair in pairs:
                query_patch_indices.add(
                    _point_to_patch_index(
                        pair["point1"],
                        raw_height=raw_height,
                        raw_width=raw_width,
                        input_height=input_height,
                        input_width=input_width,
                        patch_h=patch_h,
                        patch_w=patch_w,
                    )
                )
                query_patch_indices.add(
                    _point_to_patch_index(
                        pair["point2"],
                        raw_height=raw_height,
                        raw_width=raw_width,
                        input_height=input_height,
                        input_width=input_width,
                        patch_h=patch_h,
                        patch_w=patch_w,
                    )
                )

            query_patch_list = sorted(query_patch_indices)
            global_dists, point_dists = _head_dists_for_image(model.pretrained, x, query_patch_list)
            images_evaluated += 1
            point_query_count += len(query_patch_list)
            by_scene[scene_from_path(relative_path)] += 1

            for pair in pair_keys:
                left, right = pair
                state = pair_state[pair]
                global_cos = _cosine_matrix(global_dists[left], global_dists[right])
                global_jsd = _jsd_matrix(global_dists[left], global_dists[right])
                _add_matrix_sum(state["global_cos_sum"], global_cos)
                _add_matrix_sum(state["global_jsd_sum"], global_jsd)
                state["global_count"] += 1

                if query_patch_list:
                    query_count = len(query_patch_list)
                    point_cos_sum = torch.zeros(head_count, head_count)
                    point_jsd_sum = torch.zeros(head_count, head_count)
                    for query_index in range(query_count):
                        point_cos_sum += _cosine_matrix(point_dists[left][query_index], point_dists[right][query_index])
                        point_jsd_sum += _jsd_matrix(point_dists[left][query_index], point_dists[right][query_index])
                    _add_matrix_sum(state["point_cos_sum"], point_cos_sum)
                    _add_matrix_sum(state["point_jsd_sum"], point_jsd_sum)
                    state["point_count"] += query_count

            if config.log_every > 0 and (image_index % config.log_every == 0 or image_index == len(selected)):
                elapsed = time.monotonic() - started
                print(f"analyzed {image_index}/{len(selected)} images in {elapsed:.1f}s", flush=True)

    finalized = {f"{left}-{right}": _finalize_pair_state(state) for (left, right), state in pair_state.items()}
    adjacent_pairs = {f"{i}-{i + 1}": finalized[f"{i}-{i + 1}"] for i in range(layer_count - 1)}
    readout_pairs = {
        f"{left}-{right}": finalized[f"{left}-{right}"]
        for i, left in enumerate(readout_layers)
        for right in readout_layers[i + 1 :]
    }
    global_jsd_groups = _propagated_groups([adjacent_pairs[f"{i}-{i + 1}"]["global_best_jsd_perm"] for i in range(layer_count - 1)])
    point_jsd_groups = _propagated_groups([adjacent_pairs[f"{i}-{i + 1}"]["point_best_jsd_perm"] for i in range(layer_count - 1)])
    result = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_requested": len(selected),
            "images_evaluated": images_evaluated,
            "missing_images": missing_images,
            "elapsed_seconds": time.monotonic() - started,
            "layer_count": layer_count,
            "head_count": head_count,
            "readout_layers": readout_layers,
        },
        "point_query_count": point_query_count,
        "images_by_scene": dict(sorted(by_scene.items())),
        "adjacent_pairs": adjacent_pairs,
        "readout_pairs": readout_pairs,
        "global_jsd_groups": global_jsd_groups,
        "point_jsd_groups": point_jsd_groups,
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    _write_markdown(result, config.output_md)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure DA-V2 attention-head similarity across layers.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/attention_head_similarity_full.json"))
    parser.add_argument("--output-md", type=Path, default=Path("eval_outputs/attention_head_similarity_full.md"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scene-type", choices=SCENE_CHOICES, default="")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = HeadSimilarityConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        output_md=args.output_md,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        scene_type=args.scene_type,
        max_images=args.max_images,
        log_every=args.log_every,
    )
    result = analyze(config)
    print(
        json.dumps(
            {
                "images_evaluated": result["metadata"]["images_evaluated"],
                "point_query_count": result["point_query_count"],
                "output_json": str(config.output_json),
                "output_md": str(config.output_md),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
