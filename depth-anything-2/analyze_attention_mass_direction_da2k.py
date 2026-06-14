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
class DirectionConfig:
    dataset_root: Path
    checkpoint: Path
    output_json: Path
    output_pt: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    scene_type: str = ""
    max_images: int = 0
    train_modulo: int = 5
    train_remainder: int = 0
    ridge: float = 1e-3
    log_every: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_json", Path(self.output_json))
        object.__setattr__(self, "output_pt", Path(self.output_pt))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.train_modulo <= 1:
            raise ValueError("train_modulo must be > 1")
        if not 0 <= self.train_remainder < self.train_modulo:
            raise ValueError("train_remainder must be in [0, train_modulo)")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


def _attention_probs(attn: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    batch, token_count, channels = x.shape
    head_count = int(attn.num_heads)
    head_dim = channels // head_count
    qkv = attn.qkv(x).reshape(batch, token_count, 3, head_count, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q = qkv[0] * attn.scale
    k = qkv[1]
    return (q @ k.transpose(-2, -1)).softmax(dim=-1)


def _standardize(values: torch.Tensor) -> torch.Tensor:
    values = values.double()
    std = values.std(unbiased=False)
    if float(std.item()) <= 0:
        return torch.zeros_like(values)
    return (values - values.mean()) / std


def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    xc = x - x.mean()
    yc = y - y.mean()
    denom = torch.linalg.vector_norm(xc) * torch.linalg.vector_norm(yc)
    if float(denom.item()) <= 0:
        return float("nan")
    return float((xc @ yc / denom).item())


def _ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().double().cpu()
    order = torch.argsort(values)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    count = values.numel()
    index = 0
    while index < count:
        end = index + 1
        while end < count and sorted_values[end] == sorted_values[index]:
            end += 1
        rank = (index + end - 1) / 2.0
        ranks[order[index:end]] = rank
        index = end
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    return _corr(_ranks(x), _ranks(y))


def _summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.double)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def _new_stats(dim: int) -> dict[str, Any]:
    return {
        "n": 0,
        "sum_x": torch.zeros(dim, dtype=torch.float64),
        "sum_y": 0.0,
        "sum_xx": torch.zeros((dim, dim), dtype=torch.float64),
        "sum_xy": torch.zeros(dim, dtype=torch.float64),
        "sum_yy": 0.0,
    }


def _add_stats(stats: dict[str, Any], x: torch.Tensor, y: torch.Tensor) -> None:
    x = x.double().cpu()
    y = y.double().cpu()
    stats["n"] += x.shape[0]
    stats["sum_x"] += x.sum(dim=0)
    stats["sum_y"] += float(y.sum().item())
    stats["sum_xx"] += x.transpose(0, 1) @ x
    stats["sum_xy"] += x.transpose(0, 1) @ y
    stats["sum_yy"] += float((y @ y).item())


def _fit_directions(stats: dict[str, Any], ridge: float) -> dict[str, torch.Tensor | float]:
    n = int(stats["n"])
    if n <= 1:
        raise RuntimeError("not enough training tokens")
    mean_x = stats["sum_x"] / n
    mean_y = stats["sum_y"] / n
    cov_xx = stats["sum_xx"] / n - torch.outer(mean_x, mean_x)
    cov_xy = stats["sum_xy"] / n - mean_x * mean_y
    var_y = stats["sum_yy"] / n - mean_y * mean_y

    cov_direction = cov_xy / torch.linalg.vector_norm(cov_xy).clamp_min(1e-12)
    scale = cov_xx.diag().mean().clamp_min(1e-12)
    reg = cov_xx + ridge * scale * torch.eye(cov_xx.shape[0], dtype=cov_xx.dtype)
    ridge_direction = torch.linalg.solve(reg, cov_xy)
    ridge_direction = ridge_direction / torch.linalg.vector_norm(ridge_direction).clamp_min(1e-12)
    train_projection_var = float((ridge_direction @ cov_xx @ ridge_direction).item())
    train_cov = float((ridge_direction @ cov_xy).item())
    train_corr = train_cov / math.sqrt(max(train_projection_var, 1e-30) * max(float(var_y), 1e-30))
    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "cov_xy": cov_xy,
        "cov_direction": cov_direction,
        "ridge_direction": ridge_direction,
        "train_ridge_pearson": train_corr,
        "train_var_y": float(var_y),
        "ridge_scale": float(scale),
    }


def _mahalanobis_diag(x: torch.Tensor) -> torch.Tensor:
    centered = x - x.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0).clamp_min(1e-12)
    return torch.sqrt((centered.square() / variance).sum(dim=1).clamp_min(0.0))


def _collect_image(
    model: torch.nn.Module,
    image: Any,
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vit = model.pretrained
    block0 = vit.blocks[0]
    register_count = int(getattr(vit, "num_register_tokens", 0))
    patch_start = 1 + register_count

    x, _ = model.image2tensor(image, input_size)
    x = x.to(next(model.parameters()).device)
    tokens = vit.prepare_tokens_with_masks(x)
    attn_input = block0.norm1(tokens)
    probs = _attention_probs(block0.attn, attn_input)[0].float()
    tokens_after = block0(tokens)[0].float()
    patch_slice = slice(patch_start, tokens.shape[1])

    patch_tokens_after = tokens_after[patch_slice].detach().cpu()
    patch_attn = probs[:, patch_slice, patch_slice]
    incoming = patch_attn.mean(dim=(0, 1)).detach().cpu()
    incoming_z = _standardize(incoming)
    return patch_tokens_after, incoming, incoming_z, x.detach().cpu()


def _eval_scores(scores: dict[str, torch.Tensor], y: torch.Tensor) -> dict[str, dict[str, float]]:
    return {
        name: {
            "pearson": _corr(score, y),
            "spearman": _spearman(score, y),
        }
        for name, score in scores.items()
    }


def analyze(config: DirectionConfig) -> dict[str, Any]:
    require_ready(config.dataset_root, config.checkpoint, config.encoder)
    cv2 = load_cv2()
    device = resolve_device(config.device)
    model = load_model(config.encoder, config.checkpoint, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    selected = selected_annotations(config.dataset_root, scene_type=config.scene_type, max_images=config.max_images)
    if not selected:
        raise RuntimeError("no DA-2K annotations selected")

    hidden_size = int(model.pretrained.embed_dim)
    train_stats = _new_stats(hidden_size)
    split_counts = {"train_images": 0, "test_images": 0, "train_tokens": 0, "test_tokens": 0}
    missing_images: list[str] = []
    started = time.monotonic()

    with torch.inference_mode():
        for image_index, (relative_path, _pairs) in enumerate(selected):
            image = cv2.imread(str(config.dataset_root / relative_path))
            if image is None:
                missing_images.append(str(config.dataset_root / relative_path))
                continue
            is_test = image_index % config.train_modulo == config.train_remainder
            tokens_after, _incoming, incoming_z, _x = _collect_image(model, image, config.input_size)
            if not is_test:
                _add_stats(train_stats, tokens_after, incoming_z)
                split_counts["train_images"] += 1
                split_counts["train_tokens"] += int(tokens_after.shape[0])
            else:
                split_counts["test_images"] += 1
                split_counts["test_tokens"] += int(tokens_after.shape[0])
            if config.log_every > 0 and ((image_index + 1) % config.log_every == 0 or image_index + 1 == len(selected)):
                print(f"fit pass: {image_index + 1}/{len(selected)} images in {time.monotonic() - started:.1f}s", flush=True)

    fit = _fit_directions(train_stats, config.ridge)
    mean_x = fit["mean_x"].cpu()
    cov_direction = fit["cov_direction"].cpu()
    ridge_direction = fit["ridge_direction"].cpu()
    cov_xy = fit["cov_xy"].cpu()

    global_scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    global_target: list[torch.Tensor] = []
    per_image_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    scene_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    projection_pair_corrs: dict[str, list[float]] = defaultdict(list)
    eval_started = time.monotonic()

    with torch.inference_mode():
        for image_index, (relative_path, _pairs) in enumerate(selected):
            if image_index % config.train_modulo != config.train_remainder:
                continue
            image = cv2.imread(str(config.dataset_root / relative_path))
            if image is None:
                continue
            tokens_after, incoming, incoming_z, _x = _collect_image(model, image, config.input_size)
            centered = tokens_after.double() - mean_x
            scores = {
                "ridge_direction": centered @ ridge_direction,
                "cov_direction": centered @ cov_direction,
                "l2_norm": torch.linalg.vector_norm(tokens_after, ord=2, dim=1),
                "centered_l2_norm": torch.linalg.vector_norm(centered, ord=2, dim=1),
                "mahalanobis_diag": _mahalanobis_diag(tokens_after),
            }
            raw_scores = _eval_scores(scores, incoming_z)
            for score_name, metrics in raw_scores.items():
                for metric_name, value in metrics.items():
                    per_image_metrics[score_name][metric_name].append(value)
                    scene_metrics[scene_from_path(relative_path)][f"{score_name}_{metric_name}"].append(value)
            for score_name, score in scores.items():
                global_scores[score_name].append(score.detach().cpu())
            global_target.append(incoming_z.detach().cpu())
            projection_pair_corrs["ridge_vs_l2"].append(_corr(scores["ridge_direction"], scores["l2_norm"]))
            projection_pair_corrs["ridge_vs_mahalanobis_diag"].append(_corr(scores["ridge_direction"], scores["mahalanobis_diag"]))
            projection_pair_corrs["ridge_vs_raw_incoming"].append(_corr(scores["ridge_direction"], incoming))
            if config.log_every > 0 and ((image_index + 1) % config.log_every == 0 or image_index + 1 == len(selected)):
                print(f"eval pass: {image_index + 1}/{len(selected)} images in {time.monotonic() - eval_started:.1f}s", flush=True)

    target = torch.cat(global_target)
    global_eval = {
        score_name: {
            "pearson": _corr(torch.cat(parts), target),
            "spearman": _spearman(torch.cat(parts), target),
        }
        for score_name, parts in global_scores.items()
    }
    per_image_summary = {
        score_name: {
            metric_name: _summarize(values)
            for metric_name, values in metrics.items()
        }
        for score_name, metrics in per_image_metrics.items()
    }
    by_scene = {
        scene: {
            metric_name: _summarize(values)["mean"]
            for metric_name, values in metrics.items()
        }
        for scene, metrics in sorted(scene_metrics.items())
    }
    projection_relationships = {
        name: _summarize(values)
        for name, values in projection_pair_corrs.items()
    }

    result = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_requested": len(selected),
            "missing_images": missing_images,
            "elapsed_seconds": time.monotonic() - started,
            "target": "within-image z-scored incoming block-0 patch attention mass",
            "token_space": "patch tokens after transformer block 0 output",
        },
        "split": split_counts,
        "fit": {
            "train_ridge_pearson": fit["train_ridge_pearson"],
            "train_var_y": fit["train_var_y"],
            "ridge_scale": fit["ridge_scale"],
            "cov_xy_norm": float(torch.linalg.vector_norm(cov_xy).item()),
        },
        "global_test": global_eval,
        "per_image_test": per_image_summary,
        "projection_relationships_test": projection_relationships,
        "by_scene_test": by_scene,
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    config.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "mean_x": mean_x.float(),
            "cov_direction": cov_direction.float(),
            "ridge_direction": ridge_direction.float(),
            "cov_xy": cov_xy.float(),
        },
        config.output_pt,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find a post-block-0 token-space direction correlated with incoming attention mass.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/attention_mass_direction_block0.json"))
    parser.add_argument("--output-pt", type=Path, default=Path("eval_outputs/attention_mass_direction_block0.pt"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scene-type", choices=SCENE_CHOICES, default="")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--train-modulo", type=int, default=5)
    parser.add_argument("--train-remainder", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = DirectionConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        output_pt=args.output_pt,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        scene_type=args.scene_type,
        max_images=args.max_images,
        train_modulo=args.train_modulo,
        train_remainder=args.train_remainder,
        ridge=args.ridge,
        log_every=args.log_every,
    )
    result = analyze(config)
    print(
        json.dumps(
            {
                "split": result["split"],
                "fit": result["fit"],
                "global_test": result["global_test"],
                "projection_relationships_test": result["projection_relationships_test"],
                "output_json": str(config.output_json),
                "output_pt": str(config.output_pt),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
