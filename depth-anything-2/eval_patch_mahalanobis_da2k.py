from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

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


ScoreKind = Literal["mahalanobis-diag", "mahalanobis-full"]
FillMode = Literal["zero", "input"]


@dataclass(frozen=True)
class MahalanobisPatchConfig:
    dataset_root: Path
    checkpoint: Path
    output_json: Path
    output_md: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    keep_percentages: tuple[float, ...] = (0.99, 0.98, 0.95, 0.90, 0.85, 0.80)
    score_kind: ScoreKind = "mahalanobis-diag"
    keep_high: bool = True
    fill_mode: FillMode = "zero"
    eps: float = 1e-5
    shrinkage: float = 0.05
    scene_type: str = ""
    max_images: int = 0
    log_every: int = 50

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
        if self.score_kind not in {"mahalanobis-diag", "mahalanobis-full"}:
            raise ValueError(f"unknown score kind: {self.score_kind}")
        if self.fill_mode not in {"zero", "input"}:
            raise ValueError("fill_mode must be zero or input")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


def _parse_keep_percentages(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _threshold_key(keep: float) -> str:
    return f"{keep:.4f}".rstrip("0").rstrip(".")


def _stable_topk_indices(scores: torch.Tensor, keep_count: int, *, keep_high: bool) -> torch.Tensor:
    if keep_count >= scores.numel():
        return torch.arange(scores.numel(), dtype=torch.long, device=scores.device)
    values = torch.topk(scores, k=keep_count, largest=keep_high, sorted=False).values
    cutoff = values.min() if keep_high else values.max()
    better = scores > cutoff if keep_high else scores < cutoff
    selected = torch.nonzero(better, as_tuple=False).flatten()
    needed = keep_count - selected.numel()
    if needed > 0:
        ties = torch.nonzero(scores == cutoff, as_tuple=False).flatten()
        selected = torch.cat((selected, ties[:needed]))
    return torch.sort(selected.to(dtype=torch.long))[0]


def mahalanobis_scores(
    patch_embeddings: torch.Tensor,
    *,
    score_kind: ScoreKind,
    eps: float,
    shrinkage: float,
) -> torch.Tensor:
    if patch_embeddings.ndim != 3 or patch_embeddings.shape[0] != 1:
        raise ValueError("patch_embeddings must have shape [1, patch_count, hidden_size]")
    values = patch_embeddings.detach()[0].float()
    centered = values - values.mean(dim=0, keepdim=True)
    if score_kind == "mahalanobis-diag":
        variance = centered.square().mean(dim=0).clamp_min(eps)
        return torch.sqrt((centered.square() / variance).sum(dim=1).clamp_min(0.0))
    if score_kind == "mahalanobis-full":
        token_count = centered.shape[0]
        covariance = centered.transpose(0, 1) @ centered / max(token_count - 1, 1)
        diagonal_mean = covariance.diag().mean().clamp_min(eps)
        identity = torch.eye(covariance.shape[0], dtype=covariance.dtype, device=covariance.device)
        covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal_mean * identity
        covariance = covariance + eps * diagonal_mean * identity
        solved = torch.linalg.solve(covariance, centered.transpose(0, 1)).transpose(0, 1)
        return torch.sqrt((centered * solved).sum(dim=1).clamp_min(0.0))
    raise ValueError(f"unknown score kind: {score_kind}")


def select_kept_patch_indices(
    patch_embeddings: torch.Tensor,
    *,
    keep_percentage: float,
    score_kind: ScoreKind,
    keep_high: bool,
    eps: float,
    shrinkage: float,
) -> torch.Tensor:
    scores = mahalanobis_scores(
        patch_embeddings,
        score_kind=score_kind,
        eps=eps,
        shrinkage=shrinkage,
    )
    return select_kept_patch_indices_from_scores(
        scores,
        keep_percentage=keep_percentage,
        keep_high=keep_high,
    )


def select_kept_patch_indices_from_scores(
    scores: torch.Tensor,
    *,
    keep_percentage: float,
    keep_high: bool,
) -> torch.Tensor:
    keep_count = max(1, math.ceil(scores.numel() * keep_percentage))
    return _stable_topk_indices(scores, keep_count, keep_high=keep_high)


def _prepare_pruned_tokens_from_indices(
    vit: torch.nn.Module,
    x: torch.Tensor,
    kept_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if x.shape[0] != 1:
        raise ValueError("patch pruning currently supports batch size 1")
    batch, _channels, width, height = x.shape
    patch_embeddings = vit.patch_embed(x)
    cls_token = vit.cls_token.expand(batch, -1, -1)
    full_tokens_for_pos = torch.cat((cls_token, patch_embeddings), dim=1)
    pos = vit.interpolate_pos_encoding(full_tokens_for_pos, width, height)
    cls_token = cls_token + pos[:, :1]
    patch_tokens = patch_embeddings + pos[:, 1:]

    kept_indices = kept_indices.to(device=x.device, dtype=torch.long)
    if getattr(vit, "register_tokens", None) is not None:
        register_tokens = vit.register_tokens.expand(batch, -1, -1)
        sequence = torch.cat((cls_token, register_tokens, patch_tokens[:, kept_indices]), dim=1)
    else:
        sequence = torch.cat((cls_token, patch_tokens[:, kept_indices]), dim=1)
    return sequence, patch_tokens, kept_indices, patch_embeddings.shape[1]


def get_pruned_intermediate_layers_from_indices(
    vit: torch.nn.Module,
    x: torch.Tensor,
    layers: list[int],
    *,
    kept_indices: torch.Tensor,
    fill_mode: FillMode,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    sequence, input_patch_tokens, kept_indices, patch_count = _prepare_pruned_tokens_from_indices(vit, x, kept_indices)
    layers_to_take = set(layers)
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    register_count = int(getattr(vit, "num_register_tokens", 0))
    patch_start = 1 + register_count

    for block_index, block in enumerate(vit.blocks):
        sequence = block(sequence)
        if block_index not in layers_to_take:
            continue
        normalized = vit.norm(sequence)
        class_token = normalized[:, 0]
        kept_patch_tokens = normalized[:, patch_start:]
        if fill_mode == "input":
            restored = vit.norm(input_patch_tokens)
        else:
            restored = torch.zeros(
                (1, patch_count, normalized.shape[-1]),
                dtype=normalized.dtype,
                device=normalized.device,
            )
        restored = restored.clone()
        restored[:, kept_indices] = kept_patch_tokens
        outputs.append((restored, class_token))

    if len(outputs) != len(layers):
        raise RuntimeError(f"only captured {len(outputs)} / {len(layers)} requested layers")
    return tuple(outputs)


@torch.no_grad()
def infer_depth_from_kept_indices(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    kept_indices: torch.Tensor,
    fill_mode: FillMode,
    raw_height: int,
    raw_width: int,
) -> torch.Tensor:
    patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
    layers = model.intermediate_layer_idx[model.encoder]
    features = get_pruned_intermediate_layers_from_indices(
        model.pretrained,
        x,
        layers,
        kept_indices=kept_indices,
        fill_mode=fill_mode,
    )
    depth = model.depth_head(features, patch_h, patch_w)
    depth = F.relu(depth).squeeze(1)
    depth = F.interpolate(depth[:, None], (raw_height, raw_width), mode="bilinear", align_corners=True)[0, 0]
    return depth.detach().float().cpu()


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Mahalanobis Patch-Norm Token Pruning",
        "",
        f"- Images: {result['metadata']['images_evaluated']}/{result['metadata']['images_requested']}",
        f"- Score: `{result['metadata']['score_kind']}`",
        f"- Keep high scores: `{result['metadata']['keep_high']}`",
        f"- Fill mode: `{result['metadata']['fill_mode']}`",
        "",
        "| keep | kept patches at 518 square | accuracy | correct | direction |",
        "|---:|---:|---:|---:|---|",
    ]
    for key, summary in result["thresholds"].items():
        overall = summary["overall"]
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(summary["kept_patch_count_at_square_input"]),
                    f"{overall['best_accuracy']:.6f}",
                    f"{overall['larger_correct']}/{overall['pairs']}",
                    str(overall["best_direction"]),
                ]
            )
            + " |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def evaluate(config: MahalanobisPatchConfig) -> dict[str, Any]:
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

    thresholds = {_threshold_key(keep): keep for keep in config.keep_percentages}
    totals = {key: empty_counts() for key in thresholds}
    by_scene = {key: defaultdict(empty_counts) for key in thresholds}
    missing_images: list[str] = []
    started = time.monotonic()
    images_evaluated = 0

    with torch.inference_mode():
        for index, (relative_path, pairs) in enumerate(selected, start=1):
            image_path = config.dataset_root / relative_path
            image = cv2.imread(str(image_path))
            if image is None:
                missing_images.append(str(image_path))
                continue
            raw_height, raw_width = image.shape[:2]
            x, _ = model.image2tensor(image, config.input_size)
            x = x.to(device)
            patch_embeddings = model.pretrained.patch_embed(x)
            scores = mahalanobis_scores(
                patch_embeddings,
                score_kind=config.score_kind,
                eps=config.eps,
                shrinkage=config.shrinkage,
            )
            kept_by_threshold = {
                key: select_kept_patch_indices_from_scores(
                    scores,
                    keep_percentage=keep,
                    keep_high=config.keep_high,
                )
                for key, keep in thresholds.items()
            }
            depths = {
                key: infer_depth_from_kept_indices(
                    model,
                    x,
                    kept_indices=kept_indices,
                    fill_mode=config.fill_mode,
                    raw_height=raw_height,
                    raw_width=raw_width,
                )
                for key, kept_indices in kept_by_threshold.items()
            }
            scene = scene_from_path(relative_path)
            for pair in pairs:
                if pair.get("closer_point") != "point1":
                    raise ValueError(f"unsupported closer_point in {relative_path}: {pair}")
                for key, depth in depths.items():
                    d1 = point_value(depth, pair["point1"])
                    d2 = point_value(depth, pair["point2"])
                    add_pair(totals[key], d1, d2)
                    add_pair(by_scene[key][scene], d1, d2)
            images_evaluated += 1
            if config.log_every > 0 and (index % config.log_every == 0 or index == len(selected)):
                elapsed = time.monotonic() - started
                print(f"evaluated {index}/{len(selected)} images in {elapsed:.1f}s", flush=True)

    patch_count = (config.input_size // 14) * (config.input_size // 14)
    result = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_requested": len(selected),
            "images_evaluated": images_evaluated,
            "missing_images": missing_images,
            "elapsed_seconds": time.monotonic() - started,
            "patch_count_at_square_input": patch_count,
            "rule": "DA-2K labels point1 as closer; Depth Anything V2 vits uses larger predicted values for closer points.",
        },
        "thresholds": {
            key: {
                "keep_percentage": keep,
                "kept_patch_count_at_square_input": math.ceil(patch_count * keep),
                "overall": finalize_counts(totals[key]),
                "by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(by_scene[key].items())},
            }
            for key, keep in thresholds.items()
        },
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    _write_markdown(result, config.output_md)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Mahalanobis patch-token pruning on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/da2k_vits_patch_mahalanobis.json"))
    parser.add_argument("--output-md", type=Path, default=Path("eval_outputs/da2k_vits_patch_mahalanobis.md"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keep-percentages", type=_parse_keep_percentages, default=(0.99, 0.98, 0.95, 0.90, 0.85, 0.80))
    parser.add_argument("--score-kind", choices=["mahalanobis-diag", "mahalanobis-full"], default="mahalanobis-diag")
    parser.add_argument("--keep-low", action="store_true", help="Keep low Mahalanobis-distance tokens instead of high-distance tokens.")
    parser.add_argument("--fill-mode", choices=["zero", "input"], default="zero")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    parser.add_argument("--scene-type", choices=SCENE_CHOICES, default="")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = MahalanobisPatchConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        output_md=args.output_md,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        keep_percentages=args.keep_percentages,
        score_kind=args.score_kind,
        keep_high=not args.keep_low,
        fill_mode=args.fill_mode,
        eps=args.eps,
        shrinkage=args.shrinkage,
        scene_type=args.scene_type,
        max_images=args.max_images,
        log_every=args.log_every,
    )
    result = evaluate(config)
    print(
        json.dumps(
            {
                key: summary["overall"]
                for key, summary in result["thresholds"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {config.output_json}")


if __name__ == "__main__":
    main()
