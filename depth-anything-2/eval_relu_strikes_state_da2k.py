from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import torch

from eval_da2k import (
    MODEL_CONFIGS,
    add_pair,
    empty_counts,
    finalize_counts,
    point_value,
    resolve_device,
    scene_from_path,
)
from eval_gelu_relu_compensation_da2k import (
    infer_depth,
    load_model,
    selected_annotations,
    write_summary,
)
from eval_relu_strikes_da2k import (
    ActivationSpec,
    activation_spec,
    install_mlp_activation,
    install_stage2,
)


def metadata_from_summary(summary_path: Path, variant_key: str) -> tuple[ActivationSpec, str, float]:
    summary = json.loads(summary_path.read_text())
    try:
        metadata = summary["variants"][variant_key]["metadata"]
    except KeyError as exc:
        raise KeyError(f"could not find variant {variant_key!r} in {summary_path}") from exc
    activation = ActivationSpec(**metadata["activation"])
    stage2 = metadata["stage2"]
    stage2_shift = float(metadata.get("stage2_shift", 0.0))
    return activation, stage2, stage2_shift


def evaluate_da2k_model_flushed(
    *,
    model: torch.nn.Module,
    dataset_root: Path,
    items: list[tuple[str, list[dict[str, Any]]]],
    input_size: int,
    device: torch.device,
    log_every: int,
) -> dict[str, Any]:
    total = empty_counts()
    by_scene = defaultdict(empty_counts)
    missing_images: list[str] = []
    started = time.monotonic()

    model.eval()
    for index, (relative_path, pairs) in enumerate(items, start=1):
        image = cv2.imread(str(dataset_root / relative_path))
        if image is None:
            missing_images.append(str(dataset_root / relative_path))
            continue
        depth = infer_depth(model, image, input_size, device)
        scene = scene_from_path(relative_path)
        for pair in pairs:
            if pair.get("closer_point") != "point1":
                raise ValueError(f"unsupported closer_point in {relative_path}: {pair}")
            d1 = point_value(depth, pair["point1"])
            d2 = point_value(depth, pair["point2"])
            add_pair(total, d1, d2)
            add_pair(by_scene[scene], d1, d2)
        del depth
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if log_every > 0 and (index % log_every == 0 or index == len(items)):
            print(f"evaluated {index}/{len(items)} images", flush=True)

    return {
        "metadata": {
            "images_requested": len(items),
            "missing_images": missing_images,
            "elapsed_seconds": time.monotonic() - started,
        },
        "overall": finalize_counts(total),
        "by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(by_scene.items())},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    if args.summary_json is not None:
        activation, stage2, stage2_shift = metadata_from_summary(args.summary_json, args.variant_key)
    else:
        activation = activation_spec(args.activation)
        stage2 = args.stage2
        stage2_shift = args.stage2_shift

    items = selected_annotations(
        args.dataset_root,
        scene_type=args.scene_type,
        max_images=args.max_images,
        max_pairs=args.max_pairs,
    )
    all_items_count = len(items)
    if args.image_start < 0:
        raise ValueError("--image-start must be non-negative")
    if args.image_count < 0:
        raise ValueError("--image-count must be non-negative")
    if args.image_start or args.image_count:
        end = None if args.image_count == 0 else args.image_start + args.image_count
        items = items[args.image_start : end]
    model = load_model(args.encoder, args.checkpoint, device)
    changed_mlp = install_mlp_activation(model, activation)
    changed_stage2 = install_stage2(model, mode=stage2, shift=stage2_shift)
    state = torch.load(args.state_dict, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing}, unexpected={unexpected}")
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(device=device).eval()

    result = {
        "metadata": {
            "state_dict": str(args.state_dict),
            "summary_json": str(args.summary_json) if args.summary_json is not None else None,
            "variant_key": args.variant_key,
            "activation": activation.__dict__,
            "stage2": stage2,
            "stage2_shift": stage2_shift,
            "changed_mlp_modules": changed_mlp,
            "changed_stage2_modules": changed_stage2,
            "dataset_root": str(args.dataset_root),
            "checkpoint": str(args.checkpoint),
            "encoder": args.encoder,
            "input_size": args.input_size,
            "max_images": args.max_images,
            "max_pairs": args.max_pairs,
            "all_items_before_slice": all_items_count,
            "image_start": args.image_start,
            "image_count": args.image_count,
            "items_after_slice": len(items),
            "scene_type": args.scene_type,
            "device": str(device),
        },
        "evaluation": evaluate_da2k_model_flushed(
            model=model,
            dataset_root=args.dataset_root,
            items=items,
            input_size=args.input_size,
            device=device,
            log_every=args.log_every,
        ),
    }
    write_summary(args.output_json, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a folded ReLU Strikes Depth Anything state dict.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--variant-key", default="")
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--stage2", choices=["none", "norm2", "norm12"], default="none")
    parser.add_argument("--stage2-shift", type=float, default=0.0)
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--image-start", type=int, default=0)
    parser.add_argument("--image-count", type=int, default=0)
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--log-every", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result["evaluation"]["overall"], indent=2))


if __name__ == "__main__":
    main()
