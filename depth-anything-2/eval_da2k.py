import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def load_model(encoder: str, checkpoint: Path, device: torch.device) -> DepthAnythingV2:
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def infer_depth(model: DepthAnythingV2, image, input_size: int, device: torch.device):
    tensor, (height, width) = model.image2tensor(image, input_size)
    tensor = tensor.to(device)
    depth = model(tensor)
    depth = F.interpolate(
        depth[:, None],
        (height, width),
        mode="bilinear",
        align_corners=True,
    )[0, 0]
    return depth.detach().float().cpu()


def point_value(depth: torch.Tensor, point: list[int]) -> float:
    row = max(0, min(int(point[0]), depth.shape[0] - 1))
    col = max(0, min(int(point[1]), depth.shape[1] - 1))
    return float(depth[row, col].item())


def scene_from_path(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 2 and parts[0] == "images":
        return parts[1]
    return "unknown"


def empty_counts() -> dict[str, int]:
    return {"pairs": 0, "smaller_correct": 0, "larger_correct": 0, "ties": 0}


def add_pair(counts: dict[str, int], d1: float, d2: float) -> None:
    counts["pairs"] += 1
    if d1 < d2:
        counts["smaller_correct"] += 1
    elif d1 > d2:
        counts["larger_correct"] += 1
    else:
        counts["ties"] += 1


def finalize_counts(counts: dict[str, int]) -> dict[str, float | int]:
    pairs = max(counts["pairs"], 1)
    smaller = counts["smaller_correct"] / pairs
    larger = counts["larger_correct"] / pairs
    return {
        **counts,
        "smaller_is_closer_accuracy": smaller,
        "larger_is_closer_accuracy": larger,
        "best_direction": "smaller" if smaller >= larger else "larger",
        "best_accuracy": max(smaller, larger),
        "tie_fraction": counts["ties"] / pairs,
    }


def evaluate(args: argparse.Namespace) -> dict:
    dataset_root = args.dataset_root
    annotations_path = dataset_root / "annotations.json"
    annotations = json.loads(annotations_path.read_text())
    selected = [
        (image_path, pairs)
        for image_path, pairs in annotations.items()
        if not args.scene_type or scene_from_path(image_path) == args.scene_type
    ]
    if args.max_images > 0:
        selected = selected[: args.max_images]

    device = resolve_device(args.device)
    model = load_model(args.encoder, args.checkpoint, device)

    total = empty_counts()
    by_scene = defaultdict(empty_counts)
    missing_images = []
    started = time.time()

    for index, (relative_path, pairs) in enumerate(selected, start=1):
        image_path = dataset_root / relative_path
        image = cv2.imread(str(image_path))
        if image is None:
            missing_images.append(str(image_path))
            continue

        depth = infer_depth(model, image, args.input_size, device)
        scene = scene_from_path(relative_path)
        for pair in pairs:
            if pair.get("closer_point") != "point1":
                raise ValueError(f"unsupported closer_point in {relative_path}: {pair}")
            d1 = point_value(depth, pair["point1"])
            d2 = point_value(depth, pair["point2"])
            add_pair(total, d1, d2)
            add_pair(by_scene[scene], d1, d2)

        if index % args.log_every == 0 or index == len(selected):
            print(f"evaluated {index}/{len(selected)} images")

    result = {
        "metadata": {
            "dataset_root": str(dataset_root),
            "checkpoint": str(args.checkpoint),
            "encoder": args.encoder,
            "input_size": args.input_size,
            "device": str(device),
            "scene_type": args.scene_type,
            "max_images": args.max_images,
            "images_requested": len(selected),
            "missing_images": missing_images,
            "elapsed_seconds": time.time() - started,
            "rule": "DA-2K labels point1 as closer; report both d(point1) < d(point2) and d(point1) > d(point2).",
        },
        "overall": finalize_counts(total),
        "by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(by_scene.items())},
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Depth Anything V2 on DA-2K point-pair annotations.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory containing annotations.json and images/.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scene-type", default="", choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"])
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/da2k_vits.json"))
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    summary = evaluate(parsed)
    print(json.dumps(summary["overall"], indent=2))
