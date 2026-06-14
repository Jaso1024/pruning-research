from __future__ import annotations

import argparse
import importlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

CHECKPOINT_URLS = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true",
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
}

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


class SetupError(RuntimeError):
    pass


@dataclass(frozen=True)
class DA2KConfig:
    dataset_root: Path
    checkpoint: Path
    output_json: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    scene_type: str = ""
    max_images: int = 0
    log_every: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_json", Path(self.output_json))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


def setup_instructions(
    *,
    dataset_root: Path = Path("datasets/DA-2K/extracted/DA-2K"),
    checkpoint: Path = Path("checkpoints/depth_anything_v2_vits.pth"),
    encoder: str = "vits",
) -> str:
    checkpoint_url = CHECKPOINT_URLS.get(encoder, "<official Depth Anything V2 checkpoint URL>")
    return f"""Depth Anything V2 / DA-2K setup required.

Expected files:
  dataset root: {dataset_root}
  annotations:  {dataset_root / "annotations.json"}
  images dir:    {dataset_root / "images"}
  checkpoint:    {checkpoint}

Install code and Python dependencies:
  cd /home/ubuntu
  git clone https://github.com/DepthAnything/Depth-Anything-V2.git
  /home/ubuntu/not_jason/cot2_eval_venv/bin/python -m pip install -r /home/ubuntu/Depth-Anything-V2/requirements.txt opencv-python huggingface_hub
  export PYTHONPATH=/home/ubuntu/Depth-Anything-V2:$PYTHONPATH

Place the {encoder} checkpoint:
  mkdir -p {checkpoint.parent}
  wget -O {checkpoint} '{checkpoint_url}'

Fetch DA-2K without changing this harness:
  mkdir -p {dataset_root}
  /home/ubuntu/not_jason/cot2_eval_venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="depth-anything/DA-2K",
    repo_type="dataset",
    local_dir="{dataset_root}",
    local_dir_use_symlinks=False,
)
PY
"""


def check_dataset(dataset_root: Path) -> list[str]:
    problems: list[str] = []
    if not dataset_root.exists():
        problems.append(f"dataset root is missing: {dataset_root}")
    if not (dataset_root / "annotations.json").is_file():
        problems.append(f"annotations.json is missing: {dataset_root / 'annotations.json'}")
    if not (dataset_root / "images").is_dir():
        problems.append(f"images directory is missing: {dataset_root / 'images'}")
    return problems


def require_ready(dataset_root: Path, checkpoint: Path, encoder: str) -> None:
    problems = check_dataset(dataset_root)
    if not checkpoint.is_file():
        problems.append(f"checkpoint is missing: {checkpoint}")
    if problems:
        raise SetupError("\n".join(problems) + "\n\n" + setup_instructions(
            dataset_root=dataset_root,
            checkpoint=checkpoint,
            encoder=encoder,
        ))


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def load_model(encoder: str, checkpoint: Path, device: torch.device):
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"unknown encoder: {encoder}")
    if not checkpoint.is_file():
        raise SetupError(f"checkpoint is missing: {checkpoint}\n\n" + setup_instructions(
            checkpoint=checkpoint,
            encoder=encoder,
        ))
    try:
        dpt = importlib.import_module("depth_anything_v2.dpt")
    except ModuleNotFoundError as exc:
        raise SetupError("Python package depth_anything_v2 is not importable.\n\n" + setup_instructions(
            checkpoint=checkpoint,
            encoder=encoder,
        )) from exc
    DepthAnythingV2 = dpt.DepthAnythingV2
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def load_cv2():
    try:
        return importlib.import_module("cv2")
    except ModuleNotFoundError as exc:
        raise SetupError("Python package cv2 is not importable. Install it with:\n"
                         "  /home/ubuntu/not_jason/cot2_eval_venv/bin/python -m pip install opencv-python") from exc


def scene_from_path(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 2 and parts[0] == "images":
        return parts[1]
    return "unknown"


def selected_annotations(
    dataset_root: Path,
    *,
    scene_type: str,
    max_images: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    annotation_path = dataset_root / "annotations.json"
    if not annotation_path.is_file():
        raise SetupError(f"annotations.json is missing: {annotation_path}\n\n" + setup_instructions(dataset_root=dataset_root))
    annotations = json.loads(annotation_path.read_text())
    selected = [
        (image_path, pairs)
        for image_path, pairs in annotations.items()
        if not scene_type or scene_from_path(image_path) == scene_type
    ]
    if max_images > 0:
        selected = selected[:max_images]
    return selected


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


def finalize_counts(counts: dict[str, int]) -> dict[str, float | int | str]:
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


def point_value(depth: torch.Tensor, point: list[int]) -> float:
    row = max(0, min(int(point[0]), depth.shape[0] - 1))
    col = max(0, min(int(point[1]), depth.shape[1] - 1))
    return float(depth[row, col].item())


@torch.no_grad()
def infer_depth(model: torch.nn.Module, image, input_size: int, device: torch.device) -> torch.Tensor:
    tensor, (height, width) = model.image2tensor(image, input_size)
    tensor = tensor.to(device)
    depth = model(tensor)
    depth = F.interpolate(depth[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
    return depth.detach().float().cpu()


def evaluate(config: DA2KConfig) -> dict[str, Any]:
    require_ready(config.dataset_root, config.checkpoint, config.encoder)
    cv2 = load_cv2()
    device = resolve_device(config.device)
    model = load_model(config.encoder, config.checkpoint, device)
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
            print(f"baseline: evaluated {index}/{len(selected)} images", flush=True)

    patch_count = (config.input_size // 14) * (config.input_size // 14)
    result = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "variant": "baseline",
            "images_requested": len(selected),
            "missing_images": missing_images,
            "elapsed_seconds": time.monotonic() - started,
            "patch_count_at_square_input": patch_count,
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
    parser = argparse.ArgumentParser(description="Evaluate baseline Depth Anything V2 on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/da2k_vits_baseline.json"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scene-type", default="", choices=SCENE_CHOICES)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = DA2KConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
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
