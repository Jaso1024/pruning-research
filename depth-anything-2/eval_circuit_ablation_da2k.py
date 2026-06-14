from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from eval_da2k import MODEL_CONFIGS, add_pair, empty_counts, finalize_counts, point_value, resolve_device, scene_from_path
from eval_gelu_relu_compensation_da2k import load_model, selected_annotations
from eval_attribution_patching_da2k import CircuitNodeSpec, build_circuit_nodes, image_to_tensor, parse_csv


@dataclass(frozen=True)
class CircuitAblationConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    node_types: tuple[str, ...] = ("attn", "mlp", "head")
    components: tuple[str, ...] = ()
    scene_type: str = ""
    max_images: int = 64
    max_pairs: int = 0
    seed: int = 123
    ablation_mode: str = "zero"
    log_every: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        allowed_node_types = {"block", "attn", "mlp", "linear", "mlp_linear", "head", "head_conv"}
        unknown_node_types = sorted(set(self.node_types) - allowed_node_types)
        if unknown_node_types:
            raise ValueError(f"unknown node type(s): {unknown_node_types}")
        if self.ablation_mode != "zero":
            raise ValueError("only zero ablation is implemented")


def zero_output(output: Any) -> Any:
    if torch.is_tensor(output):
        return torch.zeros_like(output)
    if isinstance(output, tuple):
        return tuple(zero_output(item) for item in output)
    if isinstance(output, list):
        return [zero_output(item) for item in output]
    if isinstance(output, dict):
        return {key: zero_output(value) for key, value in output.items()}
    return output


@contextmanager
def ablate_module(model: torch.nn.Module, module_name: str) -> Iterator[None]:
    module = model.get_submodule(module_name)

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        return zero_output(output)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def evaluate_items(
    *,
    model: torch.nn.Module,
    items: list[tuple[str, list[dict[str, Any]]]],
    dataset_root: Path,
    input_size: int,
    device: torch.device,
    desc: str,
    log_every: int,
) -> dict[str, Any]:
    counts = empty_counts()
    by_scene = defaultdict(empty_counts)
    mean_margins: list[float] = []
    missing_images: list[str] = []
    evaluated_images = 0
    for index, (relative_path, pairs) in enumerate(tqdm(items, desc=desc, unit="image", leave=False), start=1):
        image_path = dataset_root / relative_path
        image = cv2.imread(str(image_path))
        if image is None:
            missing_images.append(str(image_path))
            continue
        tensor, height, width = image_to_tensor(model, image, input_size, device)
        depth = model(tensor)
        depth = F.interpolate(depth[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
        scene = scene_from_path(relative_path)
        margins: list[float] = []
        for pair in pairs:
            d1 = point_value(depth.detach().float().cpu(), pair["point1"])
            d2 = point_value(depth.detach().float().cpu(), pair["point2"])
            add_pair(counts, d1, d2)
            add_pair(by_scene[scene], d1, d2)
            margins.append(d1 - d2)
        if margins:
            mean_margins.append(float(sum(margins) / len(margins)))
        evaluated_images += 1
        if log_every > 0 and (index % log_every == 0 or index == len(items)):
            overall = finalize_counts(counts)
            print(
                json.dumps(
                    {
                        "desc": desc,
                        "images": index,
                        "larger_accuracy": overall["larger_is_closer_accuracy"],
                        "pairs": overall["pairs"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    overall = finalize_counts(counts)
    return {
        "overall": overall,
        "by_scene": {scene: finalize_counts(scene_counts) for scene, scene_counts in sorted(by_scene.items())},
        "mean_margin": float(sum(mean_margins) / max(len(mean_margins), 1)),
        "evaluated_images": evaluated_images,
        "missing_images": missing_images,
    }


def run(config: CircuitAblationConfig) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    started = time.monotonic()
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True, default=str) + "\n")

    items = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
        max_pairs=config.max_pairs,
    )
    if not items:
        raise RuntimeError("no DA-2K annotations selected")

    model = load_model(config.encoder, config.checkpoint, device).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    specs = build_circuit_nodes(model, node_types=config.node_types)
    if config.components:
        wanted = set(config.components)
        specs = [spec for spec in specs if spec.name in wanted or spec.module_name in wanted]
        missing = sorted(wanted - {spec.name for spec in specs} - {spec.module_name for spec in specs})
        if missing:
            raise ValueError(f"requested component(s) not found: {missing}")
    if not specs:
        raise RuntimeError("no circuit nodes selected")
    (config.output_dir / "nodes.json").write_text(json.dumps([asdict(spec) for spec in specs], indent=2, sort_keys=True) + "\n")

    baseline = evaluate_items(
        model=model,
        items=items,
        dataset_root=config.dataset_root,
        input_size=config.input_size,
        device=device,
        desc="baseline",
        log_every=0,
    )
    baseline_acc = float(baseline["overall"]["larger_is_closer_accuracy"])
    baseline_correct = int(baseline["overall"]["larger_correct"])
    baseline_margin = float(baseline["mean_margin"])

    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        with ablate_module(model, spec.module_name):
            result = evaluate_items(
                model=model,
                items=items,
                dataset_root=config.dataset_root,
                input_size=config.input_size,
                device=device,
                desc=f"ablate {spec.name}",
                log_every=0,
            )
        row = {
            "component": spec.name,
            "kind": spec.kind,
            "module_name": spec.module_name,
            "layer_index": spec.layer_index,
            "overall": result["overall"],
            "by_scene": result["by_scene"],
            "mean_margin": result["mean_margin"],
            "accuracy_drop": baseline_acc - float(result["overall"]["larger_is_closer_accuracy"]),
            "correct_drop": baseline_correct - int(result["overall"]["larger_correct"]),
            "mean_margin_drop": baseline_margin - float(result["mean_margin"]),
        }
        rows.append(row)
        if config.log_every > 0 and (index % config.log_every == 0 or index == len(specs)):
            print(
                json.dumps(
                    {
                        "nodes_done": index,
                        "nodes_total": len(specs),
                        "last_component": spec.name,
                        "last_accuracy_drop": row["accuracy_drop"],
                        "last_correct_drop": row["correct_drop"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows_by_accuracy_drop = sorted(rows, key=lambda row: (-float(row["accuracy_drop"]), -float(row["mean_margin_drop"]), str(row["component"])))
    rows_by_margin_drop = sorted(rows, key=lambda row: (-float(row["mean_margin_drop"]), -float(row["accuracy_drop"]), str(row["component"])))
    rows_by_safe = sorted(rows, key=lambda row: (float(row["accuracy_drop"]), float(row["mean_margin_drop"]), str(row["component"])))
    summary = {
        "config": asdict(config),
        "device": str(device),
        "baseline": baseline,
        "node_count": len(specs),
        "image_count": len(items),
        "rows_by_accuracy_drop": rows_by_accuracy_drop,
        "rows_by_margin_drop": rows_by_margin_drop,
        "rows_by_safe": rows_by_safe,
        "metadata": {
            "elapsed_seconds": time.monotonic() - started,
            "method": "Circuit-wise zero ablation: each selected module output is replaced by zeros during DA-2K evaluation.",
        },
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate circuit-wise zero ablations for Depth Anything V2 on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/circuit_ablation_da2k"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--node-types", default="attn,mlp,head")
    parser.add_argument("--components", default="")
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ablation-mode", choices=["zero"], default="zero")
    parser.add_argument("--log-every", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = CircuitAblationConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        node_types=parse_csv(args.node_types),
        components=parse_csv(args.components),
        scene_type=args.scene_type,
        max_images=args.max_images,
        max_pairs=args.max_pairs,
        seed=args.seed,
        ablation_mode=args.ablation_mode,
        log_every=args.log_every,
    )
    summary = run(config)
    print(
        json.dumps(
            {
                "baseline": summary["baseline"]["overall"],
                "output_dir": str(config.output_dir),
                "top_drop": [
                    {
                        "component": row["component"],
                        "accuracy_drop": row["accuracy_drop"],
                        "correct_drop": row["correct_drop"],
                        "mean_margin_drop": row["mean_margin_drop"],
                    }
                    for row in summary["rows_by_accuracy_drop"][:12]
                ],
                "safest": [
                    {
                        "component": row["component"],
                        "accuracy_drop": row["accuracy_drop"],
                        "correct_drop": row["correct_drop"],
                        "mean_margin_drop": row["mean_margin_drop"],
                    }
                    for row in summary["rows_by_safe"][:12]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
