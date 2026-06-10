from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import torch
from tqdm.auto import tqdm

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


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    kind: str
    module_name: str
    layer_index: int | None
    ablation: str


@dataclass(frozen=True)
class AblationSweepConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    component_types: tuple[str, ...] = ("block", "attn", "mlp")
    components: tuple[str, ...] = ()
    scene_type: str = ""
    max_images: int = 100
    max_components: int = 0
    score_direction: str = "larger"
    log_every: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        allowed = {"block", "attn", "mlp", "linear"}
        unknown = [kind for kind in self.component_types if kind not in allowed]
        if unknown:
            raise ValueError(f"unknown component type(s): {unknown}")
        if self.score_direction not in {"larger", "smaller", "best"}:
            raise ValueError("score_direction must be larger, smaller, or best")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.max_components < 0:
            raise ValueError("max_components must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


def parse_component_types(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_components(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def selected_annotations(
    dataset_root: Path,
    *,
    scene_type: str,
    max_images: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    annotations = json.loads((dataset_root / "annotations.json").read_text())
    selected = [
        (image_path, pairs)
        for image_path, pairs in annotations.items()
        if not scene_type or scene_from_path(image_path) == scene_type
    ]
    if max_images > 0:
        selected = selected[:max_images]
    return selected


def transformer_layer_index(module_name: str) -> int | None:
    prefix = "pretrained.blocks."
    if not module_name.startswith(prefix):
        return None
    parts = module_name.split(".")
    if len(parts) < 3:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def build_component_specs(
    model: torch.nn.Module,
    *,
    component_types: tuple[str, ...],
) -> list[ComponentSpec]:
    wanted = set(component_types)
    specs: list[ComponentSpec] = []
    if {"block", "attn", "mlp"} & wanted:
        for layer_index, block in enumerate(model.pretrained.blocks):
            block_name = f"pretrained.blocks.{layer_index}"
            if "block" in wanted:
                specs.append(
                    ComponentSpec(
                        name=f"block_{layer_index:02d}",
                        kind="block",
                        module_name=block_name,
                        layer_index=layer_index,
                        ablation="return_block_input",
                    )
                )
            if "attn" in wanted and hasattr(block, "attn"):
                specs.append(
                    ComponentSpec(
                        name=f"block_{layer_index:02d}_attn",
                        kind="attn",
                        module_name=f"{block_name}.attn",
                        layer_index=layer_index,
                        ablation="zero_module_output",
                    )
                )
            if "mlp" in wanted and hasattr(block, "mlp"):
                specs.append(
                    ComponentSpec(
                        name=f"block_{layer_index:02d}_mlp",
                        kind="mlp",
                        module_name=f"{block_name}.mlp",
                        layer_index=layer_index,
                        ablation="zero_module_output",
                    )
                )
    if "linear" in wanted:
        for module_name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            layer_index = transformer_layer_index(module_name)
            if layer_index is None:
                continue
            specs.append(
                ComponentSpec(
                    name=module_name.replace(".", "_"),
                    kind="linear",
                    module_name=module_name,
                    layer_index=layer_index,
                    ablation="zero_module_output",
                )
            )
    return specs


@contextmanager
def ablate_component(model: torch.nn.Module, spec: ComponentSpec) -> Iterator[None]:
    module = model.get_submodule(spec.module_name)

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        if spec.ablation == "return_block_input":
            return inputs[0]
        if spec.ablation == "zero_module_output":
            return torch.zeros_like(output)
        raise ValueError(f"unsupported ablation mode: {spec.ablation}")

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def infer_depth(model: torch.nn.Module, image, input_size: int, device: torch.device) -> torch.Tensor:
    tensor, (height, width) = model.image2tensor(image, input_size)
    tensor = tensor.to(device)
    depth = model(tensor)
    depth = torch.nn.functional.interpolate(
        depth[:, None],
        (height, width),
        mode="bilinear",
        align_corners=True,
    )[0, 0]
    return depth.detach().float().cpu()


def evaluate_da2k_model(
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

    for index, (relative_path, pairs) in enumerate(items, start=1):
        image_path = dataset_root / relative_path
        image = cv2.imread(str(image_path))
        if image is None:
            missing_images.append(str(image_path))
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


def score_overall(overall: dict[str, Any], direction: str) -> float:
    if direction == "best":
        return float(overall["best_accuracy"])
    return float(overall[f"{direction}_is_closer_accuracy"])


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    direction = summary["config"]["score_direction"]
    lines = [
        f"Ranking metric: `{direction}` DA-2K accuracy.",
        "",
        "| rank | component | kind | layer | score | delta | larger acc | smaller acc | pairs |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(summary["rows_by_score"], start=1):
        overall = row["overall"]
        layer = "" if row["layer_index"] is None else str(row["layer_index"])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    row["component"],
                    row["kind"],
                    layer,
                    f"{float(row['score']):.6f}",
                    f"{float(row['score_delta']):.6f}",
                    f"{float(overall['larger_is_closer_accuracy']):.6f}",
                    f"{float(overall['smaller_is_closer_accuracy']):.6f}",
                    str(overall["pairs"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def run_sweep(config: AblationSweepConfig) -> dict[str, Any]:
    torch.manual_seed(17)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True, default=str) + "\n"
    )

    items = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
    )
    if not items:
        raise RuntimeError("no DA-2K annotations selected")

    model = load_model(config.encoder, config.checkpoint, device)
    for param in model.parameters():
        param.requires_grad_(False)

    specs = build_component_specs(model, component_types=config.component_types)
    if config.components:
        wanted_components = set(config.components)
        specs = [spec for spec in specs if spec.name in wanted_components or spec.module_name in wanted_components]
        missing = sorted(wanted_components - {spec.name for spec in specs} - {spec.module_name for spec in specs})
        if missing:
            raise ValueError(f"requested component(s) not found: {missing}")
    if config.max_components > 0:
        specs = specs[: config.max_components]
    if not specs:
        raise RuntimeError("no components selected for ablation")
    (config.output_dir / "components.json").write_text(
        json.dumps([asdict(spec) for spec in specs], indent=2, sort_keys=True) + "\n"
    )

    baseline = evaluate_da2k_model(
        model=model,
        dataset_root=config.dataset_root,
        items=items,
        input_size=config.input_size,
        device=device,
        log_every=config.log_every,
    )
    baseline_score = score_overall(baseline["overall"], config.score_direction)

    rows: list[dict[str, Any]] = []
    rows_path = config.output_dir / "rows.jsonl"
    for spec in tqdm(specs, desc="ablation sweep", unit="component"):
        started = time.monotonic()
        with ablate_component(model, spec):
            result = evaluate_da2k_model(
                model=model,
                dataset_root=config.dataset_root,
                items=items,
                input_size=config.input_size,
                device=device,
                log_every=0,
            )
        score = score_overall(result["overall"], config.score_direction)
        row = {
            "component": spec.name,
            "kind": spec.kind,
            "module_name": spec.module_name,
            "layer_index": spec.layer_index,
            "ablation": spec.ablation,
            "score_direction": config.score_direction,
            "score": score,
            "baseline_score": baseline_score,
            "score_delta": score - baseline_score,
            "elapsed_seconds": time.monotonic() - started,
            **result,
        }
        rows.append(row)
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        print(
            json.dumps(
                {
                    "component": row["component"],
                    "kind": row["kind"],
                    "layer_index": row["layer_index"],
                    "score": row["score"],
                    "score_delta": row["score_delta"],
                    "overall": row["overall"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    rows_by_score = sorted(
        rows,
        key=lambda row: (
            -float(row["score"]),
            int(row["layer_index"]) if row["layer_index"] is not None else math.inf,
            str(row["component"]),
        ),
    )
    rows_by_damage = sorted(
        rows,
        key=lambda row: (
            float(row["score_delta"]),
            int(row["layer_index"]) if row["layer_index"] is not None else math.inf,
            str(row["component"]),
        ),
    )
    summary = {
        "config": asdict(config),
        "device": str(device),
        "baseline": baseline,
        "baseline_score": baseline_score,
        "component_count": len(specs),
        "rows_by_score": rows_by_score,
        "rows_by_damage": rows_by_damage,
        "rule": (
            "Each run temporarily ablates exactly one component. block ablation returns the block input; "
            "attn/mlp/linear ablation replaces that module output with zeros. Hooks are removed before the next run."
        ),
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    write_markdown(config.output_dir / "summary.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep one-component-at-a-time ablations on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/da2k_vits_component_ablation"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--component-types", default="block,attn,mlp")
    parser.add_argument("--components", default="", help="Comma-separated component names or module names to evaluate.")
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--max-components", type=int, default=0)
    parser.add_argument("--score-direction", choices=["larger", "smaller", "best"], default="larger")
    parser.add_argument("--log-every", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = AblationSweepConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        component_types=parse_component_types(args.component_types),
        components=parse_components(args.components),
        scene_type=args.scene_type,
        max_images=args.max_images,
        max_components=args.max_components,
        score_direction=args.score_direction,
        log_every=args.log_every,
    )
    summary = run_sweep(config)
    print(
        json.dumps(
            {
                "baseline": summary["baseline"]["overall"],
                "best_ablations": [
                    {
                        "component": row["component"],
                        "kind": row["kind"],
                        "layer_index": row["layer_index"],
                        "score": row["score"],
                        "score_delta": row["score_delta"],
                        "overall": row["overall"],
                    }
                    for row in summary["rows_by_score"][:10]
                ],
                "worst_ablations": [
                    {
                        "component": row["component"],
                        "kind": row["kind"],
                        "layer_index": row["layer_index"],
                        "score": row["score"],
                        "score_delta": row["score_delta"],
                        "overall": row["overall"],
                    }
                    for row in summary["rows_by_damage"][:10]
                ],
                "output_dir": str(config.output_dir),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
