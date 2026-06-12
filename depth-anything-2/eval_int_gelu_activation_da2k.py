from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_gelu_relu_compensation_da2k import (
    MODEL_CONFIGS,
    evaluate_da2k_model,
    load_model,
    selected_annotations,
    write_summary,
)
from eval_da2k import resolve_device


PWL_GRIDS: dict[str, list[float]] = {
    "pwl7": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    "pwl9": [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    "pwl13": [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "pwl17": [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
}

DEFAULT_ACTIVATIONS: tuple[str, ...] = (
    "dense",
    "relu",
    "hardgelu_s0_25",
    "hardgelu_s0_3125",
    "hardgelu_s0_375",
    "hardswish",
    "shifted_square_r2",
    "pwl7_q8",
    "pwl9_q8",
    "pwl13_q8",
    "pwl13_q12",
    "pwl17_q8",
    "pwl17_q12",
    "intpwl13_xq8_cq8",
    "intpwl13_xq10_cq8",
    "intpwl13_xq12_cq8",
)


def gelu_values(points: list[float]) -> list[float]:
    x = torch.tensor(points, dtype=torch.float64)
    y = F.gelu(x)
    return [float(v) for v in y]


class HardGateGELU(nn.Module):
    def __init__(self, slope: float) -> None:
        super().__init__()
        self.slope = float(slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.clamp(0.5 + self.slope * x, 0.0, 1.0)
        return x * gate


class HardSwishLike(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


class ShiftedClippedSquare(nn.Module):
    def __init__(self, radius: float) -> None:
        super().__init__()
        self.radius = float(radius)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        radius = self.radius
        mid = x * (x + radius) / (2.0 * radius)
        return torch.where(x <= -radius, torch.zeros_like(x), torch.where(x >= radius, x, mid))


class FixedPointPwlGELU(nn.Module):
    def __init__(self, points: list[float], fractional_bits: int) -> None:
        super().__init__()
        if len(points) < 2:
            raise ValueError("PWL GELU requires at least two points")
        if sorted(points) != points:
            raise ValueError("PWL points must be sorted")
        scale = 1 << int(fractional_bits)
        values = gelu_values(points)
        slopes: list[float] = []
        intercepts: list[float] = []
        for left_x, right_x, left_y, right_y in zip(points[:-1], points[1:], values[:-1], values[1:]):
            slope = (right_y - left_y) / (right_x - left_x)
            intercept = left_y - slope * left_x
            slopes.append(round(slope * scale) / scale)
            intercepts.append(round(intercept * scale) / scale)
        self.points = tuple(float(p) for p in points)
        self.fractional_bits = int(fractional_bits)
        self.register_buffer("breaks", torch.tensor(points, dtype=torch.float32))
        self.register_buffer("slopes", torch.tensor(slopes, dtype=torch.float32))
        self.register_buffer("intercepts", torch.tensor(intercepts, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        breaks = self.breaks.to(device=x.device, dtype=x.dtype)
        slopes = self.slopes.to(device=x.device, dtype=x.dtype)
        intercepts = self.intercepts.to(device=x.device, dtype=x.dtype)
        out = torch.where(x <= breaks[0], torch.zeros_like(x), x)
        for index in range(slopes.numel()):
            y = slopes[index] * x + intercepts[index]
            mask = (x > breaks[index]) & (x <= breaks[index + 1])
            out = torch.where(mask, y, out)
        return out

    def extra_repr(self) -> str:
        return f"points={self.points}, fractional_bits={self.fractional_bits}"


def rounded_shift(values: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0:
        return values
    half = 1 << (bits - 1)
    positive = torch.div(values + half, 1 << bits, rounding_mode="trunc")
    negative = -torch.div(-values + half, 1 << bits, rounding_mode="trunc")
    return torch.where(values >= 0, positive, negative)


class QuantizedPwlGELU(nn.Module):
    def __init__(self, points: list[float], input_fractional_bits: int, coeff_fractional_bits: int) -> None:
        super().__init__()
        if len(points) < 2:
            raise ValueError("PWL GELU requires at least two points")
        if sorted(points) != points:
            raise ValueError("PWL points must be sorted")
        input_scale = 1 << int(input_fractional_bits)
        coeff_scale = 1 << int(coeff_fractional_bits)
        values = gelu_values(points)
        break_ints: list[int] = []
        slope_ints: list[int] = []
        intercept_ints: list[int] = []
        for point in points:
            break_ints.append(round(point * input_scale))
        for left_x, right_x, left_y, right_y in zip(points[:-1], points[1:], values[:-1], values[1:]):
            slope = (right_y - left_y) / (right_x - left_x)
            intercept = left_y - slope * left_x
            slope_ints.append(round(slope * coeff_scale))
            intercept_ints.append(round(intercept * input_scale))
        self.points = tuple(float(p) for p in points)
        self.input_fractional_bits = int(input_fractional_bits)
        self.coeff_fractional_bits = int(coeff_fractional_bits)
        self.register_buffer("break_ints", torch.tensor(break_ints, dtype=torch.int64))
        self.register_buffer("slope_ints", torch.tensor(slope_ints, dtype=torch.int64))
        self.register_buffer("intercept_ints", torch.tensor(intercept_ints, dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_scale = 1 << self.input_fractional_bits
        x_int = torch.round(x.float() * input_scale).to(torch.int64)
        breaks = self.break_ints.to(device=x.device)
        slopes = self.slope_ints.to(device=x.device)
        intercepts = self.intercept_ints.to(device=x.device)
        out_int = torch.where(x_int <= breaks[0], torch.zeros_like(x_int), x_int)
        for index in range(slopes.numel()):
            y_int = rounded_shift(slopes[index] * x_int, self.coeff_fractional_bits) + intercepts[index]
            mask = (x_int > breaks[index]) & (x_int <= breaks[index + 1])
            out_int = torch.where(mask, y_int, out_int)
        return (out_int.to(torch.float32) / input_scale).to(dtype=x.dtype)

    def extra_repr(self) -> str:
        return (
            f"points={self.points}, input_fractional_bits={self.input_fractional_bits}, "
            f"coeff_fractional_bits={self.coeff_fractional_bits}"
        )


def pwl_points(grid_name: str) -> list[float]:
    try:
        return PWL_GRIDS[grid_name]
    except KeyError as exc:
        raise ValueError(f"unknown PWL grid: {grid_name}") from exc


def activation_factory(name: str) -> Callable[[], nn.Module]:
    if name == "relu":
        return lambda: nn.ReLU(inplace=False)
    if name == "hardswish":
        return HardSwishLike
    if name.startswith("hardgelu_s"):
        slope = float(name.removeprefix("hardgelu_s").replace("_", "."))
        return lambda: HardGateGELU(slope)
    if name.startswith("shifted_square_r"):
        radius = float(name.removeprefix("shifted_square_r").replace("_", "."))
        return lambda: ShiftedClippedSquare(radius)
    if name.startswith("intpwl"):
        # Format: intpwl13_xq10_cq8. This simulates integer activation values
        # plus integer affine PWL segments, then dequantizes back for the float model.
        grid_name, bit_spec = name.removeprefix("int").split("_xq", 1)
        input_bits_text, coeff_bits_text = bit_spec.split("_cq", 1)
        input_fractional_bits = int(input_bits_text)
        coeff_fractional_bits = int(coeff_bits_text)
        points = pwl_points(grid_name)
        return lambda: QuantizedPwlGELU(points, input_fractional_bits, coeff_fractional_bits)
    if name.startswith("pwl"):
        # Format: pwl7_q8, pwl9_q10, pwl13_q8.
        grid_name, q_name = name.split("_q", 1)
        fractional_bits = int(q_name)
        points = pwl_points(grid_name)
        return lambda: FixedPointPwlGELU(points, fractional_bits)
    raise ValueError(f"unknown activation: {name}")


def replace_gelu(model: nn.Module, factory: Callable[[], nn.Module]) -> list[str]:
    replaced: list[str] = []
    for module_name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.GELU):
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                setattr(module, child_name, factory())
                replaced.append(full_name)
    return replaced


def parse_activations(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class ActivationEvalConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    max_images: int = 32
    max_pairs: int = 0
    scene_type: str = ""
    activations: tuple[str, ...] = DEFAULT_ACTIVATIONS
    log_every: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        if not self.activations:
            raise ValueError("at least one activation must be selected")


def run(config: ActivationEvalConfig) -> dict:
    started = time.monotonic()
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    selected_items = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
        max_pairs=config.max_pairs,
    )
    if not selected_items:
        raise RuntimeError("no DA-2K annotations selected")

    result: dict = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_selected": len(selected_items),
            "pairs_selected": sum(len(pairs) for _path, pairs in selected_items),
            "note": "Drop-in GELU replacement sweep. No weight repair, no calibration fitting, no fine-tuning.",
        },
        "variants": {},
    }
    summary_path = config.output_dir / "summary.json"
    write_summary(summary_path, result)

    for activation in config.activations:
        model = load_model(config.encoder, config.checkpoint, device)
        for param in model.parameters():
            param.requires_grad_(False)
        if activation == "dense":
            replaced: list[str] = []
            activation_note = "original nn.GELU"
        else:
            replaced = replace_gelu(model, activation_factory(activation))
            activation_note = "integer-friendly drop-in GELU replacement"
        result["variants"][activation] = {
            "metadata": {
                "activation": activation,
                "activation_note": activation_note,
                "replaced_modules": replaced,
                "integer_ops": "int/fixed-point implementation can use compares, clamps, adds, and fixed-point multiplies/shifts; PWL variants use interval affine segments.",
            },
            "evaluation": evaluate_da2k_model(
                model=model,
                dataset_root=config.dataset_root,
                items=selected_items,
                input_size=config.input_size,
                device=device,
                log_every=config.log_every,
            ),
        }
        write_summary(summary_path, result)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result["metadata"]["elapsed_seconds"] = time.monotonic() - started
    write_summary(summary_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate integer-friendly GELU drop-in replacements on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/int_gelu_activation_sweep"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument(
        "--activations",
        default=",".join(DEFAULT_ACTIVATIONS),
    )
    parser.add_argument("--log-every", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ActivationEvalConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        max_images=args.max_images,
        max_pairs=args.max_pairs,
        scene_type=args.scene_type,
        activations=parse_activations(args.activations),
        log_every=args.log_every,
    )
    summary = run(config)
    print(json.dumps({name: row["evaluation"]["overall"] for name, row in summary["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
