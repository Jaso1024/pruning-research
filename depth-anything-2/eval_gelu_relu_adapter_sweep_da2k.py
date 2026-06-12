from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from eval_gelu_relu_compensation_da2k import (
    MODEL_CONFIGS,
    _sample_batch,
    _select_dense_record_subset,
    collect_dense_mlp_calibration,
    evaluate_da2k_model,
    load_calibration_tensors,
    load_model,
    make_lora_optimizer,
    replace_gelu_with_relu,
    selected_annotations,
    transformer_mlp_names,
    write_summary,
)


ADAPTER_METHODS = {
    "lora",
    "dora",
    "loha",
    "lokr",
    "vera",
    "ia3",
    "glora",
    "fact_tucker",
}


@dataclass(frozen=True)
class AdapterSweepConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    calibration_images: int = 32
    calibration_tokens: int = 8192
    max_images: int = 32
    max_pairs: int = 0
    scene_type: str = ""
    methods: tuple[str, ...] = ("lora", "dora", "loha", "lokr", "vera", "ia3", "glora", "fact_tucker")
    rank: int = 32
    alpha: float = 32.0
    steps: int = 100
    lr: float = 3e-3
    batch_tokens: int = 2048
    weight_decay: float = 0.0
    optimizer: str = "radam"
    log_every: int = 16
    kronecker_factor: int = 4
    seed: int = 47

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "optimizer", self.optimizer.lower())
        methods = tuple(method.strip().lower() for method in self.methods if method.strip())
        object.__setattr__(self, "methods", methods)
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if not methods:
            raise ValueError("at least one method must be selected")
        unknown = set(methods) - ADAPTER_METHODS
        if unknown:
            raise ValueError(f"unknown adapter method(s): {sorted(unknown)}")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.batch_tokens <= 0:
            raise ValueError("batch_tokens must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.calibration_images <= 0:
            raise ValueError("calibration_images must be positive")
        if self.calibration_tokens <= 0:
            raise ValueError("calibration_tokens must be positive")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        if self.kronecker_factor <= 0:
            raise ValueError("kronecker_factor must be positive")


def parse_methods(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


class MatrixAdapter(nn.Module):
    def __init__(
        self,
        base_weight: torch.Tensor,
        *,
        method: str,
        rank: int,
        alpha: float,
        device: torch.device,
        ia3_axis: str = "out",
        kronecker_factor: int = 4,
        seed: int = 47,
    ) -> None:
        super().__init__()
        self.method = method
        self.rank = int(min(rank, base_weight.shape[0], base_weight.shape[1]))
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(max(self.rank, 1))
        self.ia3_axis = ia3_axis
        self.kronecker_factor = int(kronecker_factor)
        self.register_buffer("base_weight", base_weight.detach().to(device=device, dtype=torch.float32).clone())
        out_features, in_features = self.base_weight.shape
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + out_features * 17 + in_features)

        def normal(shape: tuple[int, ...], std: float) -> torch.Tensor:
            tensor = torch.empty(shape, dtype=torch.float32)
            torch.nn.init.normal_(tensor, mean=0.0, std=std, generator=generator)
            return tensor.to(device=device)

        if method == "lora":
            self.a = nn.Parameter(normal((self.rank, in_features), 1.0 / math.sqrt(in_features)))
            self.b = nn.Parameter(torch.zeros((out_features, self.rank), device=device))
        elif method == "dora":
            self.a = nn.Parameter(normal((self.rank, in_features), 1.0 / math.sqrt(in_features)))
            self.b = nn.Parameter(torch.zeros((out_features, self.rank), device=device))
            self.magnitude = nn.Parameter(self.base_weight.norm(dim=1).clamp_min(1e-6).clone())
        elif method == "loha":
            self.a1 = nn.Parameter(normal((self.rank, in_features), 1.0 / math.sqrt(in_features)))
            self.b1 = nn.Parameter(normal((out_features, self.rank), 1.0 / math.sqrt(self.rank)))
            self.a2 = nn.Parameter(normal((self.rank, in_features), 1.0 / math.sqrt(in_features)))
            self.b2 = nn.Parameter(torch.zeros((out_features, self.rank), device=device))
        elif method == "lokr":
            factor = self._valid_kronecker_factor(out_features, in_features, self.kronecker_factor)
            self.actual_kronecker_factor = factor
            sub_out = out_features // factor
            sub_in = in_features // factor
            sub_rank = min(self.rank, sub_out, sub_in)
            self.sub_rank = sub_rank
            self.a = nn.Parameter(normal((sub_rank, sub_in), 1.0 / math.sqrt(sub_in)))
            self.b = nn.Parameter(normal((sub_out, sub_rank), 1.0 / math.sqrt(sub_rank)))
            self.c = nn.Parameter(torch.zeros((factor, factor), device=device))
        elif method == "vera":
            self.register_buffer("a_fixed", normal((self.rank, in_features), 1.0 / math.sqrt(in_features)))
            self.register_buffer("b_fixed", normal((out_features, self.rank), 1.0 / math.sqrt(self.rank)))
            self.diag = nn.Parameter(torch.zeros((self.rank,), device=device))
        elif method == "ia3":
            size = out_features if ia3_axis == "out" else in_features
            self.log_scale = nn.Parameter(torch.zeros((size,), device=device))
        elif method == "glora":
            self.a = nn.Parameter(normal((self.rank, in_features), 1.0 / math.sqrt(in_features)))
            self.b = nn.Parameter(torch.zeros((out_features, self.rank), device=device))
            self.a_in = nn.Parameter(normal((self.rank, in_features), 1.0 / math.sqrt(in_features)))
            self.b_in = nn.Parameter(torch.zeros((in_features, self.rank), device=device))
        elif method == "fact_tucker":
            self.u = nn.Parameter(normal((out_features, self.rank), 1.0 / math.sqrt(self.rank)))
            self.v = nn.Parameter(normal((self.rank, in_features), 1.0 / math.sqrt(in_features)))
            self.core = nn.Parameter(torch.zeros((self.rank, self.rank), device=device))
        else:
            raise ValueError(f"unsupported method: {method}")

    @staticmethod
    def _valid_kronecker_factor(out_features: int, in_features: int, requested: int) -> int:
        for factor in range(requested, 0, -1):
            if out_features % factor == 0 and in_features % factor == 0:
                return factor
        return 1

    def additive_delta(self) -> torch.Tensor:
        if self.method == "lora":
            return self.scale * (self.b @ self.a)
        if self.method == "dora":
            return self.scale * (self.b @ self.a)
        if self.method == "loha":
            return self.scale * ((self.b1 @ self.a1) * (self.b2 @ self.a2))
        if self.method == "lokr":
            low_rank = self.b @ self.a
            return self.scale * torch.kron(self.c, low_rank)
        if self.method == "vera":
            return self.scale * ((self.b_fixed * self.diag.unsqueeze(0)) @ self.a_fixed)
        if self.method == "fact_tucker":
            return self.scale * (self.u @ self.core @ self.v)
        raise RuntimeError(f"{self.method} does not expose a plain additive delta")

    def effective_weight(self) -> torch.Tensor:
        if self.method == "dora":
            direction = self.base_weight + self.additive_delta()
            direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
            return direction * self.magnitude.unsqueeze(1)
        if self.method == "ia3":
            scale = torch.exp(self.log_scale)
            if self.ia3_axis == "out":
                return self.base_weight * scale.unsqueeze(1)
            return self.base_weight * scale.unsqueeze(0)
        if self.method == "glora":
            additive = self.scale * (self.b @ self.a)
            input_delta = self.scale * (self.b_in @ self.a_in)
            eye = torch.eye(input_delta.shape[0], device=input_delta.device, dtype=input_delta.dtype)
            return (self.base_weight + additive) @ (eye + input_delta)
        return self.base_weight + self.additive_delta()

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        if self.method == "glora":
            input_delta = self.scale * F.linear(F.linear(x.float(), self.a_in), self.b_in)
            additive = self.scale * (self.b @ self.a)
            return F.linear(x.float() + input_delta, self.base_weight + additive, bias)
        return F.linear(x.float(), self.effective_weight(), bias)

    def summary(self) -> dict[str, Any]:
        trainable = sum(param.numel() for param in self.parameters())
        delta = self.effective_weight().detach() - self.base_weight
        return {
            "method": self.method,
            "rank": self.rank,
            "alpha": self.alpha,
            "scale": self.scale,
            "trainable_parameters": int(trainable),
            "delta_weight_norm": float(delta.norm().item()),
            "base_weight_norm": float(self.base_weight.norm().item()),
            "delta_to_weight_norm": float(delta.norm().item() / max(self.base_weight.norm().item(), 1e-12)),
            "ia3_axis": self.ia3_axis if self.method == "ia3" else None,
            "kronecker_factor": getattr(self, "actual_kronecker_factor", None),
        }


def fit_adapter_method(
    *,
    relu_model: nn.Module,
    dense_records: dict[str, dict[str, torch.Tensor]],
    mlp_names: list[str],
    method: str,
    calibration_tokens: int,
    rank: int,
    alpha: float,
    steps: int,
    lr: float,
    batch_tokens: int,
    weight_decay: float,
    optimizer_name: str,
    kronecker_factor: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    summaries: list[dict[str, Any]] = []
    for layer_index, name in enumerate(tqdm(mlp_names, desc=f"fit {method} adapters", unit="mlp")):
        mlp = relu_model.get_submodule(name)
        x_in, y, token_count = _select_dense_record_subset(
            dense_records,
            name,
            calibration_tokens=calibration_tokens,
            generator=generator,
            device=device,
        )
        fc1_weight = mlp.fc1.weight.detach().float()
        fc1_bias = mlp.fc1.bias.detach().float() if mlp.fc1.bias is not None else None
        fc2_weight = mlp.fc2.weight.detach().float()
        fc2_bias = mlp.fc2.bias.detach().float() if mlp.fc2.bias is not None else None

        fc1_adapter = MatrixAdapter(
            fc1_weight,
            method=method,
            rank=rank,
            alpha=alpha,
            device=device,
            ia3_axis="out",
            kronecker_factor=kronecker_factor,
            seed=seed + layer_index * 101,
        )
        fc2_adapter = MatrixAdapter(
            fc2_weight,
            method=method,
            rank=rank,
            alpha=alpha,
            device=device,
            ia3_axis="in" if method == "ia3" else "out",
            kronecker_factor=kronecker_factor,
            seed=seed + layer_index * 101 + 1,
        )
        params = list(fc1_adapter.parameters()) + list(fc2_adapter.parameters())
        optimizer = make_lora_optimizer(params, name=optimizer_name, lr=lr, weight_decay=weight_decay)
        batch_size = min(batch_tokens, x_in.shape[0])

        with torch.no_grad():
            base_hidden = mlp.act(F.linear(x_in, fc1_weight, fc1_bias)).float()
            base_y = F.linear(base_hidden, fc2_weight, fc2_bias)
            initial_mse = F.mse_loss(base_y, y).item()

        losses: list[float] = []
        for step in range(steps):
            x_batch, y_batch = _sample_batch(x_in, y, batch_size=batch_size)
            hidden = mlp.act(fc1_adapter(x_batch, fc1_bias)).float()
            pred = fc2_adapter(hidden, fc2_bias)
            loss = F.mse_loss(pred, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step in {0, steps - 1}:
                losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            hidden = mlp.act(fc1_adapter(x_in, fc1_bias)).float()
            final_y = fc2_adapter(hidden, fc2_bias)
            final_mse = F.mse_loss(final_y, y).item()
            fc1_effective = fc1_adapter.effective_weight()
            fc2_effective = fc2_adapter.effective_weight()
            mlp.fc1.weight.copy_(fc1_effective.to(dtype=mlp.fc1.weight.dtype))
            mlp.fc2.weight.copy_(fc2_effective.to(dtype=mlp.fc2.weight.dtype))

        summaries.append(
            {
                "module": name,
                "method": method,
                "tokens_available": token_count,
                "tokens_used": int(x_in.shape[0]),
                "steps": int(steps),
                "lr": float(lr),
                "optimizer": optimizer_name,
                "initial_mse": initial_mse,
                "final_mse": final_mse,
                "first_last_batch_losses": losses,
                "fc1": fc1_adapter.summary(),
                "fc2": fc2_adapter.summary(),
            }
        )
        del fc1_adapter, fc2_adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summaries


def run(config: AdapterSweepConfig) -> dict[str, Any]:
    started = time.monotonic()
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    selected_items = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
        max_pairs=config.max_pairs,
    )
    if len(selected_items) < config.calibration_images:
        raise RuntimeError(f"selected {len(selected_items)} images, but calibration_images={config.calibration_images}")

    result: dict[str, Any] = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_selected": len(selected_items),
            "pairs_selected": sum(len(pairs) for _path, pairs in selected_items),
            "note": "GELU is replaced with ReLU, then each transformer MLP fc1/fc2 pair receives one adapter-family repair that is folded into dense weights before DA-2K evaluation.",
        },
        "variants": {},
    }
    summary_path = config.output_dir / "summary.json"
    write_summary(summary_path, result)

    dense_model = load_model(config.encoder, config.checkpoint, device)
    for param in dense_model.parameters():
        param.requires_grad_(False)
    mlp_names = transformer_mlp_names(dense_model)
    result["metadata"]["transformer_mlp_names"] = mlp_names
    calibration_tensors, calibration_paths = load_calibration_tensors(
        dense_model,
        dataset_root=config.dataset_root,
        items=selected_items,
        input_size=config.input_size,
        device=device,
        limit=config.calibration_images,
    )
    result["metadata"]["calibration_relative_paths"] = calibration_paths
    dense_records = collect_dense_mlp_calibration(
        dense_model=dense_model,
        mlp_names=mlp_names,
        calibration_tensors=calibration_tensors,
        device=device,
    )
    write_summary(summary_path, result)

    relu_template = load_model(config.encoder, config.checkpoint, device)
    replaced = replace_gelu_with_relu(relu_template)
    for param in relu_template.parameters():
        param.requires_grad_(False)

    for method in config.methods:
        model = copy.deepcopy(relu_template).to(device=device).eval()
        repair = fit_adapter_method(
            relu_model=model,
            dense_records=dense_records,
            mlp_names=mlp_names,
            method=method,
            calibration_tokens=config.calibration_tokens,
            rank=config.rank,
            alpha=config.alpha,
            steps=config.steps,
            lr=config.lr,
            batch_tokens=config.batch_tokens,
            weight_decay=config.weight_decay,
            optimizer_name=config.optimizer,
            kronecker_factor=config.kronecker_factor,
            seed=config.seed,
            device=device,
        )
        result["variants"][method] = {
            "metadata": {
                "activation": "GELU replaced with ReLU",
                "adapter_method": method,
                "rank": config.rank,
                "alpha": config.alpha,
                "steps": config.steps,
                "lr": config.lr,
                "optimizer": config.optimizer,
                "batch_tokens": config.batch_tokens,
                "replaced_modules": replaced,
                "repair": repair,
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
    parser = argparse.ArgumentParser(description="Depth Anything V2 GELU->ReLU low-rank adapter family sweep.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/gelu_relu_adapter_sweep"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-images", type=int, default=32)
    parser.add_argument("--calibration-tokens", type=int, default=8192)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--methods", default="lora,dora,loha,lokr,vera,ia3,glora,fact_tucker")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-tokens", type=int, default=2048)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", default="radam")
    parser.add_argument("--log-every", type=int, default=16)
    parser.add_argument("--kronecker-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=47)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = AdapterSweepConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        calibration_images=args.calibration_images,
        calibration_tokens=args.calibration_tokens,
        max_images=args.max_images,
        max_pairs=args.max_pairs,
        scene_type=args.scene_type,
        methods=parse_methods(args.methods),
        rank=args.rank,
        alpha=args.alpha,
        steps=args.steps,
        lr=args.lr,
        batch_tokens=args.batch_tokens,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        log_every=args.log_every,
        kronecker_factor=args.kronecker_factor,
        seed=args.seed,
    )
    summary = run(config)
    print(json.dumps({name: row["evaluation"]["overall"] for name, row in summary["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
