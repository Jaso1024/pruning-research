from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
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
class ExperimentConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    calibration_images: int = 8
    calibration_tokens: int = 4096
    max_images: int = 32
    max_pairs: int = 0
    modes: tuple[str, ...] = ("dense", "relu", "newton", "hadamard", "lora")
    scene_type: str = ""
    log_every: int = 8
    ridge_lambda: float = 1e-3
    hadamard_block_size: int = 1024
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_steps: int = 200
    lora_lr: float = 3e-3
    lora_batch_tokens: int = 2048
    lora_weight_decay: float = 0.0
    lora_optimizer: str = "adamw"
    lora_relex_target_step: int = 200
    lora_relex_space: str = "folded"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.calibration_images <= 0:
            raise ValueError("calibration_images must be positive")
        if self.calibration_tokens <= 0:
            raise ValueError("calibration_tokens must be positive")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        if self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be non-negative")
        if self.hadamard_block_size < 0:
            raise ValueError("hadamard_block_size must be non-negative")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if self.lora_alpha <= 0.0:
            raise ValueError("lora_alpha must be positive")
        if self.lora_steps <= 0:
            raise ValueError("lora_steps must be positive")
        if self.lora_lr <= 0.0:
            raise ValueError("lora_lr must be positive")
        if self.lora_batch_tokens <= 0:
            raise ValueError("lora_batch_tokens must be positive")
        if self.lora_weight_decay < 0.0:
            raise ValueError("lora_weight_decay must be non-negative")
        if self.lora_relex_target_step <= 0:
            raise ValueError("lora_relex_target_step must be positive")
        object.__setattr__(self, "lora_relex_space", self.lora_relex_space.lower())
        if self.lora_relex_space not in {"folded", "factor"}:
            raise ValueError("lora_relex_space must be 'folded' or 'factor'")
        object.__setattr__(self, "lora_optimizer", self.lora_optimizer.lower())
        if self.lora_optimizer not in LORA_OPTIMIZERS:
            raise ValueError(f"unknown lora_optimizer: {self.lora_optimizer}")
        allowed = {
            "dense",
            "relu",
            "newton",
            "hadamard",
            "lora",
            "lora_hidden",
            "lora_fc2",
            "lora_fc1",
            "lora_output",
            "lora_sandwich",
            "lora_sandwich_relex",
        }
        unknown = set(self.modes) - allowed
        if unknown:
            raise ValueError(f"unknown mode(s): {sorted(unknown)}")


def parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in value.split(",") if part.strip())
    return modes or ("dense", "relu", "newton", "hadamard", "lora")


LORA_OPTIMIZERS = {
    "adagrad",
    "adam",
    "adamax",
    "adamw",
    "cmuon_full",
    "cmuon_iso",
    "muon",
    "nadam",
    "radam",
    "rmsprop",
    "sgd",
    "sgd_momentum",
}


class CompositionalLoRAOptimizer(torch.optim.Optimizer):
    """CM-inspired optimizer for LoRA factors whose effective matrix is B @ A.

    For A in R^{r x in} and B in R^{out x r}, the first-order composed
    perturbation is dB @ A + B @ dA. The "full" variant uses damped partner
    inverse square roots, while "iso" uses the paper's isotropic scalar
    approximation. Gradients are accumulated as raw momentum and whitened with
    the current partner geometry at every step.
    """

    def __init__(
        self,
        params: list[torch.Tensor],
        *,
        lr: float,
        weight_decay: float = 0.0,
        beta: float = 0.9,
        damping: float = 1e-2,
        variant: str = "full",
    ) -> None:
        if len(params) % 2 != 0:
            raise ValueError("CompositionalLoRAOptimizer expects [A, B] parameter pairs")
        if variant not in {"full", "iso"}:
            raise ValueError("variant must be 'full' or 'iso'")
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index in range(0, len(params), 2):
            a = params[index]
            b = params[index + 1]
            if a.ndim != 2 or b.ndim != 2:
                raise ValueError("CompositionalLoRAOptimizer only supports 2D LoRA factors")
            if a.shape[0] != b.shape[1]:
                raise ValueError(
                    "LoRA factor pair must satisfy A.shape[0] == B.shape[1]; "
                    f"got {tuple(a.shape)} and {tuple(b.shape)}"
                )
            pairs.append((a, b))
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta": beta,
            "damping": damping,
            "variant": variant,
        }
        super().__init__(params, defaults)
        self.pairs = pairs

    @staticmethod
    def _matrix_sign(matrix: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(matrix).all() or float(matrix.abs().max().detach().cpu()) == 0.0:
            return torch.zeros_like(matrix)
        u, _s, vh = torch.linalg.svd(matrix, full_matrices=False)
        return u @ vh

    @staticmethod
    def _inverse_sqrt_psd(matrix: torch.Tensor, damping: float) -> torch.Tensor:
        rank = matrix.shape[0]
        eye = torch.eye(rank, device=matrix.device, dtype=matrix.dtype)
        gram = 0.5 * (matrix + matrix.T) + float(damping) * eye
        eigvals, eigvecs = torch.linalg.eigh(gram)
        inv_sqrt = eigvals.clamp_min(1e-12).rsqrt()
        return (eigvecs * inv_sqrt.unsqueeze(0)) @ eigvecs.T

    def _momentum(self, param: torch.Tensor, grad: torch.Tensor, beta: float) -> torch.Tensor:
        state = self.state[param]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        momentum = state["momentum"]
        momentum.mul_(beta).add_(grad)
        return momentum

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        beta = float(group["beta"])
        damping = float(group["damping"])
        variant = str(group["variant"])

        for a, b in self.pairs:
            if a.grad is None and b.grad is None:
                continue
            grad_a = torch.zeros_like(a) if a.grad is None else a.grad.detach()
            grad_b = torch.zeros_like(b) if b.grad is None else b.grad.detach()
            mom_a = self._momentum(a, grad_a, beta)
            mom_b = self._momentum(b, grad_b, beta)

            if weight_decay:
                a.mul_(1.0 - lr * weight_decay)
                b.mul_(1.0 - lr * weight_decay)

            a_f = a.detach().float()
            b_f = b.detach().float()
            mom_a_f = mom_a.float()
            mom_b_f = mom_b.float()

            if variant == "iso":
                rank = max(int(a_f.shape[0]), 1)
                inv_a = float((a_f.square().sum() / rank + damping).rsqrt().detach().cpu())
                inv_b = float((b_f.square().sum() / rank + damping).rsqrt().detach().cpu())
                update_b = 0.5 * inv_a * self._matrix_sign(mom_b_f)
                update_a = 0.5 * inv_b * self._matrix_sign(mom_a_f)
            else:
                inv_a = self._inverse_sqrt_psd(a_f @ a_f.T, damping)
                inv_b = self._inverse_sqrt_psd(b_f.T @ b_f, damping)
                update_b = 0.5 * (self._matrix_sign(mom_b_f @ inv_a) @ inv_a)
                update_a = 0.5 * (inv_b @ self._matrix_sign(inv_b @ mom_a_f))

            b.add_(update_b.to(dtype=b.dtype), alpha=-lr)
            a.add_(update_a.to(dtype=a.dtype), alpha=-lr)
        return loss


class MuonLoRAOptimizer(torch.optim.Optimizer):
    """Plain per-factor Muon for 2D LoRA parameters."""

    def __init__(
        self,
        params: list[torch.Tensor],
        *,
        lr: float,
        weight_decay: float = 0.0,
        beta: float = 0.9,
    ) -> None:
        for param in params:
            if param.ndim != 2:
                raise ValueError("MuonLoRAOptimizer only supports 2D LoRA factors")
        defaults = {"lr": lr, "weight_decay": weight_decay, "beta": beta}
        super().__init__(params, defaults)

    def _momentum(self, param: torch.Tensor, grad: torch.Tensor, beta: float) -> torch.Tensor:
        state = self.state[param]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        momentum = state["momentum"]
        momentum.mul_(beta).add_(grad)
        return momentum

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            beta = float(group["beta"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)
                momentum = self._momentum(param, param.grad.detach(), beta)
                update = CompositionalLoRAOptimizer._matrix_sign(momentum.float())
                param.add_(update.to(dtype=param.dtype), alpha=-lr)
        return loss


def make_lora_optimizer(
    params: list[torch.Tensor],
    *,
    name: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if name == "cmuon_full":
        return CompositionalLoRAOptimizer(params, lr=lr, weight_decay=weight_decay, variant="full")
    if name == "cmuon_iso":
        return CompositionalLoRAOptimizer(params, lr=lr, weight_decay=weight_decay, variant="iso")
    if name == "muon":
        return MuonLoRAOptimizer(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "nadam":
        return torch.optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    if name == "radam":
        return torch.optim.RAdam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamax":
        return torch.optim.Adamax(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    raise ValueError(f"unknown lora optimizer: {name}")


def selected_annotations(
    dataset_root: Path,
    *,
    scene_type: str,
    max_images: int,
    max_pairs: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    annotations = json.loads((dataset_root / "annotations.json").read_text())
    selected: list[tuple[str, list[dict[str, Any]]]] = []
    pair_count = 0
    for image_path, pairs in annotations.items():
        if scene_type and scene_from_path(image_path) != scene_type:
            continue
        kept_pairs = list(pairs)
        if max_pairs > 0:
            remaining = max_pairs - pair_count
            if remaining <= 0:
                break
            kept_pairs = kept_pairs[:remaining]
        if kept_pairs:
            selected.append((image_path, kept_pairs))
            pair_count += len(kept_pairs)
        if max_images > 0 and len(selected) >= max_images:
            break
    return selected


def load_calibration_tensors(
    model: torch.nn.Module,
    *,
    dataset_root: Path,
    items: list[tuple[str, list[dict[str, Any]]]],
    input_size: int,
    device: torch.device,
    limit: int,
) -> tuple[list[torch.Tensor], list[str]]:
    tensors: list[torch.Tensor] = []
    paths: list[str] = []
    for relative_path, _pairs in items:
        if len(tensors) >= limit:
            break
        image = cv2.imread(str(dataset_root / relative_path))
        if image is None:
            continue
        tensor, _shape = model.image2tensor(image, input_size)
        tensors.append(tensor.to(device=device, non_blocking=True))
        paths.append(relative_path)
    if len(tensors) < limit:
        raise RuntimeError(f"loaded {len(tensors)} calibration images, requested {limit}")
    return tensors, paths


@torch.no_grad()
def infer_depth(model: torch.nn.Module, image, input_size: int, device: torch.device) -> torch.Tensor:
    tensor, (height, width) = model.image2tensor(image, input_size)
    tensor = tensor.to(device=device, non_blocking=True)
    depth = model(tensor)
    depth = F.interpolate(depth[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
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


def replace_gelu_with_relu(model: torch.nn.Module) -> list[str]:
    replaced: list[str] = []
    for module_name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.GELU):
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                setattr(module, child_name, nn.ReLU(inplace=False))
                replaced.append(full_name)
    return replaced


def transformer_mlp_names(model: torch.nn.Module) -> list[str]:
    names: list[str] = []
    for name, module in model.named_modules():
        if not name.startswith("pretrained.blocks."):
            continue
        if hasattr(module, "fc1") and hasattr(module, "fc2") and hasattr(module, "act"):
            if isinstance(module.fc1, nn.Linear) and isinstance(module.fc2, nn.Linear):
                names.append(name)
    return names


def collect_dense_mlp_calibration(
    *,
    dense_model: torch.nn.Module,
    mlp_names: list[str],
    calibration_tensors: list[torch.Tensor],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    records: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"inputs": [], "targets": []} for name in mlp_names
    }
    handles = []

    for name in mlp_names:
        module = dense_model.get_submodule(name)

        def make_hook(module_name: str):
            def hook(_module, inputs, output) -> None:
                records[module_name]["inputs"].append(inputs[0].detach().flatten(0, 1).float().cpu())
                records[module_name]["targets"].append(output.detach().flatten(0, 1).float().cpu())

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

    dense_model.eval()
    try:
        with torch.inference_mode():
            for tensor in tqdm(calibration_tensors, desc="collect dense MLP targets", unit="image"):
                _ = dense_model(tensor.to(device=device, non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()

    packed: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensors in records.items():
        packed[name] = {
            "inputs": torch.cat(tensors["inputs"], dim=0),
            "targets": torch.cat(tensors["targets"], dim=0),
        }
    return packed


def fit_fc2_least_squares_repair(
    *,
    relu_model: torch.nn.Module,
    dense_records: dict[str, dict[str, torch.Tensor]],
    mlp_names: list[str],
    calibration_tokens: int,
    ridge_lambda: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    summaries: list[dict[str, Any]] = []

    for name in tqdm(mlp_names, desc="fit fc2 repairs", unit="mlp"):
        mlp = relu_model.get_submodule(name)
        inputs = dense_records[name]["inputs"]
        targets = dense_records[name]["targets"]
        token_count = inputs.shape[0]
        if token_count > calibration_tokens:
            indices = torch.randperm(token_count, generator=generator)[:calibration_tokens]
            inputs = inputs.index_select(0, indices)
            targets = targets.index_select(0, indices)

        x_in = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        y = targets.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad():
            x = mlp.act(mlp.fc1(x_in)).float()
            ones = torch.ones((x.shape[0], 1), device=device, dtype=x.dtype)
            x_aug = torch.cat([x, ones], dim=1)

            old_y = mlp.fc2(x)
            old_mse = F.mse_loss(old_y, y).item()

            gram = x_aug.T @ x_aug
            if ridge_lambda > 0.0:
                gram.diagonal().add_(ridge_lambda)
                gram[-1, -1].sub_(ridge_lambda)
            rhs = x_aug.T @ y
            beta = torch.linalg.solve(gram, rhs)
            new_weight = beta[:-1].T.contiguous()
            new_bias = beta[-1].contiguous()
            new_y = x @ new_weight.T + new_bias
            new_mse = F.mse_loss(new_y, y).item()

            mlp.fc2.weight.copy_(new_weight.to(dtype=mlp.fc2.weight.dtype))
            if mlp.fc2.bias is not None:
                mlp.fc2.bias.copy_(new_bias.to(dtype=mlp.fc2.bias.dtype))

        summaries.append(
            {
                "module": name,
                "tokens_available": int(token_count),
                "tokens_used": int(inputs.shape[0]),
                "hidden_features": int(x.shape[1]),
                "out_features": int(y.shape[1]),
                "old_mse": old_mse,
                "new_mse": new_mse,
            }
        )
    return summaries


def fit_hidden_lora_identity_repair(
    *,
    relu_model: torch.nn.Module,
    dense_records: dict[str, dict[str, torch.Tensor]],
    mlp_names: list[str],
    calibration_tokens: int,
    rank: int,
    alpha: float,
    steps: int,
    lr: float,
    batch_tokens: int,
    weight_decay: float,
    optimizer_name: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(23)
    summaries: list[dict[str, Any]] = []

    for name in tqdm(mlp_names, desc="fit hidden LoRA identity repairs", unit="mlp"):
        mlp = relu_model.get_submodule(name)
        inputs = dense_records[name]["inputs"]
        targets = dense_records[name]["targets"]
        token_count = inputs.shape[0]
        if token_count > calibration_tokens:
            indices = torch.randperm(token_count, generator=generator)[:calibration_tokens]
            inputs = inputs.index_select(0, indices)
            targets = targets.index_select(0, indices)

        x_in = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        y = targets.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad():
            hidden = mlp.act(mlp.fc1(x_in)).float()
            weight = mlp.fc2.weight.detach().float()
            bias = mlp.fc2.bias.detach().float() if mlp.fc2.bias is not None else None
            base_y = F.linear(hidden, weight, bias)
            initial_mse = F.mse_loss(base_y, y).item()

        hidden_features = hidden.shape[1]
        used_rank = min(rank, hidden_features)
        scale = float(alpha) / float(used_rank)
        lora_a = torch.empty((used_rank, hidden_features), device=device, dtype=torch.float32)
        torch.nn.init.normal_(lora_a, mean=0.0, std=1.0 / math.sqrt(hidden_features))
        lora_b = torch.zeros((hidden_features, used_rank), device=device, dtype=torch.float32)
        lora_a.requires_grad_(True)
        lora_b.requires_grad_(True)
        optimizer = make_lora_optimizer([lora_a, lora_b], name=optimizer_name, lr=lr, weight_decay=weight_decay)

        batch_size = min(batch_tokens, hidden.shape[0])
        losses: list[float] = []
        for step in range(steps):
            if batch_size < hidden.shape[0]:
                batch_indices = torch.randint(
                    0,
                    hidden.shape[0],
                    (batch_size,),
                    device=device,
                )
                hidden_batch = hidden.index_select(0, batch_indices)
                y_batch = y.index_select(0, batch_indices)
            else:
                hidden_batch = hidden
                y_batch = y

            adapted = hidden_batch + scale * ((hidden_batch @ lora_a.T) @ lora_b.T)
            pred = F.linear(adapted, weight, bias)
            loss = F.mse_loss(pred, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step in {0, steps - 1}:
                losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            adapted = hidden + scale * ((hidden @ lora_a.T) @ lora_b.T)
            final_y = F.linear(adapted, weight, bias)
            final_mse = F.mse_loss(final_y, y).item()
            delta_weight = scale * ((weight @ lora_b) @ lora_a)
            mlp.fc2.weight.add_(delta_weight.to(dtype=mlp.fc2.weight.dtype))
            delta_norm = float(delta_weight.norm().item())
            weight_norm = float(weight.norm().item())

        summaries.append(
            {
                "module": name,
                "tokens_available": int(token_count),
                "tokens_used": int(inputs.shape[0]),
                "hidden_features": int(hidden_features),
                "out_features": int(y.shape[1]),
                "rank": int(used_rank),
                "alpha": float(alpha),
                "scale": float(scale),
                "steps": int(steps),
                "lr": float(lr),
                "optimizer": optimizer_name,
                "batch_tokens": int(batch_size),
                "initial_mse": initial_mse,
                "final_mse": final_mse,
                "first_last_batch_losses": losses,
                "delta_weight_norm": delta_norm,
                "base_weight_norm": weight_norm,
                "delta_to_weight_norm": delta_norm / max(weight_norm, 1e-12),
                "folded_delta": "fc2.weight <- fc2.weight + (alpha/rank) * fc2.weight @ B @ A",
                "identity_initialization": "adapter is h -> h + (alpha/rank) * B(Ah), with B initialized to zero so the initial adapter is exactly identity.",
            }
        )
        del hidden, y, lora_a, lora_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summaries


def _select_dense_record_subset(
    dense_records: dict[str, dict[str, torch.Tensor]],
    name: str,
    *,
    calibration_tokens: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    inputs = dense_records[name]["inputs"]
    targets = dense_records[name]["targets"]
    token_count = inputs.shape[0]
    if token_count > calibration_tokens:
        indices = torch.randperm(token_count, generator=generator)[:calibration_tokens]
        inputs = inputs.index_select(0, indices)
        targets = targets.index_select(0, indices)
    return (
        inputs.to(device=device, dtype=torch.float32, non_blocking=True),
        targets.to(device=device, dtype=torch.float32, non_blocking=True),
        int(token_count),
    )


def _init_lora_factors(
    *,
    in_features: int,
    out_features: int,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    used_rank = min(rank, in_features, out_features)
    lora_a = torch.empty((used_rank, in_features), device=device, dtype=torch.float32)
    torch.nn.init.normal_(lora_a, mean=0.0, std=1.0 / math.sqrt(in_features))
    lora_b = torch.zeros((out_features, used_rank), device=device, dtype=torch.float32)
    lora_a.requires_grad_(True)
    lora_b.requires_grad_(True)
    return lora_a, lora_b, used_rank


def _sample_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size >= x.shape[0]:
        return x, y
    indices = torch.randint(0, x.shape[0], (batch_size,), device=x.device)
    return x.index_select(0, indices), y.index_select(0, indices)


def relex_rank1_extrapolate_delta(
    trajectory: list[torch.Tensor],
    *,
    target_step: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not trajectory:
        raise ValueError("RELEX trajectory must be non-empty")
    if len(trajectory) == 1:
        return trajectory[0].to(device=device, dtype=torch.float32), {
            "observed_steps": 1,
            "target_step": int(target_step),
            "r2": 1.0,
            "explained_variance": 1.0,
            "observed_final_coeff": float(trajectory[0].float().norm().item()),
            "extrapolated_coeff": float(trajectory[0].float().norm().item()),
            "coefficient_ratio": 1.0,
        }

    shape = trajectory[0].shape
    matrix = torch.stack([delta.reshape(-1).float() for delta in trajectory], dim=0).to(device=device)
    gram = matrix @ matrix.T
    evals, evecs = torch.linalg.eigh(gram)
    top_eval = torch.clamp(evals[-1], min=0.0)
    total_eval = torch.clamp(evals, min=0.0).sum()
    sigma = torch.sqrt(top_eval)
    if float(sigma.item()) <= 1e-12:
        zero = torch.zeros(shape, device=device, dtype=torch.float32)
        return zero, {
            "observed_steps": len(trajectory),
            "target_step": int(target_step),
            "r2": 0.0,
            "explained_variance": 0.0,
            "observed_final_coeff": 0.0,
            "extrapolated_coeff": 0.0,
            "coefficient_ratio": 0.0,
        }

    direction = (matrix.T @ evecs[:, -1]) / sigma
    coeff = matrix @ direction
    steps = torch.arange(1, len(trajectory) + 1, device=device, dtype=torch.float32)
    step_mean = steps.mean()
    coeff_mean = coeff.mean()
    slope = ((steps - step_mean) * (coeff - coeff_mean)).sum() / ((steps - step_mean).square().sum() + 1e-12)
    intercept = coeff_mean - slope * step_mean
    pred_coeff = slope * float(target_step) + intercept
    pred = (pred_coeff * direction).reshape(shape)

    fitted = slope * steps + intercept
    residual = (coeff - fitted).square().sum()
    centered = (coeff - coeff_mean).square().sum()
    r2 = 1.0 - float((residual / (centered + 1e-12)).item())
    observed_final_coeff = float(coeff[-1].item())
    extrapolated_coeff = float(pred_coeff.item())
    return pred, {
        "observed_steps": len(trajectory),
        "target_step": int(target_step),
        "r2": r2,
        "explained_variance": float((top_eval / (total_eval + 1e-12)).item()),
        "observed_final_coeff": observed_final_coeff,
        "extrapolated_coeff": extrapolated_coeff,
        "coefficient_ratio": extrapolated_coeff / observed_final_coeff if abs(observed_final_coeff) > 1e-12 else 0.0,
        "slope": float(slope.item()),
        "intercept": float(intercept.item()),
    }


def fit_fc2_lora_identity_repair(
    *,
    relu_model: torch.nn.Module,
    dense_records: dict[str, dict[str, torch.Tensor]],
    mlp_names: list[str],
    calibration_tokens: int,
    rank: int,
    alpha: float,
    steps: int,
    lr: float,
    batch_tokens: int,
    weight_decay: float,
    optimizer_name: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(29)
    summaries: list[dict[str, Any]] = []
    for name in tqdm(mlp_names, desc="fit fc2 LoRA identity repairs", unit="mlp"):
        mlp = relu_model.get_submodule(name)
        x_in, y, token_count = _select_dense_record_subset(
            dense_records,
            name,
            calibration_tokens=calibration_tokens,
            generator=generator,
            device=device,
        )
        with torch.no_grad():
            hidden = mlp.act(mlp.fc1(x_in)).float()
            weight = mlp.fc2.weight.detach().float()
            bias = mlp.fc2.bias.detach().float() if mlp.fc2.bias is not None else None
            base_y = F.linear(hidden, weight, bias)
            initial_mse = F.mse_loss(base_y, y).item()

        lora_a, lora_b, used_rank = _init_lora_factors(
            in_features=hidden.shape[1],
            out_features=y.shape[1],
            rank=rank,
            device=device,
        )
        scale = float(alpha) / float(used_rank)
        optimizer = make_lora_optimizer([lora_a, lora_b], name=optimizer_name, lr=lr, weight_decay=weight_decay)
        batch_size = min(batch_tokens, hidden.shape[0])
        losses: list[float] = []
        for step in range(steps):
            hidden_batch, y_batch = _sample_batch(hidden, y, batch_size=batch_size)
            pred = F.linear(hidden_batch, weight, bias)
            pred = pred + scale * F.linear(F.linear(hidden_batch, lora_a), lora_b)
            loss = F.mse_loss(pred, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step in {0, steps - 1}:
                losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            final = F.linear(hidden, weight, bias) + scale * F.linear(F.linear(hidden, lora_a), lora_b)
            final_mse = F.mse_loss(final, y).item()
            delta_weight = scale * (lora_b @ lora_a)
            mlp.fc2.weight.add_(delta_weight.to(dtype=mlp.fc2.weight.dtype))
            delta_norm = float(delta_weight.norm().item())
            weight_norm = float(weight.norm().item())

        summaries.append(
            {
                "module": name,
                "placement": "fc2_weight",
                "tokens_available": token_count,
                "tokens_used": int(x_in.shape[0]),
                "rank": int(used_rank),
                "alpha": float(alpha),
                "steps": int(steps),
                "lr": float(lr),
                "optimizer": optimizer_name,
                "initial_mse": initial_mse,
                "final_mse": final_mse,
                "first_last_batch_losses": losses,
                "delta_weight_norm": delta_norm,
                "base_weight_norm": weight_norm,
                "delta_to_weight_norm": delta_norm / max(weight_norm, 1e-12),
                "identity_initialization": "standard fc2 LoRA delta has B initialized to zero, so the initial fc2 function is unchanged.",
                "folded_delta": "fc2.weight <- fc2.weight + (alpha/rank) * B @ A",
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summaries


def fit_fc1_lora_identity_repair(
    *,
    relu_model: torch.nn.Module,
    dense_records: dict[str, dict[str, torch.Tensor]],
    mlp_names: list[str],
    calibration_tokens: int,
    rank: int,
    alpha: float,
    steps: int,
    lr: float,
    batch_tokens: int,
    weight_decay: float,
    optimizer_name: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(31)
    summaries: list[dict[str, Any]] = []
    for name in tqdm(mlp_names, desc="fit fc1 LoRA identity repairs", unit="mlp"):
        mlp = relu_model.get_submodule(name)
        x_in, y, token_count = _select_dense_record_subset(
            dense_records,
            name,
            calibration_tokens=calibration_tokens,
            generator=generator,
            device=device,
        )
        with torch.no_grad():
            fc1_weight = mlp.fc1.weight.detach().float()
            fc1_bias = mlp.fc1.bias.detach().float() if mlp.fc1.bias is not None else None
            fc2_weight = mlp.fc2.weight.detach().float()
            fc2_bias = mlp.fc2.bias.detach().float() if mlp.fc2.bias is not None else None
            base_hidden = mlp.act(F.linear(x_in, fc1_weight, fc1_bias)).float()
            base_y = F.linear(base_hidden, fc2_weight, fc2_bias)
            initial_mse = F.mse_loss(base_y, y).item()

        lora_a, lora_b, used_rank = _init_lora_factors(
            in_features=x_in.shape[1],
            out_features=fc1_weight.shape[0],
            rank=rank,
            device=device,
        )
        scale = float(alpha) / float(used_rank)
        optimizer = make_lora_optimizer([lora_a, lora_b], name=optimizer_name, lr=lr, weight_decay=weight_decay)
        batch_size = min(batch_tokens, x_in.shape[0])
        losses: list[float] = []
        for step in range(steps):
            x_batch, y_batch = _sample_batch(x_in, y, batch_size=batch_size)
            pre = F.linear(x_batch, fc1_weight, fc1_bias) + scale * F.linear(F.linear(x_batch, lora_a), lora_b)
            hidden = mlp.act(pre).float()
            pred = F.linear(hidden, fc2_weight, fc2_bias)
            loss = F.mse_loss(pred, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step in {0, steps - 1}:
                losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            pre = F.linear(x_in, fc1_weight, fc1_bias) + scale * F.linear(F.linear(x_in, lora_a), lora_b)
            final_y = F.linear(mlp.act(pre).float(), fc2_weight, fc2_bias)
            final_mse = F.mse_loss(final_y, y).item()
            delta_weight = scale * (lora_b @ lora_a)
            mlp.fc1.weight.add_(delta_weight.to(dtype=mlp.fc1.weight.dtype))
            delta_norm = float(delta_weight.norm().item())
            weight_norm = float(fc1_weight.norm().item())

        summaries.append(
            {
                "module": name,
                "placement": "fc1_weight",
                "tokens_available": token_count,
                "tokens_used": int(x_in.shape[0]),
                "rank": int(used_rank),
                "alpha": float(alpha),
                "steps": int(steps),
                "lr": float(lr),
                "optimizer": optimizer_name,
                "initial_mse": initial_mse,
                "final_mse": final_mse,
                "first_last_batch_losses": losses,
                "delta_weight_norm": delta_norm,
                "base_weight_norm": weight_norm,
                "delta_to_weight_norm": delta_norm / max(weight_norm, 1e-12),
                "identity_initialization": "fc1 LoRA delta has B initialized to zero, so the initial ReLU MLP is unchanged.",
                "folded_delta": "fc1.weight <- fc1.weight + (alpha/rank) * B @ A",
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summaries


class OutputLoRAAdapter(nn.Module):
    def __init__(self, base_mlp: nn.Module, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        self.register_buffer("lora_a", lora_a.detach().clone())
        self.register_buffer("lora_b", lora_b.detach().clone())
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_mlp(x)
        delta = self.scale * F.linear(F.linear(x.float(), self.lora_a.float()), self.lora_b.float())
        return base + delta.to(dtype=base.dtype)


def set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def fit_output_lora_identity_repair(
    *,
    relu_model: torch.nn.Module,
    dense_records: dict[str, dict[str, torch.Tensor]],
    mlp_names: list[str],
    calibration_tokens: int,
    rank: int,
    alpha: float,
    steps: int,
    lr: float,
    batch_tokens: int,
    weight_decay: float,
    optimizer_name: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(37)
    summaries: list[dict[str, Any]] = []
    for name in tqdm(mlp_names, desc="fit output LoRA identity repairs", unit="mlp"):
        mlp = relu_model.get_submodule(name)
        x_in, y, token_count = _select_dense_record_subset(
            dense_records,
            name,
            calibration_tokens=calibration_tokens,
            generator=generator,
            device=device,
        )
        with torch.no_grad():
            base_y = mlp(x_in).float()
            initial_mse = F.mse_loss(base_y, y).item()

        lora_a, lora_b, used_rank = _init_lora_factors(
            in_features=x_in.shape[1],
            out_features=y.shape[1],
            rank=rank,
            device=device,
        )
        scale = float(alpha) / float(used_rank)
        optimizer = make_lora_optimizer([lora_a, lora_b], name=optimizer_name, lr=lr, weight_decay=weight_decay)
        batch_size = min(batch_tokens, x_in.shape[0])
        losses: list[float] = []
        for step in range(steps):
            x_batch, y_batch = _sample_batch(x_in, y, batch_size=batch_size)
            pred = mlp(x_batch).float() + scale * F.linear(F.linear(x_batch, lora_a), lora_b)
            loss = F.mse_loss(pred, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step in {0, steps - 1}:
                losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            final_y = mlp(x_in).float() + scale * F.linear(F.linear(x_in, lora_a), lora_b)
            final_mse = F.mse_loss(final_y, y).item()
            delta_norm = float((scale * (lora_b @ lora_a)).norm().item())

        set_submodule(
            relu_model,
            name,
            OutputLoRAAdapter(
                mlp,
                lora_a.detach().to(dtype=torch.float32),
                lora_b.detach().to(dtype=torch.float32),
                scale,
            ),
        )
        summaries.append(
            {
                "module": name,
                "placement": "parallel_mlp_output_residual",
                "tokens_available": token_count,
                "tokens_used": int(x_in.shape[0]),
                "rank": int(used_rank),
                "alpha": float(alpha),
                "steps": int(steps),
                "lr": float(lr),
                "optimizer": optimizer_name,
                "initial_mse": initial_mse,
                "final_mse": final_mse,
                "first_last_batch_losses": losses,
                "delta_linear_norm": delta_norm,
                "identity_initialization": "output LoRA residual has B initialized to zero, so the initial MLP output is unchanged.",
                "architecture_change": "adds a parallel low-rank residual from MLP input to MLP output; this cannot be folded into the existing GELU/ReLU MLP weights exactly.",
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summaries


def fit_sandwich_lora_identity_repair(
    *,
    relu_model: torch.nn.Module,
    dense_records: dict[str, dict[str, torch.Tensor]],
    mlp_names: list[str],
    calibration_tokens: int,
    rank: int,
    alpha: float,
    steps: int,
    lr: float,
    batch_tokens: int,
    weight_decay: float,
    optimizer_name: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(41)
    summaries: list[dict[str, Any]] = []
    for name in tqdm(mlp_names, desc="fit sandwich LoRA identity repairs", unit="mlp"):
        mlp = relu_model.get_submodule(name)
        x_in, y, token_count = _select_dense_record_subset(
            dense_records,
            name,
            calibration_tokens=calibration_tokens,
            generator=generator,
            device=device,
        )
        with torch.no_grad():
            fc1_weight = mlp.fc1.weight.detach().float()
            fc1_bias = mlp.fc1.bias.detach().float() if mlp.fc1.bias is not None else None
            fc2_weight = mlp.fc2.weight.detach().float()
            fc2_bias = mlp.fc2.bias.detach().float() if mlp.fc2.bias is not None else None
            hidden = mlp.act(F.linear(x_in, fc1_weight, fc1_bias)).float()
            base_y = F.linear(hidden, fc2_weight, fc2_bias)
            initial_mse = F.mse_loss(base_y, y).item()

        pre_a, pre_b, pre_rank = _init_lora_factors(
            in_features=x_in.shape[1],
            out_features=fc1_weight.shape[0],
            rank=rank,
            device=device,
        )
        post_a, post_b, post_rank = _init_lora_factors(
            in_features=fc1_weight.shape[0],
            out_features=fc1_weight.shape[0],
            rank=rank,
            device=device,
        )
        pre_scale = float(alpha) / float(pre_rank)
        post_scale = float(alpha) / float(post_rank)
        optimizer = make_lora_optimizer(
            [pre_a, pre_b, post_a, post_b],
            name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
        )
        batch_size = min(batch_tokens, x_in.shape[0])
        losses: list[float] = []

        for step in range(steps):
            x_batch, y_batch = _sample_batch(x_in, y, batch_size=batch_size)
            pre = F.linear(x_batch, fc1_weight, fc1_bias)
            pre = pre + pre_scale * F.linear(F.linear(x_batch, pre_a), pre_b)
            h = mlp.act(pre).float()
            h = h + post_scale * F.linear(F.linear(h, post_a), post_b)
            pred = F.linear(h, fc2_weight, fc2_bias)
            loss = F.mse_loss(pred, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step in {0, steps - 1}:
                losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            pre = F.linear(x_in, fc1_weight, fc1_bias)
            pre = pre + pre_scale * F.linear(F.linear(x_in, pre_a), pre_b)
            h = mlp.act(pre).float()
            h = h + post_scale * F.linear(F.linear(h, post_a), post_b)
            final_y = F.linear(h, fc2_weight, fc2_bias)
            final_mse = F.mse_loss(final_y, y).item()

            delta_fc1 = pre_scale * (pre_b @ pre_a)
            delta_fc2 = post_scale * ((fc2_weight @ post_b) @ post_a)
            mlp.fc1.weight.add_(delta_fc1.to(dtype=mlp.fc1.weight.dtype))
            mlp.fc2.weight.add_(delta_fc2.to(dtype=mlp.fc2.weight.dtype))

            delta_fc1_norm = float(delta_fc1.norm().item())
            delta_fc2_norm = float(delta_fc2.norm().item())
            fc1_norm = float(fc1_weight.norm().item())
            fc2_norm = float(fc2_weight.norm().item())

        summaries.append(
            {
                "module": name,
                "placement": "pre_relu_fc1_and_post_relu_hidden",
                "tokens_available": token_count,
                "tokens_used": int(x_in.shape[0]),
                "pre_rank": int(pre_rank),
                "post_rank": int(post_rank),
                "alpha": float(alpha),
                "steps": int(steps),
                "lr": float(lr),
                "optimizer": optimizer_name,
                "initial_mse": initial_mse,
                "final_mse": final_mse,
                "first_last_batch_losses": losses,
                "delta_fc1_weight_norm": delta_fc1_norm,
                "base_fc1_weight_norm": fc1_norm,
                "delta_fc1_to_weight_norm": delta_fc1_norm / max(fc1_norm, 1e-12),
                "delta_fc2_weight_norm": delta_fc2_norm,
                "base_fc2_weight_norm": fc2_norm,
                "delta_fc2_to_weight_norm": delta_fc2_norm / max(fc2_norm, 1e-12),
                "identity_initialization": "both pre-ReLU and post-ReLU LoRA B factors are initialized to zero, so the initial MLP is exactly the plain ReLU MLP.",
                "folded_delta": "fc1.weight <- fc1.weight + pre_B @ pre_A * alpha/rank; fc2.weight <- fc2.weight + fc2.weight @ post_B @ post_A * alpha/rank",
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summaries


def fit_sandwich_lora_relex_repair(
    *,
    relu_model: torch.nn.Module,
    dense_records: dict[str, dict[str, torch.Tensor]],
    mlp_names: list[str],
    calibration_tokens: int,
    rank: int,
    alpha: float,
    steps: int,
    lr: float,
    batch_tokens: int,
    weight_decay: float,
    optimizer_name: str,
    target_step: int,
    relex_space: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(43)
    summaries: list[dict[str, Any]] = []
    for name in tqdm(mlp_names, desc="fit sandwich LoRA RELEX repairs", unit="mlp"):
        mlp = relu_model.get_submodule(name)
        x_in, y, token_count = _select_dense_record_subset(
            dense_records,
            name,
            calibration_tokens=calibration_tokens,
            generator=generator,
            device=device,
        )
        with torch.no_grad():
            fc1_weight = mlp.fc1.weight.detach().float()
            fc1_bias = mlp.fc1.bias.detach().float() if mlp.fc1.bias is not None else None
            fc2_weight = mlp.fc2.weight.detach().float()
            fc2_bias = mlp.fc2.bias.detach().float() if mlp.fc2.bias is not None else None
            hidden = mlp.act(F.linear(x_in, fc1_weight, fc1_bias)).float()
            base_y = F.linear(hidden, fc2_weight, fc2_bias)
            initial_mse = F.mse_loss(base_y, y).item()

        pre_a, pre_b, pre_rank = _init_lora_factors(
            in_features=x_in.shape[1],
            out_features=fc1_weight.shape[0],
            rank=rank,
            device=device,
        )
        post_a, post_b, post_rank = _init_lora_factors(
            in_features=fc1_weight.shape[0],
            out_features=fc1_weight.shape[0],
            rank=rank,
            device=device,
        )
        pre_scale = float(alpha) / float(pre_rank)
        post_scale = float(alpha) / float(post_rank)
        pre_a_init = pre_a.detach().clone()
        pre_b_init = pre_b.detach().clone()
        post_a_init = post_a.detach().clone()
        post_b_init = post_b.detach().clone()
        optimizer = make_lora_optimizer(
            [pre_a, pre_b, post_a, post_b],
            name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
        )
        batch_size = min(batch_tokens, x_in.shape[0])
        losses: list[float] = []
        delta_fc1_trajectory: list[torch.Tensor] = []
        delta_fc2_trajectory: list[torch.Tensor] = []
        pre_a_trajectory: list[torch.Tensor] = []
        pre_b_trajectory: list[torch.Tensor] = []
        post_a_trajectory: list[torch.Tensor] = []
        post_b_trajectory: list[torch.Tensor] = []

        for step in range(steps):
            x_batch, y_batch = _sample_batch(x_in, y, batch_size=batch_size)
            pre = F.linear(x_batch, fc1_weight, fc1_bias)
            pre = pre + pre_scale * F.linear(F.linear(x_batch, pre_a), pre_b)
            h = mlp.act(pre).float()
            h = h + post_scale * F.linear(F.linear(h, post_a), post_b)
            pred = F.linear(h, fc2_weight, fc2_bias)
            loss = F.mse_loss(pred, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if relex_space == "folded":
                    delta_fc1_now = pre_scale * (pre_b @ pre_a)
                    delta_fc2_now = post_scale * ((fc2_weight @ post_b) @ post_a)
                    delta_fc1_trajectory.append(delta_fc1_now.detach().cpu())
                    delta_fc2_trajectory.append(delta_fc2_now.detach().cpu())
                else:
                    pre_a_trajectory.append((pre_a - pre_a_init).detach().cpu())
                    pre_b_trajectory.append((pre_b - pre_b_init).detach().cpu())
                    post_a_trajectory.append((post_a - post_a_init).detach().cpu())
                    post_b_trajectory.append((post_b - post_b_init).detach().cpu())
            if step in {0, steps - 1}:
                losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            observed_delta_fc1 = pre_scale * (pre_b @ pre_a)
            observed_delta_fc2 = post_scale * ((fc2_weight @ post_b) @ post_a)
            observed_y = F.linear(
                mlp.act(F.linear(x_in, fc1_weight + observed_delta_fc1, fc1_bias)).float(),
                fc2_weight + observed_delta_fc2,
                fc2_bias,
            )
            observed_mse = F.mse_loss(observed_y, y).item()

            if relex_space == "folded":
                relex_delta_fc1, fc1_relex = relex_rank1_extrapolate_delta(
                    delta_fc1_trajectory,
                    target_step=target_step,
                    device=device,
                )
                relex_delta_fc2, fc2_relex = relex_rank1_extrapolate_delta(
                    delta_fc2_trajectory,
                    target_step=target_step,
                    device=device,
                )
                factor_relex: dict[str, Any] = {}
            else:
                pre_a_delta, pre_a_relex = relex_rank1_extrapolate_delta(
                    pre_a_trajectory,
                    target_step=target_step,
                    device=device,
                )
                pre_b_delta, pre_b_relex = relex_rank1_extrapolate_delta(
                    pre_b_trajectory,
                    target_step=target_step,
                    device=device,
                )
                post_a_delta, post_a_relex = relex_rank1_extrapolate_delta(
                    post_a_trajectory,
                    target_step=target_step,
                    device=device,
                )
                post_b_delta, post_b_relex = relex_rank1_extrapolate_delta(
                    post_b_trajectory,
                    target_step=target_step,
                    device=device,
                )
                pre_a_hat = pre_a_init + pre_a_delta
                pre_b_hat = pre_b_init + pre_b_delta
                post_a_hat = post_a_init + post_a_delta
                post_b_hat = post_b_init + post_b_delta
                relex_delta_fc1 = pre_scale * (pre_b_hat @ pre_a_hat)
                relex_delta_fc2 = post_scale * ((fc2_weight @ post_b_hat) @ post_a_hat)
                fc1_relex = {
                    "space": "factor_product",
                    "pre_a": pre_a_relex,
                    "pre_b": pre_b_relex,
                }
                fc2_relex = {
                    "space": "factor_product",
                    "post_a": post_a_relex,
                    "post_b": post_b_relex,
                }
                factor_relex = {
                    "pre_a": pre_a_relex,
                    "pre_b": pre_b_relex,
                    "post_a": post_a_relex,
                    "post_b": post_b_relex,
                }
            final_y = F.linear(
                mlp.act(F.linear(x_in, fc1_weight + relex_delta_fc1, fc1_bias)).float(),
                fc2_weight + relex_delta_fc2,
                fc2_bias,
            )
            final_mse = F.mse_loss(final_y, y).item()

            mlp.fc1.weight.add_(relex_delta_fc1.to(dtype=mlp.fc1.weight.dtype))
            mlp.fc2.weight.add_(relex_delta_fc2.to(dtype=mlp.fc2.weight.dtype))

            delta_fc1_norm = float(relex_delta_fc1.norm().item())
            delta_fc2_norm = float(relex_delta_fc2.norm().item())
            observed_fc1_norm = float(observed_delta_fc1.norm().item())
            observed_fc2_norm = float(observed_delta_fc2.norm().item())
            fc1_norm = float(fc1_weight.norm().item())
            fc2_norm = float(fc2_weight.norm().item())

        summaries.append(
            {
                "module": name,
                "placement": "pre_relu_fc1_and_post_relu_hidden_relex_folded_weights",
                "tokens_available": token_count,
                "tokens_used": int(x_in.shape[0]),
                "pre_rank": int(pre_rank),
                "post_rank": int(post_rank),
                "alpha": float(alpha),
                "relex_space": relex_space,
                "observed_steps": int(steps),
                "target_step": int(target_step),
                "lr": float(lr),
                "optimizer": optimizer_name,
                "initial_mse": initial_mse,
                "observed_mse": observed_mse,
                "final_mse": final_mse,
                "first_last_batch_losses": losses,
                "observed_delta_fc1_weight_norm": observed_fc1_norm,
                "observed_delta_fc2_weight_norm": observed_fc2_norm,
                "delta_fc1_weight_norm": delta_fc1_norm,
                "base_fc1_weight_norm": fc1_norm,
                "delta_fc1_to_weight_norm": delta_fc1_norm / max(fc1_norm, 1e-12),
                "delta_fc2_weight_norm": delta_fc2_norm,
                "base_fc2_weight_norm": fc2_norm,
                "delta_fc2_to_weight_norm": delta_fc2_norm / max(fc2_norm, 1e-12),
                "fc1_relex": fc1_relex,
                "fc2_relex": fc2_relex,
                "factor_relex": factor_relex,
                "identity_initialization": "both LoRA B factors start at zero; RELEX observes the folded weight-delta trajectory from this exact plain-ReLU initialization.",
                "folded_delta": "RELEX rank-1 extrapolates either folded fc1/fc2 weight deltas or trainable LoRA factor deltas, then folds the resulting effective deltas into the model.",
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summaries


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def normalized_fwht_rows(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"FWHT row count must be a power of two, got {n}")
    y = x.clone()
    h = 1
    while h < n:
        y = y.reshape(n // (2 * h), 2, h, *y.shape[1:])
        a = y[:, 0].clone()
        b = y[:, 1].clone()
        y[:, 0] = a + b
        y[:, 1] = a - b
        y = y.reshape(n, *x.shape[1:])
        h *= 2
    return y / math.sqrt(n)


def hadamard_rows_blockwise(x: torch.Tensor, *, block_size: int) -> tuple[torch.Tensor, list[dict[str, int]]]:
    dim = x.shape[0]
    out = torch.empty_like(x)
    chunks: list[dict[str, int]] = []
    if block_size <= 0:
        block_size = next_power_of_two(dim)
    start = 0
    while start < dim:
        size = min(block_size, dim - start)
        padded_size = next_power_of_two(size)
        chunk = x[start : start + size]
        if padded_size != size:
            pad_shape = (padded_size - size, *chunk.shape[1:])
            chunk = torch.cat([chunk, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=0)
        transformed = normalized_fwht_rows(chunk)[:size]
        out[start : start + size] = transformed
        chunks.append({"start": start, "size": size, "padded_size": padded_size})
        start += size
    return out, chunks


def apply_hadamard_hidden_basis(
    *,
    model: torch.nn.Module,
    mlp_names: list[str],
    block_size: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    with torch.no_grad():
        for name in mlp_names:
            mlp = model.get_submodule(name)
            fc1_weight, chunks = hadamard_rows_blockwise(mlp.fc1.weight.data.float(), block_size=block_size)
            mlp.fc1.weight.copy_(fc1_weight.to(dtype=mlp.fc1.weight.dtype))
            if mlp.fc1.bias is not None:
                bias, _chunks = hadamard_rows_blockwise(mlp.fc1.bias.data.float()[:, None], block_size=block_size)
                mlp.fc1.bias.copy_(bias[:, 0].to(dtype=mlp.fc1.bias.dtype))

            fc2_columns, _chunks = hadamard_rows_blockwise(mlp.fc2.weight.data.float().T, block_size=block_size)
            mlp.fc2.weight.copy_(fc2_columns.T.contiguous().to(dtype=mlp.fc2.weight.dtype))
            summaries.append(
                {
                    "module": name,
                    "hidden_features": int(mlp.fc1.out_features),
                    "chunks": chunks,
                    "rule": "fc1 rows and bias transformed by H; fc2 hidden columns transformed by H^T. H is normalized blockwise FWHT with padding only for non-power-of-two final blocks.",
                }
            )
    return summaries


def write_summary(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")


def run(config: ExperimentConfig) -> dict[str, Any]:
    torch.manual_seed(17)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
    if len(selected_items) < config.calibration_images:
        raise RuntimeError(
            f"selected {len(selected_items)} images, but calibration_images={config.calibration_images}"
        )

    result: dict[str, Any] = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_selected": len(selected_items),
            "pairs_selected": sum(len(pairs) for _path, pairs in selected_items),
            "rule": "DA-2K labels point1 as closer; Depth Anything V2 vits uses larger predicted values for closer points.",
        },
        "variants": {},
    }
    summary_path = config.output_dir / "summary.json"
    write_summary(summary_path, result)

    dense_model = load_model(config.encoder, config.checkpoint, device)
    for param in dense_model.parameters():
        param.requires_grad_(False)
    mlp_names = transformer_mlp_names(dense_model)
    result["metadata"]["transformer_mlp_count"] = len(mlp_names)
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
    write_summary(summary_path, result)

    if "dense" in config.modes:
        result["variants"]["dense"] = {
            "metadata": {"activation": "original GELU checkpoint"},
            "evaluation": evaluate_da2k_model(
                model=dense_model,
                dataset_root=config.dataset_root,
                items=selected_items,
                input_size=config.input_size,
                device=device,
                log_every=config.log_every,
            ),
        }
        write_summary(summary_path, result)

    relu_template = load_model(config.encoder, config.checkpoint, device)
    replaced = replace_gelu_with_relu(relu_template)
    for param in relu_template.parameters():
        param.requires_grad_(False)

    if "relu" in config.modes:
        result["variants"]["relu"] = {
            "metadata": {"activation": "all nn.GELU modules replaced with nn.ReLU", "replaced_modules": replaced},
            "evaluation": evaluate_da2k_model(
                model=relu_template,
                dataset_root=config.dataset_root,
                items=selected_items,
                input_size=config.input_size,
                device=device,
                log_every=config.log_every,
            ),
        }
        write_summary(summary_path, result)

    lora_mode_names = {
        "lora",
        "lora_hidden",
        "lora_fc2",
        "lora_fc1",
        "lora_output",
        "lora_sandwich",
        "lora_sandwich_relex",
    }
    requested_lora_modes = tuple(mode for mode in config.modes if mode in lora_mode_names)
    dense_records = None
    if "newton" in config.modes or requested_lora_modes:
        dense_records = collect_dense_mlp_calibration(
            dense_model=dense_model,
            mlp_names=mlp_names,
            calibration_tensors=calibration_tensors,
            device=device,
        )

    lora_specs = {
        "lora": (
            fit_hidden_lora_identity_repair,
            "hidden_identity_folded_fc2",
            "Low-rank identity-initialized hidden adapter repair for each transformer MLP. Adapter is h -> h + (alpha/rank) * B(Ah), with B initialized to zero so the starting function is exactly the plain ReLU model. The trained adapter is folded into fc2.weight for evaluation.",
        ),
        "lora_hidden": (
            fit_hidden_lora_identity_repair,
            "hidden_identity_folded_fc2",
            "Explicit alias for lora: hidden adapter h -> h + (alpha/rank) * B(Ah), folded into fc2.weight.",
        ),
        "lora_fc2": (
            fit_fc2_lora_identity_repair,
            "standard_fc2_lora",
            "Standard low-rank fc2 weight delta. B is initialized to zero so the starting function is exactly the plain ReLU model; the trained delta is folded into fc2.weight.",
        ),
        "lora_fc1": (
            fit_fc1_lora_identity_repair,
            "preactivation_fc1_lora",
            "Low-rank fc1 pre-activation weight delta trained through the ReLU nonlinearity. B is initialized to zero so the starting function is exactly the plain ReLU model; the trained delta is folded into fc1.weight.",
        ),
        "lora_output": (
            fit_output_lora_identity_repair,
            "parallel_mlp_output_lora",
            "Parallel low-rank MLP-output residual from MLP input to MLP output. B is initialized to zero so the starting function is exactly the plain ReLU model; this variant remains as an explicit adapter module for evaluation.",
        ),
        "lora_sandwich": (
            fit_sandwich_lora_identity_repair,
            "pre_relu_fc1_and_post_relu_hidden",
            "Joint before/after ReLU low-rank repair. A pre-ReLU low-rank delta is folded into fc1.weight, and a post-ReLU hidden adapter is folded into fc2.weight. Both B factors are initialized to zero, so the starting function is exactly the plain ReLU model.",
        ),
        "lora_sandwich_relex": (
            fit_sandwich_lora_relex_repair,
            "pre_relu_fc1_and_post_relu_hidden_relex_folded_weights",
            "RELEX-style rank-1 trajectory extrapolation for the sandwich LoRA repair. It observes the folded fc1/fc2 weight-delta trajectory for lora_steps optimizer updates, fits a rank-1 coefficient line per folded matrix, extrapolates to lora_relex_target_step, and folds the predicted deltas.",
        ),
    }
    for mode in requested_lora_modes:
        if dense_records is None:
            raise RuntimeError("dense MLP calibration records were not collected")
        fit_fn, placement, approximation = lora_specs[mode]
        lora_model = copy.deepcopy(relu_template).to(device=device).eval()
        fit_kwargs: dict[str, Any] = {
            "relu_model": lora_model,
            "dense_records": dense_records,
            "mlp_names": mlp_names,
            "calibration_tokens": config.calibration_tokens,
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "steps": config.lora_steps,
            "lr": config.lora_lr,
            "batch_tokens": config.lora_batch_tokens,
            "weight_decay": config.lora_weight_decay,
            "optimizer_name": config.lora_optimizer,
            "device": device,
        }
        if mode == "lora_sandwich_relex":
            fit_kwargs["target_step"] = config.lora_relex_target_step
            fit_kwargs["relex_space"] = config.lora_relex_space
        repair = fit_fn(**fit_kwargs)
        result["variants"][mode] = {
            "metadata": {
                "activation": "GELU replaced with ReLU",
                "placement": placement,
                "approximation": approximation,
                "rank": config.lora_rank,
                "alpha": config.lora_alpha,
                "steps": config.lora_steps,
                "lr": config.lora_lr,
                "optimizer": config.lora_optimizer,
                "batch_tokens": config.lora_batch_tokens,
                "weight_decay": config.lora_weight_decay,
                "relex_target_step": config.lora_relex_target_step if mode == "lora_sandwich_relex" else None,
                "relex_space": config.lora_relex_space if mode == "lora_sandwich_relex" else None,
                "calibration_tokens_per_mlp": config.calibration_tokens,
                "replaced_modules": replaced,
                "repair": repair,
            },
            "evaluation": evaluate_da2k_model(
                model=lora_model,
                dataset_root=config.dataset_root,
                items=selected_items,
                input_size=config.input_size,
                device=device,
                log_every=config.log_every,
            ),
        }
        write_summary(summary_path, result)
        del lora_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "newton" in config.modes:
        if dense_records is None:
            raise RuntimeError("dense MLP calibration records were not collected")
        newton_model = copy.deepcopy(relu_template).to(device=device).eval()
        repair = fit_fc2_least_squares_repair(
            relu_model=newton_model,
            dense_records=dense_records,
            mlp_names=mlp_names,
            calibration_tokens=config.calibration_tokens,
            ridge_lambda=config.ridge_lambda,
            device=device,
        )
        result["variants"]["newton"] = {
            "metadata": {
                "activation": "GELU replaced with ReLU",
                "approximation": "Damped Gauss-Newton / ridge least-squares closed-form repair of each transformer MLP fc2 weight and bias, matching original GELU MLP outputs on calibration tokens.",
                "ridge_lambda": config.ridge_lambda,
                "calibration_tokens_per_mlp": config.calibration_tokens,
                "replaced_modules": replaced,
                "repair": repair,
            },
            "evaluation": evaluate_da2k_model(
                model=newton_model,
                dataset_root=config.dataset_root,
                items=selected_items,
                input_size=config.input_size,
                device=device,
                log_every=config.log_every,
            ),
        }
        write_summary(summary_path, result)
        del newton_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "hadamard" in config.modes:
        hadamard_model = copy.deepcopy(relu_template).to(device=device).eval()
        hadamard = apply_hadamard_hidden_basis(
            model=hadamard_model,
            mlp_names=mlp_names,
            block_size=config.hadamard_block_size,
        )
        result["variants"]["hadamard"] = {
            "metadata": {
                "activation": "GELU replaced with ReLU",
                "compensation": "Orthogonal hidden-channel basis transform around each transformer MLP fc1/fc2 pair before evaluation.",
                "hadamard_block_size": config.hadamard_block_size,
                "replaced_modules": replaced,
                "hadamard": hadamard,
            },
            "evaluation": evaluate_da2k_model(
                model=hadamard_model,
                dataset_root=config.dataset_root,
                items=selected_items,
                input_size=config.input_size,
                device=device,
                log_every=config.log_every,
            ),
        }
        write_summary(summary_path, result)

    result["metadata"]["elapsed_seconds"] = time.monotonic() - started
    write_summary(summary_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Depth Anything V2 GELU to ReLU activation-swap compensation on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/da2k_vits_gelu_relu_compensation"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-images", type=int, default=8)
    parser.add_argument("--calibration-tokens", type=int, default=4096)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--modes", default="dense,relu,newton,hadamard,lora")
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--hadamard-block-size", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-steps", type=int, default=200)
    parser.add_argument("--lora-lr", type=float, default=3e-3)
    parser.add_argument("--lora-batch-tokens", type=int, default=2048)
    parser.add_argument("--lora-weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-optimizer", choices=sorted(LORA_OPTIMIZERS), default="adamw")
    parser.add_argument("--lora-relex-target-step", type=int, default=200)
    parser.add_argument("--lora-relex-space", choices=["folded", "factor"], default="folded")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
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
        modes=parse_modes(args.modes),
        scene_type=args.scene_type,
        log_every=args.log_every,
        ridge_lambda=args.ridge_lambda,
        hadamard_block_size=args.hadamard_block_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_steps=args.lora_steps,
        lora_lr=args.lora_lr,
        lora_batch_tokens=args.lora_batch_tokens,
        lora_weight_decay=args.lora_weight_decay,
        lora_optimizer=args.lora_optimizer,
        lora_relex_target_step=args.lora_relex_target_step,
        lora_relex_space=args.lora_relex_space,
    )
    summary = run(config)
    print(json.dumps({name: row["evaluation"]["overall"] for name, row in summary["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
