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

from eval_da2k import MODEL_CONFIGS, resolve_device
from eval_gelu_relu_compensation_da2k import (
    _sample_batch,
    _select_dense_record_subset,
    collect_dense_mlp_calibration,
    evaluate_da2k_model,
    load_calibration_tensors,
    load_model,
    selected_annotations,
    transformer_mlp_names,
    write_summary,
)


DEFAULT_VARIANTS: tuple[str, ...] = (
    "identity:relu",
    "identity:leaky0_125",
    "identity:fit2_t0",
    "identity:fit2_tm0_5",
    "pca:fit2_t0",
    "pca_signed:fit2_t0",
    "random:fit2_t0",
    "learned_identity:relu",
    "learned_identity:fit2_t0",
    "learned_pca:fit2_t0",
)


@dataclass(frozen=True)
class TwoPieceSpec:
    name: str
    threshold: float
    left_slope: float
    left_intercept: float
    right_slope: float
    right_intercept: float


class TwoPieceActivation(nn.Module):
    def __init__(self, spec: TwoPieceSpec) -> None:
        super().__init__()
        self.spec = spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        threshold = torch.as_tensor(self.spec.threshold, device=x.device, dtype=x.dtype)
        left = self.spec.left_slope * x + self.spec.left_intercept
        right = self.spec.right_slope * x + self.spec.right_intercept
        return torch.where(x <= threshold, left, right)

    def extra_repr(self) -> str:
        return (
            f"name={self.spec.name}, threshold={self.spec.threshold}, "
            f"left=({self.spec.left_slope}, {self.spec.left_intercept}), "
            f"right=({self.spec.right_slope}, {self.spec.right_intercept})"
        )


def parse_variants(value: str) -> tuple[str, ...]:
    variants = tuple(part.strip() for part in value.split(",") if part.strip())
    return variants or DEFAULT_VARIANTS


def sanitize_name(value: str) -> str:
    out = []
    for ch in value:
        out.append(ch if ch.isalnum() or ch in {"_", "-"} else "_")
    return "".join(out)


def parse_threshold(text: str) -> float:
    text = text.replace("m", "-").replace("p", "+").replace("_", ".")
    return float(text)


def weighted_affine_fit(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> tuple[float, float]:
    x = x.double()
    y = y.double()
    weight = weight.double()
    design = torch.stack([x, torch.ones_like(x)], dim=1)
    weighted_design = design * weight.sqrt().unsqueeze(1)
    weighted_y = y * weight.sqrt()
    gram = weighted_design.T @ weighted_design
    rhs = weighted_design.T @ weighted_y
    beta = torch.linalg.solve(gram + 1e-10 * torch.eye(2, dtype=torch.float64), rhs)
    return float(beta[0].item()), float(beta[1].item())


def fitted_two_piece_gelu(threshold: float) -> TwoPieceSpec:
    x = torch.linspace(-6.0, 6.0, 6001, dtype=torch.float64)
    y = F.gelu(x)
    # Standard-normal weighting roughly matches normalized transformer pre-activations
    # without requiring a data pass just to define the cheap activation shape.
    weight = torch.exp(-0.5 * x.square()).clamp_min(1e-12)
    left_mask = x <= threshold
    right_mask = ~left_mask
    left_slope, left_intercept = weighted_affine_fit(x[left_mask], y[left_mask], weight[left_mask])
    right_slope, right_intercept = weighted_affine_fit(x[right_mask], y[right_mask], weight[right_mask])
    return TwoPieceSpec(
        name=f"fit2_t{threshold:g}",
        threshold=threshold,
        left_slope=left_slope,
        left_intercept=left_intercept,
        right_slope=right_slope,
        right_intercept=right_intercept,
    )


def two_piece_spec(name: str) -> TwoPieceSpec:
    key = name.lower()
    if key == "relu":
        return TwoPieceSpec("relu", 0.0, 0.0, 0.0, 1.0, 0.0)
    if key.startswith("leaky"):
        slope = parse_threshold(key.removeprefix("leaky"))
        return TwoPieceSpec(name, 0.0, slope, 0.0, 1.0, 0.0)
    if key.startswith("fit2_t"):
        threshold = parse_threshold(key.removeprefix("fit2_t"))
        return fitted_two_piece_gelu(threshold)
    raise ValueError(f"unknown two-piece activation: {name}")


def parse_variant(variant: str) -> tuple[str, str]:
    if ":" not in variant:
        return variant, "fit2_t0"
    rotation, activation = variant.split(":", 1)
    return rotation.strip().lower(), activation.strip().lower()


def ensure_block_size(hidden_size: int, block_size: int) -> int:
    if block_size <= 0:
        return hidden_size
    if hidden_size % block_size != 0:
        raise ValueError(f"hidden_size={hidden_size} must be divisible by block_size={block_size}")
    return block_size


def apply_block_rotation(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    block_count, block_size, _ = rotation.shape
    x_blocks = x.reshape(x.shape[0], block_count, block_size)
    y = torch.einsum("tni,nji->tnj", x_blocks, rotation)
    return y.reshape_as(x)


def fold_rotation_into_fc1(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    block_count, block_size, _ = rotation.shape
    weight_blocks = weight.reshape(block_count, block_size, weight.shape[1])
    new_weight = torch.einsum("nji,nik->njk", rotation, weight_blocks).reshape_as(weight)
    if bias is None:
        return new_weight.contiguous(), None
    bias_blocks = bias.reshape(block_count, block_size)
    new_bias = torch.einsum("nji,ni->nj", rotation, bias_blocks).reshape_as(bias)
    return new_weight.contiguous(), new_bias.contiguous()


def nearest_orthogonal_blocks(rotation: torch.Tensor) -> torch.Tensor:
    projected = []
    for block in rotation:
        u, _s, vh = torch.linalg.svd(block.float(), full_matrices=False)
        projected.append((u @ vh).to(dtype=rotation.dtype))
    return torch.stack(projected, dim=0)


def random_orthogonal_blocks(
    *,
    block_count: int,
    block_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    blocks = []
    for _ in range(block_count):
        matrix = torch.randn((block_size, block_size), generator=generator, dtype=torch.float32)
        q, r = torch.linalg.qr(matrix)
        q = q * r.diagonal().sign().clamp(min=-1.0, max=1.0).unsqueeze(0)
        blocks.append(q)
    return torch.stack(blocks, dim=0).to(device=device)


def pca_blocks(z: torch.Tensor, *, block_size: int, signed: bool) -> torch.Tensor:
    block_count = z.shape[1] // block_size
    blocks = []
    for block_index in range(block_count):
        chunk = z[:, block_index * block_size : (block_index + 1) * block_size].float()
        centered = chunk - chunk.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
        _evals, evecs = torch.linalg.eigh(cov)
        rotation = evecs.flip(dims=(1,)).T.contiguous()
        if signed:
            rotated = chunk @ rotation.T
            sign = torch.where(rotated.mean(dim=0) < 0, -torch.ones(block_size, device=z.device), torch.ones(block_size, device=z.device))
            rotation = rotation * sign.unsqueeze(1)
        blocks.append(rotation)
    return torch.stack(blocks, dim=0).to(device=z.device)


def initial_rotation(
    *,
    mode: str,
    z: torch.Tensor,
    block_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    block_count = z.shape[1] // block_size
    eye = torch.eye(block_size, device=device, dtype=torch.float32).expand(block_count, -1, -1).clone()
    if mode in {"identity", "learned", "learned_identity"}:
        return eye
    if mode in {"random", "learned_random"}:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return random_orthogonal_blocks(
            block_count=block_count,
            block_size=block_size,
            generator=generator,
            device=device,
        )
    if mode in {"pca", "learned_pca"}:
        return pca_blocks(z, block_size=block_size, signed=False)
    if mode in {"pca_signed", "learned_pca_signed"}:
        return pca_blocks(z, block_size=block_size, signed=True)
    raise ValueError(f"unknown rotation mode: {mode}")


def solve_output_linear(
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    ones = torch.ones((features.shape[0], 1), device=features.device, dtype=features.dtype)
    design = torch.cat([features, ones], dim=1)
    gram = design.T @ design
    if ridge_lambda > 0.0:
        gram.diagonal().add_(ridge_lambda)
        gram[-1, -1].sub_(ridge_lambda)
    rhs = design.T @ target
    beta = torch.linalg.solve(gram, rhs)
    weight = beta[:-1].T.contiguous()
    bias = beta[-1].contiguous()
    pred = F.linear(features, weight, bias)
    mse = F.mse_loss(pred, target).item()
    return weight, bias, mse


def learned_rotation_fit(
    *,
    z: torch.Tensor,
    target: torch.Tensor,
    init_rotation: torch.Tensor,
    activation: TwoPieceActivation,
    ridge_lambda: float,
    steps: int,
    lr: float,
    batch_tokens: int,
    orth_lambda: float,
    drift_lambda: float,
    project: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        init_features = activation(apply_block_rotation(z, init_rotation))
        init_weight, init_bias, init_mse = solve_output_linear(init_features, target, ridge_lambda=ridge_lambda)

    rotation = nn.Parameter(init_rotation.detach().clone())
    output_weight = nn.Parameter(init_weight.detach().clone())
    output_bias = nn.Parameter(init_bias.detach().clone())
    optimizer = torch.optim.AdamW([rotation, output_weight, output_bias], lr=lr)
    batch_size = min(batch_tokens, z.shape[0])
    eye = torch.eye(rotation.shape[-1], device=z.device, dtype=torch.float32).expand(rotation.shape[0], -1, -1)
    losses: list[float] = []
    ortho_losses: list[float] = []

    for step in range(steps):
        z_batch, target_batch = _sample_batch(z, target, batch_size=batch_size)
        features = activation(apply_block_rotation(z_batch, rotation))
        pred = F.linear(features, output_weight, output_bias)
        mse = F.mse_loss(pred, target_batch)
        gram = rotation.float() @ rotation.float().transpose(-1, -2)
        orth_loss = (gram - eye).square().mean()
        drift_loss = (rotation - init_rotation).float().square().mean()
        loss = mse + orth_lambda * orth_loss + drift_lambda * drift_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {0, steps - 1}:
            losses.append(float(mse.detach().cpu()))
            ortho_losses.append(float(orth_loss.detach().cpu()))

    final_rotation = rotation.detach()
    if project:
        final_rotation = nearest_orthogonal_blocks(final_rotation)
    with torch.no_grad():
        final_features = activation(apply_block_rotation(z, final_rotation))
        _weight, _bias, final_mse = solve_output_linear(final_features, target, ridge_lambda=ridge_lambda)
        gram = final_rotation.float() @ final_rotation.float().transpose(-1, -2)
        final_orth_error = (gram - eye).square().mean().item()
    return final_rotation.detach(), {
        "init_ls_mse": init_mse,
        "final_ls_mse": final_mse,
        "first_last_batch_mse": losses,
        "first_last_orth_loss": ortho_losses,
        "final_orth_error": final_orth_error,
        "projected_to_nearest_orthogonal": project,
    }


def fit_rotated_twopiece_mlp(
    *,
    relu_model: nn.Module,
    dense_records: dict[str, dict[str, torch.Tensor]],
    mlp_names: list[str],
    rotation_mode: str,
    activation_spec: TwoPieceSpec,
    calibration_tokens: int,
    block_size: int,
    ridge_lambda: float,
    learned_steps: int,
    learned_lr: float,
    batch_tokens: int,
    orth_lambda: float,
    drift_lambda: float,
    project_learned: bool,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    summaries: list[dict[str, Any]] = []
    activation = TwoPieceActivation(activation_spec).to(device=device)
    is_learned = rotation_mode.startswith("learned")

    for layer_index, name in enumerate(tqdm(mlp_names, desc=f"fit {rotation_mode}:{activation_spec.name}", unit="mlp")):
        mlp = relu_model.get_submodule(name)
        x_in, target, token_count = _select_dense_record_subset(
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
        hidden_size = fc1_weight.shape[0]
        actual_block_size = ensure_block_size(hidden_size, block_size)
        z = F.linear(x_in, fc1_weight, fc1_bias)
        with torch.no_grad():
            old_output = F.linear(activation(z), fc2_weight, fc2_bias)
            initial_mse = F.mse_loss(old_output, target).item()

        rotation = initial_rotation(
            mode=rotation_mode,
            z=z,
            block_size=actual_block_size,
            seed=seed + 997 * layer_index,
            device=device,
        )
        learned_summary: dict[str, Any] = {}
        if is_learned and learned_steps > 0:
            rotation, learned_summary = learned_rotation_fit(
                z=z,
                target=target,
                init_rotation=rotation,
                activation=activation,
                ridge_lambda=ridge_lambda,
                steps=learned_steps,
                lr=learned_lr,
                batch_tokens=batch_tokens,
                orth_lambda=orth_lambda,
                drift_lambda=drift_lambda,
                project=project_learned,
            )
        elif is_learned and project_learned:
            rotation = nearest_orthogonal_blocks(rotation)

        with torch.no_grad():
            features = activation(apply_block_rotation(z, rotation))
            new_fc2_weight, new_fc2_bias, final_mse = solve_output_linear(
                features,
                target,
                ridge_lambda=ridge_lambda,
            )
            new_fc1_weight, new_fc1_bias = fold_rotation_into_fc1(fc1_weight, fc1_bias, rotation)
            mlp.fc1.weight.copy_(new_fc1_weight.to(dtype=mlp.fc1.weight.dtype))
            if mlp.fc1.bias is not None and new_fc1_bias is not None:
                mlp.fc1.bias.copy_(new_fc1_bias.to(dtype=mlp.fc1.bias.dtype))
            mlp.fc2.weight.copy_(new_fc2_weight.to(dtype=mlp.fc2.weight.dtype))
            if mlp.fc2.bias is not None:
                mlp.fc2.bias.copy_(new_fc2_bias.to(dtype=mlp.fc2.bias.dtype))
            mlp.act = copy.deepcopy(activation).to(device=device)
            gram = rotation.float() @ rotation.float().transpose(-1, -2)
            eye = torch.eye(actual_block_size, device=device, dtype=torch.float32).expand(rotation.shape[0], -1, -1)
            orth_error = (gram - eye).square().mean().item()

        summaries.append(
            {
                "module": name,
                "rotation_mode": rotation_mode,
                "activation": asdict(activation_spec),
                "tokens_available": token_count,
                "tokens_used": int(x_in.shape[0]),
                "hidden_size": int(hidden_size),
                "block_size": int(actual_block_size),
                "block_count": int(rotation.shape[0]),
                "initial_mse_with_old_fc2": initial_mse,
                "final_ls_mse": final_mse,
                "orthogonality_mse": orth_error,
                "learned": learned_summary,
                "folding": "fc1.weight <- R @ fc1.weight, fc1.bias <- R @ fc1.bias, fc2 solved by ridge LS against dense GELU MLP outputs.",
            }
        )
        del x_in, target, z, old_output, rotation, features
        del fc1_weight, fc2_weight, new_fc1_weight, new_fc2_weight, new_fc2_bias
        if fc1_bias is not None:
            del fc1_bias
        if fc2_bias is not None:
            del fc2_bias
        if new_fc1_bias is not None:
            del new_fc1_bias
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    return summaries


@dataclass(frozen=True)
class RotationSweepConfig:
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
    scene_type: str = ""
    variants: tuple[str, ...] = DEFAULT_VARIANTS
    block_size: int = 128
    ridge_lambda: float = 1e-3
    learned_steps: int = 50
    learned_lr: float = 1e-3
    batch_tokens: int = 2048
    orth_lambda: float = 1e-2
    drift_lambda: float = 1e-4
    project_learned: bool = True
    save_models: bool = False
    skip_variant_eval: bool = False
    seed: int = 61
    log_every: int = 16

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
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be non-negative")
        if self.learned_steps < 0:
            raise ValueError("learned_steps must be non-negative")
        if self.learned_lr <= 0.0:
            raise ValueError("learned_lr must be positive")
        if self.batch_tokens <= 0:
            raise ValueError("batch_tokens must be positive")
        if self.orth_lambda < 0.0:
            raise ValueError("orth_lambda must be non-negative")
        if self.drift_lambda < 0.0:
            raise ValueError("drift_lambda must be non-negative")
        if not self.variants:
            raise ValueError("at least one variant is required")


def run(config: RotationSweepConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
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
    if len(selected_items) < config.calibration_images:
        raise RuntimeError(f"selected {len(selected_items)} images, but calibration_images={config.calibration_images}")

    result: dict[str, Any] = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_selected": len(selected_items),
            "pairs_selected": sum(len(pairs) for _path, pairs in selected_items),
            "note": (
                "Layerwise folded rotation experiment: z'=R z, cheap two-piece q(z'), "
                "fc2 solved by ridge least squares against dense GELU MLP outputs. R is folded into fc1."
            ),
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
    calibration_tensors = [tensor.detach().cpu() for tensor in calibration_tensors]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result["metadata"]["calibration_relative_paths"] = calibration_paths
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dense_records = collect_dense_mlp_calibration(
        dense_model=dense_model,
        mlp_names=mlp_names,
        calibration_tensors=calibration_tensors,
        device=device,
    )
    del calibration_tensors
    del dense_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for variant in config.variants:
        rotation_mode, activation_name = parse_variant(variant)
        activation_spec = two_piece_spec(activation_name)
        model = load_model(config.encoder, config.checkpoint, device)
        for param in model.parameters():
            param.requires_grad_(False)
        repair = fit_rotated_twopiece_mlp(
            relu_model=model,
            dense_records=dense_records,
            mlp_names=mlp_names,
            rotation_mode=rotation_mode,
            activation_spec=activation_spec,
            calibration_tokens=config.calibration_tokens,
            block_size=config.block_size,
            ridge_lambda=config.ridge_lambda,
            learned_steps=config.learned_steps,
            learned_lr=config.learned_lr,
            batch_tokens=config.batch_tokens,
            orth_lambda=config.orth_lambda,
            drift_lambda=config.drift_lambda,
            project_learned=config.project_learned,
            seed=config.seed,
            device=device,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        key = sanitize_name(variant)
        saved_model_path = None
        if config.save_models:
            saved_model_path = config.output_dir / f"{key}.state_dict.pt"
            saved_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, saved_model_path)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        if config.skip_variant_eval:
            evaluation = {
                "metadata": {
                    "images_requested": len(selected_items),
                    "skipped": True,
                    "reason": "skip_variant_eval was set; evaluate the saved folded state dict in a fresh process.",
                },
                "overall": {},
                "by_scene": {},
            }
        else:
            evaluation = evaluate_da2k_model(
                model=model,
                dataset_root=config.dataset_root,
                items=selected_items,
                input_size=config.input_size,
                device=device,
                log_every=config.log_every,
            )
        result["variants"][key] = {
            "metadata": {
                "variant": variant,
                "rotation_mode": rotation_mode,
                "activation": asdict(activation_spec),
                "block_size": config.block_size,
                "ridge_lambda": config.ridge_lambda,
                "learned_steps": config.learned_steps if rotation_mode.startswith("learned") else 0,
                "learned_lr": config.learned_lr if rotation_mode.startswith("learned") else None,
                "orth_lambda": config.orth_lambda if rotation_mode.startswith("learned") else None,
                "drift_lambda": config.drift_lambda if rotation_mode.startswith("learned") else None,
                "project_learned": config.project_learned if rotation_mode.startswith("learned") else None,
                "saved_model_path": str(saved_model_path) if saved_model_path is not None else None,
                "repair": repair,
            },
            "evaluation": evaluation,
        }
        write_summary(summary_path, result)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result["metadata"]["elapsed_seconds"] = time.monotonic() - started
    write_summary(summary_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Folded rotation + two-piece GELU replacement sweep for Depth Anything V2.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/rotated_twopiece_gelu"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-images", type=int, default=8)
    parser.add_argument("--calibration-tokens", type=int, default=4096)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--learned-steps", type=int, default=50)
    parser.add_argument("--learned-lr", type=float, default=1e-3)
    parser.add_argument("--batch-tokens", type=int, default=2048)
    parser.add_argument("--orth-lambda", type=float, default=1e-2)
    parser.add_argument("--drift-lambda", type=float, default=1e-4)
    parser.add_argument("--no-project-learned", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--skip-variant-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--log-every", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = RotationSweepConfig(
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
        variants=parse_variants(args.variants),
        block_size=args.block_size,
        ridge_lambda=args.ridge_lambda,
        learned_steps=args.learned_steps,
        learned_lr=args.learned_lr,
        batch_tokens=args.batch_tokens,
        orth_lambda=args.orth_lambda,
        drift_lambda=args.drift_lambda,
        project_learned=not args.no_project_learned,
        save_models=args.save_models,
        skip_variant_eval=args.skip_variant_eval,
        seed=args.seed,
        log_every=args.log_every,
    )
    summary = run(config)
    print(json.dumps({name: row["evaluation"]["overall"] for name, row in summary["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
