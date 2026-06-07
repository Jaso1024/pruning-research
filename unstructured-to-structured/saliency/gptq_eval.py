from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from saliency.calibration import batched, build_causal_lm_batch
from saliency.experiment import resolve_device, resolve_torch_dtype
from saliency.prune_eval import PerplexityStats, evaluate_perplexity, summarize_ppl_change


FP8_E4M3_MAX = 448.0


@dataclass(slots=True)
class GPTQConfig:
    output_dir: str | Path
    model_name: str = "EleutherAI/pythia-31m"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    calibration_split: str = "train"
    eval_split: str = "test"
    max_calibration_examples: int = 0
    max_eval_examples: int = 0
    calibration_batch_size: int = 32
    eval_batch_size: int = 32
    max_length: int = 512
    dtype: str = "fp32"
    device: str = "auto"
    answer_only_loss: bool = True
    damp_percent: float = 0.01
    blocksize: int = 128
    gptq_steps: int = 1
    eval_steps: tuple[int, ...] | None = None
    staged_to_wq: bool = False
    iterative_damped_gptq: bool = False
    gradient_descent_gptq: bool = False
    newton_step_alpha: float | None = None
    gradient_step_scale: float = 1.0
    gradient_step_scales: tuple[float, ...] | None = None
    hessian_approximation: str = "full"
    revision: str | None = None


def quantize_fp8_e4m3_per_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    source = weight.detach().float()
    scale = source.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_E4M3_MAX
    scaled = (source / scale).clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    quantized = scaled.to(torch.float8_e4m3fn).to(torch.float32) * scale
    return quantized.to(dtype=weight.dtype), scale


def _quantize_fp8_e4m3_with_scale(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scaled = (values.float() / scale).clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    return scaled.to(torch.float8_e4m3fn).to(torch.float32) * scale


def _invert_hessian(hessian: torch.Tensor, damp_percent: float) -> tuple[torch.Tensor, float]:
    h = hessian.double().cpu()
    diag = torch.diag(h)
    mean_diag = float(diag[diag > 0].mean().item()) if torch.any(diag > 0) else 1.0
    damp = max(mean_diag * damp_percent, 1e-8)
    eye = torch.eye(h.shape[0], dtype=h.dtype)
    for multiplier in (1.0, 10.0, 100.0, 1000.0):
        try:
            chol = torch.linalg.cholesky(h + eye * (damp * multiplier))
            return torch.cholesky_inverse(chol), damp * multiplier
        except torch.linalg.LinAlgError:
            continue
    return torch.linalg.pinv(h + eye * (damp * 1000.0)), damp * 1000.0


def _diagonal_hessian_damp(hessian_diag: torch.Tensor, damp_percent: float) -> float:
    diag = hessian_diag.double().cpu()
    mean_diag = float(diag[diag > 0].mean().item()) if torch.any(diag > 0) else 1.0
    return max(mean_diag * damp_percent, 1e-8)


def gptq_quantize_weight(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    damp_percent: float = 0.01,
    blocksize: int = 128,
) -> tuple[torch.Tensor, dict[str, object]]:
    if weight.ndim != 2:
        raise ValueError("GPTQ weight must be a 2D matrix")

    original = weight.detach().float().cpu()
    columns = original.shape[1]
    if hessian.ndim == 1:
        if hessian.shape != (columns,):
            raise ValueError(f"diagonal hessian shape {tuple(hessian.shape)} does not match input columns {columns}")
        quantized, _ = quantize_fp8_e4m3_per_row(original)
        diff = original - quantized
        return quantized.to(dtype=weight.dtype), {
            "columns": columns,
            "weights": original.numel(),
            "damp": _diagonal_hessian_damp(hessian, damp_percent),
            "blocksize": blocksize,
            "hessian_approximation": "diagonal",
            "mean_abs_error": float(diff.abs().mean().item()),
            "max_abs_error": float(diff.abs().max().item()),
        }

    if hessian.shape != (columns, columns):
        raise ValueError(f"hessian shape {tuple(hessian.shape)} does not match input columns {columns}")

    h_inv, damp = _invert_hessian(hessian, damp_percent)
    scale = original.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_E4M3_MAX
    working = original.clone()
    quantized = torch.empty_like(working)

    blocksize = max(1, int(blocksize))
    for start in range(0, columns, blocksize):
        end = min(start + blocksize, columns)
        count = end - start
        block = working[:, start:end].clone()
        err_block = torch.zeros_like(block)
        h_block = h_inv[start:end, start:end].float()

        for idx in range(count):
            values = block[:, idx]
            q = _quantize_fp8_e4m3_with_scale(values[:, None], scale).squeeze(1)
            quantized[:, start + idx] = q
            denom = float(h_block[idx, idx].item())
            if abs(denom) < 1e-12:
                denom = 1e-12
            err = (values - q) / denom
            block[:, idx:] -= err[:, None] * h_block[idx, idx:][None, :]
            err_block[:, idx] = err

        if end < columns:
            working[:, end:] -= err_block @ h_inv[start:end, end:].float()

    diff = original - quantized
    return quantized.to(dtype=weight.dtype), {
        "columns": columns,
        "weights": original.numel(),
        "damp": damp,
        "blocksize": blocksize,
        "hessian_approximation": "full",
        "mean_abs_error": float(diff.abs().mean().item()),
        "max_abs_error": float(diff.abs().max().item()),
    }


def linear_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Linear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]


def snapshot_linear_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: module.weight.detach().float().cpu().clone() for name, module in linear_modules(model)}


def set_linear_weights(model: torch.nn.Module, weights: dict[str, torch.Tensor]) -> None:
    for name, module in linear_modules(model):
        weight = weights.get(name)
        if weight is None:
            raise KeyError(f"missing linear weight for {name}")
        with torch.no_grad():
            module.weight.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))


class LinearHessianCollector:
    def __init__(self, modules: list[tuple[str, torch.nn.Linear]], *, hessian_approximation: str = "full") -> None:
        if hessian_approximation not in {"full", "diagonal"}:
            raise ValueError(f"unknown hessian approximation: {hessian_approximation}")
        self.hessian_approximation = hessian_approximation
        if hessian_approximation == "diagonal":
            self.hessians = {name: torch.zeros(module.in_features, dtype=torch.float64) for name, module in modules}
        else:
            self.hessians = {
                name: torch.zeros(module.in_features, module.in_features, dtype=torch.float64) for name, module in modules
            }
        self.tokens = {name: 0 for name, _ in modules}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def install(self, modules: list[tuple[str, torch.nn.Linear]]) -> None:
        for name, module in modules:
            self._handles.append(module.register_forward_pre_hook(self._make_hook(name)))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            if self.hessian_approximation == "diagonal":
                hessian = x.square().sum(dim=0).double().cpu()
            else:
                hessian = (x.transpose(0, 1) @ x).double().cpu()
            self.hessians[name].add_(hessian)
            self.tokens[name] += int(x.shape[0])

        return hook


def _prepare_tokenizer(model_name: str, revision: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _prepare_model(model_name: str, revision: str | None, dtype: torch.dtype, device: torch.device):
    load_dtype = dtype if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=load_dtype)
    model.config.use_cache = False
    model.to(device)
    model.eval()
    return model


def _load_records(config: GPTQConfig, split: str, max_examples: int) -> list[dict[str, Any]]:
    dataset = load_dataset(config.dataset_name, config.dataset_config, split=split)
    limit = min(max_examples, len(dataset)) if max_examples > 0 else len(dataset)
    return [dict(row) for row in dataset.select(range(limit))]


def collect_linear_hessians(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: GPTQConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    modules = linear_modules(model)
    collector = LinearHessianCollector(modules, hessian_approximation=config.hessian_approximation)
    collector.install(modules)
    try:
        for record_batch in tqdm(list(batched(records, config.calibration_batch_size)), desc="gptq_calib", unit="batch"):
            batch = build_causal_lm_batch(
                tokenizer,
                record_batch,
                config.max_length,
                answer_only_loss=config.answer_only_loss,
                device=device,
            )
            with torch.inference_mode():
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    finally:
        collector.close()
    return collector.hessians, collector.tokens


def apply_gptq_fp8_to_linear(
    name: str,
    module: torch.nn.Linear,
    hessian: torch.Tensor,
    config: GPTQConfig,
) -> dict[str, object]:
    before = module.weight.detach().float().cpu()
    quantized, stats = gptq_quantize_weight(
        before,
        hessian,
        damp_percent=config.damp_percent,
        blocksize=config.blocksize,
    )
    with torch.no_grad():
        module.weight.copy_(quantized.to(device=module.weight.device, dtype=module.weight.dtype))
    return {
        "name": name,
        "shape": list(before.shape),
        "format": "fp8_e4m3",
        **stats,
    }


def apply_gptq_fp8(model: torch.nn.Module, hessians: dict[str, torch.Tensor], config: GPTQConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, module in tqdm(linear_modules(model), desc="gptq_quant", unit="layer"):
        hessian = hessians.get(name)
        if hessian is None:
            raise KeyError(f"missing GPTQ hessian for {name}")
        rows.append(apply_gptq_fp8_to_linear(name, module, hessian, config))
    return rows


def apply_gptq_fp8_from_originals(
    model: torch.nn.Module,
    hessians: dict[str, torch.Tensor],
    original_weights: dict[str, torch.Tensor],
    config: GPTQConfig,
    *,
    step: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, module in tqdm(linear_modules(model), desc=f"gptq_quant_step_{step}", unit="layer"):
        hessian = hessians.get(name)
        if hessian is None:
            raise KeyError(f"missing GPTQ hessian for {name}")
        before = original_weights.get(name)
        if before is None:
            raise KeyError(f"missing original weight for {name}")
        quantized, stats = gptq_quantize_weight(
            before,
            hessian,
            damp_percent=config.damp_percent,
            blocksize=config.blocksize,
        )
        with torch.no_grad():
            module.weight.copy_(quantized.to(device=module.weight.device, dtype=module.weight.dtype))
        rows.append(
            {
                "step": step,
                "name": name,
                "shape": list(before.shape),
                "format": "fp8_e4m3",
                **stats,
            }
        )
    return rows


def compute_gptq_fp8_targets(
    model: torch.nn.Module,
    hessians: dict[str, torch.Tensor],
    original_weights: dict[str, torch.Tensor],
    config: GPTQConfig,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    targets: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for name, module in tqdm(linear_modules(model), desc="gptq_target", unit="layer"):
        del module
        hessian = hessians.get(name)
        if hessian is None:
            raise KeyError(f"missing GPTQ hessian for {name}")
        before = original_weights.get(name)
        if before is None:
            raise KeyError(f"missing original weight for {name}")
        quantized, stats = gptq_quantize_weight(
            before,
            hessian,
            damp_percent=config.damp_percent,
            blocksize=config.blocksize,
        )
        targets[name] = quantized
        rows.append(
            {
                "step": 1,
                "name": name,
                "shape": list(before.shape),
                "format": "fp8_e4m3",
                **stats,
            }
        )
    return targets, rows


def apply_staged_gptq_weights(
    model: torch.nn.Module,
    original_weights: dict[str, torch.Tensor],
    quantized_targets: dict[str, torch.Tensor],
    *,
    alpha: float,
    step: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, module in linear_modules(model):
        before = original_weights.get(name)
        target = quantized_targets.get(name)
        if before is None:
            raise KeyError(f"missing original weight for {name}")
        if target is None:
            raise KeyError(f"missing GPTQ target for {name}")
        staged = before + float(alpha) * (target.float() - before)
        with torch.no_grad():
            module.weight.copy_(staged.to(device=module.weight.device, dtype=module.weight.dtype))
        diff = before - staged
        rows.append(
            {
                "step": step,
                "alpha": float(alpha),
                "name": name,
                "shape": list(before.shape),
                "format": "staged_fp8_e4m3_target",
                "weights": before.numel(),
                "mean_abs_delta_from_original": float(diff.abs().mean().item()),
                "max_abs_delta_from_original": float(diff.abs().max().item()),
            }
        )
    return rows


def apply_damped_gptq_update(
    model: torch.nn.Module,
    current_weights: dict[str, torch.Tensor],
    quantized_targets: dict[str, torch.Tensor],
    *,
    step_alpha: float,
    step: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, module in linear_modules(model):
        before = current_weights.get(name)
        target = quantized_targets.get(name)
        if before is None:
            raise KeyError(f"missing current weight for {name}")
        if target is None:
            raise KeyError(f"missing GPTQ target for {name}")
        updated = before + float(step_alpha) * (target.float() - before)
        with torch.no_grad():
            module.weight.copy_(updated.to(device=module.weight.device, dtype=module.weight.dtype))
        delta = updated - before
        target_gap = target.float() - updated
        rows.append(
            {
                "step": step,
                "step_alpha": float(step_alpha),
                "name": name,
                "shape": list(before.shape),
                "format": "damped_gptq_fp8_target",
                "weights": before.numel(),
                "mean_abs_step_delta": float(delta.abs().mean().item()),
                "max_abs_step_delta": float(delta.abs().max().item()),
                "mean_abs_remaining_to_target": float(target_gap.abs().mean().item()),
                "max_abs_remaining_to_target": float(target_gap.abs().max().item()),
            }
        )
    return rows


def _hessian_damp_from_tensor(hessian: torch.Tensor, damp_percent: float) -> float:
    if hessian.ndim == 1:
        return _diagonal_hessian_damp(hessian, damp_percent)
    diag = torch.diag(hessian.double().cpu())
    mean_diag = float(diag[diag > 0].mean().item()) if torch.any(diag > 0) else 1.0
    return max(mean_diag * damp_percent, 1e-8)


def apply_gradient_descent_gptq_update(
    model: torch.nn.Module,
    current_weights: dict[str, torch.Tensor],
    quantized_targets: dict[str, torch.Tensor],
    hessians: dict[str, torch.Tensor],
    *,
    damp_percent: float,
    gradient_step_scale: float,
    step: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, module in linear_modules(model):
        before = current_weights.get(name)
        target = quantized_targets.get(name)
        hessian = hessians.get(name)
        if before is None:
            raise KeyError(f"missing current weight for {name}")
        if target is None:
            raise KeyError(f"missing GPTQ target for {name}")
        if hessian is None:
            raise KeyError(f"missing GPTQ hessian for {name}")

        device = module.weight.device
        delta_to_target = before.to(device=device, dtype=torch.float32) - target.to(device=device, dtype=torch.float32)
        damp = _hessian_damp_from_tensor(hessian, damp_percent)
        if hessian.ndim == 1:
            h_diag = hessian.to(device=device, dtype=torch.float32) + float(damp)
            lipschitz_bound = float(h_diag.abs().max().clamp_min(1e-12).item())
            grad = delta_to_target * h_diag[None, :]
        else:
            h = hessian.to(device=device, dtype=torch.float32).clone()
            h.diagonal().add_(float(damp))
            lipschitz_bound = float(h.abs().sum(dim=1).max().clamp_min(1e-12).item())
            grad = delta_to_target @ h

        lr = float(gradient_step_scale) / lipschitz_bound
        before_device = before.to(device=device, dtype=torch.float32)
        updated = before_device - lr * grad
        with torch.no_grad():
            module.weight.copy_(updated.to(dtype=module.weight.dtype))
        step_delta = updated - before_device
        remaining = target.to(device=device, dtype=torch.float32) - updated
        rows.append(
            {
                "step": step,
                "gradient_step_scale": float(gradient_step_scale),
                "gradient_step_lr": lr,
                "gradient_lipschitz_bound": lipschitz_bound,
                "name": name,
                "shape": list(before.shape),
                "format": "gradient_descent_gptq_fp8_target",
                "weights": before.numel(),
                "mean_abs_step_delta": float(step_delta.abs().mean().item()),
                "max_abs_step_delta": float(step_delta.abs().max().item()),
                "mean_abs_remaining_to_target": float(remaining.abs().mean().item()),
                "max_abs_remaining_to_target": float(remaining.abs().max().item()),
            }
        )
    return rows


def _normalized_eval_steps(gptq_steps: int, eval_steps: tuple[int, ...] | None) -> tuple[int, ...]:
    if gptq_steps <= 0:
        raise ValueError("gptq_steps must be positive")
    requested = eval_steps if eval_steps else (gptq_steps,)
    normalized = tuple(sorted(set(int(step) for step in requested)))
    if any(step <= 0 or step > gptq_steps for step in normalized):
        raise ValueError("eval_steps must be between 1 and gptq_steps")
    return normalized


def _ppl_config(config: GPTQConfig, split: str, max_examples: int):
    from saliency.prune_eval import PruneEvalConfig

    return PruneEvalConfig(
        output_dir=config.output_dir,
        saliency_path="unused",
        model_name=config.model_name,
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=split,
        max_examples=max_examples,
        batch_size=config.eval_batch_size,
        max_length=config.max_length,
        dtype=config.dtype,
        device=config.device,
        answer_only_loss=config.answer_only_loss,
        revision=config.revision,
    )


def run_gptq_fp8_experiment(config: GPTQConfig) -> dict[str, object]:
    started = time.time()
    device = resolve_device(config.device)
    dtype = resolve_torch_dtype(config.dtype)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    tokenizer = _prepare_tokenizer(config.model_name, config.revision)
    model = _prepare_model(config.model_name, config.revision, dtype, device)
    calibration_records = _load_records(config, config.calibration_split, config.max_calibration_examples)
    eval_records = _load_records(config, config.eval_split, config.max_eval_examples)
    eval_config = _ppl_config(config, config.eval_split, config.max_eval_examples)
    eval_steps = _normalized_eval_steps(config.gptq_steps, config.eval_steps)
    original_weights = snapshot_linear_weights(model)
    enabled_step_modes = sum(bool(value) for value in (config.staged_to_wq, config.iterative_damped_gptq, config.gradient_descent_gptq))
    if enabled_step_modes > 1:
        raise ValueError("staged_to_wq, iterative_damped_gptq, and gradient_descent_gptq are mutually exclusive")
    if config.hessian_approximation not in {"full", "diagonal"}:
        raise ValueError("hessian_approximation must be 'full' or 'diagonal'")
    step_alpha = config.newton_step_alpha if config.newton_step_alpha is not None else 1.0 / config.gptq_steps
    if step_alpha <= 0.0:
        raise ValueError("newton_step_alpha must be positive")
    if config.gradient_step_scale <= 0.0:
        raise ValueError("gradient_step_scale must be positive")
    if config.gradient_step_scales is not None and any(scale <= 0.0 for scale in config.gradient_step_scales):
        raise ValueError("gradient_step_scales must all be positive")

    baseline = evaluate_perplexity(model, tokenizer, eval_records, eval_config, device, desc="baseline_test_ppl")
    if config.gradient_descent_gptq and config.gradient_step_scales is not None:
        if config.gptq_steps != 1:
            raise ValueError("gradient_step_scales sweep currently supports gptq_steps=1")
        hessians, hessian_tokens = collect_linear_hessians(model, tokenizer, calibration_records, config, device)
        quantized_targets, target_layers = compute_gptq_fp8_targets(model, hessians, original_weights, config)
        target_weights = sum(int(row["weights"]) for row in target_layers)
        target_mean_abs_error = (
            sum(float(row["mean_abs_error"]) * int(row["weights"]) for row in target_layers) / max(target_weights, 1)
        )

        gradient_layers: list[dict[str, object]] = []
        step_results: list[dict[str, object]] = []
        best_result: dict[str, object] | None = None
        best_ppl = float("inf")

        for idx, scale in enumerate(config.gradient_step_scales, start=1):
            set_linear_weights(model, original_weights)
            step_layers = apply_gradient_descent_gptq_update(
                model,
                original_weights,
                quantized_targets,
                hessians,
                damp_percent=config.damp_percent,
                gradient_step_scale=float(scale),
                step=idx,
            )
            gradient_layers.extend(step_layers)
            mean_abs_step_delta = (
                sum(float(row["mean_abs_step_delta"]) * int(row["weights"]) for row in step_layers)
                / max(target_weights, 1)
            )
            mean_abs_remaining = (
                sum(float(row["mean_abs_remaining_to_target"]) * int(row["weights"]) for row in step_layers)
                / max(target_weights, 1)
            )
            mean_lipschitz_bound = (
                sum(float(row["gradient_lipschitz_bound"]) * int(row["weights"]) for row in step_layers)
                / max(target_weights, 1)
            )
            quantized = evaluate_perplexity(
                model,
                tokenizer,
                eval_records,
                eval_config,
                device,
                desc=f"gptq_fp8_gradient_scale_{idx}_test_ppl",
            )
            ppl_change = summarize_ppl_change(baseline, quantized)
            ppl_change["quantized"] = ppl_change.pop("pruned")
            step_result: dict[str, object] = {
                "step": idx,
                "gradient_step_scale": float(scale),
                "linear_layers": len(step_layers),
                "quantized_weights": target_weights,
                "target_weighted_mean_abs_error": target_mean_abs_error,
                "weighted_mean_abs_step_delta": mean_abs_step_delta,
                "weighted_mean_abs_remaining_to_target": mean_abs_remaining,
                "weighted_mean_lipschitz_bound": mean_lipschitz_bound,
                "ppl_change": ppl_change,
            }
            step_results.append(step_result)
            ppl = float(ppl_change["quantized"]["perplexity"])
            if ppl < best_ppl:
                best_ppl = ppl
                best_result = step_result

        if best_result is None:
            raise RuntimeError("no gradient descent scale was evaluated")
        result = {
            "metadata": {
                **asdict(config),
                "output_dir": str(config.output_dir),
                "device": str(device),
                "torch_dtype": str(dtype),
                "elapsed_seconds": time.time() - started,
                "quantization_method": f"one-step gradient descent step-size sweep toward GPTQ FP8 target with {config.hessian_approximation} activation Hessian",
                "calibration_examples": len(calibration_records),
                "eval_examples": len(eval_records),
                "eval_steps": list(eval_steps),
                "best_gradient_step_scale": best_result["gradient_step_scale"],
            },
            "gptq": {
                "format": "fp8_e4m3",
                "steps": 1,
                "linear_layers": len(target_layers),
                "quantized_weights": target_weights,
                "weighted_mean_abs_error": target_mean_abs_error,
                "hessian_tokens": hessian_tokens,
            },
            "ppl_change": best_result["ppl_change"],
            "step_results": step_results,
            "target_layers": target_layers,
            "layers": gradient_layers,
        }

        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gptq_fp8_summary.json").write_text(json.dumps(result, indent=2) + "\n")
        with (out / "gptq_layers.jsonl").open("w") as handle:
            for row in target_layers:
                handle.write(json.dumps(row) + "\n")
        return result

    if config.gradient_descent_gptq:
        gradient_layers: list[dict[str, object]] = []
        target_layers: list[dict[str, object]] = []
        step_results: list[dict[str, object]] = []
        hessian_tokens_by_step: dict[str, dict[str, int]] = {}
        final_ppl_change: dict[str, object] | None = None
        final_eval_step = eval_steps[-1]

        for step in range(1, config.gptq_steps + 1):
            current_weights = snapshot_linear_weights(model)
            hessians, hessian_tokens = collect_linear_hessians(model, tokenizer, calibration_records, config, device)
            hessian_tokens_by_step[str(step)] = hessian_tokens
            quantized_targets, step_target_layers = compute_gptq_fp8_targets(model, hessians, current_weights, config)
            for row in step_target_layers:
                row["step"] = step
            target_layers.extend(step_target_layers)
            step_layers = apply_gradient_descent_gptq_update(
                model,
                current_weights,
                quantized_targets,
                hessians,
                damp_percent=config.damp_percent,
                gradient_step_scale=config.gradient_step_scale,
                step=step,
            )
            gradient_layers.extend(step_layers)

            total_step_weights = sum(int(row["weights"]) for row in step_target_layers)
            target_mean_abs_error = (
                sum(float(row["mean_abs_error"]) * int(row["weights"]) for row in step_target_layers)
                / max(total_step_weights, 1)
            )
            mean_abs_step_delta = (
                sum(float(row["mean_abs_step_delta"]) * int(row["weights"]) for row in step_layers)
                / max(total_step_weights, 1)
            )
            mean_abs_remaining = (
                sum(float(row["mean_abs_remaining_to_target"]) * int(row["weights"]) for row in step_layers)
                / max(total_step_weights, 1)
            )
            mean_lipschitz_bound = (
                sum(float(row["gradient_lipschitz_bound"]) * int(row["weights"]) for row in step_layers)
                / max(total_step_weights, 1)
            )
            step_result: dict[str, object] = {
                "step": step,
                "gradient_step_scale": config.gradient_step_scale,
                "linear_layers": len(step_layers),
                "quantized_weights": total_step_weights,
                "target_weighted_mean_abs_error": target_mean_abs_error,
                "weighted_mean_abs_step_delta": mean_abs_step_delta,
                "weighted_mean_abs_remaining_to_target": mean_abs_remaining,
                "weighted_mean_lipschitz_bound": mean_lipschitz_bound,
            }
            if step in eval_steps:
                quantized = evaluate_perplexity(
                    model,
                    tokenizer,
                    eval_records,
                    eval_config,
                    device,
                    desc=f"gptq_fp8_gradient_step_{step}_test_ppl",
                )
                ppl_change = summarize_ppl_change(baseline, quantized)
                ppl_change["quantized"] = ppl_change.pop("pruned")
                step_result["ppl_change"] = ppl_change
                if step == final_eval_step:
                    final_ppl_change = ppl_change
            step_results.append(step_result)

        if final_ppl_change is None:
            raise RuntimeError("no gradient descent GPTQ step was evaluated")

        final_targets = [row for row in target_layers if int(row["step"]) == config.gptq_steps]
        total_weights = sum(int(row["weights"]) for row in final_targets)
        mean_abs_error = (
            sum(float(row["mean_abs_error"]) * int(row["weights"]) for row in final_targets) / max(total_weights, 1)
        )
        result = {
            "metadata": {
                **asdict(config),
                "output_dir": str(config.output_dir),
                "device": str(device),
                "torch_dtype": str(dtype),
                "elapsed_seconds": time.time() - started,
                "quantization_method": f"gradient descent steps toward per-step GPTQ FP8 target with {config.hessian_approximation} activation Hessian",
                "calibration_examples": len(calibration_records),
                "eval_examples": len(eval_records),
                "eval_steps": list(eval_steps),
            },
            "gptq": {
                "format": "fp8_e4m3",
                "steps": config.gptq_steps,
                "linear_layers": len(final_targets),
                "quantized_weights": total_weights,
                "weighted_mean_abs_error": mean_abs_error,
                "hessian_tokens_by_step": hessian_tokens_by_step,
            },
            "ppl_change": final_ppl_change,
            "step_results": step_results,
            "target_layers": target_layers,
            "layers": gradient_layers,
        }

        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gptq_fp8_summary.json").write_text(json.dumps(result, indent=2) + "\n")
        with (out / "gptq_layers.jsonl").open("w") as handle:
            for row in target_layers:
                handle.write(json.dumps(row) + "\n")
        return result

    if config.iterative_damped_gptq:
        damped_layers: list[dict[str, object]] = []
        target_layers: list[dict[str, object]] = []
        step_results: list[dict[str, object]] = []
        hessian_tokens_by_step: dict[str, dict[str, int]] = {}
        final_ppl_change: dict[str, object] | None = None
        final_eval_step = eval_steps[-1]

        for step in range(1, config.gptq_steps + 1):
            current_weights = snapshot_linear_weights(model)
            hessians, hessian_tokens = collect_linear_hessians(model, tokenizer, calibration_records, config, device)
            hessian_tokens_by_step[str(step)] = hessian_tokens
            quantized_targets, step_target_layers = compute_gptq_fp8_targets(model, hessians, current_weights, config)
            for row in step_target_layers:
                row["step"] = step
            target_layers.extend(step_target_layers)
            step_layers = apply_damped_gptq_update(
                model,
                current_weights,
                quantized_targets,
                step_alpha=step_alpha,
                step=step,
            )
            damped_layers.extend(step_layers)

            total_step_weights = sum(int(row["weights"]) for row in step_target_layers)
            target_mean_abs_error = (
                sum(float(row["mean_abs_error"]) * int(row["weights"]) for row in step_target_layers)
                / max(total_step_weights, 1)
            )
            mean_abs_step_delta = (
                sum(float(row["mean_abs_step_delta"]) * int(row["weights"]) for row in step_layers)
                / max(total_step_weights, 1)
            )
            mean_abs_remaining = (
                sum(float(row["mean_abs_remaining_to_target"]) * int(row["weights"]) for row in step_layers)
                / max(total_step_weights, 1)
            )
            step_result: dict[str, object] = {
                "step": step,
                "step_alpha": step_alpha,
                "nominal_cumulative_alpha": step * step_alpha,
                "linear_layers": len(step_layers),
                "quantized_weights": total_step_weights,
                "target_weighted_mean_abs_error": target_mean_abs_error,
                "weighted_mean_abs_step_delta": mean_abs_step_delta,
                "weighted_mean_abs_remaining_to_target": mean_abs_remaining,
            }
            if step in eval_steps:
                quantized = evaluate_perplexity(
                    model,
                    tokenizer,
                    eval_records,
                    eval_config,
                    device,
                    desc=f"gptq_fp8_damped_step_{step}_test_ppl",
                )
                ppl_change = summarize_ppl_change(baseline, quantized)
                ppl_change["quantized"] = ppl_change.pop("pruned")
                step_result["ppl_change"] = ppl_change
                if step == final_eval_step:
                    final_ppl_change = ppl_change
            step_results.append(step_result)

        if final_ppl_change is None:
            raise RuntimeError("no iterative damped GPTQ step was evaluated")

        final_targets = [row for row in target_layers if int(row["step"]) == config.gptq_steps]
        total_weights = sum(int(row["weights"]) for row in final_targets)
        mean_abs_error = (
            sum(float(row["mean_abs_error"]) * int(row["weights"]) for row in final_targets) / max(total_weights, 1)
        )
        result = {
            "metadata": {
                **asdict(config),
                "output_dir": str(config.output_dir),
                "device": str(device),
                "torch_dtype": str(dtype),
                "elapsed_seconds": time.time() - started,
                "quantization_method": f"iterative damped GPTQ/Newton steps with {config.hessian_approximation} activation Hessian: collect current Hessian, compute current FP8 target, move partway toward it",
                "calibration_examples": len(calibration_records),
                "eval_examples": len(eval_records),
                "eval_steps": list(eval_steps),
                "effective_step_alpha": step_alpha,
            },
            "gptq": {
                "format": "fp8_e4m3",
                "steps": config.gptq_steps,
                "linear_layers": len(final_targets),
                "quantized_weights": total_weights,
                "weighted_mean_abs_error": mean_abs_error,
                "hessian_tokens_by_step": hessian_tokens_by_step,
            },
            "ppl_change": final_ppl_change,
            "step_results": step_results,
            "target_layers": target_layers,
            "layers": damped_layers,
        }

        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gptq_fp8_summary.json").write_text(json.dumps(result, indent=2) + "\n")
        with (out / "gptq_layers.jsonl").open("w") as handle:
            for row in target_layers:
                handle.write(json.dumps(row) + "\n")
        return result

    if config.staged_to_wq:
        hessians, hessian_tokens = collect_linear_hessians(model, tokenizer, calibration_records, config, device)
        quantized_targets, target_layers = compute_gptq_fp8_targets(model, hessians, original_weights, config)
        target_weights = sum(int(row["weights"]) for row in target_layers)
        target_mean_abs_error = (
            sum(float(row["mean_abs_error"]) * int(row["weights"]) for row in target_layers) / max(target_weights, 1)
        )

        staged_layers: list[dict[str, object]] = []
        step_results: list[dict[str, object]] = []
        final_ppl_change: dict[str, object] | None = None
        final_eval_step = eval_steps[-1]

        for step in range(1, config.gptq_steps + 1):
            alpha = step / config.gptq_steps
            step_layers = apply_staged_gptq_weights(model, original_weights, quantized_targets, alpha=alpha, step=step)
            staged_layers.extend(step_layers)
            step_result: dict[str, object] = {
                "step": step,
                "alpha": alpha,
                "linear_layers": len(step_layers),
                "quantized_weights": target_weights,
                "target_weighted_mean_abs_error": target_mean_abs_error,
            }
            if step in eval_steps:
                quantized = evaluate_perplexity(
                    model,
                    tokenizer,
                    eval_records,
                    eval_config,
                    device,
                    desc=f"gptq_fp8_staged_step_{step}_test_ppl",
                )
                ppl_change = summarize_ppl_change(baseline, quantized)
                ppl_change["quantized"] = ppl_change.pop("pruned")
                step_result["ppl_change"] = ppl_change
                if step == final_eval_step:
                    final_ppl_change = ppl_change
            step_results.append(step_result)

        if final_ppl_change is None:
            raise RuntimeError("no staged GPTQ step was evaluated")

        result = {
            "metadata": {
                **asdict(config),
                "output_dir": str(config.output_dir),
                "device": str(device),
                "torch_dtype": str(dtype),
                "elapsed_seconds": time.time() - started,
                "quantization_method": f"staged path from original weights to one-step GPTQ FP8 target with {config.hessian_approximation} activation Hessian",
                "calibration_examples": len(calibration_records),
                "eval_examples": len(eval_records),
                "eval_steps": list(eval_steps),
            },
            "gptq": {
                "format": "fp8_e4m3",
                "steps": config.gptq_steps,
                "linear_layers": len(target_layers),
                "quantized_weights": target_weights,
                "weighted_mean_abs_error": target_mean_abs_error,
                "hessian_tokens": hessian_tokens,
            },
            "ppl_change": final_ppl_change,
            "step_results": step_results,
            "target_layers": target_layers,
            "layers": staged_layers,
        }

        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gptq_fp8_summary.json").write_text(json.dumps(result, indent=2) + "\n")
        with (out / "gptq_layers.jsonl").open("w") as handle:
            for row in target_layers:
                handle.write(json.dumps(row) + "\n")
        return result

    quantized_layers: list[dict[str, object]] = []
    step_results: list[dict[str, object]] = []
    hessian_tokens_by_step: dict[str, dict[str, int]] = {}
    final_ppl_change: dict[str, object] | None = None
    final_eval_step = eval_steps[-1]

    for step in range(1, config.gptq_steps + 1):
        hessians, hessian_tokens = collect_linear_hessians(model, tokenizer, calibration_records, config, device)
        hessian_tokens_by_step[str(step)] = hessian_tokens
        step_layers = apply_gptq_fp8_from_originals(model, hessians, original_weights, config, step=step)
        quantized_layers.extend(step_layers)

        total_step_weights = sum(int(row["weights"]) for row in step_layers)
        step_mean_abs_error = (
            sum(float(row["mean_abs_error"]) * int(row["weights"]) for row in step_layers) / max(total_step_weights, 1)
        )
        step_result: dict[str, object] = {
            "step": step,
            "linear_layers": len(step_layers),
            "quantized_weights": total_step_weights,
            "weighted_mean_abs_error": step_mean_abs_error,
        }
        if step in eval_steps:
            quantized = evaluate_perplexity(
                model,
                tokenizer,
                eval_records,
                eval_config,
                device,
                desc=f"gptq_fp8_step_{step}_test_ppl",
            )
            ppl_change = summarize_ppl_change(baseline, quantized)
            ppl_change["quantized"] = ppl_change.pop("pruned")
            step_result["ppl_change"] = ppl_change
            if step == final_eval_step:
                final_ppl_change = ppl_change
        step_results.append(step_result)

    if final_ppl_change is None:
        raise RuntimeError("no GPTQ step was evaluated")

    final_layers = [row for row in quantized_layers if int(row["step"]) == config.gptq_steps]
    total_weights = sum(int(row["weights"]) for row in final_layers)
    mean_abs_error = (
        sum(float(row["mean_abs_error"]) * int(row["weights"]) for row in final_layers) / max(total_weights, 1)
    )
    result = {
        "metadata": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "device": str(device),
            "torch_dtype": str(dtype),
            "elapsed_seconds": time.time() - started,
            "quantization_method": f"multi-step GPTQ with per-step {config.hessian_approximation} activation Hessians from calibration split and per-row scaled torch.float8_e4m3fn weights",
            "calibration_examples": len(calibration_records),
            "eval_examples": len(eval_records),
            "eval_steps": list(eval_steps),
        },
        "gptq": {
            "format": "fp8_e4m3",
            "steps": config.gptq_steps,
            "linear_layers": len(final_layers),
            "quantized_weights": total_weights,
            "weighted_mean_abs_error": mean_abs_error,
            "hessian_tokens_by_step": hessian_tokens_by_step,
        },
        "ppl_change": final_ppl_change,
        "step_results": step_results,
        "layers": quantized_layers,
    }

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gptq_fp8_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "gptq_layers.jsonl").open("w") as handle:
        for row in quantized_layers:
            handle.write(json.dumps(row) + "\n")
    return result
