from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from saliency.approx import sequential_wanda_matrix_parameter_groups
from saliency.calibration import batched, build_causal_lm_batch
from saliency.experiment import resolve_device, resolve_torch_dtype
from saliency.gptq_eval import linear_modules
from saliency.prune_eval import (
    evaluate_perplexity,
    lowest_saliency_mask,
    lowest_saliency_mask_per_output_row,
    summarize_ppl_change,
)


@dataclass(slots=True)
class QronosConfig:
    output_dir: str | Path
    model_name: str = "EleutherAI/pythia-31m"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    eval_split: str = ""
    max_examples: int = 128
    max_eval_examples: int = 0
    batch_size: int = 8
    max_length: int = 512
    dtype: str = "bf16"
    device: str = "auto"
    answer_only_loss: bool = True
    weight_bits: int = 4
    beta: float = 1.0
    percdamp: float = 1e-6
    cholesky_scale: float = 1e4
    num_blocks: int = 100
    use_activation_order: bool = True
    use_attention_mask: bool = True
    quantize_last_layer: bool = False
    prune_fraction: float = 0.25
    pruning_scope: str = "per_output_row"
    revision: str | None = None


def asymmetric_minmax_quantize(
    values: torch.Tensor,
    *,
    bits: int,
    beta: float = 1.0,
    scale: torch.Tensor | None = None,
    zero: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if bits < 1:
        raise ValueError("bits must be positive")
    if beta <= 0:
        raise ValueError("beta must be positive")

    source = values.detach().float()
    qmin = 0
    qmax = (1 << int(bits)) - 1
    if scale is None or zero is None:
        row_min = source.amin(dim=1, keepdim=True)
        row_max = source.amax(dim=1, keepdim=True)
        scale = (float(beta) * (row_max - row_min) / max(qmax - qmin, 1)).clamp_min(1e-12)
        zero = row_min / scale

    quant_int = torch.round(source / scale - zero).clamp(qmin, qmax)
    quantized = scale * (quant_int + zero)
    return quantized.to(dtype=values.dtype), scale, zero


def _spectral_norm_power_iteration(matrix: torch.Tensor, *, steps: int = 8) -> float:
    if matrix.numel() == 0:
        return 1.0
    work = matrix.float()
    vec = torch.ones(work.shape[0], dtype=work.dtype, device=work.device)
    vec = vec / vec.norm().clamp_min(1e-12)
    for _ in range(max(1, steps)):
        vec = work @ vec
        vec = vec / vec.norm().clamp_min(1e-12)
    return float((vec @ (work @ vec)).abs().item())


def _stabilized_hessian(hessian: torch.Tensor, percdamp: float) -> tuple[torch.Tensor, float]:
    h = hessian.detach().float()
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("Qronos hessian must be a square 2D tensor")
    eye = torch.eye(h.shape[0], dtype=h.dtype, device=h.device)
    base = max(_spectral_norm_power_iteration(h), 1e-8) * max(float(percdamp), 0.0)
    for multiplier in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        damp = base * multiplier
        try:
            torch.linalg.cholesky(h + eye * damp)
            return h + eye * damp, damp
        except torch.linalg.LinAlgError:
            continue
    damp = max(base, 1e-6) * 10000.0
    return h + eye * damp, damp


def _upper_cholesky_with_jitter(matrix: torch.Tensor) -> tuple[torch.Tensor, float]:
    jitter = 0.0
    try:
        return torch.linalg.cholesky(matrix, upper=True), jitter
    except torch.linalg.LinAlgError:
        diag = torch.diag(matrix)
        positive = diag[diag > 0]
        base_jitter = max(float(positive.mean().item()) if positive.numel() else 1.0, 1.0) * 1e-7
        eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        for multiplier in (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0):
            jitter = base_jitter * multiplier
            try:
                return torch.linalg.cholesky(matrix + eye * jitter, upper=True), jitter
            except torch.linalg.LinAlgError:
                continue
        raise


def qronos_quantize_weight(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    cross: torch.Tensor,
    *,
    bits: int = 4,
    beta: float = 1.0,
    percdamp: float = 1e-6,
    cholesky_scale: float = 1e4,
    num_blocks: int = 100,
    use_activation_order: bool = True,
) -> tuple[torch.Tensor, dict[str, object]]:
    if weight.ndim != 2:
        raise ValueError("Qronos weight must be a 2D matrix")

    original = weight.detach().float()
    if original.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    rows, columns = original.shape
    if tuple(hessian.shape) != (columns, columns):
        raise ValueError(f"hessian shape {tuple(hessian.shape)} does not match input columns {columns}")
    if tuple(cross.shape) != (columns, columns):
        raise ValueError(f"cross shape {tuple(cross.shape)} does not match input columns {columns}")

    h = hessian.detach().to(device=original.device, dtype=torch.float32)
    g = cross.detach().to(device=original.device, dtype=torch.float32)
    if use_activation_order:
        order = torch.argsort(torch.diag(h), descending=True)
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(columns, dtype=order.dtype, device=order.device)
        h = h.index_select(0, order).index_select(1, order)
        g = g.index_select(0, order).index_select(1, order)
        working = original.index_select(1, order).clone()
    else:
        inv_order = None
        working = original.clone()

    quantized = torch.empty_like(working)
    _, scale, zero = asymmetric_minmax_quantize(working, bits=bits, beta=beta)

    def quantize_current_column(column: int) -> torch.Tensor:
        q, _, _ = asymmetric_minmax_quantize(
            working[:, column : column + 1],
            bits=bits,
            beta=beta,
            scale=scale,
            zero=zero,
        )
        return q.squeeze(1)

    dh = torch.diag(h)
    dhi = torch.where(dh != 0, 1.0 / dh, torch.zeros_like(dh))
    uh = torch.triu(h, diagonal=1)
    h_damped, damp = _stabilized_hessian(h, percdamp)
    h_inv = torch.cholesky_inverse(torch.linalg.cholesky(h_damped))
    cholesky_jitter = 0.0

    first_proxy = (working @ (g[:, 0] * dhi[0])) - (working @ (uh[0, :] * dhi[0]))
    working[:, 0] = first_proxy
    quantized[:, 0] = quantize_current_column(0)

    if columns > 1:
        tail_inv = h_inv[1:, 1:].clone()
        c0 = h_inv[0, 0].clamp_min(1e-12)
        b0 = h_inv[1:, [0]]
        tail_inv -= (b0 @ b0.T) / c0

        gh = g + torch.eye(columns, dtype=g.dtype, device=g.device) * float(damp)
        future = (original if not use_activation_order else original.index_select(1, order))
        working[:, 1:] = future @ (gh[:, 1:] @ tail_inv)
        working[:, 1:] -= quantized[:, :1] @ (h[:1, 1:] @ tail_inv)

        scaled_inv = tail_inv * float(cholesky_scale)
        chol, cholesky_jitter = _upper_cholesky_with_jitter(scaled_inv)
        chol = chol / (float(cholesky_scale) ** 0.5)
        blocksize = max(1, (columns + max(1, int(num_blocks)) - 1) // max(1, int(num_blocks)))
        for start in range(1, columns, blocksize):
            end = min(start + blocksize, columns)
            count = end - start
            error_block = torch.zeros(rows, count, dtype=working.dtype, device=working.device)
            h_inv_block = chol[start - 1 : end - 1, start - 1 : end - 1]
            for idx in range(count):
                column = start + idx
                quantized[:, column] = quantize_current_column(column)
                denom = h_inv_block[idx, idx].clamp_min(1e-12)
                error = (working[:, column] - quantized[:, column]) / denom
                error_block[:, idx] = error
                working[:, column:end] -= error[:, None] @ h_inv_block[idx, idx:][None, :]
            if end < columns:
                working[:, end:] -= error_block @ chol[start - 1 : end - 1, end - 1 :]

    if inv_order is not None:
        quantized = quantized.index_select(1, inv_order)

    diff = original - quantized
    return quantized.to(dtype=weight.dtype), {
        "format": f"asymmetric_int{bits}",
        "rows": rows,
        "columns": columns,
        "weights": int(original.numel()),
        "damp": float(damp),
        "beta": float(beta),
        "percdamp": float(percdamp),
        "cholesky_scale": float(cholesky_scale),
        "cholesky_jitter": float(cholesky_jitter),
        "num_blocks": int(num_blocks),
        "activation_order": bool(use_activation_order),
        "mean_abs_error": float(diff.abs().mean().item()),
        "max_abs_error": float(diff.abs().max().item()),
    }


def _base_precision_pruning_mask(weight: torch.Tensor, *, fraction: float, pruning_scope: str) -> torch.Tensor:
    score = weight.detach().abs()
    if pruning_scope == "per_matrix":
        return lowest_saliency_mask(score, fraction=fraction)
    if pruning_scope == "per_output_row":
        return lowest_saliency_mask_per_output_row(score, fraction=fraction)
    raise ValueError(f"unknown QRONOS pruning_scope: {pruning_scope}")


def qronos_pruning_target_names(model: torch.nn.Module) -> list[str]:
    return [name for _, target_names in sequential_wanda_matrix_parameter_groups(model) for name in target_names]


def _prune_base_precision_magnitude_fallback(
    param: torch.nn.Parameter,
    *,
    prune_fraction: float,
    pruning_scope: str,
) -> dict[str, object]:
    original = param.detach().float()
    mask = _base_precision_pruning_mask(original, fraction=prune_fraction, pruning_scope=pruning_scope).to(device=param.device)
    with torch.no_grad():
        param.masked_fill_(mask, 0)
    zeroed = int(mask.sum().item())
    diff = original - param.detach().float()
    return {
        "format": "base_precision_magnitude_pruned_fallback",
        "rows": int(original.shape[0]),
        "columns": int(original.shape[1]),
        "weights": int(original.numel()),
        "zeroed": zeroed,
        "actual_zero_fraction": zeroed / max(int(original.numel()), 1),
        "prune_fraction": float(prune_fraction),
        "pruning_scope": pruning_scope,
        "activation_order": False,
        "mean_abs_delta": float(diff.abs().mean().item()),
        "max_abs_delta": float(diff.abs().max().item()),
    }


def qronos_prune_weight(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    cross: torch.Tensor,
    *,
    prune_fraction: float = 0.25,
    pruning_scope: str = "per_output_row",
    percdamp: float = 1e-6,
    cholesky_scale: float = 1e4,
    num_blocks: int = 100,
    use_activation_order: bool = True,
) -> tuple[torch.Tensor, dict[str, object]]:
    if weight.ndim != 2:
        raise ValueError("Qronos pruning weight must be a 2D matrix")

    original = weight.detach().float()
    if original.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    rows, columns = original.shape
    if tuple(hessian.shape) != (columns, columns):
        raise ValueError(f"hessian shape {tuple(hessian.shape)} does not match input columns {columns}")
    if tuple(cross.shape) != (columns, columns):
        raise ValueError(f"cross shape {tuple(cross.shape)} does not match input columns {columns}")

    h = hessian.detach().to(device=original.device, dtype=torch.float32)
    g = cross.detach().to(device=original.device, dtype=torch.float32)
    if use_activation_order:
        order = torch.argsort(torch.diag(h), descending=True)
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(columns, dtype=order.dtype, device=order.device)
        h = h.index_select(0, order).index_select(1, order)
        g = g.index_select(0, order).index_select(1, order)
        working = original.index_select(1, order).clone()
    else:
        inv_order = None
        working = original.clone()

    prune_mask = _base_precision_pruning_mask(working, fraction=prune_fraction, pruning_scope=pruning_scope).to(
        device=working.device
    )
    pruned = torch.empty_like(working)

    def prune_current_column(column: int) -> torch.Tensor:
        values = working[:, column].clone()
        return values.masked_fill(prune_mask[:, column], 0.0)

    dh = torch.diag(h)
    dhi = torch.where(dh != 0, 1.0 / dh, torch.zeros_like(dh))
    uh = torch.triu(h, diagonal=1)
    h_damped, damp = _stabilized_hessian(h, percdamp)
    h_inv = torch.cholesky_inverse(torch.linalg.cholesky(h_damped))
    cholesky_jitter = 0.0

    first_proxy = (working @ (g[:, 0] * dhi[0])) - (working @ (uh[0, :] * dhi[0]))
    working[:, 0] = first_proxy
    pruned[:, 0] = prune_current_column(0)

    if columns > 1:
        tail_inv = h_inv[1:, 1:].clone()
        c0 = h_inv[0, 0].clamp_min(1e-12)
        b0 = h_inv[1:, [0]]
        tail_inv -= (b0 @ b0.T) / c0

        gh = g + torch.eye(columns, dtype=g.dtype, device=g.device) * float(damp)
        future = original if not use_activation_order else original.index_select(1, order)
        working[:, 1:] = future @ (gh[:, 1:] @ tail_inv)
        working[:, 1:] -= pruned[:, :1] @ (h[:1, 1:] @ tail_inv)

        scaled_inv = tail_inv * float(cholesky_scale)
        chol, cholesky_jitter = _upper_cholesky_with_jitter(scaled_inv)
        chol = chol / (float(cholesky_scale) ** 0.5)
        blocksize = max(1, (columns + max(1, int(num_blocks)) - 1) // max(1, int(num_blocks)))
        for start in range(1, columns, blocksize):
            end = min(start + blocksize, columns)
            count = end - start
            error_block = torch.zeros(rows, count, dtype=working.dtype, device=working.device)
            h_inv_block = chol[start - 1 : end - 1, start - 1 : end - 1]
            for idx in range(count):
                column = start + idx
                pruned[:, column] = prune_current_column(column)
                denom = h_inv_block[idx, idx].clamp_min(1e-12)
                error = (working[:, column] - pruned[:, column]) / denom
                error_block[:, idx] = error
                working[:, column:end] -= error[:, None] @ h_inv_block[idx, idx:][None, :]
            if end < columns:
                working[:, end:] -= error_block @ chol[start - 1 : end - 1, end - 1 :]

    if inv_order is not None:
        pruned = pruned.index_select(1, inv_order)
        prune_mask = prune_mask.index_select(1, inv_order)

    diff = original - pruned
    zeroed = int(prune_mask.sum().item())
    return pruned.to(dtype=weight.dtype), {
        "format": "base_precision_pruned",
        "rows": rows,
        "columns": columns,
        "weights": int(original.numel()),
        "zeroed": zeroed,
        "actual_zero_fraction": zeroed / max(int(original.numel()), 1),
        "prune_fraction": float(prune_fraction),
        "pruning_scope": pruning_scope,
        "damp": float(damp),
        "percdamp": float(percdamp),
        "cholesky_scale": float(cholesky_scale),
        "cholesky_jitter": float(cholesky_jitter),
        "num_blocks": int(num_blocks),
        "activation_order": bool(use_activation_order),
        "mean_abs_delta": float(diff.abs().mean().item()),
        "max_abs_delta": float(diff.abs().max().item()),
    }


class _SingleLinearInputHook:
    def __init__(self, module: torch.nn.Linear, *, use_attention_mask: bool):
        self.use_attention_mask = use_attention_mask
        self.attention_mask: torch.Tensor | None = None
        self.batch_examples = 0
        self.value: torch.Tensor | None = None
        self.handle = module.register_forward_pre_hook(self._hook)

    def close(self) -> None:
        self.handle.remove()

    def _hook(self, _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        x = inputs[0].detach()
        if (
            self.use_attention_mask
            and x.ndim == 3
            and self.attention_mask is not None
            and tuple(x.shape[:2]) == tuple(self.attention_mask.shape)
        ):
            flat = x[self.attention_mask].float()
        else:
            flat = x.reshape(-1, x.shape[-1]).float()
        self.batch_examples = int(x.shape[0]) if x.ndim > 1 else 1
        self.value = flat.contiguous().clone()


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


def _load_records(config: QronosConfig, split: str, max_examples: int) -> list[dict[str, Any]]:
    dataset = load_dataset(config.dataset_name, config.dataset_config, split=split)
    limit = min(max_examples, len(dataset)) if max_examples > 0 else len(dataset)
    return [dict(row) for row in dataset.select(range(limit))]


def collect_qronos_pair_stats(
    original_model: torch.nn.Module,
    quantized_model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: QronosConfig,
    device: torch.device,
    module_name: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    original_modules = dict(linear_modules(original_model))
    quantized_modules = dict(linear_modules(quantized_model))
    original_module = original_modules[module_name]
    quantized_module = quantized_modules[module_name]

    features = quantized_module.in_features
    hessian = torch.zeros(features, features, dtype=torch.float32, device=device)
    cross = torch.zeros(features, features, dtype=torch.float32, device=device)
    tokens = 0
    nsamples = 0
    original_hook = _SingleLinearInputHook(original_module, use_attention_mask=config.use_attention_mask)
    quantized_hook = _SingleLinearInputHook(quantized_module, use_attention_mask=config.use_attention_mask)
    try:
        for record_batch in tqdm(list(batched(records, config.batch_size)), desc=f"qronos_stats_{module_name}", unit="batch"):
            batch = build_causal_lm_batch(
                tokenizer,
                record_batch,
                config.max_length,
                answer_only_loss=config.answer_only_loss,
                device=device,
            )
            mask = batch["attention_mask"].detach().bool() if config.use_attention_mask else None
            original_hook.attention_mask = mask
            quantized_hook.attention_mask = mask
            with torch.inference_mode():
                original_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
                quantized_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            if original_hook.value is None or quantized_hook.value is None:
                raise RuntimeError(f"failed to collect Qronos inputs for {module_name}")
            x = original_hook.value.to(device=device, dtype=torch.float32)
            x_tilde = quantized_hook.value.to(device=device, dtype=torch.float32)
            if x.shape != x_tilde.shape:
                raise RuntimeError(f"Qronos input shape mismatch for {module_name}: {tuple(x.shape)} != {tuple(x_tilde.shape)}")
            batch_examples = max(1, int(quantized_hook.batch_examples))
            nsamples += batch_examples
            hessian.mul_((nsamples - batch_examples) / nsamples)
            cross.mul_((nsamples - batch_examples) / nsamples)
            hessian.add_((x_tilde.T @ x_tilde) / nsamples)
            cross.add_((x.T @ x_tilde) / nsamples)
            tokens += int(x.shape[0])
    finally:
        original_hook.close()
        quantized_hook.close()
    return hessian, cross, tokens


def apply_qronos_weight_only_(
    original_model: torch.nn.Module,
    quantized_model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: QronosConfig,
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    modules = linear_modules(quantized_model)
    if not config.quantize_last_layer:
        modules = [(name, module) for name, module in modules if name not in {"lm_head", "embed_out"}]
    for name, module in tqdm(modules, desc="qronos_quant", unit="layer"):
        hessian, cross, tokens = collect_qronos_pair_stats(
            original_model,
            quantized_model,
            tokenizer,
            records,
            config,
            device,
            name,
        )
        quantized, stats = qronos_quantize_weight(
            module.weight,
            hessian,
            cross,
            bits=config.weight_bits,
            beta=config.beta,
            percdamp=config.percdamp,
            cholesky_scale=config.cholesky_scale,
            num_blocks=config.num_blocks,
            use_activation_order=config.use_activation_order,
        )
        with torch.no_grad():
            module.weight.copy_(quantized.to(device=module.weight.device, dtype=module.weight.dtype))
        rows.append(
            {
                "name": name,
                "shape": list(module.weight.shape),
                "tokens": tokens,
                **stats,
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def apply_qronos_pruning_(
    original_model: torch.nn.Module,
    pruned_model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: QronosConfig,
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    linear_by_name = dict(linear_modules(pruned_model))
    parameters_by_name = dict(pruned_model.named_parameters())
    for name in tqdm(qronos_pruning_target_names(pruned_model), desc="qronos_prune", unit="tensor"):
        param = parameters_by_name[name]
        module_name = name.removesuffix(".weight") if name.endswith(".weight") else ""
        module = linear_by_name.get(module_name)
        if module is not None and module.weight is param:
            hessian, cross, tokens = collect_qronos_pair_stats(
                original_model,
                pruned_model,
                tokenizer,
                records,
                config,
                device,
                module_name,
            )
            pruned, stats = qronos_prune_weight(
                module.weight,
                hessian,
                cross,
                prune_fraction=config.prune_fraction,
                pruning_scope=config.pruning_scope,
                percdamp=config.percdamp,
                cholesky_scale=config.cholesky_scale,
                num_blocks=config.num_blocks,
                use_activation_order=config.use_activation_order,
            )
            with torch.no_grad():
                module.weight.copy_(pruned.to(device=module.weight.device, dtype=module.weight.dtype))
        else:
            tokens = 0
            stats = _prune_base_precision_magnitude_fallback(
                param,
                prune_fraction=config.prune_fraction,
                pruning_scope=config.pruning_scope,
            )
        rows.append(
            {
                "name": name,
                "shape": list(param.shape),
                "tokens": tokens,
                **stats,
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def run_qronos_weight_only_experiment(config: QronosConfig) -> dict[str, object]:
    started = time.time()
    device = resolve_device(config.device)
    dtype = resolve_torch_dtype(config.dtype)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    tokenizer = _prepare_tokenizer(config.model_name, config.revision)
    calibration_records = _load_records(config, config.split, config.max_examples)
    if config.eval_split or config.max_eval_examples > 0:
        eval_records = _load_records(
            config,
            config.eval_split or config.split,
            config.max_eval_examples if config.max_eval_examples > 0 else config.max_examples,
        )
    else:
        eval_records = calibration_records

    original_model = _prepare_model(config.model_name, config.revision, dtype, device)
    quantized_model = _prepare_model(config.model_name, config.revision, dtype, device)
    baseline = evaluate_perplexity(original_model, tokenizer, eval_records, config, device, desc="baseline_ppl")
    layers = apply_qronos_weight_only_(original_model, quantized_model, tokenizer, calibration_records, config, device)
    quantized = evaluate_perplexity(quantized_model, tokenizer, eval_records, config, device, desc="qronos_ppl")

    ppl_change = summarize_ppl_change(baseline, quantized)
    ppl_change["quantized"] = ppl_change.pop("pruned")
    result = {
        "metadata": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "device": str(device),
            "torch_dtype": str(dtype),
            "elapsed_seconds": time.time() - started,
            "calibration_examples": len(calibration_records),
            "eval_examples": len(eval_records),
            "quantization_method": "qronos_weight_only_asymmetric_minmax",
        },
        "quantization": {
            "format": f"asymmetric_int{config.weight_bits}",
            "layers_quantized": len(layers),
            "quantized_weights": sum(int(row["weights"]) for row in layers),
            "beta": float(config.beta),
            "percdamp": float(config.percdamp),
            "cholesky_scale": float(config.cholesky_scale),
            "num_blocks": int(config.num_blocks),
            "use_activation_order": bool(config.use_activation_order),
            "use_attention_mask": bool(config.use_attention_mask),
            "quantize_last_layer": bool(config.quantize_last_layer),
            "layers": layers,
        },
        "ppl_change": ppl_change,
    }

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qronos_weight_only_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "qronos_layers.jsonl").open("w") as handle:
        for row in layers:
            handle.write(json.dumps(row) + "\n")
    return result


def run_qronos_prune_experiment(config: QronosConfig) -> dict[str, object]:
    started = time.time()
    device = resolve_device(config.device)
    dtype = resolve_torch_dtype(config.dtype)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    tokenizer = _prepare_tokenizer(config.model_name, config.revision)
    calibration_records = _load_records(config, config.split, config.max_examples)
    if config.eval_split or config.max_eval_examples > 0:
        eval_records = _load_records(
            config,
            config.eval_split or config.split,
            config.max_eval_examples if config.max_eval_examples > 0 else config.max_examples,
        )
    else:
        eval_records = calibration_records

    original_model = _prepare_model(config.model_name, config.revision, dtype, device)
    pruned_model = _prepare_model(config.model_name, config.revision, dtype, device)
    baseline = evaluate_perplexity(original_model, tokenizer, eval_records, config, device, desc="baseline_ppl")
    layers = apply_qronos_pruning_(original_model, pruned_model, tokenizer, calibration_records, config, device)
    pruned = evaluate_perplexity(pruned_model, tokenizer, eval_records, config, device, desc="qronos_pruned_ppl")

    weights_seen = sum(int(row["weights"]) for row in layers)
    weights_zeroed = sum(int(row["zeroed"]) for row in layers)
    result = {
        "metadata": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "device": str(device),
            "torch_dtype": str(dtype),
            "elapsed_seconds": time.time() - started,
            "calibration_examples": len(calibration_records),
            "eval_examples": len(eval_records),
            "method": "qronos_base_precision_pruning",
        },
        "pruning": {
            "pruning_scope": config.pruning_scope,
            "prune_fraction": float(config.prune_fraction),
            "matrix_tensors_pruned": len(layers),
            "weights_seen": weights_seen,
            "weights_zeroed": weights_zeroed,
            "actual_zero_fraction": weights_zeroed / max(weights_seen, 1),
            "use_activation_order": bool(config.use_activation_order),
            "use_attention_mask": bool(config.use_attention_mask),
            "quantize_last_layer": bool(config.quantize_last_layer),
            "layers": layers,
        },
        "ppl_change": summarize_ppl_change(baseline, pruned),
    }

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qronos_prune_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "qronos_pruned_layers.jsonl").open("w") as handle:
        for row in layers:
            handle.write(json.dumps(row) + "\n")
    return result
