from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .experiment import GpuStatsSampler, JsonlStepLogger, _extract_layers, _torch_dtype
from .hybrid_attention import _causal_lm_backbone, _load_causal_lm
from .low_qk_model import (
    _load_wikitext_eval_tokens,
    _make_eval_batches,
    causal_lm_nll_from_logits,
    perplexity_from_nll,
)


@dataclass(frozen=True)
class AttentionBasisPPLConfig:
    output_dir: Path
    model_name: str = "EleutherAI/pythia-1.4b"
    basis_sizes: tuple[int, ...] = (2, 4, 8)
    nmf_iterations: int = 8
    eval_steps: int = 8
    batch_size: int = 4
    seq_len: int = 128
    seed: int = 17
    data_split: str = "test"
    max_dataset_tokens: int = 2_000_000
    dtype: str = "bf16"
    ce_chunk_tokens: int = 32768
    include_baseline: bool = True
    log_gpu_stats: bool = True
    combine_mode: str = "linear"
    basis_quantization_bits: int = 0
    basis_quantization_format: str = ""
    basis_quantization_target: str = "reconstructed"

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        basis_sizes = tuple(sorted({int(size) for size in self.basis_sizes}))
        object.__setattr__(self, "basis_sizes", basis_sizes)
        if not basis_sizes:
            raise ValueError("basis_sizes must be non-empty")
        if any(size <= 0 for size in basis_sizes):
            raise ValueError("basis_sizes must be positive")
        if self.nmf_iterations <= 0:
            raise ValueError("nmf_iterations must be positive")
        if self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 1:
            raise ValueError("seq_len must be greater than 1")
        if self.max_dataset_tokens <= self.seq_len:
            raise ValueError("max_dataset_tokens must be greater than seq_len")
        if self.data_split not in {"train", "validation", "test"}:
            raise ValueError("data_split must be train, validation, or test")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be bf16, fp16, or fp32")
        if self.ce_chunk_tokens <= 0:
            raise ValueError("ce_chunk_tokens must be positive")
        if self.combine_mode not in {"linear", "exponential"}:
            raise ValueError("combine_mode must be linear or exponential")
        if self.basis_quantization_bits < 0:
            raise ValueError("basis_quantization_bits must be non-negative")
        object.__setattr__(self, "basis_quantization_format", _normalize_quantization_format(self.basis_quantization_format))
        if self.basis_quantization_bits and self.basis_quantization_format:
            raise ValueError("only one of basis_quantization_bits or basis_quantization_format may be set")
        if self.basis_quantization_target not in {"reconstructed", "factors"}:
            raise ValueError("basis_quantization_target must be reconstructed or factors")


@dataclass(frozen=True)
class AttentionBasisLayerGroupEvalConfig:
    output_dir: Path
    model_name: str = "EleutherAI/pythia-1.4b"
    basis_size: int = 8
    nmf_iterations: int = 6
    layer_groups: tuple[tuple[int, ...], ...] = ((),)
    eval_steps: int = 8
    batch_size: int = 4
    seq_len: int = 128
    seed: int = 17
    data_split: str = "test"
    max_dataset_tokens: int = 2_000_000
    dtype: str = "bf16"
    ce_chunk_tokens: int = 32768
    log_gpu_stats: bool = True
    combine_mode: str = "linear"
    basis_quantization_bits: int = 0
    basis_quantization_format: str = ""
    basis_quantization_target: str = "reconstructed"

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        groups = tuple(tuple(sorted({int(layer) for layer in group})) for group in self.layer_groups)
        object.__setattr__(self, "layer_groups", groups)
        if self.basis_size <= 0:
            raise ValueError("basis_size must be positive")
        if self.nmf_iterations <= 0:
            raise ValueError("nmf_iterations must be positive")
        if not groups:
            raise ValueError("layer_groups must be non-empty")
        if any(any(layer < 0 for layer in group) for group in groups):
            raise ValueError("layer_groups must be non-negative")
        if self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 1:
            raise ValueError("seq_len must be greater than 1")
        if self.max_dataset_tokens <= self.seq_len:
            raise ValueError("max_dataset_tokens must be greater than seq_len")
        if self.data_split not in {"train", "validation", "test"}:
            raise ValueError("data_split must be train, validation, or test")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be bf16, fp16, or fp32")
        if self.ce_chunk_tokens <= 0:
            raise ValueError("ce_chunk_tokens must be positive")
        if self.combine_mode not in {"linear", "exponential"}:
            raise ValueError("combine_mode must be linear or exponential")
        if self.basis_quantization_bits < 0:
            raise ValueError("basis_quantization_bits must be non-negative")
        object.__setattr__(self, "basis_quantization_format", _normalize_quantization_format(self.basis_quantization_format))
        if self.basis_quantization_bits and self.basis_quantization_format:
            raise ValueError("only one of basis_quantization_bits or basis_quantization_format may be set")
        if self.basis_quantization_target not in {"reconstructed", "factors"}:
            raise ValueError("basis_quantization_target must be reconstructed or factors")


_SUPPORTED_QUANTIZATION_FORMATS = {
    "",
    "int8",
    "int4",
    "uint8",
    "uint4",
    "fp8_e4m3",
    "fp8_e5m2",
    "nvfp4",
    "mxfp4",
    "nf4",
}


def _normalize_quantization_format(value: str | None) -> str:
    fmt = (value or "").strip().lower().replace("-", "_")
    if fmt == "none":
        fmt = ""
    if fmt not in _SUPPORTED_QUANTIZATION_FORMATS:
        raise ValueError(f"basis_quantization_format must be one of {sorted(_SUPPORTED_QUANTIZATION_FORMATS - {''})}")
    return fmt


def _uniform_quantize_nonnegative(values: torch.Tensor, *, bits: int, eps: float, normalize_rows: bool) -> torch.Tensor:
    if bits < 0:
        raise ValueError("quantization_bits must be non-negative")
    if bits == 0:
        return values
    levels = float((1 << bits) - 1)
    if levels <= 0:
        return values
    row_max = values.amax(dim=-1, keepdim=True)
    scale = (row_max / levels).clamp_min(eps)
    quantized = torch.round(values / scale) * scale
    quantized = torch.where(row_max > eps, quantized, values)
    if normalize_rows:
        quantized = quantized / quantized.sum(dim=-1, keepdim=True).clamp_min(eps)
    return quantized


def _float_codebook(*, exponent_bits: int, mantissa_bits: int, bias: int, include_max_exponent: bool = True) -> tuple[float, ...]:
    values = {0.0}
    mantissa_count = 1 << mantissa_bits
    for mantissa in range(1, mantissa_count):
        values.add((mantissa / mantissa_count) * (2.0 ** (1 - bias)))
    max_exponent_code = (1 << exponent_bits) - 1
    if not include_max_exponent:
        max_exponent_code -= 1
    for exponent_code in range(1, max_exponent_code + 1):
        exponent = exponent_code - bias
        for mantissa in range(mantissa_count):
            values.add((1.0 + mantissa / mantissa_count) * (2.0**exponent))
    return tuple(sorted(values))


def _nearest_codebook(values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    indices = torch.bucketize(values, codebook)
    lower_indices = (indices - 1).clamp(0, codebook.numel() - 1)
    upper_indices = indices.clamp(0, codebook.numel() - 1)
    lower = codebook[lower_indices]
    upper = codebook[upper_indices]
    use_upper = (upper - values).abs() < (values - lower).abs()
    return torch.where(use_upper, upper, lower)


def _codebook_quantize_scaled(
    values: torch.Tensor,
    *,
    codebook_values: tuple[float, ...],
    eps: float,
    normalize_rows: bool,
    block_size: int | None = None,
    power_of_two_scale: bool = False,
) -> torch.Tensor:
    codebook = values.new_tensor(codebook_values)
    max_code = float(codebook[-1])
    if block_size is None:
        row_max = values.amax(dim=-1, keepdim=True)
        scale = (row_max / max_code).clamp_min(eps)
        scaled = values / scale
        quantized = _nearest_codebook(scaled, codebook) * scale
        quantized = torch.where(row_max > eps, quantized, values)
        if normalize_rows:
            quantized = quantized / quantized.sum(dim=-1, keepdim=True).clamp_min(eps)
        return quantized

    original_width = values.shape[-1]
    pad = (-original_width) % block_size
    if pad:
        work = torch.nn.functional.pad(values, (0, pad))
    else:
        work = values
    blocked = work.reshape(*work.shape[:-1], work.shape[-1] // block_size, block_size)
    block_max = blocked.amax(dim=-1, keepdim=True)
    scale = (block_max / max_code).clamp_min(eps)
    if power_of_two_scale:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale.clamp_min(eps))))
    scaled = blocked / scale
    quantized = _nearest_codebook(scaled, codebook) * scale
    quantized = torch.where(block_max > eps, quantized, blocked)
    quantized = quantized.reshape(*work.shape)
    if pad:
        quantized = quantized[..., :original_width]
    if normalize_rows:
        quantized = quantized / quantized.sum(dim=-1, keepdim=True).clamp_min(eps)
    return quantized


def _format_quantize_nonnegative(values: torch.Tensor, *, fmt: str, eps: float, normalize_rows: bool) -> torch.Tensor:
    if fmt in {"int8", "uint8"}:
        return _uniform_quantize_nonnegative(values, bits=8, eps=eps, normalize_rows=normalize_rows)
    if fmt in {"int4", "uint4"}:
        return _uniform_quantize_nonnegative(values, bits=4, eps=eps, normalize_rows=normalize_rows)
    if fmt == "fp8_e4m3":
        return _codebook_quantize_scaled(
            values,
            codebook_values=_float_codebook(exponent_bits=4, mantissa_bits=3, bias=7),
            eps=eps,
            normalize_rows=normalize_rows,
        )
    if fmt == "fp8_e5m2":
        return _codebook_quantize_scaled(
            values,
            codebook_values=_float_codebook(exponent_bits=5, mantissa_bits=2, bias=15, include_max_exponent=False),
            eps=eps,
            normalize_rows=normalize_rows,
        )
    if fmt == "nvfp4":
        return _codebook_quantize_scaled(
            values,
            codebook_values=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
            eps=eps,
            normalize_rows=normalize_rows,
            block_size=16,
        )
    if fmt == "mxfp4":
        return _codebook_quantize_scaled(
            values,
            codebook_values=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
            eps=eps,
            normalize_rows=normalize_rows,
            block_size=32,
            power_of_two_scale=True,
        )
    if fmt == "nf4":
        return _codebook_quantize_scaled(
            values,
            codebook_values=(0.0, 0.0795803, 0.1609302, 0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1.0),
            eps=eps,
            normalize_rows=normalize_rows,
            block_size=64,
        )
    raise ValueError("quantization_format must be supported")


def _quantize_nonnegative(values: torch.Tensor, *, bits: int, fmt: str, eps: float, normalize_rows: bool) -> torch.Tensor:
    fmt = _normalize_quantization_format(fmt)
    if bits and fmt:
        raise ValueError("only one of quantization_bits or quantization_format may be set")
    if fmt:
        return _format_quantize_nonnegative(values, fmt=fmt, eps=eps, normalize_rows=normalize_rows)
    return _uniform_quantize_nonnegative(values, bits=bits, eps=eps, normalize_rows=normalize_rows)


def compress_attention_heads_nmf(
    attention_weights: torch.Tensor,
    *,
    basis_size: int,
    iterations: int = 8,
    combine_mode: str = "linear",
    quantization_bits: int = 0,
    quantization_format: str = "",
    quantization_target: str = "reconstructed",
    eps: float = 1e-12,
) -> torch.Tensor:
    if attention_weights.ndim != 4:
        raise ValueError("attention_weights must have shape [batch, heads, query, key]")
    if basis_size <= 0:
        raise ValueError("basis_size must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if combine_mode not in {"linear", "exponential"}:
        raise ValueError("combine_mode must be linear or exponential")
    if quantization_bits < 0:
        raise ValueError("quantization_bits must be non-negative")
    quantization_format = _normalize_quantization_format(quantization_format)
    if quantization_bits and quantization_format:
        raise ValueError("only one of quantization_bits or quantization_format may be set")
    if quantization_target not in {"reconstructed", "factors"}:
        raise ValueError("quantization_target must be reconstructed or factors")
    batch, heads, query_len, key_len = attention_weights.shape
    if basis_size >= heads:
        raise ValueError("basis_size must be smaller than the number of heads")

    source = attention_weights.float().clamp_min(0.0)
    support = source.sum(dim=1, keepdim=True) > eps
    outputs = []
    source_indices = torch.linspace(0, heads - 1, steps=basis_size, device=source.device).round().long().unique(sorted=True)
    if source_indices.numel() < basis_size:
        extra = torch.arange(heads, device=source.device)
        source_indices = torch.cat([source_indices, extra])[:basis_size]
    source_indices = source_indices[:basis_size]

    for batch_idx in range(batch):
        target = source[batch_idx].reshape(heads, query_len * key_len)
        valid = support[batch_idx, 0].reshape(query_len * key_len)
        basis = target.index_select(0, source_indices).clone()
        basis = torch.where(valid.view(1, -1), basis.clamp_min(eps), torch.zeros_like(basis))
        similarity = target @ basis.T
        nearest = similarity.argmax(dim=-1)
        coeffs = target.new_full((heads, basis_size), eps)
        coeffs.scatter_(1, nearest.view(-1, 1), 1.0)
        for basis_idx, head_idx in enumerate(source_indices.tolist()):
            coeffs[head_idx].fill_(eps)
            coeffs[head_idx, basis_idx] = 1.0
        for _ in range(iterations):
            recon = coeffs @ basis
            coeffs = coeffs * ((target @ basis.T) / ((recon @ basis.T).clamp_min(eps)))
            recon = coeffs @ basis
            basis = basis * ((coeffs.T @ target) / ((coeffs.T @ recon).clamp_min(eps)))
            basis = torch.where(valid.view(1, -1), basis, torch.zeros_like(basis))
        if quantization_target == "factors":
            basis = _quantize_nonnegative(basis, bits=quantization_bits, fmt=quantization_format, eps=eps, normalize_rows=False)
            basis = torch.where(valid.view(1, -1), basis, torch.zeros_like(basis))
            coeffs = _quantize_nonnegative(coeffs, bits=quantization_bits, fmt=quantization_format, eps=eps, normalize_rows=False)
        if combine_mode == "linear":
            reconstructed = (coeffs @ basis).view(heads, query_len, key_len)
        else:
            basis_rows = basis.view(basis_size, query_len, key_len)
            basis_rows = basis_rows / basis_rows.sum(dim=-1, keepdim=True).clamp_min(eps)
            target_rows = target.view(heads, query_len, key_len)
            log_basis_rows = basis_rows.clamp_min(eps).log()
            basis_norm = basis / basis.norm(dim=-1, keepdim=True).clamp_min(eps)
            target_norm = target / target.norm(dim=-1, keepdim=True).clamp_min(eps)
            theta = (target_norm @ basis_norm.T) / math.sqrt(float(query_len * key_len))
            for _ in range(iterations):
                mixture_weights = torch.softmax(theta, dim=-1)
                logits = torch.einsum("hk,kqv->hqv", mixture_weights, log_basis_rows)
                log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                probs = torch.exp(log_probs).masked_fill(~support[batch_idx], 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
                grad = torch.einsum("hqv,kqv->hk", probs - target_rows, log_basis_rows) / max(query_len, 1)
                theta = theta - grad
            mixture_weights = torch.softmax(theta, dim=-1)
            logits = torch.einsum("hk,kqv->hqv", mixture_weights, log_basis_rows)
            reconstructed = torch.exp(logits - torch.logsumexp(logits, dim=-1, keepdim=True))
        reconstructed = reconstructed.masked_fill(~support[batch_idx], 0.0)
        reconstructed = reconstructed / reconstructed.sum(dim=-1, keepdim=True).clamp_min(eps)
        if quantization_target == "reconstructed":
            reconstructed = _quantize_nonnegative(reconstructed, bits=quantization_bits, fmt=quantization_format, eps=eps, normalize_rows=True)
            reconstructed = reconstructed.masked_fill(~support[batch_idx], 0.0)
            reconstructed = reconstructed / reconstructed.sum(dim=-1, keepdim=True).clamp_min(eps)
        outputs.append(reconstructed)
    compressed = torch.stack(outputs, dim=0)
    return compressed.to(dtype=attention_weights.dtype)


class HeadBasisCompressedGPTNeoXAttention(torch.nn.Module):
    def __init__(
        self,
        source_attention: torch.nn.Module,
        *,
        basis_size: int,
        nmf_iterations: int,
        combine_mode: str = "linear",
        basis_quantization_bits: int = 0,
        basis_quantization_format: str = "",
        basis_quantization_target: str = "reconstructed",
    ):
        super().__init__()
        self.source_attention = source_attention
        self.basis_size = basis_size
        self.nmf_iterations = nmf_iterations
        if combine_mode not in {"linear", "exponential"}:
            raise ValueError("combine_mode must be linear or exponential")
        self.combine_mode = combine_mode
        if basis_quantization_bits < 0:
            raise ValueError("basis_quantization_bits must be non-negative")
        self.basis_quantization_format = _normalize_quantization_format(basis_quantization_format)
        if basis_quantization_bits and self.basis_quantization_format:
            raise ValueError("only one of basis_quantization_bits or basis_quantization_format may be set")
        self.basis_quantization_bits = basis_quantization_bits
        if basis_quantization_target not in {"reconstructed", "factors"}:
            raise ValueError("basis_quantization_target must be reconstructed or factors")
        self.basis_quantization_target = basis_quantization_target
        self.head_size = int(getattr(source_attention, "head_size"))
        dense = getattr(source_attention, "dense", None)
        if hasattr(source_attention, "num_attention_heads"):
            self.num_attention_heads = int(getattr(source_attention, "num_attention_heads"))
        elif dense is not None and hasattr(dense, "in_features"):
            self.num_attention_heads = int(dense.in_features // self.head_size)
        else:
            config = getattr(source_attention, "config", None)
            self.num_attention_heads = int(getattr(config, "num_attention_heads"))
        if basis_size >= self.num_attention_heads:
            raise ValueError("basis_size must be smaller than the number of heads")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_past: Any | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_past is not None:
            raise NotImplementedError("head-basis compression does not implement KV cache")
        from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, 3 * self.head_size)
        qkv = self.source_attention.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        scaling = float(getattr(self.source_attention, "scaling", self.head_size**-0.5))
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.training:
            dropout = float(getattr(self.source_attention, "attention_dropout", 0.0))
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=True)
        compressed_weights = compress_attention_heads_nmf(
            attn_weights,
            basis_size=self.basis_size,
            iterations=self.nmf_iterations,
            combine_mode=self.combine_mode,
            quantization_bits=self.basis_quantization_bits,
            quantization_format=self.basis_quantization_format,
            quantization_target=self.basis_quantization_target,
        )
        attn_output = torch.matmul(compressed_weights.to(dtype=value_states.dtype), value_states)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.source_attention.dense(attn_output), compressed_weights


def patch_attentions_with_head_basis(
    model: torch.nn.Module,
    *,
    basis_size: int,
    nmf_iterations: int,
    combine_mode: str = "linear",
    basis_quantization_bits: int = 0,
    basis_quantization_format: str = "",
    basis_quantization_target: str = "reconstructed",
    layer_indices: tuple[int, ...] | None = None,
) -> dict[int, torch.nn.Module]:
    layers = _extract_layers(_causal_lm_backbone(model))
    if layer_indices is None:
        layer_indices = tuple(range(len(layers)))
    originals: dict[int, torch.nn.Module] = {}
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"layer index out of range: {layer_idx}")
        layer = layers[layer_idx]
        originals[layer_idx] = layer.attention
        layer.attention = HeadBasisCompressedGPTNeoXAttention(
            layer.attention,
            basis_size=basis_size,
            nmf_iterations=nmf_iterations,
            combine_mode=combine_mode,
            basis_quantization_bits=basis_quantization_bits,
            basis_quantization_format=basis_quantization_format,
            basis_quantization_target=basis_quantization_target,
        )
    return originals


def restore_attentions(model: torch.nn.Module, originals: dict[int, torch.nn.Module]) -> None:
    layers = _extract_layers(_causal_lm_backbone(model))
    for layer_idx, attention in originals.items():
        layers[layer_idx].attention = attention


def build_next_layer_candidates(*, frontier: list[tuple[int, ...]] | tuple[tuple[int, ...], ...], layer_count: int) -> tuple[tuple[int, ...], ...]:
    candidates: set[tuple[int, ...]] = set()
    for group in frontier:
        selected = set(group)
        for layer_idx in range(layer_count):
            if layer_idx not in selected:
                candidates.add(tuple(sorted((*group, layer_idx))))
    return tuple(sorted(candidates, key=lambda group: (len(group), group)))


def select_top_layer_groups(rows: list[dict[str, Any]], *, beam_width: int) -> tuple[tuple[int, ...], ...]:
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    ranked = sorted(rows, key=lambda row: (float(row["ppl"]), tuple(row.get("layer_group", ()))))
    return tuple(tuple(int(layer) for layer in row.get("layer_group", ())) for row in ranked[:beam_width])


def run_attention_basis_ppl_eval(config: AttentionBasisPPLConfig) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(config.dtype)
    model_dtype = dtype if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokens = _load_wikitext_eval_tokens(tokenizer=tokenizer, split=config.data_split, max_tokens=config.max_dataset_tokens)
    batches = _make_eval_batches(tokens, batch_size=config.batch_size, seq_len=config.seq_len, max_steps=config.eval_steps)
    if not batches:
        raise ValueError("not enough tokens to build an evaluation batch")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True, default=str) + "\n")
    runs: list[dict[str, Any]] = []

    if config.include_baseline:
        baseline = _load_causal_lm(AutoModelForCausalLM, config.model_name, model_dtype).to(device)
        baseline.eval()
        runs.append(
            _evaluate_model_ppl(
                model=baseline,
                batches=batches,
                output_dir=config.output_dir / "baseline",
                run_name="baseline",
                device=device,
                dtype=dtype,
                config=config,
            )
        )
        del baseline
        _empty_cache(device)

    for basis_size in config.basis_sizes:
        model = _load_causal_lm(AutoModelForCausalLM, config.model_name, model_dtype).to(device)
        model.eval()
        originals = patch_attentions_with_head_basis(
            model,
            basis_size=basis_size,
            nmf_iterations=config.nmf_iterations,
            combine_mode=config.combine_mode,
            basis_quantization_bits=config.basis_quantization_bits,
            basis_quantization_format=config.basis_quantization_format,
            basis_quantization_target=config.basis_quantization_target,
        )
        try:
            record = _evaluate_model_ppl(
                model=model,
                batches=batches,
                output_dir=config.output_dir / f"basis_{basis_size}",
                run_name=f"basis_{basis_size}",
                device=device,
                dtype=dtype,
                config=config,
            )
            record["basis_size"] = basis_size
            record["nmf_iterations"] = config.nmf_iterations
            record["combine_mode"] = config.combine_mode
            record["basis_quantization_bits"] = config.basis_quantization_bits
            record["basis_quantization_format"] = config.basis_quantization_format
            record["basis_quantization_target"] = config.basis_quantization_target
            runs.append(record)
        finally:
            restore_attentions(model, originals)
            del model
            _empty_cache(device)

    return summarize_attention_basis_ppl(output_dir=config.output_dir, runs=runs)


def evaluate_attention_basis_layer_groups(config: AttentionBasisLayerGroupEvalConfig) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(config.dtype)
    model_dtype = dtype if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokens = _load_wikitext_eval_tokens(tokenizer=tokenizer, split=config.data_split, max_tokens=config.max_dataset_tokens)
    batches = _make_eval_batches(tokens, batch_size=config.batch_size, seq_len=config.seq_len, max_steps=config.eval_steps)
    if not batches:
        raise ValueError("not enough tokens to build an evaluation batch")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True, default=str) + "\n")
    model = _load_causal_lm(AutoModelForCausalLM, config.model_name, model_dtype).to(device)
    model.eval()
    layer_count = len(_extract_layers(_causal_lm_backbone(model)))
    runs: list[dict[str, Any]] = []
    try:
        for group in config.layer_groups:
            if group:
                run_name = "layers_" + "_".join(f"{layer_idx:02d}" for layer_idx in group)
                originals = patch_attentions_with_head_basis(
                    model,
                    basis_size=config.basis_size,
                    nmf_iterations=config.nmf_iterations,
                    combine_mode=config.combine_mode,
                    basis_quantization_bits=config.basis_quantization_bits,
                    basis_quantization_format=config.basis_quantization_format,
                    basis_quantization_target=config.basis_quantization_target,
                    layer_indices=group,
                )
            else:
                run_name = "baseline"
                originals = {}
            try:
                record = _evaluate_layer_group_model_ppl(
                    model=model,
                    batches=batches,
                    output_dir=config.output_dir / run_name,
                    run_name=run_name,
                    device=device,
                    dtype=dtype,
                    config=config,
                )
                record["basis_size"] = config.basis_size if group else None
                record["nmf_iterations"] = config.nmf_iterations if group else None
                record["combine_mode"] = config.combine_mode if group else None
                record["basis_quantization_bits"] = config.basis_quantization_bits if group else None
                record["basis_quantization_format"] = config.basis_quantization_format if group else None
                record["basis_quantization_target"] = config.basis_quantization_target if group else None
                record["layer_group"] = list(group)
                record["compressed_layers"] = len(group)
                record["layer_count"] = layer_count
                runs.append(record)
            finally:
                restore_attentions(model, originals)
    finally:
        del model
        _empty_cache(device)
    summary = {"layer_count": layer_count, "runs": runs}
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary


def _evaluate_model_ppl(
    *,
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    output_dir: Path,
    run_name: str,
    device: torch.device,
    dtype: torch.dtype,
    config: AttentionBasisPPLConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_nll = 0.0
    total_tokens = 0
    start_time = time.monotonic()
    sampler = GpuStatsSampler() if config.log_gpu_stats and device.type == "cuda" else None
    if sampler is not None:
        sampler.start()
    try:
        with JsonlStepLogger(output_dir / "steps.jsonl") as logger:
            with torch.inference_mode():
                for step, batch in enumerate(batches, start=1):
                    input_ids = batch.to(device=device, non_blocking=True)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    step_start = time.monotonic()
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                        output = model(input_ids=input_ids, use_cache=False)
                    step_nll, label_tokens = causal_lm_nll_from_logits(
                        output.logits,
                        input_ids,
                        chunk_tokens=config.ce_chunk_tokens,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    step_end = time.monotonic()
                    total_nll += float(step_nll.detach().float().cpu())
                    total_tokens += label_tokens
                    record = {
                        "step": step,
                        "run_name": run_name,
                        "loss": float((step_nll / label_tokens).detach().float().cpu()),
                        "ppl_so_far": perplexity_from_nll(total_nll=total_nll, total_tokens=total_tokens),
                        "tokens": label_tokens,
                        "total_tokens": total_tokens,
                        "elapsed_sec": max(step_end - step_start, 1e-12),
                        "tokens_per_sec": float(label_tokens / max(step_end - step_start, 1e-12)),
                    }
                    if sampler is not None:
                        record.update(sampler.stats_between(step_start, step_end))
                    logger.log(record)
                    print(json.dumps(record, sort_keys=True), flush=True)
                    del output
    finally:
        if sampler is not None:
            sampler.stop()
    elapsed = max(time.monotonic() - start_time, 1e-12)
    result = {
        "run_name": run_name,
        "loss": total_nll / total_tokens,
        "ppl": perplexity_from_nll(total_nll=total_nll, total_tokens=total_tokens),
        "total_tokens": total_tokens,
        "steps": len(batches),
        "elapsed_sec": elapsed,
        "tokens_per_sec": float(total_tokens / elapsed),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _evaluate_layer_group_model_ppl(
    *,
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    output_dir: Path,
    run_name: str,
    device: torch.device,
    dtype: torch.dtype,
    config: AttentionBasisLayerGroupEvalConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_nll = 0.0
    total_tokens = 0
    start_time = time.monotonic()
    sampler = GpuStatsSampler() if config.log_gpu_stats and device.type == "cuda" else None
    if sampler is not None:
        sampler.start()
    try:
        with JsonlStepLogger(output_dir / "steps.jsonl") as logger:
            with torch.inference_mode():
                for step, batch in enumerate(batches, start=1):
                    input_ids = batch.to(device=device, non_blocking=True)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    step_start = time.monotonic()
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                        output = model(input_ids=input_ids, use_cache=False)
                    step_nll, label_tokens = causal_lm_nll_from_logits(output.logits, input_ids, chunk_tokens=config.ce_chunk_tokens)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    step_end = time.monotonic()
                    total_nll += float(step_nll.detach().float().cpu())
                    total_tokens += label_tokens
                    record = {
                        "step": step,
                        "run_name": run_name,
                        "loss": float((step_nll / label_tokens).detach().float().cpu()),
                        "ppl_so_far": perplexity_from_nll(total_nll=total_nll, total_tokens=total_tokens),
                        "tokens": label_tokens,
                        "total_tokens": total_tokens,
                        "elapsed_sec": max(step_end - step_start, 1e-12),
                        "tokens_per_sec": float(label_tokens / max(step_end - step_start, 1e-12)),
                    }
                    if sampler is not None:
                        record.update(sampler.stats_between(step_start, step_end))
                    logger.log(record)
                    print(json.dumps(record, sort_keys=True), flush=True)
                    del output
    finally:
        if sampler is not None:
            sampler.stop()
    elapsed = max(time.monotonic() - start_time, 1e-12)
    result = {
        "run_name": run_name,
        "loss": total_nll / total_tokens,
        "ppl": perplexity_from_nll(total_nll=total_nll, total_tokens=total_tokens),
        "total_tokens": total_tokens,
        "steps": len(batches),
        "elapsed_sec": elapsed,
        "tokens_per_sec": float(total_tokens / elapsed),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def summarize_attention_basis_ppl(*, output_dir: Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next((run for run in runs if run.get("run_name") == "baseline"), None)
    if baseline is None and runs:
        baseline = min(runs, key=lambda run: math.inf if run.get("ppl") is None else float(run["ppl"]))
    if baseline is not None:
        base_ppl = float(baseline["ppl"])
        for run in runs:
            run["ppl_ratio_vs_baseline"] = float(run["ppl"]) / base_ppl
            run["ppl_delta_vs_baseline"] = float(run["ppl"]) - base_ppl
    summary = {"baseline": baseline, "runs": runs}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    _write_summary_markdown(output_dir / "summary.md", summary)
    return summary


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Attention Basis Perplexity Eval",
        "",
        "| run | basis | combine | quant bits | quant format | quant target | ppl | ratio vs baseline | loss | tokens/sec |",
        "| --- | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for run in summary["runs"]:
        basis = run.get("basis_size", "")
        quant_bits = run.get("basis_quantization_bits", "")
        quant_format = run.get("basis_quantization_format", "")
        quant_target = run.get("basis_quantization_target", "")
        ratio = run.get("ppl_ratio_vs_baseline")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run["run_name"]),
                    str(basis),
                    str(run.get("combine_mode", "")),
                    str(quant_bits),
                    str(quant_format),
                    str(quant_target),
                    f"{float(run['ppl']):.6f}",
                    "" if ratio is None else f"{float(ratio):.6f}",
                    f"{float(run['loss']):.6f}",
                    "" if run.get("tokens_per_sec") is None else f"{float(run['tokens_per_sec']):.1f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def _empty_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _parse_basis_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dynamic attention-head basis compression perplexity.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/attention_basis_ppl"))
    parser.add_argument("--model-name", default="EleutherAI/pythia-1.4b")
    parser.add_argument("--basis-sizes", default="2,4,8")
    parser.add_argument("--nmf-iterations", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--data-split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max-dataset-tokens", type=int, default=2_000_000)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--ce-chunk-tokens", type=int, default=32768)
    parser.add_argument("--combine-mode", default="linear", choices=["linear", "exponential"])
    parser.add_argument("--basis-quantization-bits", type=int, default=0)
    parser.add_argument("--basis-quantization-format", default="")
    parser.add_argument("--basis-quantization-target", default="reconstructed", choices=["reconstructed", "factors"])
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-gpu-stats", action="store_true")
    args = parser.parse_args()
    config = AttentionBasisPPLConfig(
        output_dir=args.output_dir,
        model_name=args.model_name,
        basis_sizes=_parse_basis_sizes(args.basis_sizes),
        nmf_iterations=args.nmf_iterations,
        eval_steps=args.eval_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_split=args.data_split,
        max_dataset_tokens=args.max_dataset_tokens,
        dtype=args.dtype,
        ce_chunk_tokens=args.ce_chunk_tokens,
        include_baseline=not args.no_baseline,
        log_gpu_stats=not args.no_gpu_stats,
        combine_mode=args.combine_mode,
        basis_quantization_bits=args.basis_quantization_bits,
        basis_quantization_format=args.basis_quantization_format,
        basis_quantization_target=args.basis_quantization_target,
    )
    print(json.dumps(run_attention_basis_ppl_eval(config), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
