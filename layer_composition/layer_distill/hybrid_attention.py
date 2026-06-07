from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .attention_analysis import _exponential_attention_combo, _normalize
from .experiment import GpuStatsSampler, JsonlStepLogger, _extract_layers, _torch_dtype
from .low_qk_model import (
    _load_wikitext_eval_tokens,
    _make_eval_batches,
    causal_lm_nll_from_logits,
    perplexity_from_nll,
)


@dataclass(frozen=True)
class HybridAttentionEvalConfig:
    output_dir: Path
    combo_path: Path
    small_model: str = "EleutherAI/pythia-1.4b"
    big_model: str = "EleutherAI/pythia-2.8b"
    eval_steps: int = 8
    batch_size: int = 8
    seq_len: int = 256
    seed: int = 17
    data_split: str = "test"
    max_dataset_tokens: int = 2_000_000
    dtype: str = "bf16"
    ce_chunk_tokens: int = 32768
    include_big_baseline: bool = True
    include_small_baseline: bool = False
    per_layer_sweep: bool = False
    greedy_layer_sweep: bool = False
    greedy_max_layers: int | None = None
    replace_layers: tuple[int, ...] | None = None
    log_gpu_stats: bool = True

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "combo_path", Path(self.combo_path))
        if self.replace_layers is not None:
            layers = tuple(sorted({int(layer) for layer in self.replace_layers}))
            object.__setattr__(self, "replace_layers", layers)
            object.__setattr__(self, "per_layer_sweep", False)
            object.__setattr__(self, "greedy_layer_sweep", False)
        if self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 1:
            raise ValueError("seq_len must be greater than 1")
        if self.data_split not in {"train", "validation", "test"}:
            raise ValueError("data_split must be train, validation, or test")
        if self.max_dataset_tokens <= self.seq_len:
            raise ValueError("max_dataset_tokens must be greater than seq_len")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be bf16, fp16, or fp32")
        if self.ce_chunk_tokens <= 0:
            raise ValueError("ce_chunk_tokens must be positive")
        if self.replace_layers is not None and any(layer < 0 for layer in self.replace_layers):
            raise ValueError("replace_layers must be non-negative")
        if self.greedy_max_layers is not None and self.greedy_max_layers <= 0:
            raise ValueError("greedy_max_layers must be positive")


class ExternalAttentionWeightsGPTNeoXAttention(torch.nn.Module):
    def __init__(self, source_attention: torch.nn.Module, *, layer_idx: int, provider: "ExternalAttentionProvider"):
        super().__init__()
        self.source_attention = source_attention
        self.layer_idx = layer_idx
        self.provider = provider
        self.head_size = int(getattr(source_attention, "head_size"))
        dense = getattr(source_attention, "dense", None)
        if hasattr(source_attention, "num_attention_heads"):
            self.num_attention_heads = int(getattr(source_attention, "num_attention_heads"))
        elif dense is not None and hasattr(dense, "in_features"):
            self.num_attention_heads = int(dense.in_features // self.head_size)
        else:
            config = getattr(source_attention, "config", None)
            self.num_attention_heads = int(getattr(config, "num_attention_heads"))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_past: Any | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_past is not None:
            raise NotImplementedError("external-attention hybrid does not implement KV cache")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, 3 * self.head_size)
        qkv = self.source_attention.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        _, _, value_states = qkv.chunk(3, dim=-1)
        attn_weights = self.provider.require(self.layer_idx, value_states)
        attn_output = torch.matmul(attn_weights.to(dtype=value_states.dtype), value_states)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.source_attention.dense(attn_output), attn_weights


class ExternalAttentionProvider:
    def __init__(self):
        self.weights_by_layer: dict[int, torch.Tensor] = {}

    def set(self, weights_by_layer: dict[int, torch.Tensor]) -> None:
        self.weights_by_layer = weights_by_layer

    def clear(self) -> None:
        self.weights_by_layer = {}

    def require(self, layer_idx: int, value_states: torch.Tensor) -> torch.Tensor:
        weights = self.weights_by_layer.get(layer_idx)
        if weights is None:
            raise RuntimeError(f"missing external attention weights for layer {layer_idx}")
        expected = value_states.shape[:3] + (value_states.shape[-2],)
        if tuple(weights.shape) != tuple(expected):
            raise RuntimeError(f"external attention shape mismatch for layer {layer_idx}: {tuple(weights.shape)} != {tuple(expected)}")
        return weights.to(device=value_states.device)


def load_exponential_combo_weights(path: Path) -> torch.Tensor:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("combo file is empty")
    max_layer = max(int(row["layer"]) for row in rows)
    max_big_head = max(int(row["big_head"]) for row in rows)
    small_heads = int(rows[0]["small_heads"])
    weights = torch.zeros(max_layer + 1, max_big_head + 1, small_heads, dtype=torch.float32)
    counts = torch.zeros(max_layer + 1, max_big_head + 1, 1, dtype=torch.float32)
    for row in rows:
        layer = int(row["layer"])
        big_head = int(row["big_head"])
        row_weights = torch.tensor(row["weights"], dtype=torch.float32)
        if row_weights.numel() != small_heads:
            raise ValueError("inconsistent small head count in combo rows")
        weights[layer, big_head] += row_weights
        counts[layer, big_head] += 1.0
    if torch.any(counts == 0):
        raise ValueError("combo file is missing one or more layer/head rows")
    weights = weights / counts
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def patch_attentions_with_external_weights(
    model: torch.nn.Module,
    provider: ExternalAttentionProvider,
    *,
    layer_indices: tuple[int, ...],
) -> dict[int, torch.nn.Module]:
    layers = _extract_layers(_causal_lm_backbone(model))
    originals: dict[int, torch.nn.Module] = {}
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"layer index out of range: {layer_idx}")
        originals[layer_idx] = layers[layer_idx].attention
        layers[layer_idx].attention = ExternalAttentionWeightsGPTNeoXAttention(
            layers[layer_idx].attention,
            layer_idx=layer_idx,
            provider=provider,
        )
    return originals


def restore_attentions(model: torch.nn.Module, originals: dict[int, torch.nn.Module]) -> None:
    layers = _extract_layers(_causal_lm_backbone(model))
    for layer_idx, attention in originals.items():
        layers[layer_idx].attention = attention


def replace_comparable_attentions_with_external_weights(model: torch.nn.Module, provider: ExternalAttentionProvider, *, layer_count: int) -> int:
    layers = _extract_layers(_causal_lm_backbone(model))
    layer_indices = tuple(range(min(layer_count, len(layers))))
    patch_attentions_with_external_weights(model, provider, layer_indices=layer_indices)
    return len(layer_indices)


def build_exponential_combo_attentions(
    small_attentions: tuple[torch.Tensor, ...] | list[torch.Tensor],
    combo_weights: torch.Tensor,
    *,
    target_dtype: torch.dtype,
    layer_indices: tuple[int, ...] | None = None,
    eps: float = 1e-12,
) -> dict[int, torch.Tensor]:
    layer_count = min(len(small_attentions), combo_weights.shape[0])
    if layer_indices is None:
        layer_indices = tuple(range(layer_count))
    result: dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= layer_count:
            continue
        small = _normalize(small_attentions[layer_idx], eps).float()
        weights = combo_weights[layer_idx].to(device=small.device, dtype=torch.float32)
        if small.shape[1] != weights.shape[1]:
            raise ValueError(f"small head mismatch at layer {layer_idx}: {small.shape[1]} != {weights.shape[1]}")
        support = small.sum(dim=1, keepdim=True) > eps
        combo = _exponential_attention_combo(weights, small.clamp_min(eps).log(), support, eps=eps)
        result[layer_idx] = combo.to(dtype=target_dtype)
    return result


def greedy_layer_sweep(
    *,
    comparable_layer_count: int,
    output_dir: Path,
    evaluate_layer_group,
    max_selected_layers: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if comparable_layer_count <= 0:
        return [], []
    max_layers = comparable_layer_count if max_selected_layers is None else min(max_selected_layers, comparable_layer_count)
    selected: tuple[int, ...] = ()
    remaining = set(range(comparable_layer_count))
    runs: list[dict[str, Any]] = []
    path: list[dict[str, Any]] = []

    for round_idx in range(1, max_layers + 1):
        candidates: list[dict[str, Any]] = []
        for candidate_layer in sorted(remaining):
            layer_group = tuple(sorted((*selected, candidate_layer)))
            layer_slug = "_".join(f"{layer_idx:02d}" for layer_idx in layer_group)
            run_name = f"hybrid_expcombo_greedy_r{round_idx:02d}_add_{candidate_layer:02d}_layers_{layer_slug}"
            record = evaluate_layer_group(layer_group, run_name, output_dir / run_name)
            record["greedy_round"] = round_idx
            record["candidate_layer"] = candidate_layer
            record["candidate_base_layers"] = list(selected)
            record["replaced_layers"] = len(layer_group)
            record["replaced_layer_indices"] = list(layer_group)
            record["is_selected_candidate"] = False
            candidates.append(record)
            runs.append(record)

        best = min(candidates, key=lambda row: (float(row["ppl"]), float(row["loss"]), int(row["candidate_layer"])))
        best["is_selected_candidate"] = True
        selected = tuple(best["replaced_layer_indices"])
        remaining.remove(int(best["candidate_layer"]))
        path.append(
            {
                "round": round_idx,
                "selected_layer": int(best["candidate_layer"]),
                "selected_layers": list(selected),
                "loss": best["loss"],
                "ppl": best["ppl"],
                "run_name": best["run_name"],
            }
        )

    return runs, path


def run_hybrid_attention_eval(config: HybridAttentionEvalConfig) -> dict[str, Any]:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(config.dtype)
    model_dtype = dtype if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(config.big_model)
    tokens = _load_wikitext_eval_tokens(tokenizer=tokenizer, split=config.data_split, max_tokens=config.max_dataset_tokens)
    batches = _make_eval_batches(tokens, batch_size=config.batch_size, seq_len=config.seq_len, max_steps=config.eval_steps)
    if not batches:
        raise ValueError("not enough tokens to build an evaluation batch")

    combo_weights = load_exponential_combo_weights(config.combo_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True, default=str) + "\n")

    runs: list[dict[str, Any]] = []
    if config.include_small_baseline:
        small_lm = _load_causal_lm(AutoModelForCausalLM, config.small_model, model_dtype).to(device)
        small_lm.eval()
        runs.append(
            _evaluate_standard_lm(
                model=small_lm,
                batches=batches,
                output_dir=config.output_dir / "small_baseline",
                run_name="small_baseline",
                device=device,
                dtype=dtype,
                config=config,
            )
        )
        del small_lm
        _empty_cache(device)

    if config.include_big_baseline:
        big_lm = _load_causal_lm(AutoModelForCausalLM, config.big_model, model_dtype).to(device)
        big_lm.eval()
        runs.append(
            _evaluate_standard_lm(
                model=big_lm,
                batches=batches,
                output_dir=config.output_dir / "big_baseline",
                run_name="big_baseline",
                device=device,
                dtype=dtype,
                config=config,
            )
        )
        del big_lm
        _empty_cache(device)

    small_model = _load_base_model(AutoModel, config.small_model, model_dtype).to(device)
    big_hybrid = _load_causal_lm(AutoModelForCausalLM, config.big_model, model_dtype).to(device)
    small_model.eval()
    big_hybrid.eval()
    provider = ExternalAttentionProvider()
    target_layer_count = len(_extract_layers(_causal_lm_backbone(big_hybrid)))
    comparable_layer_count = min(combo_weights.shape[0], target_layer_count)
    def evaluate_layer_group(layer_group: tuple[int, ...], run_name: str, output_dir: Path) -> dict[str, Any]:
        originals = patch_attentions_with_external_weights(big_hybrid, provider, layer_indices=tuple(layer_group))
        try:
            return _evaluate_hybrid_lm(
                small_model=small_model,
                big_model=big_hybrid,
                provider=provider,
                combo_weights=combo_weights.to(device=device),
                layer_indices=tuple(layer_group),
                batches=batches,
                output_dir=output_dir,
                run_name=run_name,
                device=device,
                dtype=dtype,
                config=config,
            )
        finally:
            restore_attentions(big_hybrid, originals)

    greedy_path: list[dict[str, Any]] = []
    if config.greedy_layer_sweep:
        greedy_runs, greedy_path = greedy_layer_sweep(
            comparable_layer_count=comparable_layer_count,
            max_selected_layers=config.greedy_max_layers,
            output_dir=config.output_dir,
            evaluate_layer_group=evaluate_layer_group,
        )
        for hybrid_record in greedy_runs:
            hybrid_record["native_tail_layers"] = target_layer_count - int(hybrid_record["replaced_layers"])
            hybrid_record["combo_path"] = str(config.combo_path)
            runs.append(hybrid_record)
    else:
        if config.replace_layers is not None:
            layer_groups = (config.replace_layers,)
        elif config.per_layer_sweep:
            layer_groups = tuple((layer_idx,) for layer_idx in range(comparable_layer_count))
        else:
            layer_groups = (tuple(range(comparable_layer_count)),)

        for layer_group in layer_groups:
            if config.per_layer_sweep:
                run_name = f"hybrid_expcombo_layer_{layer_group[0]:02d}"
                output_dir = config.output_dir / run_name
            elif config.replace_layers is not None:
                layer_slug = "_".join(str(layer_idx) for layer_idx in layer_group)
                run_name = f"hybrid_expcombo_layers_{layer_slug}"
                output_dir = config.output_dir / run_name
            else:
                run_name = "hybrid_expcombo_qk_big_vo_ffn"
                output_dir = config.output_dir / run_name
            hybrid_record = evaluate_layer_group(tuple(layer_group), run_name, output_dir)
            hybrid_record["replaced_layers"] = len(layer_group)
            hybrid_record["replaced_layer_indices"] = list(layer_group)
            hybrid_record["native_tail_layers"] = target_layer_count - len(layer_group)
            hybrid_record["combo_path"] = str(config.combo_path)
            runs.append(hybrid_record)

    summary = {
        "runs": runs,
        "combo_shape": list(combo_weights.shape),
        "comparable_layers": comparable_layer_count,
        "per_layer_sweep": config.per_layer_sweep,
        "greedy_layer_sweep": config.greedy_layer_sweep,
        "greedy_path": greedy_path,
    }
    if greedy_path:
        (config.output_dir / "greedy_layer_path.json").write_text(json.dumps(greedy_path, indent=2, sort_keys=True, default=str) + "\n")
        _write_greedy_layer_path_markdown(config.output_dir / "greedy_layer_path.md", greedy_path)
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary


def _write_greedy_layer_path_markdown(path: Path, greedy_path: list[dict[str, Any]]) -> None:
    lines = [
        "| round | added layer | selected layers | loss | ppl |",
        "| ---: | ---: | --- | ---: | ---: |",
    ]
    for row in greedy_path:
        layers = ",".join(str(layer_idx) for layer_idx in row["selected_layers"])
        lines.append(f"| {row['round']} | {row['selected_layer']} | {layers} | {float(row['loss']):.6f} | {float(row['ppl']):.6f} |")
    path.write_text("\n".join(lines) + "\n")


def _evaluate_standard_lm(
    *,
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    output_dir: Path,
    run_name: str,
    device: torch.device,
    dtype: torch.dtype,
    config: HybridAttentionEvalConfig,
) -> dict[str, Any]:
    return _evaluate_lm_loop(
        batches=batches,
        output_dir=output_dir,
        run_name=run_name,
        device=device,
        dtype=dtype,
        config=config,
        forward_fn=lambda input_ids: model(input_ids=input_ids, use_cache=False).logits,
    )


def _evaluate_hybrid_lm(
    *,
    small_model: torch.nn.Module,
    big_model: torch.nn.Module,
    provider: ExternalAttentionProvider,
    combo_weights: torch.Tensor,
    layer_indices: tuple[int, ...],
    batches: list[torch.Tensor],
    output_dir: Path,
    run_name: str,
    device: torch.device,
    dtype: torch.dtype,
    config: HybridAttentionEvalConfig,
) -> dict[str, Any]:
    def forward_fn(input_ids: torch.Tensor) -> torch.Tensor:
        small_out = small_model(input_ids=input_ids, use_cache=False, output_attentions=True)
        if small_out.attentions is None:
            raise RuntimeError("small model did not return attentions")
        provider.set(
            build_exponential_combo_attentions(
                small_out.attentions,
                combo_weights,
                target_dtype=dtype if device.type == "cuda" else torch.float32,
                layer_indices=layer_indices,
            )
        )
        try:
            return big_model(input_ids=input_ids, use_cache=False).logits
        finally:
            provider.clear()

    return _evaluate_lm_loop(
        batches=batches,
        output_dir=output_dir,
        run_name=run_name,
        device=device,
        dtype=dtype,
        config=config,
        forward_fn=forward_fn,
    )


def _evaluate_lm_loop(
    *,
    batches: list[torch.Tensor],
    output_dir: Path,
    run_name: str,
    device: torch.device,
    dtype: torch.dtype,
    config: HybridAttentionEvalConfig,
    forward_fn,
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
                        logits = forward_fn(input_ids)
                    step_nll, label_tokens = causal_lm_nll_from_logits(logits, input_ids, chunk_tokens=config.ce_chunk_tokens)
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
                    del logits
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


def _load_base_model(auto_model, model_name: str, dtype: torch.dtype) -> torch.nn.Module:
    try:
        return auto_model.from_pretrained(model_name, dtype=dtype, attn_implementation="eager")
    except TypeError:
        return auto_model.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="eager")


def _load_causal_lm(auto_model, model_name: str, dtype: torch.dtype) -> torch.nn.Module:
    try:
        return auto_model.from_pretrained(model_name, dtype=dtype, attn_implementation="eager")
    except TypeError:
        return auto_model.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="eager")


def _causal_lm_backbone(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox
    if hasattr(model, "model"):
        return model.model
    base = getattr(model, "base_model", None)
    if base is not None:
        return base
    return model


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hybrid_attention_eval"))
    parser.add_argument("--combo-path", type=Path, required=True)
    parser.add_argument("--small-model", default="EleutherAI/pythia-1.4b")
    parser.add_argument("--big-model", default="EleutherAI/pythia-2.8b")
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--data-split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--max-dataset-tokens", type=int, default=2_000_000)
    parser.add_argument("--ce-chunk-tokens", type=int, default=32768)
    parser.add_argument("--include-small-baseline", action="store_true")
    parser.add_argument("--skip-big-baseline", action="store_true")
    parser.add_argument("--per-layer-sweep", action="store_true")
    parser.add_argument("--greedy-layer-sweep", action="store_true")
    parser.add_argument("--greedy-max-layers", type=int, default=None)
    parser.add_argument("--replace-layers", default=None, help="Comma-separated target layer indices to replace.")
    args = parser.parse_args(argv)
    replace_layers = None
    if args.replace_layers:
        replace_layers = tuple(int(part.strip()) for part in args.replace_layers.split(",") if part.strip())
    config = HybridAttentionEvalConfig(
        output_dir=args.output_dir,
        combo_path=args.combo_path,
        small_model=args.small_model,
        big_model=args.big_model,
        eval_steps=args.eval_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dtype=args.dtype,
        data_split=args.data_split,
        max_dataset_tokens=args.max_dataset_tokens,
        ce_chunk_tokens=args.ce_chunk_tokens,
        include_big_baseline=not args.skip_big_baseline,
        include_small_baseline=args.include_small_baseline,
        per_layer_sweep=args.per_layer_sweep,
        greedy_layer_sweep=args.greedy_layer_sweep,
        greedy_max_layers=args.greedy_max_layers,
        replace_layers=replace_layers,
    )
    print(json.dumps(run_hybrid_attention_eval(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
