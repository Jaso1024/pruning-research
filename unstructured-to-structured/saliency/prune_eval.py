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
from saliency.experiment import _loss_sum, resolve_device, resolve_torch_dtype


@dataclass(slots=True)
class PerplexityStats:
    loss_sum: float
    supervised_tokens: int
    num_batches: int
    num_examples: int

    @property
    def loss_per_token(self) -> float:
        return self.loss_sum / max(self.supervised_tokens, 1)

    @property
    def perplexity(self) -> float:
        return math.exp(self.loss_per_token)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "loss_sum": self.loss_sum,
            "supervised_tokens": self.supervised_tokens,
            "num_batches": self.num_batches,
            "num_examples": self.num_examples,
            "loss_per_token": self.loss_per_token,
            "perplexity": self.perplexity,
        }


@dataclass(slots=True)
class PruneEvalConfig:
    output_dir: str | Path
    saliency_path: str | Path
    model_name: str = "EleutherAI/pythia-31m"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    max_examples: int = 0
    batch_size: int = 32
    max_length: int = 512
    dtype: str = "fp32"
    device: str = "auto"
    answer_only_loss: bool = True
    prune_fraction: float = 0.5
    pruning_scope: str = "per_matrix"
    revision: str | None = None


def lowest_saliency_mask(saliency: torch.Tensor, *, fraction: float) -> torch.Tensor:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1")
    flat = saliency.detach().flatten().float()
    count = int(flat.numel() * fraction)
    if count <= 0:
        return torch.zeros_like(saliency, dtype=torch.bool)
    order = torch.argsort(flat, stable=True)
    mask = torch.zeros(flat.numel(), dtype=torch.bool, device=flat.device)
    mask[order[:count]] = True
    return mask.reshape(saliency.shape)


def lowest_saliency_mask_per_output_row(saliency: torch.Tensor, *, fraction: float) -> torch.Tensor:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1")
    if saliency.ndim != 2:
        raise ValueError("per-output-row pruning requires a 2D saliency tensor")
    score = saliency.detach().float()
    count = int(score.shape[1] * fraction)
    if count <= 0:
        return torch.zeros_like(score, dtype=torch.bool)
    order = torch.argsort(score, dim=1, stable=True)
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(1, order[:, :count], True)
    return mask


def structured_2to4_stats(mask: torch.Tensor, *, group_dim: int = 1) -> dict[str, int | float | bool]:
    if mask.ndim != 2:
        raise ValueError("2:4 stats require a 2D mask")
    dim = group_dim % mask.ndim
    if mask.shape[dim] % 4 != 0:
        raise ValueError(f"group dimension size must be divisible by 4, got {mask.shape[dim]}")

    grouped = mask.bool().movedim(dim, -1).reshape(-1, mask.shape[dim] // 4, 4)
    zero_counts = grouped.sum(dim=-1)
    groups = zero_counts.numel()
    compliant = zero_counts >= 2
    extra = torch.clamp_min(2 - zero_counts, 0)
    existing_zeros = int(mask.sum().item())
    extra_zeros = int(extra.sum().item())
    return {
        "groups": groups,
        "compliant_groups": int(compliant.sum().item()),
        "compliant_group_fraction": float(compliant.float().mean().item()) if groups else 0.0,
        "existing_zeros": existing_zeros,
        "existing_zero_fraction": existing_zeros / max(mask.numel(), 1),
        "extra_zeros_needed": extra_zeros,
        "target_total_zeros": existing_zeros + extra_zeros,
        "target_zero_fraction": (existing_zeros + extra_zeros) / max(mask.numel(), 1),
        "already_2to4": bool(torch.all(compliant).item()) if groups else False,
    }


def apply_saliency_pruning_(
    model: torch.nn.Module,
    saliency_scores: dict[str, torch.Tensor],
    *,
    fraction: float = 0.5,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    weights_seen = 0
    weights_zeroed = 0
    tensors_seen = 0
    tensors_pruned = 0
    missing_saliency: list[str] = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim != 2:
                continue
            tensors_seen += 1
            weights_seen += param.numel()
            score = saliency_scores.get(name)
            if score is None:
                missing_saliency.append(name)
                continue
            if tuple(score.shape) != tuple(param.shape):
                raise ValueError(f"saliency shape mismatch for {name}: {tuple(score.shape)} != {tuple(param.shape)}")

            mask = lowest_saliency_mask(score.to(device=param.device), fraction=fraction)
            param.masked_fill_(mask, 0)
            zeroed = int(mask.sum().item())
            weights_zeroed += zeroed
            tensors_pruned += 1
            rows.append({"name": name, "shape": list(param.shape), "weights": param.numel(), "zeroed": zeroed})

    return {
        "pruning_scope": "per_matrix",
        "prune_fraction": fraction,
        "matrix_tensors_seen": tensors_seen,
        "matrix_tensors_pruned": tensors_pruned,
        "weights_seen": weights_seen,
        "weights_zeroed": weights_zeroed,
        "actual_zero_fraction": weights_zeroed / max(weights_seen, 1),
        "missing_saliency": missing_saliency,
        "pruned_tensors": rows,
    }


def apply_row_saliency_pruning_(
    model: torch.nn.Module,
    saliency_scores: dict[str, torch.Tensor],
    *,
    fraction: float = 0.5,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    weights_seen = 0
    weights_zeroed = 0
    tensors_seen = 0
    tensors_pruned = 0
    missing_saliency: list[str] = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim != 2:
                continue
            tensors_seen += 1
            weights_seen += param.numel()
            score = saliency_scores.get(name)
            if score is None:
                missing_saliency.append(name)
                continue
            if tuple(score.shape) != tuple(param.shape):
                raise ValueError(f"saliency shape mismatch for {name}: {tuple(score.shape)} != {tuple(param.shape)}")

            mask = lowest_saliency_mask_per_output_row(score.to(device=param.device), fraction=fraction)
            param.masked_fill_(mask, 0)
            zeroed = int(mask.sum().item())
            weights_zeroed += zeroed
            tensors_pruned += 1
            rows.append({"name": name, "shape": list(param.shape), "weights": param.numel(), "zeroed": zeroed})

    return {
        "pruning_scope": "per_output_row",
        "prune_fraction": fraction,
        "matrix_tensors_seen": tensors_seen,
        "matrix_tensors_pruned": tensors_pruned,
        "weights_seen": weights_seen,
        "weights_zeroed": weights_zeroed,
        "actual_zero_fraction": weights_zeroed / max(weights_seen, 1),
        "missing_saliency": missing_saliency,
        "pruned_tensors": rows,
    }


def apply_global_saliency_pruning_(
    model: torch.nn.Module,
    saliency_scores: dict[str, torch.Tensor],
    *,
    fraction: float = 0.5,
) -> dict[str, object]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1")

    entries: list[tuple[str, torch.nn.Parameter, torch.Tensor]] = []
    weights_seen = 0
    missing_saliency: list[str] = []
    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        weights_seen += param.numel()
        score = saliency_scores.get(name)
        if score is None:
            missing_saliency.append(name)
            continue
        if tuple(score.shape) != tuple(param.shape):
            raise ValueError(f"saliency shape mismatch for {name}: {tuple(score.shape)} != {tuple(param.shape)}")
        entries.append((name, param, score.detach().flatten().float()))

    prune_count = int(weights_seen * fraction)
    available_scores = sum(score.numel() for _, _, score in entries)
    if prune_count > available_scores:
        raise ValueError("not enough saliency-covered matrix weights to prune requested fraction")

    all_scores = torch.cat([score for _, _, score in entries])
    threshold = torch.topk(all_scores, k=prune_count, largest=False).values.max()
    threshold_count = int(torch.count_nonzero(all_scores < threshold).item())
    ties_needed = prune_count - threshold_count

    rows: list[dict[str, object]] = []
    weights_zeroed = 0
    with torch.no_grad():
        for name, param, score in entries:
            below = score < threshold
            equal = score == threshold
            mask = below.clone()
            if ties_needed > 0:
                equal_indices = torch.nonzero(equal, as_tuple=False).flatten()
                take = min(ties_needed, equal_indices.numel())
                if take:
                    mask[equal_indices[:take]] = True
                    ties_needed -= int(take)
            mask = mask.reshape(param.shape).to(device=param.device)
            param.masked_fill_(mask, 0)
            zeroed = int(mask.sum().item())
            weights_zeroed += zeroed
            rows.append({"name": name, "shape": list(param.shape), "weights": param.numel(), "zeroed": zeroed})

    return {
        "pruning_scope": "global",
        "prune_fraction": fraction,
        "matrix_tensors_seen": len(entries) + len(missing_saliency),
        "matrix_tensors_pruned": len(entries),
        "weights_seen": weights_seen,
        "weights_zeroed": weights_zeroed,
        "actual_zero_fraction": weights_zeroed / max(weights_seen, 1),
        "global_threshold": float(threshold.item()) if prune_count else 0.0,
        "missing_saliency": missing_saliency,
        "pruned_tensors": rows,
    }


def summarize_ppl_change(baseline: PerplexityStats, pruned: PerplexityStats) -> dict[str, object]:
    baseline_ppl = baseline.perplexity
    pruned_ppl = pruned.perplexity
    return {
        "baseline": baseline.to_dict(),
        "pruned": pruned.to_dict(),
        "delta_loss_per_token": pruned.loss_per_token - baseline.loss_per_token,
        "delta_perplexity": pruned_ppl - baseline_ppl,
        "perplexity_ratio": pruned_ppl / baseline_ppl,
    }


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


def _load_records(config: PruneEvalConfig) -> list[dict[str, Any]]:
    dataset = load_dataset(config.dataset_name, config.dataset_config, split=config.split)
    limit = min(config.max_examples, len(dataset)) if config.max_examples > 0 else len(dataset)
    return [dict(row) for row in dataset.select(range(limit))]


def evaluate_perplexity(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: PruneEvalConfig,
    device: torch.device,
    *,
    desc: str,
) -> PerplexityStats:
    loss_sum = 0.0
    supervised_tokens = 0
    num_batches = 0

    for record_batch in tqdm(list(batched(records, config.batch_size)), desc=desc, unit="batch"):
        batch = build_causal_lm_batch(
            tokenizer,
            record_batch,
            config.max_length,
            answer_only_loss=config.answer_only_loss,
            device=device,
        )
        with torch.inference_mode():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            loss, target_tokens = _loss_sum(outputs.logits, batch["labels"])
        loss_sum += float(loss.detach().cpu().item())
        supervised_tokens += target_tokens
        num_batches += 1

    return PerplexityStats(
        loss_sum=loss_sum,
        supervised_tokens=supervised_tokens,
        num_batches=num_batches,
        num_examples=len(records),
    )


def load_saliency_scores(path: str | Path) -> dict[str, torch.Tensor]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(artifact, dict) and "scores" in artifact:
        return artifact["scores"]
    if isinstance(artifact, dict) and all(isinstance(value, torch.Tensor) for value in artifact.values()):
        return artifact
    raise ValueError(f"unrecognized saliency artifact: {path}")


def run_prune_ppl_experiment(config: PruneEvalConfig) -> dict[str, object]:
    started = time.time()
    device = resolve_device(config.device)
    dtype = resolve_torch_dtype(config.dtype)
    tokenizer = _prepare_tokenizer(config.model_name, config.revision)
    records = _load_records(config)

    baseline_model = _prepare_model(config.model_name, config.revision, dtype, device)
    baseline = evaluate_perplexity(baseline_model, tokenizer, records, config, device, desc="baseline_ppl")
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pruned_model = _prepare_model(config.model_name, config.revision, dtype, device)
    saliency_scores = load_saliency_scores(config.saliency_path)
    if config.pruning_scope == "per_matrix":
        prune_summary = apply_saliency_pruning_(pruned_model, saliency_scores, fraction=config.prune_fraction)
    elif config.pruning_scope == "per_output_row":
        prune_summary = apply_row_saliency_pruning_(pruned_model, saliency_scores, fraction=config.prune_fraction)
    elif config.pruning_scope == "global":
        prune_summary = apply_global_saliency_pruning_(pruned_model, saliency_scores, fraction=config.prune_fraction)
    else:
        raise ValueError(f"unknown pruning_scope: {config.pruning_scope}")
    pruned = evaluate_perplexity(pruned_model, tokenizer, records, config, device, desc="pruned_ppl")

    result = {
        "metadata": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "saliency_path": str(config.saliency_path),
            "device": str(device),
            "torch_dtype": str(dtype),
            "elapsed_seconds": time.time() - started,
            "pruning_rule": (
                "per_matrix: zero lowest-saliency fraction within each 2D weight matrix; "
                "per_output_row: zero lowest-saliency fraction within each output row of each 2D weight matrix; "
                "global: zero lowest-saliency fraction across all 2D weight matrices; "
                "leave non-matrix parameters unchanged"
            ),
        },
        "pruning": prune_summary,
        "ppl_change": summarize_ppl_change(baseline, pruned),
    }

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prune_ppl_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "pruned_tensors.jsonl").open("w") as handle:
        for row in prune_summary["pruned_tensors"]:
            handle.write(json.dumps(row) + "\n")
    return result
