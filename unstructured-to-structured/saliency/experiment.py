from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from saliency.calibration import batched, build_causal_lm_batch
from saliency.saliency import ParameterSaliencyAccumulator, save_saliency_artifacts


@dataclass(slots=True)
class SaliencyConfig:
    output_dir: str | Path
    model_name: str = "EleutherAI/pythia-31m"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    max_examples: int = 128
    batch_size: int = 2
    max_length: int = 512
    dtype: str = "bf16"
    device: str = "auto"
    answer_only_loss: bool = True
    top_k: int = 50
    revision: str | None = None


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    normalized = dtype.lower()
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unsupported dtype: {dtype}")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


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
    model.train()
    return model


def _loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    supervised_tokens = int(torch.count_nonzero(shift_labels != -100).item())
    if supervised_tokens == 0:
        raise ValueError("batch has no supervised target tokens after causal shift")
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss, supervised_tokens


def _load_records(config: SaliencyConfig) -> list[dict[str, Any]]:
    dataset = load_dataset(config.dataset_name, config.dataset_config, split=config.split)
    limit = min(config.max_examples, len(dataset)) if config.max_examples > 0 else len(dataset)
    return [dict(row) for row in dataset.select(range(limit))]


def run_saliency_experiment(config: SaliencyConfig) -> dict[str, object]:
    started = time.time()
    device = resolve_device(config.device)
    dtype = resolve_torch_dtype(config.dtype)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    tokenizer = _prepare_tokenizer(config.model_name, config.revision)
    model = _prepare_model(config.model_name, config.revision, dtype, device)
    records = _load_records(config)
    accumulator = ParameterSaliencyAccumulator(model.named_parameters())

    total_loss = 0.0
    total_supervised_tokens = 0
    batches = 0

    progress = tqdm(list(batched(records, config.batch_size)), desc="saliency", unit="batch")
    for record_batch in progress:
        batch = build_causal_lm_batch(
            tokenizer,
            record_batch,
            config.max_length,
            answer_only_loss=config.answer_only_loss,
            device=device,
        )
        model.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        loss, supervised_tokens = _loss_sum(outputs.logits, batch["labels"])
        loss.backward()
        accumulator.accumulate(model.named_parameters())

        total_loss += float(loss.detach().cpu().item())
        total_supervised_tokens += supervised_tokens
        batches += 1
        progress.set_postfix(loss_per_token=f"{total_loss / max(total_supervised_tokens, 1):.4f}")

    scores = accumulator.finalize(normalizer=total_supervised_tokens)
    metadata = {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "device": str(device),
        "torch_dtype": str(dtype),
        "num_examples": len(records),
        "num_batches": batches,
        "supervised_tokens": total_supervised_tokens,
        "loss_sum": total_loss,
        "loss_per_token": total_loss / max(total_supervised_tokens, 1),
        "elapsed_seconds": time.time() - started,
        "saliency_method": "abs(parameter * gradient), accumulated per batch and normalized by supervised target tokens",
    }
    return save_saliency_artifacts(config.output_dir, scores, metadata, top_k=config.top_k)
