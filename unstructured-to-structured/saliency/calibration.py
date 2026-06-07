from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch


def format_gsm8k_prompt_answer(record: Mapping[str, Any]) -> tuple[str, str]:
    question = str(record["question"]).strip()
    answer = str(record["answer"]).strip()
    return f"Question: {question}\nAnswer:", answer


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _eos_suffix(tokenizer: Any) -> str:
    return tokenizer.eos_token if getattr(tokenizer, "eos_token", None) else ""


def build_causal_lm_batch(
    tokenizer: Any,
    records: Iterable[Mapping[str, Any]],
    max_length: int,
    *,
    answer_only_loss: bool = True,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor | int]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer must expose pad_token_id or eos_token_id")

    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    supervised_tokens = 0

    for record in records:
        prompt, answer = format_gsm8k_prompt_answer(record)
        prompt_ids = _token_ids(tokenizer, prompt)
        full_ids = _token_ids(tokenizer, f"{prompt} {answer}{_eos_suffix(tokenizer)}")[:max_length]
        if len(full_ids) < 2:
            continue

        labels = full_ids.copy()
        if answer_only_loss:
            prompt_len = min(len(prompt_ids), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
        target_count = sum(label != -100 for label in labels)
        if target_count == 0:
            continue

        input_rows.append(full_ids)
        label_rows.append(labels)
        supervised_tokens += target_count

    if not input_rows:
        raise ValueError("no usable calibration records after tokenization/truncation")

    width = max(len(row) for row in input_rows)
    input_ids = torch.full((len(input_rows), width), int(pad_id), dtype=torch.long)
    labels = torch.full((len(input_rows), width), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(input_rows), width), dtype=torch.long)

    for row_idx, (ids, row_labels) in enumerate(zip(input_rows, label_rows, strict=True)):
        length = len(ids)
        input_ids[row_idx, :length] = torch.tensor(ids, dtype=torch.long)
        labels[row_idx, :length] = torch.tensor(row_labels, dtype=torch.long)
        attention_mask[row_idx, :length] = 1

    if device is not None:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "supervised_tokens": supervised_tokens,
    }


def batched(records: Iterable[Mapping[str, Any]], batch_size: int) -> Iterable[list[Mapping[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[Mapping[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
