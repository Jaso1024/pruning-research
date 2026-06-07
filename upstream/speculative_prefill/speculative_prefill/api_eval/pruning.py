import math
from dataclasses import dataclass
from typing import Sequence

import torch

from speculative_prefill.vllm_patch.selector import (
    embedding_norms_from_weight,
    hidden_state_norms,
    select_kept_indices_from_scores,
)
from speculative_prefill.vllm_patch.middle_layer_scorer import (
    capture_layer_hidden_states,
    inference_scorer_mode,
    middle_layer_index,
)


@dataclass(frozen=True, order=True)
class TextSpan:
    start: int
    end: int


@dataclass(frozen=True)
class CompressionResult:
    text: str
    original_tokens: int
    kept_tokens: int
    spans: list[TextSpan]


def merge_adjacent_spans(spans: Sequence[TextSpan], max_gap: int = 1) -> list[TextSpan]:
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [ordered[0]]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start - last.end <= max_gap:
            merged[-1] = TextSpan(last.start, max(last.end, span.end))
        else:
            merged.append(span)
    return merged


def spans_to_text(text: str, spans: Sequence[TextSpan], delimiter: str) -> str:
    return delimiter.join(text[span.start:span.end].strip() for span in spans)


def token_chunk_spans_from_offsets(
    offsets: Sequence[tuple[int, int]],
    kept_chunks: torch.Tensor,
    chunk_size: int,
) -> list[TextSpan]:
    spans = []
    for chunk_idx in kept_chunks.tolist():
        start_token = chunk_idx * chunk_size
        end_token = min(len(offsets), start_token + chunk_size)
        non_empty = [
            offset for offset in offsets[start_token:end_token]
            if offset[1] > offset[0]
        ]
        if not non_empty:
            continue
        spans.append(TextSpan(non_empty[0][0], non_empty[-1][1]))
    return merge_adjacent_spans(spans)


def _chunk_scores(token_scores: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk_cnt = math.ceil(token_scores.numel() / chunk_size)
    scores = []
    for chunk_idx in range(chunk_cnt):
        chunk = token_scores[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
        scores.append(chunk.mean())
    return torch.stack(scores)


def token_scores_from_query_attentions(
    attentions_by_step: Sequence[Sequence[torch.Tensor]],
    seq_len: int,
    layer_index: int | None = None,
) -> torch.Tensor:
    layer_scores = []
    for step_attentions in attentions_by_step:
        if layer_index is None:
            attentions = step_attentions
        else:
            if not 1 <= layer_index <= len(step_attentions):
                raise IndexError(
                    f"layer_index must be in [1, {len(step_attentions)}]."
                )
            attentions = (step_attentions[layer_index - 1],)
        for attn in attentions:
            scores = attn[0, :, -1, :seq_len].mean(dim=0)
            layer_scores.append(scores)
    return torch.stack(layer_scores).max(dim=0).values.float().cpu()


class EmbeddingNormTextCompressor:
    def __init__(
        self,
        model_name: str,
        *,
        percentage: float = 0.3,
        chunk_size: int = 32,
        norm: str = "l2",
        keep_high: bool = True,
        delimiter: str = "\n...\n",
        device: str = "cpu",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
        )
        weight = model.get_input_embeddings().weight.detach()
        self.embedding_norms = embedding_norms_from_weight(weight, norm=norm).cpu()
        del model
        self.percentage = percentage
        self.chunk_size = chunk_size
        self.keep_high = keep_high
        self.delimiter = delimiter
        self.device = device

    def compress(self, text: str) -> CompressionResult:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        if input_ids.numel() == 0:
            return CompressionResult(text=text, original_tokens=0, kept_tokens=0, spans=[])
        token_scores = self.embedding_norms[input_ids]
        chunk_scores = _chunk_scores(token_scores, self.chunk_size)
        kept_chunks = select_kept_indices_from_scores(
            chunk_scores,
            percentage=self.percentage,
            keep_high=self.keep_high,
        )
        spans = token_chunk_spans_from_offsets(
            encoded["offset_mapping"],
            kept_chunks,
            self.chunk_size,
        )
        compressed = spans_to_text(text, spans, self.delimiter)
        kept_tokens = sum(
            min(self.chunk_size, input_ids.numel() - chunk_idx * self.chunk_size)
            for chunk_idx in kept_chunks.tolist()
        )
        return CompressionResult(
            text=compressed,
            original_tokens=input_ids.numel(),
            kept_tokens=kept_tokens,
            spans=spans,
        )


class AttentionTextCompressor:
    def __init__(
        self,
        model_name: str,
        *,
        percentage: float = 0.3,
        chunk_size: int = 32,
        lookahead: int = 1,
        attention_layer_index: int | None = None,
        delimiter: str = "\n...\n",
        device: str | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device in ("cuda", "mps") else torch.float32,
        ).to(self.device)
        self.model.eval()
        self.percentage = percentage
        self.chunk_size = chunk_size
        self.lookahead = lookahead
        self.attention_layer_index = attention_layer_index
        self.delimiter = delimiter

    @torch.inference_mode()
    def compress(self, text: str) -> CompressionResult:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        offsets = encoded["offset_mapping"][0].tolist()
        seq_len = input_ids.shape[1]
        if seq_len == 0:
            return CompressionResult(text=text, original_tokens=0, kept_tokens=0, spans=[])

        if seq_len == 1:
            outputs = self.model(
                input_ids=input_ids,
                output_attentions=True,
                use_cache=False,
            )
            attentions_by_step = [outputs.attentions]
        else:
            prefill = self.model(
                input_ids=input_ids[:, :-1],
                output_attentions=False,
                use_cache=True,
            )
            past_key_values = prefill.past_key_values
            query_ids = input_ids[:, -1:]
            attentions_by_step = []
            for _ in range(self.lookahead):
                outputs = self.model(
                    input_ids=query_ids,
                    past_key_values=past_key_values,
                    output_attentions=True,
                    use_cache=True,
                )
                attentions_by_step.append(outputs.attentions)
                past_key_values = outputs.past_key_values
                query_ids = outputs.logits[:, -1:].argmax(dim=-1)
        token_scores = token_scores_from_query_attentions(
            attentions_by_step,
            seq_len,
            layer_index=self.attention_layer_index,
        )
        chunk_scores = _chunk_scores(token_scores, self.chunk_size)
        kept_chunks = select_kept_indices_from_scores(
            chunk_scores,
            percentage=self.percentage,
            keep_high=True,
        )
        spans = token_chunk_spans_from_offsets(offsets, kept_chunks, self.chunk_size)
        compressed = spans_to_text(text, spans, self.delimiter)
        kept_tokens = sum(
            min(self.chunk_size, seq_len - chunk_idx * self.chunk_size)
            for chunk_idx in kept_chunks.tolist()
        )
        return CompressionResult(
            text=compressed,
            original_tokens=seq_len,
            kept_tokens=kept_tokens,
            spans=spans,
        )


class MiddleLayerNormTextCompressor:
    def __init__(
        self,
        model_name: str,
        *,
        percentage: float = 0.3,
        chunk_size: int = 32,
        norm: str = "l2",
        keep_high: bool = True,
        layer_fraction: float = 0.5,
        layer_index: int | None = None,
        activation_target: str = "layer",
        delimiter: str = "\n...\n",
        device: str | None = None,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        dtype = torch.bfloat16 if self.device == "cuda" else (
            torch.float16 if self.device == "mps" else torch.float32
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,
        ).to(self.device)
        self.model.eval()
        num_layers = int(getattr(self.model.config, "num_hidden_layers"))
        if layer_index is None:
            self.layer_index = middle_layer_index(num_layers, layer_fraction)
        else:
            self.layer_index = max(1, min(num_layers, int(layer_index)))
        self.activation_target = activation_target
        self.percentage = percentage
        self.chunk_size = chunk_size
        self.norm = norm
        self.keep_high = keep_high
        self.delimiter = delimiter

    @torch.inference_mode()
    def compress(self, text: str) -> CompressionResult:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        offsets = encoded["offset_mapping"][0].tolist()
        seq_len = input_ids.shape[1]
        if seq_len == 0:
            return CompressionResult(text=text, original_tokens=0, kept_tokens=0, spans=[])

        with inference_scorer_mode(self.model):
            states = capture_layer_hidden_states(
                self.model,
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                layer_index=self.layer_index,
                activation_target=self.activation_target,
            )
        token_scores = hidden_state_norms(states, norm=self.norm)
        chunk_scores = _chunk_scores(token_scores, self.chunk_size)
        kept_chunks = select_kept_indices_from_scores(
            chunk_scores,
            percentage=self.percentage,
            keep_high=self.keep_high,
        )
        spans = token_chunk_spans_from_offsets(offsets, kept_chunks, self.chunk_size)
        compressed = spans_to_text(text, spans, self.delimiter)
        kept_tokens = sum(
            min(self.chunk_size, seq_len - chunk_idx * self.chunk_size)
            for chunk_idx in kept_chunks.tolist()
        )
        return CompressionResult(
            text=compressed,
            original_tokens=seq_len,
            kept_tokens=kept_tokens,
            spans=spans,
        )
