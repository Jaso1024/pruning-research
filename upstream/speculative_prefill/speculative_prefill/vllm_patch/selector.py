import math
from typing import Literal

import torch


def _validate_percentage(percentage: float) -> None:
    if not 0.0 < percentage <= 1.0:
        raise ValueError("percentage must be in the interval (0, 1].")


def _stable_topk_indices(
    scores: torch.Tensor,
    keep_count: int,
    keep_high: bool,
) -> torch.Tensor:
    if keep_count >= scores.numel():
        return torch.arange(scores.numel(), dtype=torch.long, device=scores.device)

    topk_values = torch.topk(
        scores,
        k=keep_count,
        largest=keep_high,
        sorted=False,
    ).values
    cutoff = topk_values.min() if keep_high else topk_values.max()
    better = scores > cutoff if keep_high else scores < cutoff
    selected = torch.nonzero(better, as_tuple=False).flatten()
    needed = keep_count - selected.numel()
    if needed > 0:
        ties = torch.nonzero(scores == cutoff, as_tuple=False).flatten()
        selected = torch.cat((selected, ties[:needed]))
    return selected


def select_kept_indices_from_scores(
    scores: torch.Tensor,
    percentage: float,
    *,
    chunk: bool = False,
    chunk_size: int = 32,
    keep_high: bool = True,
) -> torch.LongTensor:
    _validate_percentage(percentage)
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D tensor.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    seq_len = scores.numel()
    if seq_len == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    if chunk:
        full_chunk_cnt = seq_len // chunk_size
        tail_len = seq_len - full_chunk_cnt * chunk_size
        if full_chunk_cnt:
            full_scores = scores[:full_chunk_cnt * chunk_size].view(
                full_chunk_cnt, chunk_size)
            chunk_scores = full_scores.mean(dim=1)
            if tail_len:
                tail_score = scores[full_chunk_cnt * chunk_size:].mean().view(1)
                chunk_scores = torch.cat((chunk_scores, tail_score))
        else:
            chunk_scores = scores.mean().view(1)
        chunk_cnt = chunk_scores.numel()
        keep_chunk_cnt = math.ceil(chunk_cnt * percentage)
        selected_chunks = _stable_topk_indices(
            chunk_scores,
            keep_chunk_cnt,
            keep_high,
        )
        offsets = torch.arange(chunk_size, dtype=torch.long, device=scores.device)
        kept = selected_chunks[:, None] * chunk_size + offsets[None, :]
        kept = kept.flatten()
        kept = kept[kept < seq_len]
    else:
        keep_cnt = math.ceil(seq_len * percentage)
        kept = _stable_topk_indices(scores, keep_cnt, keep_high)

    return torch.sort(kept.to(dtype=torch.long))[0]


def token_scores_from_embedding_norms(
    prompt_token_ids: torch.Tensor,
    embedding_norms: torch.Tensor,
) -> torch.Tensor:
    if prompt_token_ids.ndim != 1:
        raise ValueError("prompt_token_ids must be a 1D tensor.")
    if prompt_token_ids.device != embedding_norms.device:
        prompt_token_ids = prompt_token_ids.to(device=embedding_norms.device)
    return embedding_norms[prompt_token_ids]


def embedding_norms_from_weight(
    weight: torch.Tensor,
    *,
    norm: Literal["l1", "l2"] = "l2",
) -> torch.Tensor:
    weight = weight.detach()
    if norm == "l2":
        return torch.linalg.vector_norm(weight.float(), ord=2, dim=1)
    if norm == "l1":
        return torch.linalg.vector_norm(weight.float(), ord=1, dim=1)
    raise ValueError(f"Unsupported embedding norm: {norm}")


def hidden_state_norms(
    hidden_states: torch.Tensor,
    *,
    norm: Literal["l1", "l2"] = "l2",
) -> torch.Tensor:
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("hidden_states must have shape [1, seq_len, hidden_size].")
    states = hidden_states.detach()[0].float()
    if norm == "l2":
        return torch.linalg.vector_norm(states, ord=2, dim=1).cpu()
    if norm == "l1":
        return torch.linalg.vector_norm(states, ord=1, dim=1).cpu()
    raise ValueError(f"Unsupported hidden-state norm: {norm}")
