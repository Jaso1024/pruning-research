from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from .experiment import GpuStatsSampler, JsonlStepLogger, _torch_dtype
from .hybrid_attention import _evaluate_lm_loop, _load_causal_lm
from .low_qk_model import _load_wikitext_eval_tokens, _make_eval_batches


SPARSE24_METHODS = {
    "magnitude",
    "wanda",
    "sparsegpt",
    "gptq-cae",
    "gptaq-cae",
    "gptaq-cae-diag",
    "gptaq-cae-gd",
    "qronos",
    "rescomp",
    "rescomp-diag",
    "rescomp-gd",
}
_PYTHIA_LAYER_RE = re.compile(r"^(gpt_neox\.layers\.(\d+))\.")


@dataclass(frozen=True)
class Sparse24LayerGroup:
    layer_name: str
    layer_index: int
    module_names: tuple[str, ...]


@dataclass(frozen=True)
class Sparse24EvalConfig:
    output_dir: Path
    model_name: str = "EleutherAI/pythia-31m"
    methods: tuple[str, ...] = ("magnitude", "wanda", "sparsegpt", "gptaq-cae", "qronos")
    calibration_steps: int = 4
    calibration_batch_size: int = 64
    calibration_seq_len: int = 256
    calibration_tokens: int = 32768
    eval_steps: int = 16
    eval_batch_size: int = 64
    eval_seq_len: int = 256
    seed: int = 17
    data_split: str = "test"
    calibration_split: str = "train"
    max_dataset_tokens: int = 4_000_000
    dtype: str = "bf16"
    damp: float = 0.01
    blocksize: int = 128
    sparsity_n: int = 2
    sparsity_m: int = 4
    alpha: float = 0.25
    cae_alpha: float = 0.25
    gd_steps: int = 1
    gd_lr: float = 0.25
    gd_chunk_tokens: int = 8192
    ce_chunk_tokens: int = 32768
    include_baseline: bool = True
    save_sparse_model: bool = False
    log_gpu_stats: bool = True

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "methods", tuple(str(method) for method in self.methods))
        if not self.methods:
            raise ValueError("methods must be non-empty")
        unknown = sorted(set(self.methods) - SPARSE24_METHODS)
        if unknown:
            raise ValueError(f"unknown sparse24 method(s): {unknown}")
        if self.calibration_steps <= 0:
            raise ValueError("calibration_steps must be positive")
        if self.calibration_batch_size <= 0:
            raise ValueError("calibration_batch_size must be positive")
        if self.calibration_seq_len <= 1:
            raise ValueError("calibration_seq_len must be greater than 1")
        if self.calibration_tokens <= 0:
            raise ValueError("calibration_tokens must be positive")
        if self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive")
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        if self.eval_seq_len <= 1:
            raise ValueError("eval_seq_len must be greater than 1")
        if self.data_split not in {"train", "validation", "test"}:
            raise ValueError("data_split must be train, validation, or test")
        if self.calibration_split not in {"train", "validation", "test"}:
            raise ValueError("calibration_split must be train, validation, or test")
        if self.max_dataset_tokens <= max(self.eval_seq_len, self.calibration_seq_len):
            raise ValueError("max_dataset_tokens must exceed sequence lengths")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be bf16, fp16, or fp32")
        if self.damp < 0:
            raise ValueError("damp must be non-negative")
        if self.sparsity_n <= 0 or self.sparsity_m <= 0 or self.sparsity_n >= self.sparsity_m:
            raise ValueError("sparsity_n and sparsity_m must satisfy 0 < n < m")
        if self.blocksize <= 0 or self.blocksize % self.sparsity_m != 0:
            raise ValueError("blocksize must be positive and divisible by sparsity_m")
        if self.alpha < 0 or self.cae_alpha < 0:
            raise ValueError("alpha values must be non-negative")
        if self.gd_steps <= 0:
            raise ValueError("gd_steps must be positive")
        if self.gd_lr <= 0:
            raise ValueError("gd_lr must be positive")
        if self.gd_chunk_tokens <= 0:
            raise ValueError("gd_chunk_tokens must be positive")
        if self.ce_chunk_tokens <= 0:
            raise ValueError("ce_chunk_tokens must be positive")


@dataclass(frozen=True)
class Sparse24GreedyLayerConfig(Sparse24EvalConfig):
    method: str = "gptaq-cae"
    greedy_max_layers: int | None = None

    def __post_init__(self):
        method = "gptaq-cae" if self.method == "rescomp" else self.method
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "methods", (method,))
        super().__post_init__()
        if method != "gptaq-cae":
            raise ValueError("greedy layer sparse24 currently supports only gptaq-cae/rescomp")
        if self.greedy_max_layers is not None and self.greedy_max_layers <= 0:
            raise ValueError("greedy_max_layers must be positive when set")


@dataclass(frozen=True)
class Sparse24LayerStateConfig(Sparse24EvalConfig):
    method: str = "gptaq-cae"

    def __post_init__(self):
        method = "gptaq-cae" if self.method == "rescomp" else self.method
        method = "gptaq-cae-diag" if method == "rescomp-diag" else method
        method = "gptaq-cae-gd" if method == "rescomp-gd" else method
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "methods", (method,))
        super().__post_init__()


def structured_n_m_mask(weight: torch.Tensor, *, n: int, m: int) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("weight must be 2D")
    if n <= 0 or m <= 0 or n >= m:
        raise ValueError("n and m must satisfy 0 < n < m")
    if weight.shape[1] % m != 0:
        raise ValueError(f"input dimension must be divisible by {m} for {n}:{m} sparsity")
    grouped = weight.abs().view(weight.shape[0], weight.shape[1] // m, m)
    keep = torch.topk(grouped, k=n, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(-1, keep, True)
    return mask.view_as(weight)


def structured_2_4_mask(weight: torch.Tensor) -> torch.Tensor:
    return structured_n_m_mask(weight, n=2, m=4)


def sparse24_state_name(state: tuple[int, ...]) -> str:
    return "baseline" if not state else "layers_" + "_".join(f"{idx:02d}" for idx in state)


def sparse24_worker_cache_key(spec: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(spec["model_name"]),
        str(spec["dtype"]),
        int(spec["calibration_steps"]),
        int(spec["calibration_batch_size"]),
        int(spec["calibration_seq_len"]),
        str(spec["calibration_split"]),
        int(spec["eval_steps"]),
        int(spec["eval_batch_size"]),
        int(spec["eval_seq_len"]),
        str(spec["data_split"]),
        int(spec["sparsity_m"]),
    )


def read_sparse24_state_records(
    output_dir: Path,
    state_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir = Path(output_dir)
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for item in state_items:
        state_name = str(item["state_name"])
        expected_layers = [int(idx) for idx in item.get("layer_indices", [])]
        path = output_dir / state_name / "summary.json"
        if not path.exists():
            missing.append(item)
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            missing.append(item)
            continue
        if [int(idx) for idx in record.get("layer_indices", [])] != expected_layers:
            missing.append(item)
            continue
        record["state_name"] = state_name
        record["mcts_path"] = item.get("path", [])
        records.append(record)
    return records, missing


def sparse24_mcts_select_rollout(
    *,
    root: tuple[int, ...],
    layer_indices: tuple[int, ...],
    stats: dict[tuple[int, ...], dict[str, float]],
    rollout_depth: int,
    exploration: float,
    rng: random.Random,
) -> tuple[tuple[int, ...], list[tuple[int, ...]]]:
    if rollout_depth <= 0:
        raise ValueError("rollout_depth must be positive")
    if len(set(root)) != len(root):
        raise ValueError("root must not contain duplicate layers")
    if not (set(root) <= set(layer_indices)):
        raise ValueError("root contains unknown layers")
    if len(root) >= len(layer_indices):
        return root, [root]

    state = root
    path = [state]
    target_len = min(len(root) + rollout_depth, len(layer_indices))
    while len(state) < target_len:
        children = [state + (idx,) for idx in layer_indices if idx not in state]
        unvisited = [child for child in children if child not in stats or stats[child].get("visits", 0.0) <= 0.0]
        if unvisited:
            state = rng.choice(unvisited)
            path.append(state)
            break
        parent_visits = max(1.0, stats.get(state, {}).get("visits", 1.0))

        def score(child: tuple[int, ...]) -> float:
            child_stats = stats[child]
            visits = max(1.0, child_stats.get("visits", 0.0))
            mean_reward = child_stats.get("value_sum", 0.0) / visits
            return mean_reward + exploration * math.sqrt(math.log(parent_visits + 1.0) / visits)

        state = max(children, key=score)
        path.append(state)

    while len(state) < target_len:
        choices = [idx for idx in layer_indices if idx not in state]
        if not choices:
            break
        state = state + (rng.choice(choices),)
        path.append(state)
    return state, path


def sparse24_mcts_backpropagate(
    stats: dict[tuple[int, ...], dict[str, float]],
    *,
    path: list[tuple[int, ...]],
    ppl: float,
) -> None:
    reward = -float(ppl)
    for state in path:
        node = stats.setdefault(state, {"visits": 0.0, "value_sum": 0.0, "best_ppl": math.inf})
        node["visits"] += 1.0
        node["value_sum"] += reward
        node["best_ppl"] = min(float(node.get("best_ppl", math.inf)), float(ppl))


def sparse24_mcts_best_child(
    *,
    root: tuple[int, ...],
    layer_indices: tuple[int, ...],
    stats: dict[tuple[int, ...], dict[str, float]],
) -> tuple[int, ...] | None:
    children = [root + (idx,) for idx in layer_indices if idx not in root and root + (idx,) in stats]
    if not children:
        return None

    def rank(child: tuple[int, ...]) -> tuple[float, float, int]:
        node = stats[child]
        visits = max(1.0, node.get("visits", 0.0))
        mean_ppl = -node.get("value_sum", 0.0) / visits
        return (mean_ppl, float(node.get("best_ppl", math.inf)), child[-1])

    return min(children, key=rank)


def find_prunable_linear_names(model: torch.nn.Module, *, sparsity_m: int = 4) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if module.weight.ndim != 2 or module.weight.shape[1] % sparsity_m != 0:
            continue
        names.append(name)
    return names


def group_prunable_linear_names_by_layer(module_names: list[str]) -> list[Sparse24LayerGroup]:
    grouped: dict[int, tuple[str, list[str]]] = {}
    for name in module_names:
        match = _PYTHIA_LAYER_RE.match(name)
        if match is None:
            continue
        layer_name = match.group(1)
        layer_index = int(match.group(2))
        grouped.setdefault(layer_index, (layer_name, []))[1].append(name)
    return [
        Sparse24LayerGroup(layer_name=layer_name, layer_index=layer_index, module_names=tuple(names))
        for layer_index, (layer_name, names) in sorted(grouped.items())
    ]


def collect_linear_inputs(
    *,
    model: torch.nn.Module,
    module_name: str,
    batches: list[torch.Tensor],
    device: torch.device,
    max_tokens: int,
) -> torch.Tensor:
    module = model.get_submodule(module_name)
    captured: list[torch.Tensor] = []
    token_count = 0

    def hook(_module, inputs, _output):
        nonlocal token_count
        if token_count >= max_tokens:
            return
        x = inputs[0].detach()
        flat = x.reshape(-1, x.shape[-1]).float().cpu()
        remaining = max_tokens - token_count
        if flat.shape[0] > remaining:
            flat = flat[:remaining]
        captured.append(flat)
        token_count += flat.shape[0]

    handle = module.register_forward_hook(hook)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                if token_count >= max_tokens:
                    break
                input_ids = batch.to(device=device, non_blocking=True) if batch.dtype == torch.long else batch.to(device=device)
                try:
                    model(input_ids=input_ids, use_cache=False)
                except TypeError:
                    model(input_ids)
    finally:
        handle.remove()
        model.train(was_training)
    if not captured:
        raise RuntimeError(f"no inputs captured for module {module_name}")
    return torch.cat(captured, dim=0).contiguous()


def collect_paired_linear_inputs(
    *,
    fp_model: torch.nn.Module,
    sparse_model: torch.nn.Module,
    module_name: str,
    batches: list[torch.Tensor],
    device: torch.device,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        collect_linear_inputs(model=sparse_model, module_name=module_name, batches=batches, device=device, max_tokens=max_tokens),
        collect_linear_inputs(model=fp_model, module_name=module_name, batches=batches, device=device, max_tokens=max_tokens),
    )


def _safe_cholesky_upper_inverse_factor(hessian: torch.Tensor) -> torch.Tensor:
    diag = torch.arange(hessian.shape[0], device=hessian.device)
    jitter = 0.0
    for _ in range(5):
        try:
            chol = torch.linalg.cholesky(hessian)
            inv = torch.cholesky_inverse(chol)
            return torch.linalg.cholesky(inv, upper=True)
        except torch.linalg.LinAlgError:
            jitter = 1e-6 if jitter == 0.0 else jitter * 10
            hessian = hessian.clone()
            hessian[diag, diag] += jitter
    chol = torch.linalg.cholesky(hessian)
    inv = torch.cholesky_inverse(chol)
    return torch.linalg.cholesky(inv, upper=True)


def _safe_cholesky_lower_inverse_factor(hessian: torch.Tensor) -> torch.Tensor:
    diag = torch.arange(hessian.shape[0], device=hessian.device)
    jitter = 0.0
    for _ in range(5):
        try:
            chol = torch.linalg.cholesky(hessian)
            inv = torch.cholesky_inverse(chol)
            return torch.linalg.cholesky(inv, upper=False)
        except torch.linalg.LinAlgError:
            jitter = 1e-6 if jitter == 0.0 else jitter * 10
            hessian = hessian.clone()
            hessian[diag, diag] += jitter
    chol = torch.linalg.cholesky(hessian)
    inv = torch.cholesky_inverse(chol)
    return torch.linalg.cholesky(inv, upper=False)


def _estimate_largest_eigenvalue(matrix: torch.Tensor, *, iterations: int = 12) -> torch.Tensor:
    vec = torch.ones(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    vec = vec / vec.norm().clamp_min(1e-12)
    for _ in range(iterations):
        next_vec = matrix @ vec
        norm = next_vec.norm()
        if float(norm.item()) == 0.0:
            return torch.tensor(1.0, device=matrix.device, dtype=matrix.dtype)
        vec = next_vec / norm
    return (vec @ (matrix @ vec)).abs().clamp_min(1e-8)


def sparsify_weight_2_4(
    weight: torch.Tensor,
    *,
    x_quant: torch.Tensor,
    x_fp: torch.Tensor | None = None,
    method: str = "sparsegpt",
    damp: float = 0.01,
    blocksize: int = 128,
    sparsity_n: int = 2,
    sparsity_m: int = 4,
    alpha: float = 0.25,
    cae_alpha: float = 0.25,
    gd_steps: int = 1,
    gd_lr: float = 0.25,
    gd_chunk_tokens: int = 8192,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if method == "rescomp":
        method = "gptaq-cae"
    if method == "rescomp-diag":
        method = "gptaq-cae-diag"
    if method == "rescomp-gd":
        method = "gptaq-cae-gd"
    if method not in SPARSE24_METHODS:
        raise ValueError(f"unknown sparse24 method: {method}")
    if weight.ndim != 2:
        raise ValueError("weight must be 2D")
    if sparsity_n <= 0 or sparsity_m <= 0 or sparsity_n >= sparsity_m:
        raise ValueError("sparsity_n and sparsity_m must satisfy 0 < n < m")
    if weight.shape[1] % sparsity_m != 0:
        raise ValueError("input dimension must be divisible by sparsity_m")
    if x_quant.ndim != 2 or x_quant.shape[1] != weight.shape[1]:
        raise ValueError("x_quant must have shape [tokens, in_features]")
    if x_fp is not None and (x_fp.ndim != 2 or x_fp.shape != x_quant.shape):
        raise ValueError("x_fp must match x_quant shape")
    if blocksize <= 0 or blocksize % sparsity_m != 0:
        raise ValueError("blocksize must be positive and divisible by sparsity_m")
    if gd_steps <= 0:
        raise ValueError("gd_steps must be positive")
    if gd_lr <= 0:
        raise ValueError("gd_lr must be positive")
    if gd_chunk_tokens <= 0:
        raise ValueError("gd_chunk_tokens must be positive")

    work = weight.detach().float().clone()
    original = work.clone()
    if method in {"magnitude", "wanda"}:
        if method == "magnitude":
            scores = work
        else:
            activation_norm = x_quant.detach().float().pow(2).mean(dim=0).sqrt()
            scores = work.abs() * activation_norm.unsqueeze(0)
        sparse = work * structured_n_m_mask(scores, n=sparsity_n, m=sparsity_m)
        return sparse.to(dtype=weight.dtype), _sparse_stats(
            weight=weight,
            sparse=sparse,
            method=method,
            has_fp_inputs=x_fp is not None,
            sparsity_n=sparsity_n,
            sparsity_m=sparsity_m,
        )

    xq = x_quant.detach().float()
    xf = xq if x_fp is None else x_fp.detach().float()
    if method == "qronos":
        return _sparsify_weight_2_4_qronos(
            weight,
            x_quant=xq,
            x_fp=xf,
            damp=damp,
            sparsity_n=sparsity_n,
            sparsity_m=sparsity_m,
        )

    hessian = (xq.t() @ xq) / max(xq.shape[0], 1)
    hessian_raw = hessian.clone()
    diag = torch.arange(hessian.shape[0], device=hessian.device)
    dead = torch.diag(hessian) == 0
    if dead.any():
        hessian[dead, dead] = 1
        hessian_raw[dead, dead] = 1
        work[:, dead] = 0
        original[:, dead] = 0
    hessian[diag, diag] += damp * torch.mean(torch.diag(hessian)).clamp_min(1e-8)
    if method == "gptaq-cae-diag":
        hdiag = torch.diag(hessian).clamp_min(1e-8)
        upper = torch.diag(torch.rsqrt(hdiag))
    else:
        upper = _safe_cholesky_upper_inverse_factor(hessian)
    if method == "gptaq-cae-gd":
        return _sparsify_weight_n_m_rescomp_gd(
            weight=weight,
            work=work,
            original=original,
            hessian=hessian,
            hessian_raw=hessian_raw,
            upper=upper,
            x_quant=xq,
            x_fp=xf,
            blocksize=blocksize,
            damp=damp,
            sparsity_n=sparsity_n,
            sparsity_m=sparsity_m,
            alpha=alpha,
            cae_alpha=cae_alpha,
            gd_steps=gd_steps,
            gd_lr=gd_lr,
            gd_chunk_tokens=gd_chunk_tokens,
        )

    p_term = None
    r_term = None
    if method in {"gptq-cae", "gptaq-cae", "gptaq-cae-diag"}:
        d_xxt = ((xf - xq).t() @ xq) / max(xq.shape[0], 1)
        xhat_x = hessian_raw + d_xxt
        if method in {"gptaq-cae", "gptaq-cae-diag"}:
            p_term = alpha * torch.triu(d_xxt @ upper.t(), diagonal=1) @ upper
        r_term = cae_alpha * torch.triu(xhat_x @ upper.t(), diagonal=1) @ upper

    columns = work.shape[1]
    quantized = torch.zeros_like(work)
    start_time = time.monotonic()
    for block_start in range(0, columns, blocksize):
        block_end = min(block_start + blocksize, columns)
        count = block_end - block_start
        block_work = work[:, block_start:block_end].clone()
        block_original = original[:, block_start:block_end].clone()
        block_quant = torch.zeros_like(block_work)
        block_err = torch.zeros_like(block_work)
        upper_block = upper[block_start:block_end, block_start:block_end]
        p_block = None if p_term is None else p_term[block_start:block_end, block_start:block_end]
        r_block = None if r_term is None else r_term[block_start:block_end, block_start:block_end]
        for group_start in range(0, count, sparsity_m):
            group_end = min(group_start + sparsity_m, count)
            if group_end - group_start != sparsity_m:
                continue
            w_group = block_work[:, group_start:group_end].clone()
            diag_group = torch.diag(upper_block)[group_start:group_end].clamp_min(1e-12)
            score_group = w_group.pow(2) / diag_group.unsqueeze(0).pow(2)
            q_group = w_group * structured_n_m_mask(score_group, n=sparsity_n, m=sparsity_m)
            for local_col in range(group_start, group_end):
                w_col = block_work[:, local_col].clone()
                q_col = q_group[:, local_col - group_start]
                diag_value = upper_block[local_col, local_col].clamp_min(1e-12)
                err = (w_col - q_col) / diag_value
                block_quant[:, local_col] = q_col
                block_err[:, local_col] = err

                tail = slice(local_col, count)
                block_work[:, tail] -= err.unsqueeze(1) @ upper_block[local_col, tail].unsqueeze(0)
                if p_block is not None:
                    block_work[:, tail] += w_col.unsqueeze(1) @ p_block[local_col, tail].unsqueeze(0)
                if r_block is not None:
                    cae_col = block_original[:, local_col] - w_col
                    block_work[:, tail] += cae_col.unsqueeze(1) @ r_block[local_col, tail].unsqueeze(0)
        quantized[:, block_start:block_end] = block_quant

        if block_end < columns:
            tail = slice(block_end, columns)
            work[:, tail] -= block_err @ upper[block_start:block_end, tail]
            if p_term is not None:
                work[:, tail] += block_work @ p_term[block_start:block_end, tail]
            if r_term is not None:
                work[:, tail] += (block_original - block_work) @ r_term[block_start:block_end, tail]

    sparse = quantized
    stats = _sparse_stats(
        weight=weight,
        sparse=sparse,
        method=method,
        has_fp_inputs=x_fp is not None,
        sparsity_n=sparsity_n,
        sparsity_m=sparsity_m,
    )
    stats["sparsify_elapsed_sec"] = time.monotonic() - start_time
    stats["calibration_tokens"] = int(xq.shape[0])
    if method == "gptaq-cae-diag":
        stats["hessian_approx"] = "diagonal"
    return sparse.to(dtype=weight.dtype), stats


def _sparsify_weight_n_m_rescomp_gd(
    weight: torch.Tensor,
    *,
    work: torch.Tensor,
    original: torch.Tensor,
    hessian: torch.Tensor,
    hessian_raw: torch.Tensor,
    upper: torch.Tensor,
    x_quant: torch.Tensor,
    x_fp: torch.Tensor,
    blocksize: int,
    damp: float,
    sparsity_n: int,
    sparsity_m: int,
    alpha: float,
    cae_alpha: float,
    gd_steps: int,
    gd_lr: float,
    gd_chunk_tokens: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if gd_steps != 1:
        raise ValueError("gptaq-cae-gd currently implements exactly one single local GD step")
    start_time = time.monotonic()
    d_xxt = ((x_fp - x_quant).t() @ x_quant) / max(x_quant.shape[0], 1)
    xhat_x = hessian_raw + d_xxt
    step = gd_lr / _estimate_largest_eigenvalue(hessian).clamp_min(1e-8)
    columns = work.shape[1]
    quantized = torch.zeros_like(work)

    for block_start in range(0, columns, blocksize):
        block_end = min(block_start + blocksize, columns)
        count = block_end - block_start
        block_work = work[:, block_start:block_end].clone()
        block_original = original[:, block_start:block_end].clone()
        block_quant = torch.zeros_like(block_work)
        block_std = torch.zeros_like(block_work)
        block_p = torch.zeros_like(block_work)
        block_r = torch.zeros_like(block_work)
        upper_block = upper[block_start:block_end, block_start:block_end]
        h_block = hessian_raw[block_start:block_end, block_start:block_end]
        d_block = d_xxt[block_start:block_end, block_start:block_end]
        xhat_block = xhat_x[block_start:block_end, block_start:block_end]

        for group_start in range(0, count, sparsity_m):
            group_end = min(group_start + sparsity_m, count)
            if group_end - group_start != sparsity_m:
                continue
            w_group = block_work[:, group_start:group_end].clone()
            diag_group = torch.diag(upper_block)[group_start:group_end].clamp_min(1e-12)
            score_group = w_group.pow(2) / diag_group.unsqueeze(0).pow(2)
            q_group = w_group * structured_n_m_mask(score_group, n=sparsity_n, m=sparsity_m)
            for local_col in range(group_start, group_end):
                w_col = block_work[:, local_col].clone()
                q_col = q_group[:, local_col - group_start]
                std_col = w_col - q_col
                p_col = w_col
                r_col = block_original[:, local_col] - w_col
                block_quant[:, local_col] = q_col
                block_work[:, local_col] = q_col
                block_std[:, local_col] = std_col
                block_p[:, local_col] = p_col
                block_r[:, local_col] = r_col

                tail = slice(local_col + 1, count)
                if local_col + 1 < count:
                    update = std_col.unsqueeze(1) @ h_block[local_col, tail].unsqueeze(0)
                    if alpha:
                        update += alpha * (p_col.unsqueeze(1) @ d_block[local_col, tail].unsqueeze(0))
                    if cae_alpha:
                        update += cae_alpha * (r_col.unsqueeze(1) @ xhat_block[local_col, tail].unsqueeze(0))
                    block_work[:, tail] += step * update

        quantized[:, block_start:block_end] = block_quant
        if block_end < columns:
            tail = slice(block_end, columns)
            work[:, tail] += step * (block_std @ hessian_raw[block_start:block_end, tail])
            if alpha:
                work[:, tail] += step * alpha * (block_p @ d_xxt[block_start:block_end, tail])
            if cae_alpha:
                work[:, tail] += step * cae_alpha * (block_r @ xhat_x[block_start:block_end, tail])

    sparse = quantized

    stats = _sparse_stats(
        weight=weight,
        sparse=sparse,
        method="gptaq-cae-gd",
        has_fp_inputs=True,
        sparsity_n=sparsity_n,
        sparsity_m=sparsity_m,
    )
    stats["sparsify_elapsed_sec"] = time.monotonic() - start_time
    stats["calibration_tokens"] = int(x_quant.shape[0])
    stats["gd_steps"] = int(gd_steps)
    stats["gd_lr"] = float(gd_lr)
    stats["gd_chunk_tokens"] = int(gd_chunk_tokens)
    stats["gd_effective_step"] = float(step.item())
    return sparse.to(dtype=weight.dtype), stats


def _sparsify_weight_2_4_qronos(
    weight: torch.Tensor,
    *,
    x_quant: torch.Tensor,
    x_fp: torch.Tensor,
    damp: float,
    sparsity_n: int,
    sparsity_m: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    work = weight.detach().float().clone()
    original = work.clone()
    xq = x_quant.detach().float()
    xf = x_fp.detach().float()
    hessian = (xq.t() @ xq) / max(xq.shape[0], 1)
    cross = (xq.t() @ xf) / max(xq.shape[0], 1)
    diag = torch.arange(hessian.shape[0], device=hessian.device)
    dead = torch.diag(hessian) == 0
    if dead.any():
        hessian[dead, dead] = 1
        cross[dead, :] = 0
        cross[dead, dead] = 1
        work[:, dead] = 0
        original[:, dead] = 0
    hessian[diag, diag] += damp * torch.linalg.svdvals(hessian).amax().clamp_min(1e-8)
    lower = _safe_cholesky_lower_inverse_factor(hessian)
    columns = work.shape[1]
    quantized = torch.zeros_like(work)
    start_time = time.monotonic()

    for group_start in range(0, columns, sparsity_m):
        group_end = min(group_start + sparsity_m, columns)
        if group_end - group_start != sparsity_m:
            continue
        if group_start == 0:
            future = slice(group_end, columns)
            rhs = original @ cross[group_start:group_end, :].t()
            if group_end < columns:
                rhs -= work[:, future] @ hessian[group_start:group_end, future].t()
            p_real = torch.linalg.solve(hessian[group_start:group_end, group_start:group_end], rhs.t()).t()
            q_group = p_real * structured_n_m_mask(p_real, n=sparsity_n, m=sparsity_m)
        else:
            q_group = work[:, group_start:group_end] * structured_n_m_mask(
                work[:, group_start:group_end],
                n=sparsity_n,
                m=sparsity_m,
            )

        for col in range(group_start, group_end):
            q_col = q_group[:, col - group_start]
            err = work[:, col] - q_col
            quantized[:, col] = q_col
            future_start = col + 1
            if future_start < columns:
                scale = lower[future_start:, col] / lower[col, col].clamp_min(1e-12)
                work[:, future_start:] -= err.unsqueeze(1) @ scale.unsqueeze(0)

    sparse = quantized
    stats = _sparse_stats(
        weight=weight,
        sparse=sparse,
        method="qronos",
        has_fp_inputs=True,
        sparsity_n=sparsity_n,
        sparsity_m=sparsity_m,
    )
    stats["sparsify_elapsed_sec"] = time.monotonic() - start_time
    stats["calibration_tokens"] = int(xq.shape[0])
    return sparse.to(dtype=weight.dtype), stats


def _sparse_stats(
    *,
    weight: torch.Tensor,
    sparse: torch.Tensor,
    method: str,
    has_fp_inputs: bool,
    sparsity_n: int,
    sparsity_m: int,
) -> dict[str, Any]:
    dense = weight.detach().float()
    sparse_f = sparse.detach().float()
    nonzero = int((sparse_f != 0).sum().item())
    total = sparse_f.numel()
    return {
        "method": method,
        "shape": list(weight.shape),
        "density": nonzero / total,
        "zeros": total - nonzero,
        "nonzeros": nonzero,
        "total": total,
        "weight_mse": float(torch.nn.functional.mse_loss(sparse_f, dense).cpu()),
        "weight_rel_mse": float((torch.nn.functional.mse_loss(sparse_f, dense) / dense.pow(2).mean().clamp_min(1e-12)).cpu()),
        "has_fp_inputs": has_fp_inputs,
        "sparsity_n": sparsity_n,
        "sparsity_m": sparsity_m,
    }


def _module_reconstruction_stats(linear: torch.nn.Linear, dense_weight: torch.Tensor, x: torch.Tensor, *, max_tokens: int = 4096) -> dict[str, float]:
    sample = x[:max_tokens].to(device=linear.weight.device, dtype=torch.float32)
    dense = dense_weight.to(device=linear.weight.device, dtype=torch.float32)
    sparse = linear.weight.detach().float()
    target = sample @ dense.t()
    pred = sample @ sparse.t()
    if linear.bias is not None:
        bias = linear.bias.detach().float()
        target = target + bias
        pred = pred + bias
    mse = torch.nn.functional.mse_loss(pred, target)
    rel = mse / target.pow(2).mean().clamp_min(1e-12)
    return {"recon_mse": float(mse.cpu()), "recon_rel_mse": float(rel.cpu())}


def sparsify_model_2_4(
    *,
    fp_model: torch.nn.Module,
    sparse_model: torch.nn.Module,
    module_names: list[str],
    calibration_batches: list[torch.Tensor],
    config: Sparse24EvalConfig,
    method: str,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    sampler = GpuStatsSampler() if config.log_gpu_stats and device.type == "cuda" else None
    if sampler is not None:
        sampler.start()
    try:
        with JsonlStepLogger(output_dir / "modules.jsonl") as logger:
            for idx, name in enumerate(module_names, start=1):
                module = sparse_model.get_submodule(name)
                dense_module = fp_model.get_submodule(name)
                if not isinstance(module, torch.nn.Linear) or not isinstance(dense_module, torch.nn.Linear):
                    continue
                if module.weight.shape[1] % config.sparsity_m != 0:
                    continue
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.monotonic()
                if method in {"gptaq-cae", "gptaq-cae-diag", "gptaq-cae-gd", "qronos", "rescomp", "rescomp-diag", "rescomp-gd"}:
                    x_quant, x_fp = collect_paired_linear_inputs(
                        fp_model=fp_model,
                        sparse_model=sparse_model,
                        module_name=name,
                        batches=calibration_batches,
                        device=device,
                        max_tokens=config.calibration_tokens,
                    )
                else:
                    x_quant = collect_linear_inputs(
                        model=sparse_model,
                        module_name=name,
                        batches=calibration_batches,
                        device=device,
                        max_tokens=config.calibration_tokens,
                    )
                    x_fp = None
                sparse_weight, stats = sparsify_weight_2_4(
                    dense_module.weight.detach(),
                    x_quant=x_quant.to(device=device, non_blocking=True),
                    x_fp=x_fp.to(device=device, non_blocking=True) if x_fp is not None else None,
                    method=method,
                    damp=config.damp,
                    blocksize=config.blocksize,
                    sparsity_n=config.sparsity_n,
                    sparsity_m=config.sparsity_m,
                    alpha=config.alpha,
                    cae_alpha=config.cae_alpha,
                    gd_steps=config.gd_steps,
                    gd_lr=config.gd_lr,
                    gd_chunk_tokens=config.gd_chunk_tokens,
                )
                module.weight.data.copy_(sparse_weight.to(device=module.weight.device, dtype=module.weight.dtype))
                recon = _module_reconstruction_stats(module, dense_module.weight.detach(), x_quant)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.monotonic()
                record = {
                    "module_index": idx,
                    "module_count": len(module_names),
                    "module_name": name,
                    "method": method,
                    "elapsed_sec": end - start,
                    **stats,
                    **recon,
                }
                if sampler is not None:
                    record.update(sampler.stats_between(start, end))
                logger.log(record)
                print(json.dumps(record, sort_keys=True), flush=True)
                records.append(record)
                del x_quant, x_fp, sparse_weight
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        if sampler is not None:
            sampler.stop()
    return records


def run_sparse24_eval(config: Sparse24EvalConfig) -> dict[str, Any]:
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
    calib_tokens = _load_wikitext_eval_tokens(tokenizer=tokenizer, split=config.calibration_split, max_tokens=config.max_dataset_tokens)
    eval_tokens = _load_wikitext_eval_tokens(tokenizer=tokenizer, split=config.data_split, max_tokens=config.max_dataset_tokens)
    calibration_batches = _make_eval_batches(
        calib_tokens,
        batch_size=config.calibration_batch_size,
        seq_len=config.calibration_seq_len,
        max_steps=config.calibration_steps,
    )
    eval_batches = _make_eval_batches(eval_tokens, batch_size=config.eval_batch_size, seq_len=config.eval_seq_len, max_steps=config.eval_steps)
    if not calibration_batches:
        raise ValueError("not enough tokens for calibration batches")
    if not eval_batches:
        raise ValueError("not enough tokens for eval batches")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True, default=str) + "\n")

    fp_model = _load_causal_lm(AutoModelForCausalLM, config.model_name, model_dtype).to(device)
    fp_model.eval()
    for param in fp_model.parameters():
        param.requires_grad_(False)
    module_names = find_prunable_linear_names(fp_model)
    (config.output_dir / "module_names.json").write_text(json.dumps(module_names, indent=2, sort_keys=True) + "\n")

    runs: list[dict[str, Any]] = []
    if config.include_baseline:
        runs.append(
            _evaluate_lm_loop(
                batches=eval_batches,
                output_dir=config.output_dir / "baseline",
                run_name="baseline",
                device=device,
                dtype=dtype,
                config=config,
                forward_fn=lambda input_ids: fp_model(input_ids=input_ids, use_cache=False).logits,
            )
        )

    for method in config.methods:
        sparse_model = _load_causal_lm(AutoModelForCausalLM, config.model_name, model_dtype).to(device)
        sparse_model.eval()
        for param in sparse_model.parameters():
            param.requires_grad_(False)
        method_dir = config.output_dir / method
        start = time.monotonic()
        module_records = sparsify_model_2_4(
            fp_model=fp_model,
            sparse_model=sparse_model,
            module_names=module_names,
            calibration_batches=calibration_batches,
            config=config,
            method=method,
            device=device,
            output_dir=method_dir,
        )
        eval_record = _evaluate_lm_loop(
            batches=eval_batches,
            output_dir=method_dir / "eval",
            run_name=method,
            device=device,
            dtype=dtype,
            config=config,
            forward_fn=lambda input_ids, model=sparse_model: model(input_ids=input_ids, use_cache=False).logits,
        )
        eval_record["method"] = method
        eval_record["sparsify_elapsed_sec"] = time.monotonic() - start
        eval_record["module_count"] = len(module_records)
        eval_record["mean_module_recon_rel_mse"] = (
            sum(row["recon_rel_mse"] for row in module_records) / len(module_records) if module_records else math.nan
        )
        eval_record["mean_weight_rel_mse"] = (
            sum(row["weight_rel_mse"] for row in module_records) / len(module_records) if module_records else math.nan
        )
        eval_record["density"] = (
            sum(row["nonzeros"] for row in module_records) / sum(row["total"] for row in module_records) if module_records else math.nan
        )
        (method_dir / "summary.json").write_text(json.dumps(eval_record, indent=2, sort_keys=True) + "\n")
        if config.save_sparse_model:
            sparse_model.save_pretrained(method_dir / "model")
            tokenizer.save_pretrained(method_dir / "model")
        runs.append(eval_record)
        del sparse_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "model_name": config.model_name,
        "module_count": len(module_names),
        "methods": list(config.methods),
        "sparsity_n": config.sparsity_n,
        "sparsity_m": config.sparsity_m,
        "runs": runs,
    }
    _attach_ratios(summary)
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    _write_sparse24_markdown(config.output_dir / "summary.md", summary)
    return summary


def _attach_ratios(summary: dict[str, Any]) -> None:
    baseline = next((run for run in summary["runs"] if run.get("run_name") == "baseline"), None)
    if baseline is None:
        return
    base_ppl = float(baseline["ppl"])
    base_loss = float(baseline["loss"])
    for run in summary["runs"]:
        run["ppl_ratio_vs_baseline"] = float(run["ppl"]) / base_ppl
        run["loss_delta_vs_baseline"] = float(run["loss"]) - base_loss
        run["ppl_delta_vs_baseline"] = float(run["ppl"]) - base_ppl


def _write_sparse24_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "| method | ppl | ratio vs baseline | loss | density | mean recon rel mse | module count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in summary["runs"]:
        method = run.get("method", run.get("run_name"))
        ratio = run.get("ppl_ratio_vs_baseline")
        density = run.get("density")
        recon = run.get("mean_module_recon_rel_mse")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(method),
                    f"{float(run['ppl']):.6f}",
                    "" if ratio is None else f"{float(ratio):.3f}x",
                    f"{float(run['loss']):.6f}",
                    "" if density is None or math.isnan(float(density)) else f"{float(density):.3f}",
                    "" if recon is None or math.isnan(float(recon)) else f"{float(recon):.6f}",
                    str(run.get("module_count", "")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def _snapshot_module_weights(model: torch.nn.Module, module_names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    snapshot = {}
    for name in module_names:
        module = model.get_submodule(name)
        if isinstance(module, torch.nn.Linear):
            snapshot[name] = module.weight.detach().clone()
    return snapshot


def _restore_module_weights(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    for name, weight in snapshot.items():
        module = model.get_submodule(name)
        if isinstance(module, torch.nn.Linear):
            module.weight.data.copy_(weight)


def _summarize_sparse_module_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "module_count": len(records),
        "mean_module_recon_rel_mse": sum(row["recon_rel_mse"] for row in records) / len(records) if records else math.nan,
        "mean_weight_rel_mse": sum(row["weight_rel_mse"] for row in records) / len(records) if records else math.nan,
        "density": sum(row["nonzeros"] for row in records) / sum(row["total"] for row in records) if records else math.nan,
        "sparsified_modules": [row["module_name"] for row in records],
    }


def _layer_slug(group: Sparse24LayerGroup) -> str:
    return f"layer_{group.layer_index:02d}"


def _evaluate_sparse_model(
    *,
    model: torch.nn.Module,
    eval_batches: list[torch.Tensor],
    output_dir: Path,
    run_name: str,
    device: torch.device,
    dtype: torch.dtype,
    config: Sparse24EvalConfig,
) -> dict[str, Any]:
    return _evaluate_lm_loop(
        batches=eval_batches,
        output_dir=output_dir,
        run_name=run_name,
        device=device,
        dtype=dtype,
        config=config,
        forward_fn=lambda input_ids: model(input_ids=input_ids, use_cache=False).logits,
    )


def run_sparse24_layer_state_eval_loaded(
    config: Sparse24LayerStateConfig,
    *,
    fp_model: torch.nn.Module,
    sparse_model: torch.nn.Module,
    calibration_batches: list[torch.Tensor],
    eval_batches: list[torch.Tensor],
    module_names: list[str],
    groups: list[Sparse24LayerGroup],
    layer_indices: tuple[int, ...],
    run_name: str | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    if device is None:
        device = next(sparse_model.parameters()).device
    if dtype is None:
        dtype = _torch_dtype(config.dtype)
    group_by_index = {group.layer_index: group for group in groups}
    unknown = [idx for idx in layer_indices if idx not in group_by_index]
    if unknown:
        raise ValueError(f"unknown layer index/indices: {unknown}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True, default=str) + "\n")
    (config.output_dir / "layer_groups.json").write_text(
        json.dumps([dataclasses.asdict(group) for group in groups], indent=2, sort_keys=True) + "\n"
    )

    start = time.monotonic()
    module_records: list[dict[str, Any]] = []
    for order_idx, layer_index in enumerate(layer_indices, start=1):
        group = group_by_index[layer_index]
        layer_records = sparsify_model_2_4(
            fp_model=fp_model,
            sparse_model=sparse_model,
            module_names=list(group.module_names),
            calibration_batches=calibration_batches,
            config=config,
            method=config.method,
            device=device,
            output_dir=config.output_dir / f"apply_{order_idx:02d}_{_layer_slug(group)}",
        )
        for record in layer_records:
            record["layer_apply_order"] = order_idx
            record["layer_index"] = layer_index
        module_records.extend(layer_records)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    state_name = run_name or ("baseline" if not layer_indices else "layers_" + "_".join(str(idx) for idx in layer_indices))
    eval_record = _evaluate_sparse_model(
        model=sparse_model,
        eval_batches=eval_batches,
        output_dir=config.output_dir / "eval",
        run_name=state_name,
        device=device,
        dtype=dtype,
        config=config,
    )
    eval_record.update(_summarize_sparse_module_records(module_records))
    eval_record["method"] = config.method
    eval_record["layer_indices"] = list(layer_indices)
    eval_record["layer_count"] = len(layer_indices)
    eval_record["available_layer_count"] = len(groups)
    eval_record["module_count_total"] = len(module_names)
    eval_record["sparsify_elapsed_sec"] = time.monotonic() - start
    (config.output_dir / "summary.json").write_text(json.dumps(eval_record, indent=2, sort_keys=True, default=str) + "\n")
    return eval_record


def run_sparse24_layer_state_eval(
    config: Sparse24LayerStateConfig,
    *,
    layer_indices: tuple[int, ...],
    run_name: str | None = None,
) -> dict[str, Any]:
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
    calib_tokens = _load_wikitext_eval_tokens(tokenizer=tokenizer, split=config.calibration_split, max_tokens=config.max_dataset_tokens)
    eval_tokens = _load_wikitext_eval_tokens(tokenizer=tokenizer, split=config.data_split, max_tokens=config.max_dataset_tokens)
    calibration_batches = _make_eval_batches(
        calib_tokens,
        batch_size=config.calibration_batch_size,
        seq_len=config.calibration_seq_len,
        max_steps=config.calibration_steps,
    )
    eval_batches = _make_eval_batches(eval_tokens, batch_size=config.eval_batch_size, seq_len=config.eval_seq_len, max_steps=config.eval_steps)
    if not calibration_batches:
        raise ValueError("not enough tokens for calibration batches")
    if not eval_batches:
        raise ValueError("not enough tokens for eval batches")

    fp_model = _load_causal_lm(AutoModelForCausalLM, config.model_name, model_dtype).to(device)
    fp_model.eval()
    sparse_model = _load_causal_lm(AutoModelForCausalLM, config.model_name, model_dtype).to(device)
    sparse_model.eval()
    for model in (fp_model, sparse_model):
        for param in model.parameters():
            param.requires_grad_(False)

    module_names = find_prunable_linear_names(fp_model, sparsity_m=config.sparsity_m)
    groups = group_prunable_linear_names_by_layer(module_names)
    return run_sparse24_layer_state_eval_loaded(
        config,
        fp_model=fp_model,
        sparse_model=sparse_model,
        calibration_batches=calibration_batches,
        eval_batches=eval_batches,
        module_names=module_names,
        groups=groups,
        layer_indices=layer_indices,
        run_name=run_name,
        device=device,
        dtype=dtype,
    )


def _write_sparse24_greedy_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "| round | selected layer | ppl | ratio vs baseline | loss | candidate count | mean recon rel mse |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("rounds", []):
        ratio = row.get("ppl_ratio_vs_baseline")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["round"]),
                    row["selected_layer"],
                    f"{float(row['ppl']):.6f}",
                    "" if ratio is None else f"{float(ratio):.3f}x",
                    f"{float(row['loss']):.6f}",
                    str(row["candidate_count"]),
                    f"{float(row['mean_module_recon_rel_mse']):.6f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        import os

        os.fsync(handle.fileno())


def _trim_resumed_candidate_records(
    candidates: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    completed_counts = {int(row["round"]): int(row["candidate_count"]) for row in rounds}
    seen: dict[int, int] = {}
    trimmed: list[dict[str, Any]] = []
    for candidate in candidates:
        round_index = int(candidate["round"])
        seen[round_index] = seen.get(round_index, 0) + 1
        if round_index in completed_counts and seen[round_index] > completed_counts[round_index]:
            continue
        trimmed.append(candidate)
    return trimmed


def run_sparse24_greedy_layer_eval(
    config: Sparse24GreedyLayerConfig,
    *,
    commit_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
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
    calib_tokens = _load_wikitext_eval_tokens(tokenizer=tokenizer, split=config.calibration_split, max_tokens=config.max_dataset_tokens)
    eval_tokens = _load_wikitext_eval_tokens(tokenizer=tokenizer, split=config.data_split, max_tokens=config.max_dataset_tokens)
    calibration_batches = _make_eval_batches(
        calib_tokens,
        batch_size=config.calibration_batch_size,
        seq_len=config.calibration_seq_len,
        max_steps=config.calibration_steps,
    )
    eval_batches = _make_eval_batches(eval_tokens, batch_size=config.eval_batch_size, seq_len=config.eval_seq_len, max_steps=config.eval_steps)
    if not calibration_batches:
        raise ValueError("not enough tokens for calibration batches")
    if not eval_batches:
        raise ValueError("not enough tokens for eval batches")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True, default=str) + "\n")

    fp_model = _load_causal_lm(AutoModelForCausalLM, config.model_name, model_dtype).to(device)
    fp_model.eval()
    for param in fp_model.parameters():
        param.requires_grad_(False)

    sparse_model = _load_causal_lm(AutoModelForCausalLM, config.model_name, model_dtype).to(device)
    sparse_model.eval()
    for param in sparse_model.parameters():
        param.requires_grad_(False)

    module_names = find_prunable_linear_names(fp_model, sparsity_m=config.sparsity_m)
    groups = group_prunable_linear_names_by_layer(module_names)
    if config.greedy_max_layers is not None:
        groups = groups[: config.greedy_max_layers]
    (config.output_dir / "module_names.json").write_text(json.dumps(module_names, indent=2, sort_keys=True) + "\n")
    (config.output_dir / "layer_groups.json").write_text(
        json.dumps([dataclasses.asdict(group) for group in groups], indent=2, sort_keys=True) + "\n"
    )
    if not groups:
        raise ValueError("no transformer-layer linear groups found")

    def maybe_commit() -> None:
        if commit_callback is not None:
            commit_callback()

    baseline = None
    if config.include_baseline:
        baseline_path = config.output_dir / "baseline" / "summary.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
        else:
            baseline = _evaluate_sparse_model(
                model=fp_model,
                eval_batches=eval_batches,
                output_dir=config.output_dir / "baseline",
                run_name="baseline",
                device=device,
                dtype=dtype,
                config=config,
            )
            maybe_commit()

    rounds_path = config.output_dir / "rounds.jsonl"
    candidates_path = config.output_dir / "candidates.jsonl"
    existing_rounds = sorted(_read_jsonl_records(rounds_path), key=lambda row: int(row["round"]))
    existing_candidates = _trim_resumed_candidate_records(_read_jsonl_records(candidates_path), existing_rounds)
    group_by_layer = {group.layer_index: group for group in groups}
    selected_indices = [int(row["selected_layer_index"]) for row in existing_rounds]
    for layer_index in selected_indices:
        if layer_index not in group_by_layer:
            raise ValueError(f"cannot resume unknown selected layer index {layer_index}")

    for row in existing_rounds:
        selected_group = group_by_layer[int(row["selected_layer_index"])]
        sparsify_model_2_4(
            fp_model=fp_model,
            sparse_model=sparse_model,
            module_names=list(selected_group.module_names),
            calibration_batches=calibration_batches,
            config=config,
            method=config.method,
            device=device,
            output_dir=config.output_dir / f"round_{int(row['round']):02d}" / "resume_selected" / _layer_slug(selected_group),
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    remaining = list(groups)
    if selected_indices:
        selected_set = set(selected_indices)
        remaining = [group for group in remaining if group.layer_index not in selected_set]
    rounds: list[dict[str, Any]] = list(existing_rounds)
    candidate_records_all: list[dict[str, Any]] = list(existing_candidates)
    max_rounds = len(remaining)
    start_round = len(rounds) + 1
    for round_offset in range(max_rounds):
        round_index = start_round + round_offset
        round_dir = config.output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[dict[str, Any]] = [row for row in existing_candidates if int(row.get("round", -1)) == round_index]
        completed_candidate_indices = {int(row["candidate_layer_index"]) for row in candidates}
        for group in remaining:
            if group.layer_index in completed_candidate_indices:
                continue
            slug = _layer_slug(group)
            candidate_dir = round_dir / "candidates" / slug
            snapshot = _snapshot_module_weights(sparse_model, group.module_names)
            start = time.monotonic()
            module_records = sparsify_model_2_4(
                fp_model=fp_model,
                sparse_model=sparse_model,
                module_names=list(group.module_names),
                calibration_batches=calibration_batches,
                config=config,
                method=config.method,
                device=device,
                output_dir=candidate_dir,
            )
            eval_record = _evaluate_sparse_model(
                model=sparse_model,
                eval_batches=eval_batches,
                output_dir=candidate_dir / "eval",
                run_name=f"candidate_{round_index:02d}_{slug}",
                device=device,
                dtype=dtype,
                config=config,
            )
            candidate = {
                **eval_record,
                **_summarize_sparse_module_records(module_records),
                "round": round_index,
                "candidate_layer": group.layer_name,
                "candidate_layer_index": group.layer_index,
                "candidate_elapsed_sec": time.monotonic() - start,
                "method": config.method,
            }
            _append_jsonl_record(candidates_path, candidate)
            print(json.dumps(candidate, sort_keys=True), flush=True)
            candidates.append(candidate)
            candidate_records_all.append(candidate)
            maybe_commit()
            _restore_module_weights(sparse_model, snapshot)
            del snapshot, module_records
            if device.type == "cuda":
                torch.cuda.empty_cache()

        best = min(candidates, key=lambda row: (float(row["ppl"]), float(row["loss"])))
        selected_group = next(group for group in remaining if group.layer_index == best["candidate_layer_index"])
        selected_slug = _layer_slug(selected_group)
        selected_dir = round_dir / "selected" / selected_slug
        start = time.monotonic()
        selected_module_records = sparsify_model_2_4(
            fp_model=fp_model,
            sparse_model=sparse_model,
            module_names=list(selected_group.module_names),
            calibration_batches=calibration_batches,
            config=config,
            method=config.method,
            device=device,
            output_dir=selected_dir,
        )
        eval_record = _evaluate_sparse_model(
            model=sparse_model,
            eval_batches=eval_batches,
            output_dir=round_dir / "eval",
            run_name=f"greedy_round_{round_index:02d}_{selected_slug}",
            device=device,
            dtype=dtype,
            config=config,
        )
        round_record = {
            **eval_record,
            **_summarize_sparse_module_records(selected_module_records),
            "round": round_index,
            "selected_layer": selected_group.layer_name,
            "selected_layer_index": selected_group.layer_index,
            "candidate_count": len(candidates),
            "selected_candidate_ppl": best["ppl"],
            "selected_candidate_loss": best["loss"],
            "selected_elapsed_sec": time.monotonic() - start,
            "method": config.method,
            "selected_layers_so_far": [row["selected_layer"] for row in rounds] + [selected_group.layer_name],
        }
        rounds.append(round_record)
        _append_jsonl_record(rounds_path, round_record)
        print(json.dumps(round_record, sort_keys=True), flush=True)
        maybe_commit()
        remaining = [group for group in remaining if group.layer_index != selected_group.layer_index]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "model_name": config.model_name,
        "method": config.method,
        "module_count": len(module_names),
        "layer_count": len(groups),
        "sparsity_n": config.sparsity_n,
        "sparsity_m": config.sparsity_m,
        "baseline": baseline,
        "rounds": rounds,
        "candidate_count": len(candidate_records_all),
    }
    if baseline is not None:
        base_ppl = float(baseline["ppl"])
        base_loss = float(baseline["loss"])
        for row in rounds:
            row["ppl_ratio_vs_baseline"] = float(row["ppl"]) / base_ppl
            row["loss_delta_vs_baseline"] = float(row["loss"]) - base_loss
            row["ppl_delta_vs_baseline"] = float(row["ppl"]) - base_ppl
        for row in candidate_records_all:
            row["ppl_ratio_vs_baseline"] = float(row["ppl"]) / base_ppl
            row["loss_delta_vs_baseline"] = float(row["loss"]) - base_loss
            row["ppl_delta_vs_baseline"] = float(row["ppl"]) - base_ppl

    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    _write_sparse24_greedy_markdown(config.output_dir / "summary.md", summary)
    if config.save_sparse_model:
        sparse_model.save_pretrained(config.output_dir / "model")
        tokenizer.save_pretrained(config.output_dir / "model")
    return summary


def _parse_methods(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sparse24_eval"))
    parser.add_argument("--model-name", default="EleutherAI/pythia-31m")
    parser.add_argument("--methods", default="magnitude,wanda,sparsegpt,gptaq-cae,qronos")
    parser.add_argument("--calibration-steps", type=int, default=4)
    parser.add_argument("--calibration-batch-size", type=int, default=64)
    parser.add_argument("--calibration-seq-len", type=int, default=256)
    parser.add_argument("--calibration-tokens", type=int, default=32768)
    parser.add_argument("--eval-steps", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-seq-len", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--data-split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--calibration-split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--max-dataset-tokens", type=int, default=4_000_000)
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--sparsity-n", type=int, default=2)
    parser.add_argument("--sparsity-m", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--cae-alpha", type=float, default=0.25)
    parser.add_argument("--gd-steps", type=int, default=1)
    parser.add_argument("--gd-lr", type=float, default=0.25)
    parser.add_argument("--gd-chunk-tokens", type=int, default=8192)
    parser.add_argument("--ce-chunk-tokens", type=int, default=32768)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--save-sparse-model", action="store_true")
    args = parser.parse_args(argv)
    config = Sparse24EvalConfig(
        output_dir=args.output_dir,
        model_name=args.model_name,
        methods=_parse_methods(args.methods),
        calibration_steps=args.calibration_steps,
        calibration_batch_size=args.calibration_batch_size,
        calibration_seq_len=args.calibration_seq_len,
        calibration_tokens=args.calibration_tokens,
        eval_steps=args.eval_steps,
        eval_batch_size=args.eval_batch_size,
        eval_seq_len=args.eval_seq_len,
        dtype=args.dtype,
        data_split=args.data_split,
        calibration_split=args.calibration_split,
        max_dataset_tokens=args.max_dataset_tokens,
        damp=args.damp,
        blocksize=args.blocksize,
        sparsity_n=args.sparsity_n,
        sparsity_m=args.sparsity_m,
        alpha=args.alpha,
        cae_alpha=args.cae_alpha,
        gd_steps=args.gd_steps,
        gd_lr=args.gd_lr,
        gd_chunk_tokens=args.gd_chunk_tokens,
        ce_chunk_tokens=args.ce_chunk_tokens,
        include_baseline=not args.skip_baseline,
        save_sparse_model=args.save_sparse_model,
    )
    print(json.dumps(run_sparse24_eval(config), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
