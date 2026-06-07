from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class AffineScaffoldConfig:
    output_dir: str | Path
    num_pairs: int = 1024
    input_dim: int = 128
    output_dim: int = 128
    seed: int = 0
    dtype: str = "float32"
    input_scale: float = 1.0
    output_scale: float = 1.0
    task: str = "regression"
    solver: str = "auto"
    ridge: float = 0.0
    device: str = "cpu"


@dataclass(slots=True)
class AffinePruneEvalConfig:
    input_dir: str | Path
    output_dir: str | Path
    methods: tuple[str, ...] = ("random", "magnitude", "wanda", "squared_wanda", "exact_weight_loss", "exact_grad", "gptq", "qronos")
    prune_fractions: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)
    pruning_scope: str = "global"
    seed: int = 0
    device: str = "cpu"
    damp_percent: float = 0.01
    blocksize: int = 128
    percdamp: float = 1e-6
    cholesky_scale: float = 1e4
    num_blocks: int = 100
    use_activation_order: bool = True


@dataclass(slots=True)
class LayeredAffineScaffoldConfig:
    output_dir: str | Path
    layer_dims: tuple[int, ...] = (128, 128)
    num_pairs: int = 1024
    seed: int = 0
    dtype: str = "float32"
    input_scale: float = 1.0
    solver: str = "auto"
    ridge: float = 0.0
    device: str = "cpu"


@dataclass(slots=True)
class LayeredAffinePruneEvalConfig:
    input_dir: str | Path
    output_dir: str | Path
    methods: tuple[str, ...] = ("random", "magnitude", "wanda", "squared_wanda", "exact_weight_loss", "exact_grad", "gptq", "qronos")
    prune_fractions: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)
    pruning_scope: str = "global"
    seed: int = 0
    device: str = "cpu"
    damp_percent: float = 0.01
    blocksize: int = 128
    percdamp: float = 1e-6
    cholesky_scale: float = 1e4
    num_blocks: int = 100
    use_activation_order: bool = True


@dataclass(slots=True)
class RandomVectorPairs:
    inputs: torch.Tensor
    outputs: torch.Tensor
    labels: torch.Tensor | None = None


@dataclass(slots=True)
class AffineLeastSquaresSolution:
    a: torch.Tensor
    b: torch.Tensor
    mse: float
    residual_sum_squares: float
    rank: int


DEFAULT_AFFINE_PRUNING_METHODS = ("random", "magnitude", "wanda", "squared_wanda", "exact_weight_loss", "exact_grad", "gptq", "qronos")


def _resolve_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"fp64", "float64", "double"}:
        return torch.float64
    raise ValueError(f"unsupported dtype: {name}")


def _resolve_device(name: str) -> torch.device:
    normalized = name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda requested but unavailable")
    return device


def generate_random_pairs(
    *,
    num_pairs: int,
    input_dim: int,
    output_dim: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    input_scale: float = 1.0,
    output_scale: float = 1.0,
) -> RandomVectorPairs:
    if num_pairs <= 0 or input_dim <= 0 or output_dim <= 0:
        raise ValueError("num_pairs, input_dim, and output_dim must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    inputs = torch.randn((num_pairs, input_dim), generator=generator, dtype=dtype).mul_(float(input_scale))
    outputs = torch.randn((num_pairs, output_dim), generator=generator, dtype=dtype).mul_(float(output_scale))
    return RandomVectorPairs(inputs=inputs, outputs=outputs)


def generate_random_classification_pairs(
    *,
    num_pairs: int,
    input_dim: int,
    num_classes: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    input_scale: float = 1.0,
) -> RandomVectorPairs:
    if num_pairs <= 0 or input_dim <= 0 or num_classes <= 1:
        raise ValueError("num_pairs and input_dim must be positive; num_classes must be greater than 1")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    inputs = torch.randn((num_pairs, input_dim), generator=generator, dtype=dtype).mul_(float(input_scale))
    labels = torch.randint(num_classes, (num_pairs,), generator=generator, dtype=torch.long)
    outputs = F.one_hot(labels, num_classes=num_classes).to(dtype=dtype)
    return RandomVectorPairs(inputs=inputs, outputs=outputs, labels=labels)


def solve_affine_least_squares(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    *,
    solver: str = "lstsq",
    ridge: float = 0.0,
    device: torch.device | str = "cpu",
) -> AffineLeastSquaresSolution:
    if inputs.ndim != 2 or outputs.ndim != 2:
        raise ValueError("inputs and outputs must be 2D tensors")
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError(f"inputs/output pair count mismatch: {inputs.shape[0]} != {outputs.shape[0]}")
    solve_device = _resolve_device(str(device)) if not isinstance(device, torch.device) else device
    solve_dtype = torch.float64 if inputs.dtype == torch.float64 or outputs.dtype == torch.float64 else torch.float32
    x = inputs.detach().to(device=solve_device, dtype=solve_dtype)
    y = outputs.detach().to(device=solve_device, dtype=solve_dtype)
    ones = torch.ones((x.shape[0], 1), device=solve_device, dtype=solve_dtype)
    design = torch.cat((x, ones), dim=1)
    normalized_solver = solver.strip().lower().replace("-", "_")
    if normalized_solver == "auto":
        normalized_solver = "normal" if design.shape[0] >= design.shape[1] else "lstsq"
    if normalized_solver == "normal":
        gram = design.t().matmul(design)
        if float(ridge) > 0.0:
            gram.diagonal().add_(float(ridge))
        rhs = design.t().matmul(y)
        try:
            coeff = torch.linalg.solve(gram, rhs)
            rank = min(design.shape)
        except RuntimeError:
            result = torch.linalg.lstsq(design, y)
            coeff = result.solution
            rank = int(result.rank.item()) if result.rank.numel() else 0
    elif normalized_solver == "lstsq":
        result = torch.linalg.lstsq(design, y)
        coeff = result.solution
        rank = int(result.rank.item()) if result.rank.numel() else 0
    else:
        raise ValueError(f"unknown least-squares solver: {solver}")
    a = coeff[:-1].t().contiguous()
    b = coeff[-1].contiguous()
    prediction = x.matmul(a.t()).add(b)
    residual = prediction.sub(y)
    residual_sum_squares = float(residual.square().sum().item())
    mse = residual_sum_squares / max(y.numel(), 1)
    return AffineLeastSquaresSolution(
        a=a.to(device="cpu", dtype=inputs.dtype),
        b=b.to(device="cpu", dtype=inputs.dtype),
        mse=mse,
        residual_sum_squares=residual_sum_squares,
        rank=rank,
    )


def predict_affine(inputs: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return inputs.matmul(a.t()).add(b)


def affine_mse(inputs: torch.Tensor, outputs: torch.Tensor, a: torch.Tensor, b: torch.Tensor, *, device: torch.device | str = "cpu") -> float:
    eval_device = _resolve_device(str(device)) if not isinstance(device, torch.device) else device
    x = inputs.detach().to(device=eval_device, dtype=torch.float32)
    y = outputs.detach().to(device=eval_device, dtype=torch.float32)
    weight = a.detach().to(device=eval_device, dtype=torch.float32)
    bias = b.detach().to(device=eval_device, dtype=torch.float32)
    residual = predict_affine(x, weight, bias).sub(y)
    return float(residual.square().mean().item())


def affine_classification_metrics(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    eval_device = _resolve_device(str(device)) if not isinstance(device, torch.device) else device
    logits = predict_affine(
        inputs.detach().to(device=eval_device, dtype=torch.float32),
        a.detach().to(device=eval_device, dtype=torch.float32),
        b.detach().to(device=eval_device, dtype=torch.float32),
    )
    target = labels.detach().to(device=eval_device, dtype=torch.long)
    predictions = logits.argmax(dim=1)
    return {
        "cross_entropy": float(F.cross_entropy(logits, target).item()),
        "accuracy": float(predictions.eq(target).float().mean().item()),
    }


def affine_pruning_scores(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    score_device = _resolve_device(str(device)) if not isinstance(device, torch.device) else device
    x = inputs.detach().to(device=score_device, dtype=torch.float32)
    y = outputs.detach().to(device=score_device, dtype=torch.float32)
    weight = a.detach().to(device=score_device, dtype=torch.float32)
    bias = b.detach().to(device=score_device, dtype=torch.float32)
    residual = predict_affine(x, weight, bias).sub(y)
    input_second_moment = x.square().mean(dim=0).clamp_min(0.0)
    input_rms = input_second_moment.sqrt()
    generator = torch.Generator(device=score_device)
    generator.manual_seed(int(seed))
    exact_delta = weight.square().mul(input_second_moment.unsqueeze(0)).sub(
        2.0 * weight.mul(residual.t().matmul(x).div(max(x.shape[0], 1)))
    )
    grad = residual.t().matmul(x).mul(2.0 / max(y.numel(), 1))
    return {
        "random": torch.rand(weight.shape, generator=generator, device=score_device, dtype=torch.float32),
        "magnitude": weight.abs(),
        "wanda": weight.abs().mul(input_rms.unsqueeze(0)),
        "squared_wanda": weight.square().mul(input_second_moment.unsqueeze(0)),
        "exact_weight_loss": exact_delta,
        "exact_grad": weight.mul(grad).abs(),
    }


def _lowest_score_mask(score: torch.Tensor, *, fraction: float, pruning_scope: str) -> torch.Tensor:
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("prune fraction must be between 0 and 1")
    normalized = pruning_scope.strip().lower().replace("-", "_")
    if normalized == "global":
        flat = score.flatten()
        count = int(flat.numel() * float(fraction))
        if count <= 0:
            return torch.zeros_like(score, dtype=torch.bool)
        order = torch.argsort(flat, stable=True)
        mask = torch.zeros(flat.numel(), dtype=torch.bool, device=score.device)
        mask[order[:count]] = True
        return mask.reshape(score.shape)
    if normalized in {"per_output_row", "row"}:
        count = int(score.shape[1] * float(fraction))
        if count <= 0:
            return torch.zeros_like(score, dtype=torch.bool)
        order = torch.argsort(score, dim=1, stable=True)
        mask = torch.zeros_like(score, dtype=torch.bool)
        mask.scatter_(1, order[:, :count], True)
        return mask
    raise ValueError(f"unknown pruning_scope: {pruning_scope}")


def _affine_input_hessian(inputs: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    x = inputs.detach().to(device=device, dtype=torch.float32)
    return x.t().matmul(x).div(max(int(x.shape[0]), 1))


def _invert_hessian(hessian: torch.Tensor, damp_percent: float) -> tuple[torch.Tensor, float]:
    h = hessian.detach().to(dtype=torch.float32)
    diag = torch.diag(h)
    positive = diag[diag > 0]
    mean_diag = float(positive.mean().item()) if positive.numel() else 1.0
    damp = max(mean_diag * float(damp_percent), 1e-8)
    eye = torch.eye(h.shape[0], dtype=h.dtype, device=h.device)
    for multiplier in (1.0, 10.0, 100.0, 1000.0):
        try:
            chol = torch.linalg.cholesky(h + eye * (damp * multiplier))
            return torch.cholesky_inverse(chol), damp * multiplier
        except torch.linalg.LinAlgError:
            continue
    return torch.linalg.pinv(h + eye * (damp * 1000.0)), damp * 1000.0


def gptq_prune_weight(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    prune_fraction: float,
    pruning_scope: str,
    damp_percent: float = 0.01,
    blocksize: int = 128,
    use_activation_order: bool = True,
) -> tuple[torch.Tensor, dict[str, object]]:
    if weight.ndim != 2:
        raise ValueError("GPTQ pruning weight must be a 2D matrix")
    original = weight.detach().to(dtype=torch.float32)
    rows, columns = original.shape
    if tuple(hessian.shape) != (columns, columns):
        raise ValueError(f"hessian shape {tuple(hessian.shape)} does not match input columns {columns}")

    h = hessian.detach().to(device=original.device, dtype=torch.float32)
    if use_activation_order:
        order = torch.argsort(torch.diag(h), descending=True)
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(columns, dtype=order.dtype, device=order.device)
        h = h.index_select(0, order).index_select(1, order)
        working = original.index_select(1, order).clone()
    else:
        inv_order = None
        working = original.clone()

    h_inv, damp = _invert_hessian(h, damp_percent)
    mask = _lowest_score_mask(working.abs(), fraction=prune_fraction, pruning_scope=pruning_scope)
    pruned = torch.empty_like(working)
    blocksize = max(1, int(blocksize))

    for start in range(0, columns, blocksize):
        end = min(start + blocksize, columns)
        count = end - start
        block = working[:, start:end].clone()
        error_block = torch.zeros_like(block)
        h_inv_block = h_inv[start:end, start:end].float()
        for idx in range(count):
            column = start + idx
            values = block[:, idx]
            q = values.masked_fill(mask[:, column], 0.0)
            pruned[:, column] = q
            denom = h_inv_block[idx, idx].clamp_min(1e-12)
            error = (values - q) / denom
            block[:, idx:] -= error[:, None] * h_inv_block[idx, idx:][None, :]
            error_block[:, idx] = error
        if end < columns:
            working[:, end:] -= error_block @ h_inv[start:end, end:].float()

    if inv_order is not None:
        pruned = pruned.index_select(1, inv_order)
        mask = mask.index_select(1, inv_order)

    diff = original - pruned
    zeroed = int(mask.sum().item())
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
        "damp_percent": float(damp_percent),
        "blocksize": blocksize,
        "activation_order": bool(use_activation_order),
        "mean_abs_delta": float(diff.abs().mean().item()),
        "max_abs_delta": float(diff.abs().max().item()),
    }


def _validate_layer_dims(layer_dims: tuple[int, ...]) -> None:
    if len(layer_dims) < 2:
        raise ValueError("layer_dims must include input and output dimensions")
    if any(int(dim) <= 0 for dim in layer_dims):
        raise ValueError("all layer dimensions must be positive")


def _orthonormal_columns(rows: int, columns: int, *, seed: int) -> torch.Tensor:
    if columns <= 0:
        return torch.empty((rows, 0), dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn((rows, columns), generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.where(torch.diag(r) < 0, -1.0, 1.0)
    return q.mul(signs.unsqueeze(0))


def factor_affine_weight(a: torch.Tensor, layer_dims: tuple[int, ...]) -> list[torch.Tensor]:
    _validate_layer_dims(layer_dims)
    if tuple(a.shape) != (int(layer_dims[-1]), int(layer_dims[0])):
        raise ValueError(f"A shape {tuple(a.shape)} does not match layer dims {layer_dims}")
    if len(layer_dims) == 2:
        return [a.detach().clone()]

    original = a.detach().to(dtype=torch.float32)
    u, s, vh = torch.linalg.svd(original, full_matrices=False)
    tol = max(original.shape) * torch.finfo(original.dtype).eps * float(s.max().item()) if s.numel() else 0.0
    rank = int((s > tol).sum().item()) if s.numel() else 0
    bottleneck = min(int(dim) for dim in layer_dims)
    kept = min(rank, bottleneck, int(s.numel()))
    if kept == 0:
        return [
            torch.zeros((int(layer_dims[idx + 1]), int(layer_dims[idx])), dtype=a.dtype)
            for idx in range(len(layer_dims) - 1)
        ]

    root_s = s[:kept].sqrt()
    right = root_s[:, None] * vh[:kept, :]
    left = u[:, :kept] * root_s[None, :]
    bases = [
        _orthonormal_columns(int(hidden_dim), kept, seed=10_000 + idx)
        for idx, hidden_dim in enumerate(layer_dims[1:-1])
    ]
    weights = [bases[0].matmul(right)]
    for idx in range(len(bases) - 1):
        weights.append(bases[idx + 1].matmul(bases[idx].t()))
    weights.append(left.matmul(bases[-1].t()))
    return [weight.to(dtype=a.dtype) for weight in weights]


def predict_layered_affine(inputs: torch.Tensor, weights: list[torch.Tensor], bias: torch.Tensor) -> torch.Tensor:
    x = inputs
    for weight in weights:
        x = x.matmul(weight.t())
    return x.add(bias)


def layered_affine_activations(inputs: torch.Tensor, weights: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
    activations = [inputs]
    x = inputs
    for weight in weights:
        x = x.matmul(weight.t())
        activations.append(x)
    return activations[:-1], x


def layered_affine_classification_metrics(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    weights: list[torch.Tensor],
    bias: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    eval_device = _resolve_device(str(device)) if not isinstance(device, torch.device) else device
    x = inputs.detach().to(device=eval_device, dtype=torch.float32)
    target = labels.detach().to(device=eval_device, dtype=torch.long)
    ws = [weight.detach().to(device=eval_device, dtype=torch.float32) for weight in weights]
    b = bias.detach().to(device=eval_device, dtype=torch.float32)
    logits = predict_layered_affine(x, ws, b)
    return {
        "cross_entropy": float(F.cross_entropy(logits, target).item()),
        "accuracy": float(logits.argmax(dim=1).eq(target).float().mean().item()),
    }


def _load_layered_affine_artifacts(input_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor, str]:
    root = Path(input_dir)
    pairs = torch.load(root / "pairs.pt", map_location="cpu", weights_only=False)
    network = torch.load(root / "network.pt", map_location="cpu", weights_only=False)
    metadata = pairs.get("metadata", {})
    return pairs["inputs"], pairs["outputs"], pairs["labels"], network["weights"], network["bias"], str(metadata.get("task", "classification"))


def _downstream_matrices(weights: list[torch.Tensor]) -> list[torch.Tensor]:
    downstream: list[torch.Tensor] = []
    current = torch.eye(weights[-1].shape[0], dtype=weights[-1].dtype, device=weights[-1].device)
    for idx in range(len(weights) - 1, -1, -1):
        downstream.append(current)
        current = current.matmul(weights[idx])
    downstream.reverse()
    return downstream


def _layered_score(
    *,
    method: str,
    weight: torch.Tensor,
    activation: torch.Tensor,
    residual: torch.Tensor,
    downstream: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    if method == "random":
        return torch.rand(weight.shape, generator=generator, device=weight.device, dtype=torch.float32)
    if method == "magnitude":
        return weight.abs()
    second = activation.square().mean(dim=0).clamp_min(0.0)
    if method == "wanda":
        return weight.abs().mul(second.sqrt().unsqueeze(0))
    if method == "squared_wanda":
        return weight.square().mul(second.unsqueeze(0))
    projected_residual = residual.matmul(downstream)
    if method == "exact_weight_loss":
        cross = projected_residual.t().matmul(activation).div(max(int(activation.shape[0]), 1))
        downstream_norm = downstream.square().sum(dim=0).unsqueeze(1)
        return weight.square().mul(downstream_norm).mul(second.unsqueeze(0)).sub(2.0 * weight.mul(cross))
    if method == "exact_grad":
        grad = projected_residual.t().matmul(activation).mul(2.0 / max(residual.numel(), 1))
        return weight.mul(grad).abs()
    raise ValueError(f"unknown layered affine scoring method: {method}")


def _prune_layered_weights(
    *,
    weights: list[torch.Tensor],
    activations: list[torch.Tensor],
    outputs: torch.Tensor,
    bias: torch.Tensor,
    method: str,
    fraction: float,
    pruning_scope: str,
    seed: int,
    device: torch.device,
    damp_percent: float,
    blocksize: int,
    percdamp: float,
    cholesky_scale: float,
    num_blocks: int,
    use_activation_order: bool,
) -> tuple[list[torch.Tensor], int, list[dict[str, object]], bool]:
    normalized = method.strip().lower().replace("-", "_")
    pruned_weights = [weight.clone() for weight in weights]
    layer_stats: list[dict[str, object]] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    logits = predict_layered_affine(activations[0], weights, bias)
    residual = logits.sub(outputs)
    downstream = _downstream_matrices(weights)
    compensated = normalized in {"gptq", "qronos"}
    total_zeroed = 0

    for idx, weight in enumerate(weights):
        if normalized == "gptq":
            hessian = _affine_input_hessian(activations[idx], device=device)
            pruned, stats = gptq_prune_weight(
                weight,
                hessian,
                prune_fraction=float(fraction),
                pruning_scope=pruning_scope,
                damp_percent=damp_percent,
                blocksize=blocksize,
                use_activation_order=use_activation_order,
            )
            pruned_weights[idx] = pruned
            zeroed = int(stats["zeroed"])
        elif normalized == "qronos":
            from saliency.qronos_eval import qronos_prune_weight

            hessian = _affine_input_hessian(activations[idx], device=device)
            pruned, stats = qronos_prune_weight(
                weight,
                hessian,
                hessian,
                prune_fraction=float(fraction),
                pruning_scope=_qronos_scope(pruning_scope),
                percdamp=percdamp,
                cholesky_scale=cholesky_scale,
                num_blocks=num_blocks,
                use_activation_order=use_activation_order,
            )
            pruned_weights[idx] = pruned
            zeroed = int(stats["zeroed"])
        else:
            score = _layered_score(
                method=normalized,
                weight=weight,
                activation=activations[idx],
                residual=residual,
                downstream=downstream[idx],
                generator=generator,
            )
            mask = _lowest_score_mask(score, fraction=fraction, pruning_scope=pruning_scope)
            pruned_weights[idx].masked_fill_(mask.to(dtype=torch.bool), 0.0)
            zeroed = int(mask.sum().item())
            stats = {
                "format": "base_precision_pruned",
                "weights": int(weight.numel()),
                "zeroed": zeroed,
                "actual_zero_fraction": zeroed / max(int(weight.numel()), 1),
                "prune_fraction": float(fraction),
                "pruning_scope": pruning_scope,
            }
        total_zeroed += zeroed
        layer_stats.append(
            {
                "layer": idx,
                "rows": int(weight.shape[0]),
                "columns": int(weight.shape[1]),
                "weights": int(weight.numel()),
                "zeroed": zeroed,
                "actual_zero_fraction": zeroed / max(int(weight.numel()), 1),
                "stats": stats,
            }
        )

    return pruned_weights, total_zeroed, layer_stats, compensated


def _load_affine_artifacts(input_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, str]:
    root = Path(input_dir)
    pairs = torch.load(root / "pairs.pt", map_location="cpu", weights_only=False)
    weights = torch.load(root / "weights.pt", map_location="cpu", weights_only=False)
    metadata = pairs.get("metadata", {})
    return pairs["inputs"], pairs["outputs"], weights["A"], weights["b"], pairs.get("labels"), str(metadata.get("task", "regression"))


def _qronos_scope(pruning_scope: str) -> str:
    normalized = pruning_scope.strip().lower().replace("-", "_")
    if normalized == "global":
        return "per_matrix"
    if normalized in {"per_output_row", "row"}:
        return "per_output_row"
    raise ValueError(f"unknown pruning_scope: {pruning_scope}")


def _affine_result_row(
    *,
    method: str,
    fraction: float,
    pruning_scope: str,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    labels: torch.Tensor | None,
    a_numel: int,
    pruned_a: torch.Tensor,
    b: torch.Tensor,
    zeroed: int,
    baseline_mse: float,
    baseline_classification: dict[str, float],
    device: torch.device,
    compensation: dict[str, object] | None = None,
) -> dict[str, Any]:
    pruned_mse = affine_mse(inputs, outputs, pruned_a, b, device=device)
    pruned_classification = affine_classification_metrics(inputs, labels, pruned_a, b, device=device) if labels is not None else {}
    row: dict[str, Any] = {
        "method": method,
        "prune_fraction": float(fraction),
        "pruning_scope": pruning_scope,
        "baseline_mse": baseline_mse,
        "pruned_mse": pruned_mse,
        "delta_mse": pruned_mse - baseline_mse,
        "mse_ratio": pruned_mse / max(baseline_mse, 1e-30),
        "weights_seen": int(a_numel),
        "weights_zeroed": int(zeroed),
        "actual_zero_fraction": int(zeroed) / max(int(a_numel), 1),
        "compensated": compensation is not None,
    }
    if compensation is not None:
        row["compensation"] = compensation
    if pruned_classification:
        row.update(
            {
                "baseline_cross_entropy": baseline_classification["cross_entropy"],
                "baseline_accuracy": baseline_classification["accuracy"],
                "pruned_cross_entropy": pruned_classification["cross_entropy"],
                "pruned_accuracy": pruned_classification["accuracy"],
                "delta_cross_entropy": pruned_classification["cross_entropy"] - baseline_classification["cross_entropy"],
                "delta_accuracy": pruned_classification["accuracy"] - baseline_classification["accuracy"],
            }
        )
    return row


def run_affine_prune_eval(config: AffinePruneEvalConfig) -> dict[str, Any]:
    started = time.time()
    inputs, outputs, a, b, labels, task = _load_affine_artifacts(config.input_dir)
    device = _resolve_device(config.device)
    baseline_mse = affine_mse(inputs, outputs, a, b, device=device)
    baseline_classification = affine_classification_metrics(inputs, labels, a, b, device=device) if labels is not None else {}
    score_methods = {"random", "magnitude", "wanda", "squared_wanda", "exact_weight_loss", "exact_grad"}
    requested_methods = {method.strip().lower().replace("-", "_") for method in config.methods}
    all_scores = affine_pruning_scores(inputs, outputs, a, b, seed=config.seed, device=device) if requested_methods & score_methods else {}
    a_device = a.detach().to(device=device, dtype=torch.float32)
    b_device = b.detach().to(device=device, dtype=torch.float32)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hessian: torch.Tensor | None = None

    rows: list[dict[str, Any]] = []
    for method in config.methods:
        normalized_method = method.strip().lower().replace("-", "_")
        if normalized_method == "gptq":
            if hessian is None:
                hessian = _affine_input_hessian(inputs, device=device)
            for fraction in config.prune_fractions:
                pruned_a, stats = gptq_prune_weight(
                    a_device,
                    hessian,
                    prune_fraction=float(fraction),
                    pruning_scope=config.pruning_scope,
                    damp_percent=config.damp_percent,
                    blocksize=config.blocksize,
                    use_activation_order=config.use_activation_order,
                )
                rows.append(
                    _affine_result_row(
                        method=normalized_method,
                        fraction=float(fraction),
                        pruning_scope=config.pruning_scope,
                        inputs=inputs,
                        outputs=outputs,
                        labels=labels,
                        a_numel=a.numel(),
                        pruned_a=pruned_a,
                        b=b_device,
                        zeroed=int(stats["zeroed"]),
                        baseline_mse=baseline_mse,
                        baseline_classification=baseline_classification,
                        device=device,
                        compensation=stats,
                    )
                )
            continue
        if normalized_method == "qronos":
            if hessian is None:
                hessian = _affine_input_hessian(inputs, device=device)
            from saliency.qronos_eval import qronos_prune_weight

            for fraction in config.prune_fractions:
                pruned_a, stats = qronos_prune_weight(
                    a_device,
                    hessian,
                    hessian,
                    prune_fraction=float(fraction),
                    pruning_scope=_qronos_scope(config.pruning_scope),
                    percdamp=config.percdamp,
                    cholesky_scale=config.cholesky_scale,
                    num_blocks=config.num_blocks,
                    use_activation_order=config.use_activation_order,
                )
                rows.append(
                    _affine_result_row(
                        method=normalized_method,
                        fraction=float(fraction),
                        pruning_scope=config.pruning_scope,
                        inputs=inputs,
                        outputs=outputs,
                        labels=labels,
                        a_numel=a.numel(),
                        pruned_a=pruned_a,
                        b=b_device,
                        zeroed=int(stats["zeroed"]),
                        baseline_mse=baseline_mse,
                        baseline_classification=baseline_classification,
                        device=device,
                        compensation=stats,
                    )
                )
            continue
        score = all_scores.get(normalized_method)
        if score is None:
            raise ValueError(f"unknown affine pruning method: {method}")
        for fraction in config.prune_fractions:
            mask = _lowest_score_mask(score, fraction=fraction, pruning_scope=config.pruning_scope)
            pruned_a = a_device.clone()
            pruned_a.masked_fill_(mask.to(dtype=torch.bool), 0)
            zeroed = int(mask.sum().item())
            rows.append(
                _affine_result_row(
                    method=normalized_method,
                    fraction=float(fraction),
                    pruning_scope=config.pruning_scope,
                    inputs=inputs,
                    outputs=outputs,
                    labels=labels,
                    a_numel=a.numel(),
                    pruned_a=pruned_a,
                    b=b_device,
                    zeroed=zeroed,
                    baseline_mse=baseline_mse,
                    baseline_classification=baseline_classification,
                    device=device,
                )
            )

    rows.sort(key=lambda row: (row["prune_fraction"], row["method"]))
    summary = {
        "metadata": {
            **asdict(config),
            "methods": list(config.methods),
            "prune_fractions": [float(fraction) for fraction in config.prune_fractions],
            "input_dir": str(config.input_dir),
            "output_dir": str(output_dir),
            "bias_pruned": False,
            "task": task,
            "device": str(device),
            "model": "y = A x + b",
        },
        "baseline": {
            "mse": baseline_mse,
            "num_pairs": int(inputs.shape[0]),
            "input_dim": int(inputs.shape[1]),
            "output_dim": int(outputs.shape[1]),
            "weights": int(a.numel()),
            **baseline_classification,
        },
        "results": rows,
        "elapsed_seconds": time.time() - started,
    }
    with (output_dir / "prune_results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_affine_scaffold(config: AffineScaffoldConfig) -> dict[str, Any]:
    started = time.time()
    dtype = _resolve_dtype(config.dtype)
    device = _resolve_device(config.device)
    normalized_task = config.task.strip().lower()
    if normalized_task == "regression":
        pairs = generate_random_pairs(
            num_pairs=config.num_pairs,
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            seed=config.seed,
            dtype=dtype,
            input_scale=config.input_scale,
            output_scale=config.output_scale,
        )
    elif normalized_task == "classification":
        pairs = generate_random_classification_pairs(
            num_pairs=config.num_pairs,
            input_dim=config.input_dim,
            num_classes=config.output_dim,
            seed=config.seed,
            dtype=dtype,
            input_scale=config.input_scale,
        )
    else:
        raise ValueError(f"unknown affine scaffold task: {config.task}")
    solution = solve_affine_least_squares(pairs.inputs, pairs.outputs, solver=config.solver, ridge=config.ridge, device=device)
    classification_metrics = (
        affine_classification_metrics(pairs.inputs, pairs.labels, solution.a, solution.b, device=device)
        if pairs.labels is not None
        else {}
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_artifact = {
        "inputs": pairs.inputs,
        "outputs": pairs.outputs,
        "metadata": {
            "seed": config.seed,
            "num_pairs": config.num_pairs,
            "input_dim": config.input_dim,
            "output_dim": config.output_dim,
            "dtype": config.dtype,
            "input_scale": config.input_scale,
            "output_scale": config.output_scale,
            "task": normalized_task,
        },
    }
    if pairs.labels is not None:
        pair_artifact["labels"] = pairs.labels
    torch.save(pair_artifact, output_dir / "pairs.pt")
    torch.save(
        {
            "A": solution.a,
            "b": solution.b,
            "metadata": {
                "model": "y = A x + b",
                "orientation": "A has shape [output_dim, input_dim]",
            },
        },
        output_dir / "weights.pt",
    )

    summary = {
        "metadata": {**asdict(config), "output_dir": str(output_dir), "torch_dtype": str(dtype), "resolved_device": str(device)},
        "fit": {
            "mse": solution.mse,
            "residual_sum_squares": solution.residual_sum_squares,
            "rank": solution.rank,
            **classification_metrics,
        },
        "artifacts": {
            "pairs_pt": str(output_dir / "pairs.pt"),
            "weights_pt": str(output_dir / "weights.pt"),
            "summary_json": str(output_dir / "summary.json"),
        },
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_layered_affine_scaffold(config: LayeredAffineScaffoldConfig) -> dict[str, Any]:
    started = time.time()
    _validate_layer_dims(tuple(config.layer_dims))
    dtype = _resolve_dtype(config.dtype)
    device = _resolve_device(config.device)
    input_dim = int(config.layer_dims[0])
    output_dim = int(config.layer_dims[-1])
    pairs = generate_random_classification_pairs(
        num_pairs=config.num_pairs,
        input_dim=input_dim,
        num_classes=output_dim,
        seed=config.seed,
        dtype=dtype,
        input_scale=config.input_scale,
    )
    solution = solve_affine_least_squares(pairs.inputs, pairs.outputs, solver=config.solver, ridge=config.ridge, device=device)
    weights = factor_affine_weight(solution.a, tuple(config.layer_dims))
    metrics = layered_affine_classification_metrics(pairs.inputs, pairs.labels, weights, solution.b, device=device)
    logits = predict_layered_affine(
        pairs.inputs.to(device=device, dtype=torch.float32),
        [weight.to(device=device, dtype=torch.float32) for weight in weights],
        solution.b.to(device=device, dtype=torch.float32),
    )
    mse = float(logits.sub(pairs.outputs.to(device=device, dtype=torch.float32)).square().mean().item())

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_artifact = {
        "inputs": pairs.inputs,
        "outputs": pairs.outputs,
        "labels": pairs.labels,
        "metadata": {
            "seed": config.seed,
            "num_pairs": config.num_pairs,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "dtype": config.dtype,
            "input_scale": config.input_scale,
            "task": "classification",
        },
    }
    torch.save(pair_artifact, output_dir / "pairs.pt")
    torch.save(
        {
            "weights": weights,
            "bias": solution.b,
            "equivalent_A": solution.a,
            "metadata": {
                "model": "linear stack ending in softmax evaluation",
                "layer_dims": [int(dim) for dim in config.layer_dims],
                "bias_location": "final_logits",
            },
        },
        output_dir / "network.pt",
    )

    summary = {
        "metadata": {
            "output_dir": str(output_dir),
            "layer_dims": [int(dim) for dim in config.layer_dims],
            "num_pairs": int(config.num_pairs),
            "seed": int(config.seed),
            "dtype": config.dtype,
            "input_scale": float(config.input_scale),
            "solver": config.solver,
            "ridge": float(config.ridge),
            "resolved_device": str(device),
        },
        "fit": {
            "single_affine_mse": solution.mse,
            "layered_mse": mse,
            "residual_sum_squares": solution.residual_sum_squares,
            "rank": solution.rank,
            **metrics,
        },
        "artifacts": {
            "pairs_pt": str(output_dir / "pairs.pt"),
            "network_pt": str(output_dir / "network.pt"),
            "summary_json": str(output_dir / "summary.json"),
        },
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_layered_affine_prune_eval(config: LayeredAffinePruneEvalConfig) -> dict[str, Any]:
    started = time.time()
    inputs, outputs, labels, loaded_weights, bias, task = _load_layered_affine_artifacts(config.input_dir)
    device = _resolve_device(config.device)
    x = inputs.detach().to(device=device, dtype=torch.float32)
    y = outputs.detach().to(device=device, dtype=torch.float32)
    weights = [weight.detach().to(device=device, dtype=torch.float32) for weight in loaded_weights]
    b = bias.detach().to(device=device, dtype=torch.float32)
    activations, logits_no_bias = layered_affine_activations(x, weights)
    baseline_logits = logits_no_bias.add(b)
    target = labels.detach().to(device=device, dtype=torch.long)
    baseline_mse = float(baseline_logits.sub(y).square().mean().item())
    baseline_classification = {
        "cross_entropy": float(F.cross_entropy(baseline_logits, target).item()),
        "accuracy": float(baseline_logits.argmax(dim=1).eq(target).float().mean().item()),
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_weights = sum(int(weight.numel()) for weight in weights)
    rows: list[dict[str, Any]] = []

    for method in config.methods:
        normalized = method.strip().lower().replace("-", "_")
        for fraction in config.prune_fractions:
            pruned_weights, zeroed, layer_stats, compensated = _prune_layered_weights(
                weights=weights,
                activations=activations,
                outputs=y,
                bias=b,
                method=normalized,
                fraction=float(fraction),
                pruning_scope=config.pruning_scope,
                seed=config.seed,
                device=device,
                damp_percent=config.damp_percent,
                blocksize=config.blocksize,
                percdamp=config.percdamp,
                cholesky_scale=config.cholesky_scale,
                num_blocks=config.num_blocks,
                use_activation_order=config.use_activation_order,
            )
            pruned_logits = predict_layered_affine(x, pruned_weights, b)
            pruned_mse = float(pruned_logits.sub(y).square().mean().item())
            pruned_ce = float(F.cross_entropy(pruned_logits, target).item())
            pruned_acc = float(pruned_logits.argmax(dim=1).eq(target).float().mean().item())
            rows.append(
                {
                    "method": normalized,
                    "prune_fraction": float(fraction),
                    "pruning_scope": config.pruning_scope,
                    "baseline_mse": baseline_mse,
                    "pruned_mse": pruned_mse,
                    "delta_mse": pruned_mse - baseline_mse,
                    "mse_ratio": pruned_mse / max(baseline_mse, 1e-30),
                    "baseline_cross_entropy": baseline_classification["cross_entropy"],
                    "baseline_accuracy": baseline_classification["accuracy"],
                    "pruned_cross_entropy": pruned_ce,
                    "pruned_accuracy": pruned_acc,
                    "delta_cross_entropy": pruned_ce - baseline_classification["cross_entropy"],
                    "delta_accuracy": pruned_acc - baseline_classification["accuracy"],
                    "layers_seen": len(weights),
                    "weights_seen": total_weights,
                    "weights_zeroed": int(zeroed),
                    "actual_zero_fraction": int(zeroed) / max(total_weights, 1),
                    "compensated": compensated,
                    "layer_stats": layer_stats,
                }
            )

    rows.sort(key=lambda row: (row["prune_fraction"], row["method"]))
    summary = {
        "metadata": {
            "input_dir": str(config.input_dir),
            "output_dir": str(output_dir),
            "methods": list(config.methods),
            "prune_fractions": [float(fraction) for fraction in config.prune_fractions],
            "pruning_scope": config.pruning_scope,
            "seed": int(config.seed),
            "device": str(device),
            "damp_percent": float(config.damp_percent),
            "blocksize": int(config.blocksize),
            "percdamp": float(config.percdamp),
            "cholesky_scale": float(config.cholesky_scale),
            "num_blocks": int(config.num_blocks),
            "use_activation_order": bool(config.use_activation_order),
            "bias_pruned": False,
            "task": task,
            "model": "layered linear stack",
        },
        "baseline": {
            "mse": baseline_mse,
            "num_pairs": int(inputs.shape[0]),
            "input_dim": int(inputs.shape[1]),
            "output_dim": int(outputs.shape[1]),
            "layers": len(weights),
            "weights": total_weights,
            **baseline_classification,
        },
        "results": rows,
        "elapsed_seconds": time.time() - started,
    }
    with (output_dir / "prune_results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_layered_affine_suite_specs(specs: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for raw in specs.split(";"):
        item = raw.strip()
        if not item:
            continue
        name, pairs, dims = item.split(":", 2)
        cases.append(
            {
                "name": name.strip(),
                "num_pairs": int(pairs.strip()),
                "layer_dims": tuple(int(part.strip()) for part in dims.split(",") if part.strip()),
            }
        )
    if not cases:
        raise ValueError("at least one layered affine suite spec is required")
    return cases


def run_layered_affine_suite(
    *,
    output_dir: str | Path,
    specs: str,
    methods: tuple[str, ...] = DEFAULT_AFFINE_PRUNING_METHODS,
    prune_fractions: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50),
    pruning_scope: str = "global",
    seed: int = 17,
    dtype: str = "float32",
    solver: str = "auto",
    ridge: float = 0.0,
    device: str = "cpu",
    damp_percent: float = 0.01,
    blocksize: int = 128,
    percdamp: float = 1e-6,
    cholesky_scale: float = 1e4,
    num_blocks: int = 100,
    use_activation_order: bool = True,
) -> dict[str, Any]:
    started = time.time()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cases = parse_layered_affine_suite_specs(specs)
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for case_idx, case in enumerate(cases):
        case_dir = root / case["name"]
        scaffold = run_layered_affine_scaffold(
            LayeredAffineScaffoldConfig(
                output_dir=case_dir / "scaffold",
                layer_dims=case["layer_dims"],
                num_pairs=case["num_pairs"],
                seed=seed + case_idx,
                dtype=dtype,
                solver=solver,
                ridge=ridge,
                device=device,
            )
        )
        prune = run_layered_affine_prune_eval(
            LayeredAffinePruneEvalConfig(
                input_dir=case_dir / "scaffold",
                output_dir=case_dir / "prune_grid",
                methods=methods,
                prune_fractions=prune_fractions,
                pruning_scope=pruning_scope,
                seed=seed + 1000 + case_idx,
                device=device,
                damp_percent=damp_percent,
                blocksize=blocksize,
                percdamp=percdamp,
                cholesky_scale=cholesky_scale,
                num_blocks=num_blocks,
                use_activation_order=use_activation_order,
            )
        )
        summaries.append({"case": case, "scaffold": scaffold, "prune": prune})
        for row in prune["results"]:
            rows.append(
                {
                    "case": case["name"],
                    "layer_dims": list(case["layer_dims"]),
                    "num_pairs": int(case["num_pairs"]),
                    **row,
                }
            )

    summary = {
        "metadata": {
            "output_dir": str(root),
            "specs": specs,
            "methods": list(methods),
            "prune_fractions": [float(fraction) for fraction in prune_fractions],
            "pruning_scope": pruning_scope,
            "seed": int(seed),
            "dtype": dtype,
            "solver": solver,
            "ridge": float(ridge),
            "device": device,
            "case_count": len(cases),
        },
        "cases": summaries,
        "results": rows,
        "elapsed_seconds": time.time() - started,
    }
    (root / "suite_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (root / "suite_results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return summary
