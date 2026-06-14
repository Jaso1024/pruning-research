from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from eval_da2k import (
    MODEL_CONFIGS,
    add_pair,
    empty_counts,
    finalize_counts,
    point_value,
    resolve_device,
    scene_from_path,
)
from eval_gelu_relu_compensation_da2k import (
    infer_depth,
    load_calibration_tensors,
    load_model,
    selected_annotations,
    transformer_mlp_names,
    write_summary,
)
from eval_relu_strikes_da2k import (
    ActivationSpec,
    activation_spec,
    install_mlp_activation,
    install_stage2,
)


PARTITION_MODES = {
    "contiguous",
    "weight_kmeans",
    "activation_kmeans",
    "activation_rank",
    "balanced_weight_kmeans",
    "balanced_activation_kmeans",
}
ROUTING_MODES = {"oracle", "router", "channel_oracle"}
SCORE_MODES = {"activation", "weighted_activation", "max_activation", "weighted_max_activation"}


@dataclass(frozen=True)
class MoEficationConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    activation: str = "relu"
    stage2: str = "none"
    stage2_shift: float = 0.0
    summary_json: Path | None = None
    variant_key: str = ""
    state_dict: Path | None = None
    partition: str = "contiguous"
    routing: str = "oracle"
    score: str = "weighted_activation"
    num_experts: int = 16
    top_k: int = 4
    calibration_images: int = 8
    calibration_tokens: int = 4096
    max_images: int = 32
    max_pairs: int = 0
    scene_type: str = ""
    router_steps: int = 200
    router_lr: float = 1e-3
    router_hidden: int = 0
    router_batch_tokens: int = 2048
    kmeans_iters: int = 25
    seed: int = 89
    log_every: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.summary_json is not None:
            object.__setattr__(self, "summary_json", Path(self.summary_json))
        if self.state_dict is not None:
            object.__setattr__(self, "state_dict", Path(self.state_dict))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.partition not in PARTITION_MODES:
            raise ValueError(f"unknown partition mode: {self.partition}")
        if self.routing not in ROUTING_MODES:
            raise ValueError(f"unknown routing mode: {self.routing}")
        if self.score not in SCORE_MODES:
            raise ValueError(f"unknown score mode: {self.score}")
        if self.stage2 not in {"none", "norm2", "norm12"}:
            raise ValueError("stage2 must be one of none, norm2, norm12")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if self.calibration_images <= 0:
            raise ValueError("calibration_images must be positive")
        if self.calibration_tokens <= 0:
            raise ValueError("calibration_tokens must be positive")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        if self.router_steps < 0:
            raise ValueError("router_steps must be non-negative")
        if self.router_lr <= 0.0:
            raise ValueError("router_lr must be positive")
        if self.router_hidden < 0:
            raise ValueError("router_hidden must be non-negative")
        if self.router_batch_tokens <= 0:
            raise ValueError("router_batch_tokens must be positive")
        if self.kmeans_iters <= 0:
            raise ValueError("kmeans_iters must be positive")


class RouterMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int) -> None:
        super().__init__()
        if hidden_features > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_features, out_features),
            )
        else:
            self.net = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEifiedMlp(nn.Module):
    def __init__(
        self,
        source: nn.Module,
        *,
        group_ids: torch.Tensor,
        top_k: int,
        routing: str,
        score: str,
        router: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = source.fc1
        self.act = source.act
        self.fc2 = source.fc2
        self.drop = source.drop
        self.top_k = int(top_k)
        self.routing = routing
        self.score = score
        self.router = router
        self.register_buffer("group_ids", group_ids.detach().long().clone(), persistent=True)
        col_norm = self.fc2.weight.detach().float().norm(dim=0).clamp_min(1e-8)
        self.register_buffer("channel_weight", col_norm, persistent=False)
        self.reset_moe_stats()

    @property
    def num_experts(self) -> int:
        return int(self.group_ids.max().item()) + 1

    def reset_moe_stats(self) -> None:
        self.tokens_seen = 0
        self.channels_selected = 0
        self.expert_histogram = torch.zeros(self.num_experts, dtype=torch.long)

    def _channel_values(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.score in {"weighted_activation", "weighted_max_activation"}:
            return hidden.float().abs() * self.channel_weight.to(device=hidden.device).unsqueeze(0)
        return hidden.float().abs()

    def _oracle_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        values = self._channel_values(hidden)
        return expert_scores_from_values(
            values,
            self.group_ids.to(device=hidden.device),
            score_mode=self.score,
            num_experts=self.num_experts,
        )

    def _channel_oracle_mask(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self._channel_values(hidden)
        hidden_features = int(values.shape[1])
        channel_count = max(1, min(hidden_features, round(hidden_features * self.top_k / self.num_experts)))
        selected_channels = values.topk(channel_count, dim=1).indices
        group_ids = self.group_ids.to(device=hidden.device)
        selected_groups = group_ids[selected_channels]
        scores = torch.zeros(
            (hidden.shape[0], self.num_experts),
            device=hidden.device,
            dtype=torch.bool,
        )
        scores.scatter_(1, selected_groups, True)
        channel_mask = scores[:, group_ids]
        return channel_mask, scores

    def _router_scores(self, x_flat: torch.Tensor) -> torch.Tensor:
        if self.router is None:
            raise RuntimeError("routing='router' requires a trained router")
        return self.router(x_flat.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])
        hidden = self.act(self.fc1(x)).reshape(-1, self.group_ids.numel())
        hidden = self.drop(hidden)

        if self.routing == "channel_oracle":
            channel_mask, expert_mask = self._channel_oracle_mask(hidden)
        elif self.routing == "oracle":
            scores = self._oracle_scores(hidden)
            selected_experts = scores.topk(self.top_k, dim=1).indices
            expert_mask = torch.zeros_like(scores, dtype=torch.bool)
            expert_mask.scatter_(1, selected_experts, True)
            channel_mask = expert_mask[:, self.group_ids.to(device=hidden.device)]
        elif self.routing == "router":
            scores = self._router_scores(x_flat)
            selected_experts = scores.topk(self.top_k, dim=1).indices
            expert_mask = torch.zeros_like(scores, dtype=torch.bool)
            expert_mask.scatter_(1, selected_experts, True)
            channel_mask = expert_mask[:, self.group_ids.to(device=hidden.device)]
        else:
            raise RuntimeError(f"unknown routing mode: {self.routing}")
        hidden = hidden * channel_mask.to(dtype=hidden.dtype)

        with torch.no_grad():
            self.tokens_seen += int(hidden.shape[0])
            self.channels_selected += int(channel_mask.sum().item())
            hist = expert_mask.detach().sum(dim=0).cpu().long()
            self.expert_histogram += hist[: self.num_experts]

        out = self.fc2(hidden.reshape(*original_shape[:-1], -1))
        out = self.drop(out)
        return out

    def moe_summary(self) -> dict[str, Any]:
        hidden = int(self.group_ids.numel())
        tokens = max(self.tokens_seen, 1)
        selected_fraction = self.channels_selected / float(tokens * hidden)
        sizes = torch.bincount(self.group_ids.detach().cpu(), minlength=self.num_experts)
        return {
            "routing": self.routing,
            "score": self.score,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "hidden_features": hidden,
            "tokens_seen": int(self.tokens_seen),
            "selected_channel_fraction": selected_fraction,
            "nominal_expert_fraction": self.top_k / float(self.num_experts),
            "expert_sizes": [int(v) for v in sizes.tolist()],
            "expert_histogram": [int(v) for v in self.expert_histogram.tolist()],
        }


def parse_activation_from_summary(summary_path: Path, variant_key: str) -> tuple[ActivationSpec, str, float]:
    summary = json.loads(summary_path.read_text())
    metadata = summary["variants"][variant_key]["metadata"]
    activation = ActivationSpec(**metadata["activation"])
    stage2 = metadata["stage2"]
    stage2_shift = float(metadata.get("stage2_shift", 0.0))
    return activation, stage2, stage2_shift


def sample_rows(tensor: torch.Tensor, limit: int, generator: torch.Generator) -> torch.Tensor:
    if tensor.shape[0] <= limit:
        return tensor
    indices = torch.randperm(tensor.shape[0], generator=generator)[:limit]
    return tensor.index_select(0, indices)


def kmeans_assign(
    features: torch.Tensor,
    *,
    num_clusters: int,
    iterations: int,
    seed: int,
) -> torch.Tensor:
    features = F.normalize(features.float(), dim=1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if features.shape[0] < num_clusters:
        raise ValueError("num_clusters exceeds number of feature rows")
    init = torch.randperm(features.shape[0], generator=generator)[:num_clusters]
    centroids = features.index_select(0, init).clone()
    labels = torch.zeros(features.shape[0], dtype=torch.long)

    for _ in range(iterations):
        distances = torch.cdist(features, centroids)
        labels = distances.argmin(dim=1)
        for cluster in range(num_clusters):
            mask = labels == cluster
            if bool(mask.any()):
                centroids[cluster] = features[mask].mean(dim=0)
            else:
                replacement = int(torch.randint(0, features.shape[0], (1,), generator=generator).item())
                centroids[cluster] = features[replacement]
        centroids = F.normalize(centroids, dim=1)
    return labels


def balanced_assign_to_centroids(features: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(F.normalize(features.float(), dim=1), F.normalize(centroids.float(), dim=1))
    row_count, cluster_count = distances.shape
    base_quota = row_count // cluster_count
    remainder = row_count % cluster_count
    quotas = torch.full((cluster_count,), base_quota, dtype=torch.long)
    quotas[:remainder] += 1
    labels = torch.full((row_count,), -1, dtype=torch.long)
    counts = torch.zeros((cluster_count,), dtype=torch.long)
    ranked_clusters = distances.argsort(dim=1)
    if cluster_count > 1:
        nearest = distances.gather(1, ranked_clusters[:, :1]).squeeze(1)
        second = distances.gather(1, ranked_clusters[:, 1:2]).squeeze(1)
        row_order = torch.argsort(second - nearest, descending=True)
    else:
        row_order = torch.arange(row_count)

    # Approximate balanced assignment: lock high-confidence rows first, then
    # fall back to each row's next nearest centroid when a quota is full.
    for row in row_order.tolist():
        for cluster in ranked_clusters[row].tolist():
            if counts[cluster] < quotas[cluster]:
                labels[row] = cluster
                counts[cluster] += 1
                break
    if bool((labels < 0).any()):
        raise RuntimeError("balanced assignment failed to fill all rows")
    return labels


def balanced_kmeans_assign(
    features: torch.Tensor,
    *,
    num_clusters: int,
    iterations: int,
    seed: int,
) -> torch.Tensor:
    features = F.normalize(features.float(), dim=1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if features.shape[0] < num_clusters:
        raise ValueError("num_clusters exceeds number of feature rows")
    init = torch.randperm(features.shape[0], generator=generator)[:num_clusters]
    centroids = features.index_select(0, init).clone()
    labels = torch.zeros(features.shape[0], dtype=torch.long)
    for _ in range(iterations):
        labels = balanced_assign_to_centroids(features, centroids)
        for cluster in range(num_clusters):
            mask = labels == cluster
            if bool(mask.any()):
                centroids[cluster] = features[mask].mean(dim=0)
        centroids = F.normalize(centroids, dim=1)
    return balanced_assign_to_centroids(features, centroids)


def contiguous_partition(hidden_features: int, num_experts: int) -> torch.Tensor:
    ids = torch.arange(hidden_features, dtype=torch.long) * num_experts // hidden_features
    return ids.clamp_max(num_experts - 1)


def weight_kmeans_partition(
    mlp: nn.Module,
    num_experts: int,
    iterations: int,
    seed: int,
    *,
    balanced: bool,
) -> torch.Tensor:
    fc1 = F.normalize(mlp.fc1.weight.detach().float(), dim=1)
    fc2 = F.normalize(mlp.fc2.weight.detach().float().T, dim=1)
    features = torch.cat([fc1, fc2], dim=1).cpu()
    assign = balanced_kmeans_assign if balanced else kmeans_assign
    return assign(features, num_clusters=num_experts, iterations=iterations, seed=seed)


def activation_kmeans_partition(
    activations: torch.Tensor,
    num_experts: int,
    iterations: int,
    seed: int,
    *,
    balanced: bool,
) -> torch.Tensor:
    # Cluster channels by token activation profile. Binary profiles emphasize
    # co-activation patterns; magnitudes still matter through channel frequency.
    features = (activations.float() > 0).T.contiguous().cpu()
    assign = balanced_kmeans_assign if balanced else kmeans_assign
    return assign(features, num_clusters=num_experts, iterations=iterations, seed=seed)


def activation_rank_partition(
    activations: torch.Tensor,
    channel_weight: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    hidden = activations.shape[1]
    order = (activations.float().abs() * channel_weight.unsqueeze(0)).mean(dim=0).argsort(descending=True)
    group_ids = torch.empty(hidden, dtype=torch.long)
    ranked_groups = contiguous_partition(hidden, num_experts)
    group_ids[order] = ranked_groups
    return group_ids


def collect_mlp_calibration(
    *,
    model: nn.Module,
    mlp_names: list[str],
    calibration_tensors: list[torch.Tensor],
    calibration_tokens: int,
    device: torch.device,
    seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    records: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"inputs": [], "activations": []} for name in mlp_names
    }
    handles = []

    for name in mlp_names:
        module = model.get_submodule(name)

        def make_hook(module_name: str):
            def hook(module, inputs, _output) -> None:
                x = inputs[0].detach()
                hidden = module.act(module.fc1(x)).detach()
                records[module_name]["inputs"].append(x.flatten(0, 1).float().cpu())
                records[module_name]["activations"].append(hidden.flatten(0, 1).float().cpu())

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    try:
        with torch.inference_mode():
            for tensor in tqdm(calibration_tensors, desc="collect MoE calibration", unit="image"):
                _ = model(tensor.to(device=device, non_blocking=True))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
    finally:
        for handle in handles:
            handle.remove()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    packed: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensors in records.items():
        inputs = torch.cat(tensors["inputs"], dim=0)
        activations = torch.cat(tensors["activations"], dim=0)
        if inputs.shape[0] > calibration_tokens:
            indices = torch.randperm(inputs.shape[0], generator=generator)[:calibration_tokens]
            inputs = inputs.index_select(0, indices)
            activations = activations.index_select(0, indices)
        packed[name] = {"inputs": inputs, "activations": activations}
    return packed


def expert_scores_from_activations(
    activations: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    channel_weight: torch.Tensor,
    score_mode: str,
    num_experts: int,
) -> torch.Tensor:
    if score_mode in {"weighted_activation", "weighted_max_activation"}:
        values = activations.float().abs() * channel_weight.unsqueeze(0)
    else:
        values = activations.float().abs()
    return expert_scores_from_values(values, group_ids, score_mode=score_mode, num_experts=num_experts)


def expert_scores_from_values(
    values: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    score_mode: str,
    num_experts: int,
) -> torch.Tensor:
    scores = torch.zeros((values.shape[0], num_experts), device=values.device, dtype=torch.float32)
    expanded_groups = group_ids.to(device=values.device).expand(values.shape[0], -1)
    if score_mode in {"max_activation", "weighted_max_activation"}:
        if hasattr(scores, "scatter_reduce_"):
            scores.scatter_reduce_(1, expanded_groups, values.float(), reduce="amax", include_self=True)
        else:
            for expert in range(num_experts):
                mask = expanded_groups == expert
                scores[:, expert] = values.masked_fill(~mask, 0.0).amax(dim=1)
    else:
        scores.scatter_add_(1, expanded_groups, values.float())
    return scores


def fit_router(
    *,
    inputs: torch.Tensor,
    target_scores: torch.Tensor,
    hidden_features: int,
    steps: int,
    lr: float,
    batch_tokens: int,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    router = RouterMLP(inputs.shape[1], target_scores.shape[1], hidden_features).to(device=device)
    if steps == 0:
        return router.cpu(), {"steps": 0, "initial_mse": None, "final_mse": None, "topk_recall": None}

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = inputs.to(device=device, dtype=torch.float32)
    y = target_scores.to(device=device, dtype=torch.float32)
    y = y / y.max(dim=1, keepdim=True).values.clamp_min(1e-6)
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr)
    batch_size = min(batch_tokens, x.shape[0])
    with torch.no_grad():
        initial_mse = F.mse_loss(router(x), y).item()
    losses: list[float] = []
    for step in range(steps):
        if batch_size < x.shape[0]:
            idx = torch.randint(0, x.shape[0], (batch_size,), device=device)
            xb = x.index_select(0, idx)
            yb = y.index_select(0, idx)
        else:
            xb = x
            yb = y
        pred = router(xb)
        loss = F.mse_loss(pred, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {0, steps - 1}:
            losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        pred = router(x)
        final_mse = F.mse_loss(pred, y).item()
        true_top = y.topk(min(4, y.shape[1]), dim=1).indices
        pred_top = pred.topk(min(4, pred.shape[1]), dim=1).indices
        pred_mask = torch.zeros_like(y, dtype=torch.bool).scatter_(1, pred_top, True)
        recall = pred_mask.gather(1, true_top).float().mean().item()
    summary = {
        "steps": steps,
        "lr": lr,
        "batch_tokens": int(batch_size),
        "initial_mse": initial_mse,
        "final_mse": final_mse,
        "first_last_batch_losses": losses,
        "top4_recall": recall,
    }
    return router.cpu(), summary


def install_moe_layers(
    *,
    model: nn.Module,
    mlp_names: list[str],
    calibration: dict[str, dict[str, torch.Tensor]],
    config: MoEficationConfig,
    device: torch.device,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for layer_index, name in enumerate(tqdm(mlp_names, desc="install MoEified MLPs", unit="mlp")):
        mlp = model.get_submodule(name)
        activations = calibration[name]["activations"]
        channel_weight = mlp.fc2.weight.detach().float().norm(dim=0).cpu().clamp_min(1e-8)
        if config.partition == "contiguous":
            group_ids = contiguous_partition(mlp.fc1.out_features, config.num_experts)
        elif config.partition == "activation_rank":
            group_ids = activation_rank_partition(
                activations,
                channel_weight,
                config.num_experts,
            )
        elif config.partition in {"weight_kmeans", "balanced_weight_kmeans"}:
            group_ids = weight_kmeans_partition(
                mlp,
                config.num_experts,
                iterations=config.kmeans_iters,
                seed=config.seed + 997 * layer_index,
                balanced=config.partition.startswith("balanced_"),
            )
        elif config.partition in {"activation_kmeans", "balanced_activation_kmeans"}:
            group_ids = activation_kmeans_partition(
                activations,
                config.num_experts,
                iterations=config.kmeans_iters,
                seed=config.seed + 997 * layer_index,
                balanced=config.partition.startswith("balanced_"),
            )
        else:
            raise RuntimeError(f"unknown partition mode: {config.partition}")

        target_scores = expert_scores_from_activations(
            activations,
            group_ids,
            channel_weight=channel_weight,
            score_mode=config.score,
            num_experts=config.num_experts,
        )
        router = None
        router_summary = None
        if config.routing == "router":
            router, router_summary = fit_router(
                inputs=calibration[name]["inputs"],
                target_scores=target_scores,
                hidden_features=config.router_hidden,
                steps=config.router_steps,
                lr=config.router_lr,
                batch_tokens=config.router_batch_tokens,
                seed=config.seed + 1301 * layer_index,
                device=device,
            )

        moe_mlp = MoEifiedMlp(
            mlp,
            group_ids=group_ids,
            top_k=config.top_k,
            routing=config.routing,
            score=config.score,
            router=router,
        ).to(device=device)
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, moe_mlp)

        oracle_top = target_scores.topk(config.top_k, dim=1).indices
        histogram = torch.bincount(oracle_top.reshape(-1), minlength=config.num_experts)
        sizes = torch.bincount(group_ids, minlength=config.num_experts)
        summaries[name] = {
            "partition": config.partition,
            "routing": config.routing,
            "score": config.score,
            "num_experts": config.num_experts,
            "top_k": config.top_k,
            "expert_sizes": [int(v) for v in sizes.tolist()],
            "calibration_oracle_expert_histogram": [int(v) for v in histogram.tolist()],
            "calibration_tokens": int(activations.shape[0]),
            "router": router_summary,
        }
    return summaries


def reset_moe_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, MoEifiedMlp):
            module.reset_moe_stats()


def collect_moe_stats(model: nn.Module) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for name, module in model.named_modules():
        if isinstance(module, MoEifiedMlp):
            stats[name] = module.moe_summary()
    if stats:
        selected = [row["selected_channel_fraction"] for row in stats.values()]
        stats["_mean"] = {
            "selected_channel_fraction": sum(selected) / len(selected),
            "nominal_expert_fraction": next(iter(stats.values()))["nominal_expert_fraction"],
        }
    return stats


def evaluate_da2k_model_with_flush(
    *,
    model: nn.Module,
    dataset_root: Path,
    items: list[tuple[str, list[dict[str, Any]]]],
    input_size: int,
    device: torch.device,
    log_every: int,
) -> dict[str, Any]:
    total = empty_counts()
    by_scene = defaultdict(empty_counts)
    missing_images: list[str] = []
    started = time.monotonic()
    model.eval()
    reset_moe_stats(model)

    for index, (relative_path, pairs) in enumerate(items, start=1):
        image = cv2.imread(str(dataset_root / relative_path))
        if image is None:
            missing_images.append(str(dataset_root / relative_path))
            continue
        depth = infer_depth(model, image, input_size, device)
        scene = scene_from_path(relative_path)
        for pair in pairs:
            if pair.get("closer_point") != "point1":
                raise ValueError(f"unsupported closer_point in {relative_path}: {pair}")
            d1 = point_value(depth, pair["point1"])
            d2 = point_value(depth, pair["point2"])
            add_pair(total, d1, d2)
            add_pair(by_scene[scene], d1, d2)
        del depth
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if log_every > 0 and (index % log_every == 0 or index == len(items)):
            print(f"evaluated {index}/{len(items)} images", flush=True)

    return {
        "metadata": {
            "images_requested": len(items),
            "missing_images": missing_images,
            "elapsed_seconds": time.monotonic() - started,
        },
        "overall": finalize_counts(total),
        "by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(by_scene.items())},
        "moe_stats": collect_moe_stats(model),
    }


def load_moefication_base(config: MoEficationConfig, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    if config.summary_json is not None:
        activation, stage2, stage2_shift = parse_activation_from_summary(config.summary_json, config.variant_key)
    else:
        activation = activation_spec(config.activation)
        stage2 = config.stage2
        stage2_shift = config.stage2_shift

    model = load_model(config.encoder, config.checkpoint, device)
    changed_mlp = install_mlp_activation(model, activation)
    changed_stage2 = install_stage2(model, mode=stage2, shift=stage2_shift)
    load_summary: dict[str, Any] = {
        "activation": asdict(activation),
        "stage2": stage2,
        "stage2_shift": stage2_shift,
        "changed_mlp_modules": changed_mlp,
        "changed_stage2_modules": changed_stage2,
    }
    if config.state_dict is not None:
        state = torch.load(config.state_dict, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"state dict mismatch: missing={missing}, unexpected={unexpected}")
        load_summary["state_dict"] = str(config.state_dict)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model, load_summary


def run(config: MoEficationConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    started = time.monotonic()
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    selected_items = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=config.max_images,
        max_pairs=config.max_pairs,
    )
    if len(selected_items) < config.calibration_images:
        raise RuntimeError(f"selected {len(selected_items)} images, but calibration_images={config.calibration_images}")

    model, load_summary = load_moefication_base(config, device)
    mlp_names = transformer_mlp_names(model)
    calibration_tensors, calibration_paths = load_calibration_tensors(
        model,
        dataset_root=config.dataset_root,
        items=selected_items,
        input_size=config.input_size,
        device=device,
        limit=config.calibration_images,
    )
    calibration_tensors = [tensor.detach().cpu() for tensor in calibration_tensors]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result: dict[str, Any] = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "images_selected": len(selected_items),
            "pairs_selected": sum(len(pairs) for _path, pairs in selected_items),
            "mlp_names": mlp_names,
            "calibration_relative_paths": calibration_paths,
            "loaded_model": load_summary,
            "note": (
                "MoEification prototype. Hidden FFN channels are partitioned into experts; top-k experts are "
                "selected per token. This script masks dense PyTorch activations for accuracy measurement, "
                "so selected_channel_fraction is the target compute fraction for a later sparse kernel, not a "
                "wall-clock speed claim."
            ),
        },
        "variants": {},
    }
    summary_path = config.output_dir / "summary.json"
    write_summary(summary_path, result)

    calibration = collect_mlp_calibration(
        model=model,
        mlp_names=mlp_names,
        calibration_tensors=calibration_tensors,
        calibration_tokens=config.calibration_tokens,
        device=device,
        seed=config.seed,
    )
    del calibration_tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    install_summary = install_moe_layers(
        model=model,
        mlp_names=mlp_names,
        calibration=calibration,
        config=config,
        device=device,
    )
    result["metadata"]["moe_install"] = install_summary
    write_summary(summary_path, result)

    evaluation = evaluate_da2k_model_with_flush(
        model=model,
        dataset_root=config.dataset_root,
        items=selected_items,
        input_size=config.input_size,
        device=device,
        log_every=config.log_every,
    )
    key = f"{config.partition}_{config.routing}_{config.num_experts}e_top{config.top_k}_{config.score}"
    result["variants"][key] = {
        "metadata": {
            "partition": config.partition,
            "routing": config.routing,
            "score": config.score,
            "num_experts": config.num_experts,
            "top_k": config.top_k,
        },
        "evaluation": evaluation,
    }
    result["metadata"]["elapsed_seconds"] = time.monotonic() - started
    write_summary(summary_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Depth Anything V2 MoEification / FFN expert sparsification prototype.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/moefication"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--stage2", choices=["none", "norm2", "norm12"], default="none")
    parser.add_argument("--stage2-shift", type=float, default=0.0)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--variant-key", default="")
    parser.add_argument("--state-dict", type=Path, default=None)
    parser.add_argument("--partition", choices=sorted(PARTITION_MODES), default="contiguous")
    parser.add_argument("--routing", choices=sorted(ROUTING_MODES), default="oracle")
    parser.add_argument("--score", choices=sorted(SCORE_MODES), default="weighted_activation")
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--calibration-images", type=int, default=8)
    parser.add_argument("--calibration-tokens", type=int, default=4096)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--router-steps", type=int, default=200)
    parser.add_argument("--router-lr", type=float, default=1e-3)
    parser.add_argument("--router-hidden", type=int, default=0)
    parser.add_argument("--router-batch-tokens", type=int, default=2048)
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=89)
    parser.add_argument("--log-every", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = MoEficationConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        activation=args.activation,
        stage2=args.stage2,
        stage2_shift=args.stage2_shift,
        summary_json=args.summary_json,
        variant_key=args.variant_key,
        state_dict=args.state_dict,
        partition=args.partition,
        routing=args.routing,
        score=args.score,
        num_experts=args.num_experts,
        top_k=args.top_k,
        calibration_images=args.calibration_images,
        calibration_tokens=args.calibration_tokens,
        max_images=args.max_images,
        max_pairs=args.max_pairs,
        scene_type=args.scene_type,
        router_steps=args.router_steps,
        router_lr=args.router_lr,
        router_hidden=args.router_hidden,
        router_batch_tokens=args.router_batch_tokens,
        kmeans_iters=args.kmeans_iters,
        seed=args.seed,
        log_every=args.log_every,
    )
    summary = run(config)
    print(json.dumps({name: row["evaluation"]["overall"] for name, row in summary["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
