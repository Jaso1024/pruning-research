from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path

import modal


AIRBENCH_REPO_URL = "https://github.com/KellerJordan/cifar10-airbench.git"
AIRBENCH_REF = "4c1b6d1e3889b037efadcfd5c0ea65b246592362"
AIRBENCH_ROOT = Path("/opt/cifar10-airbench")
CACHE_ROOT = Path("/cache")
RESULTS_ROOT = Path("/results")
AIRBENCH_CLONE_COMMAND = "git clone https://github.com/KellerJordan/cifar10-airbench.git /opt/cifar10-airbench"
AIRBENCH_CHECKOUT_COMMAND = "cd /opt/cifar10-airbench && git checkout 4c1b6d1e3889b037efadcfd5c0ea65b246592362"
AIRBENCH_INSTALL_COMMAND = "python -m pip install --no-cache-dir -e /opt/cifar10-airbench"

app = modal.App("cifar-airbench")
results_volume = modal.Volume.from_name("cifar-airbench-results", create_if_missing=True)
cache_volume = modal.Volume.from_name("cifar-airbench-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("torch==2.4.1", "torchvision==0.19.1")
    .run_commands(
        AIRBENCH_CLONE_COMMAND,
        AIRBENCH_CHECKOUT_COMMAND,
        AIRBENCH_INSTALL_COMMAND,
    )
)


@dataclass(frozen=True)
class AirBenchRunConfig:
    run_name: str = "airbench94_muon_h100_smoke"
    script_name: str = "airbench94_muon.py"
    n_runs: int = 1
    warmup: bool = True
    compile_mode: str = "max-autotune"
    seed: int = 0


@dataclass(frozen=True)
class AirBenchPartialCheckpointConfig:
    run_name: str = "airbench94_muon_three_quarter_checkpoint"
    script_name: str = "airbench94_muon.py"
    total_epochs: float = 8.0
    checkpoint_fraction: float = 0.75
    batch_size: int = 2000
    compile_mode: str = "max-autotune"
    seed: int = 0
    chosen_layer_name: str = "layers.2"


@dataclass(frozen=True)
class AirBenchProxyOptimizeConfig:
    run_name: str = "airbench94_muon_proxy_optimize_smoke"
    checkpoint_run_name: str = "airbench94_muon_three_quarter_checkpoint_20260527"
    chosen_layer_name: str = "layers.2"
    proxy_loss_name: str = "activation_l2"
    proxy_steps: int = 10
    batch_size: int = 2000
    learning_rate: float = 1e-4
    seed: int = 0
    compile_mode: str = ""


@dataclass(frozen=True)
class AirBenchPlateauCheckpointConfig:
    run_name: str = "airbench94_muon_plateau_500step"
    script_name: str = "airbench94_muon.py"
    train_steps: int = 500
    batch_size: int = 2000
    compile_mode: str = "max-autotune"
    seed: int = 0
    chosen_layer_name: str = "layers.2"


@dataclass(frozen=True)
class AirBenchPruneRecoveryConfig:
    run_name: str = "airbench94_muon_prune_recovery"
    checkpoint_run_name: str = "airbench94_muon_plateau_500step_20260528"
    chosen_layer_name: str = "layers.2"
    prune_layer_name: str = "layers.1"
    prune_fraction: float = 0.10
    prune_seed: int = 12345
    recovery_loss_name: str = "activation_mse"
    recovery_steps: int = 10
    calibration_sample_count: int = 2000
    learning_rate: float = 1e-4
    seed: int = 0
    compile_mode: str = ""


def compute_checkpoint_steps(
    steps_per_epoch: int,
    total_epochs: float,
    checkpoint_fraction: float,
) -> tuple[int, int]:
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be at least 1")
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if not 0 < checkpoint_fraction <= 1:
        raise ValueError("checkpoint_fraction must be in (0, 1]")
    total_steps = ceil(total_epochs * steps_per_epoch)
    checkpoint_steps = ceil(total_steps * checkpoint_fraction)
    return total_steps, checkpoint_steps


def _parse_layers_index(layer_name: str) -> int:
    prefix = "layers."
    if not layer_name.startswith(prefix):
        raise ValueError("Only CifarNet layers.N names are supported")
    index = int(layer_name[len(prefix) :])
    if index < 0:
        raise ValueError("Layer index must be non-negative")
    return index


def prefix_parameter_names(model, layer_name: str) -> set[str]:
    layer_index = _parse_layers_index(layer_name)
    prefixes = ["whiten."] + [f"layers.{idx}." for idx in range(layer_index + 1)]
    return {name for name, param in model.named_parameters() if param.requires_grad and any(name.startswith(p) for p in prefixes)}


def layer_weight_parameter_names(model, layer_name: str) -> set[str]:
    _parse_layers_index(layer_name)
    prefix = f"{layer_name}."
    return {name for name, param in model.named_parameters() if name.startswith(prefix) and param.ndim > 1}


def random_prune_layer_weights_(
    model,
    layer_name: str,
    prune_fraction: float,
    seed: int,
):
    import torch

    if not 0 < prune_fraction < 1:
        raise ValueError("prune_fraction must be in (0, 1)")
    selected_names = layer_weight_parameter_names(model, layer_name)
    selected = [(name, param) for name, param in model.named_parameters() if name in selected_names]
    if not selected:
        raise RuntimeError(f"No weight tensors found for {layer_name}")

    total_weights = sum(param.numel() for _name, param in selected)
    pruned_weights = int(total_weights * prune_fraction)
    if pruned_weights < 1:
        raise ValueError("prune_fraction prunes zero weights")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    keep_flat = torch.ones(total_weights, dtype=torch.bool)
    keep_flat[torch.randperm(total_weights, generator=generator)[:pruned_weights]] = False

    masks = {}
    offset = 0
    with torch.no_grad():
        for name, param in selected:
            next_offset = offset + param.numel()
            mask = keep_flat[offset:next_offset].view_as(param).to(device=param.device)
            masks[name] = mask
            param.mul_(mask.to(dtype=param.dtype))
            offset = next_offset

    stats = {
        "layer_name": layer_name,
        "total_weights": int(total_weights),
        "pruned_weights": int(pruned_weights),
        "kept_weights": int(total_weights - pruned_weights),
        "prune_fraction": float(pruned_weights / total_weights),
        "seed": int(seed),
        "parameter_names": sorted(selected_names),
    }
    return masks, stats


def apply_prune_masks_to_grads_(model, masks) -> None:
    for name, param in model.named_parameters():
        if name in masks and param.grad is not None:
            param.grad.mul_(masks[name].to(device=param.grad.device, dtype=param.grad.dtype))


def apply_prune_masks_to_params_(model, masks) -> None:
    import torch

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                mask = masks[name].to(device=param.device)
                param.masked_fill_(~mask, 0)


def forward_to_chosen_layer(model, inputs, layer_name: str, whiten_bias_grad: bool = True):
    import torch.nn.functional as F

    layer_index = _parse_layers_index(layer_name)
    if layer_index >= len(model.layers):
        raise ValueError(f"Layer {layer_name} does not exist")
    bias = model.whiten.bias if whiten_bias_grad else model.whiten.bias.detach()
    outputs = F.conv2d(inputs, model.whiten.weight, bias)
    for idx, layer in enumerate(model.layers):
        outputs = layer(outputs)
        if idx == layer_index:
            return outputs
    raise RuntimeError(f"Layer {layer_name} was not reached")


def forward_layer_activation_sites(model, inputs, site_names: set[str], whiten_bias_grad: bool = True):
    import torch.nn.functional as F

    requested = set(site_names)
    layer_indices = {_parse_layers_index(site.split(".input", 1)[0].split(".output", 1)[0]) for site in requested}
    max_layer_index = max(layer_indices)
    if max_layer_index >= len(model.layers):
        raise ValueError(f"Layer layers.{max_layer_index} does not exist")
    activations = {}
    bias = model.whiten.bias if whiten_bias_grad else model.whiten.bias.detach()
    outputs = F.conv2d(inputs, model.whiten.weight, bias)
    for idx, layer in enumerate(model.layers):
        input_key = f"layers.{idx}.input"
        if input_key in requested:
            activations[input_key] = outputs
        outputs = layer(outputs)
        output_key = f"layers.{idx}.output"
        if output_key in requested:
            activations[output_key] = outputs
        if idx == max_layer_index:
            break
    missing = requested - set(activations)
    if missing:
        raise RuntimeError(f"Activation sites were not reached: {sorted(missing)}")
    return activations


def _activation_features(activations):
    if activations.ndim == 4:
        return activations.float().mean(dim=(2, 3))
    return activations.float().flatten(1)


def proxy_loss_activation_l2(activations, **_kwargs):
    return activations.float().square().mean()


def proxy_loss_activation_l1(activations, **_kwargs):
    return activations.float().abs().mean()


def proxy_loss_activation_linf(activations, **_kwargs):
    return activations.float().flatten(1).abs().amax(dim=1).mean()


def proxy_loss_cosine_sim(activations, *, second_activations=None, **_kwargs):
    if second_activations is None:
        raise ValueError("second_activations are required for cosine_sim")
    import torch.nn.functional as F

    first_features = F.normalize(_activation_features(activations), dim=1)
    second_features = F.normalize(_activation_features(second_activations), dim=1)
    return -F.cosine_similarity(first_features, second_features, dim=1).mean()


def proxy_loss_same_class_contrastive(activations, *, labels=None, temperature: float = 0.2, **_kwargs):
    if labels is None:
        raise ValueError("labels are required for same_class_contrastive")
    import torch
    import torch.nn.functional as F

    features = F.normalize(_activation_features(activations), dim=1)
    logits = features @ features.T / temperature
    eye = torch.eye(len(features), device=features.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, -torch.inf)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~eye
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return features.sum() * 0.0
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor = -(log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1))
    return per_anchor[valid].mean()


PROXY_LOSSES = {
    "activation_l2": proxy_loss_activation_l2,
    "activation_l1": proxy_loss_activation_l1,
    "activation_linf": proxy_loss_activation_linf,
    "cosine_sim": proxy_loss_cosine_sim,
    "same_class_contrastive": proxy_loss_same_class_contrastive,
}

PROXY_LOSSES_WITH_SECOND_VIEW = {"cosine_sim"}


DEFAULT_PRUNE_RECOVERY_LOSS_NAMES = (
    "activation_l1,activation_l2,activation_squared_l2,activation_linf,"
    "fractional_l05,canberra,clark,bray_curtis,soergel,kulczynski,lorentzian,"
    "gower,mape,max_relative,huber,pseudo_huber,tukey_biweight,cauchy_loss,"
    "fair_loss,channelwise_fair_loss,channelwise_fair_layer1_input,"
    "channelwise_fair_layer1_output,channelwise_fair_layer2_input,"
    "channelwise_fair_layer2_output,channelwise_fair_layer3_input,"
    "channelwise_fair_layer3_output,raw_l2_layer1_input,raw_l2_layer1_output,"
    "raw_l2_layer2_input,raw_l2_layer2_output,welsch_loss,cosine_distance,"
    "angular_distance,chordal_spherical,correlation_distance,sign_invariant_angular,"
    "projector_frobenius,best_rescaled_l2,best_shifted_l2,best_affine_l2,"
    "mahalanobis,weighted_l2,gram_quadratic,group_l21,group_l12,"
    "relative_l2,symmetric_relative_l2,relative_l1,log_ratio_positive,aitchison,"
    "alr_distance,hilbert_projective,thompson_metric,norm_ratio,"
    "cosine_preservation,parallel_l2,orthogonal_l2,"
    "projection_residual,bregman_log_cosh,total_variation,kl_divergence,reverse_kl,"
    "jeffreys,js_divergence,hellinger,bhattacharyya,chernoff_alpha_half,"
    "matusita,itakura_saito,chi_square_pearson,chi_square_neyman,"
    "symmetric_chi_square,squared_chord,triangular_discrimination,topsoe,"
    "renyi_alpha_half,alpha_divergence,cressie_read,"
    "tsallis_alpha_half,generalized_i_divergence,beta_divergence_half,"
    "wasserstein_1d,sinkhorn_ot,sliced_wasserstein,max_sliced_wasserstein,"
    "gromov_wasserstein_surrogate,rbf_kernel_point,polynomial_kernel_point,"
    "laplacian_kernel_point,mmd_rbf,energy_distance,cramer_distance,"
    "kolmogorov_smirnov,rsa_distance,linear_cka_distance,"
    "svcca_distance,pwcca_distance,orthogonal_procrustes,linear_regression_distance,"
    "ridge_regression_distance,mutual_nn_overlap,trustworthiness_surrogate,"
    "shape_distance,normalized_bures,covariance_frobenius,covariance_spectral,"
    "covariance_nuclear,covariance_log_euclidean,covariance_affine_invariant,"
    "covariance_bures,covariance_stein,covariance_condition,grassmann_geodesic,"
    "grassmann_chordal,projection_frobenius_subspace,spectral_projection_subspace,"
    "fubini_study,binet_cauchy,sign_hamming,soft_jaccard_topk,dice_topk,"
    "overlap_topk,overlap_coefficient_topk,tversky_topk,simple_matching,"
    "sokal_sneath,rogers_tanimoto,russell_rao,yule_distance,tanimoto,ochiai,"
    "histogram_intersection,ruzicka,motyka,czekanowski,wave_hedges,"
    "inner_product_dissimilarity,fidelity_distance,rank_biased_overlap,"
    "ndcg_topk,rank_correlation,spearman_distance,kendall_distance,distance_correlation,"
    "hsic_rbf,logdet_covariance,fisher_rao_categorical,poincare_distance,"
    "sequence_l2,derivative_sequence_l2,weighted_sequence_l2,soft_dtw,"
    "shape_based_sequence,frechet_surrogate,hausdorff_surrogate,chamfer_distance,"
    "layernorm_l2,cosine_relative,orthogonal_relative,covariance_whitened_cosine,"
    "rare_feature_weighted_l2"
)


def _activation_vectors(activations):
    return activations.float().flatten(1)


def _activation_probabilities(activations):
    return _activation_vectors(activations).softmax(dim=1)


def _positive_activation_vectors(activations):
    import torch.nn.functional as F

    return F.softplus(_activation_vectors(activations)).clamp_min(1e-6)


def _limited_feature_matrix(activations, max_samples: int = 256, max_features: int = 128):
    features = _activation_features(activations)
    return features[: min(features.shape[0], max_samples), : min(features.shape[1], max_features)]


def _center_columns(features):
    return features - features.mean(dim=0, keepdim=True)


def _covariance(features, ridge_scale: float = 1e-3):
    import torch

    centered = _center_columns(features)
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    ridge = cov.diag().mean().abs().clamp_min(1e-6) * ridge_scale
    return cov + ridge * torch.eye(cov.shape[0], device=cov.device)


def _audit_features(activations):
    return _activation_features(activations).float()


def linear_cka_similarity(first_features, second_features):
    first = _audit_features(first_features) if first_features.ndim > 2 else first_features.float()
    second = _audit_features(second_features) if second_features.ndim > 2 else second_features.float()
    first = _center_columns(first)
    second = _center_columns(second)
    cross = (first.T @ second).square().sum()
    first_norm = (first.T @ first).square().sum().sqrt()
    second_norm = (second.T @ second).square().sum().sqrt()
    return cross / (first_norm * second_norm).clamp_min(1e-12)


def fisher_class_separability(features, labels):
    import torch

    values = _audit_features(features) if features.ndim > 2 else features.float()
    labels = labels.to(device=values.device)
    scores = []
    for label in labels.unique(sorted=True):
        positive = values[labels == label]
        negative = values[labels != label]
        if len(positive) < 2 or len(negative) < 2:
            continue
        positive_centered = positive - positive.mean(dim=0, keepdim=True)
        negative_centered = negative - negative.mean(dim=0, keepdim=True)
        numerator = (positive.mean(dim=0) - negative.mean(dim=0)).square().sum()
        denominator = positive_centered.square().sum() / max(1, len(positive) - 1)
        denominator = denominator + negative_centered.square().sum() / max(1, len(negative) - 1)
        scores.append(numerator / denominator.clamp_min(1e-12))
    if not scores:
        return torch.tensor(0.0, device=values.device)
    return torch.stack(scores).mean()


def _ridge_probe_train_test_accuracy(train_x, train_y, test_x, test_y, class_count: int, ridge: float = 1e-3):
    import torch
    import torch.nn.functional as F

    mean = train_x.mean(dim=0, keepdim=True)
    scale = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale
    train_x = torch.cat([train_x, torch.ones(train_x.shape[0], 1, device=train_x.device, dtype=train_x.dtype)], dim=1)
    test_x = torch.cat([test_x, torch.ones(test_x.shape[0], 1, device=test_x.device, dtype=test_x.dtype)], dim=1)

    train_targets = F.one_hot(train_y, num_classes=class_count).to(dtype=train_x.dtype)
    gram = train_x.T @ train_x
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    weights = torch.linalg.solve(gram + ridge * eye, train_x.T @ train_targets)
    predictions = (test_x @ weights).argmax(dim=1)
    return float((predictions == test_y).float().mean().item())


def ridge_probe_accuracy(features, labels, train_fraction: float = 0.75, ridge: float = 1e-3):
    import torch

    values = _audit_features(features) if features.ndim > 2 else features.float()
    labels = labels.to(device=values.device, dtype=torch.long)
    sample_count = values.shape[0]
    train_count = int(sample_count * train_fraction)
    train_count = min(max(train_count, 1), sample_count - 1)
    return _ridge_probe_train_test_accuracy(
        values[:train_count],
        labels[:train_count],
        values[train_count:],
        labels[train_count:],
        class_count=int(labels.max().item()) + 1,
        ridge=ridge,
    )


def _paired_origin_probe_accuracy(original_features, current_features, train_fraction: float = 0.75):
    import torch

    original = _audit_features(original_features) if original_features.ndim > 2 else original_features.float()
    current = _audit_features(current_features) if current_features.ndim > 2 else current_features.float()
    sample_count = min(original.shape[0], current.shape[0])
    train_count = int(sample_count * train_fraction)
    train_count = min(max(train_count, 1), sample_count - 1)
    train_features = torch.cat([original[:train_count], current[:train_count]], dim=0)
    train_labels = torch.cat(
        [
            torch.zeros(train_count, device=original.device, dtype=torch.long),
            torch.ones(train_count, device=original.device, dtype=torch.long),
        ]
    )
    test_features = torch.cat([original[train_count:sample_count], current[train_count:sample_count]], dim=0)
    test_labels = torch.cat(
        [
            torch.zeros(sample_count - train_count, device=original.device, dtype=torch.long),
            torch.ones(sample_count - train_count, device=original.device, dtype=torch.long),
        ]
    )
    return _ridge_probe_train_test_accuracy(train_features, train_labels, test_features, test_labels, class_count=2)


def mirage_activation_audit(original_activations, current_activations, labels) -> dict:
    import torch

    with torch.no_grad():
        original_features = _audit_features(original_activations)
        current_features = _audit_features(current_activations)
        labels = labels.to(device=current_features.device, dtype=torch.long)
        delta = current_features - original_features
        relative_l2 = (
            torch.linalg.vector_norm(delta, dim=1)
            / torch.linalg.vector_norm(original_features, dim=1).clamp_min(1e-8)
        ).mean()
        return {
            "cka_to_original": float(linear_cka_similarity(original_features, current_features).item()),
            "origin_probe_accuracy": float(_paired_origin_probe_accuracy(original_features, current_features)),
            "class_probe_accuracy": float(ridge_probe_accuracy(current_features, labels)),
            "original_class_probe_accuracy": float(ridge_probe_accuracy(original_features, labels)),
            "class_separability": float(fisher_class_separability(current_features, labels).item()),
            "original_class_separability": float(fisher_class_separability(original_features, labels).item()),
            "relative_l2": float(relative_l2.item()),
        }


def mirage_audit_site_names(chosen_layer_name: str) -> set[str]:
    max_index = _parse_layers_index(chosen_layer_name)
    return {
        f"layers.{idx}.{kind}"
        for idx in range(1, max_index + 1)
        for kind in ("input", "output")
    }


def mirage_site_audit(original_sites, current_sites, labels) -> dict:
    return {
        site_name: mirage_activation_audit(original_sites[site_name], current_sites[site_name], labels)
        for site_name in sorted(original_sites)
    }


def _loader_labels(loader):
    for attr in ("labels", "targets"):
        if hasattr(loader, attr):
            return getattr(loader, attr)
    raise AttributeError("CifarLoader does not expose labels or targets")


def _matrix_sqrt_psd(matrix):
    import torch

    vals, vecs = torch.linalg.eigh((matrix + matrix.T) * 0.5)
    return vecs @ torch.diag(vals.clamp_min(1e-8).sqrt()) @ vecs.T


def _matrix_invsqrt_psd(matrix):
    import torch

    vals, vecs = torch.linalg.eigh((matrix + matrix.T) * 0.5)
    return vecs @ torch.diag(vals.clamp_min(1e-8).rsqrt()) @ vecs.T


def _matrix_log_psd(matrix):
    import torch

    vals, vecs = torch.linalg.eigh((matrix + matrix.T) * 0.5)
    return vecs @ torch.diag(vals.clamp_min(1e-8).log()) @ vecs.T


def _deterministic_projections(dim: int, count: int, device, dtype):
    import torch

    rows = torch.arange(1, dim + 1, device=device, dtype=dtype)[:, None]
    cols = torch.arange(1, count + 1, device=device, dtype=dtype)[None, :]
    projections = torch.sin(rows * cols * 0.173) + torch.cos(rows * (cols + 1) * 0.119)
    return projections / projections.norm(dim=0, keepdim=True).clamp_min(1e-8)


def _topk_soft_supports(original_activations, current_activations, fraction: float = 0.10):
    import torch

    original_abs = _activation_vectors(original_activations).abs()
    current_abs = _activation_vectors(current_activations).abs()
    k = max(1, int(original_abs.shape[1] * fraction))
    threshold = original_abs.topk(k, dim=1).values[:, -1:].detach()
    original_support = (original_abs >= threshold).float()
    temperature = original_abs.detach().std().clamp_min(1e-3)
    current_support = torch.sigmoid((current_abs - threshold) / temperature)
    return original_support, current_support


def _soft_binary_counts(original_activations, current_activations):
    original_support, current_support = _topk_soft_supports(original_activations, current_activations)
    n11 = (original_support * current_support).sum(dim=1)
    n10 = (original_support * (1 - current_support)).sum(dim=1)
    n01 = ((1 - original_support) * current_support).sum(dim=1)
    n00 = ((1 - original_support) * (1 - current_support)).sum(dim=1)
    return n00, n01, n10, n11


def _positive_normalized_vectors(activations):
    values = _positive_activation_vectors(activations)
    return values / values.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _spatial_sequence(activations, max_samples: int = 128, max_steps: int = 16, max_channels: int = 64):
    values = activations.float()
    if values.ndim == 4:
        sequence = values.permute(0, 2, 3, 1).flatten(1, 2)
    else:
        sequence = values.flatten(1)[:, :, None]
    return sequence[: min(sequence.shape[0], max_samples), : min(sequence.shape[1], max_steps), : min(sequence.shape[2], max_channels)]


def _log_cosh(values):
    import math
    import torch.nn.functional as F

    return values + F.softplus(-2 * values) - math.log(2.0)


def recovery_loss_activation_mse(original_weights, original_activations, current_activations):
    import torch.nn.functional as F

    return F.mse_loss(current_activations.float(), original_activations.float())


def recovery_loss_activation_l1(original_weights, original_activations, current_activations):
    return (_activation_vectors(current_activations) - _activation_vectors(original_activations)).abs().sum(dim=1).mean()


def recovery_loss_activation_l2(original_weights, original_activations, current_activations):
    import torch

    delta = _activation_vectors(current_activations) - _activation_vectors(original_activations)
    return torch.sqrt(delta.square().sum(dim=1) + 1e-12).mean()


def recovery_loss_activation_squared_l2(original_weights, original_activations, current_activations):
    delta = _activation_vectors(current_activations) - _activation_vectors(original_activations)
    return delta.square().sum(dim=1).mean()


def recovery_loss_activation_linf(original_weights, original_activations, current_activations):
    delta = _activation_vectors(current_activations) - _activation_vectors(original_activations)
    return delta.abs().amax(dim=1).mean()


def recovery_loss_group_l21(original_weights, original_activations, current_activations):
    import torch

    delta = current_activations.float() - original_activations.float()
    if delta.ndim == 4:
        grouped = torch.sqrt(delta.square().sum(dim=(2, 3)) + 1e-12)
    else:
        grouped = torch.sqrt(delta.flatten(1).square() + 1e-12)
    return grouped.sum(dim=1).mean()


def recovery_loss_group_l12(original_weights, original_activations, current_activations):
    import torch

    delta = current_activations.float() - original_activations.float()
    if delta.ndim == 4:
        grouped = delta.abs().sum(dim=(2, 3))
    else:
        grouped = delta.flatten(1).abs()
    return torch.sqrt(grouped.square().sum(dim=1) + 1e-12).mean()


def recovery_loss_fractional_l05(original_weights, original_activations, current_activations):
    delta = _activation_vectors(current_activations) - _activation_vectors(original_activations)
    return (delta.abs() + 1e-8).sqrt().sum(dim=1).mean()


def recovery_loss_canberra(original_weights, original_activations, current_activations):
    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    return ((current_features - original_features).abs() / (current_features.abs() + original_features.abs()).clamp_min(1e-6)).sum(dim=1).mean()


def recovery_loss_clark(original_weights, original_activations, current_activations):
    import torch

    original_features = _positive_activation_vectors(original_activations)
    current_features = _positive_activation_vectors(current_activations)
    ratio = (current_features - original_features) / (current_features + original_features).clamp_min(1e-6)
    return torch.sqrt(ratio.square().sum(dim=1) + 1e-12).mean()


def recovery_loss_bray_curtis(original_weights, original_activations, current_activations):
    original_features = _positive_activation_vectors(original_activations)
    current_features = _positive_activation_vectors(current_activations)
    return ((current_features - original_features).abs().sum(dim=1) / (current_features + original_features).sum(dim=1).clamp_min(1e-6)).mean()


def recovery_loss_soergel(original_weights, original_activations, current_activations):
    import torch

    original_features = _positive_activation_vectors(original_activations)
    current_features = _positive_activation_vectors(current_activations)
    return ((current_features - original_features).abs().sum(dim=1) / torch.maximum(current_features, original_features).sum(dim=1).clamp_min(1e-6)).mean()


def recovery_loss_kulczynski(original_weights, original_activations, current_activations):
    import torch

    original_features = _positive_activation_vectors(original_activations)
    current_features = _positive_activation_vectors(current_activations)
    return ((current_features - original_features).abs().sum(dim=1) / torch.minimum(current_features, original_features).sum(dim=1).clamp_min(1e-6)).mean()


def recovery_loss_lorentzian(original_weights, original_activations, current_activations):
    import torch

    delta = _activation_vectors(current_activations) - _activation_vectors(original_activations)
    return torch.log1p(delta.abs()).sum(dim=1).mean()


def recovery_loss_gower(original_weights, original_activations, current_activations):
    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    ranges = (original_features.amax(dim=0, keepdim=True) - original_features.amin(dim=0, keepdim=True)).clamp_min(1e-4)
    return ((current_features - original_features).abs() / ranges).mean()


def recovery_loss_mape(original_weights, original_activations, current_activations):
    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    return ((current_features - original_features).abs() / original_features.abs().clamp_min(1e-3)).mean()


def recovery_loss_max_relative(original_weights, original_activations, current_activations):
    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    return ((current_features - original_features).abs() / original_features.abs().clamp_min(1e-3)).amax(dim=1).mean()


def recovery_loss_huber(original_weights, original_activations, current_activations):
    import torch

    delta = (_activation_vectors(current_activations) - _activation_vectors(original_activations)).abs()
    return torch.where(delta <= 1, 0.5 * delta.square(), delta - 0.5).sum(dim=1).mean()


def recovery_loss_pseudo_huber(original_weights, original_activations, current_activations):
    import torch

    delta = _activation_vectors(current_activations) - _activation_vectors(original_activations)
    return (torch.sqrt(1 + delta.square()) - 1).sum(dim=1).mean()


def recovery_loss_tukey_biweight(original_weights, original_activations, current_activations):
    import torch

    delta = _activation_vectors(current_activations) - _activation_vectors(original_activations)
    c = delta.detach().abs().median().clamp_min(1e-3) * 4.685
    u = (delta / c).clamp(-1, 1)
    return ((c**2 / 6) * (1 - (1 - u.square()).clamp_min(0).pow(3))).sum(dim=1).mean()


def recovery_loss_cauchy_loss(original_weights, original_activations, current_activations):
    import torch

    delta = _activation_vectors(current_activations) - _activation_vectors(original_activations)
    c = delta.detach().abs().median().clamp_min(1e-3)
    return torch.log1p(delta.square() / c.square()).sum(dim=1).mean()


def recovery_loss_fair_loss(original_weights, original_activations, current_activations):
    import torch

    delta = (_activation_vectors(current_activations) - _activation_vectors(original_activations)).abs()
    c = delta.detach().median().clamp_min(1e-3)
    return (c.square() * (delta / c - torch.log1p(delta / c))).sum(dim=1).mean()


def recovery_loss_channelwise_fair_loss(original_weights, original_activations, current_activations):
    import torch

    return _channelwise_fair_tensor_loss(original_activations, current_activations)


def _channelwise_fair_tensor_loss(original_activations, current_activations):
    import torch

    delta = (current_activations.float() - original_activations.float()).abs()
    if delta.ndim == 4:
        c = delta.detach().permute(1, 0, 2, 3).flatten(1).median(dim=1).values.clamp_min(1e-3)
        c = c.view(1, -1, 1, 1)
        return (c.square() * (delta / c - torch.log1p(delta / c))).flatten(1).sum(dim=1).mean()
    delta = _activation_vectors(current_activations) - _activation_vectors(original_activations)
    delta = delta.abs()
    c = delta.detach().median(dim=0, keepdim=True).values.clamp_min(1e-3)
    return (c.square() * (delta / c - torch.log1p(delta / c))).sum(dim=1).mean()


def _site_activation_pair(original_activations, current_activations, site_name: str):
    if not isinstance(original_activations, dict) or not isinstance(current_activations, dict):
        raise TypeError(f"{site_name} loss requires activation-site dictionaries")
    return original_activations[site_name], current_activations[site_name]


def _site_channelwise_fair_loss(original_activations, current_activations, site_name: str):
    original_site, current_site = _site_activation_pair(original_activations, current_activations, site_name)
    return _channelwise_fair_tensor_loss(original_site, current_site)


def recovery_loss_channelwise_fair_layer1_input(original_weights, original_activations, current_activations):
    return _site_channelwise_fair_loss(original_activations, current_activations, "layers.1.input")


def recovery_loss_channelwise_fair_layer1_output(original_weights, original_activations, current_activations):
    return _site_channelwise_fair_loss(original_activations, current_activations, "layers.1.output")


def recovery_loss_channelwise_fair_layer2_input(original_weights, original_activations, current_activations):
    return _site_channelwise_fair_loss(original_activations, current_activations, "layers.2.input")


def recovery_loss_channelwise_fair_layer2_output(original_weights, original_activations, current_activations):
    return _site_channelwise_fair_loss(original_activations, current_activations, "layers.2.output")


def recovery_loss_channelwise_fair_layer3_input(original_weights, original_activations, current_activations):
    return _site_channelwise_fair_loss(original_activations, current_activations, "layers.3.input")


def recovery_loss_channelwise_fair_layer3_output(original_weights, original_activations, current_activations):
    return _site_channelwise_fair_loss(original_activations, current_activations, "layers.3.output")


def _site_raw_l2_loss(original_activations, current_activations, site_name: str):
    import torch

    original_site, current_site = _site_activation_pair(original_activations, current_activations, site_name)
    delta = _activation_vectors(current_site) - _activation_vectors(original_site)
    return torch.sqrt(delta.square().sum(dim=1) + 1e-12).mean()


def recovery_loss_raw_l2_layer1_input(original_weights, original_activations, current_activations):
    return _site_raw_l2_loss(original_activations, current_activations, "layers.1.input")


def recovery_loss_raw_l2_layer1_output(original_weights, original_activations, current_activations):
    return _site_raw_l2_loss(original_activations, current_activations, "layers.1.output")


def recovery_loss_raw_l2_layer2_input(original_weights, original_activations, current_activations):
    return _site_raw_l2_loss(original_activations, current_activations, "layers.2.input")


def recovery_loss_raw_l2_layer2_output(original_weights, original_activations, current_activations):
    return _site_raw_l2_loss(original_activations, current_activations, "layers.2.output")


def recovery_loss_welsch_loss(original_weights, original_activations, current_activations):
    import torch

    delta = _activation_vectors(current_activations) - _activation_vectors(original_activations)
    c = delta.detach().abs().median().clamp_min(1e-3)
    return (1 - torch.exp(-delta.square() / (2 * c.square()))).sum(dim=1).mean()


def recovery_loss_activation_cosine(original_weights, original_activations, current_activations):
    import torch.nn.functional as F

    original_features = F.normalize(_activation_vectors(original_activations), dim=1)
    current_features = F.normalize(_activation_vectors(current_activations), dim=1)
    return 1 - F.cosine_similarity(current_features, original_features, dim=1).mean()


def recovery_loss_angular_distance(original_weights, original_activations, current_activations):
    import torch
    import torch.nn.functional as F

    original_features = F.normalize(_activation_vectors(original_activations), dim=1)
    current_features = F.normalize(_activation_vectors(current_activations), dim=1)
    cosine = F.cosine_similarity(current_features, original_features, dim=1).clamp(-1 + 1e-5, 1 - 1e-5)
    return torch.acos(cosine).mean()


def recovery_loss_chordal_spherical(original_weights, original_activations, current_activations):
    import torch
    import torch.nn.functional as F

    original_features = F.normalize(_activation_vectors(original_activations), dim=1)
    current_features = F.normalize(_activation_vectors(current_activations), dim=1)
    return torch.linalg.vector_norm(current_features - original_features, dim=1).mean()


def recovery_loss_correlation_distance(original_weights, original_activations, current_activations):
    import torch.nn.functional as F

    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    original_centered = original_features - original_features.mean(dim=1, keepdim=True)
    current_centered = current_features - current_features.mean(dim=1, keepdim=True)
    original_centered = F.normalize(original_centered, dim=1)
    current_centered = F.normalize(current_centered, dim=1)
    return 1 - F.cosine_similarity(current_centered, original_centered, dim=1).mean()


def recovery_loss_sign_invariant_angular(original_weights, original_activations, current_activations):
    import torch
    import torch.nn.functional as F

    original_features = F.normalize(_activation_vectors(original_activations), dim=1)
    current_features = F.normalize(_activation_vectors(current_activations), dim=1)
    cosine = F.cosine_similarity(current_features, original_features, dim=1).abs().clamp(max=1 - 1e-5)
    return torch.acos(cosine).mean()


def recovery_loss_projector_frobenius(original_weights, original_activations, current_activations):
    import torch
    import torch.nn.functional as F

    original_features = F.normalize(_activation_vectors(original_activations), dim=1)
    current_features = F.normalize(_activation_vectors(current_activations), dim=1)
    cosine = F.cosine_similarity(current_features, original_features, dim=1).clamp(-1, 1)
    return torch.sqrt((2 - 2 * cosine.square()).clamp_min(0) + 1e-12).mean()


def recovery_loss_best_rescaled_l2(original_weights, original_activations, current_activations):
    import torch

    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    scale = (original_features * current_features).sum(dim=1, keepdim=True) / current_features.square().sum(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.linalg.vector_norm(original_features - scale * current_features, dim=1).mean()


def recovery_loss_best_shifted_l2(original_weights, original_activations, current_activations):
    import torch

    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    shift = (original_features - current_features).mean(dim=1, keepdim=True)
    return torch.linalg.vector_norm(original_features - (current_features + shift), dim=1).mean()


def recovery_loss_best_affine_l2(original_weights, original_activations, current_activations):
    import torch

    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    original_centered = original_features - original_features.mean(dim=1, keepdim=True)
    current_centered = current_features - current_features.mean(dim=1, keepdim=True)
    scale = (original_centered * current_centered).sum(dim=1, keepdim=True) / current_centered.square().sum(dim=1, keepdim=True).clamp_min(1e-8)
    shift = original_features.mean(dim=1, keepdim=True) - scale * current_features.mean(dim=1, keepdim=True)
    return torch.linalg.vector_norm(original_features - (scale * current_features + shift), dim=1).mean()


def recovery_loss_mahalanobis(original_weights, original_activations, current_activations):
    import torch

    original_features = _activation_features(original_activations)
    current_features = _activation_features(current_activations)
    delta = current_features - original_features
    centered = original_features - original_features.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    ridge = covariance.diag().mean().clamp_min(1e-6) * 1e-3
    precision = torch.linalg.pinv(covariance + ridge * torch.eye(covariance.shape[0], device=covariance.device))
    quadratic = (delta @ precision * delta).sum(dim=1).clamp_min(0)
    return torch.sqrt(quadratic + 1e-12).mean()


def recovery_loss_weighted_l2(original_weights, original_activations, current_activations):
    import torch

    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    weights = 1 / (original_features.var(dim=0, unbiased=False, keepdim=True) + 1e-4)
    quadratic = ((current_features - original_features).square() * weights).sum(dim=1)
    return torch.sqrt(quadratic + 1e-12).mean()


def recovery_loss_gram_quadratic(original_weights, original_activations, current_activations):
    original_features = _activation_features(original_activations)
    current_features = _activation_features(current_activations)
    delta = current_features - original_features
    gram = original_features.T @ original_features / max(1, original_features.shape[0])
    return (delta @ gram * delta).sum(dim=1).mean()


def recovery_loss_relative_l2(original_weights, original_activations, current_activations):
    import torch

    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    delta_norm = torch.linalg.vector_norm(current_features - original_features, dim=1)
    original_norm = torch.linalg.vector_norm(original_features, dim=1).clamp_min(1e-8)
    return (delta_norm / original_norm).mean()


def recovery_loss_symmetric_relative_l2(original_weights, original_activations, current_activations):
    import torch

    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    delta_norm = torch.linalg.vector_norm(current_features - original_features, dim=1)
    denom = (torch.linalg.vector_norm(original_features, dim=1) + torch.linalg.vector_norm(current_features, dim=1)).clamp_min(1e-8)
    return (delta_norm / denom).mean()


def recovery_loss_relative_l1(original_weights, original_activations, current_activations):
    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    delta_norm = (current_features - original_features).abs().sum(dim=1)
    original_norm = original_features.abs().sum(dim=1).clamp_min(1e-8)
    return (delta_norm / original_norm).mean()


def recovery_loss_log_ratio_positive(original_weights, original_activations, current_activations):
    import torch

    original_features = _positive_activation_vectors(original_activations)
    current_features = _positive_activation_vectors(current_activations)
    return torch.linalg.vector_norm(current_features.log() - original_features.log(), dim=1).mean()


def recovery_loss_aitchison(original_weights, original_activations, current_activations):
    import torch

    original_probs = _activation_probabilities(original_activations).clamp_min(1e-12)
    current_probs = _activation_probabilities(current_activations).clamp_min(1e-12)
    original_clr = original_probs.log() - original_probs.log().mean(dim=1, keepdim=True)
    current_clr = current_probs.log() - current_probs.log().mean(dim=1, keepdim=True)
    return torch.linalg.vector_norm(current_clr - original_clr, dim=1).mean()


def recovery_loss_alr_distance(original_weights, original_activations, current_activations):
    import torch

    original_probs = _activation_probabilities(original_activations)[:, :128].clamp_min(1e-12)
    current_probs = _activation_probabilities(current_activations)[:, :128].clamp_min(1e-12)
    original_alr = (original_probs[:, :-1] / original_probs[:, -1:]).log()
    current_alr = (current_probs[:, :-1] / current_probs[:, -1:]).log()
    return torch.linalg.vector_norm(current_alr - original_alr, dim=1).mean()


def recovery_loss_hilbert_projective(original_weights, original_activations, current_activations):
    original_positive = _positive_activation_vectors(original_activations)
    current_positive = _positive_activation_vectors(current_activations)
    log_ratio = (current_positive / original_positive).clamp_min(1e-8).log()
    return (log_ratio.amax(dim=1) - log_ratio.amin(dim=1)).mean()


def recovery_loss_thompson_metric(original_weights, original_activations, current_activations):
    import torch

    original_positive = _positive_activation_vectors(original_activations)
    current_positive = _positive_activation_vectors(current_activations)
    forward = (current_positive / original_positive).clamp_min(1e-8).log().amax(dim=1)
    reverse = (original_positive / current_positive).clamp_min(1e-8).log().amax(dim=1)
    return torch.maximum(forward, reverse).mean()


def recovery_loss_norm_ratio(original_weights, original_activations, current_activations):
    import torch

    original_norm = torch.linalg.vector_norm(_activation_vectors(original_activations), dim=1).clamp_min(1e-8)
    current_norm = torch.linalg.vector_norm(_activation_vectors(current_activations), dim=1).clamp_min(1e-8)
    return torch.log(current_norm / original_norm).abs().mean()


def recovery_loss_parallel_l2(original_weights, original_activations, current_activations):
    import torch

    original_features = _activation_vectors(original_activations)
    delta = _activation_vectors(current_activations) - original_features
    scale = (original_features * delta).sum(dim=1, keepdim=True) / original_features.square().sum(dim=1, keepdim=True).clamp_min(1e-8)
    parallel = scale * original_features
    return torch.linalg.vector_norm(parallel, dim=1).mean()


def recovery_loss_orthogonal_l2(original_weights, original_activations, current_activations):
    import torch

    original_features = _activation_vectors(original_activations)
    delta = _activation_vectors(current_activations) - original_features
    scale = (original_features * delta).sum(dim=1, keepdim=True) / original_features.square().sum(dim=1, keepdim=True).clamp_min(1e-8)
    orthogonal = delta - scale * original_features
    return torch.linalg.vector_norm(orthogonal, dim=1).mean()


def recovery_loss_projection_residual(original_weights, original_activations, current_activations):
    import torch

    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    scale = (original_features * current_features).sum(dim=1, keepdim=True) / original_features.square().sum(dim=1, keepdim=True).clamp_min(1e-8)
    residual = current_features - scale * original_features
    return torch.linalg.vector_norm(residual, dim=1).mean()


def recovery_loss_bregman_log_cosh(original_weights, original_activations, current_activations):
    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    return (_log_cosh(current_features) - _log_cosh(original_features) - original_features.tanh() * (current_features - original_features)).sum(dim=1).mean()


def recovery_loss_total_variation(original_weights, original_activations, current_activations):
    original_probs = _activation_probabilities(original_activations)
    current_probs = _activation_probabilities(current_activations)
    return 0.5 * (current_probs - original_probs).abs().sum(dim=1).mean()


def recovery_loss_kl_divergence(original_weights, original_activations, current_activations):
    original_probs = _activation_probabilities(original_activations)
    current_log_probs = _activation_vectors(current_activations).log_softmax(dim=1)
    original_log_probs = original_probs.clamp_min(1e-12).log()
    return (original_probs * (original_log_probs - current_log_probs)).sum(dim=1).mean()


def recovery_loss_reverse_kl(original_weights, original_activations, current_activations):
    current_probs = _activation_probabilities(current_activations)
    original_log_probs = _activation_vectors(original_activations).log_softmax(dim=1)
    current_log_probs = current_probs.clamp_min(1e-12).log()
    return (current_probs * (current_log_probs - original_log_probs)).sum(dim=1).mean()


def recovery_loss_jeffreys(original_weights, original_activations, current_activations):
    return recovery_loss_kl_divergence(original_weights, original_activations, current_activations) + recovery_loss_reverse_kl(original_weights, original_activations, current_activations)


def recovery_loss_js_divergence(original_weights, original_activations, current_activations):
    original_probs = _activation_probabilities(original_activations)
    current_probs = _activation_probabilities(current_activations)
    midpoint = 0.5 * (original_probs + current_probs)
    original_kl = (original_probs * (original_probs.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(dim=1)
    current_kl = (current_probs * (current_probs.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(dim=1)
    return (0.5 * (original_kl + current_kl)).mean()


def recovery_loss_hellinger(original_weights, original_activations, current_activations):
    import torch

    original_probs = _activation_probabilities(original_activations)
    current_probs = _activation_probabilities(current_activations)
    return torch.sqrt(0.5 * (current_probs.sqrt() - original_probs.sqrt()).square().sum(dim=1) + 1e-12).mean()


def recovery_loss_bhattacharyya(original_weights, original_activations, current_activations):
    import torch

    original_probs = _activation_probabilities(original_activations)
    current_probs = _activation_probabilities(current_activations)
    affinity = (original_probs.sqrt() * current_probs.sqrt()).sum(dim=1).clamp_min(1e-12)
    return (-torch.log(affinity)).mean()


def recovery_loss_chernoff_alpha_half(original_weights, original_activations, current_activations):
    return recovery_loss_bhattacharyya(original_weights, original_activations, current_activations)


def recovery_loss_matusita(original_weights, original_activations, current_activations):
    import torch

    original_probs = _activation_probabilities(original_activations)
    current_probs = _activation_probabilities(current_activations)
    return torch.linalg.vector_norm(current_probs.sqrt() - original_probs.sqrt(), dim=1).mean()


def recovery_loss_itakura_saito(original_weights, original_activations, current_activations):
    import torch.nn.functional as F

    original_positive = F.softplus(_activation_vectors(original_activations)).clamp_min(1e-6)
    current_positive = F.softplus(_activation_vectors(current_activations)).clamp_min(1e-6)
    ratio = current_positive / original_positive
    return (ratio - ratio.log() - 1).mean()


def recovery_loss_chi_square_pearson(original_weights, original_activations, current_activations):
    original_probs = _activation_probabilities(original_activations)
    current_probs = _activation_probabilities(current_activations)
    return ((current_probs - original_probs).square() / original_probs.clamp_min(1e-8)).sum(dim=1).mean()


def recovery_loss_chi_square_neyman(original_weights, original_activations, current_activations):
    original_probs = _activation_probabilities(original_activations)
    current_probs = _activation_probabilities(current_activations)
    return ((current_probs - original_probs).square() / current_probs.clamp_min(1e-8)).sum(dim=1).mean()


def recovery_loss_symmetric_chi_square(original_weights, original_activations, current_activations):
    return recovery_loss_triangular_discrimination(original_weights, original_activations, current_activations)


def recovery_loss_squared_chord(original_weights, original_activations, current_activations):
    original_probs = _activation_probabilities(original_activations)
    current_probs = _activation_probabilities(current_activations)
    return (current_probs.sqrt() - original_probs.sqrt()).square().sum(dim=1).mean()


def recovery_loss_triangular_discrimination(original_weights, original_activations, current_activations):
    original_probs = _activation_probabilities(original_activations)
    current_probs = _activation_probabilities(current_activations)
    return ((current_probs - original_probs).square() / (current_probs + original_probs).clamp_min(1e-8)).sum(dim=1).mean()


def recovery_loss_topsoe(original_weights, original_activations, current_activations):
    return 2 * recovery_loss_js_divergence(original_weights, original_activations, current_activations)


def recovery_loss_renyi_alpha_half(original_weights, original_activations, current_activations):
    import torch

    original_probs = _activation_probabilities(original_activations)
    current_probs = _activation_probabilities(current_activations)
    affinity = (original_probs.sqrt() * current_probs.sqrt()).sum(dim=1).clamp_min(1e-12)
    return (-2 * torch.log(affinity)).mean()


def recovery_loss_alpha_divergence(original_weights, original_activations, current_activations):
    original_probs = _activation_probabilities(original_activations)
    current_probs = _activation_probabilities(current_activations)
    affinity = (original_probs.sqrt() * current_probs.sqrt()).sum(dim=1)
    return (4 * (1 - affinity)).mean()


def recovery_loss_cressie_read(original_weights, original_activations, current_activations):
    original_probs = _activation_probabilities(original_activations).clamp_min(1e-12)
    current_probs = _activation_probabilities(current_activations).clamp_min(1e-12)
    lam = 2 / 3
    ratio = (original_probs / current_probs).clamp_min(1e-8)
    return (2 / (lam * (lam + 1)) * original_probs * (ratio.pow(lam) - 1)).sum(dim=1).mean()


def recovery_loss_tsallis_alpha_half(original_weights, original_activations, current_activations):
    original_probs = _activation_probabilities(original_activations)
    current_probs = _activation_probabilities(current_activations)
    affinity = (original_probs.sqrt() * current_probs.sqrt()).sum(dim=1)
    return (2 * (1 - affinity)).mean()


def recovery_loss_generalized_i_divergence(original_weights, original_activations, current_activations):
    original_positive = _positive_activation_vectors(original_activations)
    current_positive = _positive_activation_vectors(current_activations)
    return (original_positive * (original_positive.log() - current_positive.log()) - original_positive + current_positive).mean()


def recovery_loss_beta_divergence_half(original_weights, original_activations, current_activations):
    original_positive = _positive_activation_vectors(original_activations)
    current_positive = _positive_activation_vectors(current_activations)
    beta = 0.5
    return (
        original_positive.pow(beta) / (beta * (beta - 1))
        + current_positive.pow(beta) / beta
        - original_positive * current_positive.pow(beta - 1) / (beta - 1)
    ).mean()


def recovery_loss_wasserstein_1d(original_weights, original_activations, current_activations):
    original_cdf = _activation_probabilities(original_activations).cumsum(dim=1)
    current_cdf = _activation_probabilities(current_activations).cumsum(dim=1)
    return (current_cdf - original_cdf).abs().mean()


def recovery_loss_sinkhorn_ot(original_weights, original_activations, current_activations):
    import torch

    original_probs = _limited_feature_matrix(original_activations, max_samples=512, max_features=64).softmax(dim=1).mean(dim=0)
    current_probs = _limited_feature_matrix(current_activations, max_samples=512, max_features=64).softmax(dim=1).mean(dim=0)
    n = original_probs.shape[0]
    grid = torch.linspace(0, 1, n, device=original_probs.device)
    cost = (grid[:, None] - grid[None, :]).square()
    kernel = torch.exp(-cost / 0.05)
    u = torch.ones_like(original_probs)
    v = torch.ones_like(current_probs)
    for _ in range(20):
        u = original_probs / (kernel @ v).clamp_min(1e-12)
        v = current_probs / (kernel.T @ u).clamp_min(1e-12)
    plan = u[:, None] * kernel * v[None, :]
    return (plan * cost).sum()


def recovery_loss_sliced_wasserstein(original_weights, original_activations, current_activations):
    original_features = _limited_feature_matrix(original_activations, max_samples=256, max_features=64)
    current_features = _limited_feature_matrix(current_activations, max_samples=256, max_features=64)
    projections = _deterministic_projections(
        original_features.shape[1],
        16,
        original_features.device,
        original_features.dtype,
    )
    original_sorted = (original_features @ projections).sort(dim=0).values
    current_sorted = (current_features @ projections).sort(dim=0).values
    return (current_sorted - original_sorted).abs().mean()


def recovery_loss_max_sliced_wasserstein(original_weights, original_activations, current_activations):
    original_features = _limited_feature_matrix(original_activations, max_samples=256, max_features=64)
    current_features = _limited_feature_matrix(current_activations, max_samples=256, max_features=64)
    projections = _deterministic_projections(
        original_features.shape[1],
        16,
        original_features.device,
        original_features.dtype,
    )
    original_sorted = (original_features @ projections).sort(dim=0).values
    current_sorted = (current_features @ projections).sort(dim=0).values
    return (current_sorted - original_sorted).abs().mean(dim=0).amax()


def recovery_loss_gromov_wasserstein_surrogate(original_weights, original_activations, current_activations):
    import torch

    original_features = _limited_feature_matrix(original_activations, max_samples=128, max_features=64)
    current_features = _limited_feature_matrix(current_activations, max_samples=128, max_features=64)
    original_dist = torch.cdist(original_features, original_features)
    current_dist = torch.cdist(current_features, current_features)
    original_dist = original_dist / original_dist.mean().clamp_min(1e-8)
    current_dist = current_dist / current_dist.mean().clamp_min(1e-8)
    return (current_dist - original_dist).square().mean()


def recovery_loss_rbf_kernel_point(original_weights, original_activations, current_activations):
    import torch

    original_features = _activation_features(original_activations)
    current_features = _activation_features(current_activations)
    diff2 = (current_features - original_features).square().sum(dim=1)
    bandwidth = original_features.detach().var(dim=0).sum().clamp_min(1e-4)
    return (2 - 2 * torch.exp(-diff2 / bandwidth)).mean()


def recovery_loss_polynomial_kernel_point(original_weights, original_activations, current_activations):
    original_features = _activation_features(original_activations)
    current_features = _activation_features(current_activations)
    degree = 2
    kxx = (original_features.square().sum(dim=1) / original_features.shape[1] + 1).pow(degree)
    kyy = (current_features.square().sum(dim=1) / current_features.shape[1] + 1).pow(degree)
    kxy = ((original_features * current_features).sum(dim=1) / original_features.shape[1] + 1).pow(degree)
    return (kxx + kyy - 2 * kxy).mean()


def recovery_loss_laplacian_kernel_point(original_weights, original_activations, current_activations):
    import torch

    original_features = _activation_features(original_activations)
    current_features = _activation_features(current_activations)
    distance = torch.linalg.vector_norm(current_features - original_features, dim=1)
    bandwidth = torch.linalg.vector_norm(original_features.detach() - original_features.detach().mean(dim=0), dim=1).median().clamp_min(1e-4)
    return (2 - 2 * torch.exp(-distance / bandwidth)).mean()


def recovery_loss_mmd_rbf(original_weights, original_activations, current_activations):
    import torch

    original_features = _limited_feature_matrix(original_activations, max_samples=256, max_features=64)
    current_features = _limited_feature_matrix(current_activations, max_samples=256, max_features=64)
    with torch.no_grad():
        bandwidth = torch.pdist(original_features[: min(128, len(original_features))]).median().square().clamp_min(1e-4)
    xx = torch.cdist(original_features, original_features).square()
    yy = torch.cdist(current_features, current_features).square()
    xy = torch.cdist(original_features, current_features).square()
    return torch.exp(-xx / bandwidth).mean() + torch.exp(-yy / bandwidth).mean() - 2 * torch.exp(-xy / bandwidth).mean()


def recovery_loss_energy_distance(original_weights, original_activations, current_activations):
    import torch

    original_features = _limited_feature_matrix(original_activations, max_samples=256, max_features=64)
    current_features = _limited_feature_matrix(current_activations, max_samples=256, max_features=64)
    return 2 * torch.cdist(original_features, current_features).mean() - torch.cdist(original_features, original_features).mean() - torch.cdist(current_features, current_features).mean()


def recovery_loss_cramer_distance(original_weights, original_activations, current_activations):
    original_cdf = _activation_probabilities(original_activations).cumsum(dim=1)
    current_cdf = _activation_probabilities(current_activations).cumsum(dim=1)
    return (current_cdf - original_cdf).square().mean()


def recovery_loss_kolmogorov_smirnov(original_weights, original_activations, current_activations):
    original_cdf = _activation_probabilities(original_activations).cumsum(dim=1)
    current_cdf = _activation_probabilities(current_activations).cumsum(dim=1)
    return (current_cdf - original_cdf).abs().amax(dim=1).mean()


def recovery_loss_rsa_distance(original_weights, original_activations, current_activations):
    import torch
    import torch.nn.functional as F

    original_features = _limited_feature_matrix(original_activations, max_samples=128, max_features=64)
    current_features = _limited_feature_matrix(current_activations, max_samples=128, max_features=64)
    original_rdm = torch.cdist(original_features, original_features).flatten()
    current_rdm = torch.cdist(current_features, current_features).flatten()
    original_rdm = original_rdm - original_rdm.mean()
    current_rdm = current_rdm - current_rdm.mean()
    return 1 - F.cosine_similarity(current_rdm[None], original_rdm[None], dim=1).mean()


def recovery_loss_linear_cka_distance(original_weights, original_activations, current_activations):
    original_features = _center_columns(_limited_feature_matrix(original_activations, max_samples=512, max_features=128))
    current_features = _center_columns(_limited_feature_matrix(current_activations, max_samples=512, max_features=128))
    cross = (original_features.T @ current_features).square().sum()
    original_norm = (original_features.T @ original_features).square().sum().sqrt()
    current_norm = (current_features.T @ current_features).square().sum().sqrt()
    return 1 - cross / (original_norm * current_norm).clamp_min(1e-12)


def _cca_correlations(original_activations, current_activations):
    import torch

    original_features = _center_columns(_limited_feature_matrix(original_activations, max_samples=512, max_features=64))
    current_features = _center_columns(_limited_feature_matrix(current_activations, max_samples=512, max_features=64))
    n = max(1, original_features.shape[0] - 1)
    original_cov = original_features.T @ original_features / n
    current_cov = current_features.T @ current_features / n
    cross_cov = original_features.T @ current_features / n
    eye = torch.eye(original_cov.shape[0], device=original_cov.device)
    ridge = 1e-3
    original_vals, original_vecs = torch.linalg.eigh(original_cov + ridge * eye)
    current_vals, current_vecs = torch.linalg.eigh(current_cov + ridge * eye)
    original_inv_sqrt = original_vecs @ torch.diag(original_vals.clamp_min(1e-8).rsqrt()) @ original_vecs.T
    current_inv_sqrt = current_vecs @ torch.diag(current_vals.clamp_min(1e-8).rsqrt()) @ current_vecs.T
    return torch.linalg.svdvals(original_inv_sqrt @ cross_cov @ current_inv_sqrt).clamp(0, 1)


def recovery_loss_svcca_distance(original_weights, original_activations, current_activations):
    correlations = _cca_correlations(original_activations, current_activations)
    return 1 - correlations.mean()


def recovery_loss_pwcca_distance(original_weights, original_activations, current_activations):
    correlations = _cca_correlations(original_activations, current_activations)
    weights = correlations.detach() / correlations.detach().sum().clamp_min(1e-8)
    return 1 - (weights * correlations).sum()


def recovery_loss_orthogonal_procrustes(original_weights, original_activations, current_activations):
    original_features = _center_columns(_limited_feature_matrix(original_activations, max_samples=512, max_features=128))
    current_features = _center_columns(_limited_feature_matrix(current_activations, max_samples=512, max_features=128))
    nuclear = __import__("torch").linalg.svdvals(original_features.T @ current_features).sum()
    denom = original_features.square().sum().clamp_min(1e-8)
    return (original_features.square().sum() + current_features.square().sum() - 2 * nuclear).clamp_min(0) / denom


def recovery_loss_linear_regression_distance(original_weights, original_activations, current_activations):
    import torch

    original_features = _limited_feature_matrix(original_activations, max_samples=512, max_features=128)
    current_features = _limited_feature_matrix(current_activations, max_samples=512, max_features=128)
    coefficients = torch.linalg.pinv(original_features.detach()) @ current_features
    residual = current_features - original_features @ coefficients
    return residual.square().mean()


def recovery_loss_ridge_regression_distance(original_weights, original_activations, current_activations):
    import torch

    original_features = _limited_feature_matrix(original_activations, max_samples=512, max_features=128)
    current_features = _limited_feature_matrix(current_activations, max_samples=512, max_features=128)
    gram = original_features.detach().T @ original_features.detach()
    ridge = gram.diag().mean().abs().clamp_min(1e-6) * 1e-2
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coefficients = torch.linalg.solve(gram + ridge * eye, original_features.detach().T @ current_features)
    residual = current_features - original_features @ coefficients
    return residual.square().mean()


def recovery_loss_mutual_nn_overlap(original_weights, original_activations, current_activations):
    import torch

    original_features = _limited_feature_matrix(original_activations, max_samples=128, max_features=64)
    current_features = _limited_feature_matrix(current_activations, max_samples=128, max_features=64)
    k = min(8, max(1, original_features.shape[0] - 1))
    original_neighbors = torch.cdist(original_features.detach(), original_features.detach()).topk(k + 1, largest=False).indices[:, 1:]
    current_dist = torch.cdist(current_features, current_features)
    current_soft = torch.softmax(-current_dist / current_dist.detach().median().clamp_min(1e-4), dim=1)
    target = torch.zeros_like(current_soft).scatter_(1, original_neighbors, 1.0 / k)
    return -(target * current_soft.clamp_min(1e-12).log()).sum(dim=1).mean()


def recovery_loss_trustworthiness_surrogate(original_weights, original_activations, current_activations):
    return recovery_loss_mutual_nn_overlap(original_weights, original_activations, current_activations)


def recovery_loss_shape_distance(original_weights, original_activations, current_activations):
    import torch

    original_features = _center_columns(_limited_feature_matrix(original_activations, max_samples=512, max_features=128))
    current_features = _center_columns(_limited_feature_matrix(current_activations, max_samples=512, max_features=128))
    original_features = original_features / original_features.norm().clamp_min(1e-8)
    current_features = current_features / current_features.norm().clamp_min(1e-8)
    nuclear = torch.linalg.svdvals(original_features.T @ current_features).sum()
    return (1 - nuclear).clamp_min(0)


def recovery_loss_normalized_bures(original_weights, original_activations, current_activations):
    return recovery_loss_covariance_bures(original_weights, original_activations, current_activations)


def recovery_loss_covariance_frobenius(original_weights, original_activations, current_activations):
    import torch

    original_cov = _covariance(_limited_feature_matrix(original_activations, max_samples=512, max_features=64))
    current_cov = _covariance(_limited_feature_matrix(current_activations, max_samples=512, max_features=64))
    return torch.linalg.matrix_norm(current_cov - original_cov) / torch.linalg.matrix_norm(original_cov).clamp_min(1e-8)


def recovery_loss_covariance_spectral(original_weights, original_activations, current_activations):
    import torch

    original_cov = _covariance(_limited_feature_matrix(original_activations, max_samples=512, max_features=64))
    current_cov = _covariance(_limited_feature_matrix(current_activations, max_samples=512, max_features=64))
    return torch.linalg.matrix_norm(current_cov - original_cov, ord=2) / torch.linalg.matrix_norm(original_cov, ord=2).clamp_min(1e-8)


def recovery_loss_covariance_nuclear(original_weights, original_activations, current_activations):
    import torch

    original_cov = _covariance(_limited_feature_matrix(original_activations, max_samples=512, max_features=64))
    current_cov = _covariance(_limited_feature_matrix(current_activations, max_samples=512, max_features=64))
    return torch.linalg.matrix_norm(current_cov - original_cov, ord="nuc") / torch.linalg.matrix_norm(original_cov, ord="nuc").clamp_min(1e-8)


def recovery_loss_covariance_log_euclidean(original_weights, original_activations, current_activations):
    import torch

    original_cov = _covariance(_limited_feature_matrix(original_activations, max_samples=512, max_features=64))
    current_cov = _covariance(_limited_feature_matrix(current_activations, max_samples=512, max_features=64))
    return torch.linalg.matrix_norm(_matrix_log_psd(current_cov) - _matrix_log_psd(original_cov))


def recovery_loss_covariance_affine_invariant(original_weights, original_activations, current_activations):
    import torch

    original_cov = _covariance(_limited_feature_matrix(original_activations, max_samples=512, max_features=64))
    current_cov = _covariance(_limited_feature_matrix(current_activations, max_samples=512, max_features=64))
    middle = _matrix_invsqrt_psd(original_cov) @ current_cov @ _matrix_invsqrt_psd(original_cov)
    return torch.linalg.matrix_norm(_matrix_log_psd(middle))


def recovery_loss_covariance_bures(original_weights, original_activations, current_activations):
    import torch

    original_cov = _covariance(_limited_feature_matrix(original_activations, max_samples=512, max_features=64))
    current_cov = _covariance(_limited_feature_matrix(current_activations, max_samples=512, max_features=64))
    middle_sqrt = _matrix_sqrt_psd(_matrix_sqrt_psd(original_cov) @ current_cov @ _matrix_sqrt_psd(original_cov))
    return (torch.trace(original_cov) + torch.trace(current_cov) - 2 * torch.trace(middle_sqrt)).clamp_min(0)


def recovery_loss_covariance_stein(original_weights, original_activations, current_activations):
    import torch

    original_cov = _covariance(_limited_feature_matrix(original_activations, max_samples=512, max_features=64))
    current_cov = _covariance(_limited_feature_matrix(current_activations, max_samples=512, max_features=64))
    midpoint = 0.5 * (original_cov + current_cov)
    return (torch.linalg.slogdet(midpoint).logabsdet - 0.5 * (torch.linalg.slogdet(original_cov).logabsdet + torch.linalg.slogdet(current_cov).logabsdet)).clamp_min(0)


def recovery_loss_covariance_condition(original_weights, original_activations, current_activations):
    import torch

    original_cov = _covariance(_limited_feature_matrix(original_activations, max_samples=512, max_features=64))
    current_cov = _covariance(_limited_feature_matrix(current_activations, max_samples=512, max_features=64))
    generalized = _matrix_invsqrt_psd(original_cov) @ current_cov @ _matrix_invsqrt_psd(original_cov)
    vals = torch.linalg.eigvalsh((generalized + generalized.T) * 0.5).clamp_min(1e-8)
    return (vals.amax() / vals.amin()).log()


def _subspace_cosines(original_activations, current_activations, rank: int = 16):
    import torch

    original_features = _center_columns(_limited_feature_matrix(original_activations, max_samples=512, max_features=64))
    current_features = _center_columns(_limited_feature_matrix(current_activations, max_samples=512, max_features=64))
    _uo, _so, vo = torch.linalg.svd(original_features, full_matrices=False)
    _uc, _sc, vc = torch.linalg.svd(current_features, full_matrices=False)
    basis_original = vo[: min(rank, vo.shape[0])].T
    basis_current = vc[: min(rank, vc.shape[0])].T
    return torch.linalg.svdvals(basis_original.T @ basis_current).clamp(0, 1)


def recovery_loss_grassmann_geodesic(original_weights, original_activations, current_activations):
    import torch

    cosines = _subspace_cosines(original_activations, current_activations)
    return torch.linalg.vector_norm(torch.acos(cosines.clamp(max=1 - 1e-5)))


def recovery_loss_grassmann_chordal(original_weights, original_activations, current_activations):
    import torch

    cosines = _subspace_cosines(original_activations, current_activations)
    return torch.sqrt((1 - cosines.square()).sum().clamp_min(1e-12))


def recovery_loss_projection_frobenius_subspace(original_weights, original_activations, current_activations):
    return (2**0.5) * recovery_loss_grassmann_chordal(original_weights, original_activations, current_activations)


def recovery_loss_spectral_projection_subspace(original_weights, original_activations, current_activations):
    import torch

    cosines = _subspace_cosines(original_activations, current_activations)
    return torch.sqrt((1 - cosines.square().amin()).clamp_min(1e-12))


def recovery_loss_fubini_study(original_weights, original_activations, current_activations):
    import torch

    cosines = _subspace_cosines(original_activations, current_activations)
    return torch.acos(cosines.prod().clamp(max=1 - 1e-5))


def recovery_loss_binet_cauchy(original_weights, original_activations, current_activations):
    cosines = _subspace_cosines(original_activations, current_activations)
    return 1 - cosines.square().prod()


def recovery_loss_sign_hamming(original_weights, original_activations, current_activations):
    import torch
    import torch.nn.functional as F

    original_sign = torch.where(_activation_vectors(original_activations) >= 0, 1.0, -1.0)
    scale = _activation_vectors(original_activations).detach().std().clamp_min(1e-3)
    return F.softplus(-original_sign * _activation_vectors(current_activations) / scale).mean()


def recovery_loss_soft_jaccard_topk(original_weights, original_activations, current_activations):
    import torch

    original_abs = _activation_vectors(original_activations).abs()
    current_abs = _activation_vectors(current_activations).abs()
    k = max(1, int(original_abs.shape[1] * 0.10))
    threshold = original_abs.topk(k, dim=1).values[:, -1:].detach()
    original_support = (original_abs >= threshold).float()
    temperature = original_abs.detach().std().clamp_min(1e-3)
    current_support = torch.sigmoid((current_abs - threshold) / temperature)
    intersection = (original_support * current_support).sum(dim=1)
    union = (original_support + current_support - original_support * current_support).sum(dim=1).clamp_min(1e-8)
    return (1 - intersection / union).mean()


def recovery_loss_dice_topk(original_weights, original_activations, current_activations):
    import torch

    original_abs = _activation_vectors(original_activations).abs()
    current_abs = _activation_vectors(current_activations).abs()
    k = max(1, int(original_abs.shape[1] * 0.10))
    threshold = original_abs.topk(k, dim=1).values[:, -1:].detach()
    original_support = (original_abs >= threshold).float()
    temperature = original_abs.detach().std().clamp_min(1e-3)
    current_support = torch.sigmoid((current_abs - threshold) / temperature)
    intersection = (original_support * current_support).sum(dim=1)
    denom = (original_support.sum(dim=1) + current_support.sum(dim=1)).clamp_min(1e-8)
    return (1 - 2 * intersection / denom).mean()


def recovery_loss_overlap_topk(original_weights, original_activations, current_activations):
    import torch

    original_abs = _activation_vectors(original_activations).abs()
    current_abs = _activation_vectors(current_activations).abs()
    k = max(1, int(original_abs.shape[1] * 0.10))
    threshold = original_abs.topk(k, dim=1).values[:, -1:].detach()
    original_support = (original_abs >= threshold).float()
    temperature = original_abs.detach().std().clamp_min(1e-3)
    current_support = torch.sigmoid((current_abs - threshold) / temperature)
    overlap = (original_support * current_support).sum(dim=1) / k
    return (1 - overlap).mean()


def recovery_loss_overlap_coefficient_topk(original_weights, original_activations, current_activations):
    import torch

    original_support, current_support = _topk_soft_supports(original_activations, current_activations)
    intersection = (original_support * current_support).sum(dim=1)
    denom = torch.minimum(original_support.sum(dim=1), current_support.sum(dim=1)).clamp_min(1e-8)
    return (1 - intersection / denom).mean()


def recovery_loss_tversky_topk(original_weights, original_activations, current_activations):
    _n00, n01, n10, n11 = _soft_binary_counts(original_activations, current_activations)
    alpha = 0.7
    beta = 0.3
    return (1 - n11 / (n11 + alpha * n10 + beta * n01).clamp_min(1e-8)).mean()


def recovery_loss_simple_matching(original_weights, original_activations, current_activations):
    n00, n01, n10, n11 = _soft_binary_counts(original_activations, current_activations)
    return (1 - (n11 + n00) / (n00 + n01 + n10 + n11).clamp_min(1e-8)).mean()


def recovery_loss_sokal_sneath(original_weights, original_activations, current_activations):
    _n00, n01, n10, n11 = _soft_binary_counts(original_activations, current_activations)
    return (1 - n11 / (n11 + 2 * (n10 + n01)).clamp_min(1e-8)).mean()


def recovery_loss_rogers_tanimoto(original_weights, original_activations, current_activations):
    n00, n01, n10, n11 = _soft_binary_counts(original_activations, current_activations)
    return (1 - (n11 + n00) / (n11 + n00 + 2 * (n10 + n01)).clamp_min(1e-8)).mean()


def recovery_loss_russell_rao(original_weights, original_activations, current_activations):
    n00, n01, n10, n11 = _soft_binary_counts(original_activations, current_activations)
    return (1 - n11 / (n00 + n01 + n10 + n11).clamp_min(1e-8)).mean()


def recovery_loss_yule_distance(original_weights, original_activations, current_activations):
    n00, n01, n10, n11 = _soft_binary_counts(original_activations, current_activations)
    return (2 * n10 * n01 / (n11 * n00 + n10 * n01).clamp_min(1e-8)).mean()


def recovery_loss_tanimoto(original_weights, original_activations, current_activations):
    original_features = _positive_activation_vectors(original_activations)
    current_features = _positive_activation_vectors(current_activations)
    dot = (original_features * current_features).sum(dim=1)
    denom = (original_features.square().sum(dim=1) + current_features.square().sum(dim=1) - dot).clamp_min(1e-8)
    return (1 - dot / denom).mean()


def recovery_loss_ochiai(original_weights, original_activations, current_activations):
    import torch

    original_features = _positive_activation_vectors(original_activations)
    current_features = _positive_activation_vectors(current_activations)
    dot = (original_features * current_features).sum(dim=1)
    denom = torch.sqrt(original_features.square().sum(dim=1) * current_features.square().sum(dim=1)).clamp_min(1e-8)
    return (1 - dot / denom).mean()


def recovery_loss_histogram_intersection(original_weights, original_activations, current_activations):
    import torch

    original_probs = _positive_normalized_vectors(original_activations)
    current_probs = _positive_normalized_vectors(current_activations)
    return (1 - torch.minimum(original_probs, current_probs).sum(dim=1)).mean()


def recovery_loss_ruzicka(original_weights, original_activations, current_activations):
    import torch

    original_features = _positive_activation_vectors(original_activations)
    current_features = _positive_activation_vectors(current_activations)
    return (1 - torch.minimum(original_features, current_features).sum(dim=1) / torch.maximum(original_features, current_features).sum(dim=1).clamp_min(1e-8)).mean()


def recovery_loss_motyka(original_weights, original_activations, current_activations):
    import torch

    original_features = _positive_activation_vectors(original_activations)
    current_features = _positive_activation_vectors(current_activations)
    return (1 - torch.minimum(original_features, current_features).sum(dim=1) / (original_features + current_features).sum(dim=1).clamp_min(1e-8)).mean()


def recovery_loss_czekanowski(original_weights, original_activations, current_activations):
    import torch

    original_features = _positive_activation_vectors(original_activations)
    current_features = _positive_activation_vectors(current_activations)
    return (1 - 2 * torch.minimum(original_features, current_features).sum(dim=1) / (original_features.sum(dim=1) + current_features.sum(dim=1)).clamp_min(1e-8)).mean()


def recovery_loss_wave_hedges(original_weights, original_activations, current_activations):
    import torch

    original_features = _positive_activation_vectors(original_activations)
    current_features = _positive_activation_vectors(current_activations)
    return ((current_features - original_features).abs() / torch.maximum(current_features, original_features).clamp_min(1e-8)).sum(dim=1).mean()


def recovery_loss_inner_product_dissimilarity(original_weights, original_activations, current_activations):
    original_probs = _positive_normalized_vectors(original_activations)
    current_probs = _positive_normalized_vectors(current_activations)
    return (1 - (original_probs * current_probs).sum(dim=1)).mean()


def recovery_loss_fidelity_distance(original_weights, original_activations, current_activations):
    original_probs = _positive_normalized_vectors(original_activations)
    current_probs = _positive_normalized_vectors(current_activations)
    return (1 - (original_probs.sqrt() * current_probs.sqrt()).sum(dim=1)).mean()


def recovery_loss_rank_biased_overlap(original_weights, original_activations, current_activations):
    import torch

    original_abs = _activation_vectors(original_activations).abs()
    current_abs = _activation_vectors(current_activations).abs()
    k = min(64, original_abs.shape[1])
    top = original_abs.topk(k, dim=1).indices
    gathered_current = current_abs.gather(1, top)
    weights = 0.9 ** torch.arange(k, device=original_abs.device, dtype=original_abs.dtype)
    weights = weights / weights.sum()
    target = original_abs.gather(1, top).detach()
    temperature = target.detach().std().clamp_min(1e-3)
    return (weights * torch.sigmoid((target - gathered_current) / temperature)).sum(dim=1).mean()


def recovery_loss_ndcg_topk(original_weights, original_activations, current_activations):
    import torch

    original_abs = _activation_vectors(original_activations).abs()
    current_abs = _activation_vectors(current_activations).abs()
    k = min(64, original_abs.shape[1])
    top = original_abs.topk(k, dim=1).indices
    gains = original_abs.gather(1, top).detach()
    current_scores = current_abs.gather(1, top)
    discounts = 1 / torch.log2(torch.arange(k, device=original_abs.device, dtype=original_abs.dtype) + 2)
    ideal = (gains * discounts).sum(dim=1).clamp_min(1e-8)
    temperature = gains.detach().std().clamp_min(1e-3)
    soft_present = torch.sigmoid((current_scores - gains.amin(dim=1, keepdim=True)) / temperature)
    dcg = (gains * soft_present * discounts).sum(dim=1)
    return (1 - dcg / ideal).mean()


def recovery_loss_rank_correlation(original_weights, original_activations, current_activations):
    import torch
    import torch.nn.functional as F

    original_features = _activation_features(original_activations)[:, :64]
    current_features = _activation_features(current_activations)[:, :64]
    original_diff = original_features[:, :, None] - original_features[:, None, :]
    current_diff = current_features[:, :, None] - current_features[:, None, :]
    target = original_diff.sign()
    valid = target != 0
    scale = original_diff.detach().abs().median().clamp_min(1e-3)
    return F.softplus(-target[valid] * current_diff[valid] / scale).mean()


def recovery_loss_spearman_distance(original_weights, original_activations, current_activations):
    import torch
    import torch.nn.functional as F

    original_features = _activation_features(original_activations)[:, :64]
    current_features = _activation_features(current_activations)[:, :64]
    order = original_features.argsort(dim=1).argsort(dim=1).float()
    temperature = original_features.detach().std().clamp_min(1e-3)
    soft_rank = torch.sigmoid((current_features[:, :, None] - current_features[:, None, :]) / temperature).sum(dim=2)
    order = F.normalize(order - order.mean(dim=1, keepdim=True), dim=1)
    soft_rank = F.normalize(soft_rank - soft_rank.mean(dim=1, keepdim=True), dim=1)
    return 1 - F.cosine_similarity(soft_rank, order, dim=1).mean()


def recovery_loss_kendall_distance(original_weights, original_activations, current_activations):
    return recovery_loss_rank_correlation(original_weights, original_activations, current_activations)


def recovery_loss_distance_correlation(original_weights, original_activations, current_activations):
    import torch
    import torch.nn.functional as F

    original_features = _limited_feature_matrix(original_activations, max_samples=128, max_features=64)
    current_features = _limited_feature_matrix(current_activations, max_samples=128, max_features=64)
    original_dist = torch.cdist(original_features, original_features)
    current_dist = torch.cdist(current_features, current_features)
    original_centered = original_dist - original_dist.mean(dim=0, keepdim=True) - original_dist.mean(dim=1, keepdim=True) + original_dist.mean()
    current_centered = current_dist - current_dist.mean(dim=0, keepdim=True) - current_dist.mean(dim=1, keepdim=True) + current_dist.mean()
    return 1 - F.cosine_similarity(current_centered.flatten()[None], original_centered.flatten()[None], dim=1).mean()


def recovery_loss_hsic_rbf(original_weights, original_activations, current_activations):
    import torch
    import torch.nn.functional as F

    original_features = _limited_feature_matrix(original_activations, max_samples=256, max_features=64)
    current_features = _limited_feature_matrix(current_activations, max_samples=256, max_features=64)
    original_dist = torch.cdist(original_features, original_features).square()
    current_dist = torch.cdist(current_features, current_features).square()
    bw_original = original_dist.detach().median().clamp_min(1e-4)
    bw_current = current_dist.detach().median().clamp_min(1e-4)
    original_kernel = torch.exp(-original_dist / bw_original)
    current_kernel = torch.exp(-current_dist / bw_current)
    original_centered = original_kernel - original_kernel.mean(dim=0, keepdim=True) - original_kernel.mean(dim=1, keepdim=True) + original_kernel.mean()
    current_centered = current_kernel - current_kernel.mean(dim=0, keepdim=True) - current_kernel.mean(dim=1, keepdim=True) + current_kernel.mean()
    return 1 - F.cosine_similarity(current_centered.flatten()[None], original_centered.flatten()[None], dim=1).mean()


def recovery_loss_fisher_rao_categorical(original_weights, original_activations, current_activations):
    import torch

    original_probs = _activation_probabilities(original_activations)
    current_probs = _activation_probabilities(current_activations)
    affinity = (original_probs.sqrt() * current_probs.sqrt()).sum(dim=1).clamp(max=1 - 1e-5)
    return (2 * torch.acos(affinity)).mean()


def recovery_loss_poincare_distance(original_weights, original_activations, current_activations):
    import torch

    original_features = _activation_features(original_activations)
    current_features = _activation_features(current_activations)
    original_norm = torch.linalg.vector_norm(original_features, dim=1, keepdim=True).clamp_min(1e-8)
    current_norm = torch.linalg.vector_norm(current_features, dim=1, keepdim=True).clamp_min(1e-8)
    original_ball = torch.tanh(original_norm / original_features.shape[1] ** 0.5) * original_features / original_norm
    current_ball = torch.tanh(current_norm / current_features.shape[1] ** 0.5) * current_features / current_norm
    diff2 = (current_ball - original_ball).square().sum(dim=1)
    denom = ((1 - original_ball.square().sum(dim=1)) * (1 - current_ball.square().sum(dim=1))).clamp_min(1e-8)
    return torch.acosh(1 + 2 * diff2 / denom).mean()


def recovery_loss_sequence_l2(original_weights, original_activations, current_activations):
    import torch

    original_sequence = _spatial_sequence(original_activations)
    current_sequence = _spatial_sequence(current_activations)
    return torch.linalg.vector_norm(current_sequence - original_sequence, dim=2).mean()


def recovery_loss_derivative_sequence_l2(original_weights, original_activations, current_activations):
    import torch

    original_sequence = _spatial_sequence(original_activations)
    current_sequence = _spatial_sequence(current_activations)
    original_delta = original_sequence[:, 1:] - original_sequence[:, :-1]
    current_delta = current_sequence[:, 1:] - current_sequence[:, :-1]
    return torch.linalg.vector_norm(current_delta - original_delta, dim=2).mean()


def recovery_loss_weighted_sequence_l2(original_weights, original_activations, current_activations):
    import torch

    original_sequence = _spatial_sequence(original_activations)
    current_sequence = _spatial_sequence(current_activations)
    steps = original_sequence.shape[1]
    weights = torch.linspace(1, 2, steps, device=original_sequence.device, dtype=original_sequence.dtype)
    return (weights[None] * torch.linalg.vector_norm(current_sequence - original_sequence, dim=2)).mean()


def recovery_loss_soft_dtw(original_weights, original_activations, current_activations):
    import torch

    original_sequence = _spatial_sequence(original_activations, max_samples=64)
    current_sequence = _spatial_sequence(current_activations, max_samples=64)
    distance = torch.cdist(current_sequence, original_sequence).square()
    gamma = distance.detach().median().clamp_min(1e-3)
    row_softmin = -gamma * torch.logsumexp(-distance / gamma, dim=2)
    col_softmin = -gamma * torch.logsumexp(-distance / gamma, dim=1)
    return 0.5 * (row_softmin.mean() + col_softmin.mean())


def recovery_loss_shape_based_sequence(original_weights, original_activations, current_activations):
    import torch.nn.functional as F

    original_sequence = _spatial_sequence(original_activations).flatten(1)
    current_sequence = _spatial_sequence(current_activations).flatten(1)
    original_sequence = original_sequence - original_sequence.mean(dim=1, keepdim=True)
    current_sequence = current_sequence - current_sequence.mean(dim=1, keepdim=True)
    return 1 - F.cosine_similarity(current_sequence, original_sequence, dim=1).mean()


def recovery_loss_frechet_surrogate(original_weights, original_activations, current_activations):
    import torch

    original_sequence = _spatial_sequence(original_activations)
    current_sequence = _spatial_sequence(current_activations)
    pointwise = torch.linalg.vector_norm(current_sequence - original_sequence, dim=2)
    return pointwise.amax(dim=1).mean()


def recovery_loss_hausdorff_surrogate(original_weights, original_activations, current_activations):
    import torch

    original_sequence = _spatial_sequence(original_activations, max_samples=64)
    current_sequence = _spatial_sequence(current_activations, max_samples=64)
    distance = torch.cdist(current_sequence, original_sequence)
    current_to_original = distance.amin(dim=2)
    original_to_current = distance.amin(dim=1)
    return torch.maximum(current_to_original.amax(dim=1), original_to_current.amax(dim=1)).mean()


def recovery_loss_chamfer_distance(original_weights, original_activations, current_activations):
    import torch

    original_sequence = _spatial_sequence(original_activations, max_samples=64)
    current_sequence = _spatial_sequence(current_activations, max_samples=64)
    distance = torch.cdist(current_sequence, original_sequence).square()
    return distance.amin(dim=2).mean() + distance.amin(dim=1).mean()


def recovery_loss_layernorm_l2(original_weights, original_activations, current_activations):
    import torch
    import torch.nn.functional as F

    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    original_normed = F.layer_norm(original_features, (original_features.shape[1],))
    current_normed = F.layer_norm(current_features, (current_features.shape[1],))
    return torch.linalg.vector_norm(current_normed - original_normed, dim=1).mean()


def recovery_loss_cosine_relative(original_weights, original_activations, current_activations):
    return recovery_loss_relative_l2(original_weights, original_activations, current_activations) + recovery_loss_activation_cosine(original_weights, original_activations, current_activations)


def recovery_loss_orthogonal_relative(original_weights, original_activations, current_activations):
    import torch

    original_features = _activation_vectors(original_activations)
    delta = _activation_vectors(current_activations) - original_features
    scale = (original_features * delta).sum(dim=1, keepdim=True) / original_features.square().sum(dim=1, keepdim=True).clamp_min(1e-8)
    orthogonal = delta - scale * original_features
    return (torch.linalg.vector_norm(orthogonal, dim=1) / torch.linalg.vector_norm(original_features, dim=1).clamp_min(1e-8)).mean()


def recovery_loss_covariance_whitened_cosine(original_weights, original_activations, current_activations):
    import torch.nn.functional as F

    original_features = _limited_feature_matrix(original_activations, max_samples=512, max_features=64)
    current_features = _limited_feature_matrix(current_activations, max_samples=512, max_features=64)
    invsqrt = _matrix_invsqrt_psd(_covariance(original_features)).detach()
    original_white = original_features @ invsqrt
    current_white = current_features @ invsqrt
    return 1 - F.cosine_similarity(current_white, original_white, dim=1).mean()


def recovery_loss_rare_feature_weighted_l2(original_weights, original_activations, current_activations):
    original_features = _activation_vectors(original_activations)
    current_features = _activation_vectors(current_activations)
    weights = 1 / original_features.var(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    return ((current_features - original_features).square() * weights).mean()


def recovery_loss_logdet_covariance(original_weights, original_activations, current_activations):
    import torch

    original_features = _center_columns(_limited_feature_matrix(original_activations, max_samples=512, max_features=64))
    current_features = _center_columns(_limited_feature_matrix(current_activations, max_samples=512, max_features=64))
    n = max(1, original_features.shape[0] - 1)
    original_cov = original_features.T @ original_features / n
    current_cov = current_features.T @ current_features / n
    eye = torch.eye(original_cov.shape[0], device=original_cov.device)
    ridge = original_cov.diag().mean().clamp_min(1e-6) * 1e-3
    original_cov = original_cov + ridge * eye
    current_cov = current_cov + ridge * eye
    precision = torch.linalg.pinv(original_cov)
    return (torch.trace(current_cov @ precision) - torch.linalg.slogdet(current_cov).logabsdet + torch.linalg.slogdet(original_cov).logabsdet - original_cov.shape[0]).clamp_min(0)


def _learned_feature_pair(original_activations, current_activations, max_samples: int = 512, max_features: int = 128):
    original_features = _limited_feature_matrix(original_activations, max_samples=max_samples, max_features=max_features)
    current_features = _limited_feature_matrix(current_activations, max_samples=max_samples, max_features=max_features)
    return original_features, current_features


def _learned_pca(original_features, component_count: int):
    import torch

    mean = original_features.detach().mean(dim=0, keepdim=True)
    centered = original_features.detach() - mean
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    vals, vecs = torch.linalg.eigh((cov + cov.T) * 0.5)
    order = torch.argsort(vals, descending=True)
    k = min(component_count, vecs.shape[1])
    return mean, vecs[:, order[:k]].detach(), vals[order[:k]].clamp_min(1e-6).detach()


def _learned_pca_loss(original_activations, current_activations, component_count: int, whiten: bool = False, residual: bool = False):
    import torch

    original_features, current_features = _learned_feature_pair(original_activations, current_activations)
    mean, basis, vals = _learned_pca(original_features, component_count)
    original_centered = original_features - mean
    current_centered = current_features - mean
    if residual:
        projection = basis @ basis.T
        original_residual = original_centered - original_centered @ projection
        current_residual = current_centered - current_centered @ projection
        return torch.linalg.vector_norm(current_residual - original_residual, dim=1).mean()
    delta = current_centered @ basis - original_centered @ basis
    if whiten:
        delta = delta / vals.sqrt()
    return torch.linalg.vector_norm(delta, dim=1).mean()


def _learned_kmeans_centers(features, center_count: int, iterations: int = 6):
    import torch

    values = features.detach()
    n = values.shape[0]
    k = min(center_count, n)
    indices = torch.linspace(0, n - 1, steps=k, device=values.device).round().long()
    centers = values[indices].clone()
    for _ in range(iterations):
        assignments = torch.cdist(values, centers).argmin(dim=1)
        next_centers = centers.clone()
        for idx in range(k):
            mask = assignments == idx
            if mask.any():
                next_centers[idx] = values[mask].mean(dim=0)
        centers = next_centers
    return centers.detach()


def _learned_soft_assignments(features, centers):
    import torch

    distances = torch.cdist(features, centers).square()
    scale = distances.detach().median().clamp_min(1e-3)
    return torch.softmax(-distances / scale, dim=1)


def _learned_kmeans_softdist_loss(original_activations, current_activations, center_count: int):
    original_features, current_features = _learned_feature_pair(original_activations, current_activations)
    centers = _learned_kmeans_centers(original_features, center_count)
    original_soft = _learned_soft_assignments(original_features, centers).detach()
    current_soft = _learned_soft_assignments(current_features, centers)
    return (current_soft - original_soft).square().sum(dim=1).mean()


def recovery_loss_learned_pca8_l2(original_weights, original_activations, current_activations):
    return _learned_pca_loss(original_activations, current_activations, component_count=8)


def recovery_loss_learned_pca32_l2(original_weights, original_activations, current_activations):
    return _learned_pca_loss(original_activations, current_activations, component_count=32)


def recovery_loss_learned_pca8_whitened_l2(original_weights, original_activations, current_activations):
    return _learned_pca_loss(original_activations, current_activations, component_count=8, whiten=True)


def recovery_loss_learned_pca32_whitened_l2(original_weights, original_activations, current_activations):
    return _learned_pca_loss(original_activations, current_activations, component_count=32, whiten=True)


def recovery_loss_learned_pca32_residual_l2(original_weights, original_activations, current_activations):
    return _learned_pca_loss(original_activations, current_activations, component_count=32, residual=True)


def recovery_loss_learned_shrinkage_mahalanobis(original_weights, original_activations, current_activations):
    import torch

    original_features, current_features = _learned_feature_pair(original_activations, current_activations, max_features=64)
    delta = current_features - original_features
    cov = _covariance(original_features.detach(), ridge_scale=1e-2)
    diag = torch.diag_embed(cov.diag())
    shrink = 0.25 * diag + 0.75 * cov
    precision = torch.linalg.pinv(shrink).detach()
    return ((delta @ precision) * delta).sum(dim=1).mean()


def recovery_loss_learned_variance_attention_l2(original_weights, original_activations, current_activations):
    import torch

    original_features, current_features = _learned_feature_pair(original_activations, current_activations)
    weights = torch.softmax(original_features.detach().var(dim=0, unbiased=False), dim=0)
    return ((current_features - original_features).square() * weights).sum(dim=1).mean()


def recovery_loss_learned_kmeans4_softdist(original_weights, original_activations, current_activations):
    return _learned_kmeans_softdist_loss(original_activations, current_activations, center_count=4)


def recovery_loss_learned_kmeans16_softdist(original_weights, original_activations, current_activations):
    return _learned_kmeans_softdist_loss(original_activations, current_activations, center_count=16)


def recovery_loss_learned_kmeans16_logit_l2(original_weights, original_activations, current_activations):
    import torch

    original_features, current_features = _learned_feature_pair(original_activations, current_activations)
    centers = _learned_kmeans_centers(original_features, 16)
    original_logits = -torch.cdist(original_features, centers).square().detach()
    current_logits = -torch.cdist(current_features, centers).square()
    return torch.linalg.vector_norm(current_logits - original_logits, dim=1).mean()


def recovery_loss_learned_ridge_pseudolabel_logits(original_weights, original_activations, current_activations):
    import torch
    import torch.nn.functional as F

    original_features, current_features = _learned_feature_pair(original_activations, current_activations, max_features=64)
    centers = _learned_kmeans_centers(original_features, 16)
    labels = torch.cdist(original_features.detach(), centers).argmin(dim=1)
    targets = F.one_hot(labels, num_classes=centers.shape[0]).to(dtype=original_features.dtype)
    ones = torch.ones(original_features.shape[0], 1, device=original_features.device, dtype=original_features.dtype)
    design = torch.cat([original_features.detach(), ones], dim=1)
    gram = design.T @ design
    ridge = gram.diag().mean().clamp_min(1e-6) * 1e-2
    gram = gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    weights = torch.linalg.solve(gram, design.T @ targets).detach()
    original_logits = design @ weights
    current_design = torch.cat([current_features, ones], dim=1)
    current_logits = current_design @ weights
    return torch.linalg.vector_norm(current_logits - original_logits.detach(), dim=1).mean()


def recovery_loss_learned_rbf_nystrom_l2(original_weights, original_activations, current_activations):
    import torch

    original_features, current_features = _learned_feature_pair(original_activations, current_activations, max_features=64)
    centers = _learned_kmeans_centers(original_features, 16)
    center_distances = torch.cdist(centers, centers).square()
    nonzero = center_distances > 0
    gamma = center_distances[nonzero].median().clamp_min(1e-3) if nonzero.any() else torch.tensor(1.0, device=centers.device, dtype=centers.dtype)
    original_nystrom = torch.exp(-torch.cdist(original_features, centers).square() / gamma).detach()
    current_nystrom = torch.exp(-torch.cdist(current_features, centers).square() / gamma)
    return torch.linalg.vector_norm(current_nystrom - original_nystrom, dim=1).mean()


PRUNE_RECOVERY_LOSS_SITE_NAMES = {
    "channelwise_fair_layer1_input": {"layers.1.input"},
    "channelwise_fair_layer1_output": {"layers.1.output"},
    "channelwise_fair_layer2_input": {"layers.2.input"},
    "channelwise_fair_layer2_output": {"layers.2.output"},
    "channelwise_fair_layer3_input": {"layers.3.input"},
    "channelwise_fair_layer3_output": {"layers.3.output"},
    "raw_l2_layer1_input": {"layers.1.input"},
    "raw_l2_layer1_output": {"layers.1.output"},
    "raw_l2_layer2_input": {"layers.2.input"},
    "raw_l2_layer2_output": {"layers.2.output"},
}


PRUNE_RECOVERY_LOSSES = {
    "activation_l1": recovery_loss_activation_l1,
    "activation_l2": recovery_loss_activation_l2,
    "activation_squared_l2": recovery_loss_activation_squared_l2,
    "activation_linf": recovery_loss_activation_linf,
    "group_l21": recovery_loss_group_l21,
    "group_l12": recovery_loss_group_l12,
    "fractional_l05": recovery_loss_fractional_l05,
    "canberra": recovery_loss_canberra,
    "clark": recovery_loss_clark,
    "bray_curtis": recovery_loss_bray_curtis,
    "soergel": recovery_loss_soergel,
    "kulczynski": recovery_loss_kulczynski,
    "lorentzian": recovery_loss_lorentzian,
    "gower": recovery_loss_gower,
    "mape": recovery_loss_mape,
    "max_relative": recovery_loss_max_relative,
    "huber": recovery_loss_huber,
    "pseudo_huber": recovery_loss_pseudo_huber,
    "tukey_biweight": recovery_loss_tukey_biweight,
    "cauchy_loss": recovery_loss_cauchy_loss,
    "fair_loss": recovery_loss_fair_loss,
    "channelwise_fair_loss": recovery_loss_channelwise_fair_loss,
    "channelwise_fair_layer1_input": recovery_loss_channelwise_fair_layer1_input,
    "channelwise_fair_layer1_output": recovery_loss_channelwise_fair_layer1_output,
    "channelwise_fair_layer2_input": recovery_loss_channelwise_fair_layer2_input,
    "channelwise_fair_layer2_output": recovery_loss_channelwise_fair_layer2_output,
    "channelwise_fair_layer3_input": recovery_loss_channelwise_fair_layer3_input,
    "channelwise_fair_layer3_output": recovery_loss_channelwise_fair_layer3_output,
    "raw_l2_layer1_input": recovery_loss_raw_l2_layer1_input,
    "raw_l2_layer1_output": recovery_loss_raw_l2_layer1_output,
    "raw_l2_layer2_input": recovery_loss_raw_l2_layer2_input,
    "raw_l2_layer2_output": recovery_loss_raw_l2_layer2_output,
    "welsch_loss": recovery_loss_welsch_loss,
    "cosine_distance": recovery_loss_activation_cosine,
    "angular_distance": recovery_loss_angular_distance,
    "chordal_spherical": recovery_loss_chordal_spherical,
    "correlation_distance": recovery_loss_correlation_distance,
    "sign_invariant_angular": recovery_loss_sign_invariant_angular,
    "projector_frobenius": recovery_loss_projector_frobenius,
    "best_rescaled_l2": recovery_loss_best_rescaled_l2,
    "best_shifted_l2": recovery_loss_best_shifted_l2,
    "best_affine_l2": recovery_loss_best_affine_l2,
    "mahalanobis": recovery_loss_mahalanobis,
    "weighted_l2": recovery_loss_weighted_l2,
    "gram_quadratic": recovery_loss_gram_quadratic,
    "relative_l2": recovery_loss_relative_l2,
    "symmetric_relative_l2": recovery_loss_symmetric_relative_l2,
    "relative_l1": recovery_loss_relative_l1,
    "log_ratio_positive": recovery_loss_log_ratio_positive,
    "aitchison": recovery_loss_aitchison,
    "alr_distance": recovery_loss_alr_distance,
    "hilbert_projective": recovery_loss_hilbert_projective,
    "thompson_metric": recovery_loss_thompson_metric,
    "norm_ratio": recovery_loss_norm_ratio,
    "cosine_preservation": recovery_loss_activation_cosine,
    "parallel_l2": recovery_loss_parallel_l2,
    "orthogonal_l2": recovery_loss_orthogonal_l2,
    "projection_residual": recovery_loss_projection_residual,
    "bregman_log_cosh": recovery_loss_bregman_log_cosh,
    "total_variation": recovery_loss_total_variation,
    "kl_divergence": recovery_loss_kl_divergence,
    "reverse_kl": recovery_loss_reverse_kl,
    "jeffreys": recovery_loss_jeffreys,
    "js_divergence": recovery_loss_js_divergence,
    "hellinger": recovery_loss_hellinger,
    "bhattacharyya": recovery_loss_bhattacharyya,
    "chernoff_alpha_half": recovery_loss_chernoff_alpha_half,
    "matusita": recovery_loss_matusita,
    "itakura_saito": recovery_loss_itakura_saito,
    "chi_square_pearson": recovery_loss_chi_square_pearson,
    "chi_square_neyman": recovery_loss_chi_square_neyman,
    "symmetric_chi_square": recovery_loss_symmetric_chi_square,
    "squared_chord": recovery_loss_squared_chord,
    "triangular_discrimination": recovery_loss_triangular_discrimination,
    "topsoe": recovery_loss_topsoe,
    "renyi_alpha_half": recovery_loss_renyi_alpha_half,
    "alpha_divergence": recovery_loss_alpha_divergence,
    "cressie_read": recovery_loss_cressie_read,
    "tsallis_alpha_half": recovery_loss_tsallis_alpha_half,
    "generalized_i_divergence": recovery_loss_generalized_i_divergence,
    "beta_divergence_half": recovery_loss_beta_divergence_half,
    "wasserstein_1d": recovery_loss_wasserstein_1d,
    "sinkhorn_ot": recovery_loss_sinkhorn_ot,
    "sliced_wasserstein": recovery_loss_sliced_wasserstein,
    "max_sliced_wasserstein": recovery_loss_max_sliced_wasserstein,
    "gromov_wasserstein_surrogate": recovery_loss_gromov_wasserstein_surrogate,
    "rbf_kernel_point": recovery_loss_rbf_kernel_point,
    "polynomial_kernel_point": recovery_loss_polynomial_kernel_point,
    "laplacian_kernel_point": recovery_loss_laplacian_kernel_point,
    "mmd_rbf": recovery_loss_mmd_rbf,
    "energy_distance": recovery_loss_energy_distance,
    "cramer_distance": recovery_loss_cramer_distance,
    "kolmogorov_smirnov": recovery_loss_kolmogorov_smirnov,
    "rsa_distance": recovery_loss_rsa_distance,
    "linear_cka_distance": recovery_loss_linear_cka_distance,
    "svcca_distance": recovery_loss_svcca_distance,
    "pwcca_distance": recovery_loss_pwcca_distance,
    "orthogonal_procrustes": recovery_loss_orthogonal_procrustes,
    "linear_regression_distance": recovery_loss_linear_regression_distance,
    "ridge_regression_distance": recovery_loss_ridge_regression_distance,
    "mutual_nn_overlap": recovery_loss_mutual_nn_overlap,
    "trustworthiness_surrogate": recovery_loss_trustworthiness_surrogate,
    "shape_distance": recovery_loss_shape_distance,
    "normalized_bures": recovery_loss_normalized_bures,
    "covariance_frobenius": recovery_loss_covariance_frobenius,
    "covariance_spectral": recovery_loss_covariance_spectral,
    "covariance_nuclear": recovery_loss_covariance_nuclear,
    "covariance_log_euclidean": recovery_loss_covariance_log_euclidean,
    "covariance_affine_invariant": recovery_loss_covariance_affine_invariant,
    "covariance_bures": recovery_loss_covariance_bures,
    "covariance_stein": recovery_loss_covariance_stein,
    "covariance_condition": recovery_loss_covariance_condition,
    "grassmann_geodesic": recovery_loss_grassmann_geodesic,
    "grassmann_chordal": recovery_loss_grassmann_chordal,
    "projection_frobenius_subspace": recovery_loss_projection_frobenius_subspace,
    "spectral_projection_subspace": recovery_loss_spectral_projection_subspace,
    "fubini_study": recovery_loss_fubini_study,
    "binet_cauchy": recovery_loss_binet_cauchy,
    "sign_hamming": recovery_loss_sign_hamming,
    "soft_jaccard_topk": recovery_loss_soft_jaccard_topk,
    "dice_topk": recovery_loss_dice_topk,
    "overlap_topk": recovery_loss_overlap_topk,
    "overlap_coefficient_topk": recovery_loss_overlap_coefficient_topk,
    "tversky_topk": recovery_loss_tversky_topk,
    "simple_matching": recovery_loss_simple_matching,
    "sokal_sneath": recovery_loss_sokal_sneath,
    "rogers_tanimoto": recovery_loss_rogers_tanimoto,
    "russell_rao": recovery_loss_russell_rao,
    "yule_distance": recovery_loss_yule_distance,
    "tanimoto": recovery_loss_tanimoto,
    "ochiai": recovery_loss_ochiai,
    "histogram_intersection": recovery_loss_histogram_intersection,
    "ruzicka": recovery_loss_ruzicka,
    "motyka": recovery_loss_motyka,
    "czekanowski": recovery_loss_czekanowski,
    "wave_hedges": recovery_loss_wave_hedges,
    "inner_product_dissimilarity": recovery_loss_inner_product_dissimilarity,
    "fidelity_distance": recovery_loss_fidelity_distance,
    "rank_biased_overlap": recovery_loss_rank_biased_overlap,
    "ndcg_topk": recovery_loss_ndcg_topk,
    "rank_correlation": recovery_loss_rank_correlation,
    "spearman_distance": recovery_loss_spearman_distance,
    "kendall_distance": recovery_loss_kendall_distance,
    "distance_correlation": recovery_loss_distance_correlation,
    "hsic_rbf": recovery_loss_hsic_rbf,
    "logdet_covariance": recovery_loss_logdet_covariance,
    "fisher_rao_categorical": recovery_loss_fisher_rao_categorical,
    "poincare_distance": recovery_loss_poincare_distance,
    "sequence_l2": recovery_loss_sequence_l2,
    "derivative_sequence_l2": recovery_loss_derivative_sequence_l2,
    "weighted_sequence_l2": recovery_loss_weighted_sequence_l2,
    "soft_dtw": recovery_loss_soft_dtw,
    "shape_based_sequence": recovery_loss_shape_based_sequence,
    "frechet_surrogate": recovery_loss_frechet_surrogate,
    "hausdorff_surrogate": recovery_loss_hausdorff_surrogate,
    "chamfer_distance": recovery_loss_chamfer_distance,
    "layernorm_l2": recovery_loss_layernorm_l2,
    "cosine_relative": recovery_loss_cosine_relative,
    "orthogonal_relative": recovery_loss_orthogonal_relative,
    "covariance_whitened_cosine": recovery_loss_covariance_whitened_cosine,
    "rare_feature_weighted_l2": recovery_loss_rare_feature_weighted_l2,
    "learned_pca8_l2": recovery_loss_learned_pca8_l2,
    "learned_pca32_l2": recovery_loss_learned_pca32_l2,
    "learned_pca8_whitened_l2": recovery_loss_learned_pca8_whitened_l2,
    "learned_pca32_whitened_l2": recovery_loss_learned_pca32_whitened_l2,
    "learned_pca32_residual_l2": recovery_loss_learned_pca32_residual_l2,
    "learned_shrinkage_mahalanobis": recovery_loss_learned_shrinkage_mahalanobis,
    "learned_variance_attention_l2": recovery_loss_learned_variance_attention_l2,
    "learned_kmeans4_softdist": recovery_loss_learned_kmeans4_softdist,
    "learned_kmeans16_softdist": recovery_loss_learned_kmeans16_softdist,
    "learned_kmeans16_logit_l2": recovery_loss_learned_kmeans16_logit_l2,
    "learned_ridge_pseudolabel_logits": recovery_loss_learned_ridge_pseudolabel_logits,
    "learned_rbf_nystrom_l2": recovery_loss_learned_rbf_nystrom_l2,
    "activation_mse": recovery_loss_activation_mse,
    "activation_cosine": recovery_loss_activation_cosine,
}


def _replace_with_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"Refusing to replace non-symlink path: {link_path}")
    os.symlink(target_path, link_path, target_is_directory=True)


def prepare_runtime_dirs(repo_root: Path, cache_root: Path, output_dir: Path) -> None:
    cifar_dir = cache_root / "cifar10"
    logs_dir = output_dir / "logs"
    cifar_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _replace_with_symlink(repo_root / "cifar10", cifar_dir)
    _replace_with_symlink(repo_root / "logs", logs_dir)


def _load_script_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("airbench94_muon_remote", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load AirBench script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    old_argv0 = sys.argv[0]
    sys.argv[0] = str(script_path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv[0] = old_argv0
    return module


def run_airbench94_muon_local(
    config: AirBenchRunConfig,
    repo_root: Path = AIRBENCH_ROOT,
    cache_root: Path = CACHE_ROOT,
    results_root: Path = RESULTS_ROOT,
) -> dict:
    if config.script_name != "airbench94_muon.py":
        raise ValueError("Only the official airbench94_muon.py runner is wired here")
    if config.n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    import torch

    output_dir = results_root / config.run_name
    prepare_runtime_dirs(repo_root, cache_root, output_dir)
    os.chdir(repo_root)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    module = _load_script_module(repo_root / config.script_name)
    model = module.CifarNet().cuda().to(memory_format=torch.channels_last)
    if config.compile_mode:
        model.compile(mode=config.compile_mode)

    started = time.perf_counter()
    module.print_columns(module.logging_columns_list, is_head=True)
    if config.warmup:
        module.main("warmup", model)
    accs = torch.tensor([module.main(run, model) for run in range(config.n_runs)])
    wall_time_seconds = time.perf_counter() - started

    summary = {
        "config": asdict(config),
        "airbench_repo_url": AIRBENCH_REPO_URL,
        "airbench_ref": AIRBENCH_REF,
        "mean_accuracy": float(accs.mean().item()),
        "std_accuracy": float(accs.std(unbiased=False).item()),
        "n_runs": int(config.n_runs),
        "wall_time_seconds": wall_time_seconds,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "result_dir": str(output_dir),
    }

    torch.save({"config": asdict(config), "accs": accs, "summary": summary}, output_dir / "log.pt")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_airbench94_muon_partial_checkpoint_local(
    config: AirBenchPartialCheckpointConfig,
    repo_root: Path = AIRBENCH_ROOT,
    cache_root: Path = CACHE_ROOT,
    results_root: Path = RESULTS_ROOT,
) -> dict:
    if config.script_name != "airbench94_muon.py":
        raise ValueError("Only the official airbench94_muon.py runner is wired here")

    import torch

    output_dir = results_root / config.run_name
    prepare_runtime_dirs(repo_root, cache_root, output_dir)
    os.chdir(repo_root)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    module = _load_script_module(repo_root / config.script_name)
    model = module.CifarNet().cuda().to(memory_format=torch.channels_last)
    if config.compile_mode:
        model.compile(mode=config.compile_mode)

    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * config.batch_size

    test_loader = module.CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = module.CifarLoader(
        "cifar10",
        train=True,
        batch_size=config.batch_size,
        aug=dict(flip=True, translate=2),
    )
    total_train_steps, checkpoint_steps = compute_checkpoint_steps(
        len(train_loader),
        config.total_epochs,
        config.checkpoint_fraction,
    )
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True, fused=True)
    optimizer2 = module.Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0

    def start_timer() -> None:
        starter.record()

    def stop_timer() -> None:
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    started = time.perf_counter()
    model.reset()
    step = 0

    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    module.print_columns(module.logging_columns_list, is_head=True)
    last_metrics: dict[str, float | int | str] = {}
    for epoch in range(ceil(checkpoint_steps / len(train_loader))):
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            module.F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum").backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= checkpoint_steps:
                break
        stop_timer()

        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = module.evaluate(model, test_loader, tta_level=0)
        last_metrics = {
            "epoch": epoch,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "time_seconds": time_seconds,
        }
        module.print_training_details(locals(), is_final_entry=False)

    start_timer()
    tta_val_acc = module.evaluate(model, test_loader, tta_level=2)
    stop_timer()
    wall_time_seconds = time.perf_counter() - started

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "chosen_layer_name": config.chosen_layer_name,
            "step": step,
            "total_train_steps": total_train_steps,
            "airbench_ref": AIRBENCH_REF,
        },
        checkpoint_path,
    )
    summary = {
        "config": asdict(config),
        "airbench_repo_url": AIRBENCH_REPO_URL,
        "airbench_ref": AIRBENCH_REF,
        "checkpoint_path": str(checkpoint_path),
        "chosen_layer_name": config.chosen_layer_name,
        "checkpoint_fraction": config.checkpoint_fraction,
        "checkpoint_steps": checkpoint_steps,
        "total_train_steps": total_train_steps,
        "completed_epoch_fraction": step / len(train_loader),
        "train_acc": float(last_metrics["train_acc"]),
        "val_acc": float(last_metrics["val_acc"]),
        "tta_val_acc": float(tta_val_acc),
        "time_seconds": time_seconds,
        "wall_time_seconds": wall_time_seconds,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "result_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_airbench94_muon_plateau_checkpoint_local(
    config: AirBenchPlateauCheckpointConfig,
    repo_root: Path = AIRBENCH_ROOT,
    cache_root: Path = CACHE_ROOT,
    results_root: Path = RESULTS_ROOT,
) -> dict:
    if config.script_name != "airbench94_muon.py":
        raise ValueError("Only the official airbench94_muon.py runner is wired here")
    if config.train_steps < 1:
        raise ValueError("train_steps must be at least 1")

    import torch

    output_dir = results_root / config.run_name
    prepare_runtime_dirs(repo_root, cache_root, output_dir)
    os.chdir(repo_root)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    module = _load_script_module(repo_root / config.script_name)
    model = module.CifarNet().cuda().to(memory_format=torch.channels_last)
    if config.compile_mode:
        model.compile(mode=config.compile_mode)

    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * config.batch_size

    test_loader = module.CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = module.CifarLoader(
        "cifar10",
        train=True,
        batch_size=config.batch_size,
        aug=dict(flip=True, translate=2),
    )
    total_train_steps = config.train_steps
    whiten_bias_train_steps = min(ceil(3 * len(train_loader)), total_train_steps)

    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True, fused=True)
    optimizer2 = module.Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0

    def start_timer() -> None:
        starter.record()

    def stop_timer() -> None:
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    started = time.perf_counter()
    model.reset()
    step = 0

    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    module.print_columns(module.logging_columns_list, is_head=True)
    last_metrics: dict[str, float | int | str] = {}
    val_history = []
    for epoch in range(ceil(total_train_steps / len(train_loader))):
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            module.F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum").backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * max(0.0, 1 - step / whiten_bias_train_steps)
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * max(0.0, 1 - step / total_train_steps)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()

        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = module.evaluate(model, test_loader, tta_level=0)
        last_metrics = {
            "epoch": epoch,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "time_seconds": time_seconds,
        }
        val_history.append({"epoch": int(epoch), "step": int(step), "val_acc": float(val_acc)})
        module.print_training_details(locals(), is_final_entry=False)

    start_timer()
    tta_val_acc = module.evaluate(model, test_loader, tta_level=2)
    stop_timer()
    wall_time_seconds = time.perf_counter() - started

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "chosen_layer_name": config.chosen_layer_name,
            "step": step,
            "total_train_steps": total_train_steps,
            "val_history": val_history,
            "airbench_ref": AIRBENCH_REF,
        },
        checkpoint_path,
    )
    summary = {
        "config": asdict(config),
        "airbench_repo_url": AIRBENCH_REPO_URL,
        "airbench_ref": AIRBENCH_REF,
        "checkpoint_path": str(checkpoint_path),
        "chosen_layer_name": config.chosen_layer_name,
        "train_steps": total_train_steps,
        "completed_epoch_fraction": step / len(train_loader),
        "train_acc": float(last_metrics["train_acc"]),
        "val_acc": float(last_metrics["val_acc"]),
        "tta_val_acc": float(tta_val_acc),
        "val_history": val_history,
        "time_seconds": time_seconds,
        "wall_time_seconds": wall_time_seconds,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "result_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _evaluate_full_model(module, model, test_loader) -> dict:
    val_acc = module.evaluate(model, test_loader, tta_level=0)
    tta_val_acc = module.evaluate(model, test_loader, tta_level=2)
    return {"val_acc": float(val_acc), "tta_val_acc": float(tta_val_acc)}


def run_airbench94_muon_prune_recovery_local(
    config: AirBenchPruneRecoveryConfig,
    repo_root: Path = AIRBENCH_ROOT,
    cache_root: Path = CACHE_ROOT,
    results_root: Path = RESULTS_ROOT,
) -> dict:
    if config.recovery_steps < 1:
        raise ValueError("recovery_steps must be at least 1")
    if config.recovery_loss_name not in PRUNE_RECOVERY_LOSSES:
        raise ValueError(f"Unknown recovery loss: {config.recovery_loss_name}")

    import torch

    output_dir = results_root / config.run_name
    prepare_runtime_dirs(repo_root, cache_root, output_dir)
    os.chdir(repo_root)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    module = _load_script_module(repo_root / "airbench94_muon.py")
    checkpoint_path = results_root / config.checkpoint_run_name / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda")

    original_model = module.CifarNet().cuda().to(memory_format=torch.channels_last)
    original_model.load_state_dict(checkpoint["model_state_dict"])
    original_model.eval()

    current_model = module.CifarNet().cuda().to(memory_format=torch.channels_last)
    current_model.load_state_dict(checkpoint["model_state_dict"])
    prune_masks, prune_stats = random_prune_layer_weights_(
        current_model,
        config.prune_layer_name,
        config.prune_fraction,
        config.prune_seed,
    )
    if config.compile_mode:
        current_model.compile(mode=config.compile_mode)

    train_loader = module.CifarLoader(
        "cifar10",
        train=True,
        batch_size=config.calibration_sample_count,
        aug=None,
    )
    if config.calibration_sample_count > len(train_loader.images):
        raise ValueError("calibration_sample_count exceeds the train set size")
    calibration_inputs = train_loader.normalize(train_loader.images[: config.calibration_sample_count])
    calibration_labels = _loader_labels(train_loader)[: config.calibration_sample_count]
    test_loader = module.CifarLoader("cifar10", train=False, batch_size=2000)

    original_weights = {
        name: param.detach().clone()
        for name, param in original_model.named_parameters()
    }
    required_site_names = PRUNE_RECOVERY_LOSS_SITE_NAMES.get(config.recovery_loss_name)
    if required_site_names:
        max_required_layer = max(_parse_layers_index(site.rsplit(".", 1)[0]) for site in required_site_names)
        if max_required_layer > _parse_layers_index(config.chosen_layer_name):
            raise ValueError(f"{config.recovery_loss_name} requires layers.{max_required_layer}, but chosen_layer_name is {config.chosen_layer_name}")
    with torch.no_grad():
        if required_site_names:
            original_activations = {
                key: value.detach()
                for key, value in forward_layer_activation_sites(
                    original_model,
                    calibration_inputs,
                    required_site_names,
                    whiten_bias_grad=False,
                ).items()
            }
        else:
            original_activations = forward_to_chosen_layer(
                original_model,
                calibration_inputs,
                config.chosen_layer_name,
                whiten_bias_grad=False,
            ).detach()

    original_eval = _evaluate_full_model(module, original_model, test_loader)
    pruned_eval = _evaluate_full_model(module, current_model, test_loader)
    pruned_model_state_dict = {name: value.detach().clone() for name, value in current_model.state_dict().items()}
    audit_site_names = mirage_audit_site_names(config.chosen_layer_name)
    with torch.no_grad():
        original_audit_sites = {
            key: value.detach()
            for key, value in forward_layer_activation_sites(
                original_model,
                calibration_inputs,
                audit_site_names,
                whiten_bias_grad=False,
            ).items()
        }
        pruned_audit_sites = {
            key: value.detach()
            for key, value in forward_layer_activation_sites(
                current_model,
                calibration_inputs,
                audit_site_names,
                whiten_bias_grad=False,
            ).items()
        }

    prefix_names = prefix_parameter_names(current_model, config.chosen_layer_name)
    trainable_params = []
    for name, param in current_model.named_parameters():
        param.requires_grad_(name in prefix_names)
        if param.requires_grad:
            trainable_params.append(param)
    if not trainable_params:
        raise RuntimeError(f"No trainable prefix parameters found for {config.chosen_layer_name}")

    optimizer = torch.optim.SGD(trainable_params, lr=config.learning_rate)
    recovery_loss_fn = PRUNE_RECOVERY_LOSSES[config.recovery_loss_name]
    recovery_losses = []
    started = time.perf_counter()
    current_model.eval()
    for _step in range(config.recovery_steps):
        if required_site_names:
            current_activations = forward_layer_activation_sites(
                current_model,
                calibration_inputs,
                required_site_names,
                whiten_bias_grad=False,
            )
        else:
            current_activations = forward_to_chosen_layer(
                current_model,
                calibration_inputs,
                config.chosen_layer_name,
                whiten_bias_grad=False,
            )
        loss = recovery_loss_fn(
            original_weights=original_weights,
            original_activations=original_activations,
            current_activations=current_activations,
        )
        optimizer.zero_grad(set_to_none=True)
        if loss.requires_grad:
            loss.backward()
            apply_prune_masks_to_grads_(current_model, prune_masks)
            optimizer.step()
        apply_prune_masks_to_params_(current_model, prune_masks)
        recovery_losses.append(float(loss.detach().cpu()))
    recovery_wall_time_seconds = time.perf_counter() - started

    recovered_eval = _evaluate_full_model(module, current_model, test_loader)
    with torch.no_grad():
        recovered_audit_sites = {
            key: value.detach()
            for key, value in forward_layer_activation_sites(
                current_model,
                calibration_inputs,
                audit_site_names,
                whiten_bias_grad=False,
            ).items()
        }
    mirage_audit = {
        "site_names": sorted(audit_site_names),
        "pruned_vs_original": mirage_site_audit(original_audit_sites, pruned_audit_sites, calibration_labels),
        "recovered_vs_original": mirage_site_audit(original_audit_sites, recovered_audit_sites, calibration_labels),
    }
    zero_mask_violations = 0
    with torch.no_grad():
        for name, param in current_model.named_parameters():
            if name in prune_masks:
                zero_mask_violations += int((param[~prune_masks[name]] != 0).sum().item())

    pruned_checkpoint_path = output_dir / "pruned_checkpoint.pt"
    recovered_checkpoint_path = output_dir / "recovered_checkpoint.pt"
    mask_path = output_dir / "prune_masks.pt"
    activations_path = output_dir / "original_activations.pt"
    torch.save(
        {
            "source_model_state_dict": checkpoint["model_state_dict"],
            "model_state_dict": pruned_model_state_dict,
            "config": asdict(config),
            "prune_stats": prune_stats,
            "airbench_ref": AIRBENCH_REF,
        },
        pruned_checkpoint_path,
    )
    torch.save(
        {
            "model_state_dict": current_model.state_dict(),
            "source_checkpoint_run_name": config.checkpoint_run_name,
            "source_checkpoint_path": str(checkpoint_path),
            "config": asdict(config),
            "chosen_layer_name": config.chosen_layer_name,
            "prune_layer_name": config.prune_layer_name,
            "prune_stats": prune_stats,
            "recovery_losses": recovery_losses,
            "airbench_ref": AIRBENCH_REF,
        },
        recovered_checkpoint_path,
    )
    torch.save({"masks": prune_masks, "prune_stats": prune_stats}, mask_path)
    torch.save({"original_activations": original_activations, "chosen_layer_name": config.chosen_layer_name}, activations_path)

    summary = {
        "config": asdict(config),
        "airbench_repo_url": AIRBENCH_REPO_URL,
        "airbench_ref": AIRBENCH_REF,
        "source_checkpoint_path": str(checkpoint_path),
        "pruned_checkpoint_path": str(pruned_checkpoint_path),
        "recovered_checkpoint_path": str(recovered_checkpoint_path),
        "mask_path": str(mask_path),
        "original_activations_path": str(activations_path),
        "chosen_layer_name": config.chosen_layer_name,
        "prune_layer_name": config.prune_layer_name,
        "prune_stats": prune_stats,
        "recovery_loss_name": config.recovery_loss_name,
        "recovery_losses": recovery_losses,
        "recovery_steps": config.recovery_steps,
        "learning_rate": config.learning_rate,
        "calibration_sample_count": config.calibration_sample_count,
        "prefix_parameter_names": sorted(prefix_names),
        "original_eval": original_eval,
        "pruned_eval": pruned_eval,
        "recovered_eval": recovered_eval,
        "mirage_audit": mirage_audit,
        "zero_mask_violations": zero_mask_violations,
        "recovery_wall_time_seconds": recovery_wall_time_seconds,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "result_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_airbench94_muon_prune_recovery_batch_local(
    run_name: str,
    recovery_loss_names: list[str],
    checkpoint_run_name: str,
    chosen_layer_name: str,
    prune_layer_name: str,
    prune_fraction: float,
    prune_seed: int,
    recovery_steps: int,
    calibration_sample_count: int,
    learning_rate: float,
    seed: int,
    compile_mode: str,
) -> dict:
    if not recovery_loss_names:
        raise ValueError("recovery_loss_names must not be empty")
    results = []
    for loss_name in recovery_loss_names:
        config = AirBenchPruneRecoveryConfig(
            run_name=f"{run_name}/{loss_name}",
            checkpoint_run_name=checkpoint_run_name,
            chosen_layer_name=chosen_layer_name,
            prune_layer_name=prune_layer_name,
            prune_fraction=prune_fraction,
            prune_seed=prune_seed,
            recovery_loss_name=loss_name,
            recovery_steps=recovery_steps,
            calibration_sample_count=calibration_sample_count,
            learning_rate=learning_rate,
            seed=seed,
            compile_mode=compile_mode,
        )
        results.append(run_airbench94_muon_prune_recovery_local(config))

    output_dir = RESULTS_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_name": run_name,
        "checkpoint_run_name": checkpoint_run_name,
        "chosen_layer_name": chosen_layer_name,
        "prune_layer_name": prune_layer_name,
        "prune_fraction": prune_fraction,
        "prune_seed": prune_seed,
        "recovery_loss_names": recovery_loss_names,
        "recovery_steps": recovery_steps,
        "calibration_sample_count": calibration_sample_count,
        "learning_rate": learning_rate,
        "results": results,
    }
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_airbench94_muon_proxy_optimize_local(
    config: AirBenchProxyOptimizeConfig,
    repo_root: Path = AIRBENCH_ROOT,
    cache_root: Path = CACHE_ROOT,
    results_root: Path = RESULTS_ROOT,
) -> dict:
    if config.proxy_steps < 1:
        raise ValueError("proxy_steps must be at least 1")
    if config.proxy_loss_name not in PROXY_LOSSES:
        raise ValueError(f"Unknown proxy loss: {config.proxy_loss_name}")

    import torch

    output_dir = results_root / config.run_name
    prepare_runtime_dirs(repo_root, cache_root, output_dir)
    os.chdir(repo_root)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    module = _load_script_module(repo_root / "airbench94_muon.py")
    model = module.CifarNet().cuda().to(memory_format=torch.channels_last)
    checkpoint_path = results_root / config.checkpoint_run_name / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    if config.compile_mode:
        model.compile(mode=config.compile_mode)

    train_loader = module.CifarLoader(
        "cifar10",
        train=True,
        batch_size=config.batch_size,
        aug=dict(flip=True, translate=2),
    )
    test_loader = module.CifarLoader("cifar10", train=False, batch_size=2000)

    prefix_names = prefix_parameter_names(model, config.chosen_layer_name)
    prefix_params = []
    for name, param in model.named_parameters():
        param.requires_grad_(name in prefix_names)
        if param.requires_grad:
            prefix_params.append(param)
    if not prefix_params:
        raise RuntimeError(f"No trainable prefix parameters found for {config.chosen_layer_name}")

    before_eval = _evaluate_full_model(module, model, test_loader)
    optimizer = torch.optim.SGD(prefix_params, lr=config.learning_rate)
    proxy_loss_fn = PROXY_LOSSES[config.proxy_loss_name]
    proxy_losses: list[float] = []

    started = time.perf_counter()
    model.train()
    step = 0
    while step < config.proxy_steps:
        for inputs, labels in train_loader:
            activations = forward_to_chosen_layer(model, inputs, config.chosen_layer_name)
            if config.proxy_loss_name in PROXY_LOSSES_WITH_SECOND_VIEW:
                second_activations = forward_to_chosen_layer(model, inputs.flip(-1), config.chosen_layer_name)
            else:
                second_activations = None
            loss = proxy_loss_fn(activations, labels=labels, second_activations=second_activations)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            proxy_losses.append(float(loss.detach().cpu()))
            step += 1
            if step >= config.proxy_steps:
                break
    proxy_wall_time_seconds = time.perf_counter() - started

    after_eval = _evaluate_full_model(module, model, test_loader)
    optimized_checkpoint_path = output_dir / "proxy_optimized_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "source_checkpoint_run_name": config.checkpoint_run_name,
            "source_checkpoint_path": str(checkpoint_path),
            "config": asdict(config),
            "chosen_layer_name": config.chosen_layer_name,
            "proxy_losses": proxy_losses,
            "airbench_ref": AIRBENCH_REF,
        },
        optimized_checkpoint_path,
    )
    summary = {
        "config": asdict(config),
        "airbench_repo_url": AIRBENCH_REPO_URL,
        "airbench_ref": AIRBENCH_REF,
        "source_checkpoint_path": str(checkpoint_path),
        "optimized_checkpoint_path": str(optimized_checkpoint_path),
        "chosen_layer_name": config.chosen_layer_name,
        "proxy_loss_name": config.proxy_loss_name,
        "proxy_losses": proxy_losses,
        "proxy_steps": config.proxy_steps,
        "prefix_parameter_names": sorted(prefix_names),
        "before_eval": before_eval,
        "after_eval": after_eval,
        "proxy_wall_time_seconds": proxy_wall_time_seconds,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "result_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={str(RESULTS_ROOT): results_volume, str(CACHE_ROOT): cache_volume},
)
def run_airbench94_muon(
    run_name: str = "airbench94_muon_h100_smoke",
    n_runs: int = 1,
    warmup: bool = True,
    compile_mode: str = "max-autotune",
    seed: int = 0,
) -> dict:
    config = AirBenchRunConfig(
        run_name=run_name,
        n_runs=n_runs,
        warmup=warmup,
        compile_mode=compile_mode,
        seed=seed,
    )
    summary = run_airbench94_muon_local(config)
    results_volume.commit()
    cache_volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={str(RESULTS_ROOT): results_volume, str(CACHE_ROOT): cache_volume},
)
def run_airbench94_muon_partial_checkpoint(
    run_name: str = "airbench94_muon_three_quarter_checkpoint",
    checkpoint_fraction: float = 0.75,
    compile_mode: str = "max-autotune",
    seed: int = 0,
) -> dict:
    config = AirBenchPartialCheckpointConfig(
        run_name=run_name,
        checkpoint_fraction=checkpoint_fraction,
        compile_mode=compile_mode,
        seed=seed,
    )
    summary = run_airbench94_muon_partial_checkpoint_local(config)
    results_volume.commit()
    cache_volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={str(RESULTS_ROOT): results_volume, str(CACHE_ROOT): cache_volume},
)
def run_airbench94_muon_plateau_checkpoint(
    run_name: str = "airbench94_muon_plateau_500step",
    train_steps: int = 500,
    compile_mode: str = "max-autotune",
    seed: int = 0,
) -> dict:
    config = AirBenchPlateauCheckpointConfig(
        run_name=run_name,
        train_steps=train_steps,
        compile_mode=compile_mode,
        seed=seed,
    )
    summary = run_airbench94_muon_plateau_checkpoint_local(config)
    results_volume.commit()
    cache_volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={str(RESULTS_ROOT): results_volume, str(CACHE_ROOT): cache_volume},
)
def run_airbench94_muon_prune_recovery(
    run_name: str = "airbench94_muon_prune_recovery",
    checkpoint_run_name: str = "airbench94_muon_plateau_500step_20260528",
    chosen_layer_name: str = "layers.2",
    prune_layer_name: str = "layers.1",
    prune_fraction: float = 0.10,
    prune_seed: int = 12345,
    recovery_loss_name: str = "activation_mse",
    recovery_steps: int = 10,
    calibration_sample_count: int = 2000,
    learning_rate: float = 1e-4,
    seed: int = 0,
    compile_mode: str = "",
) -> dict:
    config = AirBenchPruneRecoveryConfig(
        run_name=run_name,
        checkpoint_run_name=checkpoint_run_name,
        chosen_layer_name=chosen_layer_name,
        prune_layer_name=prune_layer_name,
        prune_fraction=prune_fraction,
        prune_seed=prune_seed,
        recovery_loss_name=recovery_loss_name,
        recovery_steps=recovery_steps,
        calibration_sample_count=calibration_sample_count,
        learning_rate=learning_rate,
        seed=seed,
        compile_mode=compile_mode,
    )
    summary = run_airbench94_muon_prune_recovery_local(config)
    results_volume.commit()
    cache_volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={str(RESULTS_ROOT): results_volume, str(CACHE_ROOT): cache_volume},
)
def run_airbench94_muon_prune_recovery_batch(
    run_name: str = "airbench94_muon_prune_recovery_batch",
    recovery_loss_names: str = DEFAULT_PRUNE_RECOVERY_LOSS_NAMES,
    checkpoint_run_name: str = "airbench94_muon_plateau_500step_20260528",
    chosen_layer_name: str = "layers.2",
    prune_layer_name: str = "layers.1",
    prune_fraction: float = 0.10,
    prune_seed: int = 12345,
    recovery_steps: int = 10,
    calibration_sample_count: int = 2000,
    learning_rate: float = 1e-4,
    seed: int = 0,
    compile_mode: str = "",
) -> dict:
    loss_names = [name.strip() for name in recovery_loss_names.split(",") if name.strip()]
    summary = run_airbench94_muon_prune_recovery_batch_local(
        run_name=run_name,
        recovery_loss_names=loss_names,
        checkpoint_run_name=checkpoint_run_name,
        chosen_layer_name=chosen_layer_name,
        prune_layer_name=prune_layer_name,
        prune_fraction=prune_fraction,
        prune_seed=prune_seed,
        recovery_steps=recovery_steps,
        calibration_sample_count=calibration_sample_count,
        learning_rate=learning_rate,
        seed=seed,
        compile_mode=compile_mode,
    )
    results_volume.commit()
    cache_volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={str(RESULTS_ROOT): results_volume, str(CACHE_ROOT): cache_volume},
)
def run_airbench94_muon_proxy_optimize(
    run_name: str = "airbench94_muon_proxy_optimize_smoke",
    checkpoint_run_name: str = "airbench94_muon_three_quarter_checkpoint_20260527",
    chosen_layer_name: str = "layers.2",
    proxy_loss_name: str = "activation_l2",
    proxy_steps: int = 10,
    learning_rate: float = 1e-4,
    seed: int = 0,
    compile_mode: str = "",
) -> dict:
    config = AirBenchProxyOptimizeConfig(
        run_name=run_name,
        checkpoint_run_name=checkpoint_run_name,
        chosen_layer_name=chosen_layer_name,
        proxy_loss_name=proxy_loss_name,
        proxy_steps=proxy_steps,
        learning_rate=learning_rate,
        seed=seed,
        compile_mode=compile_mode,
    )
    summary = run_airbench94_muon_proxy_optimize_local(config)
    results_volume.commit()
    cache_volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={str(RESULTS_ROOT): results_volume, str(CACHE_ROOT): cache_volume},
)
def run_airbench94_muon_proxy_batch(
    run_name: str = "airbench94_muon_proxy_batch",
    proxy_loss_names: str = "activation_l2,activation_l1,activation_linf,cosine_sim,same_class_contrastive",
    checkpoint_run_name: str = "airbench94_muon_three_quarter_checkpoint_20260527",
    chosen_layer_name: str = "layers.2",
    proxy_steps: int = 10,
    learning_rate: float = 1e-4,
    seed: int = 0,
    compile_mode: str = "",
) -> dict:
    loss_names = [name.strip() for name in proxy_loss_names.split(",") if name.strip()]
    results = []
    for loss_name in loss_names:
        config = AirBenchProxyOptimizeConfig(
            run_name=f"{run_name}/{loss_name}",
            checkpoint_run_name=checkpoint_run_name,
            chosen_layer_name=chosen_layer_name,
            proxy_loss_name=loss_name,
            proxy_steps=proxy_steps,
            learning_rate=learning_rate,
            seed=seed,
            compile_mode=compile_mode,
        )
        results.append(run_airbench94_muon_proxy_optimize_local(config))
    output_dir = RESULTS_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_name": run_name,
        "checkpoint_run_name": checkpoint_run_name,
        "chosen_layer_name": chosen_layer_name,
        "proxy_loss_names": loss_names,
        "proxy_steps": proxy_steps,
        "learning_rate": learning_rate,
        "results": results,
    }
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    results_volume.commit()
    cache_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    run_name: str = "airbench94_muon_h100_smoke",
    n_runs: int = 1,
    warmup: bool = True,
    compile_mode: str = "max-autotune",
    seed: int = 0,
    partial_checkpoint: bool = False,
    checkpoint_fraction: float = 0.75,
    plateau_checkpoint: bool = False,
    train_steps: int = 500,
    proxy_optimize: bool = False,
    proxy_batch: bool = False,
    prune_recovery: bool = False,
    prune_recovery_batch: bool = False,
    checkpoint_run_name: str = "airbench94_muon_three_quarter_checkpoint_20260527",
    chosen_layer_name: str = "layers.2",
    prune_layer_name: str = "layers.1",
    prune_fraction: float = 0.10,
    prune_seed: int = 12345,
    recovery_loss_name: str = "activation_mse",
    recovery_loss_names: str = DEFAULT_PRUNE_RECOVERY_LOSS_NAMES,
    recovery_steps: int = 10,
    calibration_sample_count: int = 2000,
    proxy_loss_names: str = "activation_l2,activation_l1,activation_linf,cosine_sim,same_class_contrastive",
    proxy_loss_name: str = "activation_l2",
    proxy_steps: int = 10,
    learning_rate: float = 1e-4,
    proxy_compile_mode: str = "",
) -> None:
    if prune_recovery_batch:
        if checkpoint_run_name == "airbench94_muon_three_quarter_checkpoint_20260527":
            checkpoint_run_name = AirBenchPruneRecoveryConfig.checkpoint_run_name
        summary = run_airbench94_muon_prune_recovery_batch.remote(
            run_name=run_name,
            recovery_loss_names=recovery_loss_names,
            checkpoint_run_name=checkpoint_run_name,
            chosen_layer_name=chosen_layer_name,
            prune_layer_name=prune_layer_name,
            prune_fraction=prune_fraction,
            prune_seed=prune_seed,
            recovery_steps=recovery_steps,
            calibration_sample_count=calibration_sample_count,
            learning_rate=learning_rate,
            seed=seed,
            compile_mode=proxy_compile_mode,
        )
    elif prune_recovery:
        if checkpoint_run_name == "airbench94_muon_three_quarter_checkpoint_20260527":
            checkpoint_run_name = AirBenchPruneRecoveryConfig.checkpoint_run_name
        summary = run_airbench94_muon_prune_recovery.remote(
            run_name=run_name,
            checkpoint_run_name=checkpoint_run_name,
            chosen_layer_name=chosen_layer_name,
            prune_layer_name=prune_layer_name,
            prune_fraction=prune_fraction,
            prune_seed=prune_seed,
            recovery_loss_name=recovery_loss_name,
            recovery_steps=recovery_steps,
            calibration_sample_count=calibration_sample_count,
            learning_rate=learning_rate,
            seed=seed,
            compile_mode=proxy_compile_mode,
        )
    elif proxy_batch:
        summary = run_airbench94_muon_proxy_batch.remote(
            run_name=run_name,
            proxy_loss_names=proxy_loss_names,
            checkpoint_run_name=checkpoint_run_name,
            chosen_layer_name=chosen_layer_name,
            proxy_steps=proxy_steps,
            learning_rate=learning_rate,
            seed=seed,
            compile_mode=proxy_compile_mode,
        )
    elif proxy_optimize:
        summary = run_airbench94_muon_proxy_optimize.remote(
            run_name=run_name,
            checkpoint_run_name=checkpoint_run_name,
            chosen_layer_name=chosen_layer_name,
            proxy_loss_name=proxy_loss_name,
            proxy_steps=proxy_steps,
            learning_rate=learning_rate,
            seed=seed,
            compile_mode=proxy_compile_mode,
        )
    elif partial_checkpoint:
        summary = run_airbench94_muon_partial_checkpoint.remote(
            run_name=run_name,
            checkpoint_fraction=checkpoint_fraction,
            compile_mode=compile_mode,
            seed=seed,
        )
    elif plateau_checkpoint:
        summary = run_airbench94_muon_plateau_checkpoint.remote(
            run_name=run_name,
            train_steps=train_steps,
            compile_mode=compile_mode,
            seed=seed,
        )
    else:
        summary = run_airbench94_muon.remote(
            run_name=run_name,
            n_runs=n_runs,
            warmup=warmup,
            compile_mode=compile_mode,
            seed=seed,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
