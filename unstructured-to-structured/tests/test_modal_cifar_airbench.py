from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

import modal_cifar_airbench as airbench_modal


def test_airbench_source_is_pinned_to_current_master_head():
    assert airbench_modal.AIRBENCH_REPO_URL == "https://github.com/KellerJordan/cifar10-airbench.git"
    assert airbench_modal.AIRBENCH_REF == "4c1b6d1e3889b037efadcfd5c0ea65b246592362"
    assert len(airbench_modal.AIRBENCH_REF) == 40


def test_default_config_is_fast_smoke_on_real_airbench94_muon():
    config = airbench_modal.AirBenchRunConfig(run_name="smoke")

    assert config.script_name == "airbench94_muon.py"
    assert config.n_runs == 1
    assert config.warmup is True
    assert config.compile_mode == "max-autotune"


def test_partial_checkpoint_defaults_to_three_quarters_of_airbench94():
    config = airbench_modal.AirBenchPartialCheckpointConfig(run_name="partial")

    assert config.script_name == "airbench94_muon.py"
    assert config.total_epochs == 8.0
    assert config.checkpoint_fraction == 0.75
    assert config.chosen_layer_name == "layers.2"


def test_proxy_config_defaults_to_layers2_prefix_scaffold():
    config = airbench_modal.AirBenchProxyOptimizeConfig(run_name="proxy")

    assert config.checkpoint_run_name == "airbench94_muon_three_quarter_checkpoint_20260527"
    assert config.chosen_layer_name == "layers.2"
    assert config.proxy_loss_name == "activation_l2"
    assert config.proxy_steps == 10
    assert config.learning_rate == 1e-4
    assert config.compile_mode == ""


def test_plateau_checkpoint_defaults_to_500_steps_and_layers2():
    config = airbench_modal.AirBenchPlateauCheckpointConfig(run_name="plateau")

    assert config.script_name == "airbench94_muon.py"
    assert config.train_steps == 500
    assert config.chosen_layer_name == "layers.2"
    assert config.batch_size == 2000


def test_prune_recovery_defaults_to_fixed_layer1_mask_and_layers2_targets():
    config = airbench_modal.AirBenchPruneRecoveryConfig(run_name="recover")

    assert config.checkpoint_run_name == "airbench94_muon_plateau_500step_20260528"
    assert config.prune_layer_name == "layers.1"
    assert config.chosen_layer_name == "layers.2"
    assert config.prune_fraction == 0.10
    assert config.prune_seed == 12345
    assert config.recovery_loss_name == "activation_mse"
    assert config.recovery_steps == 10
    assert config.calibration_sample_count == 2000


def test_compute_checkpoint_steps_preserves_full_schedule():
    total_steps, checkpoint_steps = airbench_modal.compute_checkpoint_steps(
        steps_per_epoch=25,
        total_epochs=8.0,
        checkpoint_fraction=0.75,
    )

    assert total_steps == 200
    assert checkpoint_steps == 150


def test_compute_checkpoint_steps_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="checkpoint_fraction"):
        airbench_modal.compute_checkpoint_steps(25, 8.0, 0.0)


def test_prefix_parameter_names_stop_at_chosen_layer():
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.whiten = torch.nn.Conv2d(3, 4, 1)
            self.layers = torch.nn.ModuleList(torch.nn.Conv2d(4, 4, 1) for _ in range(4))
            self.head = torch.nn.Linear(4, 10)

    names = airbench_modal.prefix_parameter_names(TinyModel(), "layers.2")

    assert "whiten.bias" in names
    assert "layers.0.weight" in names
    assert "layers.1.weight" in names
    assert "layers.2.weight" in names
    assert "layers.3.weight" not in names
    assert "head.weight" not in names


def test_layers3_target_can_prune_only_layer1_but_optimize_prefix_through_layer3():
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.whiten = torch.nn.Conv2d(3, 4, 1)
            self.layers = torch.nn.ModuleList(torch.nn.Conv2d(4, 4, 1, bias=False) for _ in range(5))
            self.head = torch.nn.Linear(4, 10)

    model = TinyModel()
    prefix_names = airbench_modal.prefix_parameter_names(model, "layers.3")
    prune_names = airbench_modal.layer_weight_parameter_names(model, "layers.1")

    assert "layers.1.weight" in prune_names
    assert "layers.2.weight" not in prune_names
    assert "layers.3.weight" not in prune_names
    assert "layers.1.weight" in prefix_names
    assert "layers.2.weight" in prefix_names
    assert "layers.3.weight" in prefix_names
    assert "layers.4.weight" not in prefix_names
    assert "head.weight" not in prefix_names


def test_layer_weight_parameter_names_selects_only_weights_in_requested_layer():
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.whiten = torch.nn.Conv2d(3, 4, 1)
            self.layers = torch.nn.ModuleList(torch.nn.Conv2d(4, 4, 1) for _ in range(4))
            self.layers[1].bias = torch.nn.Parameter(torch.zeros(4))
            self.head = torch.nn.Linear(4, 10)

    names = airbench_modal.layer_weight_parameter_names(TinyModel(), "layers.1")

    assert names == {"layers.1.weight"}


def test_random_prune_layer_weights_is_fixed_and_exact_global_fraction():
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(torch.nn.Linear(10, 10, bias=False) for _ in range(3))
            with torch.no_grad():
                for idx, param in enumerate(self.parameters(), start=1):
                    param.fill_(idx)

    first = TinyModel()
    second = TinyModel()

    first_masks, first_stats = airbench_modal.random_prune_layer_weights_(
        first,
        "layers.1",
        prune_fraction=0.10,
        seed=7,
    )
    second_masks, second_stats = airbench_modal.random_prune_layer_weights_(
        second,
        "layers.1",
        prune_fraction=0.10,
        seed=7,
    )

    assert first_stats["total_weights"] == 100
    assert first_stats["pruned_weights"] == 10
    assert torch.equal(first_masks["layers.1.weight"], second_masks["layers.1.weight"])
    assert torch.count_nonzero(first.layers[1].weight == 0).item() == 10
    assert torch.equal(first.layers[0].weight, torch.full_like(first.layers[0].weight, 1.0))
    assert torch.equal(first.layers[2].weight, torch.full_like(first.layers[2].weight, 3.0))


def test_prune_masks_preserve_zeroed_weights_after_update():
    layer = torch.nn.Linear(10, 10, bias=False)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Identity(), layer])
    with torch.no_grad():
        layer.weight.fill_(1.0)
    masks, _stats = airbench_modal.random_prune_layer_weights_(model, "layers.1", 0.10, seed=3)

    layer.weight.grad = torch.ones_like(layer.weight)
    with torch.no_grad():
        layer.weight.add_(layer.weight.grad)
    airbench_modal.apply_prune_masks_to_params_(model, masks)

    assert torch.count_nonzero(layer.weight == 0).item() == 10
    assert torch.all(layer.weight[masks["layers.1.weight"]] == 2.0)


def test_prune_masks_force_pruned_nan_weights_to_zero():
    layer = torch.nn.Linear(10, 10, bias=False)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Identity(), layer])
    masks, _stats = airbench_modal.random_prune_layer_weights_(model, "layers.1", 0.10, seed=3)

    with torch.no_grad():
        layer.weight[~masks["layers.1.weight"]] = float("nan")
    airbench_modal.apply_prune_masks_to_params_(model, masks)

    assert torch.count_nonzero(layer.weight[~masks["layers.1.weight"]] == 0).item() == 10
    assert torch.isfinite(layer.weight).all()


def test_forward_to_chosen_layer_does_not_call_suffix_or_head():
    class RaisingLayer(torch.nn.Module):
        def forward(self, inputs):
            raise AssertionError("suffix layer should not run")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.whiten = torch.nn.Conv2d(3, 4, 1)
            self.layers = torch.nn.ModuleList(
                [torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity(), RaisingLayer()]
            )
            self.head = RaisingLayer()

    model = TinyModel()
    outputs = airbench_modal.forward_to_chosen_layer(model, torch.randn(2, 3, 4, 4), "layers.2")

    assert outputs.shape == (2, 4, 4, 4)


def test_forward_layer_activation_sites_collects_requested_inputs_and_outputs():
    class AddLayer(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value

        def forward(self, inputs):
            return inputs + self.value

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.whiten = torch.nn.Conv2d(1, 1, 1)
            self.layers = torch.nn.ModuleList(AddLayer(float(idx + 1)) for idx in range(4))

    model = TinyModel()
    with torch.no_grad():
        model.whiten.weight.fill_(1.0)
        model.whiten.bias.zero_()

    sites = airbench_modal.forward_layer_activation_sites(
        model,
        torch.zeros(1, 1, 1, 1),
        {
            "layers.1.input",
            "layers.1.output",
            "layers.3.input",
            "layers.3.output",
        },
        whiten_bias_grad=False,
    )

    assert torch.equal(sites["layers.1.input"], torch.tensor([[[[1.0]]]]))
    assert torch.equal(sites["layers.1.output"], torch.tensor([[[[3.0]]]]))
    assert torch.equal(sites["layers.3.input"], torch.tensor([[[[6.0]]]]))
    assert torch.equal(sites["layers.3.output"], torch.tensor([[[[10.0]]]]))


def test_linear_cka_similarity_is_scale_invariant_for_same_features():
    features = torch.tensor([[1.0, 2.0], [2.0, 1.0], [4.0, 0.0]])

    score = airbench_modal.linear_cka_similarity(features, 3 * features)

    assert torch.isclose(score, torch.tensor(1.0))


def test_fisher_class_separability_increases_for_separated_classes():
    separated = torch.tensor([[-2.0, 0.0], [-2.1, 0.1], [2.0, 0.0], [2.1, -0.1]])
    mixed = torch.tensor([[-0.1, 0.0], [0.1, 0.0], [0.0, 0.1], [0.0, -0.1]])
    labels = torch.tensor([0, 0, 1, 1])

    assert airbench_modal.fisher_class_separability(separated, labels) > airbench_modal.fisher_class_separability(mixed, labels)


def test_ridge_probe_accuracy_recovers_linearly_separable_labels():
    features = torch.tensor(
        [
            [-2.0, 0.0],
            [-1.8, 0.1],
            [-2.2, -0.1],
            [2.0, 0.0],
            [1.8, -0.1],
            [2.2, 0.1],
            [-1.9, 0.2],
            [1.9, -0.2],
        ]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 0, 1])

    accuracy = airbench_modal.ridge_probe_accuracy(features, labels, train_fraction=0.75)

    assert accuracy == 1.0


def test_mirage_activation_audit_identical_variant_has_chance_origin_accuracy():
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    activations = torch.tensor(
        [
            [[[-2.0]], [[0.0]]],
            [[[-2.1]], [[0.1]]],
            [[[2.0]], [[0.0]]],
            [[[2.1]], [[-0.1]]],
            [[[-1.9]], [[0.2]]],
            [[[1.9]], [[-0.2]]],
            [[[-2.2]], [[-0.1]]],
            [[[2.2]], [[0.1]]],
        ]
    )

    audit = airbench_modal.mirage_activation_audit(activations, activations.clone(), labels)

    assert audit["cka_to_original"] == pytest.approx(1.0)
    assert audit["origin_probe_accuracy"] == pytest.approx(0.5)
    assert audit["class_probe_accuracy"] == pytest.approx(1.0)
    assert audit["class_separability"] > 0


def test_activation_l2_proxy_loss_uses_activation_only():
    activations = torch.tensor([1.0, -2.0, 3.0])

    assert torch.equal(airbench_modal.proxy_loss_activation_l2(activations), torch.tensor(14.0 / 3.0))


def test_activation_l1_proxy_loss_uses_activation_only():
    activations = torch.tensor([1.0, -2.0, 3.0])

    assert torch.equal(airbench_modal.proxy_loss_activation_l1(activations), torch.tensor(2.0))


def test_activation_linf_proxy_loss_uses_mean_per_sample_max():
    activations = torch.tensor([[[[1.0, -4.0]]], [[[2.0, -3.0]]]])

    assert torch.equal(airbench_modal.proxy_loss_activation_linf(activations), torch.tensor(3.5))


def test_cosine_proxy_loss_maximizes_two_view_similarity():
    first = torch.tensor([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]])
    second = torch.tensor([[[[1.0]], [[0.0]]], [[[1.0]], [[0.0]]]])

    loss = airbench_modal.proxy_loss_cosine_sim(first, second_activations=second)

    assert torch.isclose(loss, torch.tensor(-0.5))


def test_same_class_contrastive_requires_labels_and_is_finite():
    activations = torch.tensor([[[[1.0]], [[0.0]]], [[[0.9]], [[0.1]]], [[[0.0]], [[1.0]]]])
    labels = torch.tensor([0, 0, 1])

    loss = airbench_modal.proxy_loss_same_class_contrastive(activations, labels=labels)

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_same_class_contrastive_rejects_missing_labels():
    activations = torch.randn(4, 3, 2, 2)

    with pytest.raises(ValueError, match="labels"):
        airbench_modal.proxy_loss_same_class_contrastive(activations)


def test_proxy_loss_registry_contains_requested_losses():
    assert {
        "activation_l2",
        "activation_l1",
        "activation_linf",
        "cosine_sim",
        "same_class_contrastive",
    }.issubset(airbench_modal.PROXY_LOSSES)


def test_recovery_loss_registry_contains_only_allowed_input_signatures():
    allowed = {"original_weights", "original_activations", "current_activations"}

    requested_losses = {
        "activation_l1",
        "activation_l2",
        "activation_squared_l2",
        "activation_linf",
        "group_l21",
        "group_l12",
        "fractional_l05",
        "canberra",
        "clark",
        "bray_curtis",
        "soergel",
        "kulczynski",
        "lorentzian",
        "gower",
        "mape",
        "max_relative",
        "huber",
        "pseudo_huber",
        "tukey_biweight",
        "cauchy_loss",
        "fair_loss",
        "channelwise_fair_loss",
        "channelwise_fair_layer1_input",
        "channelwise_fair_layer1_output",
        "channelwise_fair_layer2_input",
        "channelwise_fair_layer2_output",
        "channelwise_fair_layer3_input",
        "channelwise_fair_layer3_output",
        "raw_l2_layer1_input",
        "raw_l2_layer1_output",
        "raw_l2_layer2_input",
        "raw_l2_layer2_output",
        "welsch_loss",
        "cosine_distance",
        "angular_distance",
        "chordal_spherical",
        "correlation_distance",
        "sign_invariant_angular",
        "projector_frobenius",
        "best_rescaled_l2",
        "best_shifted_l2",
        "best_affine_l2",
        "mahalanobis",
        "weighted_l2",
        "gram_quadratic",
        "relative_l2",
        "symmetric_relative_l2",
        "relative_l1",
        "log_ratio_positive",
        "aitchison",
        "alr_distance",
        "hilbert_projective",
        "thompson_metric",
        "norm_ratio",
        "cosine_preservation",
        "parallel_l2",
        "orthogonal_l2",
        "projection_residual",
        "bregman_log_cosh",
        "total_variation",
        "kl_divergence",
        "reverse_kl",
        "jeffreys",
        "js_divergence",
        "hellinger",
        "bhattacharyya",
        "chernoff_alpha_half",
        "matusita",
        "itakura_saito",
        "chi_square_pearson",
        "chi_square_neyman",
        "symmetric_chi_square",
        "squared_chord",
        "triangular_discrimination",
        "topsoe",
        "renyi_alpha_half",
        "alpha_divergence",
        "cressie_read",
        "tsallis_alpha_half",
        "generalized_i_divergence",
        "beta_divergence_half",
        "wasserstein_1d",
        "sinkhorn_ot",
        "sliced_wasserstein",
        "max_sliced_wasserstein",
        "gromov_wasserstein_surrogate",
        "rbf_kernel_point",
        "polynomial_kernel_point",
        "laplacian_kernel_point",
        "mmd_rbf",
        "energy_distance",
        "cramer_distance",
        "kolmogorov_smirnov",
        "rsa_distance",
        "linear_cka_distance",
        "svcca_distance",
        "pwcca_distance",
        "orthogonal_procrustes",
        "linear_regression_distance",
        "ridge_regression_distance",
        "mutual_nn_overlap",
        "trustworthiness_surrogate",
        "shape_distance",
        "normalized_bures",
        "covariance_frobenius",
        "covariance_spectral",
        "covariance_nuclear",
        "covariance_log_euclidean",
        "covariance_affine_invariant",
        "covariance_bures",
        "covariance_stein",
        "covariance_condition",
        "grassmann_geodesic",
        "grassmann_chordal",
        "projection_frobenius_subspace",
        "spectral_projection_subspace",
        "fubini_study",
        "binet_cauchy",
        "sign_hamming",
        "soft_jaccard_topk",
        "dice_topk",
        "overlap_topk",
        "overlap_coefficient_topk",
        "tversky_topk",
        "simple_matching",
        "sokal_sneath",
        "rogers_tanimoto",
        "russell_rao",
        "yule_distance",
        "tanimoto",
        "ochiai",
        "rank_biased_overlap",
        "ndcg_topk",
        "rank_correlation",
        "spearman_distance",
        "kendall_distance",
        "distance_correlation",
        "hsic_rbf",
        "logdet_covariance",
        "histogram_intersection",
        "ruzicka",
        "motyka",
        "czekanowski",
        "wave_hedges",
        "inner_product_dissimilarity",
        "fidelity_distance",
        "fisher_rao_categorical",
        "poincare_distance",
        "sequence_l2",
        "derivative_sequence_l2",
        "weighted_sequence_l2",
        "soft_dtw",
        "shape_based_sequence",
        "frechet_surrogate",
        "hausdorff_surrogate",
        "chamfer_distance",
        "layernorm_l2",
        "cosine_relative",
        "orthogonal_relative",
        "covariance_whitened_cosine",
        "rare_feature_weighted_l2",
        "learned_pca8_l2",
        "learned_pca32_l2",
        "learned_pca8_whitened_l2",
        "learned_pca32_whitened_l2",
        "learned_pca32_residual_l2",
        "learned_shrinkage_mahalanobis",
        "learned_variance_attention_l2",
        "learned_kmeans4_softdist",
        "learned_kmeans16_softdist",
        "learned_kmeans16_logit_l2",
        "learned_ridge_pseudolabel_logits",
        "learned_rbf_nystrom_l2",
    }
    assert requested_losses.issubset(airbench_modal.PRUNE_RECOVERY_LOSSES)
    for loss_fn in airbench_modal.PRUNE_RECOVERY_LOSSES.values():
        signature = inspect.signature(loss_fn)
        assert set(signature.parameters) <= allowed


def test_activation_mse_recovery_loss_matches_current_to_original_activations():
    original = torch.tensor([1.0, 2.0, 3.0])
    current = torch.tensor([2.0, 0.0, 3.0])

    loss = airbench_modal.recovery_loss_activation_mse(
        original_weights={},
        original_activations=original,
        current_activations=current,
    )

    assert torch.equal(loss, torch.tensor(5.0 / 3.0))


def test_channelwise_fair_loss_uses_separate_channel_scales():
    original = torch.zeros(2, 2, 1, 2)
    current = torch.tensor(
        [
            [[[1.0, 1.0]], [[10.0, 10.0]]],
            [[[1.0, 1.0]], [[10.0, 10.0]]],
        ],
        requires_grad=True,
    )

    loss = airbench_modal.recovery_loss_channelwise_fair_loss(
        original_weights={},
        original_activations=original,
        current_activations=current,
    )
    global_loss = airbench_modal.recovery_loss_fair_loss(
        original_weights={},
        original_activations=original,
        current_activations=current,
    )
    expected = 2 * ((1 - torch.log(torch.tensor(2.0))) + 100 * (1 - torch.log(torch.tensor(2.0))))

    assert torch.isclose(loss, expected)
    assert not torch.isclose(loss, global_loss)


@pytest.mark.parametrize(
    "loss_name,site_key",
    [
        ("channelwise_fair_layer1_input", "layers.1.input"),
        ("channelwise_fair_layer1_output", "layers.1.output"),
        ("channelwise_fair_layer2_input", "layers.2.input"),
        ("channelwise_fair_layer2_output", "layers.2.output"),
        ("channelwise_fair_layer3_input", "layers.3.input"),
        ("channelwise_fair_layer3_output", "layers.3.output"),
    ],
)
def test_site_channelwise_fair_losses_use_requested_activation_site(loss_name, site_key):
    original = {
        "layers.1.input": torch.zeros(2, 2, 1, 1),
        "layers.1.output": torch.zeros(2, 2, 1, 1),
        "layers.2.input": torch.zeros(2, 2, 1, 1),
        "layers.2.output": torch.zeros(2, 2, 1, 1),
        "layers.3.input": torch.zeros(2, 2, 1, 1),
        "layers.3.output": torch.zeros(2, 2, 1, 1),
    }
    current = {key: value.clone() for key, value in original.items()}
    current[site_key] = torch.tensor([[[[1.0]], [[10.0]]], [[[1.0]], [[10.0]]]], requires_grad=True)

    loss = airbench_modal.PRUNE_RECOVERY_LOSSES[loss_name](
        original_weights={},
        original_activations=original,
        current_activations=current,
    )
    loss.backward()

    expected = (1 - torch.log(torch.tensor(2.0))) + 100 * (1 - torch.log(torch.tensor(2.0)))
    assert torch.isclose(loss, expected)
    assert current[site_key].grad is not None


@pytest.mark.parametrize(
    "loss_name,site_key",
    [
        ("raw_l2_layer1_input", "layers.1.input"),
        ("raw_l2_layer1_output", "layers.1.output"),
        ("raw_l2_layer2_input", "layers.2.input"),
        ("raw_l2_layer2_output", "layers.2.output"),
    ],
)
def test_site_raw_l2_losses_use_requested_activation_site(loss_name, site_key):
    original = {
        "layers.1.input": torch.zeros(2, 2, 1, 1),
        "layers.1.output": torch.zeros(2, 2, 1, 1),
        "layers.2.input": torch.zeros(2, 2, 1, 1),
        "layers.2.output": torch.zeros(2, 2, 1, 1),
    }
    current = {key: value.clone() for key, value in original.items()}
    current[site_key] = torch.tensor([[[[3.0]], [[4.0]]], [[[5.0]], [[12.0]]]], requires_grad=True)

    loss = airbench_modal.PRUNE_RECOVERY_LOSSES[loss_name](
        original_weights={},
        original_activations=original,
        current_activations=current,
    )
    loss.backward()

    assert torch.isclose(loss, torch.tensor(9.0))
    assert current[site_key].grad is not None


@pytest.mark.parametrize(
    "loss_name",
    [
        "activation_l1",
        "activation_l2",
        "activation_squared_l2",
        "activation_linf",
        "group_l21",
        "group_l12",
        "fractional_l05",
        "canberra",
        "clark",
        "bray_curtis",
        "soergel",
        "kulczynski",
        "lorentzian",
        "gower",
        "mape",
        "max_relative",
        "huber",
        "pseudo_huber",
        "tukey_biweight",
        "cauchy_loss",
        "fair_loss",
        "channelwise_fair_loss",
        "welsch_loss",
        "cosine_distance",
        "angular_distance",
        "chordal_spherical",
        "correlation_distance",
        "sign_invariant_angular",
        "projector_frobenius",
        "best_rescaled_l2",
        "best_shifted_l2",
        "best_affine_l2",
        "mahalanobis",
        "weighted_l2",
        "gram_quadratic",
        "relative_l2",
        "symmetric_relative_l2",
        "relative_l1",
        "log_ratio_positive",
        "aitchison",
        "alr_distance",
        "hilbert_projective",
        "thompson_metric",
        "norm_ratio",
        "cosine_preservation",
        "parallel_l2",
        "orthogonal_l2",
        "projection_residual",
        "bregman_log_cosh",
        "total_variation",
        "kl_divergence",
        "reverse_kl",
        "jeffreys",
        "js_divergence",
        "hellinger",
        "bhattacharyya",
        "chernoff_alpha_half",
        "matusita",
        "itakura_saito",
        "chi_square_pearson",
        "chi_square_neyman",
        "symmetric_chi_square",
        "squared_chord",
        "triangular_discrimination",
        "topsoe",
        "renyi_alpha_half",
        "alpha_divergence",
        "cressie_read",
        "tsallis_alpha_half",
        "generalized_i_divergence",
        "beta_divergence_half",
        "wasserstein_1d",
        "sinkhorn_ot",
        "sliced_wasserstein",
        "max_sliced_wasserstein",
        "gromov_wasserstein_surrogate",
        "rbf_kernel_point",
        "polynomial_kernel_point",
        "laplacian_kernel_point",
        "mmd_rbf",
        "energy_distance",
        "cramer_distance",
        "kolmogorov_smirnov",
        "rsa_distance",
        "linear_cka_distance",
        "svcca_distance",
        "pwcca_distance",
        "orthogonal_procrustes",
        "linear_regression_distance",
        "ridge_regression_distance",
        "mutual_nn_overlap",
        "trustworthiness_surrogate",
        "shape_distance",
        "normalized_bures",
        "covariance_frobenius",
        "covariance_spectral",
        "covariance_nuclear",
        "covariance_log_euclidean",
        "covariance_affine_invariant",
        "covariance_bures",
        "covariance_stein",
        "covariance_condition",
        "grassmann_geodesic",
        "grassmann_chordal",
        "projection_frobenius_subspace",
        "spectral_projection_subspace",
        "fubini_study",
        "binet_cauchy",
        "sign_hamming",
        "soft_jaccard_topk",
        "dice_topk",
        "overlap_topk",
        "overlap_coefficient_topk",
        "tversky_topk",
        "simple_matching",
        "sokal_sneath",
        "rogers_tanimoto",
        "russell_rao",
        "yule_distance",
        "tanimoto",
        "ochiai",
        "rank_biased_overlap",
        "ndcg_topk",
        "rank_correlation",
        "spearman_distance",
        "kendall_distance",
        "distance_correlation",
        "hsic_rbf",
        "logdet_covariance",
        "histogram_intersection",
        "ruzicka",
        "motyka",
        "czekanowski",
        "wave_hedges",
        "inner_product_dissimilarity",
        "fidelity_distance",
        "fisher_rao_categorical",
        "poincare_distance",
        "sequence_l2",
        "derivative_sequence_l2",
        "weighted_sequence_l2",
        "soft_dtw",
        "shape_based_sequence",
        "frechet_surrogate",
        "hausdorff_surrogate",
        "chamfer_distance",
        "layernorm_l2",
        "cosine_relative",
        "orthogonal_relative",
        "covariance_whitened_cosine",
        "rare_feature_weighted_l2",
        "learned_pca8_l2",
        "learned_pca32_l2",
        "learned_pca8_whitened_l2",
        "learned_pca32_whitened_l2",
        "learned_pca32_residual_l2",
        "learned_shrinkage_mahalanobis",
        "learned_variance_attention_l2",
        "learned_kmeans4_softdist",
        "learned_kmeans16_softdist",
        "learned_kmeans16_logit_l2",
        "learned_ridge_pseudolabel_logits",
        "learned_rbf_nystrom_l2",
    ],
)
def test_requested_recovery_losses_are_finite_scalars_with_gradients(loss_name):
    original = torch.randn(8, 4, 2, 2)
    current = (original + 0.1 * torch.randn_like(original)).requires_grad_(True)

    loss = airbench_modal.PRUNE_RECOVERY_LOSSES[loss_name](
        original_weights={},
        original_activations=original,
        current_activations=current,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.ndim == 0
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


@pytest.mark.parametrize(
    "loss_name",
    [
        "learned_pca8_l2",
        "learned_pca8_whitened_l2",
        "learned_kmeans4_softdist",
        "learned_ridge_pseudolabel_logits",
        "learned_rbf_nystrom_l2",
    ],
)
def test_learned_recovery_losses_are_zero_or_near_zero_for_identical_activations(loss_name):
    original = torch.randn(10, 6, 2, 2)
    current = original.clone().requires_grad_(True)

    loss = airbench_modal.PRUNE_RECOVERY_LOSSES[loss_name](
        original_weights={},
        original_activations=original,
        current_activations=current,
    )

    assert torch.isfinite(loss)
    assert loss < 1e-5


def test_runtime_dirs_link_airbench_data_and_logs_to_volumes(tmp_path):
    repo_root = tmp_path / "repo"
    cache_root = tmp_path / "cache"
    output_dir = tmp_path / "results" / "run"
    repo_root.mkdir()

    airbench_modal.prepare_runtime_dirs(repo_root, cache_root, output_dir)

    assert (cache_root / "cifar10").is_dir()
    assert (output_dir / "logs").is_dir()
    assert (repo_root / "cifar10").is_symlink()
    assert (repo_root / "logs").is_symlink()
    assert (repo_root / "cifar10").resolve() == (cache_root / "cifar10").resolve()
    assert (repo_root / "logs").resolve() == (output_dir / "logs").resolve()


def test_runtime_dirs_refuse_to_replace_real_paths(tmp_path):
    repo_root = tmp_path / "repo"
    cache_root = tmp_path / "cache"
    output_dir = tmp_path / "results" / "run"
    repo_root.mkdir()
    (repo_root / "cifar10").mkdir()

    with pytest.raises(FileExistsError):
        airbench_modal.prepare_runtime_dirs(repo_root, cache_root, output_dir)


def test_modal_definition_uses_h100_and_cached_airbench_image():
    source = Path(inspect.getsourcefile(airbench_modal) or "").read_text()

    assert 'gpu="H100"' in source
    assert 'modal.Volume.from_name("cifar-airbench-results"' in source
    assert 'modal.Volume.from_name("cifar-airbench-cache"' in source
    assert "nvidia/cuda:12.8.1-devel-ubuntu22.04" in source
    assert f"git checkout {airbench_modal.AIRBENCH_REF}" in source
    assert "pip install --no-cache-dir -e" in source


def test_modal_definition_exposes_partial_checkpoint_runner():
    source = Path(inspect.getsourcefile(airbench_modal) or "").read_text()

    assert "def run_airbench94_muon_partial_checkpoint(" in source
    assert '"checkpoint.pt"' in source
    assert '"chosen_layer_name"' in source


def test_modal_definition_exposes_proxy_optimize_runner():
    source = Path(inspect.getsourcefile(airbench_modal) or "").read_text()

    assert "def run_airbench94_muon_proxy_optimize(" in source
    assert '"proxy_loss_name"' in source
    assert "proxy_compile_mode" in source
    assert "forward_to_chosen_layer" in source


def test_modal_definition_exposes_proxy_batch_runner():
    source = Path(inspect.getsourcefile(airbench_modal) or "").read_text()

    assert "def run_airbench94_muon_proxy_batch(" in source
    assert "proxy_loss_names" in source


def test_modal_definition_exposes_plateau_and_prune_recovery_runners():
    source = Path(inspect.getsourcefile(airbench_modal) or "").read_text()

    assert "def run_airbench94_muon_plateau_checkpoint(" in source
    assert "def run_airbench94_muon_prune_recovery(" in source
    assert "def run_airbench94_muon_prune_recovery_batch(" in source
    assert "plateau_checkpoint" in source
    assert "prune_recovery" in source
    assert "prune_recovery_batch" in source
    assert "apply_prune_masks_to_params_" in source
