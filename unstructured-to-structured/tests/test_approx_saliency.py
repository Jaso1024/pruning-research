from __future__ import annotations

import torch

import saliency.approx as approx
from saliency.prune_eval import PerplexityStats
from saliency.approx import (
    AngularActivationAccumulator,
    ApproxSaliencyConfig,
    DfaActivationCollector,
    FeatureCosineWandaAccumulator,
    InputActivationStatsAccumulator,
    IterativeApproxPruneConfig,
    LocalForwardWandaAccumulator,
    apply_incremental_per_matrix_pruning_,
    apply_incremental_nm_pruning_,
    apply_incremental_nm_pruning_to_parameter_,
    apply_masked_gradient_step_,
    accumulate_vjp_gradient_scores_,
    accumulate_vjp_parameter_scores_,
    attention_qkv_superset_output_gains,
    activation_forward_diff_scores,
    activation_local_jacobian_square,
    angular_saliency_scores,
    build_local_subgraph_endpoint_groups,
    input_activation_stat_scores,
    cross_entropy_residual,
    feature_wanda_cosine_scores,
    hash_project_residual,
    graph_propagated_scores,
    linear_superset_wanda_scores,
    layernorm_input_downstream_colnorm_squares,
    layernorm_forward_diff_colnorm_squares,
    layernorm_input_jacobian_colnorm_squares,
    layernorm_input_projected_gain_from_vectors,
    normalize_pruning_structure,
    qwen_attention_qkv_superset_output_gains,
    qwen_mlp_superset_output_gains,
    reapply_pruned_masks_,
    relative_importance_scores,
    rmsnorm_input_downstream_colnorm_squares,
    RowConditionedWandaAccumulator,
    run_approx_saliency_experiment,
    run_nm_global_pass_matrix_attribution_experiment,
    sequential_wanda_matrix_parameter_groups,
    sequential_wanda_parameter_groups,
    transform_superset_gain,
    weight_magnitude_scores,
    WandaActivationAccumulator,
)


def test_weight_magnitude_scores_keep_trainable_matrices_only():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.LayerNorm(2))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))
        model[0].bias.copy_(torch.tensor([5.0, -6.0]))
        model[1].weight.copy_(torch.tensor([7.0, -8.0]))

    scores = weight_magnitude_scores(model.named_parameters())

    assert set(scores) == {"0.weight"}
    assert torch.equal(scores["0.weight"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_wanda_scores_scale_weight_magnitude_by_input_rms_and_fallback_to_magnitude():
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.extra = torch.nn.Parameter(torch.tensor([[5.0, -6.0]]))
            self.linear = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.linear(inputs)

    model = Tiny()
    inputs = torch.tensor([[[3.0, 4.0], [0.0, 8.0]]])

    with WandaActivationAccumulator(model) as accumulator:
        model(inputs)
    scores = accumulator.finalize(model.named_parameters())

    input_rms = torch.sqrt(torch.tensor([(3.0**2 + 0.0**2) / 2.0, (4.0**2 + 8.0**2) / 2.0]))
    expected_linear = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) * input_rms.unsqueeze(0)

    assert torch.allclose(scores["linear.weight"], expected_linear)
    assert torch.equal(scores["extra"], torch.tensor([[5.0, 6.0]]))


def test_wanda_scores_ignore_padding_positions_with_attention_mask():
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.linear(inputs)

    model = Tiny()
    inputs = torch.tensor([[[3.0, 4.0], [100.0, 100.0], [0.0, 8.0]]])
    attention_mask = torch.tensor([[1, 0, 1]])

    with WandaActivationAccumulator(model) as accumulator:
        accumulator.set_attention_mask(attention_mask)
        model(inputs)
    scores = accumulator.finalize(model.named_parameters())
    hessians, tokens = accumulator.hessian_diagonals()

    input_rms = torch.sqrt(torch.tensor([(3.0**2 + 0.0**2) / 2.0, (4.0**2 + 8.0**2) / 2.0]))
    expected_linear = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) * input_rms.unsqueeze(0)

    assert torch.allclose(scores["linear.weight"], expected_linear)
    assert torch.equal(hessians["linear"], torch.tensor([9.0, 80.0], dtype=torch.float64))
    assert tokens["linear"] == 2


def test_original_wanda_alias_disables_attention_mask(monkeypatch, tmp_path):
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[1.0]]))

    seen: list[bool] = []

    def fake_prepare_model(*args, **kwargs):
        return Tiny()

    def fake_prepare_tokenizer(*args, **kwargs):
        return object()

    def fake_load_records(*args, **kwargs):
        return [{"question": "q", "answer": "a"}]

    def fake_wanda_scores(*args, **kwargs):
        seen.append(bool(kwargs["use_attention_mask"]))
        return {"weight": torch.tensor([[1.0]])}

    monkeypatch.setattr(approx, "_prepare_model", fake_prepare_model)
    monkeypatch.setattr(approx, "_prepare_tokenizer", fake_prepare_tokenizer)
    monkeypatch.setattr(approx, "_load_records", fake_load_records)
    monkeypatch.setattr(approx, "_wanda_scores", fake_wanda_scores)

    summary = run_approx_saliency_experiment(
        ApproxSaliencyConfig(output_dir=tmp_path, model_name="tiny", method="original_wanda")
    )

    assert seen == [False]
    assert "including padding" in str(summary["metadata"]["saliency_method"])


def test_sequential_wanda_parameter_groups_order_embeddings_layers_and_head():
    class TinyNeoX(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gpt_neox = torch.nn.Module()
            self.gpt_neox.embed_in = torch.nn.Embedding(4, 3)
            self.gpt_neox.layers = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict({"dense": torch.nn.Linear(3, 3, bias=False)}),
                    torch.nn.ModuleDict(
                        {
                            "a": torch.nn.Linear(3, 3, bias=False),
                            "b": torch.nn.Linear(3, 3, bias=False),
                        }
                    ),
                ]
            )
            self.embed_out = torch.nn.Linear(3, 4, bias=False)

    groups = sequential_wanda_parameter_groups(TinyNeoX())

    assert groups == [
        ("gpt_neox.embed_in.weight", ["gpt_neox.embed_in.weight"]),
        ("gpt_neox.layers.0", ["gpt_neox.layers.0.dense.weight"]),
        ("gpt_neox.layers.1", ["gpt_neox.layers.1.a.weight", "gpt_neox.layers.1.b.weight"]),
        ("embed_out.weight", ["embed_out.weight"]),
    ]


def test_sequential_wanda_matrix_parameter_groups_order_every_matrix():
    class TinyNeoX(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gpt_neox = torch.nn.Module()
            self.gpt_neox.embed_in = torch.nn.Embedding(4, 3)
            self.gpt_neox.layers = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict({"dense": torch.nn.Linear(3, 3, bias=False)}),
                    torch.nn.ModuleDict(
                        {
                            "a": torch.nn.Linear(3, 3, bias=False),
                            "b": torch.nn.Linear(3, 3, bias=False),
                        }
                    ),
                ]
            )
            self.embed_out = torch.nn.Linear(3, 4, bias=False)

    groups = sequential_wanda_matrix_parameter_groups(TinyNeoX())

    assert groups == [
        ("gpt_neox.embed_in.weight", ["gpt_neox.embed_in.weight"]),
        ("gpt_neox.layers.0.dense.weight", ["gpt_neox.layers.0.dense.weight"]),
        ("gpt_neox.layers.1.a.weight", ["gpt_neox.layers.1.a.weight"]),
        ("gpt_neox.layers.1.b.weight", ["gpt_neox.layers.1.b.weight"]),
        ("embed_out.weight", ["embed_out.weight"]),
    ]


def test_apply_sequential_wanda_pruning_recomputes_each_group(monkeypatch):
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Linear(2, 1, bias=False)
            self.second = torch.nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                self.first.weight.copy_(torch.tensor([[1.0, 2.0]]))
                self.second.weight.copy_(torch.tensor([[3.0, 4.0]]))

    calls: list[tuple[str, ...]] = []

    def fake_wanda_scores(*args, **kwargs):
        target_names = tuple(kwargs["target_names"])
        calls.append(target_names)
        if target_names == ("first.weight",):
            return {"first.weight": torch.tensor([[0.1, 0.9]])}, {}, {}
        if target_names == ("second.weight",):
            return {"second.weight": torch.tensor([[0.8, 0.2]])}, {}, {}
        raise AssertionError(target_names)

    monkeypatch.setattr(approx, "_wanda_scores_and_hessian_diagonal", fake_wanda_scores)

    summary = approx.apply_sequential_wanda_pruning_(
        Tiny(),
        tokenizer=object(),
        records=[{}],
        config=ApproxSaliencyConfig(output_dir="unused", model_name="tiny", max_examples=1),
        device=torch.device("cpu"),
        use_attention_mask=True,
        pruning_scope="per_output_row",
        fraction=0.5,
    )

    assert calls == [("first.weight",), ("second.weight",)]
    assert summary["weights_zeroed"] == 2
    assert summary["pruning_schedule"] == "sequential"


def test_apply_matrix_sequential_wanda_pruning_recomputes_each_matrix_inside_layer(monkeypatch):
    class TinyNeoX(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gpt_neox = torch.nn.Module()
            self.gpt_neox.layers = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            "first": torch.nn.Linear(2, 1, bias=False),
                            "second": torch.nn.Linear(2, 1, bias=False),
                        }
                    )
                ]
            )

    calls: list[tuple[str, ...]] = []

    def fake_wanda_scores(*args, **kwargs):
        target_names = tuple(kwargs["target_names"])
        calls.append(target_names)
        return {target_names[0]: torch.tensor([[0.1, 0.9]])}, {}, {}

    monkeypatch.setattr(approx, "_wanda_scores_and_hessian_diagonal", fake_wanda_scores)

    summary = approx.apply_matrix_sequential_wanda_pruning_(
        TinyNeoX(),
        tokenizer=object(),
        records=[{}],
        config=ApproxSaliencyConfig(output_dir="unused", model_name="tiny", max_examples=1),
        device=torch.device("cpu"),
        use_attention_mask=True,
        pruning_scope="per_output_row",
        fraction=0.5,
    )

    assert calls == [
        ("gpt_neox.layers.0.first.weight",),
        ("gpt_neox.layers.0.second.weight",),
    ]
    assert summary["weights_zeroed"] == 2
    assert summary["pruning_schedule"] == "matrix_sequential"


def test_apply_matrix_sequential_wanda_can_route_to_superset_scorer(monkeypatch):
    class TinyNeoX(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gpt_neox = torch.nn.Module()
            self.gpt_neox.layers = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            "first": torch.nn.Linear(2, 1, bias=False),
                            "second": torch.nn.Linear(2, 1, bias=False),
                        }
                    )
                ]
            )

    calls: list[tuple[bool, tuple[str, ...]]] = []

    def fake_superset_scores(*args, **kwargs):
        target_names = tuple(kwargs["target_names"])
        calls.append((bool(kwargs["use_attention_mask"]), target_names))
        return {target_names[0]: torch.tensor([[0.1, 0.9]])}

    monkeypatch.setattr(approx, "_superset_wanda_scores", fake_superset_scores)

    summary = approx.apply_matrix_sequential_wanda_pruning_(
        TinyNeoX(),
        tokenizer=object(),
        records=[{}],
        config=ApproxSaliencyConfig(output_dir="unused", model_name="tiny", max_examples=1),
        device=torch.device("cpu"),
        use_attention_mask=False,
        pruning_scope="per_output_row",
        fraction=0.5,
        score_method="superset_wanda",
    )

    assert calls == [
        (False, ("gpt_neox.layers.0.first.weight",)),
        (False, ("gpt_neox.layers.0.second.weight",)),
    ]
    assert summary["score_method"] == "superset_wanda"
    assert summary["weights_zeroed"] == 2


def test_apply_matrix_sequential_wanda_can_route_to_magnitude_scores(monkeypatch):
    class TinyNeoX(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gpt_neox = torch.nn.Module()
            self.gpt_neox.layers = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            "first": torch.nn.Linear(2, 1, bias=False),
                            "second": torch.nn.Linear(2, 1, bias=False),
                        }
                    )
                ]
            )
            with torch.no_grad():
                self.gpt_neox.layers[0]["first"].weight.copy_(torch.tensor([[0.1, 0.9]]))
                self.gpt_neox.layers[0]["second"].weight.copy_(torch.tensor([[0.8, 0.2]]))

    def fail_wanda(*args, **kwargs):
        raise AssertionError("magnitude should not call WANDA scoring")

    monkeypatch.setattr(approx, "_wanda_scores_and_hessian_diagonal", fail_wanda)

    summary = approx.apply_matrix_sequential_wanda_pruning_(
        TinyNeoX(),
        tokenizer=object(),
        records=[{}],
        config=ApproxSaliencyConfig(output_dir="unused", model_name="tiny", max_examples=1),
        device=torch.device("cpu"),
        use_attention_mask=False,
        pruning_scope="per_output_row",
        fraction=0.5,
        score_method="magnitude",
    )

    assert summary["score_method"] == "magnitude"
    assert summary["weights_zeroed"] == 2


def test_relative_importance_scores_balance_row_and_column_channels():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))

    scores = relative_importance_scores(model.named_parameters())

    weight_abs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = weight_abs / weight_abs.sum(dim=0, keepdim=True) + weight_abs / weight_abs.sum(dim=1, keepdim=True)

    assert torch.allclose(scores["weight"], expected)


def test_relative_importance_scores_apply_activation_rms_exponent():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))

    activation_rms = {"weight": torch.tensor([4.0, 9.0])}
    scores = relative_importance_scores(model.named_parameters(), activation_rms=activation_rms, activation_exponent=0.5)

    weight_abs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ri = weight_abs / weight_abs.sum(dim=0, keepdim=True) + weight_abs / weight_abs.sum(dim=1, keepdim=True)
    expected = ri * torch.tensor([2.0, 3.0]).unsqueeze(0)

    assert torch.allclose(scores["weight"], expected)


def test_input_activation_stat_scores_output_l2_mean_abs_var_max_and_fallbacks():
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.extra = torch.nn.Parameter(torch.tensor([[2.0, -3.0]]))
            self.linear = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.linear(inputs)

    model = Tiny()
    inputs = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])
    with InputActivationStatsAccumulator(model) as accumulator:
        model(inputs)
    stats = accumulator.finalize_stats()

    output_l2 = input_activation_stat_scores(model.named_parameters(), stats, "output_l2")
    mean_abs = input_activation_stat_scores(model.named_parameters(), stats, "mean_abs_wanda")
    var_output = input_activation_stat_scores(model.named_parameters(), stats, "var_output")
    max_abs = input_activation_stat_scores(model.named_parameters(), stats, "max_wanda")

    weight_abs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sumsq = torch.tensor([10.0, 20.0])
    mean_abs_stat = torch.tensor([2.0, 3.0])
    variance = torch.tensor([1.0, 9.0])
    max_abs_stat = torch.tensor([3.0, 4.0])

    assert torch.allclose(output_l2["linear.weight"], weight_abs.square() * sumsq.unsqueeze(0))
    assert torch.allclose(mean_abs["linear.weight"], weight_abs * mean_abs_stat.unsqueeze(0))
    assert torch.allclose(var_output["linear.weight"], weight_abs.square() * variance.unsqueeze(0))
    assert torch.allclose(max_abs["linear.weight"], weight_abs * max_abs_stat.unsqueeze(0))
    assert torch.equal(output_l2["extra"], torch.tensor([[4.0, 9.0]]))
    assert torch.equal(mean_abs["extra"], torch.tensor([[2.0, 3.0]]))


def test_input_activation_stats_stream_exact_top_tail_q95():
    model = torch.nn.Linear(2, 1, bias=False)
    inputs = torch.tensor([[[1.0, -10.0], [2.0, -20.0], [3.0, -30.0], [4.0, -40.0]]])

    with InputActivationStatsAccumulator(model, quantile=0.75, max_rows=4) as accumulator:
        model(inputs[:, :2])
        model(inputs[:, 2:])
    stats = accumulator.finalize_stats()

    assert torch.equal(stats["weight"]["q75_abs"], torch.tensor([3.0, 30.0]))


def test_angular_saliency_scores_exact_approx_hybrid_and_fallback():
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.extra = torch.nn.Parameter(torch.tensor([[3.0, -4.0]]))
            self.linear = torch.nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor([[2.0, -1.0]]))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.linear(inputs)

    model = Tiny()
    inputs = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    with AngularActivationAccumulator(model) as accumulator:
        model(inputs)
    stats = accumulator.finalize_stats()

    exact = angular_saliency_scores(model.named_parameters(), stats, "angular_exact")
    approx_scores = angular_saliency_scores(model.named_parameters(), stats, "angular_approx")
    hybrid = angular_saliency_scores(model.named_parameters(), stats, "angular_hybrid", hybrid_lambda=0.5)

    y_norm2 = torch.tensor([[8.0]])
    x_norm2 = torch.tensor([[1.0, 4.0]])
    yx_dot = torch.tensor([[2.0, -4.0]])
    weight = torch.tensor([[2.0, -1.0]])
    exact_expected = 1.0 - (y_norm2 - weight * yx_dot).div(
        y_norm2.sqrt() * (y_norm2 - 2.0 * weight * yx_dot + weight.square() * x_norm2).sqrt()
    )
    approx_expected = weight.square().div(y_norm2) * (x_norm2 - yx_dot.square().div(y_norm2))
    hybrid_expected = weight.square().div(y_norm2) * (x_norm2 - 0.5 * yx_dot.square().div(y_norm2))

    assert torch.allclose(exact["linear.weight"], exact_expected)
    assert torch.allclose(approx_scores["linear.weight"], approx_expected)
    assert torch.allclose(hybrid["linear.weight"], hybrid_expected)
    assert torch.equal(exact["extra"], torch.tensor([[9.0, 16.0]]))


def test_approx_saliency_config_accepts_angular_hybrid_lambda():
    config = ApproxSaliencyConfig(output_dir="unused", method="angular_hybrid", angular_hybrid_lambda=0.25)

    assert config.angular_hybrid_lambda == 0.25


def test_row_conditioned_wanda_gives_different_column_activations_by_row():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    inputs = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])

    with RowConditionedWandaAccumulator(model) as accumulator:
        model(inputs)
    scores = accumulator.finalize(model.named_parameters())

    outputs = torch.tensor([[[7.0, 15.0], [19.0, 43.0]]])
    x_abs = inputs.reshape(-1, 2).abs()
    y_abs = outputs.reshape(-1, 2).abs()
    row_conditioned_input = y_abs.T.matmul(x_abs) / y_abs.sum(dim=0).unsqueeze(1)
    expected = model.weight.detach().abs() * row_conditioned_input

    assert torch.allclose(scores["weight"], expected)
    assert not torch.allclose(row_conditioned_input[0], row_conditioned_input[1])
    assert scores["weight"][0, 0] != scores["weight"][1, 0]


def test_feature_wanda_cosine_scores_modulate_feature_wanda_by_exact_cosine_damage():
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.extra = torch.nn.Parameter(torch.tensor([[3.0, -4.0]]))
            self.linear = torch.nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor([[2.0, -0.5]]))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.linear(inputs)

    model = Tiny()
    inputs = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    with FeatureCosineWandaAccumulator(model) as accumulator:
        model(inputs)
    stats = accumulator.finalize_stats()
    scores = feature_wanda_cosine_scores(model.named_parameters(), stats, alpha=0.05)

    y = torch.tensor([[2.0], [-1.0]])
    x_abs = inputs.reshape(-1, 2).abs()
    y_abs = y.abs()
    base = model.linear.weight.detach().abs() * (y_abs.T.matmul(x_abs) / y_abs.sum(dim=0).unsqueeze(1))
    y_norm2 = torch.tensor([[5.0]])
    x_norm2 = torch.tensor([[1.0, 4.0]])
    yx_dot = torch.tensor([[2.0, -2.0]])
    weight = torch.tensor([[2.0, -0.5]])
    cos_damage = 1.0 - (y_norm2 - weight * yx_dot).div(
        y_norm2.sqrt() * (y_norm2 - 2.0 * weight * yx_dot + weight.square() * x_norm2).sqrt()
    )
    expected = base * (1.0 + 0.05 * cos_damage / cos_damage.mean())

    assert torch.allclose(scores["linear.weight"], expected)
    assert torch.equal(scores["extra"], torch.tensor([[3.0, 4.0]]))


def test_approx_saliency_config_accepts_feature_cosine_controls():
    config = ApproxSaliencyConfig(
        output_dir="unused",
        method="feature_cosine_wanda",
        feature_cosine_alpha=0.05,
        feature_cosine_clip=8.0,
    )

    assert config.feature_cosine_alpha == 0.05
    assert config.feature_cosine_clip == 8.0


def test_layernorm_input_jacobian_colnorm_squares_matches_autograd():
    norm = torch.nn.LayerNorm(3, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 1.5, 0.5]))
        norm.bias.copy_(torch.tensor([0.25, -0.5, 0.75]))
    inputs = torch.tensor([[1.0, 2.0, 4.0], [-2.0, 0.5, 3.0]], requires_grad=True)

    gain = layernorm_input_jacobian_colnorm_squares(inputs, norm)
    expected_rows = []
    for row in inputs.detach():
        row = row.clone().requires_grad_(True)
        jac = torch.autograd.functional.jacobian(lambda value: norm(value), row)
        expected_rows.append(jac.square().sum(dim=0))
    expected = torch.stack(expected_rows)

    assert torch.allclose(gain, expected, atol=1e-5, rtol=1e-5)


def test_layernorm_input_projected_gain_from_vectors_matches_autograd_vjps():
    norm = torch.nn.LayerNorm(3, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 1.5, 0.5]))
    inputs = torch.tensor([[1.0, 2.0, 4.0], [-2.0, 0.5, 3.0]])
    projected = torch.tensor([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]])

    gain = layernorm_input_projected_gain_from_vectors(inputs, norm, projected)
    expected_rows = []
    for row in inputs:
        per_probe = []
        for vector in projected:
            row_var = row.clone().requires_grad_(True)
            scalar = (norm(row_var) * vector).sum()
            (grad,) = torch.autograd.grad(scalar, row_var)
            per_probe.append(grad.square())
        expected_rows.append(torch.stack(per_probe).mean(dim=0))
    expected = torch.stack(expected_rows)

    assert torch.allclose(gain, expected, atol=1e-5, rtol=1e-5)


def test_layernorm_forward_diff_colnorm_squares_matches_bruteforce_positive_perturbation():
    norm = torch.nn.LayerNorm(3, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 1.5, 0.5]))
        norm.bias.copy_(torch.tensor([0.25, -0.5, 0.75]))
    inputs = torch.tensor([[1.0, 2.0, 4.0], [-2.0, 0.5, 3.0]])
    eps = 1e-3

    gain = layernorm_forward_diff_colnorm_squares(inputs, norm, eps=eps)
    baseline = norm(inputs.float())
    expected_rows = []
    for row_idx in range(inputs.shape[0]):
        per_col = []
        for col_idx in range(inputs.shape[1]):
            perturbed = inputs.float().clone()
            perturbed[row_idx, col_idx] += eps
            diff = norm(perturbed)[row_idx] - baseline[row_idx]
            per_col.append(diff.square().sum().div(eps * eps))
        expected_rows.append(torch.stack(per_col))
    expected = torch.stack(expected_rows)

    assert torch.allclose(gain, expected, atol=2e-3, rtol=2e-3)


def test_activation_local_jacobian_square_matches_gelu_autograd():
    activation = torch.nn.GELU()
    preactivation = torch.tensor([[-2.0, -0.5, 0.0, 1.5]], requires_grad=True)

    gain = activation_local_jacobian_square(preactivation, activation)
    (grad,) = torch.autograd.grad(activation(preactivation).sum(), preactivation)

    assert torch.allclose(gain, grad.detach().square(), atol=1e-6, rtol=1e-6)


def test_linear_superset_wanda_scores_match_two_matrix_counterfactual():
    inputs = torch.tensor([[2.0, -1.0], [0.5, 3.0], [-4.0, 1.5]])
    first = torch.tensor([[1.0, -2.0], [0.25, 3.0]])
    second = torch.tensor([[2.0, -1.0], [0.5, 4.0], [-3.0, 1.5]])
    baseline = inputs.matmul(first.t()).matmul(second.t())

    gain = second.square().sum(dim=0).expand(inputs.shape[0], -1)
    scores = linear_superset_wanda_scores(first, inputs, gain)

    expected = torch.zeros_like(first)
    for row_idx in range(first.shape[0]):
        for col_idx in range(first.shape[1]):
            pruned = first.clone()
            pruned[row_idx, col_idx] = 0.0
            diff = inputs.matmul(pruned.t()).matmul(second.t()) - baseline
            expected[row_idx, col_idx] = diff.square().sum()

    assert torch.allclose(scores, expected, atol=1e-6, rtol=1e-6)


def test_linear_superset_wanda_scores_include_activation_and_downstream_columns():
    inputs = torch.tensor([[1.5, -2.0], [-0.5, 3.0]])
    weight = torch.tensor([[0.1, -0.2], [0.05, 0.3]])
    preactivation = inputs.matmul(weight.t())
    activation = torch.nn.GELU()
    downstream = torch.tensor([[2.0, -1.0], [0.5, 4.0], [-3.0, 1.5]])

    gain = activation_local_jacobian_square(preactivation, activation).mul(downstream.square().sum(dim=0).unsqueeze(0))
    scores = linear_superset_wanda_scores(weight, inputs, gain)

    manual = weight.square().mul(gain.transpose(0, 1).matmul(inputs.square()))
    without_downstream = weight.square().mul(activation_local_jacobian_square(preactivation, activation).transpose(0, 1).matmul(inputs.square()))

    assert torch.allclose(scores, manual, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(scores, without_downstream)


def test_layernorm_input_downstream_colnorm_squares_matches_autograd_columns():
    norm = torch.nn.LayerNorm(3)
    downstream = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, -1.5, 2.0]))
        norm.bias.copy_(torch.tensor([0.25, -0.5, 1.0]))
        downstream.weight.copy_(torch.tensor([[1.0, -2.0, 0.5], [3.0, 0.25, -1.0]]))
    inputs = torch.tensor([[0.5, -1.0, 2.0], [3.0, 0.25, -0.75]], requires_grad=True)

    gain = layernorm_input_downstream_colnorm_squares(inputs, norm, downstream.weight)

    expected_rows = []
    for row_idx in range(inputs.shape[0]):
        row = inputs[row_idx : row_idx + 1].detach().requires_grad_(True)
        output = downstream(norm(row)).squeeze(0)
        per_col = []
        for col_idx in range(row.shape[1]):
            grads = []
            for out_idx in range(output.numel()):
                (grad,) = torch.autograd.grad(output[out_idx], row, retain_graph=True)
                grads.append(grad[0, col_idx])
            per_col.append(torch.stack(grads).square().sum())
        expected_rows.append(torch.stack(per_col))
    expected = torch.stack(expected_rows)

    assert torch.allclose(gain, expected, atol=1e-5, rtol=1e-5)


def test_rmsnorm_input_downstream_colnorm_squares_matches_autograd_columns():
    class TinyRmsNorm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.5, -1.5, 2.0]))
            self.variance_epsilon = 1e-6

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            variance = hidden.float().square().mean(dim=-1, keepdim=True)
            return self.weight * hidden.float() * torch.rsqrt(variance + self.variance_epsilon)

    norm = TinyRmsNorm()
    downstream = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        downstream.weight.copy_(torch.tensor([[1.0, -2.0, 0.5], [3.0, 0.25, -1.0]]))
    inputs = torch.tensor([[0.5, -1.0, 2.0], [3.0, 0.25, -0.75]], requires_grad=True)

    gain = rmsnorm_input_downstream_colnorm_squares(inputs, norm, downstream.weight)

    expected_rows = []
    for row_idx in range(inputs.shape[0]):
        row = inputs[row_idx : row_idx + 1].detach().requires_grad_(True)
        output = downstream(norm(row)).squeeze(0)
        per_col = []
        for col_idx in range(row.shape[1]):
            grads = []
            for out_idx in range(output.numel()):
                (grad,) = torch.autograd.grad(output[out_idx], row, retain_graph=True)
                grads.append(grad[0, col_idx])
            per_col.append(torch.stack(grads).square().sum())
        expected_rows.append(torch.stack(per_col))
    expected = torch.stack(expected_rows)

    assert torch.allclose(gain, expected, atol=1e-5, rtol=1e-5)


def test_qwen_mlp_superset_output_gains_match_autograd_endpoint_columns():
    gate = torch.tensor([[[0.25, -0.5], [1.0, 0.2]]], dtype=torch.float32)
    up = torch.tensor([[[1.5, -0.75], [0.4, 2.0]]], dtype=torch.float32)
    down = torch.tensor([[2.0, -1.0], [0.5, 4.0], [-3.0, 1.5]], dtype=torch.float32)

    gate_gain, up_gain = qwen_mlp_superset_output_gains(gate, up, down)

    gate_autograd = gate.detach().requires_grad_(True)
    up_autograd = up.detach().requires_grad_(True)
    endpoint = (torch.nn.functional.silu(gate_autograd) * up_autograd).matmul(down.t())
    flat_endpoint = endpoint.reshape(-1)
    expected_gate = torch.empty_like(gate)
    expected_up = torch.empty_like(up)
    for token_idx in range(gate.shape[1]):
        for coord_idx in range(gate.shape[-1]):
            gate_grads = []
            up_grads = []
            for out_idx in range(flat_endpoint.numel()):
                gate_grad, up_grad = torch.autograd.grad(flat_endpoint[out_idx], (gate_autograd, up_autograd), retain_graph=True)
                gate_grads.append(gate_grad[0, token_idx, coord_idx])
                up_grads.append(up_grad[0, token_idx, coord_idx])
            expected_gate[0, token_idx, coord_idx] = torch.stack(gate_grads).square().sum()
            expected_up[0, token_idx, coord_idx] = torch.stack(up_grads).square().sum()

    assert torch.allclose(gate_gain, expected_gate, atol=1e-5, rtol=1e-5)
    assert torch.allclose(up_gain, expected_up, atol=1e-5, rtol=1e-5)


def test_transform_superset_gain_clips_and_tempers_local_gain():
    gain = torch.tensor([[0.0, 1.0, 4.0, 100.0]])

    transformed = transform_superset_gain(gain, power=0.5, clip_quantile=0.75)

    assert torch.allclose(transformed, torch.tensor([[0.0, 1.0, 2.0, 2.0]]))


def test_superset_wanda_accumulator_can_temper_mlp_downstream_gain():
    class TinyMlp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense_h_to_4h = torch.nn.Linear(2, 2, bias=False)
            self.act = torch.nn.Identity()
            self.dense_4h_to_h = torch.nn.Linear(2, 2, bias=False)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.dense_4h_to_h(self.act(self.dense_h_to_4h(hidden)))

    class TinyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = TinyMlp()

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.mlp(hidden)

    class TinyNeoX(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([TinyLayer()])

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gpt_neox = TinyNeoX()

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.gpt_neox.layers[0](hidden)

    model = TinyModel()
    with torch.no_grad():
        model.gpt_neox.layers[0].mlp.dense_h_to_4h.weight.copy_(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))
        model.gpt_neox.layers[0].mlp.dense_4h_to_h.weight.copy_(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
    hidden = torch.tensor([[[2.0, -1.0], [0.5, 3.0]]])
    target = "gpt_neox.layers.0.mlp.dense_h_to_4h.weight"

    with LocalForwardWandaAccumulator(
        model,
        use_attention_mask=False,
        target_names=[target],
        closed_form=True,
        superset_gain_power=0.5,
    ) as accumulator:
        accumulator.set_batch(torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.bool))
        model(hidden)
        accumulator.clear_batch()
    scores = accumulator.finalize(model.named_parameters())

    up = model.gpt_neox.layers[0].mlp.dense_h_to_4h.weight.detach()
    down = model.gpt_neox.layers[0].mlp.dense_4h_to_h.weight.detach()
    gain = down.square().sum(dim=0).sqrt()
    expected = linear_superset_wanda_scores(up, hidden, gain)
    assert torch.allclose(scores[target], expected, atol=2e-3, rtol=1e-3)


def test_qwen_attention_qkv_superset_output_gains_match_autograd_endpoint_columns():
    class TinyRmsNorm(torch.nn.Module):
        def __init__(self, weight: torch.Tensor) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(weight.clone())
            self.variance_epsilon = 1e-6

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            variance = hidden.float().square().mean(dim=-1, keepdim=True)
            return self.weight * hidden.float() * torch.rsqrt(variance + self.variance_epsilon)

    batch = 1
    seq_len = 3
    num_heads = 2
    num_kv_heads = 1
    head_size = 2
    q = torch.tensor(
        [
            [
                [0.2, -0.1, 0.3, 0.4],
                [-0.5, 0.25, 0.1, -0.2],
                [0.3, 0.2, -0.4, 0.1],
            ]
        ],
        dtype=torch.float32,
    )
    k = torch.tensor([[[0.4, -0.1], [0.3, 0.2], [-0.2, 0.45]]], dtype=torch.float32)
    v = torch.tensor([[[-0.6, 0.15], [0.35, -0.45], [0.1, 0.7]]], dtype=torch.float32)
    dense_weight = torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.75],
            [-0.25, 0.5, -1.0, 0.3],
            [0.4, 0.2, 0.6, -0.8],
        ],
        dtype=torch.float32,
    )
    q_norm = TinyRmsNorm(torch.tensor([0.7, -1.3]))
    k_norm = TinyRmsNorm(torch.tensor([1.1, 0.6]))
    cos = torch.tensor([[[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]]], dtype=torch.float32)
    sin = (1.0 - cos.square()).sqrt()
    causal = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    scaling = head_size ** -0.5

    gains = qwen_attention_qkv_superset_output_gains(
        q,
        k,
        v,
        dense_weight,
        q_norm,
        k_norm,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_size=head_size,
        attention_mask=causal,
        scaling=scaling,
        position_embeddings=(cos, sin),
    )

    q_auto = q.detach().requires_grad_(True)
    k_auto = k.detach().requires_grad_(True)
    v_auto = v.detach().requires_grad_(True)
    q_states = q_norm(q_auto.view(batch, seq_len, num_heads, head_size)).transpose(1, 2)
    k_states = k_norm(k_auto.view(batch, seq_len, num_kv_heads, head_size)).transpose(1, 2)
    v_states = v_auto.view(batch, seq_len, num_kv_heads, head_size).transpose(1, 2)
    q_states = approx._qwen_apply_rotary_single(q_states, cos, sin)
    k_states = approx._qwen_apply_rotary_single(k_states, cos, sin)
    k_states = k_states[:, :, None].expand(batch, num_kv_heads, 2, seq_len, head_size).reshape(batch, num_heads, seq_len, head_size)
    v_states = v_states[:, :, None].expand(batch, num_kv_heads, 2, seq_len, head_size).reshape(batch, num_heads, seq_len, head_size)
    attn = torch.softmax(q_states.matmul(k_states.transpose(2, 3)).mul(scaling).add(causal), dim=-1)
    context = attn.matmul(v_states).transpose(1, 2).reshape(batch, seq_len, num_heads * head_size)
    endpoint = context.matmul(dense_weight.t())
    flat_endpoint = endpoint.reshape(-1)
    expected = [torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)]
    for tensor_idx, source in enumerate((q_auto, k_auto, v_auto)):
        for token_idx in range(source.shape[1]):
            for coord_idx in range(source.shape[-1]):
                grads = []
                for out_idx in range(flat_endpoint.numel()):
                    grad = torch.autograd.grad(flat_endpoint[out_idx], source, retain_graph=True)[0]
                    grads.append(grad[0, token_idx, coord_idx])
                expected[tensor_idx][0, token_idx, coord_idx] = torch.stack(grads).square().sum()

    for got, want in zip(gains, expected, strict=True):
        assert torch.allclose(got, want, atol=2e-5, rtol=2e-5)


def test_split_attention_superset_output_gains_without_qk_norms_match_autograd_endpoint_columns():
    batch = 1
    seq_len = 3
    num_heads = 2
    head_size = 2
    q = torch.tensor(
        [[[0.2, -0.1, 0.3, 0.4], [-0.5, 0.25, 0.1, -0.2], [0.3, 0.2, -0.4, 0.1]]],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [[[0.4, -0.1, 0.3, 0.2], [-0.2, 0.45, 0.5, 0.25], [0.1, -0.35, 0.55, 0.05]]],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [[[-0.6, 0.15, 0.35, -0.45], [0.1, 0.7, 0.6, -0.4], [0.2, 0.7, -0.5, 0.25]]],
        dtype=torch.float32,
    )
    dense_weight = torch.tensor(
        [[1.0, -0.5, 0.25, 0.75], [-0.25, 0.5, -1.0, 0.3], [0.4, 0.2, 0.6, -0.8]],
        dtype=torch.float32,
    )
    causal = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    scaling = head_size ** -0.5

    gains = qwen_attention_qkv_superset_output_gains(
        q,
        k,
        v,
        dense_weight,
        None,
        None,
        num_heads=num_heads,
        num_key_value_heads=num_heads,
        head_size=head_size,
        attention_mask=causal,
        scaling=scaling,
        position_embeddings=None,
    )

    q_auto = q.detach().requires_grad_(True)
    k_auto = k.detach().requires_grad_(True)
    v_auto = v.detach().requires_grad_(True)
    q_states = q_auto.view(batch, seq_len, num_heads, head_size).transpose(1, 2)
    k_states = k_auto.view(batch, seq_len, num_heads, head_size).transpose(1, 2)
    v_states = v_auto.view(batch, seq_len, num_heads, head_size).transpose(1, 2)
    attn = torch.softmax(q_states.matmul(k_states.transpose(2, 3)).mul(scaling).add(causal), dim=-1)
    context = attn.matmul(v_states).transpose(1, 2).reshape(batch, seq_len, num_heads * head_size)
    endpoint = context.matmul(dense_weight.t())
    flat_endpoint = endpoint.reshape(-1)
    expected = [torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)]
    for tensor_idx, source in enumerate((q_auto, k_auto, v_auto)):
        for token_idx in range(source.shape[1]):
            for coord_idx in range(source.shape[-1]):
                grads = []
                for out_idx in range(flat_endpoint.numel()):
                    grad = torch.autograd.grad(flat_endpoint[out_idx], source, retain_graph=True)[0]
                    grads.append(grad[0, token_idx, coord_idx])
                expected[tensor_idx][0, token_idx, coord_idx] = torch.stack(grads).square().sum()

    for got, want in zip(gains, expected, strict=True):
        assert torch.allclose(got, want, atol=2e-5, rtol=2e-5)


def test_superset_wanda_accumulator_uses_mlp_downstream_column_norm():
    class TinyMlp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense_h_to_4h = torch.nn.Linear(2, 2, bias=False)
            self.act = torch.nn.Identity()
            self.dense_4h_to_h = torch.nn.Linear(2, 3, bias=False)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.dense_4h_to_h(self.act(self.dense_h_to_4h(hidden)))

    class TinyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = TinyMlp()

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.mlp(hidden)

    class TinyNeoX(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([TinyLayer()])

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gpt_neox = TinyNeoX()

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.gpt_neox.layers[0](hidden)

    model = TinyModel()
    with torch.no_grad():
        model.gpt_neox.layers[0].mlp.dense_h_to_4h.weight.copy_(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))
        model.gpt_neox.layers[0].mlp.dense_4h_to_h.weight.copy_(torch.tensor([[2.0, -1.0], [0.5, 4.0], [-3.0, 1.5]]))
    hidden = torch.tensor([[[2.0, -1.0], [0.5, 3.0]]])
    target = "gpt_neox.layers.0.mlp.dense_h_to_4h.weight"

    with LocalForwardWandaAccumulator(model, use_attention_mask=False, target_names=[target], closed_form=True) as accumulator:
        accumulator.set_batch(torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.bool))
        model(hidden)
        accumulator.clear_batch()
    scores = accumulator.finalize(model.named_parameters())

    up = model.gpt_neox.layers[0].mlp.dense_h_to_4h.weight.detach()
    down = model.gpt_neox.layers[0].mlp.dense_4h_to_h.weight.detach()
    expected = linear_superset_wanda_scores(up, hidden, down.square().sum(dim=0))
    assert torch.allclose(scores[target], expected, atol=2e-3, rtol=1e-3)


def test_attention_qkv_superset_output_gains_match_autograd_endpoint_columns():
    batch = 1
    seq_len = 3
    num_heads = 2
    head_size = 2
    hidden = num_heads * head_size
    qkv = torch.tensor(
        [
            [
                [0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1, -0.3, 0.6, -0.4, 0.2, 0.7],
                [-0.5, 0.25, 0.1, -0.2, 0.4, -0.1, 0.3, 0.2, -0.6, 0.15, 0.35, -0.45],
                [0.3, 0.2, -0.4, 0.1, 0.5, 0.25, -0.2, 0.45, 0.1, -0.35, 0.55, 0.05],
            ]
        ],
        dtype=torch.float32,
    )
    dense_weight = torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.75],
            [-0.25, 0.5, -1.0, 0.3],
            [0.4, 0.2, 0.6, -0.8],
        ],
        dtype=torch.float32,
    )
    causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).view(1, 1, seq_len, seq_len)
    scaling = head_size ** -0.5

    gains = attention_qkv_superset_output_gains(
        qkv,
        dense_weight,
        num_heads=num_heads,
        head_size=head_size,
        attention_mask=causal,
        scaling=scaling,
    )

    qkv_autograd = qkv.detach().requires_grad_(True)
    raw = qkv_autograd.view(batch, seq_len, num_heads, 3 * head_size).transpose(1, 2)
    query, key, value = raw.chunk(3, dim=-1)
    attn = torch.softmax(query.matmul(key.transpose(2, 3)).mul(scaling).add(causal), dim=-1)
    context = attn.matmul(value).transpose(1, 2).reshape(batch, seq_len, hidden)
    endpoint = context.matmul(dense_weight.t())
    expected = torch.empty_like(qkv)
    flat_endpoint = endpoint.reshape(-1)
    for token_idx in range(seq_len):
        for coord_idx in range(qkv.shape[-1]):
            grads = []
            for out_idx in range(flat_endpoint.numel()):
                (grad,) = torch.autograd.grad(flat_endpoint[out_idx], qkv_autograd, retain_graph=True)
                grads.append(grad[0, token_idx, coord_idx])
            expected[0, token_idx, coord_idx] = torch.stack(grads).square().sum()

    assert torch.allclose(gains, expected, atol=1e-5, rtol=1e-5)


def test_attention_qkv_superset_output_gains_push_through_rotary_embedding():
    batch = 1
    seq_len = 2
    num_heads = 1
    head_size = 4
    hidden = num_heads * head_size
    qkv = torch.tensor(
        [
            [
                [0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1, -0.3, 0.6, -0.4, 0.2, 0.7],
                [-0.5, 0.25, 0.1, -0.2, 0.4, -0.1, 0.3, 0.2, -0.6, 0.15, 0.35, -0.45],
            ]
        ],
        dtype=torch.float32,
    )
    dense_weight = torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.75],
            [-0.25, 0.5, -1.0, 0.3],
            [0.4, 0.2, 0.6, -0.8],
        ],
        dtype=torch.float32,
    )
    cos = torch.tensor([[[0.9, 0.8, 0.9, 0.8], [0.7, 0.6, 0.7, 0.6]]], dtype=torch.float32)
    sin = (1.0 - cos.square()).sqrt()
    causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).view(1, 1, seq_len, seq_len)
    scaling = head_size ** -0.5

    gains = attention_qkv_superset_output_gains(
        qkv,
        dense_weight,
        num_heads=num_heads,
        head_size=head_size,
        attention_mask=causal,
        scaling=scaling,
        position_embeddings=(cos, sin),
    )

    qkv_autograd = qkv.detach().requires_grad_(True)
    raw = qkv_autograd.view(batch, seq_len, num_heads, 3 * head_size).transpose(1, 2)
    query, key, value = raw.chunk(3, dim=-1)
    query, key = approx.apply_rotary_pos_emb(query, key, cos, sin)
    attn = torch.softmax(query.matmul(key.transpose(2, 3)).mul(scaling).add(causal), dim=-1)
    context = attn.matmul(value).transpose(1, 2).reshape(batch, seq_len, hidden)
    endpoint = context.matmul(dense_weight.t())
    expected = torch.empty_like(qkv)
    flat_endpoint = endpoint.reshape(-1)
    for token_idx in range(seq_len):
        for coord_idx in range(qkv.shape[-1]):
            grads = []
            for out_idx in range(flat_endpoint.numel()):
                (grad,) = torch.autograd.grad(flat_endpoint[out_idx], qkv_autograd, retain_graph=True)
                grads.append(grad[0, token_idx, coord_idx])
            expected[0, token_idx, coord_idx] = torch.stack(grads).square().sum()

    assert torch.allclose(gains, expected, atol=1e-5, rtol=1e-5)


def test_superset_wanda_accumulator_uses_attention_qkv_softmax_gain():
    class TinyAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_attention_heads = 1
            self.head_size = 2
            self.scaling = self.head_size ** -0.5
            self.query_key_value = torch.nn.Linear(2, 6, bias=False)
            self.dense = torch.nn.Linear(2, 2, bias=False)

        def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None, position_embeddings=None) -> torch.Tensor:
            del position_embeddings
            batch, seq_len, _ = hidden.shape
            qkv = self.query_key_value(hidden).view(batch, seq_len, 1, 6).transpose(1, 2)
            query, key, value = qkv.chunk(3, dim=-1)
            logits = query.matmul(key.transpose(2, 3)).mul(self.scaling)
            if attention_mask is not None:
                logits = logits + attention_mask
            attn = torch.softmax(logits, dim=-1)
            context = attn.matmul(value).transpose(1, 2).reshape(batch, seq_len, 2)
            return self.dense(context)

    class TinyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = TinyAttention()

        def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
            return self.attention(hidden, attention_mask=attention_mask)

    class TinyNeoX(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([TinyLayer()])

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gpt_neox = TinyNeoX()

        def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
            return self.gpt_neox.layers[0](hidden, attention_mask=attention_mask)

    model = TinyModel()
    with torch.no_grad():
        model.gpt_neox.layers[0].attention.query_key_value.weight.copy_(
            torch.tensor(
                [
                    [0.2, -0.1],
                    [0.3, 0.4],
                    [-0.2, 0.5],
                    [0.1, -0.3],
                    [0.6, -0.4],
                    [0.2, 0.7],
                ]
            )
        )
        model.gpt_neox.layers[0].attention.dense.weight.copy_(torch.tensor([[1.0, -0.5], [0.25, 0.75]]))
    hidden = torch.tensor([[[2.0, -1.0], [0.5, 3.0], [-0.25, 1.5]]])
    mask = torch.triu(torch.full((1, 1, 3, 3), float("-inf")), diagonal=1)
    target = "gpt_neox.layers.0.attention.query_key_value.weight"

    with LocalForwardWandaAccumulator(model, use_attention_mask=False, target_names=[target], closed_form=True) as accumulator:
        accumulator.set_batch(torch.ones(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.bool))
        model(hidden, attention_mask=mask)
        accumulator.clear_batch()
    scores = accumulator.finalize(model.named_parameters())

    qkv_output = model.gpt_neox.layers[0].attention.query_key_value(hidden)
    gain = attention_qkv_superset_output_gains(
        qkv_output,
        model.gpt_neox.layers[0].attention.dense.weight,
        num_heads=1,
        head_size=2,
        attention_mask=mask,
        scaling=2 ** -0.5,
    )
    expected = linear_superset_wanda_scores(model.gpt_neox.layers[0].attention.query_key_value.weight, hidden, gain.reshape(-1, 6))

    assert torch.allclose(scores[target], expected, atol=1e-5, rtol=1e-5)


def test_superset_wanda_accumulator_uses_llama_split_qkv_without_qk_norms():
    class TinyAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = 1
            self.num_key_value_heads = 1
            self.head_dim = 2
            self.scaling = self.head_dim ** -0.5
            self.q_proj = torch.nn.Linear(2, 2, bias=False)
            self.k_proj = torch.nn.Linear(2, 2, bias=False)
            self.v_proj = torch.nn.Linear(2, 2, bias=False)
            self.o_proj = torch.nn.Linear(2, 2, bias=False)

        def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None, position_embeddings=None) -> torch.Tensor:
            del position_embeddings
            query = self.q_proj(hidden).view(1, hidden.shape[1], 1, 2).transpose(1, 2)
            key = self.k_proj(hidden).view(1, hidden.shape[1], 1, 2).transpose(1, 2)
            value = self.v_proj(hidden).view(1, hidden.shape[1], 1, 2).transpose(1, 2)
            logits = query.matmul(key.transpose(2, 3)).mul(self.scaling)
            if attention_mask is not None:
                logits = logits + attention_mask
            attn = torch.softmax(logits, dim=-1)
            context = attn.matmul(value).transpose(1, 2).reshape(1, hidden.shape[1], 2)
            return self.o_proj(context)

    class TinyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = TinyAttention()

        def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
            return self.self_attn(hidden, attention_mask=attention_mask)

    class TinyBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([TinyLayer()])

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = TinyBackbone()

        def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
            return self.model.layers[0](hidden, attention_mask=attention_mask)

    model = TinyModel()
    hidden = torch.tensor([[[2.0, -1.0], [0.5, 3.0], [-0.25, 1.5]]])
    mask = torch.triu(torch.full((1, 1, 3, 3), float("-inf")), diagonal=1)
    target = "model.layers.0.self_attn.q_proj.weight"

    with LocalForwardWandaAccumulator(model, use_attention_mask=False, target_names=[target], closed_form=True) as accumulator:
        accumulator.set_batch(torch.ones(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.bool))
        model(hidden, attention_mask=mask)
        accumulator.clear_batch()
    scores = accumulator.finalize(model.named_parameters())

    q_gain, _, _ = qwen_attention_qkv_superset_output_gains(
        model.model.layers[0].self_attn.q_proj(hidden),
        model.model.layers[0].self_attn.k_proj(hidden),
        model.model.layers[0].self_attn.v_proj(hidden),
        model.model.layers[0].self_attn.o_proj.weight,
        None,
        None,
        num_heads=1,
        num_key_value_heads=1,
        head_size=2,
        attention_mask=mask,
        scaling=2 ** -0.5,
        position_embeddings=None,
    )
    expected = linear_superset_wanda_scores(model.model.layers[0].self_attn.q_proj.weight, hidden, q_gain.reshape(-1, 2))

    assert torch.allclose(scores[target], expected, atol=1e-5, rtol=1e-5)


def test_superset_wanda_accumulator_covers_opt_decoder_block_targets():
    class TinyOptAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = 1
            self.head_dim = 2
            self.scaling = self.head_dim ** -0.5
            self.q_proj = torch.nn.Linear(2, 2, bias=False)
            self.k_proj = torch.nn.Linear(2, 2, bias=False)
            self.v_proj = torch.nn.Linear(2, 2, bias=False)
            self.out_proj = torch.nn.Linear(2, 2, bias=False)

        def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs) -> tuple[torch.Tensor, None]:
            del kwargs
            query = self.q_proj(hidden_states).view(1, hidden_states.shape[1], 1, 2).transpose(1, 2)
            key = self.k_proj(hidden_states).view(1, hidden_states.shape[1], 1, 2).transpose(1, 2)
            value = self.v_proj(hidden_states).view(1, hidden_states.shape[1], 1, 2).transpose(1, 2)
            logits = query.matmul(key.transpose(2, 3)).mul(self.scaling)
            if attention_mask is not None:
                logits = logits + attention_mask
            attn = torch.softmax(logits, dim=-1)
            context = attn.matmul(value).transpose(1, 2).reshape(1, hidden_states.shape[1], 2)
            return self.out_proj(context), None

    class TinyOptLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.do_layer_norm_before = True
            self.self_attn_layer_norm = torch.nn.LayerNorm(2)
            self.self_attn = TinyOptAttention()
            self.final_layer_norm = torch.nn.LayerNorm(2)
            self.fc1 = torch.nn.Linear(2, 3, bias=False)
            self.activation_fn = torch.nn.GELU()
            self.fc2 = torch.nn.Linear(3, 2, bias=False)

        def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
            residual = hidden_states
            hidden_states = self.self_attn_layer_norm(hidden_states)
            hidden_states = residual + self.self_attn(hidden_states, attention_mask=attention_mask)[0]
            residual = hidden_states
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
            return residual + hidden_states

    class TinyDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(5, 2)
            self.embed_positions = torch.nn.Embedding(8, 2)
            self.layers = torch.nn.ModuleList([TinyOptLayer(), TinyOptLayer()])
            self.final_layer_norm = torch.nn.LayerNorm(2)

    class TinyBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = TinyDecoder()

    class TinyOptModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = TinyBackbone()
            self.lm_head = torch.nn.Linear(2, 5, bias=False)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
            del attention_mask
            decoder = self.model.decoder
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            hidden = decoder.embed_tokens(input_ids) + decoder.embed_positions(positions)
            causal = torch.triu(torch.full((1, 1, input_ids.shape[1], input_ids.shape[1]), float("-inf")), diagonal=1)
            for layer in decoder.layers:
                hidden = layer(hidden, attention_mask=causal)
            return self.lm_head(decoder.final_layer_norm(hidden))

    model = TinyOptModel()
    input_ids = torch.tensor([[1, 2, 3]])
    target_names = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.ndim == 2 and "layer_norm" not in name
    ]

    with LocalForwardWandaAccumulator(model, use_attention_mask=False, target_names=target_names, closed_form=True) as accumulator:
        accumulator.set_batch(input_ids, torch.ones_like(input_ids, dtype=torch.bool))
        model(input_ids)
        accumulator.clear_batch()
    scores = accumulator.finalize(model.named_parameters())

    assert set(scores) == set(target_names)
    assert all(tuple(scores[name].shape) == tuple(dict(model.named_parameters())[name].shape) for name in target_names)


def test_activation_forward_diff_scores_matches_bruteforce_weight_prune_endpoint():
    weight = torch.tensor([[1.0, -2.0], [0.5, -1.5]])
    inputs = torch.tensor([[2.0, -3.0], [4.0, 5.0]])
    preactivation = inputs.matmul(weight.t())
    activation = torch.nn.GELU()
    eps = 1e-3

    scores = activation_forward_diff_scores(weight, inputs, preactivation, activation, eps=eps)
    expected = torch.zeros_like(weight)
    baseline = activation(preactivation)
    for row_idx in range(weight.shape[0]):
        for col_idx in range(weight.shape[1]):
            perturbed = preactivation.clone()
            perturbed[:, row_idx] -= weight[row_idx, col_idx] * inputs[:, col_idx]
            diff = activation(perturbed)[:, row_idx] - baseline[:, row_idx]
            expected[row_idx, col_idx] = diff.square().sum()

    assert torch.allclose(scores, expected, atol=1e-4, rtol=1e-4)


def test_graph_propagated_scores_use_propagated_stats_and_output_l2_fallback():
    model = torch.nn.Module()
    model.residual = torch.nn.Linear(2, 2, bias=False)
    model.other = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.residual.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))
        model.other.weight.copy_(torch.tensor([[5.0, -6.0]]))

    scores = graph_propagated_scores(
        model.named_parameters(),
        propagated_input_sums={
            "residual.weight": torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        },
        fallback_input_sumsq={
            "residual.weight": torch.tensor([100.0, 100.0]),
            "other.weight": torch.tensor([7.0, 11.0]),
        },
    )

    assert torch.equal(scores["residual.weight"], torch.tensor([[2.0, 12.0], [36.0, 80.0]]))
    assert torch.equal(scores["other.weight"], torch.tensor([[175.0, 396.0]]))


def test_accumulate_vjp_parameter_scores_uses_weight_times_endpoint_vjp_squared():
    linear = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))
    inputs = torch.tensor([[2.0, 5.0]])
    endpoint_vector = torch.tensor([[0.5, -1.5]])
    scores = {"weight": torch.zeros_like(linear.weight, dtype=torch.float32)}

    scalar = (linear(inputs) * endpoint_vector).sum()
    scalar.backward()
    accumulate_vjp_parameter_scores_(scores, [("weight", linear.weight)])

    expected_grad = endpoint_vector.transpose(0, 1).matmul(inputs)
    expected = linear.weight.detach().mul(expected_grad).square()
    assert torch.equal(scores["weight"], expected)


def test_accumulate_vjp_gradient_scores_updates_only_supplied_gradients_and_counts():
    linear = torch.nn.Linear(2, 2, bias=False)
    other = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))
        other.weight.copy_(torch.tensor([[5.0, -6.0]]))
    scores = {
        "linear.weight": torch.zeros_like(linear.weight, dtype=torch.float32),
        "other.weight": torch.zeros_like(other.weight, dtype=torch.float32),
    }
    normalizers: dict[str, int] = {}

    accumulate_vjp_gradient_scores_(
        scores,
        normalizers,
        [
            ("linear.weight", linear.weight, torch.tensor([[0.5, 1.0], [-1.5, 2.0]])),
            ("other.weight", other.weight, None),
        ],
        count=7,
    )

    assert torch.equal(scores["linear.weight"], torch.tensor([[0.25, 4.0], [20.25, 64.0]]))
    assert torch.equal(scores["other.weight"], torch.zeros_like(other.weight))
    assert normalizers == {"linear.weight": 7}


def test_build_local_subgraph_endpoint_groups_covers_gpt_neox_matrix_names_without_fallbacks():
    names = ["gpt_neox.embed_in.weight", "embed_out.weight"]
    for layer in range(2):
        names.extend(
            [
                f"gpt_neox.layers.{layer}.attention.query_key_value.weight",
                f"gpt_neox.layers.{layer}.attention.dense.weight",
                f"gpt_neox.layers.{layer}.mlp.dense_h_to_4h.weight",
                f"gpt_neox.layers.{layer}.mlp.dense_4h_to_h.weight",
            ]
        )

    groups = build_local_subgraph_endpoint_groups(num_layers=2, parameter_names=set(names))
    covered = {target for group in groups for target in group["targets"]}

    assert covered == set(names)
    assert {"attn_context_0", "mlp_activation_0", "attn_context_1", "mlp_activation_1", "final_norm", "logits"} <= {
        group["endpoint"] for group in groups
    }


def test_cross_entropy_residual_uses_causal_shift_and_supervised_tokens_only():
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0],
                [0.0, 2.0, -1.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )
    labels = torch.tensor([[-100, 1, -100]])

    residual, mask = cross_entropy_residual(logits, labels)

    expected = torch.softmax(logits[:, :-1, :], dim=-1)[0, 0]
    expected[1] -= 1.0
    assert mask.tolist() == [[True, False]]
    assert torch.allclose(residual.cpu()[0], expected)


def test_hash_project_residual_is_deterministic_and_shape_correct():
    residual = torch.tensor([[0.5, -0.25, 0.125], [0.0, 0.25, -0.5]])

    first = hash_project_residual(residual, out_features=5, seed=17)
    second = hash_project_residual(residual, out_features=5, seed=17)

    assert first.shape == (2, 5)
    assert torch.equal(first, second)


def test_dfa_activation_collector_captures_linear_inputs_without_gradients():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 2))
    inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    with DfaActivationCollector(model) as collector:
        model(inputs)

    assert set(collector.inputs) == {"0.weight", "1.weight"}
    assert torch.equal(collector.inputs["0.weight"], inputs)
    assert collector.inputs["1.weight"].shape == (1, 2, 3)
    assert all(param.grad is None for param in model.parameters())


def test_incremental_per_matrix_pruning_skips_weights_pruned_in_prior_steps():
    model = torch.nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    masks: dict[str, torch.Tensor] = {}
    first = apply_incremental_per_matrix_pruning_(
        model,
        {"weight": torch.tensor([[0.1, 0.2, 0.3, 0.4]])},
        pruned_masks=masks,
        target_fraction=0.5,
        chunk_fraction=0.25,
        step=1,
    )
    second = apply_incremental_per_matrix_pruning_(
        model,
        {"weight": torch.tensor([[0.0, 0.1, 0.3, 0.4]])},
        pruned_masks=masks,
        target_fraction=0.5,
        chunk_fraction=0.25,
        step=2,
    )

    assert model.weight.tolist() == [[0.0, 0.0, 3.0, 4.0]]
    assert first["weights_zeroed_this_step"] == 1
    assert second["weights_zeroed_this_step"] == 1
    assert masks["weight"].tolist() == [[True, True, False, False]]


def test_incremental_per_matrix_pruning_stops_at_target_fraction():
    model = torch.nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    masks: dict[str, torch.Tensor] = {}
    apply_incremental_per_matrix_pruning_(
        model,
        {"weight": torch.tensor([[0.1, 0.2, 0.3, 0.4]])},
        pruned_masks=masks,
        target_fraction=0.25,
        chunk_fraction=0.25,
        step=1,
    )
    summary = apply_incremental_per_matrix_pruning_(
        model,
        {"weight": torch.tensor([[0.0, 0.2, 0.3, 0.4]])},
        pruned_masks=masks,
        target_fraction=0.25,
        chunk_fraction=0.25,
        step=2,
    )

    assert model.weight.tolist() == [[0.0, 2.0, 3.0, 4.0]]
    assert summary["weights_zeroed_this_step"] == 0
    assert summary["target_reached"] is True


def test_incremental_nm_pruning_zeros_one_lowest_entry_per_quartet_each_step():
    model = torch.nn.Linear(8, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.arange(1.0, 9.0).reshape(1, 8))

    masks: dict[str, torch.Tensor] = {}
    first = apply_incremental_nm_pruning_(
        model,
        {"weight": torch.tensor([[0.4, 0.1, 0.3, 0.2, 0.8, 0.7, 0.6, 0.5]])},
        pruned_masks=masks,
        n=2,
        m=4,
        target_zeros_per_group=1,
        group_dim=1,
        step=1,
    )
    second = apply_incremental_nm_pruning_(
        model,
        {"weight": torch.tensor([[0.4, 0.0, 0.3, 0.2, 0.8, 0.7, 0.1, 0.5]])},
        pruned_masks=masks,
        n=2,
        m=4,
        target_zeros_per_group=2,
        group_dim=1,
        step=2,
    )

    assert first["weights_zeroed_this_step"] == 2
    assert second["weights_zeroed_this_step"] == 2
    assert model.weight.tolist() == [[1.0, 0.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0]]
    assert masks["weight"].tolist() == [[False, True, False, True, False, False, True, True]]
    assert second["target_reached"] is True


def test_incremental_nm_pruning_requires_native_quartets_along_input_dim():
    model = torch.nn.Linear(6, 1, bias=False)
    masks: dict[str, torch.Tensor] = {}

    try:
        apply_incremental_nm_pruning_(
            model,
            {"weight": torch.ones(1, 6)},
            pruned_masks=masks,
            n=2,
            m=4,
            target_zeros_per_group=1,
            group_dim=1,
            step=1,
        )
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected quartet divisibility check to fail")


def test_incremental_nm_pruning_to_parameter_prunes_only_requested_matrix():
    model = torch.nn.Module()
    model.first = torch.nn.Linear(8, 1, bias=False)
    model.second = torch.nn.Linear(8, 1, bias=False)
    with torch.no_grad():
        model.first.weight.copy_(torch.arange(1.0, 9.0).reshape(1, 8))
        model.second.weight.copy_(torch.arange(11.0, 19.0).reshape(1, 8))

    masks: dict[str, torch.Tensor] = {}
    summary = apply_incremental_nm_pruning_to_parameter_(
        model,
        {
            "first.weight": torch.tensor([[0.4, 0.1, 0.3, 0.2, 0.8, 0.7, 0.6, 0.5]]),
            "second.weight": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]),
        },
        pruned_masks=masks,
        parameter_name="first.weight",
        n=2,
        m=4,
        target_zeros_per_group=2,
        group_dim=1,
        step=1,
    )

    assert summary["name"] == "first.weight"
    assert summary["zeroed_this_step"] == 4
    assert model.first.weight.tolist() == [[1.0, 0.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0]]
    assert model.second.weight.tolist() == [[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]]
    assert set(masks) == {"first.weight"}


def test_nm_global_pass_matrix_attribution_repairs_only_after_each_full_pass(monkeypatch, tmp_path):
    model = torch.nn.Module()
    model.first = torch.nn.Linear(8, 1, bias=False)
    model.second = torch.nn.Linear(8, 1, bias=False)
    with torch.no_grad():
        model.first.weight.copy_(torch.arange(1.0, 9.0).reshape(1, 8))
        model.second.weight.copy_(torch.arange(11.0, 19.0).reshape(1, 8))

    mask_counts_at_eval: list[int] = []
    repair_mask_counts: list[int] = []
    wanda_calls = 0

    def fake_prepare_model(*args, **kwargs):
        return model

    def fake_prepare_tokenizer(*args, **kwargs):
        return object()

    def fake_load_records(*args, **kwargs):
        return [{"question": "q", "answer": "a"}]

    def fake_evaluate(model_arg, *args, **kwargs):
        assert model_arg is model
        mask_counts_at_eval.append(
            int((model.first.weight == 0).sum().item() + (model.second.weight == 0).sum().item())
        )
        return PerplexityStats(loss_sum=float(len(mask_counts_at_eval)), supervised_tokens=1, num_batches=1, num_examples=1)

    def fake_wanda(*args, **kwargs):
        nonlocal wanda_calls
        wanda_calls += 1
        return (
            {
                "first.weight": torch.tensor([[0.4, 0.1, 0.3, 0.2, 0.8, 0.7, 0.6, 0.5]]),
                "second.weight": torch.tensor([[0.1, 0.4, 0.2, 0.3, 0.5, 0.8, 0.6, 0.7]]),
            },
            {},
            {},
        )

    def fake_repair(*args, **kwargs):
        masks = kwargs["pruned_masks"]
        repair_mask_counts.append(sum(int(mask.sum().item()) for mask in masks.values()))
        return {"step": kwargs["step"], "weights_masked": repair_mask_counts[-1]}

    monkeypatch.setattr(approx, "_prepare_model", fake_prepare_model)
    monkeypatch.setattr(approx, "_prepare_tokenizer", fake_prepare_tokenizer)
    monkeypatch.setattr(approx, "_load_records", fake_load_records)
    monkeypatch.setattr(approx, "evaluate_perplexity", fake_evaluate)
    monkeypatch.setattr(approx, "_wanda_scores_and_hessian_diagonal", fake_wanda)
    monkeypatch.setattr(approx, "_loss_gradient_repair_step_", fake_repair)

    summary = run_nm_global_pass_matrix_attribution_experiment(
        IterativeApproxPruneConfig(
            output_dir=tmp_path,
            model_name="tiny",
            max_examples=1,
            max_eval_examples=1,
            pruning_structure="2:4",
            structured_n=2,
            structured_m=4,
            structured_group_dim=1,
            repair_with_loss_gd=True,
        )
    )

    assert wanda_calls == 2
    assert repair_mask_counts == [4, 8]
    assert mask_counts_at_eval == [0, 2, 4, 4, 6, 8, 8]
    assert [row["pass_index"] for row in summary["rows"]] == [1, 1, 2, 2]
    assert [row["name"] for row in summary["rows"]] == [
        "first.weight",
        "second.weight",
        "first.weight",
        "second.weight",
    ]


def test_normalize_pruning_structure_accepts_2to4_aliases():
    assert normalize_pruning_structure("unstructured") == "unstructured"
    assert normalize_pruning_structure("2:4") == "nm"
    assert normalize_pruning_structure("semi_structured") == "nm"


def test_reapply_pruned_masks_keeps_linear_pruned_entries_zero_after_repair():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.ones(2, 3))
    masks = {"0.weight": torch.tensor([[True, False, False], [False, True, False]])}

    reapplied = reapply_pruned_masks_(model, masks)

    assert reapplied == 2
    assert model[0].weight.tolist() == [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]


def test_masked_gradient_step_updates_only_unpruned_matrix_weights():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False), torch.nn.LayerNorm(1))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.0, 2.0]]))
        model[1].weight.copy_(torch.tensor([3.0]))

    model[0].weight.grad = torch.tensor([[10.0, 4.0]])
    model[1].weight.grad = torch.tensor([5.0])
    masks = {"0.weight": torch.tensor([[True, False]])}

    summary = apply_masked_gradient_step_(
        model,
        pruned_masks=masks,
        learning_rate=0.25,
        step=1,
    )

    assert summary["format"] == "loss_gradient_descent"
    assert summary["updated_tensors"] == 1
    assert model[0].weight[0, 0].item() == 0.0
    assert torch.allclose(model[0].weight[0, 1], torch.tensor(1.0))
    assert model[1].weight.item() == 3.0


def test_masked_gradient_step_reapplies_zero_mask_even_if_weight_drifted():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[5.0, 2.0]]))

    model[0].weight.grad = torch.tensor([[0.0, 0.0]])
    masks = {"0.weight": torch.tensor([[True, False]])}

    apply_masked_gradient_step_(
        model,
        pruned_masks=masks,
        learning_rate=1.0,
        step=1,
    )

    assert model[0].weight.tolist() == [[0.0, 2.0]]
