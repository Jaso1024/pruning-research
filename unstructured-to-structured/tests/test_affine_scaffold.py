from __future__ import annotations

import json

import torch

from saliency.affine_scaffold import (
    AffineScaffoldConfig,
    AffinePruneEvalConfig,
    affine_pruning_scores,
    factor_affine_weight,
    generate_random_classification_pairs,
    generate_random_pairs,
    LayeredAffinePruneEvalConfig,
    LayeredAffineScaffoldConfig,
    run_layered_affine_prune_eval,
    run_layered_affine_scaffold,
    run_affine_scaffold,
    run_affine_prune_eval,
    solve_affine_least_squares,
)


def test_generate_random_pairs_is_fixed_by_seed():
    first = generate_random_pairs(num_pairs=8, input_dim=3, output_dim=2, seed=123)
    second = generate_random_pairs(num_pairs=8, input_dim=3, output_dim=2, seed=123)
    different = generate_random_pairs(num_pairs=8, input_dim=3, output_dim=2, seed=124)

    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.outputs, second.outputs)
    assert not torch.equal(first.inputs, different.inputs)
    assert first.inputs.shape == (8, 3)
    assert first.outputs.shape == (8, 2)


def test_generate_random_classification_pairs_is_fixed_by_seed_and_one_hot():
    first = generate_random_classification_pairs(num_pairs=10, input_dim=4, num_classes=6, seed=17)
    second = generate_random_classification_pairs(num_pairs=10, input_dim=4, num_classes=6, seed=17)

    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.outputs, second.outputs)
    assert torch.equal(first.labels, second.labels)
    assert first.outputs.shape == (10, 6)
    assert first.labels.shape == (10,)
    assert torch.allclose(first.outputs.sum(dim=1), torch.ones(10))
    assert torch.equal(first.outputs.argmax(dim=1), first.labels)


def test_solve_affine_least_squares_recovers_known_map():
    inputs = torch.tensor(
        [
            [-2.0, -1.0],
            [-1.0, 0.5],
            [0.0, 1.0],
            [1.0, -0.5],
            [2.0, 1.5],
        ],
        dtype=torch.float32,
    )
    expected_a = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float32)
    expected_b = torch.tensor([0.25, -1.5], dtype=torch.float32)
    outputs = inputs.matmul(expected_a.t()).add(expected_b)

    solution = solve_affine_least_squares(inputs, outputs)

    assert torch.allclose(solution.a, expected_a, atol=1e-5, rtol=1e-5)
    assert torch.allclose(solution.b, expected_b, atol=1e-5, rtol=1e-5)
    assert solution.mse < 1e-10


def test_run_affine_scaffold_saves_pairs_weights_and_summary(tmp_path):
    summary = run_affine_scaffold(
        AffineScaffoldConfig(output_dir=tmp_path, num_pairs=16, input_dim=4, output_dim=3, seed=7)
    )

    pairs = torch.load(tmp_path / "pairs.pt", weights_only=False)
    weights = torch.load(tmp_path / "weights.pt", weights_only=False)
    saved_summary = json.loads((tmp_path / "summary.json").read_text())

    assert pairs["inputs"].shape == (16, 4)
    assert pairs["outputs"].shape == (16, 3)
    assert weights["A"].shape == (3, 4)
    assert weights["b"].shape == (3,)
    assert saved_summary == summary
    assert summary["metadata"]["seed"] == 7
    assert summary["fit"]["rank"] == 5


def test_run_affine_scaffold_classification_saves_labels_and_softmax_metrics(tmp_path):
    summary = run_affine_scaffold(
        AffineScaffoldConfig(
            output_dir=tmp_path,
            num_pairs=24,
            input_dim=5,
            output_dim=7,
            seed=12,
            task="classification",
        )
    )

    pairs = torch.load(tmp_path / "pairs.pt", weights_only=False)
    weights = torch.load(tmp_path / "weights.pt", weights_only=False)

    assert pairs["inputs"].shape == (24, 5)
    assert pairs["outputs"].shape == (24, 7)
    assert pairs["labels"].shape == (24,)
    assert weights["A"].shape == (7, 5)
    assert weights["b"].shape == (7,)
    assert summary["metadata"]["task"] == "classification"
    assert "cross_entropy" in summary["fit"]
    assert "accuracy" in summary["fit"]


def test_affine_pruning_scores_include_expected_methods():
    inputs = torch.tensor([[-1.0, 2.0], [0.5, -0.5], [2.0, 1.0]], dtype=torch.float32)
    a = torch.tensor([[1.0, -2.0], [0.25, 3.0]], dtype=torch.float32)
    b = torch.tensor([0.1, -0.2], dtype=torch.float32)
    outputs = inputs.matmul(a.t()).add(b)

    scores = affine_pruning_scores(inputs, outputs, a, b, seed=5)

    assert set(scores) == {"random", "magnitude", "wanda", "squared_wanda", "exact_weight_loss", "exact_grad"}
    assert all(tuple(score.shape) == tuple(a.shape) for score in scores.values())
    assert torch.equal(scores["magnitude"], a.abs())
    assert torch.all(scores["exact_weight_loss"] >= -1e-6)


def test_run_affine_prune_eval_writes_requested_pruning_grid(tmp_path):
    scaffold_dir = tmp_path / "scaffold"
    eval_dir = tmp_path / "eval"
    run_affine_scaffold(AffineScaffoldConfig(output_dir=scaffold_dir, num_pairs=32, input_dim=5, output_dim=3, seed=9))

    summary = run_affine_prune_eval(
        AffinePruneEvalConfig(
            input_dir=scaffold_dir,
            output_dir=eval_dir,
            methods=("magnitude", "wanda", "exact_grad"),
            prune_fractions=(0.05, 0.1, 0.25, 0.5),
        )
    )

    rows = [json.loads(line) for line in (eval_dir / "prune_results.jsonl").read_text().splitlines()]
    saved_summary = json.loads((eval_dir / "summary.json").read_text())

    assert len(rows) == 12
    assert saved_summary == summary
    assert {(row["method"], row["prune_fraction"]) for row in rows} == {
        (method, fraction)
        for method in ("magnitude", "wanda", "exact_grad")
        for fraction in (0.05, 0.1, 0.25, 0.5)
    }
    assert all(row["weights_seen"] == 15 for row in rows)
    assert summary["metadata"]["bias_pruned"] is False


def test_run_affine_prune_eval_classification_reports_softmax_metrics(tmp_path):
    scaffold_dir = tmp_path / "classification"
    eval_dir = tmp_path / "eval"
    run_affine_scaffold(
        AffineScaffoldConfig(
            output_dir=scaffold_dir,
            num_pairs=32,
            input_dim=6,
            output_dim=8,
            seed=21,
            task="classification",
        )
    )

    summary = run_affine_prune_eval(
        AffinePruneEvalConfig(
            input_dir=scaffold_dir,
            output_dir=eval_dir,
            methods=("magnitude", "squared_wanda"),
            prune_fractions=(0.1, 0.5),
        )
    )

    rows = [json.loads(line) for line in (eval_dir / "prune_results.jsonl").read_text().splitlines()]

    assert summary["metadata"]["task"] == "classification"
    assert "cross_entropy" in summary["baseline"]
    assert "accuracy" in summary["baseline"]
    assert len(rows) == 4
    assert all("pruned_cross_entropy" in row for row in rows)
    assert all("pruned_accuracy" in row for row in rows)


def test_run_affine_prune_eval_includes_compensated_pruning_methods(tmp_path):
    scaffold_dir = tmp_path / "classification"
    eval_dir = tmp_path / "eval"
    run_affine_scaffold(
        AffineScaffoldConfig(
            output_dir=scaffold_dir,
            num_pairs=48,
            input_dim=8,
            output_dim=6,
            seed=31,
            task="classification",
        )
    )

    summary = run_affine_prune_eval(
        AffinePruneEvalConfig(
            input_dir=scaffold_dir,
            output_dir=eval_dir,
            methods=("gptq", "qronos"),
            prune_fractions=(0.25,),
            pruning_scope="global",
            use_activation_order=False,
            blocksize=4,
            num_blocks=2,
        )
    )

    rows = [json.loads(line) for line in (eval_dir / "prune_results.jsonl").read_text().splitlines()]

    assert [row["method"] for row in rows] == ["gptq", "qronos"]
    assert all(row["weights_seen"] == 48 for row in rows)
    assert all(row["weights_zeroed"] == 12 for row in rows)
    assert all(row["compensated"] is True for row in rows)
    assert all("compensation" in row for row in rows)
    assert summary["metadata"]["bias_pruned"] is False


def test_factor_affine_weight_recovers_full_rank_map_with_wide_hidden_layers():
    a = torch.tensor([[2.0, -1.0, 0.5], [0.25, 3.0, -2.0]], dtype=torch.float32)

    weights = factor_affine_weight(a, (3, 4, 5, 2))
    product = weights[-1]
    for weight in reversed(weights[:-1]):
        product = product.matmul(weight)

    assert [tuple(weight.shape) for weight in weights] == [(4, 3), (5, 4), (2, 5)]
    assert torch.allclose(product, a, atol=1e-5, rtol=1e-5)


def test_run_layered_affine_scaffold_saves_factored_network(tmp_path):
    summary = run_layered_affine_scaffold(
        LayeredAffineScaffoldConfig(
            output_dir=tmp_path,
            layer_dims=(5, 7, 4),
            num_pairs=32,
            seed=41,
        )
    )

    network = torch.load(tmp_path / "network.pt", map_location="cpu", weights_only=False)
    pairs = torch.load(tmp_path / "pairs.pt", map_location="cpu", weights_only=False)

    assert pairs["inputs"].shape == (32, 5)
    assert pairs["outputs"].shape == (32, 4)
    assert [tuple(weight.shape) for weight in network["weights"]] == [(7, 5), (4, 7)]
    assert network["bias"].shape == (4,)
    assert summary["metadata"]["layer_dims"] == [5, 7, 4]
    assert "cross_entropy" in summary["fit"]


def test_run_layered_affine_prune_eval_reports_layered_grid(tmp_path):
    scaffold_dir = tmp_path / "layered"
    eval_dir = tmp_path / "eval"
    run_layered_affine_scaffold(
        LayeredAffineScaffoldConfig(
            output_dir=scaffold_dir,
            layer_dims=(6, 8, 5),
            num_pairs=40,
            seed=52,
        )
    )

    summary = run_layered_affine_prune_eval(
        LayeredAffinePruneEvalConfig(
            input_dir=scaffold_dir,
            output_dir=eval_dir,
            methods=("magnitude", "wanda", "gptq"),
            prune_fractions=(0.25,),
            pruning_scope="global",
            use_activation_order=False,
            blocksize=4,
        )
    )

    rows = [json.loads(line) for line in (eval_dir / "prune_results.jsonl").read_text().splitlines()]

    assert len(rows) == 3
    assert {row["method"] for row in rows} == {"magnitude", "wanda", "gptq"}
    assert all(row["layers_seen"] == 2 for row in rows)
    assert all(row["weights_zeroed"] == 22 for row in rows)
    assert all("pruned_cross_entropy" in row for row in rows)
    assert summary["baseline"]["layers"] == 2
