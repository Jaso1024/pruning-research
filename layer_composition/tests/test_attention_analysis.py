import json
from pathlib import Path

import pytest
import torch

import layer_distill.attention_analysis as attention_analysis
from layer_distill.attention_analysis import (
    AttentionAnalysisConfig,
    attention_distance,
    compare_cross_model,
    compare_cross_model_head_matching,
    compare_within_model_head_matching,
    compare_cross_model_exponential_combinations,
    compare_cross_model_linear_combinations,
    compare_cross_model_wasserstein_combinations,
    compare_within_model,
    fit_head_exponential_combinations,
    fit_attention_basis,
    fit_attention_basis_nmf,
    fit_head_linear_combinations,
    fit_head_wasserstein_combinations,
    head_summary,
    run_attention_combo_analysis,
    summarize_attention_run,
    _model_pair_slug,
    _parse_model_pairs,
)


def test_attention_distance_identical_distributions_are_zero_distance():
    a = torch.tensor([[[[1.0, 0.0], [0.25, 0.75]]]])

    metrics = attention_distance(a, a)

    assert metrics["jsd"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["tv"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["cosine_distance"] == pytest.approx(0.0, abs=1e-7)


def test_attention_distance_separates_disjoint_distributions():
    a = torch.tensor([[[[1.0, 0.0]]]])
    b = torch.tensor([[[[0.0, 1.0]]]])

    metrics = attention_distance(a, b)

    assert metrics["jsd"] > 0.6
    assert metrics["tv"] == pytest.approx(1.0)
    assert metrics["cosine_distance"] == pytest.approx(1.0)


def test_head_summary_reports_entropy_and_attention_mass():
    attn = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.2, 0.3, 0.5],
            ]
        ]
    )

    summary = head_summary(attn, local_window=1)

    assert summary["entropy"] > 0.0
    assert 0.0 <= summary["normalized_entropy"] <= 1.0
    assert summary["max_prob"] == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert summary["diagonal_mass"] == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert summary["local_1_mass"] == pytest.approx((1.0 + 0.5 + 0.5) / 3)


def test_compare_within_model_builds_head_pairs():
    layer = torch.zeros(1, 2, 2, 2)
    layer[0, 0] = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    layer[0, 1] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    rows = compare_within_model("m", [layer])

    assert len(rows) == 1
    assert rows[0]["model"] == "m"
    assert rows[0]["layer"] == 0
    assert rows[0]["head_a"] == 0
    assert rows[0]["head_b"] == 1
    assert rows[0]["jsd"] > 0.0


def test_compare_within_model_head_matching_adds_directional_best_matches():
    layer = torch.zeros(1, 3, 2, 2)
    layer[0, 0] = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
    layer[0, 1] = torch.tensor([[1.0, 0.0], [0.8, 0.2]])
    layer[0, 2] = torch.tensor([[0.0, 1.0], [0.1, 0.9]])

    rows = compare_within_model_head_matching("m", [layer])
    all_pairs = [row for row in rows if row["match_type"] == "all_pairs"]
    best = [row for row in rows if row["match_type"] == "head_to_best_head"]

    assert {(row["head_a"], row["head_b"]) for row in all_pairs} == {(0, 1), (0, 2), (1, 2)}
    assert {(row["head"], row["matched_head"]) for row in best} == {(0, 1), (1, 0), (2, 1)}
    assert all(row["head"] != row["matched_head"] for row in best)
    assert all(row["model"] == "m" and row["layer"] == 0 for row in rows)


def test_compare_within_model_head_matching_builds_min_cost_one_to_one_assignment():
    layer = torch.zeros(1, 4, 2, 2)
    layer[0, 0] = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
    layer[0, 1] = torch.tensor([[1.0, 0.0], [0.8, 0.2]])
    layer[0, 2] = torch.tensor([[0.0, 1.0], [0.1, 0.9]])
    layer[0, 3] = torch.tensor([[0.0, 1.0], [0.2, 0.8]])

    rows = compare_within_model_head_matching("m", [layer])
    one_to_one = [row for row in rows if row["match_type"] == "one_to_one"]

    assert {(row["head"], row["matched_head"]) for row in one_to_one} == {(0, 1), (1, 0), (2, 3), (3, 2)}
    assert all(row["head"] != row["matched_head"] for row in one_to_one)
    assert all(row["jsd"] < 0.01 for row in one_to_one)


def test_compare_cross_model_matches_same_layer_and_head():
    big_layer = torch.zeros(1, 2, 2, 2)
    small_layer = torch.zeros(1, 2, 2, 2)
    big_layer[0, 0] = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    big_layer[0, 1] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    small_layer.copy_(big_layer)

    rows = compare_cross_model("small", [small_layer], "big", [big_layer])

    assert len(rows) == 2
    assert {row["head"] for row in rows} == {0, 1}
    assert all(row["jsd"] == pytest.approx(0.0, abs=1e-7) for row in rows)


def test_compare_cross_model_head_matching_finds_permuted_heads():
    big_layer = torch.zeros(1, 2, 2, 2)
    small_layer = torch.zeros(1, 2, 2, 2)
    small_layer[0, 0] = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    small_layer[0, 1] = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
    big_layer[0, 0] = small_layer[0, 1]
    big_layer[0, 1] = small_layer[0, 0]

    rows = compare_cross_model_head_matching("small", [small_layer], "big", [big_layer])
    best = [row for row in rows if row["match_type"] == "small_to_best_big"]
    one_to_one = [row for row in rows if row["match_type"] == "one_to_one"]

    assert {(row["small_head"], row["big_head"]) for row in best} == {(0, 1), (1, 0)}
    assert all(row["jsd"] == pytest.approx(0.0, abs=1e-7) for row in best)
    assert {(row["small_head"], row["big_head"]) for row in one_to_one} == {(0, 1), (1, 0)}


def test_compare_cross_model_head_matching_handles_more_big_heads():
    small_layer = torch.zeros(1, 2, 2, 2)
    big_layer = torch.zeros(1, 3, 2, 2)
    small_layer[0, 0] = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    small_layer[0, 1] = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
    big_layer[0, 0] = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    big_layer[0, 1] = small_layer[0, 1]
    big_layer[0, 2] = small_layer[0, 0]

    rows = compare_cross_model_head_matching("small", [small_layer], "big", [big_layer])
    one_to_one = [row for row in rows if row["match_type"] == "one_to_one"]

    assert {(row["small_head"], row["big_head"]) for row in one_to_one} == {(0, 2), (1, 1)}
    assert all(row["jsd"] == pytest.approx(0.0, abs=1e-7) for row in one_to_one)


def test_compare_cross_model_head_matching_handles_more_small_heads():
    small_layer = torch.zeros(1, 3, 2, 2)
    big_layer = torch.zeros(1, 2, 2, 2)
    small_layer[0, 0] = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    small_layer[0, 1] = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
    small_layer[0, 2] = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    big_layer[0, 0] = small_layer[0, 2]
    big_layer[0, 1] = small_layer[0, 1]

    rows = compare_cross_model_head_matching("small", [small_layer], "big", [big_layer])
    one_to_one = [row for row in rows if row["match_type"] == "one_to_one"]

    assert {(row["small_head"], row["big_head"]) for row in one_to_one} == {(2, 0), (1, 1)}
    assert all(row["jsd"] == pytest.approx(0.0, abs=1e-7) for row in one_to_one)


def test_fit_head_linear_combinations_reconstructs_convex_mixture():
    small_layer = torch.zeros(1, 2, 2, 2)
    small_layer[0, 0] = torch.tensor([[1.0, 0.0], [0.8, 0.2]])
    small_layer[0, 1] = torch.tensor([[0.0, 1.0], [0.2, 0.8]])
    target = 0.25 * small_layer[:, 0] + 0.75 * small_layer[:, 1]
    big_layer = target.unsqueeze(1)

    result = fit_head_linear_combinations(small_layer, big_layer, steps=250, lr=0.25)

    assert result["metrics"][0]["jsd"] < 1e-4
    assert result["weights"][0][0] == pytest.approx(0.25, abs=0.04)
    assert result["weights"][0][1] == pytest.approx(0.75, abs=0.04)


def test_fit_head_exponential_combinations_reconstructs_geometric_mixture():
    small_layer = torch.zeros(1, 2, 2, 3)
    small_layer[0, 0] = torch.tensor([[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]])
    small_layer[0, 1] = torch.tensor([[0.10, 0.30, 0.60], [0.60, 0.25, 0.15]])
    weights = torch.tensor([0.30, 0.70])
    target_logits = torch.einsum("h,bhqk->bqk", weights, small_layer.log())
    target = torch.softmax(target_logits, dim=-1)
    big_layer = target.unsqueeze(1)

    result = fit_head_exponential_combinations(small_layer, big_layer, steps=350, lr=0.2)

    assert result["metrics"][0]["jsd"] < 1e-5
    assert result["weights"][0][0] == pytest.approx(0.30, abs=0.05)
    assert result["weights"][0][1] == pytest.approx(0.70, abs=0.05)


def test_fit_head_wasserstein_combinations_reconstructs_position_mixture():
    small_layer = torch.zeros(1, 2, 2, 3)
    small_layer[:, 0, :, 0] = 1.0
    small_layer[:, 1, :, 2] = 1.0
    target = torch.zeros(1, 2, 3)
    target[:, :, 1] = 0.5
    target[:, :, 2] = 0.5
    big_layer = target.unsqueeze(1)

    result = fit_head_wasserstein_combinations(small_layer, big_layer, steps=350, lr=0.2)

    assert result["metrics"][0]["jsd"] < 1e-5
    assert result["weights"][0][0] == pytest.approx(0.25, abs=0.05)
    assert result["weights"][0][1] == pytest.approx(0.75, abs=0.05)


def test_fit_attention_basis_reconstructs_heads_from_two_latent_distributions():
    basis_a = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.75, 0.25, 0.0],
            [0.50, 0.25, 0.25],
        ]
    )
    basis_b = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.10, 0.70, 0.20],
            [0.20, 0.30, 0.50],
        ]
    )
    weights = torch.tensor(
        [
            [1.00, 0.00],
            [0.70, 0.30],
            [0.25, 0.75],
            [0.00, 1.00],
        ]
    )
    layer = torch.einsum("hb,bqk->hqk", weights, torch.stack([basis_a, basis_b])).unsqueeze(0)

    one_basis = fit_attention_basis([layer], basis_size=1, steps=120, lr=0.25, seed=0)
    two_basis = fit_attention_basis([layer], basis_size=2, steps=350, lr=0.25, seed=0)

    assert two_basis["mean_jsd"] < one_basis["mean_jsd"] * 0.2
    assert two_basis["max_jsd"] < 0.02
    assert two_basis["basis_size"] == 2
    assert len(two_basis["per_head_jsd"]) == 4


def test_fit_attention_basis_nmf_runs_fast_nonnegative_basis_fit():
    pytest.importorskip("sklearn")
    layer = torch.zeros(1, 4, 2, 3)
    layer[0, 0] = torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0]])
    layer[0, 1] = torch.tensor([[0.0, 1.0, 0.0], [0.1, 0.8, 0.1]])
    layer[0, 2] = 0.6 * layer[0, 0] + 0.4 * layer[0, 1]
    layer[0, 3] = 0.2 * layer[0, 0] + 0.8 * layer[0, 1]

    result = fit_attention_basis_nmf([layer], basis_size=2, steps=100, seed=0)

    assert result["basis_size"] == 2
    assert result["mean_jsd"] < 0.03
    assert len(result["coefficient_sums"]) == 4


def test_compare_cross_model_linear_combinations_emits_big_head_rows():
    small_layer = torch.zeros(1, 2, 2, 2)
    big_layer = torch.zeros(1, 3, 2, 2)
    small_layer[0, 0] = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    small_layer[0, 1] = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
    big_layer[0, 0] = small_layer[0, 0]
    big_layer[0, 1] = small_layer[0, 1]
    big_layer[0, 2] = 0.5 * small_layer[0, 0] + 0.5 * small_layer[0, 1]

    rows = compare_cross_model_linear_combinations("small", [small_layer], "big", [big_layer], steps=200, lr=0.25)

    assert len(rows) == 3
    assert {row["big_head"] for row in rows} == {0, 1, 2}
    assert all(row["jsd"] < 1e-4 for row in rows)
    assert rows[2]["top_small_head"] in {0, 1}


def test_compare_cross_model_wasserstein_combinations_emits_big_head_rows():
    small_layer = torch.zeros(1, 2, 2, 3)
    big_layer = torch.zeros(1, 3, 2, 3)
    small_layer[:, 0, :, 0] = 1.0
    small_layer[:, 1, :, 2] = 1.0
    big_layer[:, 0] = small_layer[:, 0]
    big_layer[:, 1] = small_layer[:, 1]
    big_layer[:, 2, :, 1] = 0.5
    big_layer[:, 2, :, 2] = 0.5

    rows = compare_cross_model_wasserstein_combinations("small", [small_layer], "big", [big_layer], steps=300, lr=0.2)

    assert len(rows) == 3
    assert {row["big_head"] for row in rows} == {0, 1, 2}
    assert all(row["method"] == "wasserstein_distribution_combo" for row in rows)
    assert all(row["jsd"] < 1e-4 for row in rows)
    assert rows[2]["top_small_head"] == 1


def test_compare_cross_model_exponential_combinations_emits_big_head_rows():
    small_layer = torch.zeros(1, 2, 2, 3)
    big_layer = torch.zeros(1, 3, 2, 3)
    small_layer[0, 0] = torch.tensor([[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]])
    small_layer[0, 1] = torch.tensor([[0.10, 0.30, 0.60], [0.60, 0.25, 0.15]])
    big_layer[0, 0] = small_layer[0, 0]
    big_layer[0, 1] = small_layer[0, 1]
    big_layer[0, 2] = torch.softmax(0.5 * small_layer[0, 0].log() + 0.5 * small_layer[0, 1].log(), dim=-1)

    rows = compare_cross_model_exponential_combinations("small", [small_layer], "big", [big_layer], steps=250, lr=0.2)

    assert len(rows) == 3
    assert {row["big_head"] for row in rows} == {0, 1, 2}
    assert all(row["method"] == "exponential_distribution_combo" for row in rows)
    assert all(row["jsd"] < 1e-4 for row in rows)
    assert rows[2]["top_small_head"] in {0, 1}


def test_run_attention_combo_analysis_dispatches_exponential(monkeypatch, tmp_path: Path):
    config = AttentionAnalysisConfig(output_dir=tmp_path, prompts=("hello",), max_length=16)
    calls = []

    def fake_run(received_config, *, fit_steps, fit_lr):
        calls.append((received_config, fit_steps, fit_lr))
        return {"ok": True}

    monkeypatch.setattr(attention_analysis, "run_attention_exponential_combo_analysis", fake_run)

    summary = run_attention_combo_analysis(config, combo_method="exponential", fit_steps=7, fit_lr=0.3)

    assert summary == {"ok": True}
    assert calls == [(config, 7, 0.3)]


def test_run_attention_combo_analysis_rejects_unknown_method(tmp_path: Path):
    config = AttentionAnalysisConfig(output_dir=tmp_path, prompts=("hello",), max_length=16)

    with pytest.raises(ValueError, match="combo_method"):
        run_attention_combo_analysis(config, combo_method="bad")


def test_attention_analysis_config_validates_inputs(tmp_path: Path):
    config = AttentionAnalysisConfig(output_dir=tmp_path, prompts=("hello",), max_length=16)
    assert config.small_model == "EleutherAI/pythia-31m"
    assert config.big_model == "EleutherAI/pythia-70m"

    with pytest.raises(ValueError):
        AttentionAnalysisConfig(output_dir=tmp_path, prompts=(), max_length=16)
    with pytest.raises(ValueError):
        AttentionAnalysisConfig(output_dir=tmp_path, prompts=("hello",), max_length=1)
    with pytest.raises(ValueError):
        AttentionAnalysisConfig(output_dir=tmp_path, prompts=("hello",), dtype="bad")


def test_parse_model_pairs_and_slug():
    pairs = _parse_model_pairs("EleutherAI/pythia-160m>EleutherAI/pythia-410m, a > b")

    assert pairs == (("EleutherAI/pythia-160m", "EleutherAI/pythia-410m"), ("a", "b"))
    assert _model_pair_slug("EleutherAI/pythia-160m", "EleutherAI/pythia-410m") == "pythia-160m_to_pythia-410m"

    with pytest.raises(ValueError):
        _parse_model_pairs("missing_separator")


def test_summarize_attention_run_picks_lowest_cross_model_jsd(tmp_path: Path):
    (tmp_path / "cross_model.jsonl").write_text(
        json.dumps({"layer": 0, "head": 0, "jsd": 0.2}) + "\n"
        + json.dumps({"layer": 1, "head": 1, "jsd": 0.1}) + "\n"
    )
    (tmp_path / "cross_head_matching.jsonl").write_text(
        json.dumps({"layer": 0, "match_type": "small_to_best_big", "small_head": 0, "big_head": 1, "jsd": 0.05}) + "\n"
    )
    (tmp_path / "within_model.jsonl").write_text(
        json.dumps({"model": "m", "layer": 0, "match_type": "all_pairs", "head_a": 0, "head_b": 1, "jsd": 0.3}) + "\n"
        + json.dumps({"model": "m", "layer": 0, "match_type": "head_to_best_head", "head": 0, "matched_head": 1, "jsd": 0.2}) + "\n"
    )

    summary = summarize_attention_run(tmp_path)

    assert summary["best_cross_model"]["layer"] == 1
    assert summary["best_cross_model"]["head"] == 1
    assert summary["best_cross_head_match"]["small_head"] == 0
    assert summary["closest_within_model_pair"]["head_a"] == 0
    assert summary["best_within_head_match"]["head"] == 0
    assert summary["cross_model_count"] == 2
    assert summary["cross_head_matching_count"] == 1
    assert summary["within_model_count"] == 2
