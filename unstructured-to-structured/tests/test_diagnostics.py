from __future__ import annotations

import math

import torch

from saliency.diagnostics import (
    score_pair_diagnostics,
    score_tensor_summary,
    tensor_weight_summary,
)


def test_tensor_weight_summary_reports_tail_and_quantiles():
    tensor = torch.tensor([[0.0, -1.0, 2.0], [3.0, -4.0, 100.0]])

    summary = tensor_weight_summary("layer.weight", tensor, bottom_fraction=0.5)

    assert summary["name"] == "layer.weight"
    assert summary["numel"] == 6
    assert math.isclose(summary["zero_fraction"], 1 / 6, rel_tol=1e-6, abs_tol=1e-6)
    assert summary["abs_max"] == 100.0
    assert math.isclose(summary["bottom_abs_l1_fraction"], (0.0 + 1.0 + 2.0) / 110.0, rel_tol=1e-6)
    assert summary["top_abs_l2_fraction"] > 0.99
    assert summary["abs_q50"] == 2.5


def test_score_tensor_summary_tracks_score_weight_alignment():
    weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    score = torch.tensor([[4.0, 3.0], [2.0, 1.0]])

    summary = score_tensor_summary("layer.weight", score, weight, prune_fraction=0.5)

    assert summary["name"] == "layer.weight"
    assert summary["score_pruned_mean_abs_weight"] == 3.5
    assert summary["score_kept_mean_abs_weight"] == 1.5
    assert math.isclose(summary["score_pruned_l2_weight_fraction"], 25.0 / 30.0, rel_tol=1e-6, abs_tol=1e-6)
    assert summary["spearman_score_abs_weight"] < 0.0


def test_score_pair_diagnostics_reports_overlap_and_rank_correlation():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[4.0, 3.0], [2.0, 1.0]])

    same_scope = score_pair_diagnostics("a", "b", a, b, prune_fraction=0.5, pruning_scope="per_matrix")
    row_scope = score_pair_diagnostics("a", "b", a, b, prune_fraction=0.5, pruning_scope="per_output_row")

    assert same_scope["mask_jaccard"] == 0.0
    assert same_scope["mask_overlap_fraction"] == 0.0
    assert math.isclose(same_scope["spearman"], -1.0)
    assert row_scope["mask_jaccard"] == 0.0
