from __future__ import annotations

import math

import pytest
import torch

from saliency.prune_eval import (
    PerplexityStats,
    apply_saliency_pruning_,
    apply_global_saliency_pruning_,
    apply_row_saliency_pruning_,
    lowest_saliency_mask,
    structured_2to4_stats,
    summarize_ppl_change,
)


def test_lowest_saliency_mask_selects_requested_fraction_with_stable_tie_break():
    saliency = torch.tensor([[0.4, 0.1], [0.2, 0.3]])

    mask = lowest_saliency_mask(saliency, fraction=0.5)

    assert mask.tolist() == [[False, True], [True, False]]


def test_lowest_saliency_mask_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        lowest_saliency_mask(torch.ones(2, 2), fraction=0.0)


def test_apply_saliency_pruning_zeroes_half_of_each_matrix_only():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.LayerNorm(2))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        model[0].bias.copy_(torch.tensor([5.0, 6.0]))
        model[1].weight.copy_(torch.tensor([7.0, 8.0]))

    saliency = {
        "0.weight": torch.tensor([[0.4, 0.1], [0.2, 0.3]]),
        "0.bias": torch.zeros(2),
        "1.weight": torch.zeros(2),
    }

    summary = apply_saliency_pruning_(model, saliency, fraction=0.5)

    assert model[0].weight.tolist() == [[1.0, 0.0], [0.0, 4.0]]
    assert model[0].bias.tolist() == [5.0, 6.0]
    assert model[1].weight.tolist() == [7.0, 8.0]
    assert summary["matrix_tensors_pruned"] == 1
    assert summary["weights_zeroed"] == 2
    assert summary["weights_seen"] == 4


def test_apply_row_saliency_pruning_zeroes_fraction_within_each_output_row():
    model = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))

    saliency = {
        "0.weight": torch.tensor(
            [
                [0.1, 0.2, 100.0, 101.0],
                [0.3, 0.4, 0.5, 0.6],
            ]
        )
    }

    summary = apply_row_saliency_pruning_(model, saliency, fraction=0.5)

    assert model[0].weight.tolist() == [[0.0, 0.0, 3.0, 4.0], [0.0, 0.0, 7.0, 8.0]]
    assert summary["pruning_scope"] == "per_output_row"
    assert summary["matrix_tensors_pruned"] == 1
    assert summary["weights_zeroed"] == 4
    assert summary["weights_seen"] == 8


def test_apply_global_saliency_pruning_zeroes_lowest_scores_across_matrices():
    model = torch.nn.ModuleDict(
        {
            "a": torch.nn.Linear(2, 2, bias=False),
            "b": torch.nn.Linear(2, 1, bias=False),
        }
    )
    with torch.no_grad():
        model["a"].weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        model["b"].weight.copy_(torch.tensor([[5.0, 6.0]]))

    saliency = {
        "a.weight": torch.tensor([[0.6, 0.1], [0.2, 0.5]]),
        "b.weight": torch.tensor([[0.3, 0.4]]),
    }

    summary = apply_global_saliency_pruning_(model, saliency, fraction=0.5)

    assert model["a"].weight.tolist() == [[1.0, 0.0], [0.0, 4.0]]
    assert model["b"].weight.tolist() == [[0.0, 6.0]]
    assert summary["matrix_tensors_pruned"] == 2
    assert summary["weights_zeroed"] == 3
    assert summary["weights_seen"] == 6
    assert summary["actual_zero_fraction"] == 0.5


def test_summarize_ppl_change_reports_ratio_and_delta():
    baseline = PerplexityStats(loss_sum=20.0, supervised_tokens=10, num_batches=2, num_examples=4)
    pruned = PerplexityStats(loss_sum=30.0, supervised_tokens=10, num_batches=2, num_examples=4)

    summary = summarize_ppl_change(baseline, pruned)

    assert summary["baseline"]["loss_per_token"] == 2.0
    assert summary["pruned"]["loss_per_token"] == 3.0
    assert math.isclose(summary["baseline"]["perplexity"], math.exp(2.0))
    assert math.isclose(summary["delta_loss_per_token"], 1.0)
    assert math.isclose(summary["perplexity_ratio"], math.exp(1.0))


def test_structured_2to4_stats_reports_compliance_and_extra_zeros():
    mask = torch.tensor(
        [
            [True, False, False, False, True, True, False, False],
            [True, False, True, False, False, False, False, False],
        ]
    )

    stats = structured_2to4_stats(mask, group_dim=1)

    assert stats["groups"] == 4
    assert stats["compliant_groups"] == 2
    assert stats["compliant_group_fraction"] == 0.5
    assert stats["existing_zeros"] == 5
    assert stats["extra_zeros_needed"] == 3
    assert stats["target_zero_fraction"] == 0.5
