from pathlib import Path

import pytest
import torch

from layer_distill.sparse24 import (
    Sparse24EvalConfig,
    Sparse24GreedyLayerConfig,
    _append_jsonl_record,
    _read_jsonl_records,
    _restore_module_weights,
    _snapshot_module_weights,
    _trim_resumed_candidate_records,
    collect_linear_inputs,
    find_prunable_linear_names,
    group_prunable_linear_names_by_layer,
    sparse24_mcts_backpropagate,
    sparse24_mcts_best_child,
    sparse24_mcts_select_rollout,
    sparse24_state_name,
    sparse24_worker_cache_key,
    read_sparse24_state_records,
    sparsify_weight_2_4,
    structured_n_m_mask,
    structured_2_4_mask,
)


def test_sparse24_jsonl_helpers_append_and_read(tmp_path: Path):
    path = tmp_path / "records.jsonl"

    _append_jsonl_record(path, {"round": 1, "layer": 10})
    _append_jsonl_record(path, {"round": 2, "layer": 14})
    path.write_text(path.read_text() + "\n")

    assert _read_jsonl_records(path) == [{"layer": 10, "round": 1}, {"layer": 14, "round": 2}]


def test_sparse24_jsonl_helpers_missing_file(tmp_path: Path):
    assert _read_jsonl_records(tmp_path / "missing.jsonl") == []


def test_trim_resumed_candidate_records_drops_stale_completed_round_tail():
    candidates = [
        {"round": 1, "candidate_layer_index": 0},
        {"round": 1, "candidate_layer_index": 1},
        {"round": 1, "candidate_layer_index": 2},
        {"round": 2, "candidate_layer_index": 3},
    ]
    rounds = [{"round": 1, "candidate_count": 2}]

    trimmed = _trim_resumed_candidate_records(candidates, rounds)

    assert trimmed == [
        {"round": 1, "candidate_layer_index": 0},
        {"round": 1, "candidate_layer_index": 1},
        {"round": 2, "candidate_layer_index": 3},
    ]


def test_sparse24_mcts_backpropagates_and_selects_lowest_mean_ppl():
    stats: dict[tuple[int, ...], dict[str, float]] = {}
    sparse24_mcts_backpropagate(stats, path=[(), (1,)], ppl=30.0)
    sparse24_mcts_backpropagate(stats, path=[(), (2,)], ppl=25.0)
    sparse24_mcts_backpropagate(stats, path=[(), (1,)], ppl=20.0)

    assert stats[()]["visits"] == 3.0
    assert sparse24_mcts_best_child(root=(), layer_indices=(1, 2, 3), stats=stats) == (1,)


def test_sparse24_mcts_select_rollout_extends_root_without_duplicates():
    rng = __import__("random").Random(0)
    terminal, path = sparse24_mcts_select_rollout(
        root=(2,),
        layer_indices=(0, 1, 2, 3),
        stats={(2,): {"visits": 1.0, "value_sum": -10.0}},
        rollout_depth=3,
        exploration=1.4,
        rng=rng,
    )

    assert terminal[:1] == (2,)
    assert len(terminal) == 4
    assert len(set(terminal)) == len(terminal)
    assert path[0] == (2,)
    assert path[-1] == terminal


def test_sparse24_state_resume_reads_matching_committed_summaries(tmp_path: Path):
    state = (14, 13)
    state_dir = tmp_path / sparse24_state_name(state)
    state_dir.mkdir(parents=True)
    (state_dir / "summary.json").write_text(
        '{"layer_indices":[14,13],"loss":1.5,"ppl":4.0,"method":"magnitude"}\n'
    )

    records, missing = read_sparse24_state_records(
        tmp_path,
        [
            {
                "state_name": sparse24_state_name(state),
                "layer_indices": list(state),
                "path": [[], [14], [14, 13]],
            },
            {
                "state_name": sparse24_state_name((15,)),
                "layer_indices": [15],
            },
        ],
    )

    assert missing == [{"state_name": sparse24_state_name((15,)), "layer_indices": [15]}]
    assert records == [
        {
            "layer_indices": [14, 13],
            "loss": 1.5,
            "method": "magnitude",
            "mcts_path": [[], [14], [14, 13]],
            "ppl": 4.0,
            "state_name": "layers_14_13",
        }
    ]


def test_sparse24_state_resume_recomputes_mismatched_summary(tmp_path: Path):
    state_dir = tmp_path / sparse24_state_name((14, 13))
    state_dir.mkdir(parents=True)
    (state_dir / "summary.json").write_text('{"layer_indices":[13,14],"ppl":4.0}\n')

    records, missing = read_sparse24_state_records(
        tmp_path,
        [{"state_name": sparse24_state_name((14, 13)), "layer_indices": [14, 13]}],
    )

    assert records == []
    assert missing == [{"state_name": "layers_14_13", "layer_indices": [14, 13]}]


def test_sparse24_worker_cache_key_ignores_run_and_method_but_tracks_data_shape():
    base = {
        "run_name": "run_a",
        "batch_name": "magnitude_depth_01",
        "method": "magnitude",
        "model_name": "EleutherAI/pythia-1.4b",
        "dtype": "bf16",
        "calibration_steps": 4,
        "calibration_batch_size": 64,
        "calibration_seq_len": 256,
        "calibration_split": "train",
        "eval_steps": 16,
        "eval_batch_size": 64,
        "eval_seq_len": 256,
        "data_split": "test",
        "sparsity_m": 4,
    }
    same_load = dict(base, run_name="run_b", batch_name="wanda_depth_02", method="wanda")
    different_eval = dict(base, eval_steps=8)

    assert sparse24_worker_cache_key(base) == sparse24_worker_cache_key(same_load)
    assert sparse24_worker_cache_key(base) != sparse24_worker_cache_key(different_eval)


def test_structured_2_4_mask_keeps_two_per_group():
    weight = torch.tensor(
        [
            [1.0, -4.0, 2.0, 3.0, 0.5, -0.2, 9.0, 8.0],
            [7.0, 6.0, 5.0, 4.0, -1.0, -3.0, 2.0, 0.0],
        ]
    )

    mask = structured_2_4_mask(weight)

    assert mask.dtype == torch.bool
    assert mask.view(2, 2, 4).sum(dim=-1).tolist() == [[2, 2], [2, 2]]
    sparse = weight * mask
    assert sparse[0].tolist() == [0.0, -4.0, 0.0, 3.0, 0.0, -0.0, 9.0, 8.0]
    assert sparse[1].tolist() == [7.0, 6.0, 0.0, 0.0, 0.0, -3.0, 2.0, 0.0]


def test_structured_2_4_mask_rejects_non_multiple_of_four():
    with pytest.raises(ValueError):
        structured_2_4_mask(torch.randn(3, 6))


def test_structured_n_m_mask_supports_one_of_two():
    weight = torch.tensor([[1.0, -4.0, 2.0, 3.0, 0.5, -0.2]])

    mask = structured_n_m_mask(weight, n=1, m=2)

    assert mask.dtype == torch.bool
    assert mask.view(1, 3, 2).sum(dim=-1).tolist() == [[1, 1, 1]]
    assert (weight * mask).tolist() == [[0.0, -4.0, 0.0, 3.0, 0.5, -0.0]]


def test_sparsify_weight_2_4_preserves_shape_and_pattern():
    torch.manual_seed(0)
    weight = torch.randn(5, 8)
    x = torch.randn(32, 8)

    sparse, stats = sparsify_weight_2_4(weight, x_quant=x, method="sparsegpt")

    assert sparse.shape == weight.shape
    assert (sparse != 0).view(5, 2, 4).sum(dim=-1).tolist() == [[2, 2]] * 5
    assert stats["density"] == pytest.approx(0.5)
    assert stats["method"] == "sparsegpt"


def test_sparsify_weight_sparsegpt_uses_hessian_scaled_group_scores():
    weight = torch.tensor([[10.0, 9.0, 1.0, 1.0]])
    x = torch.eye(4).repeat(2, 1) * torch.tensor([1.0, 0.01, 20.0, 0.01])

    sparse, stats = sparsify_weight_2_4(weight, x_quant=x, method="sparsegpt", damp=0.0)

    assert sparse.tolist() == [[10.0, 0.0, 1.0, 0.0]]
    assert (sparse != 0).view(1, 1, 4).sum(dim=-1).eq(2).all()
    assert stats["method"] == "sparsegpt"


def test_sparsify_weight_wanda_uses_activation_scaled_scores():
    weight = torch.tensor([[10.0, 9.0, 1.0, 1.0]])
    x = torch.zeros(8, 4)
    x[:, 0] = 1.0
    x[:, 1] = 0.01
    x[:, 2] = 20.0
    x[:, 3] = 0.01

    sparse, stats = sparsify_weight_2_4(weight, x_quant=x, method="wanda")

    assert sparse.tolist() == [[10.0, 0.0, 1.0, 0.0]]
    assert (sparse != 0).view(1, 1, 4).sum(dim=-1).eq(2).all()
    assert stats["density"] == pytest.approx(0.5)
    assert stats["method"] == "wanda"


def test_sparsify_weight_rescomp_supports_one_of_two():
    torch.manual_seed(4)
    weight = torch.randn(4, 8)
    x_quant = torch.randn(16, 8)
    x_fp = x_quant + 0.05 * torch.randn(16, 8)

    sparse, stats = sparsify_weight_2_4(
        weight,
        x_quant=x_quant,
        x_fp=x_fp,
        method="gptaq-cae",
        sparsity_n=1,
        sparsity_m=2,
    )

    assert sparse.shape == weight.shape
    assert (sparse != 0).view(4, 4, 2).sum(dim=-1).eq(1).all()
    assert stats["density"] == pytest.approx(0.5)
    assert stats["sparsity_n"] == 1
    assert stats["sparsity_m"] == 2


def test_sparsify_weight_rescomp_accepts_fp_inputs():
    torch.manual_seed(1)
    weight = torch.randn(4, 8)
    x_quant = torch.randn(16, 8)
    x_fp = x_quant + 0.05 * torch.randn(16, 8)

    sparse, stats = sparsify_weight_2_4(weight, x_quant=x_quant, x_fp=x_fp, method="gptaq-cae")

    assert (sparse != 0).view(4, 2, 4).sum(dim=-1).eq(2).all()
    assert stats["method"] == "gptaq-cae"
    assert stats["has_fp_inputs"] is True


def test_sparsify_weight_rescomp_diag_accepts_fp_inputs():
    torch.manual_seed(11)
    weight = torch.randn(4, 8)
    x_quant = torch.randn(24, 8)
    x_fp = x_quant + 0.05 * torch.randn(24, 8)

    sparse, stats = sparsify_weight_2_4(weight, x_quant=x_quant, x_fp=x_fp, method="gptaq-cae-diag")

    assert (sparse != 0).view(4, 2, 4).sum(dim=-1).eq(2).all()
    assert stats["method"] == "gptaq-cae-diag"
    assert stats["has_fp_inputs"] is True
    assert stats["hessian_approx"] == "diagonal"


def test_sparsify_weight_rescomp_gd_keeps_pattern_and_reduces_masked_objective():
    torch.manual_seed(12)
    weight = torch.randn(5, 8)
    x_quant = torch.randn(48, 8)
    x_fp = x_quant + 0.1 * torch.randn(48, 8)
    hdiag = x_quant.pow(2).mean(dim=0).clamp_min(1e-8)
    mask = structured_n_m_mask(weight.pow(2) * hdiag.unsqueeze(0), n=2, m=4)
    initial = weight * mask
    target = x_fp @ weight.t()
    initial_loss = torch.nn.functional.mse_loss(x_quant @ initial.t(), target)

    sparse, stats = sparsify_weight_2_4(
        weight,
        x_quant=x_quant,
        x_fp=x_fp,
        method="gptaq-cae-gd",
        gd_steps=1,
        gd_lr=0.25,
        gd_chunk_tokens=16,
    )
    final_loss = torch.nn.functional.mse_loss(x_quant @ sparse.float().t(), target)

    assert (sparse != 0).view(5, 2, 4).sum(dim=-1).eq(2).all()
    assert stats["method"] == "gptaq-cae-gd"
    assert stats["has_fp_inputs"] is True
    assert stats["gd_steps"] == 1
    assert final_loss <= initial_loss


def test_sparsify_weight_rescomp_gd_rejects_multiple_local_steps():
    torch.manual_seed(13)
    weight = torch.randn(4, 8)
    x_quant = torch.randn(16, 8)
    x_fp = x_quant + 0.1 * torch.randn(16, 8)

    with pytest.raises(ValueError, match="single local GD step"):
        sparsify_weight_2_4(
            weight,
            x_quant=x_quant,
            x_fp=x_fp,
            method="gptaq-cae-gd",
            gd_steps=2,
        )


def test_sparsify_weight_gptq_cae_uses_compensation_aware_term_without_fp_inputs():
    torch.manual_seed(2)
    weight = torch.randn(4, 8)
    x_quant = torch.randn(20, 8)

    sparse, stats = sparsify_weight_2_4(weight, x_quant=x_quant, method="gptq-cae")

    assert (sparse != 0).view(4, 2, 4).sum(dim=-1).eq(2).all()
    assert stats["method"] == "gptq-cae"
    assert stats["has_fp_inputs"] is False


def test_sparsify_weight_qronos_accepts_mismatched_inputs():
    torch.manual_seed(3)
    weight = torch.randn(6, 8)
    x_quant = torch.randn(24, 8)
    x_fp = x_quant + 0.1 * torch.randn(24, 8)

    sparse, stats = sparsify_weight_2_4(weight, x_quant=x_quant, x_fp=x_fp, method="qronos")

    assert sparse.shape == weight.shape
    assert (sparse != 0).view(6, 2, 4).sum(dim=-1).eq(2).all()
    assert stats["method"] == "qronos"
    assert stats["has_fp_inputs"] is True


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.keep = torch.nn.Linear(8, 8)
        self.skip = torch.nn.Linear(6, 3)

    def forward(self, x):
        return self.keep(x)


def test_find_prunable_linear_names_skips_non_2_4_shapes():
    model = TinyBlock()

    assert find_prunable_linear_names(model) == ["keep"]


def test_group_prunable_linear_names_by_layer_orders_transformer_layers_only():
    names = [
        "embed_out",
        "gpt_neox.layers.10.mlp.dense_4h_to_h",
        "gpt_neox.layers.2.attention.dense",
        "gpt_neox.layers.2.mlp.dense_h_to_4h",
        "other.layers.0.dense",
        "gpt_neox.layers.10.attention.query_key_value",
    ]

    groups = group_prunable_linear_names_by_layer(names)

    assert [group.layer_index for group in groups] == [2, 10]
    assert [group.layer_name for group in groups] == ["gpt_neox.layers.2", "gpt_neox.layers.10"]
    assert groups[0].module_names == ("gpt_neox.layers.2.attention.dense", "gpt_neox.layers.2.mlp.dense_h_to_4h")
    assert groups[1].module_names == (
        "gpt_neox.layers.10.mlp.dense_4h_to_h",
        "gpt_neox.layers.10.attention.query_key_value",
    )


def test_snapshot_restore_module_weights_resets_sparse_state():
    model = TinyBlock()
    names = ("keep",)
    original = model.keep.weight.detach().clone()
    snapshot = _snapshot_module_weights(model, names)

    model.keep.weight.data.zero_()
    _restore_module_weights(model, snapshot)

    assert torch.equal(model.keep.weight, original)
    assert snapshot["keep"].data_ptr() != model.keep.weight.data_ptr()


def test_collect_linear_inputs_flattens_and_limits_tokens():
    model = TinyBlock()
    batches = [torch.randn(2, 5, 8), torch.randn(2, 5, 8)]

    inputs = collect_linear_inputs(
        model=model,
        module_name="keep",
        batches=batches,
        device=torch.device("cpu"),
        max_tokens=12,
    )

    assert inputs.shape == (12, 8)


def test_sparse24_config_validates_inputs(tmp_path: Path):
    config = Sparse24EvalConfig(
        output_dir=tmp_path,
        methods=("magnitude", "wanda", "gptq-cae", "gptaq-cae", "gptaq-cae-diag", "gptaq-cae-gd", "qronos"),
        eval_steps=1,
    )
    assert config.methods == ("magnitude", "wanda", "gptq-cae", "gptaq-cae", "gptaq-cae-diag", "gptaq-cae-gd", "qronos")
    assert config.sparsity_n == 2
    assert config.sparsity_m == 4
    assert config.gd_steps == 1

    with pytest.raises(ValueError):
        Sparse24EvalConfig(output_dir=tmp_path, methods=())
    with pytest.raises(ValueError):
        Sparse24EvalConfig(output_dir=tmp_path, methods=("bad",))
    with pytest.raises(ValueError):
        Sparse24EvalConfig(output_dir=tmp_path, calibration_tokens=0)
    with pytest.raises(ValueError):
        Sparse24EvalConfig(output_dir=tmp_path, sparsity_n=0, sparsity_m=2)
    with pytest.raises(ValueError):
        Sparse24EvalConfig(output_dir=tmp_path, sparsity_n=2, sparsity_m=2)
    with pytest.raises(ValueError):
        Sparse24EvalConfig(output_dir=tmp_path, sparsity_n=1, sparsity_m=3, blocksize=128)
    with pytest.raises(ValueError):
        Sparse24EvalConfig(output_dir=tmp_path, gd_steps=0)
    with pytest.raises(ValueError):
        Sparse24EvalConfig(output_dir=tmp_path, gd_lr=0.0)


def test_sparse24_greedy_layer_config_validates_limits(tmp_path: Path):
    config = Sparse24GreedyLayerConfig(output_dir=tmp_path, greedy_max_layers=2, eval_steps=1)
    assert config.method == "gptaq-cae"
    assert config.greedy_max_layers == 2

    with pytest.raises(ValueError):
        Sparse24GreedyLayerConfig(output_dir=tmp_path, method="sparsegpt")
    with pytest.raises(ValueError):
        Sparse24GreedyLayerConfig(output_dir=tmp_path, greedy_max_layers=0)
