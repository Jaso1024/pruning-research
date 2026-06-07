import json
from pathlib import Path

import pytest
import torch

from layer_distill.low_qk_attention import (
    LowQKAttention,
    LowQKDistillConfig,
    build_low_qk_lr_configs,
    init_low_qk_from_gpt_neox_attention,
    summarize_low_qk_runs,
)


class TinyTeacherAttention(torch.nn.Module):
    def __init__(self, hidden_size: int = 8, heads: int = 2):
        super().__init__()
        self.num_attention_heads = heads
        self.head_size = hidden_size // heads
        self.query_key_value = torch.nn.Linear(hidden_size, 3 * hidden_size)
        self.dense = torch.nn.Linear(hidden_size, hidden_size)


def test_low_qk_attention_output_shape_and_weights():
    module = LowQKAttention(hidden_size=16, num_heads=4, qk_dim=2)
    hidden = torch.randn(3, 5, 16)

    output, weights = module(hidden, return_weights=True)

    assert output.shape == (3, 5, 16)
    assert weights.shape == (3, 4, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 4, 5), atol=1e-6)


def test_low_qk_attention_is_causal():
    module = LowQKAttention(hidden_size=8, num_heads=2, qk_dim=2)
    hidden = torch.randn(1, 4, 8)

    _, weights = module(hidden, return_weights=True)

    assert weights[..., 0, 1:].abs().max().item() == pytest.approx(0.0)
    assert weights[..., 1, 2:].abs().max().item() == pytest.approx(0.0)
    assert weights[..., 2, 3:].abs().max().item() == pytest.approx(0.0)


def test_init_low_qk_from_teacher_copies_value_and_output_when_shapes_match():
    teacher = TinyTeacherAttention(hidden_size=8, heads=2)
    student = LowQKAttention(hidden_size=8, num_heads=2, qk_dim=2)
    with torch.no_grad():
        teacher.query_key_value.weight.copy_(torch.arange(24 * 8).float().view(24, 8))
        teacher.query_key_value.bias.copy_(torch.arange(24).float())
        teacher.dense.weight.fill_(3.0)
        teacher.dense.bias.fill_(4.0)

    init_low_qk_from_gpt_neox_attention(student, teacher)

    expected_q_rows = torch.cat([teacher.query_key_value.weight[0:2], teacher.query_key_value.weight[12:14]])
    expected_k_rows = torch.cat([teacher.query_key_value.weight[4:6], teacher.query_key_value.weight[16:18]])
    expected_v_rows = torch.cat([teacher.query_key_value.weight[8:12], teacher.query_key_value.weight[20:24]])
    expected_q_bias = torch.cat([teacher.query_key_value.bias[0:2], teacher.query_key_value.bias[12:14]])
    expected_k_bias = torch.cat([teacher.query_key_value.bias[4:6], teacher.query_key_value.bias[16:18]])
    expected_v_bias = torch.cat([teacher.query_key_value.bias[8:12], teacher.query_key_value.bias[20:24]])

    assert torch.equal(student.q_proj.weight, expected_q_rows)
    assert torch.equal(student.k_proj.weight, expected_k_rows)
    assert torch.equal(student.v_proj.weight, expected_v_rows)
    assert torch.equal(student.q_proj.bias, expected_q_bias)
    assert torch.equal(student.k_proj.bias, expected_k_bias)
    assert torch.equal(student.v_proj.bias, expected_v_bias)
    assert torch.equal(student.out_proj.weight, teacher.dense.weight)
    assert torch.equal(student.out_proj.bias, teacher.dense.bias)


def test_low_qk_distill_config_validates_inputs(tmp_path: Path):
    config = LowQKDistillConfig(output_dir=tmp_path, learning_rates=(1e-3,), qk_dim=2)
    assert config.qk_dim == 2

    with pytest.raises(ValueError):
        LowQKDistillConfig(output_dir=tmp_path, learning_rates=())
    with pytest.raises(ValueError):
        LowQKDistillConfig(output_dir=tmp_path, learning_rates=(-1e-3,))
    with pytest.raises(ValueError):
        LowQKDistillConfig(output_dir=tmp_path, qk_dim=0)
    with pytest.raises(ValueError):
        LowQKDistillConfig(output_dir=tmp_path, student_heads=0)


def test_build_low_qk_lr_configs_names_independent_dirs(tmp_path: Path):
    base = LowQKDistillConfig(output_dir=tmp_path, learning_rates=(1e-3, 3e-3), qk_dim=2)

    configs = build_low_qk_lr_configs(base)

    assert [config.learning_rate for config in configs] == [1e-3, 3e-3]
    assert [config.output_dir.name for config in configs] == ["lr_0.001", "lr_0.003"]


def test_summarize_low_qk_runs_picks_best_relative_mse(tmp_path: Path):
    run_a = tmp_path / "lr_0.001"
    run_b = tmp_path / "lr_0.003"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "steps.jsonl").write_text(json.dumps({"step": 2, "rel_mse": 0.4}) + "\n")
    (run_b / "steps.jsonl").write_text(json.dumps({"step": 2, "rel_mse": 0.2}) + "\n")

    summary = summarize_low_qk_runs(tmp_path)

    assert summary["best"]["run_dir"] == str(run_b)
    assert summary["best"]["final_rel_mse"] == 0.2
