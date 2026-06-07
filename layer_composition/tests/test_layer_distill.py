import json
from pathlib import Path

import pytest
import torch

from layer_distill.experiment import (
    DistillConfig,
    JsonlStepLogger,
    build_lr_sweep_configs,
    clone_student_layer,
    summarize_runs,
    middle_layer_pair,
)
from layer_distill.muon import Muon, split_muon_params


class TinyLayer(torch.nn.Module):
    def __init__(self, width: int = 4):
        super().__init__()
        self.proj = torch.nn.Linear(width, width)
        self.norm = torch.nn.LayerNorm(width)

    def forward(self, hidden_states, **_kwargs):
        return (self.norm(self.proj(hidden_states)),)


def test_middle_layer_pair_selects_middle_and_next():
    assert middle_layer_pair(6) == (2, 3)
    assert middle_layer_pair(7) == (3, 4)


@pytest.mark.parametrize("num_layers", [0, 1])
def test_middle_layer_pair_requires_two_layers(num_layers):
    with pytest.raises(ValueError):
        middle_layer_pair(num_layers)


def test_distill_config_validates_layer_pair_and_lrs(tmp_path: Path):
    config = DistillConfig(
        output_dir=tmp_path,
        learning_rates=(1e-4, 3e-4),
        steps=5,
        batch_size=2,
        seq_len=8,
        layer_index=2,
    )
    assert config.layer_index == 2
    assert config.learning_rates == (1e-4, 3e-4)

    with pytest.raises(ValueError):
        DistillConfig(output_dir=tmp_path, learning_rates=(), steps=5)
    with pytest.raises(ValueError):
        DistillConfig(output_dir=tmp_path, learning_rates=(-1e-4,), steps=5)
    with pytest.raises(ValueError):
        DistillConfig(output_dir=tmp_path, learning_rates=(1e-4,), steps=0)


def test_clone_student_layer_deep_copies_first_layer():
    source = TinyLayer()
    student = clone_student_layer(source)

    for source_param, student_param in zip(source.parameters(), student.parameters()):
        assert torch.equal(source_param, student_param)
        assert source_param.data_ptr() != student_param.data_ptr()

    with torch.no_grad():
        next(student.parameters()).add_(1.0)

    assert not torch.equal(next(source.parameters()), next(student.parameters()))


def test_split_muon_params_separates_matrix_from_vector_params():
    layer = TinyLayer()
    muon_params, adamw_params = split_muon_params(layer)

    assert {tuple(param.shape) for param in muon_params} == {(4, 4)}
    assert {tuple(param.shape) for param in adamw_params} == {(4,),}


def test_muon_step_updates_matrix_param_finitely():
    param = torch.nn.Parameter(torch.eye(4))
    param.grad = torch.ones_like(param)
    optimizer = Muon([param], lr=0.01, momentum=0.0, nesterov=False)

    before = param.detach().clone()
    optimizer.step()

    assert param.shape == before.shape
    assert torch.isfinite(param).all()
    assert not torch.equal(param, before)


def test_jsonl_step_logger_writes_and_flushes_records(tmp_path: Path):
    path = tmp_path / "steps.jsonl"
    with JsonlStepLogger(path) as logger:
        logger.log(
            {
                "step": 1,
                "lr": 1e-4,
                "loss": 0.25,
                "rel_mse": 0.5,
                "cosine": 0.75,
            }
        )

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {
            "step": 1,
            "lr": 1e-4,
            "loss": 0.25,
            "rel_mse": 0.5,
            "cosine": 0.75,
        }
    ]


def test_build_lr_sweep_configs_uses_independent_output_dirs(tmp_path: Path):
    base = DistillConfig(
        output_dir=tmp_path,
        learning_rates=(1e-4, 3e-4),
        steps=3,
        batch_size=2,
        seq_len=8,
    )

    configs = build_lr_sweep_configs(base)

    assert [config.learning_rate for config in configs] == [1e-4, 3e-4]
    assert [config.output_dir.name for config in configs] == ["lr_0.0001", "lr_0.0003"]
    assert configs[0].output_dir != configs[1].output_dir


def test_summarize_runs_picks_lowest_final_relative_mse(tmp_path: Path):
    run_a = tmp_path / "lr_0.0001"
    run_b = tmp_path / "lr_0.0003"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "steps.jsonl").write_text(
        json.dumps({"step": 1, "rel_mse": 0.8}) + "\n"
        + json.dumps({"step": 2, "rel_mse": 0.4}) + "\n"
    )
    (run_b / "steps.jsonl").write_text(json.dumps({"step": 2, "rel_mse": 0.2}) + "\n")

    summary = summarize_runs(tmp_path)

    assert summary["best"]["run_dir"] == str(run_b)
    assert summary["best"]["final_rel_mse"] == 0.2
    assert len(summary["runs"]) == 2
