import json
from pathlib import Path

import pytest
import torch

from layer_distill.merge import (
    MergeEvalConfig,
    build_merge_configs,
    merge_module_pair,
    slerp_tensor,
    summarize_merge_runs,
)


class TinyLayer(torch.nn.Module):
    def __init__(self, width: int = 2):
        super().__init__()
        self.proj = torch.nn.Linear(width, width)
        self.norm = torch.nn.LayerNorm(width)

    def forward(self, hidden_states, **_kwargs):
        return (self.norm(self.proj(hidden_states)),)


def test_slerp_tensor_preserves_endpoints():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])

    assert torch.allclose(slerp_tensor(a, b, 0.0), a)
    assert torch.allclose(slerp_tensor(a, b, 1.0), b)


def test_slerp_tensor_uses_spherical_midpoint_for_orthogonal_unit_vectors():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    expected = torch.tensor([2**-0.5, 2**-0.5])

    assert torch.allclose(slerp_tensor(a, b, 0.5), expected, atol=1e-6)
    assert torch.allclose(slerp_tensor(a, b, 0.5).norm(), torch.tensor(1.0), atol=1e-6)


def test_geometric_slerp_uses_geometric_norm_interpolation():
    a = torch.tensor([2.0, 0.0])
    b = torch.tensor([0.0, 8.0])
    merged = slerp_tensor(a, b, 0.5, geometric_norm=True)

    assert torch.allclose(merged.norm(), torch.tensor(4.0), atol=1e-5)
    assert merged[0] > 0
    assert merged[1] > 0


def test_slerp_tensor_falls_back_for_nearly_parallel_vectors():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([2.0, 0.0])

    assert torch.allclose(slerp_tensor(a, b, 0.25), torch.tensor([1.25, 0.0]))


def test_merge_module_pair_does_not_alias_sources():
    first = TinyLayer()
    second = TinyLayer()
    with torch.no_grad():
        for param in first.parameters():
            param.fill_(1.0)
        for param in second.parameters():
            param.fill_(3.0)

    merged = merge_module_pair(first, second, method="linear", t=0.25)

    for param in merged.parameters():
        assert torch.allclose(param, torch.full_like(param, 1.5))
    with torch.no_grad():
        next(merged.parameters()).add_(10.0)
    assert torch.allclose(next(first.parameters()), torch.ones_like(next(first.parameters())))


def test_merge_eval_config_validates_methods_and_t_values(tmp_path: Path):
    config = MergeEvalConfig(output_dir=tmp_path, methods=("slerp", "geom_slerp"), t_values=(0.0, 0.5, 1.0))
    assert config.methods == ("slerp", "geom_slerp")

    with pytest.raises(ValueError):
        MergeEvalConfig(output_dir=tmp_path, methods=(), t_values=(0.5,))
    with pytest.raises(ValueError):
        MergeEvalConfig(output_dir=tmp_path, methods=("bad",), t_values=(0.5,))
    with pytest.raises(ValueError):
        MergeEvalConfig(output_dir=tmp_path, methods=("slerp",), t_values=(-0.1,))
    with pytest.raises(ValueError):
        MergeEvalConfig(output_dir=tmp_path, methods=("slerp",), t_values=(1.1,))


def test_build_merge_configs_names_independent_output_dirs(tmp_path: Path):
    base = MergeEvalConfig(output_dir=tmp_path, methods=("slerp", "geom_slerp"), t_values=(0.25, 0.5))

    configs = build_merge_configs(base)

    assert [(config.method, config.t) for config in configs] == [
        ("slerp", 0.25),
        ("slerp", 0.5),
        ("geom_slerp", 0.25),
        ("geom_slerp", 0.5),
    ]
    assert [config.output_dir.name for config in configs] == [
        "slerp_t_0.25",
        "slerp_t_0.5",
        "geom_slerp_t_0.25",
        "geom_slerp_t_0.5",
    ]


def test_summarize_merge_runs_picks_lowest_final_relative_mse(tmp_path: Path):
    run_a = tmp_path / "slerp_t_0.25"
    run_b = tmp_path / "geom_slerp_t_0.5"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "steps.jsonl").write_text(json.dumps({"step": 2, "rel_mse": 0.4, "cosine": 0.7}) + "\n")
    (run_b / "steps.jsonl").write_text(json.dumps({"step": 2, "rel_mse": 0.2, "cosine": 0.8}) + "\n")

    summary = summarize_merge_runs(tmp_path)

    assert summary["best"]["run_dir"] == str(run_b)
    assert summary["best"]["final_rel_mse"] == 0.2
    assert len(summary["runs"]) == 2
