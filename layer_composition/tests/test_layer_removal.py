from pathlib import Path

import pytest
import torch

from layer_distill.layer_removal import (
    LayerRemovalEvalConfig,
    SkipTransformerLayer,
    greedy_removal_sweep,
    patch_layers_with_skip,
    restore_layers,
)


class TinyLayer(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, hidden_states, **kwargs):
        return hidden_states + self.value


class TinyLayerModel(torch.nn.Module):
    def __init__(self, layers: int = 3):
        super().__init__()
        self.layers = torch.nn.ModuleList(TinyLayer(float(idx)) for idx in range(layers))


def test_skip_transformer_layer_returns_hidden_states_and_rejects_cache():
    layer = SkipTransformerLayer()
    hidden = torch.randn(2, 3, 4)

    assert layer(hidden) is hidden
    with pytest.raises(NotImplementedError):
        layer(hidden, use_cache=True)
    with pytest.raises(NotImplementedError):
        layer(hidden, layer_past=object())


def test_patch_layers_with_skip_selects_layers_and_restores():
    model = TinyLayerModel(layers=4)
    original = list(model.layers)

    originals = patch_layers_with_skip(model, layer_indices=(1, 3))

    assert list(originals) == [1, 3]
    assert model.layers[0] is original[0]
    assert isinstance(model.layers[1], SkipTransformerLayer)
    assert model.layers[2] is original[2]
    assert isinstance(model.layers[3], SkipTransformerLayer)

    restore_layers(model, originals)

    assert list(model.layers) == original


def test_greedy_removal_sweep_selects_lowest_ppl_candidate_each_round():
    scores = {
        (0,): 3.0,
        (1,): 1.0,
        (2,): 2.0,
        (0, 1): 5.0,
        (1, 2): 0.5,
    }

    def evaluate(layer_group: tuple[int, ...], run_name: str, output_dir: Path) -> dict:
        return {"run_name": run_name, "loss": scores[layer_group], "ppl": scores[layer_group]}

    runs, path = greedy_removal_sweep(
        layer_count=3,
        max_removed_layers=2,
        output_dir=Path("unused"),
        evaluate_removed_layers=evaluate,
    )

    assert [row["removed_layer"] for row in path] == [1, 2]
    assert [row["removed_layers"] for row in path] == [[1], [1, 2]]
    assert len(runs) == 5
    assert runs[1]["is_selected_candidate"] is True
    assert runs[-1]["is_selected_candidate"] is True


def test_layer_removal_eval_config_validates_inputs(tmp_path: Path):
    config = LayerRemovalEvalConfig(output_dir=tmp_path, eval_steps=1, remove_layers=(2, 0), greedy_layer_removal=True)
    assert config.remove_layers == (0, 2)
    assert config.greedy_layer_removal is False

    with pytest.raises(ValueError):
        LayerRemovalEvalConfig(output_dir=tmp_path, eval_steps=0)
    with pytest.raises(ValueError):
        LayerRemovalEvalConfig(output_dir=tmp_path, seq_len=1)
    with pytest.raises(ValueError):
        LayerRemovalEvalConfig(output_dir=tmp_path, greedy_max_layers=0)
    with pytest.raises(ValueError):
        LayerRemovalEvalConfig(output_dir=tmp_path, remove_layers=(-1,))
