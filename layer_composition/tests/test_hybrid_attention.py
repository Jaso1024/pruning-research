import json
from pathlib import Path

import pytest
import torch

from layer_distill.hybrid_attention import (
    ExternalAttentionProvider,
    ExternalAttentionWeightsGPTNeoXAttention,
    HybridAttentionEvalConfig,
    build_exponential_combo_attentions,
    greedy_layer_sweep,
    load_exponential_combo_weights,
    patch_attentions_with_external_weights,
    restore_attentions,
)


class TinyGPTNeoXAttention(torch.nn.Module):
    def __init__(self, hidden_size: int = 4, heads: int = 2):
        super().__init__()
        self.num_attention_heads = heads
        self.head_size = hidden_size // heads
        self.query_key_value = torch.nn.Linear(hidden_size, 3 * hidden_size)
        self.dense = torch.nn.Linear(hidden_size, hidden_size)


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = TinyGPTNeoXAttention()


class TinyLayerModel(torch.nn.Module):
    def __init__(self, layers: int = 3):
        super().__init__()
        self.layers = torch.nn.ModuleList(TinyLayer() for _ in range(layers))


def test_external_attention_weights_adapter_uses_external_weights_and_source_vo():
    source = TinyGPTNeoXAttention()
    provider = ExternalAttentionProvider()
    adapter = ExternalAttentionWeightsGPTNeoXAttention(source, layer_idx=0, provider=provider)
    hidden = torch.randn(2, 3, 4)
    weights = torch.softmax(torch.randn(2, 2, 3, 3), dim=-1)
    provider.set({0: weights})

    actual, returned_weights = adapter(hidden)

    qkv = source.query_key_value(hidden).view(2, 3, 2, 3 * source.head_size).transpose(1, 2)
    _, _, value = qkv.chunk(3, dim=-1)
    expected = torch.matmul(weights, value).transpose(1, 2).reshape(2, 3, 4).contiguous()
    expected = source.dense(expected)
    assert torch.allclose(actual, expected)
    assert returned_weights is weights


def test_patch_attentions_with_external_weights_selects_layers_and_restores():
    model = TinyLayerModel(layers=3)
    provider = ExternalAttentionProvider()
    original = [layer.attention for layer in model.layers]

    originals = patch_attentions_with_external_weights(model, provider, layer_indices=(1,))

    assert list(originals) == [1]
    assert model.layers[0].attention is original[0]
    assert isinstance(model.layers[1].attention, ExternalAttentionWeightsGPTNeoXAttention)
    assert model.layers[2].attention is original[2]

    restore_attentions(model, originals)

    assert [layer.attention for layer in model.layers] == original


def test_external_attention_provider_rejects_missing_or_wrong_shape():
    provider = ExternalAttentionProvider()
    value = torch.randn(1, 2, 3, 4)

    with pytest.raises(RuntimeError, match="missing external attention"):
        provider.require(0, value)

    provider.set({0: torch.randn(1, 2, 3, 2)})
    with pytest.raises(RuntimeError, match="shape mismatch"):
        provider.require(0, value)


def test_load_exponential_combo_weights_averages_prompt_rows(tmp_path: Path):
    path = tmp_path / "exponential_combo.jsonl"
    rows = [
        {"layer": 0, "big_head": 0, "small_heads": 2, "weights": [0.25, 0.75]},
        {"layer": 0, "big_head": 0, "small_heads": 2, "weights": [0.75, 0.25]},
        {"layer": 0, "big_head": 1, "small_heads": 2, "weights": [0.10, 0.90]},
        {"layer": 0, "big_head": 1, "small_heads": 2, "weights": [0.30, 0.70]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    weights = load_exponential_combo_weights(path)

    assert weights.shape == (1, 2, 2)
    assert torch.allclose(weights[0, 0], torch.tensor([0.50, 0.50]))
    assert torch.allclose(weights[0, 1], torch.tensor([0.20, 0.80]))


def test_build_exponential_combo_attentions_emits_target_head_count_and_filters_layers():
    small_attn = torch.tensor(
        [
            [
                [[0.8, 0.2], [0.4, 0.6]],
                [[0.2, 0.8], [0.7, 0.3]],
            ]
        ]
    )
    combo_weights = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
        ]
    )

    combined = build_exponential_combo_attentions([small_attn, small_attn], combo_weights, target_dtype=torch.float32, layer_indices=(1,))

    assert set(combined) == {1}
    assert combined[1].shape == (1, 3, 2, 2)
    assert torch.allclose(combined[1].sum(dim=-1), torch.ones(1, 3, 2))
    assert torch.allclose(combined[1][:, 0], small_attn[:, 1])
    assert torch.allclose(combined[1][:, 1], small_attn[:, 0])


def test_greedy_layer_sweep_selects_best_candidate_each_round():
    scores = {
        (0,): 3.0,
        (1,): 1.0,
        (2,): 2.0,
        (0, 1): 5.0,
        (1, 2): 0.5,
    }

    def evaluate(layer_group: tuple[int, ...], run_name: str, output_dir: Path) -> dict:
        return {"run_name": run_name, "loss": scores[layer_group], "ppl": scores[layer_group]}

    runs, path = greedy_layer_sweep(
        comparable_layer_count=3,
        max_selected_layers=2,
        output_dir=Path("unused"),
        evaluate_layer_group=evaluate,
    )

    assert [row["selected_layer"] for row in path] == [1, 2]
    assert [row["selected_layers"] for row in path] == [[1], [1, 2]]
    assert len(runs) == 5
    assert runs[1]["is_selected_candidate"] is True
    assert runs[-1]["is_selected_candidate"] is True


def test_hybrid_attention_eval_config_validates_inputs(tmp_path: Path):
    config = HybridAttentionEvalConfig(
        output_dir=tmp_path,
        combo_path=tmp_path / "combo.jsonl",
        eval_steps=1,
        replace_layers=(2, 0),
        greedy_layer_sweep=True,
    )
    assert config.big_model == "EleutherAI/pythia-2.8b"
    assert config.replace_layers == (0, 2)
    assert config.greedy_layer_sweep is False

    with pytest.raises(ValueError):
        HybridAttentionEvalConfig(output_dir=tmp_path, combo_path=tmp_path / "combo.jsonl", eval_steps=0)
    with pytest.raises(ValueError):
        HybridAttentionEvalConfig(output_dir=tmp_path, combo_path=tmp_path / "combo.jsonl", seq_len=1)
    with pytest.raises(ValueError):
        HybridAttentionEvalConfig(output_dir=tmp_path, combo_path=tmp_path / "combo.jsonl", replace_layers=(-1,))
    with pytest.raises(ValueError):
        HybridAttentionEvalConfig(output_dir=tmp_path, combo_path=tmp_path / "combo.jsonl", greedy_max_layers=0)
