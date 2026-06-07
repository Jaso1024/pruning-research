from pathlib import Path

import pytest
import torch

from layer_distill.attention_basis_ppl import (
    AttentionBasisPPLConfig,
    build_next_layer_candidates,
    compress_attention_heads_nmf,
    patch_attentions_with_head_basis,
    restore_attentions,
    select_top_layer_groups,
    summarize_attention_basis_ppl,
)


def test_compress_attention_heads_nmf_preserves_attention_shape_and_rows():
    torch.manual_seed(0)
    weights = torch.softmax(torch.randn(2, 4, 5, 5), dim=-1)
    causal_mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    weights = weights.masked_fill(~causal_mask.view(1, 1, 5, 5), 0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    compressed = compress_attention_heads_nmf(weights, basis_size=2, iterations=4)

    assert compressed.shape == weights.shape
    assert compressed.dtype == weights.dtype
    assert torch.allclose(compressed.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-5)
    assert torch.count_nonzero(compressed.masked_select(~causal_mask.view(1, 1, 5, 5))) == 0


def test_compress_attention_heads_nmf_supports_exponential_combination():
    torch.manual_seed(1)
    weights = torch.softmax(torch.randn(1, 5, 4, 4), dim=-1)
    causal_mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    weights = weights.masked_fill(~causal_mask.view(1, 1, 4, 4), 0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    linear = compress_attention_heads_nmf(weights, basis_size=2, iterations=3, combine_mode="linear")
    exponential = compress_attention_heads_nmf(weights, basis_size=2, iterations=3, combine_mode="exponential")

    assert exponential.shape == weights.shape
    assert exponential.dtype == weights.dtype
    assert torch.allclose(exponential.sum(dim=-1), torch.ones(1, 5, 4), atol=1e-5)
    assert torch.count_nonzero(exponential.masked_select(~causal_mask.view(1, 1, 4, 4))) == 0
    assert not torch.allclose(linear, exponential)


def test_compress_attention_heads_nmf_quantizes_reconstructed_rows():
    torch.manual_seed(2)
    weights = torch.softmax(torch.randn(1, 5, 4, 4), dim=-1)
    causal_mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    weights = weights.masked_fill(~causal_mask.view(1, 1, 4, 4), 0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    unquantized = compress_attention_heads_nmf(weights, basis_size=2, iterations=3, quantization_bits=0)
    quantized = compress_attention_heads_nmf(weights, basis_size=2, iterations=3, quantization_bits=2)

    assert quantized.shape == weights.shape
    assert quantized.dtype == weights.dtype
    assert torch.allclose(quantized.sum(dim=-1), torch.ones(1, 5, 4), atol=1e-5)
    assert torch.count_nonzero(quantized.masked_select(~causal_mask.view(1, 1, 4, 4))) == 0
    assert not torch.allclose(unquantized, quantized)


def test_compress_attention_heads_nmf_supports_named_quantization_formats():
    torch.manual_seed(3)
    weights = torch.softmax(torch.randn(1, 5, 6, 6), dim=-1)
    causal_mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    weights = weights.masked_fill(~causal_mask.view(1, 1, 6, 6), 0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    unquantized = compress_attention_heads_nmf(weights, basis_size=2, iterations=3)
    for fmt in ("int8", "int4", "fp8_e4m3", "fp8_e5m2", "nvfp4", "mxfp4", "nf4"):
        quantized = compress_attention_heads_nmf(weights, basis_size=2, iterations=3, quantization_format=fmt)

        assert quantized.shape == weights.shape
        assert quantized.dtype == weights.dtype
        assert torch.allclose(quantized.sum(dim=-1), torch.ones(1, 5, 6), atol=1e-5)
        assert torch.count_nonzero(quantized.masked_select(~causal_mask.view(1, 1, 6, 6))) == 0
        assert not torch.allclose(unquantized, quantized)


def test_compress_attention_heads_nmf_quantizes_basis_factors():
    torch.manual_seed(4)
    weights = torch.softmax(torch.randn(1, 5, 6, 6), dim=-1)
    causal_mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    weights = weights.masked_fill(~causal_mask.view(1, 1, 6, 6), 0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    unquantized = compress_attention_heads_nmf(weights, basis_size=2, iterations=3)
    factor_quantized = compress_attention_heads_nmf(
        weights,
        basis_size=2,
        iterations=3,
        quantization_format="int4",
        quantization_target="factors",
    )
    reconstructed_quantized = compress_attention_heads_nmf(
        weights,
        basis_size=2,
        iterations=3,
        quantization_format="int4",
        quantization_target="reconstructed",
    )

    assert factor_quantized.shape == weights.shape
    assert torch.allclose(factor_quantized.sum(dim=-1), torch.ones(1, 5, 6), atol=1e-5)
    assert torch.count_nonzero(factor_quantized.masked_select(~causal_mask.view(1, 1, 6, 6))) == 0
    assert not torch.allclose(unquantized, factor_quantized)
    assert not torch.allclose(reconstructed_quantized, factor_quantized)


def test_compress_attention_heads_nmf_rejects_invalid_basis():
    weights = torch.softmax(torch.randn(1, 3, 4, 4), dim=-1)

    with pytest.raises(ValueError, match="basis_size"):
        compress_attention_heads_nmf(weights, basis_size=0)
    with pytest.raises(ValueError, match="smaller"):
        compress_attention_heads_nmf(weights, basis_size=3)
    with pytest.raises(ValueError, match="combine_mode"):
        compress_attention_heads_nmf(weights, basis_size=2, combine_mode="bad")
    with pytest.raises(ValueError, match="quantization_bits"):
        compress_attention_heads_nmf(weights, basis_size=2, quantization_bits=-1)
    with pytest.raises(ValueError, match="quantization_format"):
        compress_attention_heads_nmf(weights, basis_size=2, quantization_format="bad")
    with pytest.raises(ValueError, match="only one"):
        compress_attention_heads_nmf(weights, basis_size=2, quantization_bits=4, quantization_format="int4")
    with pytest.raises(ValueError, match="quantization_target"):
        compress_attention_heads_nmf(weights, basis_size=2, quantization_format="int4", quantization_target="bad")


def test_attention_basis_ppl_config_validates_combine_mode_and_quantization(tmp_path: Path):
    config = AttentionBasisPPLConfig(
        output_dir=tmp_path,
        basis_sizes=(1,),
        combine_mode="exponential",
        basis_quantization_format="nvfp4",
    )

    assert config.combine_mode == "exponential"
    assert config.basis_quantization_format == "nvfp4"

    with pytest.raises(ValueError, match="combine_mode"):
        AttentionBasisPPLConfig(output_dir=tmp_path, combine_mode="bad")
    with pytest.raises(ValueError, match="basis_quantization_bits"):
        AttentionBasisPPLConfig(output_dir=tmp_path, basis_quantization_bits=-1)
    with pytest.raises(ValueError, match="basis_quantization_format"):
        AttentionBasisPPLConfig(output_dir=tmp_path, basis_quantization_format="bad")
    with pytest.raises(ValueError, match="only one"):
        AttentionBasisPPLConfig(output_dir=tmp_path, basis_quantization_bits=4, basis_quantization_format="int4")
    with pytest.raises(ValueError, match="basis_quantization_target"):
        AttentionBasisPPLConfig(output_dir=tmp_path, basis_quantization_target="bad")


def test_attention_basis_ppl_config_validates_inputs(tmp_path: Path):
    config = AttentionBasisPPLConfig(output_dir=tmp_path, basis_sizes=(1, 2), eval_steps=1)

    assert config.basis_sizes == (1, 2)
    assert config.output_dir == tmp_path

    with pytest.raises(ValueError, match="basis_sizes"):
        AttentionBasisPPLConfig(output_dir=tmp_path, basis_sizes=())
    with pytest.raises(ValueError, match="eval_steps"):
        AttentionBasisPPLConfig(output_dir=tmp_path, eval_steps=0)
    with pytest.raises(ValueError, match="nmf_iterations"):
        AttentionBasisPPLConfig(output_dir=tmp_path, nmf_iterations=0)


class TinyAttention(torch.nn.Module):
    def __init__(self, hidden_size: int = 8, heads: int = 4):
        super().__init__()
        self.num_attention_heads = heads
        self.head_size = hidden_size // heads
        self.query_key_value = torch.nn.Linear(hidden_size, 3 * hidden_size)
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.scaling = self.head_size**-0.5
        self.attention_dropout = 0.0


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = TinyAttention()


class TinyModel(torch.nn.Module):
    def __init__(self, layers: int = 3):
        super().__init__()
        self.layers = torch.nn.ModuleList(TinyLayer() for _ in range(layers))


def test_patch_attentions_with_head_basis_can_patch_selected_layers_and_restore():
    model = TinyModel(layers=3)
    original = [layer.attention for layer in model.layers]

    originals = patch_attentions_with_head_basis(
        model,
        basis_size=2,
        nmf_iterations=1,
        basis_quantization_format="int4",
        basis_quantization_target="factors",
        layer_indices=(1,),
    )

    assert list(originals) == [1]
    assert model.layers[0].attention is original[0]
    assert model.layers[1].attention is not original[1]
    assert model.layers[1].attention.basis_quantization_format == "int4"
    assert model.layers[1].attention.basis_quantization_target == "factors"
    assert model.layers[2].attention is original[2]

    restore_attentions(model, originals)

    assert [layer.attention for layer in model.layers] == original


def test_build_next_layer_candidates_expands_frontier_without_duplicates():
    candidates = build_next_layer_candidates(frontier=[(), (1,)], layer_count=4)

    assert set(candidates) == {(0,), (1,), (2,), (3,), (0, 1), (1, 2), (1, 3)}
    assert len(candidates) == 7


def test_select_top_layer_groups_ranks_by_ppl_then_layers():
    rows = [
        {"layer_group": [2], "ppl": 13.0},
        {"layer_group": [1], "ppl": 12.0},
        {"layer_group": [0], "ppl": 12.0},
    ]

    assert select_top_layer_groups(rows, beam_width=2) == ((0,), (1,))


def test_summarize_attention_basis_ppl_adds_ratios(tmp_path: Path):
    summary = summarize_attention_basis_ppl(
        output_dir=tmp_path,
        runs=[
            {"run_name": "baseline", "ppl": 10.0, "loss": 2.0},
            {"run_name": "basis_2", "ppl": 12.5, "loss": 2.5, "basis_size": 2},
        ],
    )

    assert summary["baseline"]["ppl"] == 10.0
    assert summary["runs"][1]["ppl_ratio_vs_baseline"] == 1.25
    assert summary["runs"][1]["ppl_delta_vs_baseline"] == 2.5
    assert (tmp_path / "summary.json").exists()
