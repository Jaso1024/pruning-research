from __future__ import annotations

import torch

import saliency.qronos_eval as qronos_eval
from saliency.approx import sequential_wanda_matrix_parameter_groups
from saliency.qronos_eval import (
    _SingleLinearInputHook,
    QronosConfig,
    apply_qronos_pruning_,
    asymmetric_minmax_quantize,
    qronos_pruning_target_names,
    qronos_prune_weight,
    qronos_quantize_weight,
)


def test_asymmetric_minmax_quantize_preserves_row_endpoints_with_beta_one():
    weight = torch.tensor([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]])

    quantized, scale, zero = asymmetric_minmax_quantize(weight, bits=2, beta=1.0)

    assert torch.allclose(quantized[:, 0], weight[:, 0])
    assert torch.allclose(quantized[:, -1], weight[:, -1])
    assert scale.shape == (2, 1)
    assert zero.shape == (2, 1)


def test_qronos_quantize_weight_matches_rtn_for_identity_matched_inputs():
    weight = torch.tensor([[-1.0, -0.2, 0.4, 1.0], [2.0, 2.2, 2.8, 4.0]])
    hessian = torch.eye(weight.shape[1], dtype=torch.float64)
    cross = torch.eye(weight.shape[1], dtype=torch.float64)

    quantized, stats = qronos_quantize_weight(
        weight,
        hessian,
        cross,
        bits=2,
        beta=1.0,
        percdamp=0.0,
        use_activation_order=False,
    )
    expected, _, _ = asymmetric_minmax_quantize(weight, bits=2, beta=1.0)

    assert torch.allclose(quantized, expected)
    assert stats["columns"] == 4
    assert stats["format"] == "asymmetric_int2"


def test_qronos_quantize_weight_uses_cross_covariance_for_first_column():
    weight = torch.tensor([[0.49, 1.0]])
    hessian = torch.eye(2, dtype=torch.float64)
    matched_cross = torch.eye(2, dtype=torch.float64)
    shifted_cross = torch.tensor([[1.0, 0.0], [0.60, 1.0]], dtype=torch.float64)

    matched, _ = qronos_quantize_weight(
        weight,
        hessian,
        matched_cross,
        bits=1,
        beta=1.0,
        percdamp=0.0,
        use_activation_order=False,
    )
    shifted, _ = qronos_quantize_weight(
        weight,
        hessian,
        shifted_cross,
        bits=1,
        beta=1.0,
        percdamp=0.0,
        use_activation_order=False,
    )

    assert torch.isclose(matched[0, 0], torch.tensor(0.49))
    assert shifted[0, 0] > matched[0, 0]


def test_qronos_prune_weight_matches_magnitude_pruning_for_identity_stats():
    weight = torch.tensor([[1.0, -4.0, 2.0, -3.0], [8.0, 5.0, -6.0, -7.0]])
    hessian = torch.eye(weight.shape[1], dtype=torch.float64)
    cross = torch.eye(weight.shape[1], dtype=torch.float64)

    pruned, stats = qronos_prune_weight(
        weight,
        hessian,
        cross,
        prune_fraction=0.5,
        pruning_scope="per_output_row",
        percdamp=0.0,
        use_activation_order=False,
    )

    assert pruned.tolist() == [[0.0, -4.0, 0.0, -3.0], [8.0, 0.0, 0.0, -7.0]]
    assert stats["format"] == "base_precision_pruned"
    assert stats["zeroed"] == 4
    assert stats["pruning_scope"] == "per_output_row"


def test_qronos_prune_weight_activation_order_preserves_original_shape_and_zero_count():
    weight = torch.tensor([[0.1, 10.0, 0.2, 9.0], [4.0, 0.3, 5.0, 0.4]])
    hessian = torch.diag(torch.tensor([1.0, 4.0, 2.0, 3.0], dtype=torch.float64))
    cross = hessian.clone()

    pruned, stats = qronos_prune_weight(
        weight,
        hessian,
        cross,
        prune_fraction=0.5,
        pruning_scope="per_matrix",
        percdamp=0.0,
        use_activation_order=True,
    )

    assert pruned.shape == weight.shape
    assert int(torch.count_nonzero(pruned == 0).item()) == 4
    assert stats["activation_order"] is True
    assert stats["zeroed"] == 4


def test_qronos_prune_weight_handles_rank_deficient_hessian_tail():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    weight = torch.randn((16, 24), generator=generator)
    factors = torch.randn((24, 8), generator=generator)
    hessian = factors.matmul(factors.t())

    pruned, stats = qronos_prune_weight(
        weight,
        hessian,
        hessian,
        prune_fraction=0.25,
        pruning_scope="per_matrix",
        percdamp=1e-6,
        use_activation_order=True,
    )

    assert pruned.shape == weight.shape
    assert stats["zeroed"] == 96
    assert stats["cholesky_jitter"] >= 0.0


def test_qronos_pruning_targets_match_wanda_matrix_sequential_targets():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(4, 3)
            self.block = torch.nn.Sequential(torch.nn.Linear(3, 5, bias=False), torch.nn.ReLU())
            self.lm_head = torch.nn.Linear(5, 4, bias=False)

    model = TinyModel()
    wanda_names = [name for _, names in sequential_wanda_matrix_parameter_groups(model) for name in names]

    assert qronos_pruning_target_names(model) == wanda_names


def test_apply_qronos_pruning_covers_embedding_fallback_and_linear(monkeypatch, tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(2, 4)
            self.proj = torch.nn.Linear(4, 2, bias=False)

    original_model = TinyModel()
    pruned_model = TinyModel()
    with torch.no_grad():
        pruned_model.embed_tokens.weight.copy_(torch.tensor([[1.0, -4.0, 2.0, -3.0], [8.0, 5.0, -6.0, -7.0]]))
        pruned_model.proj.weight.copy_(torch.tensor([[1.0, -4.0, 2.0, -3.0], [8.0, 5.0, -6.0, -7.0]]))

    def fake_collect_qronos_pair_stats(*args, **kwargs):
        assert args[-1] == "proj"
        return torch.eye(4), torch.eye(4), 11

    monkeypatch.setattr(qronos_eval, "collect_qronos_pair_stats", fake_collect_qronos_pair_stats)
    rows = apply_qronos_pruning_(
        original_model,
        pruned_model,
        tokenizer=None,
        records=[],
        config=QronosConfig(
            output_dir=tmp_path,
            prune_fraction=0.5,
            pruning_scope="per_output_row",
            percdamp=0.0,
            use_activation_order=False,
        ),
        device=torch.device("cpu"),
    )

    assert [row["name"] for row in rows] == ["embed_tokens.weight", "proj.weight"]
    assert rows[0]["format"] == "base_precision_magnitude_pruned_fallback"
    assert rows[0]["zeroed"] == 4
    assert rows[1]["format"] == "base_precision_pruned"
    assert rows[1]["tokens"] == 11
    assert pruned_model.embed_tokens.weight.tolist() == [[0.0, -4.0, 0.0, -3.0], [8.0, 0.0, 0.0, -7.0]]


def test_single_linear_input_hook_snapshots_inputs():
    module = torch.nn.Linear(3, 2, bias=False)
    hook = _SingleLinearInputHook(module, use_attention_mask=False)
    try:
        inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        _ = module(inputs)
        captured = hook.value
        assert captured is not None

        inputs.zero_()

        assert torch.equal(captured, torch.arange(6, dtype=torch.float32).reshape(2, 3))
    finally:
        hook.close()


def test_qronos_quantize_weight_cpu_cuda_parity_when_cuda_available():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    weight = torch.randn(8, 16)
    activations = torch.randn(64, 16)
    quantized_activations = activations + 0.01 * torch.randn_like(activations)
    hessian = quantized_activations.T @ quantized_activations / 4
    cross = activations.T @ quantized_activations / 4

    cpu_quantized, _ = qronos_quantize_weight(
        weight,
        hessian,
        cross,
        bits=4,
        percdamp=1e-6,
        cholesky_scale=1e4,
        num_blocks=8,
        use_activation_order=True,
    )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    cuda_quantized, _ = qronos_quantize_weight(
        weight.cuda(),
        hessian.cuda(),
        cross.cuda(),
        bits=4,
        percdamp=1e-6,
        cholesky_scale=1e4,
        num_blocks=8,
        use_activation_order=True,
    )

    assert torch.allclose(cpu_quantized, cuda_quantized.cpu(), atol=1e-4, rtol=1e-4)
