from __future__ import annotations

import torch

from saliency.gptq_eval import (
    GPTQConfig,
    LinearHessianCollector,
    apply_gptq_fp8_from_originals,
    apply_damped_gptq_update,
    apply_gradient_descent_gptq_update,
    apply_staged_gptq_weights,
    apply_gptq_fp8_to_linear,
    collect_linear_hessians,
    gptq_quantize_weight,
    quantize_fp8_e4m3_per_row,
    set_linear_weights,
    snapshot_linear_weights,
)


def test_quantize_fp8_e4m3_per_row_clamps_and_preserves_zero_row():
    weight = torch.tensor(
        [
            [0.0, 1.0, -2.0, 5000.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    quantized, scale = quantize_fp8_e4m3_per_row(weight)

    assert quantized.shape == weight.shape
    assert scale.shape == (2, 1)
    assert torch.isfinite(quantized).all()
    assert torch.all(quantized[1] == 0)
    assert scale[0, 0] > 0


def test_gptq_quantize_weight_matches_plain_fp8_with_identity_hessian():
    weight = torch.tensor([[0.0, 0.25, -1.0], [4.0, -8.0, 16.0]], dtype=torch.float32)
    hessian = torch.eye(weight.shape[1], dtype=torch.float64)

    quantized, stats = gptq_quantize_weight(weight, hessian, damp_percent=0.0, blocksize=2)
    expected, _ = quantize_fp8_e4m3_per_row(weight)

    assert torch.allclose(quantized, expected)
    assert stats["columns"] == 3
    assert stats["weights"] == weight.numel()
    assert stats["mean_abs_error"] >= 0.0


def test_gptq_quantize_weight_accepts_diagonal_hessian_vector():
    weight = torch.tensor([[0.0, 0.25, -1.0], [4.0, -8.0, 16.0]], dtype=torch.float32)
    hessian_diag = torch.tensor([3.0, 5.0, 7.0], dtype=torch.float64)

    quantized, stats = gptq_quantize_weight(weight, hessian_diag, damp_percent=0.01, blocksize=2)
    expected, _ = quantize_fp8_e4m3_per_row(weight)

    assert torch.allclose(quantized, expected)
    assert stats["columns"] == 3
    assert stats["hessian_approximation"] == "diagonal"
    assert stats["damp"] > 0.0


def test_linear_hessian_collector_diagonal_accumulates_feature_squares():
    layer = torch.nn.Linear(3, 2)
    collector = LinearHessianCollector([("linear", layer)], hessian_approximation="diagonal")
    collector.install([("linear", layer)])
    x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    try:
        layer(x)
    finally:
        collector.close()

    assert collector.hessians["linear"].shape == (3,)
    assert torch.allclose(collector.hessians["linear"], torch.tensor([17.0, 29.0, 45.0], dtype=torch.float64))
    assert collector.tokens["linear"] == 2


def test_apply_gptq_fp8_to_linear_updates_weight_not_bias():
    layer = torch.nn.Linear(3, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.0, 0.25, -1.0], [4.0, -8.0, 16.0]]))
        layer.bias.copy_(torch.tensor([1.0, -2.0]))

    summary = apply_gptq_fp8_to_linear(
        "linear",
        layer,
        torch.eye(3, dtype=torch.float64),
        GPTQConfig(output_dir="unused", max_calibration_examples=1, max_eval_examples=1),
    )
    expected, _ = quantize_fp8_e4m3_per_row(torch.tensor([[0.0, 0.25, -1.0], [4.0, -8.0, 16.0]]))

    assert torch.allclose(layer.weight, expected)
    assert layer.bias.tolist() == [1.0, -2.0]
    assert summary["name"] == "linear"
    assert summary["format"] == "fp8_e4m3"


def test_apply_gptq_fp8_from_originals_reuses_saved_fp32_weights_each_step():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.0, 0.25, -1.0], [4.0, -8.0, 16.0]]))
    originals = snapshot_linear_weights(model)
    with torch.no_grad():
        model[0].weight.zero_()

    rows = apply_gptq_fp8_from_originals(
        model,
        {"0": torch.eye(3, dtype=torch.float64)},
        originals,
        GPTQConfig(output_dir="unused", gptq_steps=2, max_calibration_examples=1, max_eval_examples=1),
        step=2,
    )
    expected, _ = quantize_fp8_e4m3_per_row(originals["0"])

    assert torch.allclose(model[0].weight, expected)
    assert torch.allclose(originals["0"], torch.tensor([[0.0, 0.25, -1.0], [4.0, -8.0, 16.0]]))
    assert rows[0]["step"] == 2


def test_apply_staged_gptq_weights_interpolates_toward_target():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 3.0]]))
    originals = snapshot_linear_weights(model)
    targets = {"0": torch.tensor([[5.0, 7.0]])}

    rows = apply_staged_gptq_weights(model, originals, targets, alpha=0.25, step=1)

    assert torch.allclose(model[0].weight, torch.tensor([[2.0, 4.0]]))
    assert rows[0]["step"] == 1
    assert rows[0]["alpha"] == 0.25


def test_apply_damped_gptq_update_steps_from_current_weight():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 3.0]]))
    current = snapshot_linear_weights(model)
    targets = {"0": torch.tensor([[5.0, 7.0]])}

    rows = apply_damped_gptq_update(model, current, targets, step_alpha=0.25, step=3)

    assert torch.allclose(model[0].weight, torch.tensor([[2.0, 4.0]]))
    assert rows[0]["step"] == 3
    assert rows[0]["step_alpha"] == 0.25
    assert rows[0]["format"] == "damped_gptq_fp8_target"


def test_apply_gradient_descent_gptq_update_uses_diagonal_curvature():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[2.0, 2.0]]))
    current = snapshot_linear_weights(model)
    targets = {"0": torch.tensor([[0.0, 0.0]])}
    hessians = {"0": torch.tensor([1.0, 2.0], dtype=torch.float64)}

    rows = apply_gradient_descent_gptq_update(
        model,
        current,
        targets,
        hessians,
        damp_percent=0.0,
        gradient_step_scale=1.0,
        step=1,
    )

    assert torch.allclose(model[0].weight, torch.tensor([[1.0, 0.0]]))
    assert rows[0]["format"] == "gradient_descent_gptq_fp8_target"
    assert rows[0]["gradient_step_lr"] == 0.5


def test_apply_gradient_descent_gptq_update_identity_hessian_reaches_target():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[2.0, -3.0]]))
    current = snapshot_linear_weights(model)
    targets = {"0": torch.tensor([[0.5, 1.0]])}
    hessians = {"0": torch.eye(2, dtype=torch.float64)}

    rows = apply_gradient_descent_gptq_update(
        model,
        current,
        targets,
        hessians,
        damp_percent=0.0,
        gradient_step_scale=1.0,
        step=2,
    )

    assert torch.allclose(model[0].weight, targets["0"])
    assert rows[0]["step"] == 2
    assert rows[0]["gradient_lipschitz_bound"] == 1.0


def test_apply_gradient_descent_gptq_update_respects_step_scale():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[2.0, -3.0]]))
    current = snapshot_linear_weights(model)
    targets = {"0": torch.tensor([[0.5, 1.0]])}
    hessians = {"0": torch.eye(2, dtype=torch.float64)}

    apply_gradient_descent_gptq_update(
        model,
        current,
        targets,
        hessians,
        damp_percent=0.0,
        gradient_step_scale=0.5,
        step=1,
    )

    assert torch.allclose(model[0].weight, torch.tensor([[1.25, -1.0]]))


def test_set_linear_weights_restores_snapshot():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 2.0]]))
    snapshot = snapshot_linear_weights(model)
    with torch.no_grad():
        model[0].weight.zero_()

    set_linear_weights(model, snapshot)

    assert torch.allclose(model[0].weight, torch.tensor([[1.0, 2.0]]))
