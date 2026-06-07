import json
from pathlib import Path

import pytest
import torch

from layer_distill.low_qk_model import (
    EndToEndLowQKConfig,
    LowQKLogitDistillConfig,
    LowQKPerplexityEvalConfig,
    causal_lm_nll_from_logits,
    kl_divergence_sum_from_logits,
    LowQKGPTNeoXAttention,
    find_low_qk_adapter_checkpoints,
    load_low_qk_adapter_checkpoint,
    perplexity_from_nll,
    replace_gpt_neox_attentions_with_low_qk,
    summarize_end_to_end_low_qk_runs,
)


class TinyTeacherAttention(torch.nn.Module):
    def __init__(self, hidden_size: int = 8, heads: int = 2):
        super().__init__()
        self.num_attention_heads = heads
        self.head_size = hidden_size // heads
        self.query_key_value = torch.nn.Linear(hidden_size, 3 * hidden_size)
        self.dense = torch.nn.Linear(hidden_size, hidden_size)


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = TinyTeacherAttention()
        self.mlp = torch.nn.Linear(8, 8)


class TinyModel(torch.nn.Module):
    def __init__(self, layers: int = 3):
        super().__init__()
        self.layers = torch.nn.ModuleList(TinyLayer() for _ in range(layers))
        self.embed = torch.nn.Linear(8, 8)
        self.config = type("Config", (), {"hidden_size": 8, "num_attention_heads": 2})()


def test_low_qk_gpt_neox_attention_adapter_contract():
    adapter = LowQKGPTNeoXAttention(hidden_size=16, num_heads=4, qk_dim=2)
    hidden = torch.randn(2, 5, 16)

    output, weights = adapter(hidden, attention_mask=None, position_embeddings=None)

    assert output.shape == hidden.shape
    assert weights is None


def test_replace_gpt_neox_attentions_replaces_all_and_freezes_base():
    model = TinyModel(layers=3)

    adapters = replace_gpt_neox_attentions_with_low_qk(model, qk_dim=2)

    assert len(adapters) == 3
    assert all(isinstance(layer.attention, LowQKGPTNeoXAttention) for layer in model.layers)
    assert all(param.requires_grad for adapter in adapters for param in adapter.parameters())
    assert not any(param.requires_grad for layer in model.layers for param in layer.mlp.parameters())
    assert not any(param.requires_grad for param in model.embed.parameters())


def test_replace_gpt_neox_attentions_preserves_teacher_attention_dtype():
    model = TinyModel(layers=1).to(dtype=torch.float64)

    adapters = replace_gpt_neox_attentions_with_low_qk(model, qk_dim=2)

    assert adapters[0].low_qk.q_proj.weight.dtype == torch.float64


def test_end_to_end_low_qk_config_validates_inputs(tmp_path: Path):
    config = EndToEndLowQKConfig(output_dir=tmp_path, learning_rates=(3e-2,), qk_dim=2)
    assert config.qk_dim == 2

    with pytest.raises(ValueError):
        EndToEndLowQKConfig(output_dir=tmp_path, learning_rates=())
    with pytest.raises(ValueError):
        EndToEndLowQKConfig(output_dir=tmp_path, learning_rates=(-3e-2,))
    with pytest.raises(ValueError):
        EndToEndLowQKConfig(output_dir=tmp_path, qk_dim=0)
    with pytest.raises(ValueError):
        EndToEndLowQKConfig(output_dir=tmp_path, student_heads=0)


def test_summarize_end_to_end_low_qk_runs_picks_best_relative_mse(tmp_path: Path):
    run_a = tmp_path / "lr_0.001"
    run_b = tmp_path / "lr_0.003"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "steps.jsonl").write_text(json.dumps({"step": 2, "final_rel_mse": 0.4}) + "\n")
    (run_b / "steps.jsonl").write_text(json.dumps({"step": 2, "final_rel_mse": 0.2}) + "\n")

    summary = summarize_end_to_end_low_qk_runs(tmp_path)

    assert summary["best"]["run_dir"] == str(run_b)
    assert summary["best"]["final_rel_mse"] == 0.2


def test_find_low_qk_adapter_checkpoints_sorts_by_lr(tmp_path: Path):
    for name in ("lr_0.03", "lr_0.003", "lr_0.01"):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "low_qk_adapters.pt").write_bytes(b"placeholder")

    paths = find_low_qk_adapter_checkpoints(tmp_path)

    assert [path.parent.name for path in paths] == ["lr_0.003", "lr_0.01", "lr_0.03"]


def test_load_low_qk_adapter_checkpoint_loads_all_layers(tmp_path: Path):
    model = TinyModel(layers=2)
    adapters = replace_gpt_neox_attentions_with_low_qk(model, qk_dim=2)
    checkpoint = {}
    for idx, adapter in enumerate(adapters):
        state = {key: value.detach().clone() for key, value in adapter.state_dict().items()}
        for value in state.values():
            value.add_(1.0)
        checkpoint[str(idx)] = state
    path = tmp_path / "low_qk_adapters.pt"
    torch.save(checkpoint, path)

    load_low_qk_adapter_checkpoint(adapters, path)

    loaded = adapters[0].state_dict()["low_qk.q_proj.bias"]
    assert torch.allclose(loaded, checkpoint["0"]["low_qk.q_proj.bias"])


def test_low_qk_perplexity_eval_config_validates_inputs(tmp_path: Path):
    config = LowQKPerplexityEvalConfig(output_dir=tmp_path, adapter_root=tmp_path, eval_steps=2)
    assert config.eval_steps == 2

    with pytest.raises(ValueError):
        LowQKPerplexityEvalConfig(output_dir=tmp_path, adapter_root=tmp_path, eval_steps=0)
    with pytest.raises(ValueError):
        LowQKPerplexityEvalConfig(output_dir=tmp_path, adapter_root=tmp_path, seq_len=1)


def test_low_qk_logit_distill_config_validates_inputs(tmp_path: Path):
    config = LowQKLogitDistillConfig(output_dir=tmp_path, learning_rates=(1e-3,), temperature=2.0)
    assert config.temperature == 2.0

    with pytest.raises(ValueError):
        LowQKLogitDistillConfig(output_dir=tmp_path, learning_rates=())
    with pytest.raises(ValueError):
        LowQKLogitDistillConfig(output_dir=tmp_path, temperature=0.0)
    with pytest.raises(ValueError):
        LowQKLogitDistillConfig(output_dir=tmp_path, logit_chunk_tokens=0)


def test_perplexity_from_nll():
    assert perplexity_from_nll(total_nll=0.0, total_tokens=10) == 1.0

    with pytest.raises(ValueError):
        perplexity_from_nll(total_nll=1.0, total_tokens=0)


def test_causal_lm_nll_from_logits_matches_cross_entropy():
    logits = torch.randn(2, 5, 7)
    labels = torch.randint(0, 7, (2, 5))

    nll, tokens = causal_lm_nll_from_logits(logits, labels, chunk_tokens=3)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)).float(),
        labels[:, 1:].reshape(-1),
        reduction="sum",
    )

    assert tokens == 8
    assert torch.allclose(nll, expected)


def test_kl_divergence_sum_from_logits_matches_torch_kl():
    student_logits = torch.randn(4, 9)
    teacher_logits = torch.randn(4, 9)
    temperature = 1.7

    actual = kl_divergence_sum_from_logits(student_logits, teacher_logits, temperature=temperature)
    expected = torch.nn.functional.kl_div(
        torch.log_softmax(student_logits.float() / temperature, dim=-1),
        torch.softmax(teacher_logits.float() / temperature, dim=-1),
        reduction="sum",
    ) * (temperature * temperature)

    assert torch.allclose(actual, expected)
