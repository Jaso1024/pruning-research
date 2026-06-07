from __future__ import annotations

import pytest

from saliency.experiment import SaliencyConfig, resolve_torch_dtype


def test_saliency_config_defaults_to_gsm8k_main_for_pythia31m():
    config = SaliencyConfig(output_dir="runs/test")

    assert config.model_name == "EleutherAI/pythia-31m"
    assert config.dataset_name == "openai/gsm8k"
    assert config.dataset_config == "main"
    assert config.answer_only_loss is True


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("fp32", "float32"),
        ("float32", "float32"),
        ("bf16", "bfloat16"),
        ("bfloat16", "bfloat16"),
        ("fp16", "float16"),
        ("float16", "float16"),
    ],
)
def test_resolve_torch_dtype(name, expected):
    assert str(resolve_torch_dtype(name)).rsplit(".", 1)[-1] == expected


def test_resolve_torch_dtype_rejects_unknown_name():
    with pytest.raises(ValueError, match="dtype"):
        resolve_torch_dtype("int8")
