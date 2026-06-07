from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from layer_distill.attention_viewer import (
    AttentionModelBundle,
    build_attention_payload,
    handle_api_request,
    index_html,
    summarize_attention_rows,
)


class FakeTokenizer:
    model_max_length = 2048

    def __call__(self, text: str, *, return_tensors: str, truncation: bool, max_length: int):
        assert return_tensors == "pt"
        assert truncation is True
        token_count = min(max(len(text.split()), 1), max_length)
        return {"input_ids": torch.arange(10, 10 + token_count).view(1, token_count)}

    def convert_ids_to_tokens(self, token_ids):
        return [f"tok_{int(token_id)}" for token_id in token_ids]

    def decode(self, token_ids, *, clean_up_tokenization_spaces: bool = False):
        return f"T{int(token_ids[0])}"


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=2,
            model_type="fake-gpt",
        )
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, *, input_ids, use_cache: bool, output_attentions: bool):
        assert use_cache is False
        assert output_attentions is True
        self.calls += 1
        seq_len = input_ids.shape[-1]
        layer0 = torch.zeros(1, 2, seq_len, seq_len)
        layer1 = torch.zeros(1, 2, seq_len, seq_len)
        layer1[0, 0] = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.75, 0.25, 0.0],
                [0.10, 0.30, 0.60],
            ]
        )[:seq_len, :seq_len]
        layer1[0, 1] = torch.eye(seq_len)
        return SimpleNamespace(attentions=(layer0, layer1))


def fake_bundle() -> AttentionModelBundle:
    return AttentionModelBundle(
        model_name="fake/model",
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        device=torch.device("cpu"),
        dtype="fp32",
    )


def test_summarize_attention_rows_reports_top_tokens_and_entropy():
    matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.75, 0.25, 0.0],
            [0.10, 0.30, 0.60],
        ]
    )
    tokens = [{"index": idx, "text": text} for idx, text in enumerate(["A", "B", "C"])]

    rows = summarize_attention_rows(matrix, tokens, top_k=2)

    assert rows[0]["query_token"] == "A"
    assert rows[0]["top_keys"][0] == {"token_index": 0, "token": "A", "weight": 1.0}
    assert rows[2]["top_keys"][0]["token"] == "C"
    assert rows[2]["top_keys"][1]["token"] == "B"
    assert rows[2]["entropy"] > rows[0]["entropy"]


def test_build_attention_payload_selects_layer_head_and_formats_tokens():
    payload = build_attention_payload("alpha beta gamma", layer=1, head=0, max_length=16, bundle=fake_bundle())

    assert payload["model"]["name"] == "fake/model"
    assert payload["model"]["layers"] == 2
    assert payload["model"]["heads"] == 2
    assert payload["selection"] == {"layer": 1, "head": 0}
    assert [token["text"] for token in payload["tokens"]] == ["T10", "T11", "T12"]
    assert payload["attention"][1] == pytest.approx([0.75, 0.25, 0.0])
    assert payload["rows"][2]["top_keys"][0]["token"] == "T12"


def test_build_attention_payload_validates_bounds_and_text():
    bundle = fake_bundle()

    with pytest.raises(ValueError, match="text"):
        build_attention_payload("   ", layer=0, head=0, max_length=16, bundle=bundle)
    with pytest.raises(ValueError, match="layer"):
        build_attention_payload("hello", layer=2, head=0, max_length=16, bundle=bundle)
    with pytest.raises(ValueError, match="head"):
        build_attention_payload("hello", layer=0, head=2, max_length=16, bundle=bundle)
    with pytest.raises(ValueError, match="max_length"):
        build_attention_payload("hello", layer=0, head=0, max_length=0, bundle=bundle)


def test_api_dispatcher_serves_model_metadata_and_attention_payload():
    bundle = fake_bundle()
    provider = lambda: bundle

    status, headers, body = handle_api_request("/api/model", b"", bundle_provider=provider)
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert json.loads(body)["layers"] == 2

    status, _, body = handle_api_request(
        "/api/attention",
        json.dumps({"text": "alpha beta gamma", "layer": 1, "head": 0, "max_length": 16}).encode(),
        bundle_provider=provider,
    )

    assert status == 200
    payload = json.loads(body)
    assert payload["attention"][2] == pytest.approx([0.10, 0.30, 0.60])


def test_api_dispatcher_reports_bad_requests_as_json_errors():
    status, headers, body = handle_api_request(
        "/api/attention",
        json.dumps({"text": "", "layer": 0, "head": 0}).encode(),
        bundle_provider=fake_bundle,
    )

    assert status == 400
    assert headers["Content-Type"] == "application/json"
    assert "text" in json.loads(body)["error"]


def test_index_html_contains_viewer_controls_and_canvas():
    html = index_html()

    assert "attention-canvas" in html
    assert "layer-select" in html
    assert "head-select" in html
    assert "/api/attention" in html
