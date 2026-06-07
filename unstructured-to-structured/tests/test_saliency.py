from __future__ import annotations

import json

import torch

from saliency.saliency import ParameterSaliencyAccumulator, parameter_summary_rows, save_saliency_artifacts


def test_parameter_saliency_accumulator_tracks_abs_weight_times_grad():
    model = torch.nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -3.0]]))
        model.bias.copy_(torch.tensor([0.5]))

    output = model(torch.tensor([[4.0, -5.0]]))
    output.sum().backward()

    acc = ParameterSaliencyAccumulator(model.named_parameters())
    acc.accumulate(model.named_parameters(), scale=2.0)
    scores = acc.finalize(normalizer=4.0)

    expected_weight = torch.tensor([[abs(2.0 * 4.0) * 2.0 / 4.0, abs(-3.0 * -5.0) * 2.0 / 4.0]])
    expected_bias = torch.tensor([abs(0.5 * 1.0) * 2.0 / 4.0])
    assert torch.allclose(scores["weight"], expected_weight)
    assert torch.allclose(scores["bias"], expected_bias)


def test_parameter_summary_rows_are_sorted_by_total_score():
    scores = {
        "small": torch.tensor([1.0, 2.0]),
        "large": torch.tensor([[10.0, 1.0]]),
    }

    rows = parameter_summary_rows(scores)

    assert [row["name"] for row in rows] == ["large", "small"]
    assert rows[0]["numel"] == 2
    assert rows[0]["shape"] == [1, 2]
    assert rows[0]["sum"] == 11.0


def test_save_saliency_artifacts_writes_summary_json_jsonl_and_pt(tmp_path):
    scores = {"layer.weight": torch.tensor([[1.0, 2.0]])}
    metadata = {"model_name": "tiny", "supervised_tokens": 3}

    summary = save_saliency_artifacts(tmp_path, scores, metadata, top_k=1)

    assert (tmp_path / "saliency.pt").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "parameter_summary.jsonl").exists()
    assert summary["metadata"] == metadata
    assert summary["top_parameters"][0]["name"] == "layer.weight"

    saved_summary = json.loads((tmp_path / "summary.json").read_text())
    saved_rows = [json.loads(line) for line in (tmp_path / "parameter_summary.jsonl").read_text().splitlines()]
    saved_pt = torch.load(tmp_path / "saliency.pt", map_location="cpu", weights_only=False)

    assert saved_summary["top_parameters"][0]["sum"] == 3.0
    assert saved_rows[0]["mean"] == 1.5
    assert torch.equal(saved_pt["scores"]["layer.weight"], scores["layer.weight"])
