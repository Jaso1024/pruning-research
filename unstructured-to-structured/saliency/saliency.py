from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import torch


class ParameterSaliencyAccumulator:
    def __init__(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]):
        self._scores: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.detach(), dtype=torch.float32, device="cpu")
            for name, param in named_parameters
            if param.requires_grad
        }

    def accumulate(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]], *, scale: float = 1.0) -> None:
        for name, param in named_parameters:
            if name not in self._scores or param.grad is None:
                continue
            score = (param.detach() * param.grad.detach()).abs()
            self._scores[name].add_(score.to(device="cpu", dtype=torch.float32), alpha=float(scale))

    def finalize(self, *, normalizer: float = 1.0) -> dict[str, torch.Tensor]:
        denom = max(float(normalizer), 1.0)
        return {name: score.div(denom) for name, score in self._scores.items()}


def parameter_summary_rows(scores: Mapping[str, torch.Tensor]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, tensor in scores.items():
        flat = tensor.float()
        numel = flat.numel()
        total = float(flat.sum().item())
        rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "numel": numel,
                "sum": total,
                "mean": float(total / numel) if numel else 0.0,
                "max": float(flat.max().item()) if numel else 0.0,
                "nonzero": int(torch.count_nonzero(flat).item()),
            }
        )
    rows.sort(key=lambda row: float(row["sum"]), reverse=True)
    return rows


def save_saliency_artifacts(
    output_dir: str | Path,
    scores: Mapping[str, torch.Tensor],
    metadata: Mapping[str, object],
    *,
    top_k: int = 50,
) -> dict[str, object]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = parameter_summary_rows(scores)
    summary = {
        "metadata": dict(metadata),
        "num_parameters": len(scores),
        "total_elements": int(sum(tensor.numel() for tensor in scores.values())),
        "total_saliency": float(sum(float(tensor.float().sum().item()) for tensor in scores.values())),
        "top_parameters": rows[:top_k],
        "artifacts": {
            "saliency_pt": str(out / "saliency.pt"),
            "summary_json": str(out / "summary.json"),
            "parameter_summary_jsonl": str(out / "parameter_summary.jsonl"),
        },
    }

    torch.save({"scores": dict(scores), "metadata": dict(metadata)}, out / "saliency.pt")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "parameter_summary.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    return summary
