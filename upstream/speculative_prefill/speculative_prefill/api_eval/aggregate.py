import json
from dataclasses import dataclass
from pathlib import Path

from speculative_prefill.api_eval.modal_commands import (
    LONGBENCH_SAMPLE_COUNTS,
    sample_block_shards,
)
from speculative_prefill.api_eval.results import group_scores, score_longbench_predictions


@dataclass(frozen=True)
class Shard:
    path: Path
    dataset: str
    sample_start: int
    sample_count: int
    rows: list[dict]


def collect_shards(root: Path, *, requested_datasets: str = "all") -> list[Shard]:
    shards = []
    for path in sorted(root.glob("modal_longbench_deepseek_*_s*_n25_*")):
        config_path = path / "run_config.json"
        pred_path = path / "predictions.jsonl"
        if not config_path.exists() or not pred_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("dry_run") or config.get("requested_datasets") != requested_datasets:
            continue
        rows = [
            json.loads(line)
            for line in pred_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        shards.append(Shard(
            path=path,
            dataset=config["datasets"],
            sample_start=int(config["sample_start"]),
            sample_count=int(config["sample_count"]),
            rows=rows,
        ))
    return shards


def expected_block_counts(block_size: int = 25) -> dict[str, int]:
    counts = {}
    for dataset, total in LONGBENCH_SAMPLE_COUNTS.items():
        counts[dataset] = len(sample_block_shards([dataset], limit=total, block_size=block_size))
    return counts


def summarize_shards(shards: list[Shard], *, block_size: int = 25) -> dict:
    rows = [row for shard in shards for row in shard.rows]
    expected_counts = expected_block_counts(block_size)
    starts_by_dataset: dict[str, set[int]] = {}
    for shard in shards:
        starts_by_dataset.setdefault(shard.dataset, set()).add(shard.sample_start)

    coverage = {}
    for dataset, expected in expected_counts.items():
        starts = sorted(starts_by_dataset.get(dataset, set()))
        coverage[dataset] = {
            "completed_blocks": len(starts),
            "expected_blocks": expected,
            "completed_samples": sum(
                min(block_size, LONGBENCH_SAMPLE_COUNTS[dataset] - start)
                for start in starts
            ),
            "expected_samples": LONGBENCH_SAMPLE_COUNTS[dataset],
            "starts": starts,
        }

    scores_by_method_model = group_scores(rows, ("method", "local_model"))
    macro_scores = {
        label: round(
            sum(dataset_scores["score"] for dataset_scores in scores.values()) / len(scores),
            2,
        )
        for label, scores in scores_by_method_model.items()
        if scores
    }
    api_errors_by_method_model: dict[str, int] = {}
    for row in rows:
        if "api_error" not in row:
            continue
        label = f"{row['method']} | {row['local_model']}"
        api_errors_by_method_model[label] = api_errors_by_method_model.get(label, 0) + 1

    return {
        "shard_count": len(shards),
        "expected_shards": sum(expected_counts.values()),
        "row_count": len(rows),
        "expected_rows": sum(LONGBENCH_SAMPLE_COUNTS.values()) * 3,
        "sample_count": len(rows) // 3,
        "expected_samples": sum(LONGBENCH_SAMPLE_COUNTS.values()),
        "api_error_count": sum(api_errors_by_method_model.values()),
        "api_errors_by_method_model": api_errors_by_method_model,
        "coverage": coverage,
        "scores_by_method": score_longbench_predictions(rows),
        "scores_by_method_model": scores_by_method_model,
        "macro_scores_by_method_model": macro_scores,
    }
