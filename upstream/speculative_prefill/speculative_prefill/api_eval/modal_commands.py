import json
from pathlib import Path


FULL_LONGBENCH_DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]
LONGBENCH_SAMPLE_COUNTS = {
    "narrativeqa": 200,
    "qasper": 200,
    "multifieldqa_en": 150,
    "multifieldqa_zh": 200,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "dureader": 200,
    "gov_report": 200,
    "qmsum": 200,
    "multi_news": 200,
    "vcsum": 200,
    "trec": 200,
    "triviaqa": 200,
    "samsum": 200,
    "lsht": 200,
    "passage_count": 200,
    "passage_retrieval_en": 200,
    "passage_retrieval_zh": 200,
    "lcc": 500,
    "repobench-p": 500,
}


def expand_dataset_spec(spec: str) -> list[str]:
    normalized = spec.strip()
    if normalized in {"all", "paper", "longbench"}:
        return list(FULL_LONGBENCH_DATASETS)
    return [item.strip() for item in spec.split(",") if item.strip()]


def _limit_count(dataset: str, limit: int | str) -> int:
    total = LONGBENCH_SAMPLE_COUNTS[dataset]
    if isinstance(limit, int):
        return min(total, limit)
    normalized = limit.strip().lower()
    if normalized in {"all", "none", "null"}:
        return total
    return min(total, int(normalized))


def sample_block_shards(
    datasets: list[str],
    *,
    limit: int | str,
    block_size: int,
) -> list[tuple[str, int, int]]:
    if block_size <= 0:
        return [(dataset, 0, _limit_count(dataset, limit)) for dataset in datasets]
    shards = []
    for dataset in datasets:
        count = _limit_count(dataset, limit)
        for start in range(0, count, block_size):
            shards.append((dataset, start, min(block_size, count - start)))
    return shards


def build_deepseek_longbench_command(
    *,
    datasets: str,
    limit: int | str,
    deepseek_model: str,
    scorer_models: str,
    draft_models: str,
    first_layer_draft_models: str,
    middle_layer_models: str,
    first_attn_models: str,
    first_ffn_models: str,
    keep_rate: float,
    chunk_size: int,
    lookahead: int,
    max_tokens: int,
    concurrency: int,
    dry_run: bool,
    sample_start: int,
    sample_count: int | None,
    continue_on_api_error: bool,
) -> list[str]:
    command = [
        "python",
        "eval/deepseek_longbench_subset.py",
        "--datasets",
        datasets,
        "--limit",
        str(limit),
        "--deepseek-model",
        deepseek_model,
        "--scorer-models",
        scorer_models,
        "--draft-models",
        draft_models,
        "--first-layer-draft-models",
        first_layer_draft_models,
        "--middle-layer-models",
        middle_layer_models,
        "--first-attn-models",
        first_attn_models,
        "--first-ffn-models",
        first_ffn_models,
        "--keep-rate",
        str(keep_rate),
        "--chunk-size",
        str(chunk_size),
        "--lookahead",
        str(lookahead),
        "--max-tokens",
        str(max_tokens),
        "--concurrency",
        str(concurrency),
        "--device",
        "cuda",
        "--sample-start",
        str(sample_start),
    ]
    if sample_count is not None:
        command.extend(["--sample-count", str(sample_count)])
    if continue_on_api_error:
        command.append("--continue-on-api-error")
    if dry_run:
        command.append("--dry-run")
    return command


def output_run_name(
    timestamp: str,
    dataset: str,
    shard_by_dataset: bool,
    sample_start: int = 0,
    sample_count: int | None = None,
) -> str:
    if shard_by_dataset or sample_start or sample_count is not None:
        suffix = ""
        if sample_start or sample_count is not None:
            suffix = f"_s{sample_start}_n{sample_count if sample_count is not None else 'all'}"
        return f"modal_longbench_deepseek_{dataset}{suffix}_{timestamp}"
    return f"modal_longbench_deepseek_subset_{timestamp}"


def completed_shard_exists(
    root: Path,
    requested_datasets: str,
    dataset: str,
    sample_start: int,
    sample_count: int | None,
    dry_run: bool,
) -> bool:
    for path in root.glob(f"modal_longbench_deepseek_{dataset}_s{sample_start}_n*_*"):
        config_path = path / "run_config.json"
        pred_path = path / "predictions.jsonl"
        if not config_path.exists() or not pred_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if (
            config.get("requested_datasets") == requested_datasets
            and config.get("datasets") == dataset
            and int(config.get("sample_start", -1)) == sample_start
            and config.get("sample_count") == sample_count
            and bool(config.get("dry_run")) == dry_run
        ):
            return True
    return False
