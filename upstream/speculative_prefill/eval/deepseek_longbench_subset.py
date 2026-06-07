import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download

from speculative_prefill.api_eval.deepseek_client import (
    DeepSeekClient,
    load_deepseek_api_key,
)
from speculative_prefill.api_eval.pruning import (
    AttentionTextCompressor,
    CompressionResult,
    EmbeddingNormTextCompressor,
    MiddleLayerNormTextCompressor,
)
from speculative_prefill.api_eval.results import group_scores, score_longbench_predictions


LOCAL_DIR = ROOT / "local" / "deepseek_accuracy"
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
LONGBENCH_E_NO_8K_DATASETS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "repobench-p",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def parse_limit(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    normalized = value.strip().lower()
    if normalized in {"all", "none", "null"}:
        return None
    limit = int(normalized)
    if limit < 0:
        raise ValueError("--limit must be non-negative or all")
    return limit


def expand_dataset_spec(spec: str) -> list[str]:
    normalized = spec.strip()
    if normalized == "all":
        return list(FULL_LONGBENCH_DATASETS)
    if normalized in {"paper", "longbench"}:
        return list(FULL_LONGBENCH_DATASETS)
    if normalized in {"no_8k", "longbench_e_no_8k"}:
        return list(LONGBENCH_E_NO_8K_DATASETS)
    return [item.strip() for item in spec.split(",") if item.strip()]


def _load_longbench_dataset(dataset_name: str) -> list[dict]:
    zip_path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    member = f"data/{dataset_name}.jsonl"
    with ZipFile(zip_path) as archive:
        with archive.open(member) as handle:
            return [json.loads(line) for line in handle.read().decode("utf-8").splitlines() if line]


def _compress_context(prompt_format: str, sample: dict, compressor) -> tuple[str, CompressionResult]:
    compressed = compressor.compress(sample["context"])
    sample = dict(sample)
    sample["context"] = compressed.text
    return prompt_format.format(**sample), compressed


def _identity_compression(prompt: str) -> CompressionResult:
    return CompressionResult(
        text=prompt,
        original_tokens=0,
        kept_tokens=0,
        spans=[],
    )


def _make_compressors(args):
    compressors = [("baseline", "none", None)]
    for model_name in args.scorer_models.split(","):
        model_name = model_name.strip()
        if not model_name:
            continue
        compressors.append((
            "embedding_norm",
            model_name,
            EmbeddingNormTextCompressor(
                model_name,
                percentage=args.keep_rate,
                chunk_size=args.chunk_size,
                delimiter=args.delimiter,
                device=args.device,
            ),
        ))
    for model_name in args.draft_models.split(","):
        model_name = model_name.strip()
        if not model_name:
            continue
        compressors.append((
            "cross_family_spec_prefill",
            model_name,
            AttentionTextCompressor(
                model_name,
                percentage=args.keep_rate,
                chunk_size=args.chunk_size,
                lookahead=args.lookahead,
                delimiter=args.delimiter,
                device=args.device if args.device != "auto" else None,
            ),
        ))
    for model_name in args.first_layer_draft_models.split(","):
        model_name = model_name.strip()
        if not model_name:
            continue
        compressors.append((
            "first_layer_cross_family_spec_prefill",
            model_name,
            AttentionTextCompressor(
                model_name,
                percentage=args.keep_rate,
                chunk_size=args.chunk_size,
                lookahead=args.lookahead,
                attention_layer_index=1,
                delimiter=args.delimiter,
                device=args.device if args.device != "auto" else None,
            ),
        ))
    for model_name in args.middle_layer_models.split(","):
        model_name = model_name.strip()
        if not model_name:
            continue
        compressors.append((
            "middle_layer_norm",
            model_name,
            MiddleLayerNormTextCompressor(
                model_name,
                percentage=args.keep_rate,
                chunk_size=args.chunk_size,
                delimiter=args.delimiter,
                device=args.device if args.device != "auto" else None,
                layer_fraction=args.layer_fraction,
            ),
        ))
    for model_name in args.first_attn_models.split(","):
        model_name = model_name.strip()
        if not model_name:
            continue
        compressors.append((
            "first_attn_norm",
            model_name,
            MiddleLayerNormTextCompressor(
                model_name,
                percentage=args.keep_rate,
                chunk_size=args.chunk_size,
                delimiter=args.delimiter,
                device=args.device if args.device != "auto" else None,
                layer_index=1,
                activation_target="attn",
            ),
        ))
    for model_name in args.first_ffn_models.split(","):
        model_name = model_name.strip()
        if not model_name:
            continue
        compressors.append((
            "first_ffn_norm",
            model_name,
            MiddleLayerNormTextCompressor(
                model_name,
                percentage=args.keep_rate,
                chunk_size=args.chunk_size,
                delimiter=args.delimiter,
                device=args.device if args.device != "auto" else None,
                layer_index=1,
                activation_target="ffn",
            ),
        ))
    return compressors


def prepare_rows(args) -> list[dict]:
    dataset2prompt = _load_json(ROOT / "eval" / "long_bench" / "configs" / "dataset2prompt.json")
    dataset2maxlen = _load_json(ROOT / "eval" / "long_bench" / "configs" / "dataset2maxlen.json")
    rows = []
    compressors = _make_compressors(args)
    sample_limit = parse_limit(args.limit)
    for dataset_name in expand_dataset_spec(args.datasets):
        data = _load_longbench_dataset(dataset_name)
        prompt_format = dataset2prompt[dataset_name]
        max_gen = min(dataset2maxlen[dataset_name], args.max_tokens)
        limited = data if sample_limit is None else data[:sample_limit]
        sample_end = None if args.sample_count is None else args.sample_start + args.sample_count
        samples = limited[args.sample_start:sample_end]
        for offset, sample in enumerate(samples):
            sample_idx = args.sample_start + offset
            baseline_prompt = prompt_format.format(**sample)
            for method, local_model, compressor in compressors:
                start = time.perf_counter()
                if compressor is None:
                    prompt = baseline_prompt
                    compression = _identity_compression(prompt)
                else:
                    prompt, compression = _compress_context(prompt_format, sample, compressor)
                compression_s = time.perf_counter() - start
                rows.append({
                    "dataset": dataset_name,
                    "sample_idx": sample_idx,
                    "method": method,
                    "local_model": local_model,
                    "target_model": args.deepseek_model,
                    "keep_rate": args.keep_rate if method != "baseline" else 1.0,
                    "chunk_size": args.chunk_size,
                    "max_tokens": max_gen,
                    "messages": _build_messages(prompt),
                    "stop": "\n" if dataset_name == "samsum" else None,
                    "answers": sample["answers"],
                    "all_classes": sample["all_classes"],
                    "length": sample["length"],
                    "original_local_tokens": compression.original_tokens,
                    "kept_local_tokens": compression.kept_tokens,
                    "compressed_chars": len(prompt),
                    "compression_s": compression_s,
                })
    return rows


async def evaluate_rows(rows: list[dict], args) -> list[dict]:
    api_key = load_deepseek_api_key(args.env_file)
    client = DeepSeekClient(
        api_key=api_key,
        model=args.deepseek_model,
        timeout=args.timeout,
        retries=args.retries,
        concurrency=args.concurrency,
    )
    try:
        jobs = [
            (idx, row["messages"], row["max_tokens"], row.get("stop"))
            for idx, row in enumerate(rows)
        ]
        async for idx, result in client.run_jobs(jobs):
            apply_deepseek_result(
                rows[idx],
                result,
                continue_on_error=args.continue_on_api_error,
            )
    finally:
        await client.close()

    return rows


def apply_deepseek_result(row: dict, result, *, continue_on_error: bool) -> None:
    if isinstance(result, Exception):
        if not continue_on_error:
            raise result
        row["pred"] = ""
        row["api_error"] = str(result)
        row["usage"] = {}
        return
    row["pred"] = result.content
    row["api_latency_s"] = result.latency_s
    row["usage"] = result.usage


def write_outputs(rows: list[dict], args) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    out_dir = LOCAL_DIR / f"longbench_deepseek_subset_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            output_row = dict(row)
            if not args.include_prompts:
                output_row.pop("messages", None)
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
    scored_rows = [row for row in rows if "pred" in row]
    if scored_rows:
        by_method = {}
        for method in sorted({row["method"] for row in scored_rows}):
            method_rows = [row for row in scored_rows if row["method"] == method]
            by_method[method] = score_longbench_predictions(method_rows)
        (out_dir / "scores.json").write_text(
            json.dumps(by_method, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / "scores_by_method_model.json").write_text(
            json.dumps(
                group_scores(scored_rows, ("method", "local_model")),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return out_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="triviaqa,passage_retrieval_en")
    parser.add_argument("--limit", default="2")
    parser.add_argument("--deepseek-model", default="deepseek-v4-flash")
    parser.add_argument("--scorer-models", default="Qwen/Qwen3-0.6B,HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--draft-models", default="Qwen/Qwen3-0.6B,HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--first-layer-draft-models", default="")
    parser.add_argument("--middle-layer-models", default="")
    parser.add_argument("--first-attn-models", default="")
    parser.add_argument("--first-ffn-models", default="")
    parser.add_argument("--keep-rate", type=float, default=0.3)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--layer-fraction", type=float, default=0.5)
    parser.add_argument("--delimiter", default="\n...\n")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-prompts", action="store_true")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--continue-on-api-error", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = prepare_rows(args)
    if not args.dry_run:
        rows = asyncio.run(evaluate_rows(rows, args))
    out_dir = write_outputs(rows, args)
    print(f"Wrote {out_dir}")
    if (out_dir / "scores.json").exists():
        print((out_dir / "scores.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
