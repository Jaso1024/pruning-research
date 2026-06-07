import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal


REMOTE_ROOT = "/root/speculative_prefill"
LOCAL_ROOT = Path(__file__).resolve().parents[1]
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))
if REMOTE_ROOT not in sys.path:
    sys.path.insert(0, REMOTE_ROOT)

from speculative_prefill.api_eval.modal_commands import (
    build_deepseek_longbench_command,
    completed_shard_exists,
    expand_dataset_spec,
    output_run_name,
    sample_block_shards,
)

secret_names = ["DEEPSEEK_API_KEY"]
if "HF_TOKEN" in os.environ:
    secret_names.append("HF_TOKEN")
SECRETS = [modal.Secret.from_local_environ(secret_names)]

app = modal.App("spec-prefill-deepseek-accuracy")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("httpx")
    .add_local_dir(
        ".",
        remote_path=REMOTE_ROOT,
        copy=True,
        ignore=[
            ".git",
            "**/__pycache__",
            "**/*.pyc",
            "local",
        ],
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=3 * 60 * 60,
    secrets=SECRETS,
)
def run_accuracy(
    datasets: str,
    limit: str,
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
) -> dict[str, str]:
    command = build_deepseek_longbench_command(
        datasets=datasets,
        limit=limit,
        deepseek_model=deepseek_model,
        scorer_models=scorer_models,
        draft_models=draft_models,
        first_layer_draft_models=first_layer_draft_models,
        middle_layer_models=middle_layer_models,
        first_attn_models=first_attn_models,
        first_ffn_models=first_ffn_models,
        keep_rate=keep_rate,
        chunk_size=chunk_size,
        lookahead=lookahead,
        max_tokens=max_tokens,
        concurrency=concurrency,
        dry_run=dry_run,
        sample_start=sample_start,
        sample_count=sample_count,
        continue_on_api_error=continue_on_api_error,
    )
    before = set((Path(REMOTE_ROOT) / "local" / "deepseek_accuracy").glob("longbench_deepseek_subset_*"))
    result = subprocess.run(
        command,
        cwd=REMOTE_ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout)

    output_root = Path(REMOTE_ROOT) / "local" / "deepseek_accuracy"
    after = set(output_root.glob("longbench_deepseek_subset_*"))
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if not created:
        raise RuntimeError(f"No output directory created.\n{result.stdout}")
    out_dir = created[-1]
    payload = {"stdout": result.stdout, "remote_dir": str(out_dir)}
    for name in ("predictions.jsonl", "scores.json", "scores_by_method_model.json"):
        path = out_dir / name
        if path.exists():
            payload[name] = path.read_text(encoding="utf-8")
    return payload


@app.local_entrypoint()
def main(
    datasets: str = "passage_retrieval_en",
    limit: str = "1",
    deepseek_model: str = "deepseek-chat",
    scorer_models: str = "Qwen/Qwen3-0.6B",
    draft_models: str = "Qwen/Qwen3-0.6B",
    first_layer_draft_models: str = "",
    middle_layer_models: str = "",
    first_attn_models: str = "",
    first_ffn_models: str = "",
    keep_rate: float = 0.3,
    chunk_size: int = 32,
    lookahead: int = 1,
    max_tokens: int = 32,
    concurrency: int = 4,
    dry_run: bool = False,
    output_dir: str = "local/deepseek_accuracy",
    shard_by_dataset: bool = False,
    sample_block_size: int = 0,
    skip_existing: bool = False,
    continue_on_api_error: bool = False,
):
    if sample_block_size:
        shards = sample_block_shards(
            expand_dataset_spec(datasets),
            limit=limit,
            block_size=sample_block_size,
        )
    else:
        dataset_shards = expand_dataset_spec(datasets) if shard_by_dataset else [datasets]
        shards = [(dataset_shard, 0, None) for dataset_shard in dataset_shards]
    written = []
    for dataset_shard, sample_start, sample_count in shards:
        output_root = LOCAL_ROOT / output_dir
        if skip_existing and completed_shard_exists(
            output_root,
            datasets,
            dataset_shard,
            sample_start,
            sample_count,
            dry_run,
        ):
            print(f"Skipping existing shard {dataset_shard} start={sample_start} count={sample_count}")
            continue
        payload = run_accuracy.remote(
            datasets=dataset_shard,
            limit=limit,
            deepseek_model=deepseek_model,
            scorer_models=scorer_models,
            draft_models=draft_models,
            first_layer_draft_models=first_layer_draft_models,
            middle_layer_models=middle_layer_models,
            first_attn_models=first_attn_models,
            first_ffn_models=first_ffn_models,
            keep_rate=keep_rate,
            chunk_size=chunk_size,
            lookahead=lookahead,
            max_tokens=max_tokens,
            concurrency=concurrency,
            dry_run=dry_run,
            sample_start=sample_start,
            sample_count=sample_count,
            continue_on_api_error=continue_on_api_error,
        )

        timestamp = time.strftime("%Y%m%dT%H%M%S")
        out_dir = LOCAL_ROOT / output_dir / output_run_name(
            timestamp,
            dataset_shard,
            shard_by_dataset or bool(sample_block_size),
            sample_start,
            sample_count,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "modal_stdout.txt").write_text(payload["stdout"], encoding="utf-8")
        (out_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "datasets": dataset_shard,
                    "requested_datasets": datasets,
                    "limit": limit,
                    "deepseek_model": deepseek_model,
                    "scorer_models": scorer_models,
                    "draft_models": draft_models,
                    "first_layer_draft_models": first_layer_draft_models,
                    "middle_layer_models": middle_layer_models,
                    "first_attn_models": first_attn_models,
                    "first_ffn_models": first_ffn_models,
                    "keep_rate": keep_rate,
                    "chunk_size": chunk_size,
                    "lookahead": lookahead,
                    "max_tokens": max_tokens,
                    "concurrency": concurrency,
                    "dry_run": dry_run,
                    "shard_by_dataset": shard_by_dataset,
                    "sample_block_size": sample_block_size,
                    "skip_existing": skip_existing,
                    "continue_on_api_error": continue_on_api_error,
                    "sample_start": sample_start,
                    "sample_count": sample_count,
                    "remote_dir": payload["remote_dir"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        for name in ("predictions.jsonl", "scores.json", "scores_by_method_model.json"):
            if name in payload:
                (out_dir / name).write_text(payload[name], encoding="utf-8")

        print(f"Wrote {out_dir}")
        if "scores.json" in payload:
            print(payload["scores.json"])
        written.append(str(out_dir))
    print(json.dumps({"written": written}, indent=2))
