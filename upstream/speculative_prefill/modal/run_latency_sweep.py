import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal


REMOTE_ROOT = "/root/speculative_prefill"
SECRETS = [modal.Secret.from_local_environ(["HF_TOKEN"])] if "HF_TOKEN" in os.environ else []

app = modal.App("spec-prefill-latency-sweep")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
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
    timeout=60 * 60,
    secrets=SECRETS,
)
def run_mode_sweep(
    mode: str,
    model: str,
    spec_model: str,
    config: str,
    input_lens: str,
    batch_sizes: str,
    output_len: int,
    warmup_iters: int,
    iters: int,
) -> str:
    output_json = f"/tmp/{mode}_sweep.json"
    cmd = [
        "python",
        "-m",
        "speculative_prefill.vllm_benchmarks.sweep_latency",
        "--mode",
        mode,
        "--model",
        model,
        "--input-lens",
        input_lens,
        "--batch-sizes",
        batch_sizes,
        "--output-len",
        str(output_len),
        "--warmup-iters",
        str(warmup_iters),
        "--iters",
        str(iters),
        "--output-json",
        output_json,
    ]
    if mode != "baseline":
        cmd.extend(["--config", config])
    if mode == "spec_prefill":
        cmd.extend(["--spec-model", spec_model])

    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    result = subprocess.run(
        cmd,
        cwd=REMOTE_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    return Path(output_json).read_text(encoding="utf-8")


@app.local_entrypoint()
def main(
    modes: str = "baseline,embedding_norm,spec_prefill",
    model: str = "Qwen/Qwen3-1.7B",
    spec_model: str = "Qwen/Qwen3-0.6B",
    config: str = "configs/config_embedding_norm_p3.yaml",
    spec_config: str = "configs/config_p3.yaml",
    input_lens: str = "256,512,1024,2048,4096",
    batch_sizes: str = "1,2,4,8",
    output_len: int = 1,
    warmup_iters: int = 1,
    iters: int = 3,
    output_dir: str = "benchmark_results",
):
    from speculative_prefill.vllm_benchmarks.sweep_utils import (
        rows_from_json_payload,
        rows_to_csv,
        rows_to_jsonl,
        write_svg_latency_charts,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / f"qwen3_latency_sweep_{timestamp}"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode in [item.strip() for item in modes.split(",") if item.strip()]:
        mode_config = spec_config if mode == "spec_prefill" else config
        payload_text = run_mode_sweep.remote(
            mode=mode,
            model=model,
            spec_model=spec_model,
            config=mode_config,
            input_lens=input_lens,
            batch_sizes=batch_sizes,
            output_len=output_len,
            warmup_iters=warmup_iters,
            iters=iters,
        )
        payload_path = raw_dir / f"{mode}.json"
        payload_path.write_text(payload_text, encoding="utf-8")
        rows.extend(rows_from_json_payload(json.loads(payload_text)))

    rows_to_csv(rows, run_dir / "latency_sweep.csv")
    rows_to_jsonl(rows, run_dir / "latency_sweep.jsonl")
    chart_paths = write_svg_latency_charts(rows, run_dir / "charts")

    print(f"Wrote {run_dir}")
    print(f"CSV: {run_dir / 'latency_sweep.csv'}")
    for path in chart_paths:
        print(f"Chart: {path}")
