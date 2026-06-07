import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import modal


REMOTE_ROOT = "/root/speculative_prefill"
SECRETS = [modal.Secret.from_local_environ(["HF_TOKEN"])] if "HF_TOKEN" in os.environ else []

app = modal.App("spec-prefill-qps-throughput")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("openai", "httpx", "huggingface_hub")
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


def _wait_for_server(port: int, proc: subprocess.Popen, log_path: Path, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited during startup:\n{log_path.read_text(encoding='utf-8')[-12000:]}")
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Server did not become healthy within {timeout_s}s")


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def _client_timeout_s(num_requests: int, qps: float, request_timeout_s: float) -> int:
    return int(max(600, (num_requests / qps) + request_timeout_s + 180))


def _max_tolerance(num_requests: int) -> int:
    return min(4, max(0, num_requests - 1))


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    secrets=SECRETS,
)
def run_mode_category(
    mode: str,
    category: str,
    model: str,
    spec_model: str,
    config: str,
    qps_values: str,
    num_samples: int,
    seed: int,
    timeout_s: float,
    port: int,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    stop_on_timeout: bool,
    server_startup_timeout_s: int,
) -> str:
    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)

    from speculative_prefill.qps_benchmarks.utils import (
        CATEGORY_DATASETS,
        DEFAULT_WARMUP_MAX_TOKENS,
        DEFAULT_WARMUP_NUM_SAMPLES,
        DEFAULT_WARMUP_QPS,
        QPSResult,
        build_client_command,
        build_mode_env,
        build_server_command,
        parse_float_csv,
    )

    qps_grid = parse_float_csv(qps_values)
    server_log_path = Path("/tmp/qps_server.log")
    server_cmd = build_server_command(
        model=model,
        port=port,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    env = os.environ.copy()
    env.update(build_mode_env(mode, spec_model=spec_model, config=config))

    with server_log_path.open("w", encoding="utf-8") as server_log:
        server_proc = subprocess.Popen(
            server_cmd,
            cwd=REMOTE_ROOT,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    client_outputs = []
    rows = []
    num_requests = len(CATEGORY_DATASETS[category]) * num_samples
    try:
        _wait_for_server(port, server_proc, server_log_path, server_startup_timeout_s)

        warmup_json = "/tmp/qps_warmup.json"
        warmup_cmd = build_client_command(
            model=model,
            category=category,
            qps=DEFAULT_WARMUP_QPS,
            timeout_s=timeout_s,
            num_samples=DEFAULT_WARMUP_NUM_SAMPLES,
            output_json=warmup_json,
            max_tokens=DEFAULT_WARMUP_MAX_TOKENS,
            max_tolerance=_max_tolerance(len(CATEGORY_DATASETS[category]) * DEFAULT_WARMUP_NUM_SAMPLES),
            seed=seed,
            port=port,
        )
        warmup = subprocess.run(
            warmup_cmd,
            cwd=REMOTE_ROOT,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=_client_timeout_s(len(CATEGORY_DATASETS[category]) * DEFAULT_WARMUP_NUM_SAMPLES, DEFAULT_WARMUP_QPS, timeout_s),
        )
        client_outputs.append({"kind": "warmup", "qps": DEFAULT_WARMUP_QPS, "returncode": warmup.returncode, "stdout": warmup.stdout})
        if warmup.returncode != 0:
            raise RuntimeError(warmup.stdout)

        for qps in qps_grid:
            time.sleep(10)
            result_json = f"/tmp/qps_{category}_{qps}.json"
            client_cmd = build_client_command(
                model=model,
                category=category,
                qps=qps,
                timeout_s=timeout_s,
                num_samples=num_samples,
                output_json=result_json,
                max_tolerance=_max_tolerance(num_requests),
                seed=seed,
                port=port,
            )
            result = subprocess.run(
                client_cmd,
                cwd=REMOTE_ROOT,
                env=os.environ.copy(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=_client_timeout_s(num_requests, qps, timeout_s),
            )
            client_outputs.append({"kind": "profile", "qps": qps, "returncode": result.returncode, "stdout": result.stdout})
            if result.returncode != 0:
                row = QPSResult(
                    mode=mode,
                    category=category,
                    qps=qps,
                    status="client_error",
                    avg_latency_s=None,
                    timed_out=True,
                    num_requests=num_requests,
                    num_success=0,
                    num_timeout=num_requests,
                    timeout_s=timeout_s,
                    num_samples_per_dataset=num_samples,
                    model=model,
                    spec_model=spec_model if mode == "spec_prefill" else "",
                    config=config if mode != "baseline" else "",
                )
            else:
                summary = json.loads(Path(result_json).read_text(encoding="utf-8"))
                row = QPSResult(
                    mode=mode,
                    category=category,
                    qps=qps,
                    status=summary["status"],
                    avg_latency_s=summary["avg_latency_s"],
                    timed_out=summary["status"] != "ok",
                    num_requests=summary["num_requests"],
                    num_success=summary["num_success"],
                    num_timeout=summary["num_timeout"],
                    timeout_s=summary["timeout_s"],
                    num_samples_per_dataset=summary["num_samples_per_dataset"],
                    model=model,
                    spec_model=spec_model if mode == "spec_prefill" else "",
                    config=config if mode != "baseline" else "",
                )
            rows.append(row.to_dict())
            if stop_on_timeout and row.timed_out:
                break
    finally:
        _stop_server(server_proc)

    payload = {
        "mode": mode,
        "category": category,
        "model": model,
        "spec_model": spec_model,
        "config": config,
        "qps_values": qps_grid,
        "num_samples": num_samples,
        "timeout_s": timeout_s,
        "rows": rows,
        "client_outputs": client_outputs,
        "server_cmd": server_cmd,
        "server_env": {key: env[key] for key in sorted(build_mode_env(mode, spec_model=spec_model, config=config))},
        "server_log_tail": server_log_path.read_text(encoding="utf-8")[-120000:],
    }
    return json.dumps(payload)


@app.local_entrypoint()
def main(
    modes: str = "baseline,embedding_norm,spec_prefill",
    categories: str = "paper",
    model: str = "Qwen/Qwen3-1.7B",
    spec_model: str = "Qwen/Qwen3-0.6B",
    embedding_config: str = "configs/config_embedding_norm_p3.yaml",
    middle_layer_config: str = "configs/config_middle_layer_norm_p3.yaml",
    spec_config: str = "configs/config_p3.yaml",
    qps_values: str = "paper",
    num_samples: int = 32,
    seed: int = 227,
    port: int = 8888,
    max_model_len: int = 40960,
    max_num_seqs: int = 256,
    gpu_memory_utilization: float = 0.85,
    stop_on_timeout: bool = True,
    server_startup_timeout_s: int = 1800,
    output_dir: str = "local/qps_benchmark",
):
    from speculative_prefill.qps_benchmarks.utils import (
        PAPER_QPS_GRIDS,
        PAPER_TIMEOUTS_S,
        QPSResult,
        rows_to_csv,
        rows_to_jsonl,
    )

    selected_modes = [item.strip() for item in modes.split(",") if item.strip()]
    if categories == "paper":
        selected_categories = ["few-shot-learning", "multi-doc-qa", "summarization"]
    else:
        selected_categories = [item.strip() for item in categories.split(",") if item.strip()]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode_slug = "_".join(selected_modes)
    run_dir = Path(output_dir) / f"qps_throughput_{timestamp}_{mode_slug}"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "modes": selected_modes,
        "categories": selected_categories,
        "model": model,
        "spec_model": spec_model,
        "embedding_config": embedding_config,
        "middle_layer_config": middle_layer_config,
        "spec_config": spec_config,
        "qps_values": qps_values,
        "num_samples": num_samples,
        "seed": seed,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization,
        "stop_on_timeout": stop_on_timeout,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    rows = []
    for mode in selected_modes:
        if mode not in {"baseline", "embedding_norm", "middle_layer_norm", "spec_prefill"}:
            raise ValueError(f"Unknown mode: {mode}")
        if mode == "spec_prefill":
            mode_config = spec_config
        elif mode == "middle_layer_norm":
            mode_config = middle_layer_config
        else:
            mode_config = embedding_config
        for category in selected_categories:
            category_qps_values = ",".join(str(value) for value in PAPER_QPS_GRIDS[category]) if qps_values == "paper" else qps_values
            payload_text = run_mode_category.remote(
                mode=mode,
                category=category,
                model=model,
                spec_model=spec_model,
                config=mode_config,
                qps_values=category_qps_values,
                num_samples=num_samples,
                seed=seed,
                timeout_s=PAPER_TIMEOUTS_S[category],
                port=port,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                gpu_memory_utilization=gpu_memory_utilization,
                stop_on_timeout=stop_on_timeout,
                server_startup_timeout_s=server_startup_timeout_s,
            )
            payload = json.loads(payload_text)
            stem = f"{mode}_{category}"
            (raw_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            (raw_dir / f"{stem}_server_tail.log").write_text(payload["server_log_tail"], encoding="utf-8")
            for item in payload["client_outputs"]:
                safe_qps = str(item["qps"]).replace(".", "p")
                (raw_dir / f"{stem}_{item['kind']}_{safe_qps}.log").write_text(item["stdout"], encoding="utf-8")
            rows.extend(QPSResult(**row) for row in payload["rows"])

    rows_to_csv(rows, run_dir / "qps_results.csv")
    rows_to_jsonl(rows, run_dir / "qps_results.jsonl")

    print(f"Wrote {run_dir}")
    print(f"CSV: {run_dir / 'qps_results.csv'}")
