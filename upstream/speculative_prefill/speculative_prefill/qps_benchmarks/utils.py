import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


CATEGORY_DATASETS = {
    "single-doc-qa": ["qasper", "narrativeqa", "multifieldqa_zh", "multifieldqa_en"],
    "multi-doc-qa": ["dureader", "2wikimqa", "musique", "hotpotqa"],
    "summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "few-shot-learning": ["triviaqa", "lsht", "trec", "samsum"],
}

PAPER_QPS_GRIDS = {
    "few-shot-learning": [
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.4,
        1.6,
        1.8,
        2.0,
        2.2,
        2.4,
        2.6,
        2.8,
        3.0,
        3.2,
    ],
    "multi-doc-qa": [0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0],
    "summarization": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
}

PAPER_TIMEOUTS_S = {
    "few-shot-learning": 25.0,
    "multi-doc-qa": 15.0,
    "summarization": 45.0,
}

DEFAULT_NUM_SAMPLES = 32
DEFAULT_WARMUP_NUM_SAMPLES = 4
DEFAULT_WARMUP_QPS = 0.2
DEFAULT_WARMUP_MAX_TOKENS = 8
DEFAULT_API_KEY = "local_server"


@dataclass(frozen=True)
class ParsedQPSOutput:
    avg_latency_s: float | None
    timed_out: bool


@dataclass(frozen=True)
class QPSResult:
    mode: str
    category: str
    qps: float
    status: str
    avg_latency_s: float | None
    timed_out: bool
    num_requests: int
    num_success: int
    num_timeout: int
    timeout_s: float
    num_samples_per_dataset: int
    model: str
    spec_model: str
    config: str

    def to_dict(self) -> dict:
        return asdict(self)


def parse_qps_client_output(text: str) -> ParsedQPSOutput:
    if "Found timeout in queries" in text:
        return ParsedQPSOutput(avg_latency_s=None, timed_out=True)
    match = re.search(r"Average latency:\s*([0-9]+(?:\.[0-9]+)?)s", text)
    if not match:
        raise ValueError("Could not find QPS client latency or timeout marker")
    return ParsedQPSOutput(avg_latency_s=float(match.group(1)), timed_out=False)


def parse_float_csv(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one float")
    return parsed


def build_mode_env(mode: str, spec_model: str, config: str) -> dict[str, str]:
    if mode not in {"baseline", "embedding_norm", "middle_layer_norm", "spec_prefill"}:
        raise ValueError(
            "mode must be baseline, embedding_norm, middle_layer_norm, or spec_prefill"
        )
    env = {"VLLM_USE_V1": "0"}
    if mode == "embedding_norm":
        env["ENABLE_EMBEDDING_NORM_PREFILL"] = "1"
        env["SPEC_CONFIG_PATH"] = config
    elif mode == "middle_layer_norm":
        env["ENABLE_MIDDLE_LAYER_NORM_PREFILL"] = "1"
        env["SPEC_CONFIG_PATH"] = config
    elif mode == "spec_prefill":
        env["ENABLE_SP"] = spec_model
        env["SPEC_CONFIG_PATH"] = config
    return env


def build_server_command(
    *,
    model: str,
    port: int,
    max_model_len: int,
    max_num_seqs: int,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    api_key: str = DEFAULT_API_KEY,
    dtype: str = "auto",
    enforce_eager: bool = True,
) -> list[str]:
    cmd = [
        "python",
        "-m",
        "speculative_prefill.scripts",
        "serve",
        model,
        "--tokenizer",
        model,
        "--dtype",
        dtype,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--no-enable-chunked-prefill",
        "--disable-log-requests",
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-num-seqs",
        str(max_num_seqs),
        "--api-key",
        api_key,
        "--port",
        str(port),
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    return cmd


def build_client_command(
    *,
    model: str,
    category: str,
    qps: float,
    timeout_s: float,
    num_samples: int,
    output_json: str,
    host_name: str = "127.0.0.1",
    port: int = 8888,
    api_key: str = DEFAULT_API_KEY,
    max_tokens: int | None = None,
    max_tolerance: int | None = None,
    seed: int = 227,
) -> list[str]:
    cmd = [
        "python",
        "eval/qps_client.py",
        "--model",
        model,
        "--qps",
        str(qps),
        "--category",
        category,
        "--timeout",
        str(timeout_s),
        "--num-samples",
        str(num_samples),
        "--host-name",
        host_name,
        "--port",
        str(port),
        "--api-key",
        api_key,
        "--seed",
        str(seed),
        "--output-json",
        output_json,
    ]
    if max_tokens is not None:
        cmd.extend(["--max-tokens", str(max_tokens)])
    if max_tolerance is not None:
        cmd.extend(["--max-tolerance", str(max_tolerance)])
    return cmd


def _fieldnames(rows: list[QPSResult]) -> list[str]:
    if not rows:
        return [field for field in QPSResult.__dataclass_fields__]
    return list(rows[0].to_dict().keys())


def rows_to_csv(rows: list[QPSResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def rows_to_jsonl(rows: list[QPSResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
