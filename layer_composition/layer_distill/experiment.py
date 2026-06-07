from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .muon import Muon, split_muon_params


@dataclass(frozen=True)
class DistillConfig:
    output_dir: Path
    model_name: str = "EleutherAI/pythia-70m"
    learning_rates: tuple[float, ...] = (3e-4, 1e-3, 3e-3)
    learning_rate: float | None = None
    steps: int = 100
    batch_size: int = 512
    seq_len: int = 256
    layer_index: int | None = None
    seed: int = 17
    data: str = "wikitext"
    max_dataset_tokens: int = 2_000_000
    dtype: str = "bf16"
    adamw_lr_scale: float = 0.1
    weight_decay: float = 0.0
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    compile_student: bool = False
    save_student: bool = True
    log_gpu_stats: bool = True

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "learning_rates", tuple(float(lr) for lr in self.learning_rates))
        if not self.learning_rates:
            raise ValueError("learning_rates must be non-empty")
        if any(lr <= 0 for lr in self.learning_rates):
            raise ValueError("learning_rates must be positive")
        if self.learning_rate is not None and self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 1:
            raise ValueError("seq_len must be greater than 1")
        if self.layer_index is not None and self.layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if self.data not in {"wikitext", "synthetic"}:
            raise ValueError("data must be wikitext or synthetic")


class JsonlStepLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is not None:
            self.handle.close()
        return False

    def log(self, record: dict[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("logger is not open")
        self.handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())


def middle_layer_pair(num_layers: int) -> tuple[int, int]:
    if num_layers < 2:
        raise ValueError("need at least two layers")
    first = (num_layers - 1) // 2
    if first + 1 >= num_layers:
        raise ValueError("middle layer has no following layer")
    return first, first + 1


def clone_student_layer(layer: torch.nn.Module) -> torch.nn.Module:
    student = copy.deepcopy(layer)
    student.train()
    for param in student.parameters():
        param.requires_grad_(True)
    return student


def build_lr_sweep_configs(config: DistillConfig) -> list[DistillConfig]:
    return [
        dataclasses.replace(
            config,
            learning_rate=lr,
            learning_rates=(lr,),
            output_dir=config.output_dir / f"lr_{lr:g}",
        )
        for lr in config.learning_rates
    ]


def summarize_runs(output_dir: Path) -> dict[str, Any]:
    runs = []
    for step_path in sorted(Path(output_dir).glob("lr_*/steps.jsonl")):
        records = [json.loads(line) for line in step_path.read_text().splitlines() if line.strip()]
        if not records:
            continue
        final = records[-1]
        runs.append(
            {
                "run_dir": str(step_path.parent),
                "final_step": final.get("step"),
                "final_loss": final.get("loss"),
                "final_rel_mse": final.get("rel_mse"),
                "final_cosine": final.get("cosine"),
                "final_tokens_per_sec": final.get("tokens_per_sec"),
            }
        )
    best = min(runs, key=lambda run: math.inf if run["final_rel_mse"] is None else run["final_rel_mse"]) if runs else None
    summary = {"best": best, "runs": runs}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError("dtype must be bf16, fp16, or fp32")


def _extract_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("could not find transformer layers on model")


class TokenBatcher:
    def __init__(
        self,
        *,
        tokenizer,
        config: DistillConfig,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.vocab_size = int(getattr(tokenizer, "vocab_size", 50_000))
        self.tokens = None
        if config.data == "wikitext":
            self.tokens = self._load_wikitext_tokens(tokenizer, config.max_dataset_tokens).to(device=device, non_blocking=True)

    def _load_wikitext_tokens(self, tokenizer, max_tokens: int) -> torch.Tensor:
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError("datasets is required for data='wikitext'; use data='synthetic' for smoke tests") from exc
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(item["text"] for item in dataset if item["text"].strip())
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if encoded.numel() < self.config.seq_len + 1:
            raise ValueError("tokenized dataset is shorter than seq_len")
        return encoded[:max_tokens].contiguous()

    def next(self) -> torch.Tensor:
        batch_size = self.config.batch_size
        seq_len = self.config.seq_len
        if self.tokens is None:
            return torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device, dtype=torch.long)
        max_start = self.tokens.numel() - seq_len - 1
        starts = torch.randint(0, max_start, (batch_size, 1), device=self.device)
        offsets = torch.arange(seq_len, device=self.device).unsqueeze(0)
        return self.tokens[starts + offsets]


def _gpu_stats() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:
        return {}
    first = output.splitlines()[0].split(",")
    if len(first) < 5:
        return {}
    return {
        "gpu_util_pct": float(first[0].strip()),
        "gpu_mem_util_pct": float(first[1].strip()),
        "gpu_mem_used_mb": float(first[2].strip()),
        "gpu_mem_total_mb": float(first[3].strip()),
        "gpu_power_w": float(first[4].strip()),
    }


class GpuStatsSampler:
    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.samples: list[tuple[float, dict[str, float]]] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            self.process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                    f"--loop-ms={self.interval_ms}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception:
            self.process = None
            return
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            parsed = self._parse(line)
            if parsed:
                self.samples.append((time.monotonic(), parsed))

    def _parse(self, line: str) -> dict[str, float]:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            return {}
        try:
            return {
                "gpu_util_pct": float(parts[0]),
                "gpu_mem_util_pct": float(parts[1]),
                "gpu_mem_used_mb": float(parts[2]),
                "gpu_mem_total_mb": float(parts[3]),
                "gpu_power_w": float(parts[4]),
            }
        except ValueError:
            return {}

    def stats_between(self, start: float, end: float) -> dict[str, Any]:
        window = [sample for timestamp, sample in self.samples if start <= timestamp <= end]
        if not window:
            return _gpu_stats()
        return {
            "gpu_util_pct": max(sample["gpu_util_pct"] for sample in window),
            "gpu_util_pct_avg": sum(sample["gpu_util_pct"] for sample in window) / len(window),
            "gpu_mem_util_pct": max(sample["gpu_mem_util_pct"] for sample in window),
            "gpu_mem_used_mb": max(sample["gpu_mem_used_mb"] for sample in window),
            "gpu_mem_total_mb": window[-1]["gpu_mem_total_mb"],
            "gpu_power_w": max(sample["gpu_power_w"] for sample in window),
            "gpu_samples": len(window),
        }

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread is not None:
            self.thread.join(timeout=1)


def _make_position_ids(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)


def _student_forward(model: torch.nn.Module, student: torch.nn.Module, hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    kwargs = {"use_cache": False, "position_ids": position_ids}
    if hasattr(model, "rotary_emb"):
        kwargs["position_embeddings"] = model.rotary_emb(hidden, position_ids)
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "rotary_emb"):
        kwargs["position_embeddings"] = model.gpt_neox.rotary_emb(hidden, position_ids)
    output = student(hidden, attention_mask=None, **kwargs)
    return output[0] if isinstance(output, tuple) else output


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
    rel_mse = loss / target.float().pow(2).mean().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        prediction.float().flatten(),
        target.float().flatten(),
        dim=0,
    )
    return loss, float(rel_mse.detach().cpu()), float(cosine.detach().cpu())


def run_one_lr(config: DistillConfig) -> dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(config.dtype)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_dtype = dtype if device.type == "cuda" else torch.float32
    try:
        model = AutoModel.from_pretrained(
            config.model_name,
            dtype=model_dtype,
            attn_implementation="sdpa",
        ).to(device)
    except TypeError:
        model = AutoModel.from_pretrained(
            config.model_name,
            torch_dtype=model_dtype,
            attn_implementation="sdpa",
        ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    layers = _extract_layers(model)
    layer_idx, next_idx = (config.layer_index, config.layer_index + 1) if config.layer_index is not None else middle_layer_pair(len(layers))
    if next_idx >= len(layers):
        raise ValueError(f"layer_index {layer_idx} has no following layer in {len(layers)} layers")
    student = clone_student_layer(layers[layer_idx]).to(device=device)
    if config.compile_student and hasattr(torch, "compile"):
        student = torch.compile(student)

    lr = config.learning_rate if config.learning_rate is not None else config.learning_rates[0]
    muon_params, adamw_params = split_muon_params(student)
    optimizers: list[torch.optim.Optimizer] = [
        Muon(
            muon_params,
            lr=lr,
            momentum=config.muon_momentum,
            ns_steps=config.muon_ns_steps,
            weight_decay=config.weight_decay,
        )
    ]
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=lr * config.adamw_lr_scale, weight_decay=config.weight_decay))

    batcher = TokenBatcher(tokenizer=tokenizer, config=config, device=device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True, default=str) + "\n"
    )

    last_record: dict[str, Any] = {}
    sampler = GpuStatsSampler() if config.log_gpu_stats and device.type == "cuda" else None
    if sampler is not None:
        sampler.start()
    try:
        with JsonlStepLogger(config.output_dir / "steps.jsonl") as logger:
            for step in range(1, config.steps + 1):
                input_ids = batcher.next()
                position_ids = _make_position_ids(input_ids)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.monotonic()

                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                        outputs = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
                        hidden = outputs.hidden_states[layer_idx].detach()
                        target = outputs.hidden_states[next_idx + 1].detach()

                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                    prediction = _student_forward(model, student, hidden, position_ids)
                    loss, rel_mse, cosine = _metrics(prediction, target)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1e9)
                for optimizer in optimizers:
                    optimizer.step()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.monotonic()
                elapsed = max(end - start, 1e-12)
                record = {
                    "step": step,
                    "lr": lr,
                    "layer_index": layer_idx,
                    "next_layer_index": next_idx,
                    "loss": float(loss.detach().cpu()),
                    "rel_mse": rel_mse,
                    "cosine": cosine,
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "batch_size": config.batch_size,
                    "seq_len": config.seq_len,
                    "tokens": config.batch_size * config.seq_len,
                    "elapsed_sec": elapsed,
                    "tokens_per_sec": (config.batch_size * config.seq_len) / elapsed,
                }
                if sampler is not None:
                    record.update(sampler.stats_between(start, end))
                elif config.log_gpu_stats:
                    record.update(_gpu_stats())
                logger.log(record)
                print(json.dumps(record, sort_keys=True), flush=True)
                last_record = record
    finally:
        if sampler is not None:
            sampler.stop()

    if config.save_student:
        torch.save(student.state_dict(), config.output_dir / "student_layer.pt")
    return last_record


def run_sweep(config: DistillConfig) -> dict[str, Any]:
    for run_config in build_lr_sweep_configs(config):
        run_one_lr(run_config)
    return summarize_runs(config.output_dir)


def _parse_lrs(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/layer_distill"))
    parser.add_argument("--model-name", default="EleutherAI/pythia-70m")
    parser.add_argument("--learning-rates", default="3e-4,1e-3,3e-3")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--layer-index", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data", choices=["wikitext", "synthetic"], default="wikitext")
    parser.add_argument("--max-dataset-tokens", type=int, default=2_000_000)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--compile-student", action="store_true")
    parser.add_argument("--no-save-student", action="store_true")
    args = parser.parse_args(argv)
    config = DistillConfig(
        output_dir=args.output_dir,
        model_name=args.model_name,
        learning_rates=_parse_lrs(args.learning_rates),
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        layer_index=args.layer_index,
        seed=args.seed,
        data=args.data,
        max_dataset_tokens=args.max_dataset_tokens,
        dtype=args.dtype,
        compile_student=args.compile_student,
        save_student=not args.no_save_student,
    )
    print(json.dumps(run_sweep(config), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
