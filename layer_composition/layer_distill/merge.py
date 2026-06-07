from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .experiment import (
    GpuStatsSampler,
    JsonlStepLogger,
    TokenBatcher,
    _extract_layers,
    _gpu_stats,
    _make_position_ids,
    _metrics,
    _student_forward,
    _torch_dtype,
    middle_layer_pair,
)


MERGE_METHODS = {"linear", "slerp", "geom_slerp"}


@dataclass(frozen=True)
class MergeEvalConfig:
    output_dir: Path
    model_name: str = "EleutherAI/pythia-70m"
    methods: tuple[str, ...] = ("slerp", "geom_slerp")
    t_values: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    method: str | None = None
    t: float | None = None
    steps: int = 100
    batch_size: int = 1024
    seq_len: int = 1024
    layer_index: int | None = None
    seed: int = 17
    data: str = "wikitext"
    max_dataset_tokens: int = 2_000_000
    dtype: str = "bf16"
    save_student: bool = False
    log_gpu_stats: bool = True

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "methods", tuple(str(method) for method in self.methods))
        object.__setattr__(self, "t_values", tuple(float(t) for t in self.t_values))
        if not self.methods:
            raise ValueError("methods must be non-empty")
        if any(method not in MERGE_METHODS for method in self.methods):
            raise ValueError(f"methods must be in {sorted(MERGE_METHODS)}")
        if self.method is not None and self.method not in MERGE_METHODS:
            raise ValueError(f"method must be in {sorted(MERGE_METHODS)}")
        if not self.t_values:
            raise ValueError("t_values must be non-empty")
        if any(t < 0.0 or t > 1.0 for t in self.t_values):
            raise ValueError("t_values must be in [0, 1]")
        if self.t is not None and (self.t < 0.0 or self.t > 1.0):
            raise ValueError("t must be in [0, 1]")
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


def slerp_tensor(a: torch.Tensor, b: torch.Tensor, t: float, *, geometric_norm: bool = False, eps: float = 1e-8) -> torch.Tensor:
    if t == 0.0:
        return a.clone()
    if t == 1.0:
        return b.clone()
    if not torch.is_floating_point(a):
        return a.clone() if t < 0.5 else b.clone()

    dtype = a.dtype
    a_flat = a.float().reshape(-1)
    b_flat = b.float().reshape(-1)
    a_norm = a_flat.norm()
    b_norm = b_flat.norm()
    if a_norm <= eps or b_norm <= eps:
        return torch.lerp(a, b, t)

    a_unit = a_flat / a_norm
    b_unit = b_flat / b_norm
    dot = torch.dot(a_unit, b_unit).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    if sin_omega.abs() <= eps:
        return torch.lerp(a, b, t)

    direction = (
        torch.sin((1.0 - t) * omega) / sin_omega * a_unit
        + torch.sin(t * omega) / sin_omega * b_unit
    )
    if geometric_norm:
        norm = torch.exp((1.0 - t) * torch.log(a_norm.clamp_min(eps)) + t * torch.log(b_norm.clamp_min(eps)))
        merged = direction * norm
    else:
        merged = (
            torch.sin((1.0 - t) * omega) / sin_omega * a_flat
            + torch.sin(t * omega) / sin_omega * b_flat
        )
    return merged.reshape_as(a).to(dtype=dtype)


def merge_module_pair(first: torch.nn.Module, second: torch.nn.Module, *, method: str, t: float) -> torch.nn.Module:
    if method not in MERGE_METHODS:
        raise ValueError(f"method must be in {sorted(MERGE_METHODS)}")
    merged = copy.deepcopy(first)
    first_state = first.state_dict()
    second_state = second.state_dict()
    merged_state = {}
    for key, first_value in first_state.items():
        if key not in second_state:
            merged_state[key] = first_value.detach().clone()
            continue
        second_value = second_state[key]
        if first_value.shape != second_value.shape:
            raise ValueError(f"state tensor shape mismatch for {key}: {tuple(first_value.shape)} != {tuple(second_value.shape)}")
        if method == "linear":
            value = torch.lerp(first_value, second_value, t) if torch.is_floating_point(first_value) else first_value.clone()
        elif method == "slerp":
            value = slerp_tensor(first_value, second_value, t)
        else:
            value = slerp_tensor(first_value, second_value, t, geometric_norm=True)
        merged_state[key] = value
    merged.load_state_dict(merged_state)
    merged.eval()
    for param in merged.parameters():
        param.requires_grad_(False)
    return merged


def build_merge_configs(config: MergeEvalConfig) -> list[MergeEvalConfig]:
    configs: list[MergeEvalConfig] = []
    for method in config.methods:
        for t in config.t_values:
            configs.append(
                dataclasses.replace(
                    config,
                    method=method,
                    methods=(method,),
                    t=t,
                    t_values=(t,),
                    output_dir=config.output_dir / f"{method}_t_{t:g}",
                )
            )
    return configs


def summarize_merge_runs(output_dir: Path) -> dict[str, Any]:
    runs = []
    for step_path in sorted(Path(output_dir).glob("*_t_*/steps.jsonl")):
        records = [json.loads(line) for line in step_path.read_text().splitlines() if line.strip()]
        if not records:
            continue
        final = records[-1]
        runs.append(
            {
                "run_dir": str(step_path.parent),
                "method": final.get("method"),
                "t": final.get("t"),
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


def _load_base_model(config: MergeEvalConfig):
    from transformers import AutoModel, AutoTokenizer

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
    return tokenizer, model, device, dtype


def _evaluate_merged_layer(
    *,
    config: MergeEvalConfig,
    model: torch.nn.Module,
    tokenizer,
    merged_layer: torch.nn.Module,
    layer_idx: int,
    next_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
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
                        prediction = _student_forward(model, merged_layer, hidden, position_ids)
                        loss, rel_mse, cosine = _metrics(prediction, target)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.monotonic()
                elapsed = max(end - start, 1e-12)
                record = {
                    "step": step,
                    "method": config.method,
                    "t": config.t,
                    "layer_index": layer_idx,
                    "next_layer_index": next_idx,
                    "loss": float(loss.detach().cpu()),
                    "rel_mse": rel_mse,
                    "cosine": cosine,
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
        torch.save(merged_layer.state_dict(), config.output_dir / "merged_layer.pt")
    return last_record


def run_merge_sweep(config: MergeEvalConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    tokenizer, model, device, dtype = _load_base_model(config)
    layers = _extract_layers(model)
    layer_idx, next_idx = (config.layer_index, config.layer_index + 1) if config.layer_index is not None else middle_layer_pair(len(layers))
    if next_idx >= len(layers):
        raise ValueError(f"layer_index {layer_idx} has no following layer in {len(layers)} layers")

    for run_config in build_merge_configs(config):
        assert run_config.method is not None
        assert run_config.t is not None
        merged_layer = merge_module_pair(layers[layer_idx], layers[next_idx], method=run_config.method, t=run_config.t).to(device=device)
        _evaluate_merged_layer(
            config=run_config,
            model=model,
            tokenizer=tokenizer,
            merged_layer=merged_layer,
            layer_idx=layer_idx,
            next_idx=next_idx,
            device=device,
            dtype=dtype,
        )
        del merged_layer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return summarize_merge_runs(config.output_dir)


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/layer_merge"))
    parser.add_argument("--model-name", default="EleutherAI/pythia-70m")
    parser.add_argument("--methods", default="slerp,geom_slerp")
    parser.add_argument("--t-values", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--layer-index", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data", choices=["wikitext", "synthetic"], default="wikitext")
    parser.add_argument("--max-dataset-tokens", type=int, default=2_000_000)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--save-student", action="store_true")
    args = parser.parse_args(argv)
    config = MergeEvalConfig(
        output_dir=args.output_dir,
        model_name=args.model_name,
        methods=_parse_csv_strings(args.methods),
        t_values=_parse_csv_floats(args.t_values),
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        layer_index=args.layer_index,
        seed=args.seed,
        data=args.data,
        max_dataset_tokens=args.max_dataset_tokens,
        dtype=args.dtype,
        save_student=args.save_student,
    )
    print(json.dumps(run_merge_sweep(config), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
