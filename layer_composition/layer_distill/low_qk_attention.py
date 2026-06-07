from __future__ import annotations

import argparse
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
    _make_position_ids,
    _metrics,
    _torch_dtype,
    middle_layer_pair,
)
from .muon import Muon, split_muon_params


@dataclass(frozen=True)
class LowQKDistillConfig:
    output_dir: Path
    model_name: str = "EleutherAI/pythia-70m"
    learning_rates: tuple[float, ...] = (1e-3, 3e-3, 1e-2)
    learning_rate: float | None = None
    steps: int = 100
    batch_size: int = 512
    seq_len: int = 256
    layer_index: int | None = None
    seed: int = 17
    data: str = "wikitext"
    max_dataset_tokens: int = 2_000_000
    dtype: str = "bf16"
    qk_dim: int = 2
    student_heads: int | None = None
    adamw_lr_scale: float = 0.1
    weight_decay: float = 0.0
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
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
        if self.qk_dim <= 0:
            raise ValueError("qk_dim must be positive")
        if self.student_heads is not None and self.student_heads <= 0:
            raise ValueError("student_heads must be positive")


class LowQKAttention(torch.nn.Module):
    def __init__(self, *, hidden_size: int, num_heads: int, qk_dim: int = 2, value_dim: int | None = None):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if qk_dim <= 0:
            raise ValueError("qk_dim must be positive")
        value_dim = value_dim if value_dim is not None else hidden_size // num_heads
        if value_dim <= 0:
            raise ValueError("value_dim must be positive")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.value_dim = value_dim
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * qk_dim)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * qk_dim)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * value_dim)
        self.out_proj = torch.nn.Linear(num_heads * value_dim, hidden_size)
        self.scale = qk_dim**-0.5

    def forward(self, hidden_states: torch.Tensor, *, return_weights: bool = False):
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.qk_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_heads, self.qk_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_heads, self.value_dim).transpose(1, 2)
        if not return_weights:
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
            output = output.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.value_dim)
            return self.out_proj(output)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        causal = torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=v.dtype)
        output = torch.matmul(weights, v).transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.value_dim)
        output = self.out_proj(output)
        return (output, weights) if return_weights else output


def init_low_qk_from_gpt_neox_attention(student: LowQKAttention, teacher_attention: torch.nn.Module) -> None:
    with torch.no_grad():
        dense = getattr(teacher_attention, "dense", None)
        if dense is not None and student.out_proj.weight.shape == dense.weight.shape:
            student.out_proj.weight.copy_(dense.weight)
            if student.out_proj.bias is not None and dense.bias is not None:
                student.out_proj.bias.copy_(dense.bias)
        qkv = getattr(teacher_attention, "query_key_value", None)
        if qkv is None:
            return
        teacher_heads = int(getattr(teacher_attention, "num_attention_heads", student.num_heads))
        teacher_head_size = int(getattr(teacher_attention, "head_size", student.value_dim))
        if teacher_heads != student.num_heads or teacher_head_size != student.value_dim:
            return
        q_rows = []
        k_rows = []
        v_rows = []
        q_bias = []
        k_bias = []
        v_bias = []
        for head in range(student.num_heads):
            base = head * 3 * teacher_head_size
            q_rows.append(qkv.weight[base : base + student.qk_dim])
            k_rows.append(qkv.weight[base + teacher_head_size : base + teacher_head_size + student.qk_dim])
            v_rows.append(qkv.weight[base + 2 * teacher_head_size : base + 3 * teacher_head_size])
            if qkv.bias is not None:
                q_bias.append(qkv.bias[base : base + student.qk_dim])
                k_bias.append(qkv.bias[base + teacher_head_size : base + teacher_head_size + student.qk_dim])
                v_bias.append(qkv.bias[base + 2 * teacher_head_size : base + 3 * teacher_head_size])
        q_weight = torch.cat(q_rows, dim=0)
        k_weight = torch.cat(k_rows, dim=0)
        v_weight = torch.cat(v_rows, dim=0)
        if student.q_proj.weight.shape == q_weight.shape:
            student.q_proj.weight.copy_(q_weight)
            student.k_proj.weight.copy_(k_weight)
            student.v_proj.weight.copy_(v_weight)
        if qkv.bias is not None and student.q_proj.bias is not None:
            student.q_proj.bias.copy_(torch.cat(q_bias, dim=0))
            student.k_proj.bias.copy_(torch.cat(k_bias, dim=0))
            student.v_proj.bias.copy_(torch.cat(v_bias, dim=0))


def build_low_qk_lr_configs(config: LowQKDistillConfig) -> list[LowQKDistillConfig]:
    return [
        dataclasses.replace(
            config,
            learning_rate=lr,
            learning_rates=(lr,),
            output_dir=config.output_dir / f"lr_{lr:g}",
        )
        for lr in config.learning_rates
    ]


def summarize_low_qk_runs(output_dir: Path) -> dict[str, Any]:
    runs = []
    for step_path in sorted(Path(output_dir).glob("lr_*/steps.jsonl")):
        records = [json.loads(line) for line in step_path.read_text().splitlines() if line.strip()]
        if not records:
            continue
        final = records[-1]
        runs.append(
            {
                "run_dir": str(step_path.parent),
                "lr": final.get("lr"),
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


def run_low_qk_sweep(config: LowQKDistillConfig) -> dict[str, Any]:
    for lr_config in build_low_qk_lr_configs(config):
        run_one_low_qk_lr(lr_config)
    return summarize_low_qk_runs(config.output_dir)


def run_one_low_qk_lr(config: LowQKDistillConfig) -> dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    if config.learning_rate is None:
        raise ValueError("learning_rate must be set for a single run")
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
        model = AutoModel.from_pretrained(config.model_name, dtype=model_dtype, attn_implementation="sdpa").to(device)
    except TypeError:
        model = AutoModel.from_pretrained(config.model_name, torch_dtype=model_dtype, attn_implementation="sdpa").to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    layers = _extract_layers(model)
    layer_idx, _ = (config.layer_index, config.layer_index + 1) if config.layer_index is not None else middle_layer_pair(len(layers))
    if layer_idx >= len(layers):
        raise ValueError(f"layer_index {layer_idx} is outside {len(layers)} layers")
    teacher_attention = layers[layer_idx].attention
    hidden_size = int(getattr(model.config, "hidden_size"))
    teacher_heads = int(getattr(teacher_attention, "num_attention_heads", getattr(model.config, "num_attention_heads")))
    teacher_head_size = int(getattr(teacher_attention, "head_size", hidden_size // teacher_heads))
    student_heads = config.student_heads or teacher_heads
    student = LowQKAttention(hidden_size=hidden_size, num_heads=student_heads, qk_dim=config.qk_dim, value_dim=teacher_head_size)
    init_low_qk_from_gpt_neox_attention(student, teacher_attention)
    student.to(device=device, dtype=model_dtype)
    student.train()

    muon_params, adamw_params = split_muon_params(student)
    optimizers = []
    if muon_params:
        optimizers.append(
            Muon(
                muon_params,
                lr=config.learning_rate,
                momentum=config.muon_momentum,
                ns_steps=config.muon_ns_steps,
                weight_decay=config.weight_decay,
            )
        )
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=config.learning_rate * config.adamw_lr_scale, weight_decay=config.weight_decay))

    batcher = TokenBatcher(tokenizer=tokenizer, config=config, device=device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True, default=str) + "\n")
    sampler = GpuStatsSampler() if config.log_gpu_stats and device.type == "cuda" else None
    if sampler is not None:
        sampler.start()
    last_record: dict[str, Any] = {}
    try:
        with JsonlStepLogger(config.output_dir / "steps.jsonl") as logger:
            for step in range(1, config.steps + 1):
                input_ids = batcher.next()
                position_ids = _make_position_ids(input_ids)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.monotonic()
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                        outputs = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
                        hidden = outputs.hidden_states[layer_idx].detach()
                        target = _teacher_attention_forward(model, teacher_attention, hidden, position_ids).detach()
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                    prediction = student(hidden)
                    loss, rel_mse, cosine = _metrics(prediction, target)
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.monotonic()
                elapsed = max(end - start, 1e-12)
                record = {
                    "step": step,
                    "lr": config.learning_rate,
                    "loss": float(loss.detach().cpu()),
                    "rel_mse": rel_mse,
                    "cosine": cosine,
                    "elapsed_sec": elapsed,
                    "tokens": int(input_ids.numel()),
                    "tokens_per_sec": float(input_ids.numel() / elapsed),
                    "model_name": config.model_name,
                    "layer_index": layer_idx,
                    "qk_dim": config.qk_dim,
                    "student_heads": student_heads,
                    "teacher_heads": teacher_heads,
                    "teacher_head_size": teacher_head_size,
                    "student_params": sum(param.numel() for param in student.parameters()),
                }
                if sampler is not None:
                    record.update(sampler.stats_between(start, end))
                logger.log(record)
                print(json.dumps(record, sort_keys=True), flush=True)
                last_record = record
    finally:
        if sampler is not None:
            sampler.stop()
    if config.save_student:
        torch.save(student.state_dict(), config.output_dir / "student_attention.pt")
    return last_record


def _teacher_attention_forward(
    model: torch.nn.Module,
    teacher_attention: torch.nn.Module,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    kwargs = {}
    if hasattr(model, "rotary_emb"):
        kwargs["position_embeddings"] = model.rotary_emb(hidden, position_ids)
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "rotary_emb"):
        kwargs["position_embeddings"] = model.gpt_neox.rotary_emb(hidden, position_ids)
    output = teacher_attention(hidden, attention_mask=None, **kwargs)
    return output[0] if isinstance(output, tuple) else output


def _parse_lrs(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/low_qk_attention"))
    parser.add_argument("--model-name", default="EleutherAI/pythia-70m")
    parser.add_argument("--learning-rates", default="1e-3,3e-3,1e-2")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--layer-index", type=int, default=None)
    parser.add_argument("--qk-dim", type=int, default=2)
    parser.add_argument("--student-heads", type=int, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--data", choices=["wikitext", "synthetic"], default="wikitext")
    parser.add_argument("--max-dataset-tokens", type=int, default=2_000_000)
    parser.add_argument("--no-save-student", action="store_true")
    parser.add_argument("--no-gpu-stats", action="store_true")
    args = parser.parse_args(argv)
    config = LowQKDistillConfig(
        output_dir=args.output_dir,
        model_name=args.model_name,
        learning_rates=_parse_lrs(args.learning_rates),
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        layer_index=args.layer_index,
        qk_dim=args.qk_dim,
        student_heads=args.student_heads,
        dtype=args.dtype,
        data=args.data,
        max_dataset_tokens=args.max_dataset_tokens,
        save_student=not args.no_save_student,
        log_gpu_stats=not args.no_gpu_stats,
    )
    print(json.dumps(run_low_qk_sweep(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
