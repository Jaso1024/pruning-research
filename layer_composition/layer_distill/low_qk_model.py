from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .experiment import GpuStatsSampler, JsonlStepLogger, TokenBatcher, _extract_layers, _metrics, _torch_dtype
from .low_qk_attention import LowQKAttention, init_low_qk_from_gpt_neox_attention
from .muon import Muon, split_muon_params


@dataclass(frozen=True)
class EndToEndLowQKConfig:
    output_dir: Path
    model_name: str = "EleutherAI/pythia-70m"
    learning_rates: tuple[float, ...] = (3e-2,)
    learning_rate: float | None = None
    steps: int = 100
    batch_size: int = 512
    seq_len: int = 256
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
    save_adapters: bool = True
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
        if self.data not in {"wikitext", "synthetic"}:
            raise ValueError("data must be wikitext or synthetic")
        if self.qk_dim <= 0:
            raise ValueError("qk_dim must be positive")
        if self.student_heads is not None and self.student_heads <= 0:
            raise ValueError("student_heads must be positive")


@dataclass(frozen=True)
class LowQKPerplexityEvalConfig:
    output_dir: Path
    adapter_root: Path | None = None
    adapter_paths: tuple[Path, ...] = ()
    model_name: str = "EleutherAI/pythia-70m"
    eval_steps: int = 64
    batch_size: int = 32
    seq_len: int = 512
    seed: int = 17
    data: str = "wikitext"
    data_split: str = "test"
    max_dataset_tokens: int = 2_000_000
    dtype: str = "bf16"
    qk_dim: int = 2
    student_heads: int | None = None
    ce_chunk_tokens: int = 32768
    include_teacher: bool = True
    log_gpu_stats: bool = True

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.adapter_root is not None:
            object.__setattr__(self, "adapter_root", Path(self.adapter_root))
        object.__setattr__(self, "adapter_paths", tuple(Path(path) for path in self.adapter_paths))
        if self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 1:
            raise ValueError("seq_len must be greater than 1")
        if self.data != "wikitext":
            raise ValueError("data must be wikitext")
        if self.data_split not in {"train", "validation", "test"}:
            raise ValueError("data_split must be train, validation, or test")
        if self.max_dataset_tokens <= self.seq_len:
            raise ValueError("max_dataset_tokens must be greater than seq_len")
        if self.qk_dim <= 0:
            raise ValueError("qk_dim must be positive")
        if self.student_heads is not None and self.student_heads <= 0:
            raise ValueError("student_heads must be positive")
        if self.ce_chunk_tokens <= 0:
            raise ValueError("ce_chunk_tokens must be positive")


@dataclass(frozen=True)
class LowQKLogitDistillConfig:
    output_dir: Path
    model_name: str = "EleutherAI/pythia-70m"
    learning_rates: tuple[float, ...] = (1e-3,)
    learning_rate: float | None = None
    steps: int = 100
    batch_size: int = 256
    seq_len: int = 256
    seed: int = 17
    data: str = "wikitext"
    max_dataset_tokens: int = 2_000_000
    dtype: str = "bf16"
    qk_dim: int = 2
    student_heads: int | None = None
    temperature: float = 1.0
    logit_chunk_tokens: int = 8192
    adamw_lr_scale: float = 0.1
    weight_decay: float = 0.0
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    save_adapters: bool = True
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
        if self.data not in {"wikitext", "synthetic"}:
            raise ValueError("data must be wikitext or synthetic")
        if self.qk_dim <= 0:
            raise ValueError("qk_dim must be positive")
        if self.student_heads is not None and self.student_heads <= 0:
            raise ValueError("student_heads must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.logit_chunk_tokens <= 0:
            raise ValueError("logit_chunk_tokens must be positive")


class LowQKGPTNeoXAttention(torch.nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        qk_dim: int = 2,
        value_dim: int | None = None,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.num_attention_heads = num_heads
        self.head_size = value_dim if value_dim is not None else hidden_size // num_heads
        self.layer_idx = layer_idx
        self.low_qk = LowQKAttention(hidden_size=hidden_size, num_heads=num_heads, qk_dim=qk_dim, value_dim=self.head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_past: Any | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if layer_past is not None:
            raise NotImplementedError("low-QK replacement does not implement KV cache")
        return self.low_qk(hidden_states), None


def replace_gpt_neox_attentions_with_low_qk(
    model: torch.nn.Module,
    *,
    qk_dim: int = 2,
    student_heads: int | None = None,
) -> list[LowQKGPTNeoXAttention]:
    for param in model.parameters():
        param.requires_grad_(False)
    layers = _extract_layers(model)
    hidden_size = int(getattr(model.config, "hidden_size"))
    adapters: list[LowQKGPTNeoXAttention] = []
    for layer_idx, layer in enumerate(layers):
        teacher_attention = layer.attention
        teacher_heads = int(getattr(teacher_attention, "num_attention_heads", getattr(model.config, "num_attention_heads")))
        teacher_head_size = int(getattr(teacher_attention, "head_size", hidden_size // teacher_heads))
        heads = student_heads or teacher_heads
        adapter = LowQKGPTNeoXAttention(
            hidden_size=hidden_size,
            num_heads=heads,
            qk_dim=qk_dim,
            value_dim=teacher_head_size,
            layer_idx=layer_idx,
        )
        init_low_qk_from_gpt_neox_attention(adapter.low_qk, teacher_attention)
        reference_param = next(teacher_attention.parameters(), None)
        if reference_param is not None:
            adapter.to(device=reference_param.device, dtype=reference_param.dtype)
        layer.attention = adapter
        for param in adapter.parameters():
            param.requires_grad_(True)
        adapters.append(adapter)
    return adapters


def build_end_to_end_low_qk_lr_configs(config: EndToEndLowQKConfig) -> list[EndToEndLowQKConfig]:
    return [
        dataclasses.replace(
            config,
            learning_rate=lr,
            learning_rates=(lr,),
            output_dir=config.output_dir / f"lr_{lr:g}",
        )
        for lr in config.learning_rates
    ]


def summarize_end_to_end_low_qk_runs(output_dir: Path) -> dict[str, Any]:
    runs = []
    for step_path in sorted(Path(output_dir).glob("lr_*/steps.jsonl")):
        records = [json.loads(line) for line in step_path.read_text().splitlines() if line.strip()]
        if not records:
            continue
        final = records[-1]
        best = min(records, key=lambda row: math.inf if row.get("final_rel_mse") is None else row["final_rel_mse"])
        runs.append(
            {
                "run_dir": str(step_path.parent),
                "lr": final.get("lr"),
                "final_step": final.get("step"),
                "final_loss": final.get("final_loss"),
                "final_rel_mse": final.get("final_rel_mse"),
                "final_cosine": final.get("final_cosine"),
                "best_step": best.get("step"),
                "best_rel_mse": best.get("final_rel_mse"),
                "best_cosine": best.get("final_cosine"),
                "final_tokens_per_sec": final.get("tokens_per_sec"),
            }
        )
    best_run = min(runs, key=lambda run: math.inf if run["best_rel_mse"] is None else run["best_rel_mse"]) if runs else None
    summary = {"best": best_run, "runs": runs}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_end_to_end_low_qk_sweep(config: EndToEndLowQKConfig) -> dict[str, Any]:
    for lr_config in build_end_to_end_low_qk_lr_configs(config):
        run_one_end_to_end_low_qk_lr(lr_config)
    return summarize_end_to_end_low_qk_runs(config.output_dir)


def build_logit_low_qk_lr_configs(config: LowQKLogitDistillConfig) -> list[LowQKLogitDistillConfig]:
    return [
        dataclasses.replace(
            config,
            learning_rate=lr,
            learning_rates=(lr,),
            output_dir=config.output_dir / f"lr_{lr:g}",
        )
        for lr in config.learning_rates
    ]


def summarize_logit_low_qk_runs(output_dir: Path) -> dict[str, Any]:
    runs = []
    for step_path in sorted(Path(output_dir).glob("lr_*/steps.jsonl")):
        records = [json.loads(line) for line in step_path.read_text().splitlines() if line.strip()]
        if not records:
            continue
        final = records[-1]
        best = min(records, key=lambda row: math.inf if row.get("student_ppl") is None else row["student_ppl"])
        runs.append(
            {
                "run_dir": str(step_path.parent),
                "lr": final.get("lr"),
                "final_step": final.get("step"),
                "final_kl_loss": final.get("kl_loss"),
                "final_student_nll": final.get("student_nll"),
                "final_student_ppl": final.get("student_ppl"),
                "final_teacher_nll": final.get("teacher_nll"),
                "final_teacher_ppl": final.get("teacher_ppl"),
                "best_step": best.get("step"),
                "best_student_ppl": best.get("student_ppl"),
                "best_kl_loss": best.get("kl_loss"),
                "final_tokens_per_sec": final.get("tokens_per_sec"),
            }
        )
    best_run = min(runs, key=lambda run: math.inf if run["best_student_ppl"] is None else run["best_student_ppl"]) if runs else None
    summary = {"best": best_run, "runs": runs}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_logit_low_qk_sweep(config: LowQKLogitDistillConfig) -> dict[str, Any]:
    for lr_config in build_logit_low_qk_lr_configs(config):
        run_one_logit_low_qk_lr(lr_config)
    return summarize_logit_low_qk_runs(config.output_dir)


def run_one_end_to_end_low_qk_lr(config: EndToEndLowQKConfig) -> dict[str, Any]:
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
    teacher = _load_auto_model(config.model_name, model_dtype).to(device)
    student = _load_auto_model(config.model_name, model_dtype).to(device)
    teacher.eval()
    student.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    adapters = replace_gpt_neox_attentions_with_low_qk(student, qk_dim=config.qk_dim, student_heads=config.student_heads)
    trainable = torch.nn.ModuleList(adapters)

    muon_params, adamw_params = split_muon_params(trainable)
    optimizers: list[torch.optim.Optimizer] = []
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
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.monotonic()
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                        target = teacher(input_ids=input_ids, use_cache=False).last_hidden_state.detach()
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                    prediction = student(input_ids=input_ids, use_cache=False).last_hidden_state
                    loss, rel_mse, cosine = _metrics(prediction, target)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=1e9)
                for optimizer in optimizers:
                    optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.monotonic()
                elapsed = max(end - start, 1e-12)
                record = {
                    "step": step,
                    "lr": config.learning_rate,
                    "final_loss": float(loss.detach().cpu()),
                    "final_rel_mse": rel_mse,
                    "final_cosine": cosine,
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "elapsed_sec": elapsed,
                    "tokens": int(input_ids.numel()),
                    "tokens_per_sec": float(input_ids.numel() / elapsed),
                    "model_name": config.model_name,
                    "qk_dim": config.qk_dim,
                    "student_heads": config.student_heads,
                    "replaced_layers": len(adapters),
                    "trainable_params": sum(param.numel() for param in trainable.parameters()),
                }
                if sampler is not None:
                    record.update(sampler.stats_between(start, end))
                logger.log(record)
                print(json.dumps(record, sort_keys=True), flush=True)
                last_record = record
    finally:
        if sampler is not None:
            sampler.stop()
    if config.save_adapters:
        torch.save({str(idx): adapter.state_dict() for idx, adapter in enumerate(adapters)}, config.output_dir / "low_qk_adapters.pt")
    return last_record


def run_one_logit_low_qk_lr(config: LowQKLogitDistillConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

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
    teacher = _load_auto_causal_lm(config.model_name, model_dtype).to(device)
    student = _load_auto_causal_lm(config.model_name, model_dtype).to(device)
    teacher.eval()
    student.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    adapters = replace_gpt_neox_attentions_with_low_qk(student, qk_dim=config.qk_dim, student_heads=config.student_heads)
    trainable = torch.nn.ModuleList(adapters)
    teacher_backbone = _causal_lm_backbone(teacher)
    student_backbone = _causal_lm_backbone(student)
    teacher_head = _causal_lm_head(teacher)
    student_head = _causal_lm_head(student)

    muon_params, adamw_params = split_muon_params(trainable)
    optimizers: list[torch.optim.Optimizer] = []
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
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.monotonic()
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                        teacher_hidden = teacher_backbone(input_ids=input_ids, use_cache=False).last_hidden_state[:, :-1, :].detach()
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                    student_hidden = student_backbone(input_ids=input_ids, use_cache=False).last_hidden_state[:, :-1, :]

                flat_teacher = teacher_hidden.reshape(-1, teacher_hidden.size(-1))
                flat_student = student_hidden.reshape(-1, student_hidden.size(-1))
                flat_labels = input_ids[:, 1:].reshape(-1)
                total_tokens = int(flat_labels.numel())
                total_kl = 0.0
                total_student_nll = 0.0
                total_teacher_nll = 0.0
                num_chunks = math.ceil(total_tokens / config.logit_chunk_tokens)
                for chunk_idx, chunk_start in enumerate(range(0, total_tokens, config.logit_chunk_tokens)):
                    chunk_end = min(chunk_start + config.logit_chunk_tokens, total_tokens)
                    with torch.no_grad():
                        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                            teacher_logits = teacher_head(flat_teacher[chunk_start:chunk_end])
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                        student_logits = student_head(flat_student[chunk_start:chunk_end])
                        chunk_kl = kl_divergence_sum_from_logits(
                            student_logits,
                            teacher_logits,
                            temperature=config.temperature,
                        )
                    labels = flat_labels[chunk_start:chunk_end]
                    with torch.no_grad():
                        student_nll = torch.nn.functional.cross_entropy(student_logits.detach().float(), labels, reduction="sum")
                        teacher_nll = torch.nn.functional.cross_entropy(teacher_logits.float(), labels, reduction="sum")
                    (chunk_kl / total_tokens).backward(retain_graph=chunk_idx + 1 < num_chunks)
                    total_kl += float(chunk_kl.detach().float().cpu())
                    total_student_nll += float(student_nll.detach().float().cpu())
                    total_teacher_nll += float(teacher_nll.detach().float().cpu())

                grad_norm = torch.nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=1e9)
                for optimizer in optimizers:
                    optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.monotonic()
                elapsed = max(end_time - start_time, 1e-12)
                student_nll_mean = total_student_nll / total_tokens
                teacher_nll_mean = total_teacher_nll / total_tokens
                record = {
                    "step": step,
                    "lr": config.learning_rate,
                    "kl_loss": total_kl / total_tokens,
                    "student_nll": student_nll_mean,
                    "student_ppl": float(math.exp(student_nll_mean)),
                    "teacher_nll": teacher_nll_mean,
                    "teacher_ppl": float(math.exp(teacher_nll_mean)),
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "elapsed_sec": elapsed,
                    "tokens": total_tokens,
                    "tokens_per_sec": float(total_tokens / elapsed),
                    "model_name": config.model_name,
                    "qk_dim": config.qk_dim,
                    "student_heads": config.student_heads,
                    "temperature": config.temperature,
                    "logit_chunk_tokens": config.logit_chunk_tokens,
                    "logit_chunks": num_chunks,
                    "replaced_layers": len(adapters),
                    "trainable_params": sum(param.numel() for param in trainable.parameters()),
                }
                if sampler is not None:
                    record.update(sampler.stats_between(start_time, end_time))
                logger.log(record)
                print(json.dumps(record, sort_keys=True), flush=True)
                last_record = record
    finally:
        if sampler is not None:
            sampler.stop()
    if config.save_adapters:
        torch.save({str(idx): adapter.state_dict() for idx, adapter in enumerate(adapters)}, config.output_dir / "low_qk_adapters.pt")
    return last_record


def find_low_qk_adapter_checkpoints(adapter_root: Path) -> tuple[Path, ...]:
    root = Path(adapter_root)
    if root.is_file():
        return (root,)
    paths = tuple(root.glob("lr_*/low_qk_adapters.pt"))
    if not paths:
        paths = tuple(root.glob("**/low_qk_adapters.pt"))

    def sort_key(path: Path) -> tuple[float, str]:
        parent = path.parent.name
        if parent.startswith("lr_"):
            try:
                return (float(parent[3:]), str(path))
            except ValueError:
                pass
        return (math.inf, str(path))

    return tuple(sorted(paths, key=sort_key))


def load_low_qk_adapter_checkpoint(adapters: list[LowQKGPTNeoXAttention], path: Path) -> None:
    checkpoint = torch.load(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"adapter checkpoint {path} is not a dict")
    if len(checkpoint) != len(adapters):
        raise ValueError(f"adapter checkpoint has {len(checkpoint)} layers, expected {len(adapters)}")
    for idx, adapter in enumerate(adapters):
        state = checkpoint.get(str(idx))
        if not isinstance(state, dict):
            raise ValueError(f"adapter checkpoint is missing layer {idx}")
        adapter.load_state_dict(state, strict=True)


def perplexity_from_nll(*, total_nll: float, total_tokens: int) -> float:
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    return float(math.exp(total_nll / total_tokens))


def causal_lm_nll_from_logits(logits: torch.Tensor, labels: torch.Tensor, *, chunk_tokens: int = 32768) -> tuple[torch.Tensor, int]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1)
    total = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, shift_labels.numel(), chunk_tokens):
        end = min(start + chunk_tokens, shift_labels.numel())
        total = total + torch.nn.functional.cross_entropy(
            shift_logits[start:end].float(),
            shift_labels[start:end],
            reduction="sum",
        )
    return total, int(shift_labels.numel())


def kl_divergence_sum_from_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_log_probs = torch.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = torch.softmax(teacher_logits.float() / temperature, dim=-1)
    return torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="sum") * (temperature * temperature)


def run_low_qk_perplexity_eval(config: LowQKPerplexityEvalConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(config.dtype)
    model_dtype = dtype if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokens = _load_wikitext_eval_tokens(
        tokenizer=tokenizer,
        split=config.data_split,
        max_tokens=config.max_dataset_tokens,
    )
    batches = _make_eval_batches(tokens, batch_size=config.batch_size, seq_len=config.seq_len, max_steps=config.eval_steps)
    if not batches:
        raise ValueError("not enough tokens to build an evaluation batch")

    adapter_paths = config.adapter_paths
    if config.adapter_root is not None:
        adapter_paths = adapter_paths + find_low_qk_adapter_checkpoints(config.adapter_root)
    if not config.include_teacher and not adapter_paths:
        raise ValueError("nothing to evaluate")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True, default=str) + "\n")
    runs: list[dict[str, Any]] = []
    if config.include_teacher:
        model = _load_auto_causal_lm(config.model_name, model_dtype).to(device)
        model.eval()
        runs.append(
            _evaluate_causal_lm_perplexity(
                model=model,
                batches=batches,
                output_dir=config.output_dir / "teacher",
                run_name="teacher",
                device=device,
                dtype=dtype,
                config=config,
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seen: set[Path] = set()
    for adapter_path in adapter_paths:
        adapter_path = Path(adapter_path)
        if adapter_path in seen:
            continue
        seen.add(adapter_path)
        run_name = adapter_path.parent.name
        model = _load_auto_causal_lm(config.model_name, model_dtype).to(device)
        adapters = replace_gpt_neox_attentions_with_low_qk(model, qk_dim=config.qk_dim, student_heads=config.student_heads)
        load_low_qk_adapter_checkpoint(adapters, adapter_path)
        model.eval()
        record = _evaluate_causal_lm_perplexity(
            model=model,
            batches=batches,
            output_dir=config.output_dir / run_name,
            run_name=run_name,
            device=device,
            dtype=dtype,
            config=config,
            adapter_path=adapter_path,
        )
        record["lr"] = _lr_from_adapter_path(adapter_path)
        runs.append(record)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {"runs": runs, "best_adapter": _best_adapter_ppl(runs)}
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary


def _load_wikitext_eval_tokens(*, tokenizer, split: str, max_tokens: int) -> torch.Tensor:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for perplexity evaluation") from exc
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(item["text"] for item in dataset if item["text"].strip())
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return encoded[:max_tokens].contiguous()


def _make_eval_batches(tokens: torch.Tensor, *, batch_size: int, seq_len: int, max_steps: int) -> list[torch.Tensor]:
    usable = (tokens.numel() // seq_len) * seq_len
    if usable < seq_len:
        return []
    chunks = tokens[:usable].view(-1, seq_len)
    batches = []
    for start in range(0, chunks.size(0), batch_size):
        if len(batches) >= max_steps:
            break
        batches.append(chunks[start : start + batch_size].contiguous())
    return batches


def _evaluate_causal_lm_perplexity(
    *,
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    output_dir: Path,
    run_name: str,
    device: torch.device,
    dtype: torch.dtype,
    config: LowQKPerplexityEvalConfig,
    adapter_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_nll = 0.0
    total_tokens = 0
    start_time = time.monotonic()
    sampler = GpuStatsSampler() if config.log_gpu_stats and device.type == "cuda" else None
    if sampler is not None:
        sampler.start()
    try:
        with JsonlStepLogger(output_dir / "steps.jsonl") as logger:
            with torch.inference_mode():
                for step, batch in enumerate(batches, start=1):
                    input_ids = batch.to(device=device, non_blocking=True)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    step_start = time.monotonic()
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                        output = model(input_ids=input_ids, use_cache=False)
                    step_nll, label_tokens = causal_lm_nll_from_logits(
                        output.logits,
                        input_ids,
                        chunk_tokens=config.ce_chunk_tokens,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    step_end = time.monotonic()
                    loss = float((step_nll / label_tokens).detach().float().cpu())
                    total_nll += float(step_nll.detach().float().cpu())
                    total_tokens += label_tokens
                    record = {
                        "step": step,
                        "run_name": run_name,
                        "loss": loss,
                        "ppl_so_far": perplexity_from_nll(total_nll=total_nll, total_tokens=total_tokens),
                        "tokens": label_tokens,
                        "total_tokens": total_tokens,
                        "elapsed_sec": max(step_end - step_start, 1e-12),
                        "tokens_per_sec": float(label_tokens / max(step_end - step_start, 1e-12)),
                    }
                    if adapter_path is not None:
                        record["adapter_path"] = str(adapter_path)
                    if sampler is not None:
                        record.update(sampler.stats_between(step_start, step_end))
                    logger.log(record)
                    print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        if sampler is not None:
            sampler.stop()
    elapsed = max(time.monotonic() - start_time, 1e-12)
    result = {
        "run_name": run_name,
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "loss": total_nll / total_tokens,
        "ppl": perplexity_from_nll(total_nll=total_nll, total_tokens=total_tokens),
        "total_tokens": total_tokens,
        "steps": len(batches),
        "elapsed_sec": elapsed,
        "tokens_per_sec": float(total_tokens / elapsed),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _lr_from_adapter_path(path: Path) -> float | None:
    parent = Path(path).parent.name
    if not parent.startswith("lr_"):
        return None
    try:
        return float(parent[3:])
    except ValueError:
        return None


def _best_adapter_ppl(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    adapters = [run for run in runs if run.get("adapter_path")]
    return min(adapters, key=lambda run: math.inf if run.get("ppl") is None else run["ppl"]) if adapters else None


def _load_auto_model(model_name: str, dtype: torch.dtype) -> torch.nn.Module:
    from transformers import AutoModel

    try:
        return AutoModel.from_pretrained(model_name, dtype=dtype, attn_implementation="sdpa")
    except TypeError:
        return AutoModel.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="sdpa")


def _load_auto_causal_lm(model_name: str, dtype: torch.dtype) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation="sdpa")
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="sdpa")


def _causal_lm_backbone(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox
    if hasattr(model, "model"):
        return model.model
    base = getattr(model, "base_model", None)
    if base is not None:
        return base
    raise ValueError("could not find causal LM backbone")


def _causal_lm_head(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "embed_out"):
        return model.embed_out
    if hasattr(model, "lm_head"):
        return model.lm_head
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if head is not None:
        return head
    raise ValueError("could not find causal LM head")


def _parse_lrs(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/low_qk_model"))
    parser.add_argument("--model-name", default="EleutherAI/pythia-70m")
    parser.add_argument("--learning-rates", default="3e-2")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--qk-dim", type=int, default=2)
    parser.add_argument("--student-heads", type=int, default=None)
    parser.add_argument("--ce-chunk-tokens", type=int, default=32768)
    parser.add_argument("--logit-distill", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--logit-chunk-tokens", type=int, default=8192)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--data", choices=["wikitext", "synthetic"], default="wikitext")
    parser.add_argument("--data-split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--max-dataset-tokens", type=int, default=2_000_000)
    parser.add_argument("--eval-ppl", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=64)
    parser.add_argument("--adapter-root", type=Path, default=None)
    parser.add_argument("--adapter-path", type=Path, action="append", default=[])
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--no-save-adapters", action="store_true")
    parser.add_argument("--no-gpu-stats", action="store_true")
    args = parser.parse_args(argv)
    if args.eval_ppl:
        config = LowQKPerplexityEvalConfig(
            output_dir=args.output_dir,
            adapter_root=args.adapter_root,
            adapter_paths=tuple(args.adapter_path),
            model_name=args.model_name,
            eval_steps=args.eval_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            qk_dim=args.qk_dim,
            student_heads=args.student_heads,
            ce_chunk_tokens=args.ce_chunk_tokens,
            dtype=args.dtype,
            data=args.data,
            data_split=args.data_split,
            max_dataset_tokens=args.max_dataset_tokens,
            include_teacher=not args.skip_teacher,
            log_gpu_stats=not args.no_gpu_stats,
        )
        print(json.dumps(run_low_qk_perplexity_eval(config), indent=2, sort_keys=True))
        return
    if args.logit_distill:
        config = LowQKLogitDistillConfig(
            output_dir=args.output_dir,
            model_name=args.model_name,
            learning_rates=_parse_lrs(args.learning_rates),
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            qk_dim=args.qk_dim,
            student_heads=args.student_heads,
            temperature=args.temperature,
            logit_chunk_tokens=args.logit_chunk_tokens,
            dtype=args.dtype,
            data=args.data,
            max_dataset_tokens=args.max_dataset_tokens,
            save_adapters=not args.no_save_adapters,
            log_gpu_stats=not args.no_gpu_stats,
        )
        print(json.dumps(run_logit_low_qk_sweep(config), indent=2, sort_keys=True))
        return
    config = EndToEndLowQKConfig(
        output_dir=args.output_dir,
        model_name=args.model_name,
        learning_rates=_parse_lrs(args.learning_rates),
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        qk_dim=args.qk_dim,
        student_heads=args.student_heads,
        dtype=args.dtype,
        data=args.data,
        max_dataset_tokens=args.max_dataset_tokens,
        save_adapters=not args.no_save_adapters,
        log_gpu_stats=not args.no_gpu_stats,
    )
    print(json.dumps(run_end_to_end_low_qk_sweep(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
