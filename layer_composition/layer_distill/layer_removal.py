from __future__ import annotations

import argparse
import dataclasses
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .experiment import _extract_layers, _torch_dtype
from .hybrid_attention import _causal_lm_backbone, _evaluate_lm_loop, _load_causal_lm
from .low_qk_model import _load_wikitext_eval_tokens, _make_eval_batches


@dataclass(frozen=True)
class LayerRemovalEvalConfig:
    output_dir: Path
    model_name: str = "EleutherAI/pythia-1.4b"
    eval_steps: int = 8
    batch_size: int = 8
    seq_len: int = 256
    seed: int = 17
    data_split: str = "test"
    max_dataset_tokens: int = 2_000_000
    dtype: str = "bf16"
    ce_chunk_tokens: int = 32768
    include_baseline: bool = True
    greedy_layer_removal: bool = True
    greedy_max_layers: int | None = None
    remove_layers: tuple[int, ...] | None = None
    log_gpu_stats: bool = True

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.remove_layers is not None:
            layers = tuple(sorted({int(layer) for layer in self.remove_layers}))
            object.__setattr__(self, "remove_layers", layers)
            object.__setattr__(self, "greedy_layer_removal", False)
        if self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 1:
            raise ValueError("seq_len must be greater than 1")
        if self.data_split not in {"train", "validation", "test"}:
            raise ValueError("data_split must be train, validation, or test")
        if self.max_dataset_tokens <= self.seq_len:
            raise ValueError("max_dataset_tokens must be greater than seq_len")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be bf16, fp16, or fp32")
        if self.ce_chunk_tokens <= 0:
            raise ValueError("ce_chunk_tokens must be positive")
        if self.greedy_max_layers is not None and self.greedy_max_layers <= 0:
            raise ValueError("greedy_max_layers must be positive")
        if self.remove_layers is not None and any(layer < 0 for layer in self.remove_layers):
            raise ValueError("remove_layers must be non-negative")


class SkipTransformerLayer(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = False,
        layer_past: Any | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if use_cache or layer_past is not None:
            raise NotImplementedError("layer-removal eval does not implement KV cache")
        return hidden_states


def patch_layers_with_skip(model: torch.nn.Module, *, layer_indices: tuple[int, ...]) -> dict[int, torch.nn.Module]:
    layers = _extract_layers(_causal_lm_backbone(model))
    originals: dict[int, torch.nn.Module] = {}
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"layer index out of range: {layer_idx}")
        originals[layer_idx] = layers[layer_idx]
        layers[layer_idx] = SkipTransformerLayer()
    return originals


def restore_layers(model: torch.nn.Module, originals: dict[int, torch.nn.Module]) -> None:
    layers = _extract_layers(_causal_lm_backbone(model))
    for layer_idx, layer in originals.items():
        layers[layer_idx] = layer


def greedy_removal_sweep(
    *,
    layer_count: int,
    output_dir: Path,
    evaluate_removed_layers,
    max_removed_layers: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if layer_count <= 0:
        return [], []
    max_layers = layer_count if max_removed_layers is None else min(max_removed_layers, layer_count)
    removed: tuple[int, ...] = ()
    remaining = set(range(layer_count))
    runs: list[dict[str, Any]] = []
    path: list[dict[str, Any]] = []

    for round_idx in range(1, max_layers + 1):
        candidates: list[dict[str, Any]] = []
        for candidate_layer in sorted(remaining):
            layer_group = tuple(sorted((*removed, candidate_layer)))
            layer_slug = "_".join(f"{layer_idx:02d}" for layer_idx in layer_group)
            run_name = f"greedy_remove_r{round_idx:02d}_drop_{candidate_layer:02d}_layers_{layer_slug}"
            record = evaluate_removed_layers(layer_group, run_name, output_dir / run_name)
            record["greedy_round"] = round_idx
            record["candidate_layer"] = candidate_layer
            record["candidate_base_removed_layers"] = list(removed)
            record["removed_layers"] = len(layer_group)
            record["removed_layer_indices"] = list(layer_group)
            record["is_selected_candidate"] = False
            candidates.append(record)
            runs.append(record)

        best = min(candidates, key=lambda row: (float(row["ppl"]), float(row["loss"]), int(row["candidate_layer"])))
        best["is_selected_candidate"] = True
        removed = tuple(best["removed_layer_indices"])
        remaining.remove(int(best["candidate_layer"]))
        path.append(
            {
                "round": round_idx,
                "removed_layer": int(best["candidate_layer"]),
                "removed_layers": list(removed),
                "loss": best["loss"],
                "ppl": best["ppl"],
                "run_name": best["run_name"],
            }
        )

    return runs, path


def run_layer_removal_eval(config: LayerRemovalEvalConfig) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    tokens = _load_wikitext_eval_tokens(tokenizer=tokenizer, split=config.data_split, max_tokens=config.max_dataset_tokens)
    batches = _make_eval_batches(tokens, batch_size=config.batch_size, seq_len=config.seq_len, max_steps=config.eval_steps)
    if not batches:
        raise ValueError("not enough tokens to build an evaluation batch")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True, default=str) + "\n")

    model = _load_causal_lm(AutoModelForCausalLM, config.model_name, model_dtype).to(device)
    model.eval()
    layer_count = len(_extract_layers(_causal_lm_backbone(model)))
    runs: list[dict[str, Any]] = []

    if config.include_baseline:
        runs.append(
            _evaluate_lm_loop(
                batches=batches,
                output_dir=config.output_dir / "baseline",
                run_name="baseline",
                device=device,
                dtype=dtype,
                config=config,
                forward_fn=lambda input_ids: model(input_ids=input_ids, use_cache=False).logits,
            )
        )

    def evaluate_removed_layers(layer_group: tuple[int, ...], run_name: str, output_dir: Path) -> dict[str, Any]:
        originals = patch_layers_with_skip(model, layer_indices=layer_group)
        try:
            return _evaluate_lm_loop(
                batches=batches,
                output_dir=output_dir,
                run_name=run_name,
                device=device,
                dtype=dtype,
                config=config,
                forward_fn=lambda input_ids: model(input_ids=input_ids, use_cache=False).logits,
            )
        finally:
            restore_layers(model, originals)

    greedy_path: list[dict[str, Any]] = []
    if config.remove_layers is not None:
        layer_groups = (config.remove_layers,)
        for layer_group in layer_groups:
            layer_slug = "_".join(str(layer_idx) for layer_idx in layer_group)
            run_name = f"remove_layers_{layer_slug}"
            record = evaluate_removed_layers(tuple(layer_group), run_name, config.output_dir / run_name)
            record["removed_layers"] = len(layer_group)
            record["removed_layer_indices"] = list(layer_group)
            record["remaining_layers"] = layer_count - len(layer_group)
            runs.append(record)
    elif config.greedy_layer_removal:
        greedy_runs, greedy_path = greedy_removal_sweep(
            layer_count=layer_count,
            max_removed_layers=config.greedy_max_layers,
            output_dir=config.output_dir,
            evaluate_removed_layers=evaluate_removed_layers,
        )
        for record in greedy_runs:
            record["remaining_layers"] = layer_count - int(record["removed_layers"])
            runs.append(record)
    else:
        raise ValueError("set greedy_layer_removal=True or provide remove_layers")

    summary = {
        "runs": runs,
        "model_name": config.model_name,
        "layer_count": layer_count,
        "greedy_layer_removal": config.greedy_layer_removal,
        "greedy_path": greedy_path,
    }
    if greedy_path:
        (config.output_dir / "greedy_removal_path.json").write_text(json.dumps(greedy_path, indent=2, sort_keys=True, default=str) + "\n")
        _write_greedy_removal_path_markdown(config.output_dir / "greedy_removal_path.md", greedy_path)
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary


def _write_greedy_removal_path_markdown(path: Path, greedy_path: list[dict[str, Any]]) -> None:
    lines = [
        "| round | removed layer | removed layers | loss | ppl |",
        "| ---: | ---: | --- | ---: | ---: |",
    ]
    for row in greedy_path:
        layers = ",".join(str(layer_idx) for layer_idx in row["removed_layers"])
        lines.append(f"| {row['round']} | {row['removed_layer']} | {layers} | {float(row['loss']):.6f} | {float(row['ppl']):.6f} |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/layer_removal_eval"))
    parser.add_argument("--model-name", default="EleutherAI/pythia-1.4b")
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--data-split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--max-dataset-tokens", type=int, default=2_000_000)
    parser.add_argument("--ce-chunk-tokens", type=int, default=32768)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--greedy-max-layers", type=int, default=None)
    parser.add_argument("--remove-layers", default=None, help="Comma-separated layer indices to skip.")
    args = parser.parse_args(argv)
    remove_layers = None
    if args.remove_layers:
        remove_layers = tuple(int(part.strip()) for part in args.remove_layers.split(",") if part.strip())
    config = LayerRemovalEvalConfig(
        output_dir=args.output_dir,
        model_name=args.model_name,
        eval_steps=args.eval_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dtype=args.dtype,
        data_split=args.data_split,
        max_dataset_tokens=args.max_dataset_tokens,
        ce_chunk_tokens=args.ce_chunk_tokens,
        include_baseline=not args.skip_baseline,
        greedy_max_layers=args.greedy_max_layers,
        remove_layers=remove_layers,
    )
    print(json.dumps(run_layer_removal_eval(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
