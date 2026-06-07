from __future__ import annotations

import argparse
import json
from pathlib import Path

from saliency.prune_eval import PruneEvalConfig, run_prune_ppl_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GSM8K perplexity after saliency-guided matrix pruning.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--saliency-path", type=Path, required=True)
    parser.add_argument("--model-name", default="EleutherAI/pythia-31m")
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "float32", "bf16", "bfloat16", "fp16", "float16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prune-fraction", type=float, default=0.5)
    parser.add_argument("--pruning-scope", choices=["per_matrix", "per_output_row", "global"], default="per_matrix")
    parser.add_argument("--revision")
    parser.add_argument("--full-sequence-loss", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PruneEvalConfig(
        output_dir=args.output_dir,
        saliency_path=args.saliency_path,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
        device=args.device,
        answer_only_loss=not args.full_sequence_loss,
        prune_fraction=args.prune_fraction,
        pruning_scope=args.pruning_scope,
        revision=args.revision,
    )
    print(json.dumps(run_prune_ppl_experiment(config), indent=2))


if __name__ == "__main__":
    main()
