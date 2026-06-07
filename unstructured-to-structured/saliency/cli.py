from __future__ import annotations

import argparse
import json
from pathlib import Path

from saliency.experiment import SaliencyConfig, run_saliency_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute parameter saliency on a calibration dataset.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="EleutherAI/pythia-31m")
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "float32", "bf16", "bfloat16", "fp16", "float16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--revision")
    parser.add_argument("--full-sequence-loss", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SaliencyConfig(
        output_dir=args.output_dir,
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
        top_k=args.top_k,
        revision=args.revision,
    )
    summary = run_saliency_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
