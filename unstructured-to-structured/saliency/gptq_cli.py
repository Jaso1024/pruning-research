from __future__ import annotations

import argparse
import json
from pathlib import Path

from saliency.gptq_eval import GPTQConfig, run_gptq_fp8_experiment


def parse_steps(value: str) -> tuple[int, ...]:
    steps = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not steps:
        raise argparse.ArgumentTypeError("step list cannot be empty")
    return steps


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("float list cannot be empty")
    if any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("all float list values must be positive")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GSM8K test perplexity after GPTQ FP8 quantization.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="EleutherAI/pythia-31m")
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--max-calibration-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--calibration-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "float32", "bf16", "bfloat16", "fp16", "float16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--damp-percent", type=float, default=0.01)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--gptq-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=parse_steps)
    parser.add_argument("--staged-to-wq", action="store_true")
    parser.add_argument("--iterative-damped-gptq", action="store_true")
    parser.add_argument("--gradient-descent-gptq", action="store_true")
    parser.add_argument("--newton-step-alpha", type=float)
    parser.add_argument("--gradient-step-scale", type=float, default=1.0)
    parser.add_argument("--gradient-step-scales", type=parse_float_list)
    parser.add_argument("--hessian-approximation", choices=["full", "diagonal"], default="full")
    parser.add_argument("--revision")
    parser.add_argument("--full-sequence-loss", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = GPTQConfig(
        output_dir=args.output_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        calibration_split=args.calibration_split,
        eval_split=args.eval_split,
        max_calibration_examples=args.max_calibration_examples,
        max_eval_examples=args.max_eval_examples,
        calibration_batch_size=args.calibration_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
        device=args.device,
        answer_only_loss=not args.full_sequence_loss,
        damp_percent=args.damp_percent,
        blocksize=args.blocksize,
        gptq_steps=args.gptq_steps,
        eval_steps=args.eval_steps,
        staged_to_wq=args.staged_to_wq,
        iterative_damped_gptq=args.iterative_damped_gptq,
        gradient_descent_gptq=args.gradient_descent_gptq,
        newton_step_alpha=args.newton_step_alpha,
        gradient_step_scale=args.gradient_step_scale,
        gradient_step_scales=args.gradient_step_scales,
        hessian_approximation=args.hessian_approximation,
        revision=args.revision,
    )
    print(json.dumps(run_gptq_fp8_experiment(config), indent=2))


if __name__ == "__main__":
    main()
