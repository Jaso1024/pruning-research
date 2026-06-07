from __future__ import annotations

import argparse
import json
from pathlib import Path

from saliency.affine_scaffold import AffineScaffoldConfig, run_affine_scaffold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fixed random vector pairs and fit y = A x + b.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=8192)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--output-dim", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="float32", choices=["fp32", "float32", "fp64", "float64", "double"])
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--output-scale", type=float, default=1.0)
    parser.add_argument("--task", default="classification", choices=["regression", "classification"])
    parser.add_argument("--solver", default="auto", choices=["auto", "normal", "lstsq"])
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_affine_scaffold(
        AffineScaffoldConfig(
            output_dir=args.output_dir,
            num_pairs=args.num_pairs,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            seed=args.seed,
            dtype=args.dtype,
            input_scale=args.input_scale,
            output_scale=args.output_scale,
            task=args.task,
            solver=args.solver,
            ridge=args.ridge,
            device=args.device,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
