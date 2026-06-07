from __future__ import annotations

import argparse
import json
from pathlib import Path

from saliency.affine_scaffold import AffinePruneEvalConfig, DEFAULT_AFFINE_PRUNING_METHODS, run_affine_prune_eval


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate affine pruning methods on saved random vector pairs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_AFFINE_PRUNING_METHODS))
    parser.add_argument("--prune-fractions", default="0.05,0.10,0.25,0.50")
    parser.add_argument("--pruning-scope", default="global", choices=["global", "per_output_row"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--damp-percent", type=float, default=0.01)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=1e-6)
    parser.add_argument("--cholesky-scale", type=float, default=1e4)
    parser.add_argument("--num-blocks", type=int, default=100)
    parser.add_argument("--no-activation-order", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_affine_prune_eval(
        AffinePruneEvalConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            methods=_parse_csv_strings(args.methods),
            prune_fractions=_parse_csv_floats(args.prune_fractions),
            pruning_scope=args.pruning_scope,
            seed=args.seed,
            device=args.device,
            damp_percent=args.damp_percent,
            blocksize=args.blocksize,
            percdamp=args.percdamp,
            cholesky_scale=args.cholesky_scale,
            num_blocks=args.num_blocks,
            use_activation_order=not args.no_activation_order,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
