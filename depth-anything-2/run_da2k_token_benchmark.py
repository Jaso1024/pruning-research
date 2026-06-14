from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eval_da2k import MODEL_CONFIGS, SCENE_CHOICES, check_dataset, setup_instructions


DEFAULT_VARIANTS = (
    "baseline",
    "tome",
    "attention-calib",
    "proxy-local-horizontal",
    "proxy-local-vertical",
)

PROXY_MODES = ("local-horizontal", "local-vertical", "local-checkerboard")
TOKEN_REDUCE_METHODS = ("pitome", "adamerge", "mctf", "evit", "ats", "ppt")
ACTUAL_TOKEN_METHODS = (
    "tome_actual",
    "evit_actual",
    "ats_actual",
    "token_pooling_actual",
    "pitome_actual",
    "adamerge_actual",
    "ppt_actual",
)


@dataclass(frozen=True)
class VariantCommand:
    name: str
    output_json: Path
    command: list[str]


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _run_python_import(python: Path, module: str, env: dict[str, str]) -> tuple[bool, str]:
    proc = subprocess.run(
        [str(python), "-c", f"import {module}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()


def build_env(depth_anything_root: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if depth_anything_root and depth_anything_root.exists():
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(depth_anything_root) + (os.pathsep + current if current else "")
    return env


def preflight(args: argparse.Namespace, env: dict[str, str]) -> list[str]:
    problems = []
    if not args.python.is_file():
        problems.append(f"python executable is missing: {args.python}")
    problems.extend(check_dataset(args.dataset_root))
    if not args.checkpoint.is_file():
        problems.append(f"checkpoint is missing: {args.checkpoint}")
    if args.python.is_file():
        for module in ("cv2", "depth_anything_v2.dpt"):
            ok, detail = _run_python_import(args.python, module, env)
            if not ok:
                problems.append(f"cannot import {module} with {args.python}: {detail}")
    return problems


def common_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--dataset-root", str(args.dataset_root),
        "--checkpoint", str(args.checkpoint),
        "--encoder", args.encoder,
        "--input-size", str(args.input_size),
        "--device", args.device,
        "--max-images", str(args.max_images),
        "--log-every", str(args.log_every),
    ]
    if args.scene_type:
        values.extend(["--scene-type", args.scene_type])
    return values


def build_variant_commands(args: argparse.Namespace) -> list[VariantCommand]:
    variants = _csv(args.variants)
    unknown = sorted(set(variants) - {
        "baseline",
        "tome",
        "attention-calib",
        "proxy-local-horizontal",
        "proxy-local-vertical",
        "proxy-local-checkerboard",
        *TOKEN_REDUCE_METHODS,
        *ACTUAL_TOKEN_METHODS,
    })
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(unknown)}")

    output_dir = args.output_dir
    commands: list[VariantCommand] = []
    base = [str(args.python)]
    shared = common_args(args)

    if "baseline" in variants:
        out = output_dir / "baseline.json"
        commands.append(VariantCommand("baseline", out, [
            *base, str(Path(__file__).with_name("eval_da2k.py")),
            *shared, "--output-json", str(out),
        ]))

    if "tome" in variants:
        out = output_dir / f"tome_r{args.merge_r}.json"
        commands.append(VariantCommand("tome", out, [
            *base, str(Path(__file__).with_name("eval_tome_da2k.py")),
            *shared, "--merge-r", str(args.merge_r), "--output-json", str(out),
        ]))

    if "attention-calib" in variants:
        out = output_dir / f"attention_calib_r{args.merge_r}_calib{args.calib_images}.json"
        commands.append(VariantCommand("attention-calib", out, [
            *base, str(Path(__file__).with_name("eval_attention_calib_merge_da2k.py")),
            *shared,
            "--merge-r", str(args.merge_r),
            "--calib-images", str(args.calib_images),
            "--score-mode", args.score_mode,
            "--external-lambda", str(args.external_lambda),
            "--top-pairs-per-layer", str(args.top_pairs_per_layer),
            "--candidate-multiplier", str(args.candidate_multiplier),
            "--output-dir", str(output_dir / "attention_calib_work"),
            "--output-json", str(out),
        ]))

    for mode in PROXY_MODES:
        variant = "proxy-" + mode
        if variant not in variants:
            continue
        out = output_dir / f"{variant.replace('-', '_')}_r{args.merge_r}.json"
        commands.append(VariantCommand(variant, out, [
            *base, str(Path(__file__).with_name("eval_dense_safe_proxy_da2k.py")),
            *shared,
            "--merge-r", str(args.merge_r),
            "--proxy-mode", mode,
            "--output-json", str(out),
        ]))

    for method in TOKEN_REDUCE_METHODS:
        if method not in variants:
            continue
        adaptive_tag = "_adaptive" if args.adaptive else ""
        out = output_dir / f"{method}_proxy_r{args.merge_r}{adaptive_tag}.json"
        command = [
            *base, str(Path(__file__).with_name("eval_token_reduce_methods_da2k.py")),
            *shared,
            "--method", method,
            "--merge-r", str(args.merge_r),
            "--salience-lambda", str(args.salience_lambda),
            "--size-lambda", str(args.size_lambda),
            "--protect-fraction", str(args.protect_fraction),
            "--output-json", str(out),
        ]
        if args.adaptive:
            command.append("--adaptive")
        commands.append(VariantCommand(method, out, command))

    for method in ACTUAL_TOKEN_METHODS:
        if method not in variants:
            continue
        out = output_dir / f"{method}_r{args.merge_r}.json"
        calib_cache = output_dir / f"{method}_calib_cache.json"
        command = [
            *base, str(Path(__file__).with_name("eval_actual_token_methods_da2k.py")),
            *shared,
            "--method", method,
            "--merge-r", str(args.merge_r),
            "--target-ratio", str(args.target_ratio),
            "--ats-mass-threshold", str(args.ats_mass_threshold),
            "--pooling-iters", str(args.pooling_iters),
            "--salience-lambda", str(args.salience_lambda),
            "--size-lambda", str(args.size_lambda),
            "--calib-images", str(args.calib_images),
            "--calib-cache", str(calib_cache),
            "--output-json", str(out),
        ]
        if args.force_calib:
            command.append("--force-calib")
        commands.append(VariantCommand(method, out, command))

    return commands


def run_command(command: VariantCommand, env: dict[str, str]) -> dict[str, Any]:
    started = time.monotonic()
    print(f"running {command.name}: {' '.join(shlex.quote(part) for part in command.command)}", flush=True)
    proc = subprocess.run(command.command, text=True, env=env)
    record = {
        "variant": command.name,
        "output_json": str(command.output_json),
        "command": command.command,
        "returncode": proc.returncode,
        "elapsed_seconds": time.monotonic() - started,
    }
    if command.output_json.is_file():
        record["output_exists"] = True
    else:
        record["output_exists"] = False
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DA-2K token-reduction benchmark variants with JSON outputs.")
    parser.add_argument("--python", type=Path, default=Path("/home/ubuntu/not_jason/cot2_eval_venv/bin/python"))
    parser.add_argument("--depth-anything-root", type=Path, default=Path("/home/ubuntu/Depth-Anything-V2"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/da2k_token_benchmark"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--merge-r", type=int, default=57)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--salience-lambda", type=float, default=1.0)
    parser.add_argument("--size-lambda", type=float, default=0.25)
    parser.add_argument("--protect-fraction", type=float, default=0.15)
    parser.add_argument("--calib-images", type=int, default=64)
    parser.add_argument("--target-ratio", type=float, default=0.0)
    parser.add_argument("--ats-mass-threshold", type=float, default=0.90)
    parser.add_argument("--pooling-iters", type=int, default=4)
    parser.add_argument("--force-calib", action="store_true")
    parser.add_argument("--score-mode", choices=["high_mutual", "mutual_minus_external"], default="mutual_minus_external")
    parser.add_argument("--external-lambda", type=float, default=1.0)
    parser.add_argument("--top-pairs-per-layer", type=int, default=50_000)
    parser.add_argument("--candidate-multiplier", type=int, default=64)
    parser.add_argument("--scene-type", default="", choices=SCENE_CHOICES)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    env = build_env(args.depth_anything_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_preflight:
        problems = preflight(args, env)
        if problems:
            print("Cannot run DA-2K benchmark yet:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            print("", file=sys.stderr)
            print(setup_instructions(
                dataset_root=args.dataset_root,
                checkpoint=args.checkpoint,
                encoder=args.encoder,
            ), file=sys.stderr)
            raise SystemExit(2)

    commands = build_variant_commands(args)
    manifest: dict[str, Any] = {
        "metadata": {
            **vars(args),
            "python": str(args.python),
            "dataset_root": str(args.dataset_root),
            "checkpoint": str(args.checkpoint),
            "output_dir": str(args.output_dir),
            "depth_anything_root": str(args.depth_anything_root),
            "commands": [asdict(command) for command in commands],
        },
        "results": [],
    }

    if args.dry_run:
        for command in commands:
            print(" ".join(shlex.quote(part) for part in command.command))
        manifest["dry_run"] = True
    else:
        for command in commands:
            record = run_command(command, env)
            manifest["results"].append(record)
            if record["returncode"] != 0 and not args.continue_on_error:
                break

    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {manifest_path}")

    failures = [item for item in manifest["results"] if item.get("returncode") not in (None, 0)]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
