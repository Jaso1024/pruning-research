from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from eval_da2k import MODEL_CONFIGS, resolve_device
from eval_gelu_relu_compensation_da2k import (
    evaluate_da2k_model,
    load_model,
    selected_annotations,
    transformer_mlp_names,
    write_summary,
)
from eval_rotated_twopiece_gelu_da2k import TwoPieceActivation, TwoPieceSpec, two_piece_spec


def activation_from_summary(summary_path: Path, variant_key: str) -> TwoPieceSpec:
    summary = json.loads(summary_path.read_text())
    try:
        activation = summary["variants"][variant_key]["metadata"]["activation"]
    except KeyError as exc:
        raise KeyError(f"could not find activation metadata for variant {variant_key!r} in {summary_path}") from exc
    return TwoPieceSpec(**activation)


def install_activation(model: torch.nn.Module, spec: TwoPieceSpec) -> list[str]:
    changed: list[str] = []
    for name in transformer_mlp_names(model):
        mlp = model.get_submodule(name)
        mlp.act = TwoPieceActivation(spec)
        changed.append(f"{name}.act")
    return changed


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    if args.summary_json is not None:
        spec = activation_from_summary(args.summary_json, args.variant_key)
    else:
        spec = two_piece_spec(args.activation)

    items = selected_annotations(
        args.dataset_root,
        scene_type=args.scene_type,
        max_images=args.max_images,
        max_pairs=args.max_pairs,
    )
    model = load_model(args.encoder, args.checkpoint, device)
    changed = install_activation(model, spec)
    state = torch.load(args.state_dict, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing}, unexpected={unexpected}")
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(device=device).eval()

    result = {
        "metadata": {
            "state_dict": str(args.state_dict),
            "summary_json": str(args.summary_json) if args.summary_json is not None else None,
            "variant_key": args.variant_key,
            "activation": spec.__dict__,
            "changed_modules": changed,
            "dataset_root": str(args.dataset_root),
            "checkpoint": str(args.checkpoint),
            "encoder": args.encoder,
            "input_size": args.input_size,
            "max_images": args.max_images,
            "max_pairs": args.max_pairs,
            "scene_type": args.scene_type,
            "device": str(device),
        },
        "evaluation": evaluate_da2k_model(
            model=model,
            dataset_root=args.dataset_root,
            items=items,
            input_size=args.input_size,
            device=device,
            log_every=args.log_every,
        ),
    }
    write_summary(args.output_json, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a folded rotated-two-piece Depth Anything state dict.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--variant-key", default="")
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--log-every", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result["evaluation"]["overall"], indent=2))


if __name__ == "__main__":
    main()
