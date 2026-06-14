from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval_da2k import MODEL_CONFIGS, resolve_device
from eval_gelu_relu_compensation_da2k import load_model, selected_annotations
from eval_structured_subcircuit_pruning_metrics_da2k import (
    DEFAULT_CANDIDATE_KINDS,
    evaluate_items,
    metric_scores,
    parse_csv,
    parse_int_csv,
    selected_rows,
    structural_scores,
)


WEIGHT_MASK_METRICS = {
    "ablation_correct",
    "ablation_margin",
    "magnitude",
    "safe_magnitude",
    "stability",
    "param_eff_correct",
    "param_eff_margin",
    "stability_param",
    "random",
}


@dataclass(frozen=True)
class WeightMaskedPruningConfig:
    dataset_root: Path
    checkpoint: Path
    circuit_summary: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    metrics: tuple[str, ...] = ("stability_param",)
    candidate_kinds: tuple[str, ...] = DEFAULT_CANDIDATE_KINDS
    budget_nodes: tuple[int, ...] = (50, 100)
    budget_values: tuple[int, ...] = ()
    scene_type: str = ""
    skip_images: int = 0
    max_images: int = 0
    seed: int = 123
    log_every: int = 50
    save_checkpoints: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "circuit_summary", Path(self.circuit_summary))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if any(metric not in WEIGHT_MASK_METRICS for metric in self.metrics):
            raise ValueError(
                "weight masking only supports metrics that do not require activation calibration: "
                f"{sorted(WEIGHT_MASK_METRICS)}"
            )
        if any(budget <= 0 for budget in self.budget_nodes):
            raise ValueError("budget_nodes must be positive")
        if any(budget <= 0 for budget in self.budget_values):
            raise ValueError("budget_values must be positive")
        if self.skip_images < 0:
            raise ValueError("skip_images must be non-negative")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")


def select_nodes(rows: list[dict[str, Any]], scores: dict[str, float], budget: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (scores[str(row["name"])], str(row["name"])))
    return ranked[: min(budget, len(ranked))]


def select_nodes_by_value_budget(rows: list[dict[str, Any]], scores: dict[str, float], budget: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (scores[str(row["name"])], str(row["name"])))
    selected: list[dict[str, Any]] = []
    running = 0
    for row in ranked:
        selected.append(row)
        running += int(row.get("parameter_estimate", 0))
        if running >= budget:
            break
    return selected


def _zero_linear_rows(linear: torch.nn.Linear, start: int, end: int, *, offset: int = 0) -> int:
    row_start = offset + start
    row_end = offset + end
    with torch.no_grad():
        linear.weight[row_start:row_end, :].zero_()
        if linear.bias is not None:
            linear.bias[row_start:row_end].zero_()
    return int((row_end - row_start) * linear.weight.shape[1] + (row_end - row_start if linear.bias is not None else 0))


def _zero_linear_cols(linear: torch.nn.Linear, start: int, end: int) -> int:
    with torch.no_grad():
        linear.weight[:, start:end].zero_()
    return int(linear.weight.shape[0] * (end - start))


def _zero_conv_output(module: torch.nn.Module, start: int, end: int) -> int:
    with torch.no_grad():
        if isinstance(module, torch.nn.ConvTranspose2d):
            module.weight[:, start:end, :, :].zero_()
        elif isinstance(module, torch.nn.Conv2d):
            module.weight[start:end, :, :, :].zero_()
        else:
            raise TypeError(f"unsupported conv module: {type(module)!r}")
        if module.bias is not None:
            module.bias[start:end].zero_()
    kernel = int(module.weight.shape[-1] * module.weight.shape[-2])
    if isinstance(module, torch.nn.ConvTranspose2d):
        weight_count = int(module.weight.shape[0] * (end - start) * kernel)
    else:
        weight_count = int((end - start) * module.weight.shape[1] * kernel)
    return weight_count + int(end - start if module.bias is not None else 0)


def _zero_conv_input(module: torch.nn.Module, start: int, end: int) -> int:
    with torch.no_grad():
        if isinstance(module, torch.nn.ConvTranspose2d):
            module.weight[start:end, :, :, :].zero_()
        elif isinstance(module, torch.nn.Conv2d):
            module.weight[:, start:end, :, :].zero_()
        else:
            raise TypeError(f"unsupported conv module: {type(module)!r}")
    kernel = int(module.weight.shape[-1] * module.weight.shape[-2])
    if isinstance(module, torch.nn.ConvTranspose2d):
        return int((end - start) * module.weight.shape[1] * kernel)
    return int(module.weight.shape[0] * (end - start) * kernel)


def apply_weight_masks(model: torch.nn.Module, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["module_name"])].append(row)

    for module_name, module_rows in sorted(grouped.items()):
        module = model.get_submodule(module_name)
        for row in sorted(module_rows, key=lambda item: str(item["name"])):
            kind = str(row["kind"])
            start = int(row["start"])
            end = int(row["end"])
            masked_values = 0
            target = module_name

            if kind in {"attn_q_head", "attn_k_head", "attn_v_head", "attn_q_group", "attn_k_group", "attn_v_group"}:
                qkv = module.qkv
                if not isinstance(qkv, torch.nn.Linear):
                    raise TypeError(f"{module_name}.qkv is not Linear")
                channels = int(qkv.in_features)
                offset = {
                    "attn_q_head": 0,
                    "attn_q_group": 0,
                    "attn_k_head": channels,
                    "attn_k_group": channels,
                    "attn_v_head": 2 * channels,
                    "attn_v_group": 2 * channels,
                }[kind]
                masked_values = _zero_linear_rows(qkv, start, end, offset=offset)
                target = module_name + ".qkv"

            elif kind == "attn_proj_group":
                proj = module.proj
                if not isinstance(proj, torch.nn.Linear):
                    raise TypeError(f"{module_name}.proj is not Linear")
                masked_values = _zero_linear_rows(proj, start, end)
                target = module_name + ".proj"

            elif kind == "mlp_group":
                fc1 = module.fc1
                fc2 = module.fc2
                if not isinstance(fc1, torch.nn.Linear) or not isinstance(fc2, torch.nn.Linear):
                    raise TypeError(f"{module_name}.fc1/fc2 are not Linear")
                # Zero fc1 rows+bias to make the hidden group identically zero,
                # and zero fc2 columns so the checkpoint carries the full
                # structured mask for later shape surgery.
                masked_values = _zero_linear_rows(fc1, start, end) + _zero_linear_cols(fc2, start, end)
                target = module_name + ".fc1/fc2"

            elif kind == "head_channel_group":
                if not isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    raise TypeError(f"{module_name} is not Conv2d/ConvTranspose2d")
                masked_values = _zero_conv_output(module, start, end)

            elif kind == "head_input_channel_group":
                if not isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    raise TypeError(f"{module_name} is not Conv2d/ConvTranspose2d")
                masked_values = _zero_conv_input(module, start, end)

            else:
                raise ValueError(f"unsupported mask kind: {kind}")

            operations.append(
                {
                    "name": row["name"],
                    "kind": kind,
                    "module_name": module_name,
                    "target": target,
                    "start": start,
                    "end": end,
                    "parameter_estimate": int(row.get("parameter_estimate", 0)),
                    "masked_tensor_values": masked_values,
                }
            )

    return operations


def run(config: WeightMaskedPruningConfig) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    started = time.monotonic()
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True, default=str) + "\n")

    summary = json.loads(config.circuit_summary.read_text())
    rows = selected_rows(summary, config.candidate_kinds)
    if not rows:
        raise RuntimeError("no candidate rows selected")

    all_items = selected_annotations(
        config.dataset_root,
        scene_type=config.scene_type,
        max_images=0,
        max_pairs=0,
    )
    items = all_items[config.skip_images :]
    if config.max_images > 0:
        items = items[: config.max_images]
    if not items:
        raise RuntimeError("no DA-2K annotations selected")

    dense_model = load_model(config.encoder, config.checkpoint, device).to(device).eval()
    for param in dense_model.parameters():
        param.requires_grad_(False)
    baseline = evaluate_items(
        model=dense_model,
        items=items,
        dataset_root=config.dataset_root,
        input_size=config.input_size,
        device=device,
    )
    baseline_correct = int(baseline["overall"]["larger_correct"])
    baseline_acc = float(baseline["overall"]["larger_is_closer_accuracy"])
    magnitude_scores = (
        structural_scores(dense_model, rows, input_rms=None)
        if any(metric == "magnitude" for metric in config.metrics)
        else {}
    )
    del dense_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results: list[dict[str, Any]] = []
    total_runs = len(config.metrics) * (len(config.budget_nodes) + len(config.budget_values))
    run_index = 0
    for metric in config.metrics:
        scores = metric_scores(
            metric=metric,
            rows=rows,
            magnitude_scores=magnitude_scores,
            wanda_scores={},
            seed=config.seed,
        )
        budget_specs = [("nodes", budget) for budget in config.budget_nodes]
        budget_specs.extend(("values", budget) for budget in config.budget_values)
        for budget_kind, budget in budget_specs:
            run_index += 1
            if budget_kind == "values":
                selected = select_nodes_by_value_budget(rows, scores, budget)
            else:
                selected = select_nodes(rows, scores, budget)
            model = load_model(config.encoder, config.checkpoint, device).to(device).eval()
            for param in model.parameters():
                param.requires_grad_(False)
            operations = apply_weight_masks(model, selected)
            result = evaluate_items(
                model=model,
                items=items,
                dataset_root=config.dataset_root,
                input_size=config.input_size,
                device=device,
            )

            checkpoint_path = None
            if config.save_checkpoints:
                checkpoint_path = config.output_dir / f"{metric}_budget{budget_kind}{budget}_masked.pth"
                torch.save(model.state_dict(), checkpoint_path)

            row = {
                "metric": metric,
                "budget_kind": budget_kind,
                "budget_nodes": budget,
                "budget": budget,
                "selected_nodes": len(selected),
                "selected_parameter_estimate": int(sum(int(node.get("parameter_estimate", 0)) for node in selected)),
                "masked_tensor_values": int(sum(int(op["masked_tensor_values"]) for op in operations)),
                "overall": result["overall"],
                "by_scene": result["by_scene"],
                "mean_margin": result["mean_margin"],
                "accuracy_drop": baseline_acc - float(result["overall"]["larger_is_closer_accuracy"]),
                "correct_drop": baseline_correct - int(result["overall"]["larger_correct"]),
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                "selected": [
                    {
                        "name": node["name"],
                        "kind": node["kind"],
                        "module_name": node["module_name"],
                        "start": node["start"],
                        "end": node["end"],
                        "score": scores[str(node["name"])],
                        "correct_drop": node.get("correct_drop"),
                        "mean_margin_drop": node.get("mean_margin_drop"),
                        "mean_abs_margin_delta": node.get("pair_delta", {}).get("mean_abs_margin_delta"),
                        "parameter_estimate": node.get("parameter_estimate"),
                    }
                    for node in selected
                ],
                "mask_operations": operations,
            }
            results.append(row)
            print(
                json.dumps(
                    {
                        "runs_done": run_index,
                        "runs_total": total_runs,
                        "metric": metric,
                        "budget_kind": budget_kind,
                        "budget_nodes": budget,
                        "budget": budget,
                        "accuracy": row["overall"]["larger_is_closer_accuracy"],
                        "correct_drop": row["correct_drop"],
                        "selected_parameter_estimate": row["selected_parameter_estimate"],
                        "masked_tensor_values": row["masked_tensor_values"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results_by_accuracy = sorted(results, key=lambda row: (int(row["correct_drop"]), float(row["accuracy_drop"]), row["metric"], row["budget_nodes"]))
    output = {
        "config": asdict(config),
        "device": str(device),
        "baseline": baseline,
        "candidate_count": len(rows),
        "results": results,
        "results_by_accuracy": results_by_accuracy,
        "metadata": {
            "elapsed_seconds": time.monotonic() - started,
            "method": "Persistent structured weight masks from fine subcircuit rankings; no forward hooks during evaluation.",
        },
    }
    (config.output_dir / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply persistent structured subcircuit masks and evaluate DAV2 on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--circuit-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/weight_masked_subcircuit_pruning"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metrics", default="stability_param")
    parser.add_argument("--candidate-kinds", default=",".join(DEFAULT_CANDIDATE_KINDS))
    parser.add_argument("--budget-nodes", default="50,100")
    parser.add_argument("--budget-values", default="", help="Comma-separated target masked-value budgets. Candidates are selected until parameter_estimate reaches each target.")
    parser.add_argument("--scene-type", default="")
    parser.add_argument("--skip-images", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = WeightMaskedPruningConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        circuit_summary=args.circuit_summary,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        metrics=parse_csv(args.metrics),
        candidate_kinds=parse_csv(args.candidate_kinds),
        budget_nodes=parse_int_csv(args.budget_nodes),
        budget_values=parse_int_csv(args.budget_values),
        scene_type=args.scene_type,
        skip_images=args.skip_images,
        max_images=args.max_images,
        seed=args.seed,
        log_every=args.log_every,
        save_checkpoints=bool(args.save_checkpoints),
    )
    summary = run(config)
    baseline = summary["baseline"]["overall"]
    print(
        json.dumps(
            {
                "output_dir": str(config.output_dir),
                "baseline": {
                    "pairs": baseline["pairs"],
                    "larger_correct": baseline["larger_correct"],
                    "larger_is_closer_accuracy": baseline["larger_is_closer_accuracy"],
                },
                "best_results": [
                    {
                        "metric": row["metric"],
                        "budget_kind": row["budget_kind"],
                        "budget": row["budget"],
                        "budget_nodes": row["budget_nodes"],
                        "correct_drop": row["correct_drop"],
                        "accuracy": row["overall"]["larger_is_closer_accuracy"],
                        "selected_parameter_estimate": row["selected_parameter_estimate"],
                        "masked_tensor_values": row["masked_tensor_values"],
                    }
                    for row in summary["results_by_accuracy"][:20]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
