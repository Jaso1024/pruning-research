from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from eval_da2k import MODEL_CONFIGS, load_model, resolve_device, selected_annotations
from eval_circuit_lora_repair_da2k import (
    LORA_PLACEMENTS,
    PEFT_METHODS,
    add_peft_modules_,
    cache_teacher_targets,
    evaluate_items,
    freeze_except_peft_,
    load_train_samples,
    lora_target_names_for_placement,
    merge_peft_modules_,
    reapply_masks_,
    train_lora_repair,
    trainable_params,
)

REMOTE_DA2K_DIR = Path("/home/ubuntu/remote-work/depth-anything-2")
if str(REMOTE_DA2K_DIR) not in sys.path:
    sys.path.insert(0, str(REMOTE_DA2K_DIR))

from eval_scalar_weight_circuit_da2k import collect_scalar_scores, global_pruning_order, pruning_scores
from eval_wanda_unstructured_da2k import find_prunable_linears


SCALAR_SCORE_NAMES = (
    "abs_wgrad",
    "positive_wgrad",
    "magnitude",
    "hybrid_abs_mag",
    "hybrid_protect_mag",
    "taylor1_abs",
    "taylor2_abs",
    "taylor3_abs",
    "taylor1_damage",
    "taylor2_damage",
    "taylor3_damage",
)


@dataclass(frozen=True)
class ScalarMaskLoRARepairConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    target: str = "transformer"
    layer_indices: tuple[int, ...] = ()
    scene_type: str = ""
    train_images: int = 64
    eval_skip_images: int = 64
    eval_images: int = 128
    allow_train_eval_overlap: bool = False
    score_images: int = 32
    score_cache: Path | None = None
    save_score_cache: bool = False
    prune_score: str = "taylor2_abs"
    hybrid_alpha: float = 1.0
    loss_tau: float = 1.0
    budget_values: int = 5_014_736
    peft_method: str = "lora"
    lora_placement: str = "masked"
    lora_window: int = 1
    lora_module_set: str = "all"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    pairwise_weight: float = 0.0
    pairwise_teacher_weight: float = 1.0
    pairwise_label_weight: float = 0.0
    pairwise_tau: float = 0.25
    seed: int = 123
    log_every: int = 12
    save_checkpoint: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.score_cache is not None:
            object.__setattr__(self, "score_cache", Path(self.score_cache))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.target not in {"transformer", "all-linear"}:
            raise ValueError("target must be transformer or all-linear")
        if self.train_images <= 0 or self.score_images <= 0:
            raise ValueError("train_images and score_images must be positive")
        if self.eval_skip_images < 0 or self.eval_images < 0:
            raise ValueError("eval_skip_images/eval_images must be non-negative")
        if not self.allow_train_eval_overlap and self.eval_skip_images < self.train_images:
            raise ValueError("eval_skip_images must be >= train_images unless --allow-train-eval-overlap is set")
        if self.prune_score not in SCALAR_SCORE_NAMES:
            raise ValueError(f"unknown prune score: {self.prune_score}")
        if self.hybrid_alpha < 0:
            raise ValueError("hybrid_alpha must be non-negative")
        if self.loss_tau <= 0 or self.pairwise_tau <= 0:
            raise ValueError("loss temperatures must be positive")
        if self.budget_values <= 0:
            raise ValueError("budget_values must be positive")
        if self.peft_method not in PEFT_METHODS:
            raise ValueError(f"unknown PEFT method: {self.peft_method}")
        placements = tuple(part for part in self.lora_placement.split("+") if part)
        if not placements or any(placement not in LORA_PLACEMENTS for placement in placements):
            raise ValueError("unknown lora_placement")
        if self.lora_window < 0:
            raise ValueError("lora_window must be non-negative")
        if self.lora_module_set not in {"all", "attn", "mlp"}:
            raise ValueError("lora_module_set must be all, attn, or mlp")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if self.epochs <= 0 or self.lr <= 0:
            raise ValueError("epochs/lr must be positive")
        if self.pairwise_weight < 0 or self.pairwise_teacher_weight < 0 or self.pairwise_label_weight < 0:
            raise ValueError("pairwise weights must be non-negative")


def parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def load_or_collect_scores(
    config: ScalarMaskLoRARepairConfig,
    *,
    all_items: list[tuple[str, list[dict[str, Any]]]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[str], dict[str, Any]]:
    if config.score_cache is not None and config.score_cache.exists():
        payload = torch.load(config.score_cache, map_location="cpu", weights_only=False)
        scores = payload["pruning_scores"]
        module_names = list(payload["module_names"])
        metadata = dict(payload.get("score_metadata", {}))
        metadata["loaded_score_cache"] = str(config.score_cache)
        return scores, module_names, metadata

    score_model = load_model(config.encoder, config.checkpoint, device)
    module_names = find_prunable_linears(score_model, target=config.target, layer_indices=config.layer_indices)
    if not module_names:
        raise RuntimeError("no prunable linear modules selected")
    score_items = all_items[: config.score_images]
    accum, metadata = collect_scalar_scores(
        model=score_model,
        module_names=module_names,
        items=score_items,
        dataset_root=config.dataset_root,
        input_size=config.input_size,
        device=device,
        log_every=config.log_every,
        loss_tau=config.loss_tau,
    )
    scores = pruning_scores(accum, config.prune_score, config.hybrid_alpha)
    if config.save_score_cache:
        cache_path = config.output_dir / "scalar_scores.pt"
        torch.save(
            {
                "accumulators": accum,
                "pruning_scores": scores,
                "module_names": module_names,
                "config": asdict(config),
                "score_metadata": metadata,
            },
            cache_path,
        )
        metadata["saved_score_cache"] = str(cache_path)
    del score_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores, module_names, metadata


def apply_scalar_mask_(
    model: nn.Module,
    *,
    scores: dict[str, torch.Tensor],
    module_names: list[str],
    budget: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[dict[str, Any]]]:
    order, slices = global_pruning_order(scores, module_names, budget)
    masks: dict[str, torch.Tensor] = {}
    operations: list[dict[str, Any]] = []
    touched_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for name, start, end in slices:
            left = int(torch.searchsorted(order, start, right=False).item())
            right = int(torch.searchsorted(order, end, right=False).item())
            local = order[left:right] - start
            if local.numel() == 0:
                continue
            module = model.get_submodule(name)
            if not isinstance(module, nn.Linear):
                continue
            mask = torch.zeros_like(module.weight.detach(), dtype=torch.bool, device="cpu")
            mask_flat = mask.flatten()
            mask_flat[local] = True
            flat = module.weight.flatten()
            flat[local.to(device=flat.device)] = 0
            masks[name] = mask
            operations.append(
                {
                    "module_name": name,
                    "shape": list(module.weight.shape),
                    "masked_tensor_values": int(local.numel()),
                    "masked_fraction": float(local.numel() / max(module.weight.numel(), 1)),
                }
            )
            touched_rows.append({"module_name": name})
    summary = {
        "budget_values": int(budget),
        "masked_tensor_values": int(sum(row["masked_tensor_values"] for row in operations)),
        "modules_touched": len(operations),
        "target_weight_count": int(sum(int(scores[name].numel()) for name in module_names if name in scores)),
    }
    summary["zero_fraction"] = summary["masked_tensor_values"] / max(summary["target_weight_count"], 1)
    return masks, summary, touched_rows


def result_row(label: str, result: dict[str, Any]) -> dict[str, Any]:
    overall = result["overall"]
    return {
        "label": label,
        "accuracy": overall["larger_is_closer_accuracy"],
        "correct": overall["larger_correct"],
        "pairs": overall["pairs"],
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Scalar Mask LoRA Repair DA2K", ""]
    lines.append("## Results")
    lines.append("")
    lines.append("| model | accuracy | correct | pairs |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in summary["results"]:
        overall = row["overall"]
        lines.append(
            f"| `{row['label']}` | {overall['larger_is_closer_accuracy']:.4f} | {overall['larger_correct']} | {overall['pairs']} |"
        )
    lines.append("")
    lines.append("## Repair")
    lines.append("")
    lines.append(f"- Prune score: `{summary['config']['prune_score']}`")
    lines.append(f"- Masked tensor values: `{summary['mask_summary']['masked_tensor_values']}`")
    lines.append(f"- Target zero fraction: `{summary['mask_summary']['zero_fraction']:.4f}`")
    lines.append(f"- PEFT method: `{summary['config']['peft_method']}`")
    lines.append(f"- LoRA placement: `{summary['config']['lora_placement']}`")
    lines.append(f"- PEFT trainable params: `{summary['peft_trainable_params']}`")
    lines.append(f"- Train/eval overlap images: `{summary['train_eval_overlap_count']}`")
    path.write_text("\n".join(lines) + "\n")


def run(config: ScalarMaskLoRARepairConfig) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    started = time.monotonic()
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True, default=str) + "\n")

    all_items = selected_annotations(config.dataset_root, scene_type=config.scene_type, max_images=0)
    train_items = all_items[: config.train_images]
    eval_end = None if config.eval_images == 0 else config.eval_skip_images + config.eval_images
    eval_items = all_items[config.eval_skip_images : eval_end]
    if not train_items or not eval_items:
        raise RuntimeError("empty train/eval item selection")
    train_paths = {item[0] for item in train_items}
    eval_paths = {item[0] for item in eval_items}
    overlap_paths = sorted(train_paths & eval_paths)
    if overlap_paths and not config.allow_train_eval_overlap:
        raise RuntimeError("train/eval image overlap detected")

    scores, module_names, score_metadata = load_or_collect_scores(config, all_items=all_items, device=device)

    teacher = load_model(config.encoder, config.checkpoint, device)
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher.eval()

    student = load_model(config.encoder, config.checkpoint, device)
    masks, mask_summary, touched_rows = apply_scalar_mask_(
        student,
        scores=scores,
        module_names=module_names,
        budget=config.budget_values,
    )
    mask_operations = [
        {
            "module_name": name,
            "masked_tensor_values": int(mask.sum().item()),
            "shape": list(mask.shape),
        }
        for name, mask in sorted(masks.items())
    ]
    lora_targets = lora_target_names_for_placement(
        student,
        masked_names=sorted(masks),
        selected_rows=touched_rows,
        placement=config.lora_placement,
        window=config.lora_window,
        module_set=config.lora_module_set,
    )
    add_stats = add_peft_modules_(
        student,
        lora_targets,
        method=config.peft_method,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        masks=masks,
    )
    freeze_except_peft_(student)
    peft_param_count = trainable_params(student)

    train_samples = load_train_samples(
        teacher,
        config.dataset_root,
        train_items,
        input_size=config.input_size,
        device=device,
    )
    teacher_outputs, teacher_pair_margins = cache_teacher_targets(teacher, train_samples, device)

    results: list[dict[str, Any]] = []
    results.append(
        evaluate_items(
            teacher,
            eval_items,
            dataset_root=config.dataset_root,
            input_size=config.input_size,
            device=device,
            label="dense_teacher",
        )
    )

    pruned_eval_model = load_model(config.encoder, config.checkpoint, device)
    apply_scalar_mask_(
        pruned_eval_model,
        scores=scores,
        module_names=module_names,
        budget=config.budget_values,
    )
    results.append(
        evaluate_items(
            pruned_eval_model,
            eval_items,
            dataset_root=config.dataset_root,
            input_size=config.input_size,
            device=device,
            label="pruned_student",
        )
    )
    del pruned_eval_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    history = train_lora_repair(
        student,
        train_samples,
        teacher_outputs,
        teacher_pair_margins,
        device=device,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        pairwise_weight=config.pairwise_weight,
        pairwise_teacher_weight=config.pairwise_teacher_weight,
        pairwise_label_weight=config.pairwise_label_weight,
        pairwise_tau=config.pairwise_tau,
        log_every=config.log_every,
    )
    results.append(
        evaluate_items(
            student,
            eval_items,
            dataset_root=config.dataset_root,
            input_size=config.input_size,
            device=device,
            label="peft_repaired_unmerged",
        )
    )

    merge_stats = merge_peft_modules_(student)
    results.append(
        evaluate_items(
            student,
            eval_items,
            dataset_root=config.dataset_root,
            input_size=config.input_size,
            device=device,
            label="folded_peft_unmasked",
        )
    )

    reapply_masks_(student, masks)
    results.append(
        evaluate_items(
            student,
            eval_items,
            dataset_root=config.dataset_root,
            input_size=config.input_size,
            device=device,
            label="folded_peft_remasked",
        )
    )

    checkpoint_path = None
    if config.save_checkpoint:
        checkpoint_path = config.output_dir / "folded_peft_remasked.pth"
        torch.save(student.state_dict(), checkpoint_path)

    summary = {
        "config": asdict(config),
        "device": str(device),
        "train_items": [item[0] for item in train_items],
        "eval_items": [item[0] for item in eval_items],
        "train_eval_overlap_count": len(overlap_paths),
        "train_eval_overlap_items": overlap_paths,
        "score_metadata": {key: value for key, value in score_metadata.items() if key != "image_rows"},
        "target_modules": module_names,
        "mask_summary": mask_summary,
        "mask_operations": mask_operations,
        "mask_modules": sorted(masks),
        "peft_modules": add_stats,
        "peft_trainable_params": peft_param_count,
        "history": history,
        "results": results,
        "merge_rms_delta_by_module": merge_stats,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "elapsed_seconds": time.monotonic() - started,
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    write_report(config.output_dir / "summary.md", summary)
    print(
        json.dumps(
            {
                "output_dir": str(config.output_dir),
                "elapsed_seconds": summary["elapsed_seconds"],
                "mask_summary": mask_summary,
                "peft_trainable_params": peft_param_count,
                "results": [result_row(row["label"], row) for row in results],
                "train_eval_overlap_count": len(overlap_paths),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scalar-mask PEFT repair for Depth Anything V2 on DA-2K.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/ubuntu/checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target", choices=["transformer", "all-linear"], default="transformer")
    parser.add_argument("--layer-indices", default="")
    parser.add_argument(
        "--scene-type",
        default="",
        choices=["", "indoor", "outdoor", "non_real", "transparent_reflective", "adverse_style", "aerial", "underwater", "object"],
    )
    parser.add_argument("--train-images", type=int, default=64)
    parser.add_argument("--eval-skip-images", type=int, default=64)
    parser.add_argument("--eval-images", type=int, default=128)
    parser.add_argument("--allow-train-eval-overlap", action="store_true")
    parser.add_argument("--score-images", type=int, default=32)
    parser.add_argument("--score-cache", type=Path)
    parser.add_argument("--save-score-cache", action="store_true")
    parser.add_argument("--prune-score", choices=SCALAR_SCORE_NAMES, default="taylor2_abs")
    parser.add_argument("--hybrid-alpha", type=float, default=1.0)
    parser.add_argument("--loss-tau", type=float, default=1.0)
    parser.add_argument("--budget-values", type=int, required=True)
    parser.add_argument("--peft-method", choices=PEFT_METHODS, default="lora")
    parser.add_argument("--lora-placement", default="masked")
    parser.add_argument("--lora-window", type=int, default=1)
    parser.add_argument("--lora-module-set", choices=["all", "attn", "mlp"], default="all")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-teacher-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-label-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-tau", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=12)
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ScalarMaskLoRARepairConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        target=args.target,
        layer_indices=parse_int_tuple(args.layer_indices),
        scene_type=args.scene_type,
        train_images=args.train_images,
        eval_skip_images=args.eval_skip_images,
        eval_images=args.eval_images,
        allow_train_eval_overlap=args.allow_train_eval_overlap,
        score_images=args.score_images,
        score_cache=args.score_cache,
        save_score_cache=args.save_score_cache,
        prune_score=args.prune_score,
        hybrid_alpha=args.hybrid_alpha,
        loss_tau=args.loss_tau,
        budget_values=args.budget_values,
        peft_method=args.peft_method,
        lora_placement=args.lora_placement,
        lora_window=args.lora_window,
        lora_module_set=args.lora_module_set,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pairwise_weight=args.pairwise_weight,
        pairwise_teacher_weight=args.pairwise_teacher_weight,
        pairwise_label_weight=args.pairwise_label_weight,
        pairwise_tau=args.pairwise_tau,
        seed=args.seed,
        log_every=args.log_every,
        save_checkpoint=args.save_checkpoint,
    )
    run(config)


if __name__ == "__main__":
    main()
