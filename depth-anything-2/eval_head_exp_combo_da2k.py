import argparse
import json
import time
import types
from collections import defaultdict
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def load_model(encoder: str, checkpoint: Path, device: torch.device) -> DepthAnythingV2:
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def scene_from_path(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 2 and parts[0] == "images":
        return parts[1]
    return "unknown"


def collect_annotations(dataset_root: Path, max_images: int, scene_type: str) -> list[tuple[str, list[dict]]]:
    annotations = json.loads((dataset_root / "annotations.json").read_text())
    selected = [
        (image_path, pairs)
        for image_path, pairs in annotations.items()
        if not scene_type or scene_from_path(image_path) == scene_type
    ]
    return selected[:max_images] if max_images > 0 else selected


def empty_counts() -> dict[str, int]:
    return {"pairs": 0, "smaller_correct": 0, "larger_correct": 0, "ties": 0}


def add_pair(counts: dict[str, int], d1: float, d2: float) -> None:
    counts["pairs"] += 1
    if d1 < d2:
        counts["smaller_correct"] += 1
    elif d1 > d2:
        counts["larger_correct"] += 1
    else:
        counts["ties"] += 1


def finalize_counts(counts: dict[str, int]) -> dict[str, float | int | str]:
    pairs = max(counts["pairs"], 1)
    smaller = counts["smaller_correct"] / pairs
    larger = counts["larger_correct"] / pairs
    return {
        **counts,
        "smaller_is_closer_accuracy": smaller,
        "larger_is_closer_accuracy": larger,
        "best_direction": "smaller" if smaller >= larger else "larger",
        "best_accuracy": max(smaller, larger),
        "tie_fraction": counts["ties"] / pairs,
    }


def point_value(depth: torch.Tensor, point: list[int]) -> float:
    row = max(0, min(int(point[0]), depth.shape[0] - 1))
    col = max(0, min(int(point[1]), depth.shape[1] - 1))
    return float(depth[row, col].item())


class HeadApproximation:
    def __init__(self, coefficient_json: Path, source: str) -> None:
        self.data = json.loads(coefficient_json.read_text())
        self.source = source
        self.table = self._load_table()

    def _load_table(self) -> dict[tuple[int, int], dict]:
        table = {}
        source_result = self.data["results"][self.source]
        for block in source_result["blocks"]:
            block_index = int(block["block"])
            for head in block["heads"]:
                head_index = int(head["head"])
                if "coeffs" in head:
                    table[(block_index, head_index)] = {
                        "betas": [float(value) for value in head["betas"]],
                        "coeffs": [float(value) for value in head["coeffs"]],
                        "fit_explained": float(head["explained_energy_vs_zero"]),
                    }
                else:
                    table[(block_index, head_index)] = {
                        "betas": [float(head["beta"])],
                        "coeffs": [float(head["alpha"])],
                        "fit_explained": float(head["explained_energy_vs_zero"]),
                    }
        return table

    def keys(self) -> list[tuple[int, int]]:
        return sorted(self.table)

    def fit_explained(self, key: tuple[int, int]) -> float:
        return self.table[key]["fit_explained"]

    def apply(self, source: torch.Tensor, key: tuple[int, int]) -> torch.Tensor:
        spec = self.table[key]
        output = torch.zeros_like(source)
        for beta, coeff in zip(spec["betas"], spec["coeffs"]):
            ema = torch.empty_like(source)
            ema[:, 0, :] = source[:, 0, :]
            beta_value = float(beta)
            for index in range(1, source.shape[1]):
                ema[:, index, :] = source[:, index, :] + beta_value * ema[:, index - 1, :]
            output = output + float(coeff) * ema
        return output


class AttentionPatcher:
    def __init__(self, model: DepthAnythingV2, approximation: HeadApproximation) -> None:
        self.model = model
        self.approximation = approximation
        self.selected: set[tuple[int, int]] = set()
        self.original_forwards = []

    def install(self) -> None:
        for block_index, block in enumerate(self.model.pretrained.blocks):
            attn = block.attn
            self.original_forwards.append((attn, attn.forward))

            def patched_forward(module, x, idx=block_index):
                batch, tokens, channels = x.shape
                heads = module.num_heads
                head_dim = channels // heads
                qkv = module.qkv(x).reshape(batch, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
                q = qkv[0] * module.scale
                k = qkv[1]
                v = qkv[2]
                attn_weights = (q @ k.transpose(-2, -1)).softmax(dim=-1)
                head_out = attn_weights @ v
                for head_index in range(heads):
                    key = (idx, head_index)
                    if key in self.selected:
                        head_out[:, head_index, :, :] = self.approximation.apply(v[:, head_index, :, :], key)
                out = head_out.transpose(1, 2).reshape(batch, tokens, channels)
                out = module.proj(out)
                out = module.proj_drop(out)
                return out

            attn.forward = types.MethodType(patched_forward, attn)

    def set_selected(self, selected: set[tuple[int, int]]) -> None:
        self.selected = set(selected)

    def restore(self) -> None:
        for attn, forward in self.original_forwards:
            attn.forward = forward


@torch.no_grad()
def infer_depth(model: DepthAnythingV2, image, input_size: int, device: torch.device) -> torch.Tensor:
    tensor, (height, width) = model.image2tensor(image, input_size)
    tensor = tensor.to(device)
    depth = model(tensor)
    depth = F.interpolate(depth[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
    return depth.detach().float().cpu()


@torch.no_grad()
def evaluate(
    model: DepthAnythingV2,
    dataset_root: Path,
    selected_annotations: list[tuple[str, list[dict]]],
    input_size: int,
    device: torch.device,
    log_every: int,
    label: str,
) -> dict:
    total = empty_counts()
    by_scene = defaultdict(empty_counts)
    missing_images = []
    started = time.time()
    for index, (relative_path, pairs) in enumerate(selected_annotations, start=1):
        image_path = dataset_root / relative_path
        image = cv2.imread(str(image_path))
        if image is None:
            missing_images.append(str(image_path))
            continue
        depth = infer_depth(model, image, input_size, device)
        scene = scene_from_path(relative_path)
        for pair in pairs:
            if pair.get("closer_point") != "point1":
                raise ValueError(f"unsupported closer_point in {relative_path}: {pair}")
            d1 = point_value(depth, pair["point1"])
            d2 = point_value(depth, pair["point2"])
            add_pair(total, d1, d2)
            add_pair(by_scene[scene], d1, d2)
        if log_every > 0 and (index % log_every == 0 or index == len(selected_annotations)):
            print(f"{label}: evaluated {index}/{len(selected_annotations)} images", flush=True)
    return {
        "label": label,
        "overall": finalize_counts(total),
        "by_scene": {scene: finalize_counts(counts) for scene, counts in sorted(by_scene.items())},
        "missing_images": missing_images,
        "elapsed_seconds": time.time() - started,
    }


def with_delta(result: dict, baseline_accuracy: float) -> dict:
    result["accuracy_delta_vs_baseline"] = result["overall"]["best_accuracy"] - baseline_accuracy
    return result


def save_result(path: Path, result: dict) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")


def evaluate_single_heads(args, model, patcher, approximation, annotations, device, baseline_accuracy) -> list[dict]:
    outputs = []
    for index, key in enumerate(approximation.keys(), start=1):
        patcher.set_selected({key})
        result = evaluate(
            model,
            args.dataset_root,
            annotations,
            args.input_size,
            device,
            0,
            f"single_head_{key[0]}_{key[1]}",
        )
        result.update({"block": key[0], "head": key[1], "fit_explained": approximation.fit_explained(key)})
        with_delta(result, baseline_accuracy)
        outputs.append(result)
        if index % args.progress_every == 0 or index == len(approximation.keys()):
            print(f"single-head eval {index}/{len(approximation.keys())}", flush=True)
    return outputs


def evaluate_layer_replacements(args, model, patcher, approximation, annotations, device, baseline_accuracy) -> list[dict]:
    blocks = sorted({block for block, _head in approximation.keys()})
    outputs = []
    for block in blocks:
        selected = {key for key in approximation.keys() if key[0] == block}
        patcher.set_selected(selected)
        result = evaluate(model, args.dataset_root, annotations, args.input_size, device, 0, f"layer_{block}")
        result.update({"block": block, "heads_replaced": len(selected)})
        with_delta(result, baseline_accuracy)
        outputs.append(result)
        print(f"layer eval {block + 1}/{len(blocks)}", flush=True)
    return outputs


def cumulative_eval(args, model, patcher, approximation, annotations, device, baseline_accuracy, ordered_keys, label) -> list[dict]:
    selected = set()
    outputs = []
    for step, key in enumerate(ordered_keys, start=1):
        selected.add(key)
        patcher.set_selected(selected)
        result = evaluate(model, args.dataset_root, annotations, args.input_size, device, 0, f"{label}_{step}")
        result.update({"step": step, "added_block": key[0], "added_head": key[1], "heads_replaced": len(selected)})
        with_delta(result, baseline_accuracy)
        outputs.append(result)
        print(f"{label} cumulative eval {step}/{len(ordered_keys)}", flush=True)
        if args.stop_accuracy > 0 and result["overall"]["best_accuracy"] < args.stop_accuracy:
            break
    return outputs


def cumulative_layer_eval(args, model, patcher, approximation, annotations, device, baseline_accuracy, layer_results) -> list[dict]:
    ordered_blocks = [result["block"] for result in sorted(layer_results, key=lambda item: item["accuracy_delta_vs_baseline"], reverse=True)]
    selected = set()
    outputs = []
    for step, block in enumerate(ordered_blocks, start=1):
        selected.update({key for key in approximation.keys() if key[0] == block})
        patcher.set_selected(selected)
        result = evaluate(model, args.dataset_root, annotations, args.input_size, device, 0, f"layer_cumulative_{step}")
        result.update({"step": step, "added_block": block, "heads_replaced": len(selected)})
        with_delta(result, baseline_accuracy)
        outputs.append(result)
        print(f"layer cumulative eval {step}/{len(ordered_blocks)}", flush=True)
        if args.stop_accuracy > 0 and result["overall"]["best_accuracy"] < args.stop_accuracy:
            break
    return outputs


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    device = resolve_device(args.device)
    model = load_model(args.encoder, args.checkpoint, device)
    annotations = collect_annotations(args.dataset_root, args.max_images, args.scene_type)
    approximation = HeadApproximation(args.coefficients, args.source)
    patcher = AttentionPatcher(model, approximation)
    patcher.install()
    result = {
        "metadata": {
            "encoder": args.encoder,
            "checkpoint": str(args.checkpoint),
            "dataset_root": str(args.dataset_root),
            "coefficients": str(args.coefficients),
            "source": args.source,
            "device": str(device),
            "input_size": args.input_size,
            "max_images": args.max_images,
            "images": len(annotations),
            "scene_type": args.scene_type,
            "elapsed_seconds": None,
            "complete": False,
            "note": "Selected pre-proj attention heads are replaced by fitted EMA combinations of v_h; heads are then concatenated and passed through the original attention output projection so the rest of the model is unchanged.",
        }
    }
    try:
        patcher.set_selected(set())
        baseline = evaluate(model, args.dataset_root, annotations, args.input_size, device, args.log_every, "baseline")
        result["baseline"] = baseline
        save_result(args.output_json, result)
        baseline_accuracy = float(baseline["overall"]["best_accuracy"])
        single_heads = evaluate_single_heads(args, model, patcher, approximation, annotations, device, baseline_accuracy)
        result["single_heads"] = single_heads
        save_result(args.output_json, result)
        layers = evaluate_layer_replacements(args, model, patcher, approximation, annotations, device, baseline_accuracy)
        result["single_layers"] = layers
        save_result(args.output_json, result)
        by_least_damage = [tuple(item[key] for key in ("block", "head")) for item in sorted(single_heads, key=lambda item: item["accuracy_delta_vs_baseline"], reverse=True)]
        cumulative_least_damage = cumulative_eval(
            args, model, patcher, approximation, annotations, device, baseline_accuracy, by_least_damage, "head_cumulative_least_damage"
        )
        result["head_cumulative_least_damage"] = cumulative_least_damage
        save_result(args.output_json, result)
        if args.run_best_fit_cumulative:
            by_best_fit = sorted(approximation.keys(), key=lambda key: approximation.fit_explained(key), reverse=True)
            cumulative_best_fit = cumulative_eval(
                args, model, patcher, approximation, annotations, device, baseline_accuracy, by_best_fit, "head_cumulative_best_fit"
            )
        else:
            cumulative_best_fit = []
        result["head_cumulative_best_fit"] = cumulative_best_fit
        save_result(args.output_json, result)
        if args.run_layer_cumulative:
            cumulative_layers = cumulative_layer_eval(args, model, patcher, approximation, annotations, device, baseline_accuracy, layers)
        else:
            cumulative_layers = []
        result["layer_cumulative_least_damage"] = cumulative_layers
        result["metadata"]["complete"] = True
        result["metadata"]["elapsed_seconds"] = time.time() - started
        save_result(args.output_json, result)
    finally:
        patcher.restore()

    return result


def print_summary(result: dict) -> None:
    baseline = result["baseline"]["overall"]["best_accuracy"]
    print(f"baseline best_accuracy={baseline:.6f}")
    best_single = max(result["single_heads"], key=lambda item: item["overall"]["best_accuracy"])
    worst_single = min(result["single_heads"], key=lambda item: item["overall"]["best_accuracy"])
    print(
        "single_head best "
        f"block={best_single['block']} head={best_single['head']} "
        f"acc={best_single['overall']['best_accuracy']:.6f} "
        f"delta={best_single['accuracy_delta_vs_baseline']:.6f}"
    )
    print(
        "single_head worst "
        f"block={worst_single['block']} head={worst_single['head']} "
        f"acc={worst_single['overall']['best_accuracy']:.6f} "
        f"delta={worst_single['accuracy_delta_vs_baseline']:.6f}"
    )
    best_layer = max(result["single_layers"], key=lambda item: item["overall"]["best_accuracy"])
    worst_layer = min(result["single_layers"], key=lambda item: item["overall"]["best_accuracy"])
    print(f"single_layer best block={best_layer['block']} acc={best_layer['overall']['best_accuracy']:.6f} delta={best_layer['accuracy_delta_vs_baseline']:.6f}")
    print(f"single_layer worst block={worst_layer['block']} acc={worst_layer['overall']['best_accuracy']:.6f} delta={worst_layer['accuracy_delta_vs_baseline']:.6f}")
    for key in ["head_cumulative_least_damage", "head_cumulative_best_fit", "layer_cumulative_least_damage"]:
        series = result.get(key, [])
        if series:
            last = series[-1]
            print(f"{key} final step={last['step']} heads={last['heads_replaced']} acc={last['overall']['best_accuracy']:.6f} delta={last['accuracy_delta_vs_baseline']:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DA-2K accuracy after replacing selected DA-V2 attention heads with EMA head approximations.")
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--source", default="v")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=25)
    parser.add_argument("--scene-type", default="")
    parser.add_argument("--stop-accuracy", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=12)
    parser.add_argument("--run-best-fit-cumulative", action="store_true")
    parser.add_argument("--run-layer-cumulative", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/head_exp_combo_accuracy_vits.json"))
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    print_summary(run(parsed))
