import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2
from ema_filter import causal_ema_filter


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


def collect_da2k_images(dataset_root: Path, max_images: int, scene_type: str) -> list[Path]:
    annotations = json.loads((dataset_root / "annotations.json").read_text())
    paths = []
    for relative_path in annotations.keys():
        parts = Path(relative_path).parts
        scene = parts[1] if len(parts) >= 2 and parts[0] == "images" else ""
        if scene_type and scene != scene_type:
            continue
        image_path = dataset_root / relative_path
        if image_path.exists():
            paths.append(image_path)
        if max_images > 0 and len(paths) >= max_images:
            break
    return paths


def build_betas(args: argparse.Namespace) -> np.ndarray:
    grid = np.linspace(args.beta_min, args.beta_max, args.beta_values, dtype=np.float64)
    extras = np.array([float(value) for value in args.extra_betas.split(",") if value.strip()], dtype=np.float64)
    if extras.size:
        grid = np.concatenate([grid, extras])
    grid = grid[(grid >= args.beta_min) & (grid <= args.beta_max)]
    return np.unique(np.round(grid, 10))


def token_slice(array: np.ndarray, mode: str) -> np.ndarray:
    if mode == "all":
        return array
    if mode == "patch":
        return array[:, :, 1:, :]
    raise ValueError(f"unknown token slice: {mode}")


class HeadFitStats:
    def __init__(self, betas: np.ndarray, heads: int, alpha_mode: str) -> None:
        self.betas = betas
        self.alpha_mode = alpha_mode
        self.numerator = np.zeros((len(betas), heads), dtype=np.float64)
        self.denominator = np.zeros((len(betas), heads), dtype=np.float64)
        self.target_ss = np.zeros(heads, dtype=np.float64)
        self.count = np.zeros(heads, dtype=np.int64)

    def update(self, source: np.ndarray, target: np.ndarray) -> None:
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        self.target_ss += np.einsum("bhnd,bhnd->h", target, target, optimize=True)
        self.count += np.prod(target.shape[0:1] + target.shape[2:4])
        for index, beta in enumerate(self.betas):
            ema = causal_ema_filter(source, float(beta), axis=2)
            self.numerator[index] += np.einsum("bhnd,bhnd->h", ema, target, optimize=True)
            self.denominator[index] += np.einsum("bhnd,bhnd->h", ema, ema, optimize=True)

    def best_by_head(self) -> list[dict]:
        results = []
        for head in range(len(self.target_ss)):
            target_ss = float(self.target_ss[head])
            count = int(self.count[head])
            target_mse0 = target_ss / max(count, 1)
            best_result = None
            for beta_index, beta in enumerate(self.betas):
                numerator = float(self.numerator[beta_index, head])
                denominator = float(self.denominator[beta_index, head])
                if denominator <= 0.0:
                    alpha = 0.0
                    sse = target_ss
                else:
                    alpha = numerator / denominator
                    if self.alpha_mode == "nonnegative":
                        alpha = max(0.0, alpha)
                    sse = target_ss - 2.0 * alpha * numerator + alpha * alpha * denominator
                mse = max(0.0, sse / max(count, 1))
                rel_mse = mse / target_mse0 if target_mse0 > 0.0 else 0.0
                cosine = 0.0
                if denominator > 0.0 and target_ss > 0.0:
                    cosine = numerator / float(np.sqrt(denominator * target_ss))
                result = {
                    "head": head,
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "mse": float(mse),
                    "relative_mse_vs_zero": float(rel_mse),
                    "explained_energy_vs_zero": float(1.0 - rel_mse),
                    "cosine": float(cosine),
                    "target_rms": float(np.sqrt(target_mse0)),
                    "count": count,
                }
                if best_result is None or result["mse"] < best_result["mse"]:
                    best_result = result
            results.append(best_result or {})
        return results


def summarize_heads(heads: list[dict]) -> dict:
    if not heads:
        return {}
    explained = np.array([head["explained_energy_vs_zero"] for head in heads], dtype=np.float64)
    rel_mse = np.array([head["relative_mse_vs_zero"] for head in heads], dtype=np.float64)
    return {
        "mean_relative_mse_vs_zero": float(rel_mse.mean()),
        "mean_explained_energy_vs_zero": float(explained.mean()),
        "median_explained_energy_vs_zero": float(np.median(explained)),
        "max_explained_energy_vs_zero": float(explained.max()),
        "min_explained_energy_vs_zero": float(explained.min()),
    }


def compute_head_capture(module, x: torch.Tensor) -> dict[str, np.ndarray]:
    batch, tokens, channels = x.shape
    heads = module.num_heads
    head_dim = channels // heads
    qkv = module.qkv(x).reshape(batch, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    q = qkv[0] * module.scale
    k = qkv[1]
    v = qkv[2]
    attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
    head_out = attn @ v
    x_head = x.reshape(batch, tokens, heads, head_dim).permute(0, 2, 1, 3)
    return {
        "v": v.detach().float().cpu().numpy(),
        "x_head": x_head.detach().float().cpu().numpy(),
        "head_attn_out": head_out.detach().float().cpu().numpy(),
    }


def register_hooks(model: DepthAnythingV2, captures: dict[int, dict[str, np.ndarray]]) -> list:
    handles = []
    for block_index, block in enumerate(model.pretrained.blocks):
        def attn_hook(module, inputs, _output, idx=block_index):
            captures[idx] = compute_head_capture(module, inputs[0].detach())

        handles.append(block.attn.register_forward_hook(attn_hook))
    return handles


@torch.no_grad()
def capture_one(model: DepthAnythingV2, image_path: Path, input_size: int, device: torch.device, captures: dict) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image: {image_path}")
    captures.clear()
    tensor, _ = model.image2tensor(image, input_size)
    model(tensor.to(device))


def fit(args: argparse.Namespace) -> dict:
    started = time.time()
    device = resolve_device(args.device)
    model = load_model(args.encoder, args.checkpoint, device)
    betas = build_betas(args)
    image_paths = collect_da2k_images(args.dataset_root, args.max_images, args.scene_type)
    if not image_paths:
        raise ValueError("no calibration images found")

    blocks = list(model.pretrained.blocks)
    heads = blocks[0].attn.num_heads
    source_variants = [value.strip() for value in args.sources.split(",") if value.strip()]
    per_block_stats = {
        (source_name, block_index): HeadFitStats(betas, heads, args.alpha_mode)
        for source_name in source_variants
        for block_index in range(len(blocks))
    }
    global_stats = {source_name: HeadFitStats(betas, heads, args.alpha_mode) for source_name in source_variants}

    captures: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    handles = register_hooks(model, captures)
    try:
        for image_index, image_path in enumerate(image_paths, start=1):
            capture_one(model, image_path, args.input_size, device, captures)
            for block_index in range(len(blocks)):
                target = token_slice(captures[block_index]["head_attn_out"], args.tokens)
                for source_name in source_variants:
                    source = token_slice(captures[block_index][source_name], args.tokens)
                    per_block_stats[(source_name, block_index)].update(source, target)
                    global_stats[source_name].update(source, target)
            if image_index % args.log_every == 0 or image_index == len(image_paths):
                print(f"processed {image_index}/{len(image_paths)} calibration images", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    results = {}
    for source_name in source_variants:
        blocks_result = []
        all_block_heads = []
        for block_index in range(len(blocks)):
            head_results = per_block_stats[(source_name, block_index)].best_by_head()
            all_block_heads.extend(head_results)
            blocks_result.append(
                {
                    "block": block_index,
                    "summary": summarize_heads(head_results),
                    "heads": head_results,
                }
            )
        global_head_results = global_stats[source_name].best_by_head()
        results[source_name] = {
            "global_by_head": {
                "summary": summarize_heads(global_head_results),
                "heads": global_head_results,
            },
            "all_block_heads_summary": summarize_heads(all_block_heads),
            "blocks": blocks_result,
        }

    result = {
        "metadata": {
            "encoder": args.encoder,
            "checkpoint": str(args.checkpoint),
            "device": str(device),
            "dataset_root": str(args.dataset_root),
            "images": len(image_paths),
            "input_size": args.input_size,
            "heads": heads,
            "target": "raw per-head attn @ v before output projection",
            "sources": source_variants,
            "tokens": args.tokens,
            "alpha_mode": args.alpha_mode,
            "beta_min": args.beta_min,
            "beta_max": args.beta_max,
            "beta_values": len(betas),
            "elapsed_seconds": time.time() - started,
            "note": "For each fixed beta and each head, alpha is the closed-form least-squares optimum for head_attn_out ~= alpha * causal_ema_beta(source). No attention output projection is applied.",
        },
        "results": results,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    return result


def print_summary(result: dict) -> None:
    print(json.dumps(result["metadata"], indent=2))
    for source_name, source_result in result["results"].items():
        print(f"source={source_name} global_by_head {source_result['global_by_head']['summary']}")
        print(f"source={source_name} all_block_heads {source_result['all_block_heads_summary']}")
        best = None
        for block in source_result["blocks"]:
            for head in block["heads"]:
                item = {"block": block["block"], **head}
                if best is None or item["explained_energy_vs_zero"] > best["explained_energy_vs_zero"]:
                    best = item
        if best:
            print(
                f"source={source_name} best block={best['block']:02d} head={best['head']:02d} "
                f"alpha={best['alpha']:.6g} beta={best['beta']:.6g} "
                f"rel_mse={best['relative_mse_vs_zero']:.6g} "
                f"explained={best['explained_energy_vs_zero']:.6g} cos={best['cosine']:.6g}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit per-head causal EMA approximations to Depth Anything V2 attention heads.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--scene-type", default="")
    parser.add_argument("--sources", default="v")
    parser.add_argument("--tokens", choices=["all", "patch"], default="patch")
    parser.add_argument("--alpha-mode", choices=["unconstrained", "nonnegative"], default="unconstrained")
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--beta-max", type=float, default=0.99)
    parser.add_argument("--beta-values", type=int, default=50)
    parser.add_argument("--extra-betas", default="0,0.1,0.5,0.9,0.95,0.99")
    parser.add_argument("--log-every", type=int, default=4)
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/head_ema_attention_fit_vits.json"))
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    print_summary(fit(parsed))
