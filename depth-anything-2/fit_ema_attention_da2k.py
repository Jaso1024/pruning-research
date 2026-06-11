import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.signal import lfilter

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


def collect_example_images(example_root: Path, max_images: int) -> list[Path]:
    paths = sorted(
        [
            *example_root.glob("*.jpg"),
            *example_root.glob("*.jpeg"),
            *example_root.glob("*.png"),
        ]
    )
    return paths[:max_images] if max_images > 0 else paths


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
        return array[:, 1:, :]
    raise ValueError(f"unknown token slice: {mode}")


class FitStats:
    def __init__(self, betas: np.ndarray, alpha_mode: str) -> None:
        self.betas = betas
        self.alpha_mode = alpha_mode
        self.numerator = np.zeros(len(betas), dtype=np.float64)
        self.denominator = np.zeros(len(betas), dtype=np.float64)
        self.target_ss = 0.0
        self.count = 0

    def update(self, source: np.ndarray, target: np.ndarray) -> None:
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        self.target_ss += float(np.einsum("bnc,bnc->", target, target, optimize=True))
        self.count += int(target.size)
        for index, beta in enumerate(self.betas):
            ema = lfilter([1.0], [1.0, -float(beta)], source, axis=1).astype(np.float32, copy=False)
            self.numerator[index] += float(np.einsum("bnc,bnc->", ema, target, optimize=True))
            self.denominator[index] += float(np.einsum("bnc,bnc->", ema, ema, optimize=True))

    def best(self) -> dict:
        if self.count == 0:
            return {}
        target_mse0 = self.target_ss / self.count
        best_result = None
        for index, beta in enumerate(self.betas):
            denom = self.denominator[index]
            if denom <= 0.0:
                alpha = 0.0
                sse = self.target_ss
            else:
                alpha = self.numerator[index] / denom
                if self.alpha_mode == "nonnegative":
                    alpha = max(0.0, alpha)
                sse = self.target_ss - 2.0 * alpha * self.numerator[index] + alpha * alpha * denom
            mse = max(0.0, sse / self.count)
            rel_mse = mse / target_mse0 if target_mse0 > 0.0 else 0.0
            cosine = 0.0
            if denom > 0.0 and self.target_ss > 0.0:
                cosine = self.numerator[index] / float(np.sqrt(denom * self.target_ss))
            result = {
                "alpha": float(alpha),
                "beta": float(beta),
                "mse": float(mse),
                "relative_mse_vs_zero": float(rel_mse),
                "explained_energy_vs_zero": float(1.0 - rel_mse),
                "cosine": float(cosine),
                "target_rms": float(np.sqrt(target_mse0)),
                "count": int(self.count),
            }
            if best_result is None or result["mse"] < best_result["mse"]:
                best_result = result
        return best_result or {}


def register_hooks(model: DepthAnythingV2, captures: dict[int, dict[str, np.ndarray]]) -> list:
    handles = []
    blocks = list(model.pretrained.blocks)

    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().float().cpu().numpy()

    for block_index, block in enumerate(blocks):
        def norm_hook(_module, inputs, output, idx=block_index):
            captures[idx]["prenorm"] = to_numpy(inputs[0])
            captures[idx]["norm1"] = to_numpy(output)

        def ls1_hook(_module, _inputs, output, idx=block_index):
            captures[idx]["post_ls1"] = to_numpy(output)

        def attn_hook(_module, _inputs, output, idx=block_index):
            captures[idx]["raw_attn"] = to_numpy(output)

        handles.append(block.norm1.register_forward_hook(norm_hook))
        handles.append(block.attn.register_forward_hook(attn_hook))
        handles.append(block.ls1.register_forward_hook(ls1_hook))
    return handles


@torch.no_grad()
def capture_one(model: DepthAnythingV2, image_path: Path, input_size: int, device: torch.device, captures: dict) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image: {image_path}")
    captures.clear()
    for index in range(len(model.pretrained.blocks)):
        captures[index] = {}
    tensor, _ = model.image2tensor(image, input_size)
    model(tensor.to(device))


def fit(args: argparse.Namespace) -> dict:
    started = time.time()
    device = resolve_device(args.device)
    model = load_model(args.encoder, args.checkpoint, device)
    betas = build_betas(args)
    if args.dataset_root:
        image_paths = collect_da2k_images(args.dataset_root, args.max_images, args.scene_type)
        image_source = str(args.dataset_root)
    else:
        image_paths = collect_example_images(args.example_root, args.max_images)
        image_source = str(args.example_root)
    if not image_paths:
        raise ValueError("no calibration images found")

    source_variants = [value.strip() for value in args.sources.split(",") if value.strip()]
    stats = {}
    global_stats = {}
    blocks = list(model.pretrained.blocks)
    for source_name in source_variants:
        global_stats[source_name] = FitStats(betas, args.alpha_mode)
        for block_index in range(len(blocks)):
            stats[(source_name, block_index)] = FitStats(betas, args.alpha_mode)

    captures: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    handles = register_hooks(model, captures)
    try:
        for image_index, image_path in enumerate(image_paths, start=1):
            capture_one(model, image_path, args.input_size, device, captures)
            for block_index in range(len(blocks)):
                target = captures[block_index][args.target]
                target = token_slice(target, args.tokens)
                for source_name in source_variants:
                    source = token_slice(captures[block_index][source_name], args.tokens)
                    stats[(source_name, block_index)].update(source, target)
                    global_stats[source_name].update(source, target)
            if image_index % args.log_every == 0 or image_index == len(image_paths):
                print(f"processed {image_index}/{len(image_paths)} calibration images", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    by_source = {}
    for source_name in source_variants:
        by_source[source_name] = {
            "global": global_stats[source_name].best(),
            "blocks": [stats[(source_name, block_index)].best() for block_index in range(len(blocks))],
        }

    result = {
        "metadata": {
            "encoder": args.encoder,
            "checkpoint": str(args.checkpoint),
            "device": str(device),
            "image_source": image_source,
            "images": len(image_paths),
            "input_size": args.input_size,
            "target": args.target,
            "sources": source_variants,
            "tokens": args.tokens,
            "alpha_mode": args.alpha_mode,
            "beta_min": args.beta_min,
            "beta_max": args.beta_max,
            "beta_values": len(betas),
            "elapsed_seconds": time.time() - started,
            "note": "For each fixed beta, alpha is the closed-form least-squares optimum for target ~= alpha * causal_ema(source).",
        },
        "results": by_source,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    return result


def print_summary(result: dict) -> None:
    print(json.dumps(result["metadata"], indent=2))
    for source_name, source_result in result["results"].items():
        global_result = source_result["global"]
        print(
            f"source={source_name} global "
            f"alpha={global_result['alpha']:.6g} beta={global_result['beta']:.6g} "
            f"rel_mse={global_result['relative_mse_vs_zero']:.6g} "
            f"explained={global_result['explained_energy_vs_zero']:.6g} "
            f"cos={global_result['cosine']:.6g}"
        )
        for index, block_result in enumerate(source_result["blocks"]):
            print(
                f"  block={index:02d} alpha={block_result['alpha']:.6g} "
                f"beta={block_result['beta']:.6g} "
                f"rel_mse={block_result['relative_mse_vs_zero']:.6g} "
                f"explained={block_result['explained_energy_vs_zero']:.6g} "
                f"cos={block_result['cosine']:.6g}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit scalar causal EMA approximations to Depth Anything V2 attention residuals.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--example-root", type=Path, default=Path("assets/examples"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--scene-type", default="")
    parser.add_argument("--target", choices=["post_ls1", "raw_attn"], default="post_ls1")
    parser.add_argument("--sources", default="norm1,prenorm")
    parser.add_argument("--tokens", choices=["all", "patch"], default="all")
    parser.add_argument("--alpha-mode", choices=["unconstrained", "nonnegative"], default="unconstrained")
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--beta-max", type=float, default=0.99)
    parser.add_argument("--beta-values", type=int, default=50)
    parser.add_argument("--extra-betas", default="0,0.1,0.5,0.9,0.95,0.99")
    parser.add_argument("--log-every", type=int, default=4)
    parser.add_argument("--output-json", type=Path, default=Path("eval_outputs/ema_attention_fit_vits.json"))
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    print_summary(fit(parsed))
