from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from eval_da2k import MODEL_CONFIGS, resolve_device
from eval_gelu_relu_compensation_da2k import selected_annotations, transformer_mlp_names, write_summary
from eval_moefication_da2k import load_moefication_base, parse_activation_from_summary


@dataclass(frozen=True)
class SparseFc2BenchConfig:
    dataset_root: Path
    checkpoint: Path
    output_dir: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"
    summary_json: Path | None = None
    variant_key: str = ""
    state_dict: Path | None = None
    max_images: int = 1
    max_tokens: int = 1024
    max_layers: int = 12
    warmup: int = 5
    iters: int = 20
    channel_fractions: tuple[float, ...] = (0.25, 0.5)
    block_sizes: tuple[int, ...] = (4, 8)
    seed: int = 89

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.summary_json is not None:
            object.__setattr__(self, "summary_json", Path(self.summary_json))
        if self.state_dict is not None:
            object.__setattr__(self, "state_dict", Path(self.state_dict))
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"unknown encoder: {self.encoder}")
        if self.max_images <= 0:
            raise ValueError("max_images must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.max_layers <= 0:
            raise ValueError("max_layers must be positive")
        if self.warmup < 0 or self.iters <= 0:
            raise ValueError("warmup must be non-negative and iters must be positive")
        for fraction in self.channel_fractions:
            if not 0.0 < fraction <= 1.0:
                raise ValueError("channel fractions must be in (0, 1]")
        for block_size in self.block_sizes:
            if block_size <= 0:
                raise ValueError("block sizes must be positive")


def collect_layer_payloads(
    *,
    model: torch.nn.Module,
    mlp_names: list[str],
    dataset_root: Path,
    input_size: int,
    device: torch.device,
    max_images: int,
    max_tokens: int,
    max_layers: int,
    seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    items = selected_annotations(dataset_root, max_images=max_images, scene_type="", max_pairs=0)
    records: dict[str, list[torch.Tensor]] = {name: [] for name in mlp_names[:max_layers]}
    handles = []
    for name in mlp_names[:max_layers]:
        module = model.get_submodule(name)

        def make_hook(module_name: str):
            def hook(module, inputs, _output) -> None:
                with torch.no_grad():
                    hidden = module.act(module.fc1(inputs[0])).detach().flatten(0, -2)
                    records[module_name].append(hidden.float().cpu())

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    try:
        with torch.inference_mode():
            for relative_path, _pairs in tqdm(items, desc="collect sparse fc2 activations", unit="image"):
                image = cv2.imread(str(dataset_root / relative_path))
                if image is None:
                    continue
                tensor, _shape = model.image2tensor(image, input_size)
                _ = model(tensor.to(device=device))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    payloads: dict[str, dict[str, torch.Tensor]] = {}
    for name in mlp_names[:max_layers]:
        hidden = torch.cat(records[name], dim=0)
        if hidden.shape[0] > max_tokens:
            indices = torch.randperm(hidden.shape[0], generator=generator)[:max_tokens]
            hidden = hidden.index_select(0, indices)
        module = model.get_submodule(name)
        payloads[name] = {
            "hidden": hidden.contiguous(),
            "weight": module.fc2.weight.detach().float().cpu().contiguous(),
            "bias": (
                module.fc2.bias.detach().float().cpu().contiguous()
                if module.fc2.bias is not None
                else torch.zeros(module.fc2.out_features)
            ),
        }
    return payloads


def cuda_time_ms(fn, *, warmup: int, iters: int) -> tuple[float, Any]:
    out = None
    for _ in range(warmup):
        out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iters, out


def select_channel_indices(hidden: torch.Tensor, weight: torch.Tensor, fraction: float) -> torch.Tensor:
    hidden_size = hidden.shape[1]
    k = max(1, min(hidden_size, round(hidden_size * fraction)))
    channel_weight = weight.float().norm(dim=0).clamp_min(1e-8)
    values = hidden.float().abs() * channel_weight.unsqueeze(0)
    return values.topk(k, dim=1).indices


def dense_masked_fc2(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(hidden, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return F.linear(hidden * mask.to(dtype=hidden.dtype), weight, bias)


def gathered_channel_fc2(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    token_count, k = indices.shape
    out_features = weight.shape[0]
    h_selected = hidden.gather(1, indices)
    columns = weight.t().index_select(0, indices.reshape(-1)).reshape(token_count, k, out_features)
    return (columns * h_selected.unsqueeze(-1)).sum(dim=1) + bias


def expand_indices_to_blocks(indices: torch.Tensor, hidden_size: int, block_size: int) -> torch.Tensor:
    token_count = indices.shape[0]
    block_count = (hidden_size + block_size - 1) // block_size
    block_mask = torch.zeros((token_count, block_count), device=indices.device, dtype=torch.bool)
    block_mask.scatter_(1, indices // block_size, True)
    channel_ids = torch.arange(hidden_size, device=indices.device)
    return block_mask[:, channel_ids // block_size]


def block_loop_fc2(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, float]:
    hidden_size = hidden.shape[1]
    channel_mask = expand_indices_to_blocks(indices, hidden_size, block_size)
    out = bias.expand(hidden.shape[0], -1).clone()
    block_count = (hidden_size + block_size - 1) // block_size
    for block in range(block_count):
        start = block * block_size
        end = min(start + block_size, hidden_size)
        rows = channel_mask[:, start].nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        out.index_add_(0, rows, F.linear(hidden.index_select(0, rows)[:, start:end], weight[:, start:end], None))
    selected_fraction = channel_mask.float().mean().item()
    return out, selected_fraction


def relative_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return ((reference - candidate).float().norm() / reference.float().norm().clamp_min(1e-8)).item()


def bench_payload(
    payload: dict[str, torch.Tensor],
    *,
    device: torch.device,
    fractions: tuple[float, ...],
    block_sizes: tuple[int, ...],
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    hidden = payload["hidden"].to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    weight = payload["weight"].to(device=device, dtype=hidden.dtype)
    bias = payload["bias"].to(device=device, dtype=hidden.dtype)
    dense_ms, dense = cuda_time_ms(lambda: F.linear(hidden, weight, bias), warmup=warmup, iters=iters)
    result: dict[str, Any] = {
        "tokens": int(hidden.shape[0]),
        "hidden_features": int(hidden.shape[1]),
        "out_features": int(weight.shape[0]),
        "dense_ms": dense_ms,
        "variants": {},
    }

    for fraction in fractions:
        indices = select_channel_indices(hidden, weight, fraction)
        masked_ms, masked = cuda_time_ms(
            lambda indices=indices: dense_masked_fc2(hidden, weight, bias, indices),
            warmup=warmup,
            iters=iters,
        )
        gathered_ms, gathered = cuda_time_ms(
            lambda indices=indices: gathered_channel_fc2(hidden, weight, bias, indices),
            warmup=warmup,
            iters=iters,
        )
        result["variants"][f"channel_top_{fraction:g}"] = {
            "selected_fraction": fraction,
            "dense_masked_ms": masked_ms,
            "gathered_channel_ms": gathered_ms,
            "dense_masked_vs_dense_relerr": relative_error(dense, masked),
            "gathered_vs_dense_masked_relerr": relative_error(masked, gathered),
            "dense_speedup_vs_gathered": dense_ms / gathered_ms if gathered_ms > 0 else None,
        }
        for block_size in block_sizes:
            block_fn = lambda indices=indices, block_size=block_size: block_loop_fc2(hidden, weight, bias, indices, block_size)
            block_ms, block_result = cuda_time_ms(block_fn, warmup=warmup, iters=iters)
            block_out, selected_fraction = block_result
            result["variants"][f"channel_top_{fraction:g}_block{block_size}"] = {
                "selected_fraction": selected_fraction,
                "block_loop_ms": block_ms,
                "block_loop_vs_dense_relerr": relative_error(dense, block_out),
                "block_loop_vs_dense_masked_relerr": relative_error(masked, block_out),
                "dense_speedup_vs_block_loop": dense_ms / block_ms if block_ms > 0 else None,
            }
    return result


def run(config: SparseFc2BenchConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the exact repaired/activation-swapped loading path from the
    # MoEification evaluator so benchmarked hidden activations match DA2K runs.
    if config.summary_json is not None:
        _activation, _stage2, _stage2_shift = parse_activation_from_summary(config.summary_json, config.variant_key)
    model, load_summary = load_moefication_base(
        SimpleNamespace(
            summary_json=config.summary_json,
            variant_key=config.variant_key,
            activation="relu",
            stage2="none",
            stage2_shift=0.0,
            encoder=config.encoder,
            checkpoint=config.checkpoint,
            state_dict=config.state_dict,
        ),
        device,
    )
    mlp_names = transformer_mlp_names(model)[: config.max_layers]
    payloads = collect_layer_payloads(
        model=model,
        mlp_names=mlp_names,
        dataset_root=config.dataset_root,
        input_size=config.input_size,
        device=device,
        max_images=config.max_images,
        max_tokens=config.max_tokens,
        max_layers=config.max_layers,
        seed=config.seed,
    )

    result: dict[str, Any] = {
        "metadata": {
            **asdict(config),
            "device": str(device),
            "loaded_model": load_summary,
            "mlp_names": mlp_names,
            "note": (
                "Sparse fc2 benchmark on real post-activation MLP hidden states. "
                "dense_masked matches the accuracy evaluator's masking semantics; "
                "gathered_channel and block_loop are PyTorch execution prototypes, not optimized CUDA kernels."
            ),
        },
        "layers": {},
    }
    summary_path = config.output_dir / "summary.json"
    write_summary(summary_path, result)

    for name, payload in tqdm(payloads.items(), desc="bench sparse fc2", unit="mlp"):
        result["layers"][name] = bench_payload(
            payload,
            device=device,
            fractions=config.channel_fractions,
            block_sizes=config.block_sizes,
            warmup=config.warmup,
            iters=config.iters,
        )
        write_summary(summary_path, result)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(piece) for piece in text.split(",") if piece)


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(piece) for piece in text.split(",") if piece)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark sparse fc2 execution shapes for DA2V2 MoEification.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DA-2K/extracted/DA-2K"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/moefication_sparse_fc2_bench"))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--variant-key", default="")
    parser.add_argument("--state-dict", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-layers", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--channel-fractions", type=parse_floats, default=(0.25, 0.5))
    parser.add_argument("--block-sizes", type=parse_ints, default=(4, 8))
    parser.add_argument("--seed", type=int, default=89)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = SparseFc2BenchConfig(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        summary_json=args.summary_json,
        variant_key=args.variant_key,
        state_dict=args.state_dict,
        max_images=args.max_images,
        max_tokens=args.max_tokens,
        max_layers=args.max_layers,
        warmup=args.warmup,
        iters=args.iters,
        channel_fractions=args.channel_fractions,
        block_sizes=args.block_sizes,
        seed=args.seed,
    )
    result = run(config)
    dense_mean = sum(row["dense_ms"] for row in result["layers"].values()) / max(len(result["layers"]), 1)
    print(json.dumps({"layers": len(result["layers"]), "mean_dense_ms": dense_mean}, indent=2))


if __name__ == "__main__":
    main()
