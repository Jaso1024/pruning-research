import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from speculative_prefill.vllm_benchmarks.sweep_utils import parse_int_csv


def _enable_mode(mode: str, config: str, spec_model: str) -> None:
    os.environ.setdefault("VLLM_USE_V1", "0")
    if mode == "embedding_norm":
        from speculative_prefill import enable_embedding_norm_prefill
        enable_embedding_norm_prefill(spec_config_path=config)
    elif mode == "spec_prefill":
        from speculative_prefill import enable_prefill_spec
        enable_prefill_spec(spec_model=spec_model, spec_config_path=config)
    elif mode != "baseline":
        raise ValueError(f"Unsupported mode: {mode}")


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_shape(llm, sampling_params, input_len: int, batch_size: int,
               warmup_iters: int, iters: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    dummy_prompt_token_ids = rng.integers(
        10000,
        size=(batch_size, input_len),
        dtype=np.int64,
    )
    prompts = [
        {"prompt_token_ids": prompt}
        for prompt in dummy_prompt_token_ids.tolist()
    ]

    for _ in range(warmup_iters):
        llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

    latencies = []
    for _ in range(iters):
        _sync_cuda()
        start = time.perf_counter()
        llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        _sync_cuda()
        latencies.append(time.perf_counter() - start)

    latencies_np = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies_np, percentages)
    return {
        "input_len": input_len,
        "output_len": sampling_params.max_tokens,
        "batch_size": batch_size,
        "avg_latency": float(np.mean(latencies_np)),
        "latencies": latencies,
        "percentiles": {
            str(percentage): float(percentile)
            for percentage, percentile in zip(percentages, percentiles)
        },
    }


def main(args: argparse.Namespace) -> dict:
    input_lens = parse_int_csv(args.input_lens)
    batch_sizes = parse_int_csv(args.batch_sizes)
    max_model_len = args.max_model_len or max(input_lens) + args.output_len + 16

    _enable_mode(args.mode, args.config, args.spec_model)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        enforce_eager=True,
        enable_chunked_prefill=False,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    results = []
    for batch_size in batch_sizes:
        for input_len in input_lens:
            print(
                f"Running mode={args.mode} input_len={input_len} "
                f"batch_size={batch_size}",
                flush=True,
            )
            result = _run_shape(
                llm,
                sampling_params,
                input_len=input_len,
                batch_size=batch_size,
                warmup_iters=args.warmup_iters,
                iters=args.iters,
                seed=args.seed + batch_size * 100000 + input_len,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "mode": args.mode,
        "model": args.model,
        "spec_model": args.spec_model if args.mode == "spec_prefill" else "",
        "config": args.config if args.mode != "baseline" else "",
        "input_lens": input_lens,
        "batch_sizes": batch_sizes,
        "output_len": args.output_len,
        "warmup_iters": args.warmup_iters,
        "iters": args.iters,
        "max_model_len": max_model_len,
        "results": results,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["baseline", "embedding_norm", "spec_prefill"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--spec-model", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--input-lens", required=True)
    parser.add_argument("--batch-sizes", required=True)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    main(parser.parse_args())
