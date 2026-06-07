from __future__ import annotations

import os
from pathlib import Path

import modal


REMOTE_ROOT = "/root/layer_composition"
RESULTS_ROOT = "/results"

app = modal.App("pythia-layer-composition-distill")
volume = modal.Volume.from_name("pythia-layer-composition-results", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "datasets", "accelerate", "scikit-learn")
    .add_local_dir(
        ".",
        remote_path=REMOTE_ROOT,
        copy=True,
        ignore=[
            ".git",
            ".venv",
            "venv",
            "runs",
            ".pytest_cache",
            "**/__pycache__",
            "**/*.pyc",
        ],
    )
)

secrets = [modal.Secret.from_local_environ(["HF_TOKEN"])] if "HF_TOKEN" in os.environ else []
_SPARSE24_LAYER_STATE_WORKER_CACHE = {}


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_h100_sweep(
    learning_rates: str = "3e-4,1e-3,3e-3",
    steps: int = 100,
    batch_size: int = 512,
    seq_len: int = 256,
    layer_index: int | None = None,
    dtype: str = "bf16",
    compile_student: bool = False,
    run_name: str = "pythia70m_middle_pair",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.experiment import DistillConfig, _parse_lrs, run_sweep

    config = DistillConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        learning_rates=_parse_lrs(learning_rates),
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        layer_index=layer_index,
        dtype=dtype,
        compile_student=compile_student,
        data="wikitext",
    )
    summary = run_sweep(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_h100_merge_sweep(
    methods: str = "slerp,geom_slerp",
    t_values: str = "0,0.25,0.5,0.75,1",
    steps: int = 100,
    batch_size: int = 1024,
    seq_len: int = 1024,
    layer_index: int | None = None,
    dtype: str = "bf16",
    save_student: bool = False,
    run_name: str = "pythia70m_middle_pair_merge",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.merge import MergeEvalConfig, _parse_csv_floats, _parse_csv_strings, run_merge_sweep

    config = MergeEvalConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        methods=_parse_csv_strings(methods),
        t_values=_parse_csv_floats(t_values),
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        layer_index=layer_index,
        dtype=dtype,
        save_student=save_student,
        data="wikitext",
    )
    summary = run_merge_sweep(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_h100_attention_sweep(
    model_pairs: str,
    prompts: str,
    max_length: int = 48,
    dtype: str = "bf16",
    local_window: int = 8,
    save_tensors: bool = False,
    run_name: str = "pythia_attention_sweep",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.attention_analysis import _parse_model_pairs, _parse_prompts, run_attention_pair_sweep

    summary = run_attention_pair_sweep(
        output_dir=Path(RESULTS_ROOT) / run_name,
        model_pairs=_parse_model_pairs(model_pairs),
        prompts=_parse_prompts(prompts),
        max_length=max_length,
        dtype=dtype,
        device="cuda",
        local_window=local_window,
        save_tensors=save_tensors,
    )
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_h100_attention_combo(
    small_model: str,
    big_model: str,
    prompts: str,
    combo_method: str = "exponential",
    max_length: int = 48,
    dtype: str = "bf16",
    local_window: int = 8,
    save_tensors: bool = False,
    fit_steps: int = 300,
    fit_lr: float = 0.2,
    run_name: str = "pythia_attention_combo",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.attention_analysis import AttentionAnalysisConfig, _parse_prompts, run_attention_combo_analysis

    config = AttentionAnalysisConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        small_model=small_model,
        big_model=big_model,
        prompts=_parse_prompts(prompts),
        max_length=max_length,
        dtype=dtype,
        device="cuda",
        local_window=local_window,
        save_tensors=save_tensors,
    )
    summary = run_attention_combo_analysis(config, combo_method=combo_method, fit_steps=fit_steps, fit_lr=fit_lr)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_h100_attention_basis_sweep(
    model_names: str,
    prompts: str,
    max_length: int = 48,
    dtype: str = "bf16",
    basis_sizes: str = "",
    fit_steps: int = 400,
    fit_lr: float = 0.2,
    run_name: str = "pythia_attention_basis",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.attention_analysis import _parse_int_tuple, _parse_model_names, _parse_prompts, run_attention_basis_sweep

    summary = run_attention_basis_sweep(
        output_dir=Path(RESULTS_ROOT) / run_name,
        model_names=_parse_model_names(model_names),
        prompts=_parse_prompts(prompts),
        max_length=max_length,
        dtype=dtype,
        device="cuda",
        basis_sizes=_parse_int_tuple(basis_sizes) if basis_sizes else (),
        fit_steps=fit_steps,
        fit_lr=fit_lr,
    )
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_h100_attention_basis_ppl_eval(
    model_name: str = "EleutherAI/pythia-1.4b",
    basis_sizes: str = "2,4,8",
    nmf_iterations: int = 8,
    eval_steps: int = 8,
    batch_size: int = 4,
    seq_len: int = 128,
    dtype: str = "bf16",
    data_split: str = "test",
    ce_chunk_tokens: int = 32768,
    combine_mode: str = "linear",
    basis_quantization_bits: int = 0,
    basis_quantization_format: str = "",
    basis_quantization_target: str = "reconstructed",
    run_name: str = "pythia_attention_basis_ppl",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.attention_basis_ppl import AttentionBasisPPLConfig, _parse_basis_sizes, run_attention_basis_ppl_eval

    summary = run_attention_basis_ppl_eval(
        AttentionBasisPPLConfig(
            output_dir=Path(RESULTS_ROOT) / run_name,
            model_name=model_name,
            basis_sizes=_parse_basis_sizes(basis_sizes),
            nmf_iterations=nmf_iterations,
            eval_steps=eval_steps,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=dtype,
            data_split=data_split,
            ce_chunk_tokens=ce_chunk_tokens,
            combine_mode=combine_mode,
            basis_quantization_bits=basis_quantization_bits,
            basis_quantization_format=basis_quantization_format,
            basis_quantization_target=basis_quantization_target,
        )
    )
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
    max_containers=8,
)
def run_h100_attention_basis_candidate_batch(spec: dict) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.attention_basis_ppl import AttentionBasisLayerGroupEvalConfig, evaluate_attention_basis_layer_groups

    summary = evaluate_attention_basis_layer_groups(
        AttentionBasisLayerGroupEvalConfig(
            output_dir=Path(RESULTS_ROOT) / spec["run_name"] / spec["round_name"] / spec["shard_name"],
            model_name=spec["model_name"],
            basis_size=int(spec["basis_size"]),
            nmf_iterations=int(spec["nmf_iterations"]),
            layer_groups=tuple(tuple(int(layer) for layer in group) for group in spec["layer_groups"]),
            eval_steps=int(spec["eval_steps"]),
            batch_size=int(spec["batch_size"]),
            seq_len=int(spec["seq_len"]),
            dtype=spec["dtype"],
            data_split=spec["data_split"],
            ce_chunk_tokens=int(spec["ce_chunk_tokens"]),
            log_gpu_stats=bool(spec.get("log_gpu_stats", True)),
            combine_mode=spec.get("combine_mode", "linear"),
            basis_quantization_bits=int(spec.get("basis_quantization_bits", 0)),
            basis_quantization_format=spec.get("basis_quantization_format", ""),
            basis_quantization_target=spec.get("basis_quantization_target", "reconstructed"),
        )
    )
    volume.commit()
    return {"round_name": spec["round_name"], "shard_name": spec["shard_name"], **summary}


@app.function(
    image=image,
    timeout=10 * 60,
    volumes={RESULTS_ROOT: volume},
)
def write_h100_attention_basis_search_summary(run_name: str, summary_json: str) -> dict:
    import json
    from pathlib import Path

    output_dir = Path(RESULTS_ROOT) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_json)
    (output_dir / "search_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Attention Basis Layer Search",
        "",
        "| depth | best layers | ppl | ratio vs baseline | loss |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in summary.get("path", []):
        layers = ",".join(str(layer) for layer in row.get("layer_group", []))
        lines.append(
            f"| {row['depth']} | {layers} | {float(row['ppl']):.6f} | "
            f"{float(row['ppl_ratio_vs_baseline']):.6f} | {float(row['loss']):.6f} |"
        )
    (output_dir / "search_summary.md").write_text("\n".join(lines) + "\n")
    volume.commit()
    return {"output_dir": str(output_dir), "path_count": len(summary.get("path", []))}


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_h100_hybrid_attention_eval(
    combo_run_name: str,
    small_model: str = "EleutherAI/pythia-1.4b",
    big_model: str = "EleutherAI/pythia-2.8b",
    eval_steps: int = 8,
    batch_size: int = 8,
    seq_len: int = 256,
    dtype: str = "bf16",
    data_split: str = "test",
    ce_chunk_tokens: int = 32768,
    include_small_baseline: bool = False,
    skip_big_baseline: bool = False,
    per_layer_sweep: bool = False,
    greedy_layer_sweep: bool = False,
    greedy_max_layers: int | None = None,
    run_name: str = "pythia_hybrid_attention_eval",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.hybrid_attention import HybridAttentionEvalConfig, run_hybrid_attention_eval

    config = HybridAttentionEvalConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        combo_path=Path(RESULTS_ROOT) / combo_run_name / "exponential_combo.jsonl",
        small_model=small_model,
        big_model=big_model,
        eval_steps=eval_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        dtype=dtype,
        data_split=data_split,
        ce_chunk_tokens=ce_chunk_tokens,
        include_big_baseline=not skip_big_baseline,
        include_small_baseline=include_small_baseline,
        per_layer_sweep=per_layer_sweep,
        greedy_layer_sweep=greedy_layer_sweep,
        greedy_max_layers=greedy_max_layers,
    )
    summary = run_hybrid_attention_eval(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_h100_layer_removal_eval(
    model_name: str = "EleutherAI/pythia-1.4b",
    eval_steps: int = 8,
    batch_size: int = 8,
    seq_len: int = 256,
    dtype: str = "bf16",
    data_split: str = "test",
    ce_chunk_tokens: int = 32768,
    skip_baseline: bool = False,
    greedy_max_layers: int | None = None,
    run_name: str = "pythia_layer_removal_eval",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.layer_removal import LayerRemovalEvalConfig, run_layer_removal_eval

    config = LayerRemovalEvalConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        model_name=model_name,
        eval_steps=eval_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        dtype=dtype,
        data_split=data_split,
        ce_chunk_tokens=ce_chunk_tokens,
        include_baseline=not skip_baseline,
        greedy_layer_removal=True,
        greedy_max_layers=greedy_max_layers,
    )
    summary = run_layer_removal_eval(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_h100_sparse24_eval(
    model_name: str = "EleutherAI/pythia-31m",
    methods: str = "magnitude,wanda,sparsegpt,gptaq-cae,qronos",
    calibration_steps: int = 4,
    calibration_batch_size: int = 64,
    calibration_seq_len: int = 256,
    calibration_tokens: int = 32768,
    eval_steps: int = 16,
    eval_batch_size: int = 64,
    eval_seq_len: int = 256,
    dtype: str = "bf16",
    data_split: str = "test",
    calibration_split: str = "train",
    ce_chunk_tokens: int = 32768,
    sparsity_n: int = 2,
    sparsity_m: int = 4,
    gd_steps: int = 1,
    gd_lr: float = 0.25,
    gd_chunk_tokens: int = 8192,
    save_sparse_model: bool = False,
    run_name: str = "pythia31m_sparse24_eval",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.sparse24 import Sparse24EvalConfig, _parse_methods, run_sparse24_eval

    config = Sparse24EvalConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        model_name=model_name,
        methods=_parse_methods(methods),
        calibration_steps=calibration_steps,
        calibration_batch_size=calibration_batch_size,
        calibration_seq_len=calibration_seq_len,
        calibration_tokens=calibration_tokens,
        eval_steps=eval_steps,
        eval_batch_size=eval_batch_size,
        eval_seq_len=eval_seq_len,
        dtype=dtype,
        data_split=data_split,
        calibration_split=calibration_split,
        ce_chunk_tokens=ce_chunk_tokens,
        sparsity_n=sparsity_n,
        sparsity_m=sparsity_m,
        gd_steps=gd_steps,
        gd_lr=gd_lr,
        gd_chunk_tokens=gd_chunk_tokens,
        save_sparse_model=save_sparse_model,
    )
    summary = run_sparse24_eval(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_h100_sparse24_greedy_layer_eval(
    model_name: str = "EleutherAI/pythia-1.4b",
    calibration_steps: int = 4,
    calibration_batch_size: int = 64,
    calibration_seq_len: int = 256,
    calibration_tokens: int = 32768,
    eval_steps: int = 16,
    eval_batch_size: int = 64,
    eval_seq_len: int = 256,
    dtype: str = "bf16",
    data_split: str = "test",
    calibration_split: str = "train",
    ce_chunk_tokens: int = 32768,
    sparsity_n: int = 2,
    sparsity_m: int = 4,
    gd_steps: int = 1,
    gd_lr: float = 0.25,
    gd_chunk_tokens: int = 8192,
    greedy_max_layers: int | None = None,
    save_sparse_model: bool = False,
    run_name: str = "pythia_sparse24_greedy_layers",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.sparse24 import Sparse24GreedyLayerConfig, run_sparse24_greedy_layer_eval

    config = Sparse24GreedyLayerConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        model_name=model_name,
        calibration_steps=calibration_steps,
        calibration_batch_size=calibration_batch_size,
        calibration_seq_len=calibration_seq_len,
        calibration_tokens=calibration_tokens,
        eval_steps=eval_steps,
        eval_batch_size=eval_batch_size,
        eval_seq_len=eval_seq_len,
        dtype=dtype,
        data_split=data_split,
        calibration_split=calibration_split,
        ce_chunk_tokens=ce_chunk_tokens,
        sparsity_n=sparsity_n,
        sparsity_m=sparsity_m,
        gd_steps=gd_steps,
        gd_lr=gd_lr,
        gd_chunk_tokens=gd_chunk_tokens,
        greedy_max_layers=greedy_max_layers,
        save_sparse_model=save_sparse_model,
    )
    summary = run_sparse24_greedy_layer_eval(config, commit_callback=volume.commit)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
    max_containers=6,
)
def run_h100_sparse24_layer_state_batch(spec: dict) -> dict:
    import json
    import os
    import random
    import sys

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.sparse24 import (
        Sparse24LayerStateConfig,
        _load_causal_lm,
        _load_wikitext_eval_tokens,
        _make_eval_batches,
        _restore_module_weights,
        _snapshot_module_weights,
        _torch_dtype,
        find_prunable_linear_names,
        group_prunable_linear_names_by_layer,
        run_sparse24_layer_state_eval_loaded,
        sparse24_worker_cache_key,
    )

    run_name = spec["run_name"]
    batch_name = spec["batch_name"]
    method = spec["method"]
    seed = int(spec.get("seed", 0))
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(spec["dtype"])
    model_dtype = dtype if device.type == "cuda" else torch.float32

    template_config = Sparse24LayerStateConfig(
        output_dir=Path(RESULTS_ROOT) / run_name / method / batch_name,
        model_name=spec["model_name"],
        method=method,
        calibration_steps=spec["calibration_steps"],
        calibration_batch_size=spec["calibration_batch_size"],
        calibration_seq_len=spec["calibration_seq_len"],
        calibration_tokens=spec["calibration_tokens"],
        eval_steps=spec["eval_steps"],
        eval_batch_size=spec["eval_batch_size"],
        eval_seq_len=spec["eval_seq_len"],
        dtype=spec["dtype"],
        data_split=spec["data_split"],
        calibration_split=spec["calibration_split"],
        ce_chunk_tokens=spec["ce_chunk_tokens"],
        sparsity_n=spec["sparsity_n"],
        sparsity_m=spec["sparsity_m"],
        gd_steps=spec.get("gd_steps", 1),
        gd_lr=spec.get("gd_lr", 0.25),
        gd_chunk_tokens=spec.get("gd_chunk_tokens", 8192),
        include_baseline=False,
        log_gpu_stats=True,
    )

    global _SPARSE24_LAYER_STATE_WORKER_CACHE
    cache_key = sparse24_worker_cache_key(spec)
    worker_cache = _SPARSE24_LAYER_STATE_WORKER_CACHE.get(cache_key)
    if worker_cache is None:
        tokenizer = AutoTokenizer.from_pretrained(spec["model_name"])
        calib_tokens = _load_wikitext_eval_tokens(
            tokenizer=tokenizer,
            split=spec["calibration_split"],
            max_tokens=template_config.max_dataset_tokens,
        )
        eval_tokens = _load_wikitext_eval_tokens(
            tokenizer=tokenizer,
            split=spec["data_split"],
            max_tokens=template_config.max_dataset_tokens,
        )
        calibration_batches = _make_eval_batches(
            calib_tokens,
            batch_size=spec["calibration_batch_size"],
            seq_len=spec["calibration_seq_len"],
            max_steps=spec["calibration_steps"],
        )
        eval_batches = _make_eval_batches(
            eval_tokens,
            batch_size=spec["eval_batch_size"],
            seq_len=spec["eval_seq_len"],
            max_steps=spec["eval_steps"],
        )
        if not calibration_batches:
            raise ValueError("not enough tokens for calibration batches")
        if not eval_batches:
            raise ValueError("not enough tokens for eval batches")

        fp_model = _load_causal_lm(AutoModelForCausalLM, spec["model_name"], model_dtype).to(device)
        sparse_model = _load_causal_lm(AutoModelForCausalLM, spec["model_name"], model_dtype).to(device)
        fp_model.eval()
        sparse_model.eval()
        for model in (fp_model, sparse_model):
            for param in model.parameters():
                param.requires_grad_(False)
        module_names = find_prunable_linear_names(fp_model, sparsity_m=spec["sparsity_m"])
        groups = group_prunable_linear_names_by_layer(module_names)
        sparse_snapshot = _snapshot_module_weights(sparse_model, tuple(module_names))
        worker_cache = {
            "fp_model": fp_model,
            "sparse_model": sparse_model,
            "calibration_batches": calibration_batches,
            "eval_batches": eval_batches,
            "module_names": module_names,
            "groups": groups,
            "sparse_snapshot": sparse_snapshot,
        }
        _SPARSE24_LAYER_STATE_WORKER_CACHE.clear()
        _SPARSE24_LAYER_STATE_WORKER_CACHE[cache_key] = worker_cache
        print(json.dumps({"event": "sparse24_worker_cache_miss", "cache_key": list(cache_key)}, sort_keys=True), flush=True)
    else:
        print(json.dumps({"event": "sparse24_worker_cache_hit", "cache_key": list(cache_key)}, sort_keys=True), flush=True)

    fp_model = worker_cache["fp_model"]
    sparse_model = worker_cache["sparse_model"]
    calibration_batches = worker_cache["calibration_batches"]
    eval_batches = worker_cache["eval_batches"]
    module_names = worker_cache["module_names"]
    groups = worker_cache["groups"]
    sparse_snapshot = worker_cache["sparse_snapshot"]

    records = []
    for item in spec["states"]:
        layer_indices = tuple(int(idx) for idx in item["layer_indices"])
        state_name = item["state_name"]
        _restore_module_weights(sparse_model, sparse_snapshot)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        config = Sparse24LayerStateConfig(
            output_dir=Path(RESULTS_ROOT) / run_name / method / batch_name / state_name,
            model_name=spec["model_name"],
            method=method,
            calibration_steps=spec["calibration_steps"],
            calibration_batch_size=spec["calibration_batch_size"],
            calibration_seq_len=spec["calibration_seq_len"],
            calibration_tokens=spec["calibration_tokens"],
            eval_steps=spec["eval_steps"],
            eval_batch_size=spec["eval_batch_size"],
            eval_seq_len=spec["eval_seq_len"],
            dtype=spec["dtype"],
            data_split=spec["data_split"],
            calibration_split=spec["calibration_split"],
            ce_chunk_tokens=spec["ce_chunk_tokens"],
            sparsity_n=spec["sparsity_n"],
            sparsity_m=spec["sparsity_m"],
            gd_steps=spec.get("gd_steps", 1),
            gd_lr=spec.get("gd_lr", 0.25),
            gd_chunk_tokens=spec.get("gd_chunk_tokens", 8192),
            include_baseline=False,
            log_gpu_stats=True,
        )
        record = run_sparse24_layer_state_eval_loaded(
            config,
            fp_model=fp_model,
            sparse_model=sparse_model,
            calibration_batches=calibration_batches,
            eval_batches=eval_batches,
            module_names=module_names,
            groups=groups,
            layer_indices=layer_indices,
            run_name=state_name,
            device=device,
            dtype=dtype,
        )
        record["state_name"] = state_name
        record["mcts_path"] = item.get("path", [])
        records.append(record)
        volume.commit()
    return {"run_name": run_name, "method": method, "batch_name": batch_name, "records": records}


@app.function(image=image, volumes={RESULTS_ROOT: volume})
def read_sparse24_layer_state_records(spec: dict) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.sparse24 import read_sparse24_state_records

    run_name = spec["run_name"]
    method = spec["method"]
    batch_name = spec["batch_name"]
    records, missing = read_sparse24_state_records(
        Path(RESULTS_ROOT) / run_name / method / batch_name,
        list(spec["states"]),
    )
    return {"run_name": run_name, "method": method, "batch_name": batch_name, "records": records, "missing": missing}


@app.function(image=image, volumes={RESULTS_ROOT: volume})
def write_sparse24_mcts_summary(run_name: str, summary_json: str) -> None:
    path = Path(RESULTS_ROOT) / run_name
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(summary_json + "\n")
    volume.commit()


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_h100_low_qk_sweep(
    learning_rates: str = "1e-3,3e-3,1e-2",
    steps: int = 100,
    batch_size: int = 1024,
    seq_len: int = 1024,
    layer_index: int | None = None,
    qk_dim: int = 2,
    student_heads: int | None = None,
    dtype: str = "bf16",
    run_name: str = "pythia70m_low_qk_attention",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.low_qk_attention import LowQKDistillConfig, _parse_lrs, run_low_qk_sweep

    config = LowQKDistillConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        learning_rates=_parse_lrs(learning_rates),
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        layer_index=layer_index,
        qk_dim=qk_dim,
        student_heads=student_heads,
        dtype=dtype,
        data="wikitext",
    )
    summary = run_low_qk_sweep(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_h100_low_qk_model_sweep(
    learning_rates: str = "3e-2",
    steps: int = 100,
    batch_size: int = 512,
    seq_len: int = 256,
    qk_dim: int = 2,
    student_heads: int | None = None,
    dtype: str = "bf16",
    run_name: str = "pythia70m_low_qk_model",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.low_qk_model import EndToEndLowQKConfig, _parse_lrs, run_end_to_end_low_qk_sweep

    config = EndToEndLowQKConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        learning_rates=_parse_lrs(learning_rates),
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        qk_dim=qk_dim,
        student_heads=student_heads,
        dtype=dtype,
        data="wikitext",
    )
    summary = run_end_to_end_low_qk_sweep(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_h100_low_qk_logit_sweep(
    learning_rates: str = "1e-3",
    steps: int = 100,
    batch_size: int = 256,
    seq_len: int = 256,
    qk_dim: int = 2,
    student_heads: int | None = None,
    temperature: float = 1.0,
    logit_chunk_tokens: int = 8192,
    dtype: str = "bf16",
    run_name: str = "pythia70m_low_qk_logit_distill",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.low_qk_model import LowQKLogitDistillConfig, _parse_lrs, run_logit_low_qk_sweep

    config = LowQKLogitDistillConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        learning_rates=_parse_lrs(learning_rates),
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        qk_dim=qk_dim,
        student_heads=student_heads,
        temperature=temperature,
        logit_chunk_tokens=logit_chunk_tokens,
        dtype=dtype,
        data="wikitext",
    )
    summary = run_logit_low_qk_sweep(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_h100_low_qk_ppl_eval(
    adapter_run_name: str,
    eval_steps: int = 64,
    batch_size: int = 32,
    seq_len: int = 512,
    qk_dim: int = 2,
    student_heads: int | None = None,
    ce_chunk_tokens: int = 32768,
    dtype: str = "bf16",
    data_split: str = "test",
    run_name: str = "pythia70m_low_qk_ppl_eval",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from layer_distill.low_qk_model import LowQKPerplexityEvalConfig, run_low_qk_perplexity_eval

    config = LowQKPerplexityEvalConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        adapter_root=Path(RESULTS_ROOT) / adapter_run_name,
        eval_steps=eval_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        qk_dim=qk_dim,
        student_heads=student_heads,
        ce_chunk_tokens=ce_chunk_tokens,
        dtype=dtype,
        data_split=data_split,
    )
    summary = run_low_qk_perplexity_eval(config)
    volume.commit()
    return summary


@app.local_entrypoint()
def main(
    learning_rates: str = "3e-4,1e-3,3e-3",
    steps: int = 100,
    batch_size: int = 512,
    seq_len: int = 256,
    layer_index: int | None = None,
    dtype: str = "bf16",
    compile_student: bool = False,
    run_name: str = "pythia70m_middle_pair",
    mode: str = "distill",
    methods: str = "slerp,geom_slerp",
    t_values: str = "0,0.25,0.5,0.75,1",
    save_student: bool = False,
    model_pairs: str = "",
    layer_groups: str = "",
    small_model: str = "EleutherAI/pythia-31m",
    big_model: str = "EleutherAI/pythia-70m",
    prompts: str = "",
    combo_method: str = "exponential",
    basis_combine_mode: str = "linear",
    basis_quantization_bits: int = 0,
    basis_quantization_format: str = "",
    basis_quantization_target: str = "reconstructed",
    basis_sizes: str = "",
    fit_lr: float = 0.2,
    max_length: int = 48,
    local_window: int = 8,
    qk_dim: int = 2,
    student_heads: int | None = None,
    adapter_run_name: str = "",
    combo_run_name: str = "",
    eval_steps: int = 64,
    data_split: str = "test",
    ce_chunk_tokens: int = 32768,
    temperature: float = 1.0,
    logit_chunk_tokens: int = 8192,
    per_layer_sweep: bool = False,
    greedy_layer_sweep: bool = False,
    greedy_max_layers: int | None = None,
    beam_width: int = 4,
    parallel_workers: int = 4,
    search_layer_count: int = 24,
    calibration_steps: int = 4,
    calibration_batch_size: int = 64,
    calibration_seq_len: int = 256,
    calibration_tokens: int = 32768,
    eval_batch_size: int = 64,
    eval_seq_len: int = 256,
    calibration_split: str = "train",
    sparsity_n: int = 2,
    sparsity_m: int = 4,
    sparse_gd_steps: int = 1,
    sparse_gd_lr: float = 0.25,
    sparse_gd_chunk_tokens: int = 8192,
    mcts_iterations: int = 16,
    mcts_rollout_depth: int = 3,
    mcts_exploration: float = 1.4,
):
    if mode == "distill":
        summary = run_h100_sweep.remote(
            learning_rates=learning_rates,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            layer_index=layer_index,
            dtype=dtype,
            compile_student=compile_student,
            run_name=run_name,
        )
    elif mode == "merge":
        summary = run_h100_merge_sweep.remote(
            methods=methods,
            t_values=t_values,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            layer_index=layer_index,
            dtype=dtype,
            save_student=save_student,
            run_name=run_name,
        )
    elif mode == "attention":
        if not model_pairs:
            raise ValueError("model_pairs is required for mode=attention")
        if not prompts:
            raise ValueError("prompts is required for mode=attention")
        summary = run_h100_attention_sweep.remote(
            model_pairs=model_pairs,
            prompts=prompts,
            max_length=max_length,
            dtype=dtype,
            local_window=local_window,
            save_tensors=save_student,
            run_name=run_name,
        )
    elif mode == "attention-combo":
        if not prompts:
            raise ValueError("prompts is required for mode=attention-combo")
        summary = run_h100_attention_combo.remote(
            small_model=small_model,
            big_model=big_model,
            prompts=prompts,
            combo_method=combo_method,
            max_length=max_length,
            dtype=dtype,
            local_window=local_window,
            save_tensors=save_student,
            fit_steps=steps,
            fit_lr=fit_lr,
            run_name=run_name,
        )
    elif mode == "attention-basis":
        if not prompts:
            raise ValueError("prompts is required for mode=attention-basis")
        model_names = small_model if not model_pairs else ",".join(dict.fromkeys(side.strip() for pair in model_pairs.split(",") for side in pair.split(">") if side.strip()))
        summary = run_h100_attention_basis_sweep.remote(
            model_names=model_names,
            prompts=prompts,
            max_length=max_length,
            dtype=dtype,
            basis_sizes=basis_sizes,
            fit_steps=steps,
            fit_lr=fit_lr,
            run_name=run_name,
        )
    elif mode == "attention-basis-ppl":
        summary = run_h100_attention_basis_ppl_eval.remote(
            model_name=small_model,
            basis_sizes=basis_sizes,
            nmf_iterations=steps,
            eval_steps=eval_steps,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=dtype,
            data_split=data_split,
            ce_chunk_tokens=ce_chunk_tokens,
            combine_mode=basis_combine_mode,
            basis_quantization_bits=basis_quantization_bits,
            basis_quantization_format=basis_quantization_format,
            basis_quantization_target=basis_quantization_target,
            run_name=run_name,
        )
    elif mode == "attention-basis-layer-search":
        summary = _run_parallel_attention_basis_layer_search(
            run_name=run_name,
            model_name=small_model,
            basis_size=int((basis_sizes or "8").split(",")[0]),
            nmf_iterations=steps,
            eval_steps=eval_steps,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=dtype,
            data_split=data_split,
            ce_chunk_tokens=ce_chunk_tokens,
            layer_count=search_layer_count,
            max_depth=greedy_max_layers or 6,
            beam_width=beam_width,
            parallel_workers=parallel_workers,
            combine_mode=basis_combine_mode,
            basis_quantization_bits=basis_quantization_bits,
            basis_quantization_format=basis_quantization_format,
            basis_quantization_target=basis_quantization_target,
        )
    elif mode == "attention-basis-layer-groups":
        groups = _parse_layer_groups_arg(layer_groups)
        if not groups:
            raise ValueError("layer_groups is required for mode=attention-basis-layer-groups")
        summary = run_h100_attention_basis_candidate_batch.remote(
            {
                "run_name": run_name,
                "round_name": "layer_groups",
                "shard_name": "all",
                "model_name": small_model,
                "basis_size": int((basis_sizes or "8").split(",")[0]),
                "nmf_iterations": steps,
                "layer_groups": [list(group) for group in ((), *groups)],
                "eval_steps": eval_steps,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "dtype": dtype,
                "data_split": data_split,
                "ce_chunk_tokens": ce_chunk_tokens,
                "log_gpu_stats": True,
                "combine_mode": basis_combine_mode,
                "basis_quantization_bits": basis_quantization_bits,
                "basis_quantization_format": basis_quantization_format,
                "basis_quantization_target": basis_quantization_target,
            }
        )
    elif mode == "hybrid-attn":
        if not combo_run_name:
            raise ValueError("combo_run_name is required for mode=hybrid-attn")
        summary = run_h100_hybrid_attention_eval.remote(
            combo_run_name=combo_run_name,
            small_model=small_model,
            big_model=big_model,
            eval_steps=eval_steps,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=dtype,
            data_split=data_split,
            ce_chunk_tokens=ce_chunk_tokens,
            include_small_baseline=save_student,
            skip_big_baseline=False,
            per_layer_sweep=per_layer_sweep,
            greedy_layer_sweep=greedy_layer_sweep,
            greedy_max_layers=greedy_max_layers,
            run_name=run_name,
        )
    elif mode == "layer-removal":
        summary = run_h100_layer_removal_eval.remote(
            model_name=big_model,
            eval_steps=eval_steps,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=dtype,
            data_split=data_split,
            ce_chunk_tokens=ce_chunk_tokens,
            skip_baseline=False,
            greedy_max_layers=greedy_max_layers,
            run_name=run_name,
        )
    elif mode == "sparse24":
        summary = run_h100_sparse24_eval.remote(
            model_name=big_model,
            methods=methods,
            calibration_steps=calibration_steps,
            calibration_batch_size=calibration_batch_size,
            calibration_seq_len=calibration_seq_len,
            calibration_tokens=calibration_tokens,
            eval_steps=eval_steps,
            eval_batch_size=eval_batch_size,
            eval_seq_len=eval_seq_len,
            dtype=dtype,
            data_split=data_split,
            calibration_split=calibration_split,
            ce_chunk_tokens=ce_chunk_tokens,
            sparsity_n=sparsity_n,
            sparsity_m=sparsity_m,
            gd_steps=sparse_gd_steps,
            gd_lr=sparse_gd_lr,
            gd_chunk_tokens=sparse_gd_chunk_tokens,
            save_sparse_model=save_student,
            run_name=run_name,
        )
    elif mode == "sparse24-greedy":
        summary = run_h100_sparse24_greedy_layer_eval.remote(
            model_name=big_model,
            calibration_steps=calibration_steps,
            calibration_batch_size=calibration_batch_size,
            calibration_seq_len=calibration_seq_len,
            calibration_tokens=calibration_tokens,
            eval_steps=eval_steps,
            eval_batch_size=eval_batch_size,
            eval_seq_len=eval_seq_len,
            dtype=dtype,
            data_split=data_split,
            calibration_split=calibration_split,
            ce_chunk_tokens=ce_chunk_tokens,
            sparsity_n=sparsity_n,
            sparsity_m=sparsity_m,
            gd_steps=sparse_gd_steps,
            gd_lr=sparse_gd_lr,
            gd_chunk_tokens=sparse_gd_chunk_tokens,
            greedy_max_layers=greedy_max_layers,
            save_sparse_model=save_student,
            run_name=run_name,
        )
    elif mode == "sparse24-state":
        state = _parse_layer_order_arg(layer_groups)
        if not state:
            raise ValueError("layer_groups must provide a comma-separated layer order for mode=sparse24-state")
        if len(set(state)) != len(state):
            raise ValueError(f"layer_groups contains duplicate layers: {state}")
        state_name = "layers_" + "_".join(f"{idx:02d}" for idx in state)
        summary = run_h100_sparse24_layer_state_batch.remote(
            {
                "run_name": run_name,
                "batch_name": "fixed_state",
                "method": methods.split(",")[0].strip() if methods else "gptaq-cae",
                "model_name": big_model,
                "dtype": dtype,
                "calibration_steps": calibration_steps,
                "calibration_batch_size": calibration_batch_size,
                "calibration_seq_len": calibration_seq_len,
                "calibration_tokens": calibration_tokens,
                "eval_steps": eval_steps,
                "eval_batch_size": eval_batch_size,
                "eval_seq_len": eval_seq_len,
                "data_split": data_split,
                "calibration_split": calibration_split,
                "ce_chunk_tokens": ce_chunk_tokens,
                "sparsity_n": sparsity_n,
                "sparsity_m": sparsity_m,
                "gd_steps": sparse_gd_steps,
                "gd_lr": sparse_gd_lr,
                "gd_chunk_tokens": sparse_gd_chunk_tokens,
                "seed": 17,
                "states": [
                    {
                        "state_name": state_name,
                        "layer_indices": list(state),
                        "path": [list(state[:idx]) for idx in range(len(state) + 1)],
                    }
                ],
            }
        )
    elif mode == "sparse24-mcts":
        summary = _run_parallel_sparse24_mcts_search(
            run_name=run_name,
            model_name=big_model,
            methods=methods,
            eval_steps=eval_steps,
            eval_batch_size=eval_batch_size,
            eval_seq_len=eval_seq_len,
            dtype=dtype,
            data_split=data_split,
            ce_chunk_tokens=ce_chunk_tokens,
            calibration_steps=calibration_steps,
            calibration_batch_size=calibration_batch_size,
            calibration_seq_len=calibration_seq_len,
            calibration_tokens=calibration_tokens,
            calibration_split=calibration_split,
            sparsity_n=sparsity_n,
            sparsity_m=sparsity_m,
            max_layers=greedy_max_layers or search_layer_count,
            iterations=mcts_iterations,
            rollout_depth=mcts_rollout_depth,
            exploration=mcts_exploration,
            parallel_workers=parallel_workers,
        )
    elif mode == "low-qk":
        summary = run_h100_low_qk_sweep.remote(
            learning_rates=learning_rates,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            layer_index=layer_index,
            qk_dim=qk_dim,
            student_heads=student_heads,
            dtype=dtype,
            run_name=run_name,
        )
    elif mode == "low-qk-model":
        summary = run_h100_low_qk_model_sweep.remote(
            learning_rates=learning_rates,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            qk_dim=qk_dim,
            student_heads=student_heads,
            dtype=dtype,
            run_name=run_name,
        )
    elif mode == "low-qk-logit":
        summary = run_h100_low_qk_logit_sweep.remote(
            learning_rates=learning_rates,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            qk_dim=qk_dim,
            student_heads=student_heads,
            temperature=temperature,
            logit_chunk_tokens=logit_chunk_tokens,
            dtype=dtype,
            run_name=run_name,
        )
    elif mode == "low-qk-ppl":
        if not adapter_run_name:
            raise ValueError("adapter_run_name is required for mode=low-qk-ppl")
        summary = run_h100_low_qk_ppl_eval.remote(
            adapter_run_name=adapter_run_name,
            eval_steps=eval_steps,
            batch_size=batch_size,
            seq_len=seq_len,
            qk_dim=qk_dim,
            student_heads=student_heads,
            ce_chunk_tokens=ce_chunk_tokens,
            dtype=dtype,
            data_split=data_split,
            run_name=run_name,
        )
    else:
        raise ValueError(
            "mode must be distill, merge, attention, attention-combo, attention-basis, attention-basis-ppl, attention-basis-layer-search, attention-basis-layer-groups, hybrid-attn, layer-removal, sparse24, sparse24-greedy, "
            "sparse24-mcts, low-qk, low-qk-model, low-qk-logit, or low-qk-ppl"
        )
    print(summary)


def _parse_layer_groups_arg(value: str) -> tuple[tuple[int, ...], ...]:
    groups = []
    for group_text in value.split(";"):
        group_text = group_text.strip()
        if not group_text:
            continue
        groups.append(tuple(sorted({int(part.strip()) for part in group_text.split(",") if part.strip()})))
    return tuple(groups)


def _parse_layer_order_arg(value: str) -> tuple[int, ...]:
    parts = []
    for group_text in value.split(";"):
        group_text = group_text.strip()
        if not group_text:
            continue
        parts.extend(int(part.strip()) for part in group_text.split(",") if part.strip())
        break
    return tuple(parts)


def _chunks(values: list, shard_count: int) -> list[list]:
    shard_count = max(1, min(shard_count, len(values)))
    shards = [[] for _ in range(shard_count)]
    for idx, value in enumerate(values):
        shards[idx % shard_count].append(value)
    return [shard for shard in shards if shard]


def _run_parallel_sparse24_mcts_search(
    *,
    run_name: str,
    model_name: str,
    methods: str,
    eval_steps: int,
    eval_batch_size: int,
    eval_seq_len: int,
    dtype: str,
    data_split: str,
    ce_chunk_tokens: int,
    calibration_steps: int,
    calibration_batch_size: int,
    calibration_seq_len: int,
    calibration_tokens: int,
    calibration_split: str,
    sparsity_n: int,
    sparsity_m: int,
    max_layers: int,
    iterations: int,
    rollout_depth: int,
    exploration: float,
    parallel_workers: int,
) -> dict:
    import hashlib
    import json
    import random

    from layer_distill.sparse24 import (
        _parse_methods,
        sparse24_state_name,
        sparse24_mcts_backpropagate,
        sparse24_mcts_best_child,
        sparse24_mcts_select_rollout,
    )

    method_list = _parse_methods(methods)
    if not method_list:
        raise ValueError("methods must be non-empty for sparse24-mcts")
    if max_layers <= 0:
        raise ValueError("max_layers/search_layer_count must be positive")
    if iterations <= 0:
        raise ValueError("mcts_iterations must be positive")
    if rollout_depth <= 0:
        raise ValueError("mcts_rollout_depth must be positive")
    if parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")

    layer_indices = tuple(range(max_layers))

    def evaluate_states(method: str, states: list[tuple[int, ...]], batch_name: str) -> list[dict]:
        unique_states = []
        seen = set()
        for state in states:
            if state in seen:
                continue
            seen.add(state)
            unique_states.append(state)
        state_items = [
            {
                "state_name": sparse24_state_name(state),
                "layer_indices": list(state),
            }
            for state in unique_states
        ]
        resumed = read_sparse24_layer_state_records.remote(
            {
                "run_name": run_name,
                "batch_name": batch_name,
                "method": method,
                "states": state_items,
            }
        )
        records = list(resumed["records"])
        missing_items = list(resumed["missing"])
        if not missing_items:
            return records
        specs = []
        for shard_idx, shard in enumerate(_chunks(missing_items, parallel_workers)):
            specs.append(
                {
                    "run_name": run_name,
                    "batch_name": batch_name,
                    "method": method,
                    "model_name": model_name,
                    "states": shard,
                    "calibration_steps": calibration_steps,
                    "calibration_batch_size": calibration_batch_size,
                    "calibration_seq_len": calibration_seq_len,
                    "calibration_tokens": calibration_tokens,
                    "eval_steps": eval_steps,
                    "eval_batch_size": eval_batch_size,
                    "eval_seq_len": eval_seq_len,
                    "dtype": dtype,
                    "data_split": data_split,
                    "calibration_split": calibration_split,
                    "ce_chunk_tokens": ce_chunk_tokens,
                    "sparsity_n": sparsity_n,
                    "sparsity_m": sparsity_m,
                }
            )
        for shard_summary in run_h100_sparse24_layer_state_batch.map(
            specs,
            order_outputs=False,
            return_exceptions=True,
        ):
            if isinstance(shard_summary, BaseException):
                raise RuntimeError(f"sparse24 state batch failed for {method}/{batch_name}") from shard_summary
            records.extend(shard_summary["records"])
        return records

    method_summaries = []
    all_records = []
    for method in method_list:
        seed_bytes = hashlib.blake2s(f"{run_name}\0{method}".encode(), digest_size=4).digest()
        rng = random.Random(int.from_bytes(seed_bytes, "big") + 17)
        stats: dict[tuple[int, ...], dict[str, float]] = {}
        cache: dict[tuple[int, ...], dict] = {}

        baseline_record = evaluate_states(method, [()], f"{method}_baseline")[0]
        base_ppl = float(baseline_record["ppl"])
        base_loss = float(baseline_record["loss"])
        cache[()] = baseline_record
        sparse24_mcts_backpropagate(stats, path=[()], ppl=base_ppl)

        root: tuple[int, ...] = ()
        path_records = []
        method_records = [baseline_record]
        for depth in range(1, max_layers + 1):
            suggestion_paths: dict[tuple[int, ...], list[list[tuple[int, ...]]]] = {}
            for iteration in range(iterations):
                terminal, mcts_path = sparse24_mcts_select_rollout(
                    root=root,
                    layer_indices=layer_indices,
                    stats=stats,
                    rollout_depth=rollout_depth,
                    exploration=exploration,
                    rng=rng,
                )
                suggestion_paths.setdefault(terminal, []).append(mcts_path)

            missing = [state for state in suggestion_paths if state not in cache]
            if missing:
                records = evaluate_states(method, missing, f"{method}_depth_{depth:02d}_rollouts")
                for record in records:
                    state = tuple(int(idx) for idx in record["layer_indices"])
                    cache[state] = record
                    method_records.append(record)

            for terminal, paths in suggestion_paths.items():
                ppl = float(cache[terminal]["ppl"])
                for mcts_path in paths:
                    sparse24_mcts_backpropagate(stats, path=mcts_path, ppl=ppl)

            best_child = sparse24_mcts_best_child(root=root, layer_indices=layer_indices, stats=stats)
            if best_child is None:
                break
            if best_child not in cache:
                record = evaluate_states(method, [best_child], f"{method}_depth_{depth:02d}_selected")[0]
                cache[best_child] = record
                method_records.append(record)
            selected = cache[best_child]
            selected["ppl_ratio_vs_baseline"] = float(selected["ppl"]) / base_ppl
            selected["loss_delta_vs_baseline"] = float(selected["loss"]) - base_loss
            selected["ppl_delta_vs_baseline"] = float(selected["ppl"]) - base_ppl
            path_records.append(
                {
                    "depth": depth,
                    "layer_indices": list(best_child),
                    "selected_layer": best_child[-1],
                    "ppl": selected["ppl"],
                    "loss": selected["loss"],
                    "ppl_ratio_vs_baseline": selected["ppl_ratio_vs_baseline"],
                    "mcts_child_stats": stats.get(best_child, {}),
                    "rollout_count": iterations,
                }
            )
            root = best_child
            if len(root) >= len(layer_indices):
                break

        for record in method_records:
            record["ppl_ratio_vs_baseline"] = float(record["ppl"]) / base_ppl
            record["loss_delta_vs_baseline"] = float(record["loss"]) - base_loss
            record["ppl_delta_vs_baseline"] = float(record["ppl"]) - base_ppl
        method_summary = {
            "method": method,
            "baseline": baseline_record,
            "path": path_records,
            "record_count": len(method_records),
            "best_record": min(method_records, key=lambda row: (float(row["ppl"]), len(row.get("layer_indices", [])))),
            "records": sorted(method_records, key=lambda row: (len(row.get("layer_indices", [])), float(row["ppl"]))),
        }
        method_summaries.append(method_summary)
        all_records.extend(method_records)

    summary = {
        "model_name": model_name,
        "methods": list(method_list),
        "layer_indices": list(layer_indices),
        "max_layers": max_layers,
        "iterations": iterations,
        "rollout_depth": rollout_depth,
        "exploration": exploration,
        "parallel_workers": parallel_workers,
        "eval_steps": eval_steps,
        "eval_batch_size": eval_batch_size,
        "eval_seq_len": eval_seq_len,
        "calibration_steps": calibration_steps,
        "calibration_batch_size": calibration_batch_size,
        "calibration_seq_len": calibration_seq_len,
        "calibration_tokens": calibration_tokens,
        "sparsity_n": sparsity_n,
        "sparsity_m": sparsity_m,
        "method_summaries": method_summaries,
        "records": all_records,
    }
    write_sparse24_mcts_summary.remote(run_name, json.dumps(summary, sort_keys=True, default=str))
    return summary


def _run_parallel_attention_basis_layer_search(
    *,
    run_name: str,
    model_name: str,
    basis_size: int,
    nmf_iterations: int,
    eval_steps: int,
    batch_size: int,
    seq_len: int,
    dtype: str,
    data_split: str,
    ce_chunk_tokens: int,
    layer_count: int,
    max_depth: int,
    beam_width: int,
    parallel_workers: int,
    combine_mode: str = "linear",
    basis_quantization_bits: int = 0,
    basis_quantization_format: str = "",
    basis_quantization_target: str = "reconstructed",
) -> dict:
    import json

    from layer_distill.attention_basis_ppl import build_next_layer_candidates, select_top_layer_groups

    if layer_count <= 0:
        raise ValueError("search_layer_count must be positive")
    if max_depth <= 0:
        raise ValueError("greedy_max_layers must be positive for attention-basis-layer-search")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")

    def evaluate_groups(groups: tuple[tuple[int, ...], ...], round_name: str) -> list[dict]:
        specs = []
        for shard_idx, shard in enumerate(_chunks(list(groups), parallel_workers)):
            specs.append(
                {
                    "run_name": run_name,
                    "round_name": round_name,
                    "shard_name": f"shard_{shard_idx:02d}",
                    "model_name": model_name,
                    "basis_size": basis_size,
                    "nmf_iterations": nmf_iterations,
                    "layer_groups": [list(group) for group in shard],
                    "eval_steps": eval_steps,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "dtype": dtype,
                    "data_split": data_split,
                    "ce_chunk_tokens": ce_chunk_tokens,
                    "log_gpu_stats": True,
                    "combine_mode": combine_mode,
                    "basis_quantization_bits": basis_quantization_bits,
                    "basis_quantization_format": basis_quantization_format,
                    "basis_quantization_target": basis_quantization_target,
                }
            )
        records = []
        for shard_summary in run_h100_attention_basis_candidate_batch.map(specs):
            records.extend(shard_summary["runs"])
        return records

    all_records: list[dict] = []
    baseline_records = evaluate_groups(((),), "round_00_baseline")
    baseline = baseline_records[0]
    base_ppl = float(baseline["ppl"])
    baseline["ppl_ratio_vs_baseline"] = 1.0
    baseline["ppl_delta_vs_baseline"] = 0.0
    all_records.extend(baseline_records)

    frontier: tuple[tuple[int, ...], ...] = ((),)
    path: list[dict] = []
    seen: set[tuple[int, ...]] = {()}
    for depth in range(1, max_depth + 1):
        candidates = tuple(group for group in build_next_layer_candidates(frontier=frontier, layer_count=layer_count) if group not in seen)
        if not candidates:
            break
        seen.update(candidates)
        records = evaluate_groups(candidates, f"round_{depth:02d}")
        for record in records:
            record["depth"] = depth
            record["ppl_ratio_vs_baseline"] = float(record["ppl"]) / base_ppl
            record["ppl_delta_vs_baseline"] = float(record["ppl"]) - base_ppl
        all_records.extend(records)
        frontier = select_top_layer_groups(records, beam_width=beam_width)
        best = min(records, key=lambda row: (float(row["ppl"]), tuple(row.get("layer_group", ()))))
        path.append(
            {
                "depth": depth,
                "layer_group": best["layer_group"],
                "ppl": best["ppl"],
                "loss": best["loss"],
                "ppl_ratio_vs_baseline": best["ppl_ratio_vs_baseline"],
                "ppl_delta_vs_baseline": best["ppl_delta_vs_baseline"],
                "frontier": [list(group) for group in frontier],
                "candidate_count": len(records),
            }
        )

    summary = {
        "model_name": model_name,
        "basis_size": basis_size,
        "nmf_iterations": nmf_iterations,
        "eval_steps": eval_steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "layer_count": layer_count,
        "max_depth": max_depth,
        "beam_width": beam_width,
        "parallel_workers": parallel_workers,
        "combine_mode": combine_mode,
        "basis_quantization_bits": basis_quantization_bits,
        "basis_quantization_format": basis_quantization_format,
        "basis_quantization_target": basis_quantization_target,
        "baseline": baseline,
        "path": path,
        "records": sorted(all_records, key=lambda row: (len(row.get("layer_group", [])), float(row["ppl"]), tuple(row.get("layer_group", ())))),
    }
    write_h100_attention_basis_search_summary.remote(run_name, json.dumps(summary))
    return summary
