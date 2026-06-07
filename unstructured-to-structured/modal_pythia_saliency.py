from __future__ import annotations

import os
from pathlib import Path

import modal


REMOTE_ROOT = "/root/unstructured_to_structured"
RESULTS_ROOT = "/results"

app = modal.App("pythia-parameter-saliency")
volume = modal.Volume.from_name("pythia-parameter-saliency-results", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "datasets", "accelerate", "safetensors", "tqdm")
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


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_pythia_saliency(
    model_name: str = "EleutherAI/pythia-31m",
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: str = "train",
    max_examples: int = 128,
    batch_size: int = 2,
    max_length: int = 512,
    dtype: str = "bf16",
    answer_only_loss: bool = True,
    run_name: str = "pythia31m_gsm8k_saliency",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from saliency.experiment import SaliencyConfig, run_saliency_experiment

    config = SaliencyConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        max_examples=max_examples,
        batch_size=batch_size,
        max_length=max_length,
        dtype=dtype,
        device="cuda",
        answer_only_loss=answer_only_loss,
    )
    summary = run_saliency_experiment(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_pythia_prune_ppl(
    saliency_run_name: str = "pythia31m_gsm8k_full_saliency_20260526",
    model_name: str = "EleutherAI/pythia-31m",
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: str = "train",
    max_examples: int = 0,
    batch_size: int = 32,
    max_length: int = 512,
    dtype: str = "fp32",
    prune_fraction: float = 0.5,
    pruning_scope: str = "per_matrix",
    answer_only_loss: bool = True,
    run_name: str = "pythia31m_gsm8k_prune50_ppl",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from saliency.prune_eval import PruneEvalConfig, run_prune_ppl_experiment

    config = PruneEvalConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        saliency_path=Path(RESULTS_ROOT) / saliency_run_name / "saliency.pt",
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        max_examples=max_examples,
        batch_size=batch_size,
        max_length=max_length,
        dtype=dtype,
        device="cuda",
        answer_only_loss=answer_only_loss,
        prune_fraction=prune_fraction,
        pruning_scope=pruning_scope,
    )
    summary = run_prune_ppl_experiment(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_pythia_approx_saliency(
    model_name: str = "EleutherAI/pythia-31m",
    method: str = "wanda",
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: str = "train",
    eval_split: str = "",
    max_examples: int = 128,
    max_eval_examples: int = 0,
    batch_size: int = 8,
    max_length: int = 512,
    dtype: str = "bf16",
    answer_only_loss: bool = True,
    angular_hybrid_lambda: float = 0.5,
    feature_cosine_alpha: float = 0.05,
    feature_cosine_clip: float = 10.0,
    graph_num_probes: int = 4,
    graph_seed: int = 17,
    local_forward_eps: float = 1e-3,
    superset_gain_power: float = 1.0,
    superset_gain_clip_quantile: float = 0.0,
    run_name: str = "pythia31m_gsm8k_approx_saliency",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from saliency.approx import ApproxSaliencyConfig, run_approx_saliency_experiment

    config = ApproxSaliencyConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        model_name=model_name,
        method=method,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        eval_split=eval_split,
        max_examples=max_examples,
        max_eval_examples=max_eval_examples,
        batch_size=batch_size,
        max_length=max_length,
        dtype=dtype,
        device="cuda",
        answer_only_loss=answer_only_loss,
        angular_hybrid_lambda=angular_hybrid_lambda,
        feature_cosine_alpha=feature_cosine_alpha,
        feature_cosine_clip=feature_cosine_clip,
        graph_num_probes=graph_num_probes,
        graph_seed=graph_seed,
        local_forward_eps=local_forward_eps,
        superset_gain_power=superset_gain_power,
        superset_gain_clip_quantile=superset_gain_clip_quantile,
    )
    summary = run_approx_saliency_experiment(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_pythia_iterative_approx_prune_ppl(
    model_name: str = "EleutherAI/pythia-31m",
    method: str = "wanda",
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: str = "train",
    eval_split: str = "",
    max_examples: int = 128,
    max_eval_examples: int = 0,
    batch_size: int = 8,
    max_length: int = 512,
    dtype: str = "bf16",
    prune_fraction: float = 0.25,
    prune_chunk_fraction: float = 0.05,
    recompute_every_weights: int = 0,
    pruning_structure: str = "unstructured",
    structured_n: int = 2,
    structured_m: int = 4,
    structured_group_dim: int = 1,
    repair_with_gptq_gd: bool = False,
    repair_with_loss_gd: bool = False,
    repair_learning_rate: float = 1e-5,
    answer_only_loss: bool = True,
    run_name: str = "pythia31m_gsm8k_iterative_wanda_prune_ppl",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from saliency.approx import IterativeApproxPruneConfig, run_iterative_approx_prune_ppl_experiment

    config = IterativeApproxPruneConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        model_name=model_name,
        method=method,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        eval_split=eval_split,
        max_examples=max_examples,
        max_eval_examples=max_eval_examples,
        batch_size=batch_size,
        max_length=max_length,
        dtype=dtype,
        device="cuda",
        answer_only_loss=answer_only_loss,
        prune_fraction=prune_fraction,
        prune_chunk_fraction=prune_chunk_fraction,
        recompute_every_weights=recompute_every_weights,
        pruning_structure=pruning_structure,
        structured_n=structured_n,
        structured_m=structured_m,
        structured_group_dim=structured_group_dim,
        repair_with_gptq_gd=repair_with_gptq_gd,
        repair_with_loss_gd=repair_with_loss_gd,
        repair_learning_rate=repair_learning_rate,
    )
    summary = run_iterative_approx_prune_ppl_experiment(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_pythia_wanda_ablation_prune_ppl(
    model_name: str = "EleutherAI/pythia-31m",
    wanda_method: str = "wanda",
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: str = "train",
    eval_split: str = "",
    max_examples: int = 128,
    max_eval_examples: int = 0,
    batch_size: int = 8,
    max_length: int = 512,
    dtype: str = "bf16",
    prune_fraction: float = 0.25,
    pruning_scope: str = "per_matrix",
    wanda_activation: str = "masked",
    wanda_schedule: str = "one_shot",
    answer_only_loss: bool = True,
    superset_gain_power: float = 1.0,
    superset_gain_clip_quantile: float = 0.0,
    run_name: str = "pythia31m_gsm8k_wanda_ablation_prune_ppl",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from saliency.approx import WandaAblationPruneConfig, run_wanda_ablation_prune_ppl_experiment

    config = WandaAblationPruneConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        eval_split=eval_split,
        max_examples=max_examples,
        max_eval_examples=max_eval_examples,
        batch_size=batch_size,
        max_length=max_length,
        dtype=dtype,
        device="cuda",
        answer_only_loss=answer_only_loss,
        prune_fraction=prune_fraction,
        pruning_scope=pruning_scope,
        wanda_activation=wanda_activation,
        wanda_schedule=wanda_schedule,
        wanda_method=wanda_method,
        superset_gain_power=superset_gain_power,
        superset_gain_clip_quantile=superset_gain_clip_quantile,
    )
    summary = run_wanda_ablation_prune_ppl_experiment(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_pythia_nm_matrix_attribution(
    model_name: str = "EleutherAI/pythia-31m",
    method: str = "wanda",
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: str = "train",
    eval_split: str = "",
    max_examples: int = 128,
    max_eval_examples: int = 0,
    batch_size: int = 8,
    max_length: int = 512,
    dtype: str = "bf16",
    pruning_structure: str = "2:4",
    structured_n: int = 2,
    structured_m: int = 4,
    structured_group_dim: int = 1,
    matrix_limit: int = 0,
    repair_with_loss_gd: bool = True,
    repair_learning_rate: float = 1e-5,
    answer_only_loss: bool = True,
    run_name: str = "pythia31m_gsm8k_nm_matrix_attribution",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from saliency.approx import IterativeApproxPruneConfig, run_nm_matrix_attribution_experiment

    config = IterativeApproxPruneConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        model_name=model_name,
        method=method,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        eval_split=eval_split,
        max_examples=max_examples,
        max_eval_examples=max_eval_examples,
        batch_size=batch_size,
        max_length=max_length,
        dtype=dtype,
        device="cuda",
        answer_only_loss=answer_only_loss,
        pruning_structure=pruning_structure,
        structured_n=structured_n,
        structured_m=structured_m,
        structured_group_dim=structured_group_dim,
        matrix_limit=matrix_limit,
        repair_with_loss_gd=repair_with_loss_gd,
        repair_learning_rate=repair_learning_rate,
    )
    summary = run_nm_matrix_attribution_experiment(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_pythia_nm_global_pass_matrix_attribution(
    model_name: str = "EleutherAI/pythia-31m",
    method: str = "wanda",
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: str = "train",
    eval_split: str = "",
    max_examples: int = 128,
    max_eval_examples: int = 0,
    batch_size: int = 8,
    max_length: int = 512,
    dtype: str = "bf16",
    pruning_structure: str = "2:4",
    structured_n: int = 2,
    structured_m: int = 4,
    structured_group_dim: int = 1,
    matrix_limit: int = 0,
    repair_with_loss_gd: bool = True,
    repair_learning_rate: float = 1e-5,
    answer_only_loss: bool = True,
    run_name: str = "pythia31m_gsm8k_nm_global_pass_matrix_attribution",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from saliency.approx import IterativeApproxPruneConfig, run_nm_global_pass_matrix_attribution_experiment

    config = IterativeApproxPruneConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        model_name=model_name,
        method=method,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        eval_split=eval_split,
        max_examples=max_examples,
        max_eval_examples=max_eval_examples,
        batch_size=batch_size,
        max_length=max_length,
        dtype=dtype,
        device="cuda",
        answer_only_loss=answer_only_loss,
        pruning_structure=pruning_structure,
        structured_n=structured_n,
        structured_m=structured_m,
        structured_group_dim=structured_group_dim,
        matrix_limit=matrix_limit,
        repair_with_loss_gd=repair_with_loss_gd,
        repair_learning_rate=repair_learning_rate,
    )
    summary = run_nm_global_pass_matrix_attribution_experiment(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_pythia_gptq_fp8(
    model_name: str = "EleutherAI/pythia-31m",
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    calibration_split: str = "train",
    eval_split: str = "test",
    max_calibration_examples: int = 0,
    max_eval_examples: int = 0,
    calibration_batch_size: int = 32,
    eval_batch_size: int = 32,
    max_length: int = 512,
    dtype: str = "fp32",
    damp_percent: float = 0.01,
    blocksize: int = 128,
    gptq_steps: int = 1,
    eval_steps: str = "",
    staged_to_wq: bool = False,
    iterative_damped_gptq: bool = False,
    gradient_descent_gptq: bool = False,
    newton_step_alpha: float = 0.0,
    gradient_step_scale: float = 1.0,
    gradient_step_scales: str = "",
    hessian_approximation: str = "full",
    answer_only_loss: bool = True,
    run_name: str = "pythia31m_gsm8k_gptq_fp8",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from saliency.gptq_eval import GPTQConfig, run_gptq_fp8_experiment

    parsed_eval_steps = tuple(int(part.strip()) for part in eval_steps.split(",") if part.strip()) or None
    parsed_gradient_step_scales = tuple(float(part.strip()) for part in gradient_step_scales.split(",") if part.strip()) or None
    config = GPTQConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        calibration_split=calibration_split,
        eval_split=eval_split,
        max_calibration_examples=max_calibration_examples,
        max_eval_examples=max_eval_examples,
        calibration_batch_size=calibration_batch_size,
        eval_batch_size=eval_batch_size,
        max_length=max_length,
        dtype=dtype,
        device="cuda",
        answer_only_loss=answer_only_loss,
        damp_percent=damp_percent,
        blocksize=blocksize,
        gptq_steps=gptq_steps,
        eval_steps=parsed_eval_steps,
        staged_to_wq=staged_to_wq,
        iterative_damped_gptq=iterative_damped_gptq,
        gradient_descent_gptq=gradient_descent_gptq,
        newton_step_alpha=newton_step_alpha or None,
        gradient_step_scale=gradient_step_scale,
        gradient_step_scales=parsed_gradient_step_scales,
        hessian_approximation=hessian_approximation,
    )
    summary = run_gptq_fp8_experiment(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_pythia_qronos_weight_only_ppl(
    model_name: str = "EleutherAI/pythia-31m",
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: str = "train",
    eval_split: str = "",
    max_examples: int = 128,
    max_eval_examples: int = 0,
    batch_size: int = 8,
    max_length: int = 512,
    dtype: str = "bf16",
    weight_bits: int = 4,
    beta: float = 1.0,
    percdamp: float = 1e-6,
    cholesky_scale: float = 1e4,
    num_blocks: int = 100,
    use_activation_order: bool = True,
    use_attention_mask: bool = True,
    quantize_last_layer: bool = False,
    answer_only_loss: bool = True,
    run_name: str = "pythia31m_gsm8k_qronos_weight_only",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from saliency.qronos_eval import QronosConfig, run_qronos_weight_only_experiment

    config = QronosConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        eval_split=eval_split,
        max_examples=max_examples,
        max_eval_examples=max_eval_examples,
        batch_size=batch_size,
        max_length=max_length,
        dtype=dtype,
        device="cuda",
        answer_only_loss=answer_only_loss,
        weight_bits=weight_bits,
        beta=beta,
        percdamp=percdamp,
        cholesky_scale=cholesky_scale,
        num_blocks=num_blocks,
        use_activation_order=use_activation_order,
        use_attention_mask=use_attention_mask,
        quantize_last_layer=quantize_last_layer,
    )
    summary = run_qronos_weight_only_experiment(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_pythia_qronos_prune_ppl(
    model_name: str = "EleutherAI/pythia-31m",
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: str = "train",
    eval_split: str = "",
    max_examples: int = 128,
    max_eval_examples: int = 0,
    batch_size: int = 8,
    max_length: int = 512,
    dtype: str = "bf16",
    prune_fraction: float = 0.25,
    pruning_scope: str = "per_output_row",
    percdamp: float = 1e-6,
    cholesky_scale: float = 1e4,
    num_blocks: int = 100,
    use_activation_order: bool = True,
    use_attention_mask: bool = True,
    quantize_last_layer: bool = False,
    answer_only_loss: bool = True,
    run_name: str = "pythia31m_gsm8k_qronos_prune",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from saliency.qronos_eval import QronosConfig, run_qronos_prune_experiment

    config = QronosConfig(
        output_dir=Path(RESULTS_ROOT) / run_name,
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        eval_split=eval_split,
        max_examples=max_examples,
        max_eval_examples=max_eval_examples,
        batch_size=batch_size,
        max_length=max_length,
        dtype=dtype,
        device="cuda",
        answer_only_loss=answer_only_loss,
        percdamp=percdamp,
        cholesky_scale=cholesky_scale,
        num_blocks=num_blocks,
        use_activation_order=use_activation_order,
        use_attention_mask=use_attention_mask,
        quantize_last_layer=quantize_last_layer,
        prune_fraction=prune_fraction,
        pruning_scope=pruning_scope,
    )
    summary = run_qronos_prune_experiment(config)
    volume.commit()
    return summary


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={RESULTS_ROOT: volume},
)
def run_pythia_2to4_check(
    saliency_run_name: str,
    prune_fraction: float = 0.25,
    group_dim: int = 1,
    run_name: str = "pythia_2to4_check",
) -> dict:
    import json
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    import torch

    from saliency.prune_eval import load_saliency_scores, lowest_saliency_mask, structured_2to4_stats

    saliency_path = Path(RESULTS_ROOT) / saliency_run_name / "saliency.pt"
    scores = load_saliency_scores(saliency_path)
    rows = []
    for name, score in scores.items():
        if score.ndim != 2:
            continue
        mask = lowest_saliency_mask(score, fraction=prune_fraction)
        rows.append({"name": name, "shape": list(score.shape), **structured_2to4_stats(mask, group_dim=group_dim)})

    rows.sort(key=lambda row: (not row["already_2to4"], row["extra_zeros_needed"], -row["compliant_group_fraction"]))
    already = [row for row in rows if row["already_2to4"]]
    summary = {
        "metadata": {
            "saliency_path": str(saliency_path),
            "saliency_run_name": saliency_run_name,
            "prune_fraction": prune_fraction,
            "group_dim": group_dim,
            "definition": "2:4 compliant means every contiguous group of 4 along group_dim has at least 2 zeros",
        },
        "matrix_count": len(rows),
        "already_2to4_count": len(already),
        "already_2to4_names": [row["name"] for row in already],
        "best_by_extra_zeros": rows[:20],
        "aggregate": {
            "groups": int(sum(row["groups"] for row in rows)),
            "compliant_groups": int(sum(row["compliant_groups"] for row in rows)),
            "existing_zeros": int(sum(row["existing_zeros"] for row in rows)),
            "extra_zeros_needed": int(sum(row["extra_zeros_needed"] for row in rows)),
            "target_total_zeros": int(sum(row["target_total_zeros"] for row in rows)),
            "weights": int(sum(torch.tensor(row["shape"]).prod().item() for row in rows)),
        },
    }
    agg = summary["aggregate"]
    agg["compliant_group_fraction"] = agg["compliant_groups"] / max(agg["groups"], 1)
    agg["existing_zero_fraction"] = agg["existing_zeros"] / max(agg["weights"], 1)
    agg["target_zero_fraction"] = agg["target_total_zeros"] / max(agg["weights"], 1)

    out = Path(RESULTS_ROOT) / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "matrix_2to4_stats.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=3 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
    secrets=secrets,
)
def run_pythia_saliency_diagnostics(
    model_name: str,
    saliency_run_names: str = "",
    prune_fraction: float = 0.25,
    pruning_scope: str = "per_output_row",
    dtype: str = "bf16",
    run_name: str = "pythia_saliency_diagnostics",
) -> dict:
    import json
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    import torch

    from saliency.approx import _prepare_model
    from saliency.diagnostics import score_tensor_summary, tensor_weight_summary
    from saliency.experiment import resolve_torch_dtype
    from saliency.prune_eval import load_saliency_scores

    model = _prepare_model(model_name, None, resolve_torch_dtype(dtype), torch.device("cuda"))
    named_params = dict(model.named_parameters())
    weight_rows = []
    for name, param in named_params.items():
        if param.requires_grad and param.ndim == 2:
            weight_rows.append(
                tensor_weight_summary(
                    name,
                    param.detach().to(device="cpu", dtype=torch.float32),
                    bottom_fraction=prune_fraction,
                )
            )

    score_summaries: dict[str, object] = {}
    run_names = [part.strip() for part in saliency_run_names.split(",") if part.strip()]
    for saliency_name in run_names:
        scores = load_saliency_scores(Path(RESULTS_ROOT) / saliency_name / "saliency.pt")
        rows = []
        for name, score in scores.items():
            param = named_params.get(name)
            if (
                param is None
                or not param.requires_grad
                or param.ndim != 2
                or score.ndim != 2
                or tuple(score.shape) != tuple(param.shape)
            ):
                continue
            rows.append(
                score_tensor_summary(
                    name,
                    score,
                    param.detach().to(device="cpu", dtype=torch.float32),
                    prune_fraction=prune_fraction,
                    pruning_scope=pruning_scope,
                )
            )
        del scores
        total = sum(int(row["numel"]) for row in rows)
        score_summaries[saliency_name] = {
            "matrix_count": len(rows),
            "weights": total,
            "weighted_spearman_score_abs_weight": sum(
                float(row["spearman_score_abs_weight"]) * int(row["numel"]) for row in rows
            )
            / max(total, 1),
            "weighted_pruned_l2_weight_fraction": sum(
                float(row["score_pruned_l2_weight_fraction"]) * int(row["numel"]) for row in rows
            )
            / max(total, 1),
            "rows": rows,
        }

    total_weights = sum(int(row["numel"]) for row in weight_rows)
    summary = {
        "metadata": {
            "model_name": model_name,
            "saliency_run_names": run_names,
            "prune_fraction": prune_fraction,
            "pruning_scope": pruning_scope,
            "dtype": dtype,
            "output_dir": str(Path(RESULTS_ROOT) / run_name),
        },
        "weight_summary": {
            "matrix_count": len(weight_rows),
            "weights": total_weights,
            "weighted_bottom_abs_l2_fraction": sum(
                float(row["bottom_abs_l2_fraction"]) * int(row["numel"]) for row in weight_rows
            )
            / max(total_weights, 1),
            "weighted_top_abs_l2_fraction": sum(
                float(row["top_abs_l2_fraction"]) * int(row["numel"]) for row in weight_rows
            )
            / max(total_weights, 1),
            "weighted_abs_mean": sum(float(row["abs_mean"]) * int(row["numel"]) for row in weight_rows)
            / max(total_weights, 1),
            "rows": weight_rows,
        },
        "score_summaries": score_summaries,
        "pair_summaries": {},
    }

    out = Path(RESULTS_ROOT) / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "weight_stats.jsonl").open("w") as handle:
        for row in weight_rows:
            handle.write(json.dumps(row) + "\n")
    for saliency_name, score_summary in score_summaries.items():
        safe = saliency_name.replace("/", "__")
        with (out / f"{safe}_score_stats.jsonl").open("w") as handle:
            for row in score_summary["rows"]:
                handle.write(json.dumps(row) + "\n")
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
)
def run_affine_classification_prune_grid(
    num_pairs: int = 8192,
    input_dim: int = 4096,
    output_dim: int = 4096,
    seed: int = 0,
    dtype: str = "float32",
    solver: str = "auto",
    ridge: float = 0.0,
    prune_fractions: str = "0.05,0.10,0.25,0.50",
    pruning_scope: str = "global",
    methods: str = "random,magnitude,wanda,squared_wanda,exact_weight_loss,exact_grad,gptq,qronos",
    damp_percent: float = 0.01,
    blocksize: int = 128,
    percdamp: float = 1e-6,
    cholesky_scale: float = 1e4,
    num_blocks: int = 100,
    use_activation_order: bool = True,
    run_name: str = "affine_classification_4096_prune_grid",
) -> dict:
    import json
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from saliency.affine_scaffold import AffinePruneEvalConfig, AffineScaffoldConfig, run_affine_prune_eval, run_affine_scaffold

    def csv_floats(value: str) -> tuple[float, ...]:
        return tuple(float(part.strip()) for part in value.split(",") if part.strip())

    def csv_strings(value: str) -> tuple[str, ...]:
        return tuple(part.strip() for part in value.split(",") if part.strip())

    root = Path(RESULTS_ROOT) / run_name
    scaffold_summary = run_affine_scaffold(
        AffineScaffoldConfig(
            output_dir=root / "scaffold",
            num_pairs=num_pairs,
            input_dim=input_dim,
            output_dim=output_dim,
            seed=seed,
            dtype=dtype,
            task="classification",
            solver=solver,
            ridge=ridge,
            device="cuda",
        )
    )
    prune_summary = run_affine_prune_eval(
        AffinePruneEvalConfig(
            input_dir=root / "scaffold",
            output_dir=root / "prune_grid",
            methods=csv_strings(methods),
            prune_fractions=csv_floats(prune_fractions),
            pruning_scope=pruning_scope,
            seed=seed + 123,
            device="cuda",
            damp_percent=damp_percent,
            blocksize=blocksize,
            percdamp=percdamp,
            cholesky_scale=cholesky_scale,
            num_blocks=num_blocks,
            use_activation_order=use_activation_order,
        )
    )
    summary = {
        "metadata": {
            "run_name": run_name,
            "output_dir": str(root),
            "num_pairs": num_pairs,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "seed": seed,
            "dtype": dtype,
            "solver": solver,
            "ridge": ridge,
            "prune_fractions": csv_floats(prune_fractions),
            "pruning_scope": pruning_scope,
            "methods": csv_strings(methods),
            "damp_percent": damp_percent,
            "blocksize": blocksize,
            "percdamp": percdamp,
            "cholesky_scale": cholesky_scale,
            "num_blocks": num_blocks,
            "use_activation_order": use_activation_order,
        },
        "scaffold": scaffold_summary,
        "prune": prune_summary,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={RESULTS_ROOT: volume},
)
def run_layered_affine_suite_grid(
    specs: str = "square2048:4096:2048,2048;wide_out:8192:1024,4096;narrow_out:8192:4096,1024;two_wide:4096:2048,4096,2048;two_bottleneck:4096:2048,512,2048;three_layer:4096:1024,2048,2048,1024",
    seed: int = 17,
    dtype: str = "float32",
    solver: str = "auto",
    ridge: float = 0.0,
    prune_fractions: str = "0.05,0.10,0.25,0.50",
    pruning_scope: str = "global",
    methods: str = "random,magnitude,wanda,squared_wanda,exact_weight_loss,exact_grad,gptq,qronos",
    damp_percent: float = 0.01,
    blocksize: int = 128,
    percdamp: float = 1e-6,
    cholesky_scale: float = 1e4,
    num_blocks: int = 100,
    use_activation_order: bool = True,
    run_name: str = "layered_affine_toy_suite",
) -> dict:
    import os
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)

    from saliency.affine_scaffold import run_layered_affine_suite

    def csv_floats(value: str) -> tuple[float, ...]:
        return tuple(float(part.strip()) for part in value.split(",") if part.strip())

    def csv_strings(value: str) -> tuple[str, ...]:
        return tuple(part.strip() for part in value.split(",") if part.strip())

    root = Path(RESULTS_ROOT) / run_name
    summary = run_layered_affine_suite(
        output_dir=root,
        specs=specs,
        methods=csv_strings(methods),
        prune_fractions=csv_floats(prune_fractions),
        pruning_scope=pruning_scope,
        seed=seed,
        dtype=dtype,
        solver=solver,
        ridge=ridge,
        device="cuda",
        damp_percent=damp_percent,
        blocksize=blocksize,
        percdamp=percdamp,
        cholesky_scale=cholesky_scale,
        num_blocks=num_blocks,
        use_activation_order=use_activation_order,
    )
    volume.commit()
    return summary


@app.local_entrypoint()
def main(
    model_name: str = "EleutherAI/pythia-31m",
    max_examples: int = 128,
    batch_size: int = 2,
    max_length: int = 512,
    dtype: str = "bf16",
    run_name: str = "pythia31m_gsm8k_saliency",
    mode: str = "saliency",
    approx_method: str = "wanda",
    saliency_run_name: str = "pythia31m_gsm8k_full_saliency_20260526",
    saliency_run_names: str = "",
    eval_split: str = "",
    prune_fraction: float = 0.5,
    prune_chunk_fraction: float = 0.05,
    recompute_every_weights: int = 0,
    pruning_structure: str = "unstructured",
    structured_n: int = 2,
    structured_m: int = 4,
    structured_group_dim: int = 1,
    matrix_limit: int = 0,
    repair_with_gptq_gd: bool = False,
    repair_with_loss_gd: bool = False,
    repair_learning_rate: float = 1e-5,
    pruning_scope: str = "per_matrix",
    max_calibration_examples: int = 0,
    max_eval_examples: int = 0,
    calibration_batch_size: int = 32,
    eval_batch_size: int = 32,
    damp_percent: float = 0.01,
    blocksize: int = 128,
    gptq_steps: int = 1,
    eval_steps: str = "",
    staged_to_wq: bool = False,
    iterative_damped_gptq: bool = False,
    gradient_descent_gptq: bool = False,
    newton_step_alpha: float = 0.0,
    gradient_step_scale: float = 1.0,
    gradient_step_scales: str = "",
    hessian_approximation: str = "full",
    angular_hybrid_lambda: float = 0.5,
    feature_cosine_alpha: float = 0.05,
    feature_cosine_clip: float = 10.0,
    graph_num_probes: int = 4,
    graph_seed: int = 17,
    local_forward_eps: float = 1e-3,
    superset_gain_power: float = 1.0,
    superset_gain_clip_quantile: float = 0.0,
    wanda_activation: str = "masked",
    wanda_schedule: str = "one_shot",
    wanda_method: str = "",
    weight_bits: int = 4,
    beta: float = 1.0,
    percdamp: float = 1e-6,
    cholesky_scale: float = 1e4,
    num_blocks: int = 100,
    use_activation_order: bool = True,
    use_attention_mask: bool = True,
    quantize_last_layer: bool = False,
    answer_only_loss: bool = True,
    group_dim: int = 1,
    affine_num_pairs: int = 8192,
    affine_input_dim: int = 4096,
    affine_output_dim: int = 4096,
    affine_solver: str = "auto",
    affine_ridge: float = 0.0,
    affine_methods: str = "random,magnitude,wanda,squared_wanda,exact_weight_loss,exact_grad,gptq,qronos",
    affine_prune_fractions: str = "0.05,0.10,0.25,0.50",
    affine_suite_specs: str = "square2048:4096:2048,2048;wide_out:8192:1024,4096;narrow_out:8192:4096,1024;two_wide:4096:2048,4096,2048;two_bottleneck:4096:2048,512,2048;three_layer:4096:1024,2048,2048,1024",
) -> None:
    import json

    if mode == "saliency":
        summary = run_pythia_saliency.remote(
            model_name=model_name,
            max_examples=max_examples,
            batch_size=batch_size,
            max_length=max_length,
            dtype=dtype,
            run_name=run_name,
        )
    elif mode == "approx-saliency":
        summary = run_pythia_approx_saliency.remote(
            model_name=model_name,
            method=approx_method,
            max_examples=max_examples,
            eval_split=eval_split,
            max_eval_examples=max_eval_examples,
            batch_size=batch_size,
            max_length=max_length,
            dtype=dtype,
            angular_hybrid_lambda=angular_hybrid_lambda,
            feature_cosine_alpha=feature_cosine_alpha,
            feature_cosine_clip=feature_cosine_clip,
            graph_num_probes=graph_num_probes,
            graph_seed=graph_seed,
            local_forward_eps=local_forward_eps,
            superset_gain_power=superset_gain_power,
            superset_gain_clip_quantile=superset_gain_clip_quantile,
            run_name=run_name,
        )
    elif mode == "iterative-approx-prune-ppl":
        summary = run_pythia_iterative_approx_prune_ppl.remote(
            model_name=model_name,
            method=approx_method,
            max_examples=max_examples,
            eval_split=eval_split,
            max_eval_examples=max_eval_examples,
            batch_size=batch_size,
            max_length=max_length,
            dtype=dtype,
            prune_fraction=prune_fraction,
            prune_chunk_fraction=prune_chunk_fraction,
            recompute_every_weights=recompute_every_weights,
            pruning_structure=pruning_structure,
            structured_n=structured_n,
            structured_m=structured_m,
            structured_group_dim=structured_group_dim,
            repair_with_gptq_gd=repair_with_gptq_gd,
            repair_with_loss_gd=repair_with_loss_gd,
            repair_learning_rate=repair_learning_rate,
            run_name=run_name,
        )
    elif mode == "wanda-ablation-prune-ppl":
        summary = run_pythia_wanda_ablation_prune_ppl.remote(
            model_name=model_name,
            wanda_method=wanda_method or approx_method,
            max_examples=max_examples,
            eval_split=eval_split,
            max_eval_examples=max_eval_examples,
            batch_size=batch_size,
            max_length=max_length,
            dtype=dtype,
            prune_fraction=prune_fraction,
            pruning_scope=pruning_scope,
            wanda_activation=wanda_activation,
            wanda_schedule=wanda_schedule,
            superset_gain_power=superset_gain_power,
            superset_gain_clip_quantile=superset_gain_clip_quantile,
            run_name=run_name,
        )
    elif mode == "nm-matrix-attribution":
        summary = run_pythia_nm_matrix_attribution.remote(
            model_name=model_name,
            method=approx_method,
            max_examples=max_examples,
            eval_split=eval_split,
            max_eval_examples=max_eval_examples,
            batch_size=batch_size,
            max_length=max_length,
            dtype=dtype,
            pruning_structure=pruning_structure,
            structured_n=structured_n,
            structured_m=structured_m,
            structured_group_dim=structured_group_dim,
            matrix_limit=matrix_limit,
            repair_with_loss_gd=repair_with_loss_gd,
            repair_learning_rate=repair_learning_rate,
            run_name=run_name,
        )
    elif mode == "nm-global-pass-matrix-attribution":
        summary = run_pythia_nm_global_pass_matrix_attribution.remote(
            model_name=model_name,
            method=approx_method,
            max_examples=max_examples,
            eval_split=eval_split,
            max_eval_examples=max_eval_examples,
            batch_size=batch_size,
            max_length=max_length,
            dtype=dtype,
            pruning_structure=pruning_structure,
            structured_n=structured_n,
            structured_m=structured_m,
            structured_group_dim=structured_group_dim,
            matrix_limit=matrix_limit,
            repair_with_loss_gd=repair_with_loss_gd,
            repair_learning_rate=repair_learning_rate,
            run_name=run_name,
        )
    elif mode == "prune-ppl":
        summary = run_pythia_prune_ppl.remote(
            saliency_run_name=saliency_run_name,
            model_name=model_name,
            max_examples=max_examples,
            batch_size=batch_size,
            max_length=max_length,
            dtype=dtype,
            prune_fraction=prune_fraction,
            pruning_scope=pruning_scope,
            run_name=run_name,
        )
    elif mode == "saliency-diagnostics":
        summary = run_pythia_saliency_diagnostics.remote(
            model_name=model_name,
            saliency_run_names=saliency_run_names or saliency_run_name,
            prune_fraction=prune_fraction,
            pruning_scope=pruning_scope,
            dtype=dtype,
            run_name=run_name,
        )
    elif mode == "affine-classification-prune-grid":
        summary = run_affine_classification_prune_grid.remote(
            num_pairs=affine_num_pairs,
            input_dim=affine_input_dim,
            output_dim=affine_output_dim,
            seed=graph_seed,
            dtype="float32",
            solver=affine_solver,
            ridge=affine_ridge,
            prune_fractions=affine_prune_fractions,
            pruning_scope=pruning_scope,
            methods=affine_methods,
            damp_percent=damp_percent,
            blocksize=blocksize,
            percdamp=percdamp,
            cholesky_scale=cholesky_scale,
            num_blocks=num_blocks,
            use_activation_order=use_activation_order,
            run_name=run_name,
        )
    elif mode == "layered-affine-suite-grid":
        summary = run_layered_affine_suite_grid.remote(
            specs=affine_suite_specs,
            seed=graph_seed,
            dtype="float32",
            solver=affine_solver,
            ridge=affine_ridge,
            prune_fractions=affine_prune_fractions,
            pruning_scope=pruning_scope,
            methods=affine_methods,
            damp_percent=damp_percent,
            blocksize=blocksize,
            percdamp=percdamp,
            cholesky_scale=cholesky_scale,
            num_blocks=num_blocks,
            use_activation_order=use_activation_order,
            run_name=run_name,
        )
    elif mode == "gptq-fp8":
        summary = run_pythia_gptq_fp8.remote(
            model_name=model_name,
            max_calibration_examples=max_calibration_examples,
            max_eval_examples=max_eval_examples,
            calibration_batch_size=calibration_batch_size,
            eval_batch_size=eval_batch_size,
            max_length=max_length,
            dtype=dtype,
            damp_percent=damp_percent,
            blocksize=blocksize,
            gptq_steps=gptq_steps,
            eval_steps=eval_steps,
            staged_to_wq=staged_to_wq,
            iterative_damped_gptq=iterative_damped_gptq,
            gradient_descent_gptq=gradient_descent_gptq,
            newton_step_alpha=newton_step_alpha,
            gradient_step_scale=gradient_step_scale,
            gradient_step_scales=gradient_step_scales,
            hessian_approximation=hessian_approximation,
            run_name=run_name,
        )
    elif mode == "qronos-weight-only-ppl":
        summary = run_pythia_qronos_weight_only_ppl.remote(
            model_name=model_name,
            split="train",
            eval_split=eval_split,
            max_examples=max_examples,
            max_eval_examples=max_eval_examples,
            batch_size=batch_size,
            max_length=max_length,
            dtype=dtype,
            weight_bits=weight_bits,
            beta=beta,
            percdamp=percdamp,
            cholesky_scale=cholesky_scale,
            num_blocks=num_blocks,
            use_activation_order=use_activation_order,
            use_attention_mask=use_attention_mask,
            quantize_last_layer=quantize_last_layer,
            answer_only_loss=answer_only_loss,
            run_name=run_name,
        )
    elif mode == "qronos-prune-ppl":
        summary = run_pythia_qronos_prune_ppl.remote(
            model_name=model_name,
            split="train",
            eval_split=eval_split,
            max_examples=max_examples,
            max_eval_examples=max_eval_examples,
            batch_size=batch_size,
            max_length=max_length,
            dtype=dtype,
            prune_fraction=prune_fraction,
            pruning_scope=pruning_scope,
            percdamp=percdamp,
            cholesky_scale=cholesky_scale,
            num_blocks=num_blocks,
            use_activation_order=use_activation_order,
            use_attention_mask=use_attention_mask,
            quantize_last_layer=quantize_last_layer,
            answer_only_loss=answer_only_loss,
            run_name=run_name,
        )
    elif mode == "2to4-check":
        summary = run_pythia_2to4_check.remote(
            saliency_run_name=saliency_run_name,
            prune_fraction=prune_fraction,
            group_dim=group_dim,
            run_name=run_name,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    print(json.dumps(summary, indent=2))
