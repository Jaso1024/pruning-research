from __future__ import annotations

import argparse
import json
from pathlib import Path

from saliency.approx import ApproxSaliencyConfig, run_approx_saliency_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute cheap parameter-saliency approximations.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="EleutherAI/pythia-31m")
    parser.add_argument(
        "--method",
        default="wanda",
        choices=[
            "wanda",
            "wanda_masked",
            "proper_wanda",
            "wanda_unmasked",
            "original_wanda",
            "legacy_wanda",
            "ri",
            "ria_no_activation",
            "relative_importance_only",
            "ria",
            "relative_importance",
            "relative_importance_activation",
            "output_l2",
            "output_damage",
            "local_reconstruction",
            "squared_wanda",
            "mean_abs_wanda",
            "wanda_mean_abs",
            "mean_abs",
            "var_output",
            "variance_output",
            "q95_wanda",
            "wanda_q95",
            "outlier_q95",
            "max_wanda",
            "wanda_max",
            "outlier_max",
            "angular",
            "angular_exact",
            "pure_angular",
            "angular_approx",
            "approx_angular",
            "angular_hybrid",
            "hybrid_angular",
            "angular_energy_hybrid",
            "row_wanda",
            "token_wanda",
            "row_conditioned_wanda",
            "feature_cosine_wanda",
            "feature_wanda_cosine",
            "row_wanda_cosine",
            "cosine_feature_wanda",
            "graph_norm",
            "subgraph_norm",
            "residual_norm",
            "graph_qkv",
            "subgraph_qkv",
            "residual_norm_qkv",
            "graph_mlp",
            "subgraph_mlp",
            "residual_norm_mlp",
            "graph_qkv_mlp",
            "graph_next_projections",
            "subgraph_qkv_mlp",
            "graph_vjp_logits",
            "graph_logits",
            "subgraph_logits",
            "hutchinson_logits",
            "local_subgraph_vjp",
            "local_graph_vjp",
            "local_vjp",
            "local_subgraph_vjp_all_tokens",
            "local_graph_vjp_all_tokens",
            "local_vjp_all_tokens",
            "local_forward_wanda",
            "forward_subgraph_wanda",
            "local_wanda_diff",
            "subgraph_wanda_diff",
            "superset_wanda",
            "closed_form_superset_wanda",
            "superset_subgraph_wanda",
            "closed_form_subgraph_wanda",
            "magnitude",
            "weight_magnitude",
            "dfa",
            "dfa_gradcam",
        ],
    )
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "float32", "bf16", "bfloat16", "fp16", "float16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--revision")
    parser.add_argument("--angular-hybrid-lambda", type=float, default=0.5)
    parser.add_argument("--feature-cosine-alpha", type=float, default=0.05)
    parser.add_argument("--feature-cosine-clip", type=float, default=10.0)
    parser.add_argument("--graph-num-probes", type=int, default=4)
    parser.add_argument("--graph-seed", type=int, default=17)
    parser.add_argument("--local-forward-eps", type=float, default=1e-3)
    parser.add_argument("--superset-gain-power", type=float, default=1.0)
    parser.add_argument("--superset-gain-clip-quantile", type=float, default=0.0)
    parser.add_argument("--full-sequence-loss", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ApproxSaliencyConfig(
        output_dir=args.output_dir,
        model_name=args.model_name,
        method=args.method,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
        device=args.device,
        answer_only_loss=not args.full_sequence_loss,
        top_k=args.top_k,
        revision=args.revision,
        angular_hybrid_lambda=args.angular_hybrid_lambda,
        feature_cosine_alpha=args.feature_cosine_alpha,
        feature_cosine_clip=args.feature_cosine_clip,
        graph_num_probes=args.graph_num_probes,
        graph_seed=args.graph_seed,
        local_forward_eps=args.local_forward_eps,
        superset_gain_power=args.superset_gain_power,
        superset_gain_clip_quantile=args.superset_gain_clip_quantile,
    )
    print(json.dumps(run_approx_saliency_experiment(config), indent=2))


if __name__ == "__main__":
    main()
