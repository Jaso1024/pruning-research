# Depth Anything V2 Compression Research

This directory contains the DA-2K Depth Anything V2 compression work. It is intentionally research scaffolding: scripts are meant to preserve the tested methods, knobs, and compact result summaries rather than present one polished package.

## Baseline

- `eval_da2k.py`: DA-2K point-pair evaluator.
- `eval_outputs/groundhog_da2k_vits_25_smoke.json`: small smoke result from the groundhog setup.
- `LOCAL_SETUP.md`: dataset/checkpoint setup and older WANDA/quant/token notes.

## Token Reduction

- `eval_tome_da2k.py`: ToMe-style token merging.
- `eval_token_reduce_methods_da2k.py`: proxy methods for PiToMe, AdaMerge, MCTF, EViT, ATS, and PPT.
- `eval_actual_token_methods_da2k.py`: paper-derived actual-token implementations with dense depth-grid restoration.
- `eval_fixed_token_methods_da2k.py`: fixed-token-budget sweeps.
- `eval_attention_calib_merge_da2k.py`, `fit_ema_attention_da2k.py`, `fit_head_ema_attention_da2k.py`, `fit_head_exp_combo_attention_da2k.py`, `eval_head_exp_combo_da2k.py`: attention/statification approximations and calibrated merge scoring.
- `eval_dense_safe_proxy_da2k.py`, `run_da2k_token_benchmark.py`: benchmark harnesses for running comparable token-reduction variants.

Useful summaries:

- `eval_outputs/fixed_token_methods_summary.md`
- `eval_outputs/da2k_vits_tome_*/*.json`
- `eval_outputs/da2k_vits_attention_*/*.json`

## Activation Replacement

- `eval_gelu_relu_compensation_da2k.py`: GELU-to-ReLU and adapter/LoRA compensation sweeps.
- `eval_gelu_relu_adapter_sweep_da2k.py`: adapter-family sweeps.
- `eval_relu_strikes_da2k.py`, `eval_relu_strikes_state_da2k.py`: ReLU sparsification experiments inspired by relufication.
- `eval_rotated_twopiece_gelu_da2k.py`, `eval_rotated_twopiece_state_da2k.py`: two-piece/rotated GELU replacement trials.
- `eval_int_gelu_activation_da2k.py`: integer/PWL GELU approximation sweep.

Useful summaries:

- `eval_outputs/gelu_relu_adapter_family_summary.md`
- `eval_outputs/gelu_relu_adapter_steps_summary.md`
- `eval_outputs/gelu_relu_lora_identity_summary.md`
- `eval_outputs/gelu_relu_lora_optimizer_summary.md`
- `eval_outputs/gelu_relu_lora_placement_summary.md`
- `eval_outputs/gelu_relu_lora_relex_summary.md`
- `eval_outputs/int_gelu_activation_summary.md`

## Circuit Discovery, Ablation, And Circuit-Aware Pruning

- `eval_attribution_patching_da2k.py`, `aggregate_attribution_patching_chunks.py`: attribution-patching circuit discovery.
- `eval_circuit_ablation_da2k.py`, `eval_component_ablation_da2k.py`, `beam_component_ablation_da2k.py`: component/circuit ablation sweeps.
- `eval_subcircuit_ablation_da2k.py`, `summarize_subcircuit_ablation.py`: finer subcircuit ablation summaries.
- `eval_scalar_weight_circuit_da2k.py`: scalar-level circuit saliency probes.
- `eval_weight_masked_subcircuit_pruning_da2k.py`: circuit-derived weight masks and value-budget pruning.
- `eval_structured_subcircuit_pruning_metrics_da2k.py`: structured metric sweeps, including circuit-aware Wanda-style variants.
- `eval_wanda_unstructured_da2k.py`, `beam_sparse24_da2k.py`, `eval_quant_da2k.py`: WANDA, 2:4, and quantization comparison baselines.

Useful summaries:

- `eval_outputs/comparison_baselines/*.md`
- `eval_outputs/pruning_granularity_audit.md`
- `eval_outputs/structured_metric_sweep_summary.md`
- `eval_outputs/attrib_patch_*/summary.md`
- `eval_outputs/subcircuit_*/summary.json`
- `eval_outputs/weight_masked_*/summary.json`
- `eval_outputs/scalar_*_full/summary.json`

## LoRA/PEFT Repair After Pruning

- `eval_circuit_lora_repair_da2k.py`: circuit/component removal followed by LoRA, LoHA, DoRA, IA3, BitFit, and head/placement repair sweeps.
- `eval_scalar_mask_lora_repair_da2k.py`: scalar-mask pruning plus LoRA/LBit repair and fold-back experiments.

Useful summaries:

- `eval_outputs/circuit_lora_compensation_summary.md`
- `eval_outputs/more_params_compensation_summary.md`
- `eval_outputs/peft_repair_sweep_summary.md`
- `eval_outputs/peft_repair_sweep_tables_with_removed_params.md`
- `eval_outputs/circuit_lora_*/summary.md`
- `eval_outputs/peft_*/summary.md`
- `eval_outputs/scalar_mask_lora_*/summary.md`

## MoE / Structured Sparsity

- `eval_moefication_da2k.py`: MoE-style sparse FC2/down-projection experiments.
- `bench_moefication_sparse_fc2.py`: sparse FC2 microbenchmarks.
- `eval_combined_moe_sparsity_da2k.py`: combined sparsity experiments.
- `eval_conv_im2col_sparsity_da2k.py`: im2col-style convolution-to-FFN sparsity tests.
- `eval_outputs/cuda_fastpath_orin/*/summary.json`: Orin W4A4, affine GELU, layernorm, and MLP fast-path benchmark summaries. The raw CUDA/PTX source was not present in the local workspaces during this consolidation.

## What Is Not Committed

Large checkpoints, datasets, raw logs, calibration caches, and full run artifacts remain ignored. The committed `eval_outputs/**/summary.*` files are compact result records only. The abandoned DiffusionGemma scratch work is not imported here.
