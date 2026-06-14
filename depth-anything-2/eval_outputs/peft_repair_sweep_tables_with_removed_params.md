# PEFT Circuit Repair Tables

Clean-only tables. Excludes train/eval overlap runs, old-schema runs, and old unconstrained masked-site LoRA results.

`removed params` is `masked_tensor_values`: the exact number of weight/bias tensor entries zeroed by the circuit pruning step.
For all weak-n10 full runs this is usually `233856`; smoke tests prune only 2 nodes and remove `36928`.

## All Clean Non-Smoke Runs Ranked By Folded Accuracy

| run | method | placement | modules | rank | train | skip | epochs | PEFT params | removed params | dense | pruned | folded | remasked |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `peft_clean_rerun_prev_attn_lora_r8_t64_e5_weak_n10_evalskip64_eval128` | `lora` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 92160 | 233856 | 0.9288 | 0.8352 | 0.9101 | 0.9101 |
| `peft_focus_lbit_r16_e5_t64_previous-block_attn_weak_n10_eval128` | `lora-bitfit` | `previous-block` | `attn` | 16 | 64 | 64 | 5 | 192000 | 233856 | 0.9288 | 0.8352 | 0.9101 | 0.9101 |
| `peft_grid_method_loha_r8_previous-block_attn_weak_n10_e5_eval128` | `loha` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 184320 | 233856 | 0.9288 | 0.8352 | 0.9101 | 0.9101 |
| `peft_grid_method_lora_r4_previous-block_attn_weak_n10_e5_eval128` | `lora` | `previous-block` | `attn` | 4 | 64 | 64 | 5 | 46080 | 233856 | 0.9288 | 0.8352 | 0.9101 | 0.9101 |
| `peft_grid_method_dora_r8_previous-block_attn_weak_n10_e5_eval128` | `dora` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 99840 | 233856 | 0.9288 | 0.8352 | 0.9064 | 0.9064 |
| `peft_holdout_holdout_t64_e5_previous-block_attn_weak_n10_evalskip128_eval128` | `lora` | `previous-block` | `attn` | 8 | 64 | 128 | 5 | 92160 | 233856 | 0.9450 | 0.8511 | 0.9061 | 0.9061 |
| `peft_focus_lbit_r4_e5_t64_previous-block_attn_weak_n10_eval128` | `lora-bitfit` | `previous-block` | `attn` | 4 | 64 | 64 | 5 | 53760 | 233856 | 0.9288 | 0.8352 | 0.9026 | 0.9026 |
| `peft_focus_lora_r8_e10_t64_previous-block_attn_weak_n10_eval128` | `lora` | `previous-block` | `attn` | 8 | 64 | 64 | 10 | 92160 | 233856 | 0.9288 | 0.8352 | 0.9026 | 0.9026 |
| `peft_grid_method_loha_r4_previous-block_attn_weak_n10_e5_eval128` | `loha` | `previous-block` | `attn` | 4 | 64 | 64 | 5 | 92160 | 233856 | 0.9288 | 0.8352 | 0.9026 | 0.9026 |
| `peft_grid_method_lora_r8_previous-block_attn_weak_n10_e5_eval128` | `lora` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 92160 | 233856 | 0.9288 | 0.8352 | 0.9026 | 0.9026 |
| `peft_focus_lbit_r8_e10_t64_previous-block_attn_weak_n10_eval128` | `lora-bitfit` | `previous-block` | `attn` | 8 | 64 | 64 | 10 | 99840 | 233856 | 0.9288 | 0.8352 | 0.8989 | 0.8989 |
| `peft_grid_place_prev_same_attn_previous-block_plus_same-block_attn_weak_n10_e5_eval128` | `lora` | `previous-block+same-block` | `attn` | 8 | 64 | 64 | 5 | 147456 | 233856 | 0.9288 | 0.8352 | 0.8989 | 0.8989 |
| `peft_holdout_holdout_t128_e8_previous-block_attn_weak_n10_evalskip128_eval128` | `lora` | `previous-block` | `attn` | 8 | 128 | 128 | 8 | 92160 | 233856 | 0.9450 | 0.8511 | 0.8964 | 0.8964 |
| `peft_focus_lbit_r8_e5_t64_previous-block_attn_weak_n10_eval128` | `lora-bitfit` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 99840 | 233856 | 0.9288 | 0.8352 | 0.8951 | 0.8951 |
| `peft_grid_method_bitfit_previous-block_attn_weak_n10_e5_eval128` | `bitfit` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 7680 | 233856 | 0.9288 | 0.8352 | 0.8951 | 0.8951 |
| `peft_grid_place_prev_next_attn_previous-block_plus_next-block_attn_weak_n10_e5_eval128` | `lora` | `previous-block+next-block` | `attn` | 8 | 64 | 64 | 5 | 147456 | 233856 | 0.9288 | 0.8352 | 0.8951 | 0.8951 |
| `peft_holdout_holdout_lbit_t128_e8_previous-block_attn_weak_n10_evalskip128_eval128` | `lora-bitfit` | `previous-block` | `attn` | 8 | 128 | 128 | 8 | 99840 | 233856 | 0.9450 | 0.8511 | 0.8932 | 0.8932 |
| `peft_grid_place_same_mlp_r8_same-block_mlp_weak_n10_e5_eval128` | `lora` | `same-block` | `mlp` | 8 | 64 | 64 | 5 | 153600 | 233856 | 0.9288 | 0.8352 | 0.8914 | 0.8914 |
| `peft_grid_method_ia3out_previous-block_attn_weak_n10_e5_eval128` | `ia3-out` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 7680 | 233856 | 0.9288 | 0.8352 | 0.8876 | 0.8876 |
| `peft_grid_place_next_attn_next-block_attn_weak_n10_e5_eval128` | `lora` | `next-block` | `attn` | 8 | 64 | 64 | 5 | 92160 | 233856 | 0.9288 | 0.8352 | 0.8876 | 0.8876 |
| `peft_clean_rerun_prev_attn_lbit_r8_t128_e8_weak_n10_evalskip128_eval128` | `lora-bitfit` | `previous-block` | `attn` | 8 | 128 | 128 | 8 | 99840 | 233856 | 0.9450 | 0.8511 | 0.8867 | 0.8867 |
| `peft_grid_method_ia3in_previous-block_attn_weak_n10_e5_eval128` | `ia3-in` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 3840 | 233856 | 0.9288 | 0.8352 | 0.8764 | 0.8764 |
| `peft_clean_rerun_prev_attn_lora_r8_t128_e8_weak_n10_evalskip128_eval128` | `lora` | `previous-block` | `attn` | 8 | 128 | 128 | 8 | 92160 | 233856 | 0.9450 | 0.8511 | 0.8738 | 0.8738 |
| `peft_grid_place_prev_same_next_attn_previous-block_plus_same-block_plus_next-block_attn_weak_n10_e5_eval128` | `lora` | `previous-block+same-block+next-block` | `attn` | 8 | 64 | 64 | 5 | 184320 | 233856 | 0.9288 | 0.8352 | 0.8689 | 0.8689 |
| `peft_grid_place_prev_next_all_previous-block_plus_next-block_all_weak_n10_e5_eval128` | `lora` | `previous-block+next-block` | `all` | 8 | 64 | 64 | 5 | 393216 | 233856 | 0.9288 | 0.8352 | 0.8652 | 0.8652 |
| `peft_grid_place_masked_all_r16_masked_all_weak_n10_e5_eval128` | `lora` | `masked` | `all` | 16 | 64 | 64 | 5 | 307200 | 233856 | 0.9288 | 0.8352 | 0.8352 | 0.8352 |
| `peft_clean_rerun_masked_lora_r16_t64_e5_weak_n10_evalskip64_eval128` | `lora` | `masked` | `all` | 16 | 64 | 64 | 5 | 307200 | 233856 | 0.9288 | 0.8352 | 0.8277 | 0.8277 |

## Canonical Method Sweep: previous-block attention, train64/skip64

| run | method | placement | modules | rank | train | skip | epochs | PEFT params | removed params | dense | pruned | folded | remasked |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `peft_clean_rerun_prev_attn_lora_r8_t64_e5_weak_n10_evalskip64_eval128` | `lora` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 92160 | 233856 | 0.9288 | 0.8352 | 0.9101 | 0.9101 |
| `peft_focus_lbit_r16_e5_t64_previous-block_attn_weak_n10_eval128` | `lora-bitfit` | `previous-block` | `attn` | 16 | 64 | 64 | 5 | 192000 | 233856 | 0.9288 | 0.8352 | 0.9101 | 0.9101 |
| `peft_grid_method_loha_r8_previous-block_attn_weak_n10_e5_eval128` | `loha` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 184320 | 233856 | 0.9288 | 0.8352 | 0.9101 | 0.9101 |
| `peft_grid_method_lora_r4_previous-block_attn_weak_n10_e5_eval128` | `lora` | `previous-block` | `attn` | 4 | 64 | 64 | 5 | 46080 | 233856 | 0.9288 | 0.8352 | 0.9101 | 0.9101 |
| `peft_grid_method_dora_r8_previous-block_attn_weak_n10_e5_eval128` | `dora` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 99840 | 233856 | 0.9288 | 0.8352 | 0.9064 | 0.9064 |
| `peft_focus_lbit_r4_e5_t64_previous-block_attn_weak_n10_eval128` | `lora-bitfit` | `previous-block` | `attn` | 4 | 64 | 64 | 5 | 53760 | 233856 | 0.9288 | 0.8352 | 0.9026 | 0.9026 |
| `peft_grid_method_loha_r4_previous-block_attn_weak_n10_e5_eval128` | `loha` | `previous-block` | `attn` | 4 | 64 | 64 | 5 | 92160 | 233856 | 0.9288 | 0.8352 | 0.9026 | 0.9026 |
| `peft_grid_method_lora_r8_previous-block_attn_weak_n10_e5_eval128` | `lora` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 92160 | 233856 | 0.9288 | 0.8352 | 0.9026 | 0.9026 |
| `peft_focus_lbit_r8_e5_t64_previous-block_attn_weak_n10_eval128` | `lora-bitfit` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 99840 | 233856 | 0.9288 | 0.8352 | 0.8951 | 0.8951 |
| `peft_grid_method_bitfit_previous-block_attn_weak_n10_e5_eval128` | `bitfit` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 7680 | 233856 | 0.9288 | 0.8352 | 0.8951 | 0.8951 |
| `peft_grid_method_ia3out_previous-block_attn_weak_n10_e5_eval128` | `ia3-out` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 7680 | 233856 | 0.9288 | 0.8352 | 0.8876 | 0.8876 |
| `peft_grid_method_ia3in_previous-block_attn_weak_n10_e5_eval128` | `ia3-in` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 3840 | 233856 | 0.9288 | 0.8352 | 0.8764 | 0.8764 |

## LoRA Placement Sweep: train64/skip64

| run | method | placement | modules | rank | train | skip | epochs | PEFT params | removed params | dense | pruned | folded | remasked |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `peft_clean_rerun_prev_attn_lora_r8_t64_e5_weak_n10_evalskip64_eval128` | `lora` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 92160 | 233856 | 0.9288 | 0.8352 | 0.9101 | 0.9101 |
| `peft_grid_method_lora_r4_previous-block_attn_weak_n10_e5_eval128` | `lora` | `previous-block` | `attn` | 4 | 64 | 64 | 5 | 46080 | 233856 | 0.9288 | 0.8352 | 0.9101 | 0.9101 |
| `peft_grid_method_lora_r8_previous-block_attn_weak_n10_e5_eval128` | `lora` | `previous-block` | `attn` | 8 | 64 | 64 | 5 | 92160 | 233856 | 0.9288 | 0.8352 | 0.9026 | 0.9026 |
| `peft_grid_place_prev_same_attn_previous-block_plus_same-block_attn_weak_n10_e5_eval128` | `lora` | `previous-block+same-block` | `attn` | 8 | 64 | 64 | 5 | 147456 | 233856 | 0.9288 | 0.8352 | 0.8989 | 0.8989 |
| `peft_grid_place_prev_next_attn_previous-block_plus_next-block_attn_weak_n10_e5_eval128` | `lora` | `previous-block+next-block` | `attn` | 8 | 64 | 64 | 5 | 147456 | 233856 | 0.9288 | 0.8352 | 0.8951 | 0.8951 |
| `peft_grid_place_same_mlp_r8_same-block_mlp_weak_n10_e5_eval128` | `lora` | `same-block` | `mlp` | 8 | 64 | 64 | 5 | 153600 | 233856 | 0.9288 | 0.8352 | 0.8914 | 0.8914 |
| `peft_grid_place_next_attn_next-block_attn_weak_n10_e5_eval128` | `lora` | `next-block` | `attn` | 8 | 64 | 64 | 5 | 92160 | 233856 | 0.9288 | 0.8352 | 0.8876 | 0.8876 |
| `peft_grid_place_prev_same_next_attn_previous-block_plus_same-block_plus_next-block_attn_weak_n10_e5_eval128` | `lora` | `previous-block+same-block+next-block` | `attn` | 8 | 64 | 64 | 5 | 184320 | 233856 | 0.9288 | 0.8352 | 0.8689 | 0.8689 |
| `peft_grid_place_prev_next_all_previous-block_plus_next-block_all_weak_n10_e5_eval128` | `lora` | `previous-block+next-block` | `all` | 8 | 64 | 64 | 5 | 393216 | 233856 | 0.9288 | 0.8352 | 0.8652 | 0.8652 |
| `peft_grid_place_masked_all_r16_masked_all_weak_n10_e5_eval128` | `lora` | `masked` | `all` | 16 | 64 | 64 | 5 | 307200 | 233856 | 0.9288 | 0.8352 | 0.8352 | 0.8352 |
| `peft_clean_rerun_masked_lora_r16_t64_e5_weak_n10_evalskip64_eval128` | `lora` | `masked` | `all` | 16 | 64 | 64 | 5 | 307200 | 233856 | 0.9288 | 0.8352 | 0.8277 | 0.8277 |

## Clean Holdout Slice: eval_skip=128

| run | method | placement | modules | rank | train | skip | epochs | PEFT params | removed params | dense | pruned | folded | remasked |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `peft_holdout_holdout_t64_e5_previous-block_attn_weak_n10_evalskip128_eval128` | `lora` | `previous-block` | `attn` | 8 | 64 | 128 | 5 | 92160 | 233856 | 0.9450 | 0.8511 | 0.9061 | 0.9061 |
| `peft_holdout_holdout_t128_e8_previous-block_attn_weak_n10_evalskip128_eval128` | `lora` | `previous-block` | `attn` | 8 | 128 | 128 | 8 | 92160 | 233856 | 0.9450 | 0.8511 | 0.8964 | 0.8964 |
| `peft_holdout_holdout_lbit_t128_e8_previous-block_attn_weak_n10_evalskip128_eval128` | `lora-bitfit` | `previous-block` | `attn` | 8 | 128 | 128 | 8 | 99840 | 233856 | 0.9450 | 0.8511 | 0.8932 | 0.8932 |
| `peft_clean_rerun_prev_attn_lbit_r8_t128_e8_weak_n10_evalskip128_eval128` | `lora-bitfit` | `previous-block` | `attn` | 8 | 128 | 128 | 8 | 99840 | 233856 | 0.9450 | 0.8511 | 0.8867 | 0.8867 |
| `peft_clean_rerun_prev_attn_lora_r8_t128_e8_weak_n10_evalskip128_eval128` | `lora` | `previous-block` | `attn` | 8 | 128 | 128 | 8 | 92160 | 233856 | 0.9450 | 0.8511 | 0.8738 | 0.8738 |

## Smoke Tests

| run | method | placement | modules | rank | train | skip | epochs | PEFT params | removed params | dense | pruned | folded | remasked |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `peft_smoke_combo_lora` | `lora` | `previous-block+next-block` | `attn` | 2 | 2 | 2 | 1 | 13824 | 36928 | 0.8333 | 0.8333 | 0.8333 | 0.8333 |
| `peft_smoke_dora` | `dora` | `previous-block` | `attn` | 2 | 2 | 2 | 1 | 12288 | 36928 | 0.8333 | 0.8333 | 0.8333 | 0.8333 |
| `peft_smoke_lora_bitfit` | `lora-bitfit` | `previous-block` | `attn` | 2 | 2 | 2 | 1 | 12288 | 36928 | 0.8333 | 0.8333 | 0.8333 | 0.8333 |

## Skipped / Excluded Runs

| output | reason |
| --- | --- |
| `eval_outputs/circuit_lora_focus_all-prior_w1_attn_weak_n10_r16_a32_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_focus_around-window_w1_attn_weak_n10_r16_a32_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_focus_previous-block_w1_attn_weak_n10_r32_a64_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_focus_previous-block_w1_attn_weak_n10_r4_a8_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_focus_previous-block_w1_attn_weak_n10_r8_a16_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_focus_previous-block_w1_attn_weak_n10_r8_a16_e5_pair0p02_eval128` | old schema |
| `eval_outputs/circuit_lora_focus_previous-block_w1_attn_weak_n10_r8_a16_e5_pair0p0_eval128` | old schema |
| `eval_outputs/circuit_lora_focus_previous-block_w1_attn_weak_n10_r8_a16_e5_pair0p1_eval128` | old schema |
| `eval_outputs/circuit_lora_focus_previous-block_w1_attn_weak_n10_r8_a16_e5_pair0p2_eval128` | old schema |
| `eval_outputs/circuit_lora_focus_previous-window_w2_attn_weak_n10_r16_a32_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_masked_constrained_weak_n10_r16_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_pairwise_smoke` | old schema |
| `eval_outputs/circuit_lora_place_all-prior_w1_all_weak_n10_r16_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_place_around-window_w1_all_weak_n10_r16_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_place_before-and-masked_w1_all_weak_n10_r16_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_place_prefix_w1_all_weak_n10_r16_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_place_prefix_w1_attn_weak_n10_r16_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_place_prefix_w1_mlp_weak_n10_r16_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_place_previous-block_w1_all_weak_n10_r16_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_place_previous-block_w1_attn_weak_n10_r16_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_place_previous-block_w1_mlp_weak_n10_r16_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_place_previous-window_w2_all_weak_n10_r16_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_place_same-block_w1_all_weak_n10_r16_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_place_same-block_w1_attn_weak_n10_r16_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_place_same-block_w1_mlp_weak_n10_r16_e5_pair005_eval128` | old schema |
| `eval_outputs/circuit_lora_place_smoke_prev` | old schema |
| `eval_outputs/circuit_lora_safe_n10_r16_e5_pairwise_005_eval128` | old schema |
| `eval_outputs/circuit_lora_smoke` | old schema |
| `eval_outputs/circuit_lora_weak_n10_r16_e5_eval128` | old schema |
| `eval_outputs/circuit_lora_weak_n10_r16_e5_pairwise_0025_eval128` | old schema |
| `eval_outputs/circuit_lora_weak_n10_r16_e5_pairwise_010_eval128` | old schema |
| `eval_outputs/circuit_lora_weak_n10_r16_e5_pairwise_eval128` | old schema |
| `eval_outputs/circuit_lora_weak_n25_r16_e5_pairwise_005_eval128` | old schema |
| `eval_outputs/circuit_lora_weak_n25_r8_e3_eval128` | old schema |

## Files

- CSV: `eval_outputs/peft_repair_sweep_tables_with_removed_params.csv`
- JSON: `eval_outputs/peft_repair_sweep_tables_with_removed_params.json`
