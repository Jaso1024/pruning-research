# Pruning Granularity Audit

This compares current PEFT mask granularity against older and same-slice no-compensation pruning results.

## Same-Slice Granularity Check

| run | method | removed params | baseline acc | pruned acc | drop | granularity | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `granularity_check_stability_param_skip128_eval128` | `stability_param` | 258208 | 0.9450 | 0.9450 | 0.0000 | head_channel_group:10, head_input_channel_group:8 | skip=128 max=128 budget=258208 selected=18 |
| `granularity_check_stability_param_skip128_eval128` | `stability_param` | 202896 | 0.9450 | 0.9450 | 0.0000 | head_channel_group:9, head_input_channel_group:8 | skip=128 max=128 budget=202896 selected=17 |
| `granularity_check_stability_param_skip128_eval128` | `stability_param` | 534768 | 0.9450 | 0.9385 | 0.0065 | head_channel_group:15, head_input_channel_group:8 | skip=128 max=128 budget=534768 selected=23 |
| `granularity_check_stability_param_skip128_eval128` | `stability_param` | 614688 | 0.9450 | 0.9353 | 0.0097 | head_channel_group:16, head_input_channel_group:8, mlp_group:1 | skip=128 max=128 budget=25 selected=25 |
| `granularity_check_stability_param_skip64_eval128` | `stability_param` | 258208 | 0.9288 | 0.9326 | -0.0037 | head_channel_group:10, head_input_channel_group:8 | skip=64 max=128 budget=258208 selected=18 |
| `granularity_check_stability_param_skip64_eval128` | `stability_param` | 202896 | 0.9288 | 0.9326 | -0.0037 | head_channel_group:9, head_input_channel_group:8 | skip=64 max=128 budget=202896 selected=17 |
| `granularity_check_stability_param_skip128_eval128` | `stability_param` | 1020352 | 0.9450 | 0.9320 | 0.0129 | head_channel_group:20, head_input_channel_group:10, mlp_group:4 | skip=128 max=128 budget=1020352 selected=34 |
| `granularity_check_stability_param_skip64_eval128` | `stability_param` | 614688 | 0.9288 | 0.9288 | 0.0000 | head_channel_group:16, head_input_channel_group:8, mlp_group:1 | skip=64 max=128 budget=25 selected=25 |
| `granularity_check_stability_param_skip64_eval128` | `stability_param` | 534768 | 0.9288 | 0.9288 | 0.0000 | head_channel_group:15, head_input_channel_group:8 | skip=64 max=128 budget=534768 selected=23 |
| `granularity_check_stability_param_skip64_eval128` | `stability_param` | 1020352 | 0.9288 | 0.9213 | 0.0075 | head_channel_group:20, head_input_channel_group:10, mlp_group:4 | skip=64 max=128 budget=1020352 selected=34 |
| `peft_holdout_holdout_t64_e5_previous-block_attn_weak_n10_evalskip128_eval128` | `weak` | 233856 | 0.9450 | 0.8511 | 0.0939 | current weak-n transformer qkv/proj/mlp groups | skip=128 max=128 folded=0.9061488673139159 placement=previous-block method=lora |
| `peft_clean_rerun_prev_attn_lora_r8_t64_e5_weak_n10_evalskip64_eval128` | `weak` | 233856 | 0.9288 | 0.8352 | 0.0936 | current weak-n transformer qkv/proj/mlp groups | skip=64 max=128 folded=0.9101123595505618 placement=previous-block method=lora |

## Top 50 By Accuracy With >=200k Removed

| run | family | method | removed params | removed % | baseline acc | pruned acc | drop | granularity | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `baseline_magnitude_global_z1020352_full` | baseline | `magnitude/global_iterative` | 1020352 | 4.81 | 0.9521 | 0.9536 | -0.0015 | unstructured scalar transformer weights | target_fraction=0.04805350645088855 |
| `baseline_wanda_permatrix_pf0612_full` | baseline | `wanda/per_matrix_iterative` | 1300320 | 6.12 | 0.9521 | 0.9531 | -0.0010 | unstructured scalar transformer weights | target_fraction=0.061239 |
| `baseline_wanda_global_z534768_full` | baseline | `wanda/global_iterative` | 534768 | 2.52 | 0.9521 | 0.9531 | -0.0010 | unstructured scalar transformer weights | target_fraction=0.025184914388774356 |
| `scalar_weight_circuit_hybrid_protect_mag4_s32_full` | scalar-circuit | `scalar_weight_circuit_hybrid_protect_mag4_s32` | 1020352 | 4.81 | 0.9521 | 0.9521 | 0.0000 | scalar transformer weights | budget=1020352 zero_fraction=0.04805350597993827 |
| `weight_masked_value_budget_diagnose_full` | structured-circuit | `stability_param` | 202896 |  | 0.9521 | 0.9521 | 0.0000 | head_channel_group:9, head_input_channel_group:8 | skip=0 max=0 budget=150000 selected=17 |
| `baseline_magnitude_global_z2016304_full` | baseline | `magnitude/global_iterative` | 2016304 | 9.50 | 0.9521 | 0.9516 | 0.0005 | unstructured scalar transformer weights | target_fraction=0.09495789374834225 |
| `baseline_magnitude_permatrix_pf0612_full` | baseline | `magnitude/per_matrix_iterative` | 1300320 | 6.12 | 0.9521 | 0.9516 | 0.0005 | unstructured scalar transformer weights | target_fraction=0.061239 |
| `baseline_wanda_permatrix_pf0255_full` | baseline | `wanda/per_matrix_iterative` | 540936 | 2.55 | 0.9521 | 0.9516 | 0.0005 | unstructured scalar transformer weights | target_fraction=0.025477 |
| `baseline_magnitude_global_z534768_full` | baseline | `magnitude/global_iterative` | 534768 | 2.52 | 0.9521 | 0.9516 | 0.0005 | unstructured scalar transformer weights | target_fraction=0.025184914388774356 |
| `baseline_magnitude_global_z1518336_full` | baseline | `magnitude/global_iterative` | 1518336 | 7.15 | 0.9521 | 0.9512 | 0.0010 | unstructured scalar transformer weights | target_fraction=0.07150607685983917 |
| `baseline_wanda_global_z1020352_full` | baseline | `wanda/global_iterative` | 1020352 | 4.81 | 0.9521 | 0.9512 | 0.0010 | unstructured scalar transformer weights | target_fraction=0.04805350645088855 |
| `baseline_magnitude_permatrix_pf0255_full` | baseline | `magnitude/per_matrix_iterative` | 540936 | 2.55 | 0.9521 | 0.9512 | 0.0010 | unstructured scalar transformer weights | target_fraction=0.025477 |
| `weight_masked_value_budget_diagnose_full` | structured-circuit | `stability_param` | 258208 |  | 0.9521 | 0.9512 | 0.0010 | head_channel_group:10, head_input_channel_group:8 | skip=0 max=0 budget=250000 selected=18 |
| `baseline_magnitude_global_z3006528_full` | baseline | `magnitude/global_iterative` | 3006528 | 14.16 | 0.9521 | 0.9507 | 0.0015 | unstructured scalar transformer weights | target_fraction=0.1415925207255799 |
| `baseline_wanda_global_z1518336_full` | baseline | `wanda/global_iterative` | 1518336 | 7.15 | 0.9521 | 0.9502 | 0.0019 | unstructured scalar transformer weights | target_fraction=0.07150607685983917 |
| `scalar_taylor3_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor3_abs_s32_tau1` | 1020352 | 4.81 | 0.9521 | 0.9502 | 0.0019 | scalar transformer weights | budget=1020352 zero_fraction=0.04805350597993827 |
| `scalar_taylor2_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor2_abs_s32_tau1` | 1020352 | 4.81 | 0.9521 | 0.9502 | 0.0019 | scalar transformer weights | budget=1020352 zero_fraction=0.04805350597993827 |
| `scalar_weight_circuit_abs_wgrad_s32_full` | scalar-circuit | `scalar_weight_circuit_abs_wgrad_s32` | 1020352 | 4.81 | 0.9521 | 0.9502 | 0.0019 | scalar transformer weights | budget=1020352 zero_fraction=0.04805350597993827 |
| `scalar_taylor1_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor1_abs_s32_tau1` | 1020352 | 4.81 | 0.9521 | 0.9502 | 0.0019 | scalar transformer weights | budget=1020352 zero_fraction=0.04805350597993827 |
| `scalar_weight_circuit_hybrid_protect_mag4_s32_full` | scalar-circuit | `scalar_weight_circuit_hybrid_protect_mag4_s32` | 534768 | 2.52 | 0.9521 | 0.9502 | 0.0019 | scalar transformer weights | budget=534768 zero_fraction=0.025184913917824073 |
| `weight_masked_value_budget_diagnose_full` | structured-circuit | `stability_param` | 368832 |  | 0.9521 | 0.9502 | 0.0019 | head_channel_group:12, head_input_channel_group:8 | skip=0 max=0 budget=350000 selected=20 |
| `scalar_weight_circuit_abs_wgrad_s32_full` | scalar-circuit | `scalar_weight_circuit_abs_wgrad_s32` | 534768 | 2.52 | 0.9521 | 0.9497 | 0.0024 | scalar transformer weights | budget=534768 zero_fraction=0.025184913917824073 |
| `weight_masked_value_budget_diagnose_full` | structured-circuit | `stability_param` | 479456 |  | 0.9521 | 0.9497 | 0.0024 | head_channel_group:14, head_input_channel_group:8 | skip=0 max=0 budget=450000 selected=22 |
| `baseline_wanda_global_z2016304_full` | baseline | `wanda/global_iterative` | 2016304 | 9.50 | 0.9521 | 0.9492 | 0.0029 | unstructured scalar transformer weights | target_fraction=0.09495789374834225 |
| `weight_masked_value_budget_ladder_full` | structured-circuit | `stability_param` | 534768 |  | 0.9521 | 0.9492 | 0.0029 | head_channel_group:15, head_input_channel_group:8 | skip=0 max=0 budget=500000 selected=23 |
| `scalar_taylor2_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor2_abs_s32_tau1` | 534768 | 2.52 | 0.9521 | 0.9492 | 0.0029 | scalar transformer weights | budget=534768 zero_fraction=0.025184913917824073 |
| `scalar_taylor3_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor3_abs_s32_tau1` | 534768 | 2.52 | 0.9521 | 0.9487 | 0.0034 | scalar transformer weights | budget=534768 zero_fraction=0.025184913917824073 |
| `scalar_taylor1_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor1_abs_s32_tau1` | 534768 | 2.52 | 0.9521 | 0.9487 | 0.0034 | scalar transformer weights | budget=534768 zero_fraction=0.025184913917824073 |
| `weight_masked_value_budget_diagnose_full` | structured-circuit | `stability_param` | 590080 |  | 0.9521 | 0.9483 | 0.0039 | head_channel_group:16, head_input_channel_group:8 | skip=0 max=0 budget=550000 selected=24 |
| `weight_masked_subcircuit_full_stratified32` | structured-circuit | `stability_param` | 614688 |  | 0.9521 | 0.9478 | 0.0044 | head_channel_group:16, head_input_channel_group:8, mlp_group:1 | skip=0 max=0 budget=25 selected=25 |
| `weight_masked_subcircuit_full_stratified32_best` | structured-circuit | `stability_param` | 614688 |  | 0.9521 | 0.9478 | 0.0044 | head_channel_group:16, head_input_channel_group:8, mlp_group:1 | skip=0 max=0 budget=25 selected=25 |
| `scalar_weight_circuit_abs_wgrad_s32_full` | scalar-circuit | `scalar_weight_circuit_abs_wgrad_s32` | 1518336 | 7.15 | 0.9521 | 0.9468 | 0.0053 | scalar transformer weights | budget=1518336 zero_fraction=0.0715060763888889 |
| `scalar_weight_circuit_hybrid_protect_mag4_s32_full` | scalar-circuit | `scalar_weight_circuit_hybrid_protect_mag4_s32` | 1518336 | 7.15 | 0.9521 | 0.9463 | 0.0058 | scalar transformer weights | budget=1518336 zero_fraction=0.0715060763888889 |
| `scalar_taylor3_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor3_abs_s32_tau1` | 1518336 | 7.15 | 0.9521 | 0.9458 | 0.0063 | scalar transformer weights | budget=1518336 zero_fraction=0.0715060763888889 |
| `scalar_taylor2_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor2_abs_s32_tau1` | 1518336 | 7.15 | 0.9521 | 0.9458 | 0.0063 | scalar transformer weights | budget=1518336 zero_fraction=0.0715060763888889 |
| `scalar_taylor1_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor1_abs_s32_tau1` | 1518336 | 7.15 | 0.9521 | 0.9458 | 0.0063 | scalar transformer weights | budget=1518336 zero_fraction=0.0715060763888889 |
| `scalar_taylor2_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor2_abs_s32_tau1` | 3006528 | 14.16 | 0.9521 | 0.9454 | 0.0068 | scalar transformer weights | budget=3006528 zero_fraction=0.14159252025462962 |
| `granularity_check_stability_param_skip128_eval128` | structured-circuit | `stability_param` | 258208 |  | 0.9450 | 0.9450 | 0.0000 | head_channel_group:10, head_input_channel_group:8 | skip=128 max=128 budget=258208 selected=18 |
| `granularity_check_stability_param_skip128_eval128` | structured-circuit | `stability_param` | 202896 |  | 0.9450 | 0.9450 | 0.0000 | head_channel_group:9, head_input_channel_group:8 | skip=128 max=128 budget=202896 selected=17 |
| `scalar_taylor3_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor3_abs_s32_tau1` | 3006528 | 14.16 | 0.9521 | 0.9449 | 0.0073 | scalar transformer weights | budget=3006528 zero_fraction=0.14159252025462962 |
| `scalar_weight_circuit_abs_wgrad_s32_full` | scalar-circuit | `scalar_weight_circuit_abs_wgrad_s32` | 3006528 | 14.16 | 0.9521 | 0.9449 | 0.0073 | scalar transformer weights | budget=3006528 zero_fraction=0.14159252025462962 |
| `scalar_taylor1_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor1_abs_s32_tau1` | 3006528 | 14.16 | 0.9521 | 0.9449 | 0.0073 | scalar transformer weights | budget=3006528 zero_fraction=0.14159252025462962 |
| `scalar_taylor3_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor3_abs_s32_tau1` | 2016304 | 9.50 | 0.9521 | 0.9444 | 0.0077 | scalar transformer weights | budget=2016304 zero_fraction=0.09495789327739197 |
| `scalar_weight_circuit_abs_wgrad_s32_full` | scalar-circuit | `scalar_weight_circuit_abs_wgrad_s32` | 2016304 | 9.50 | 0.9521 | 0.9444 | 0.0077 | scalar transformer weights | budget=2016304 zero_fraction=0.09495789327739197 |
| `scalar_taylor1_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor1_abs_s32_tau1` | 2016304 | 9.50 | 0.9521 | 0.9444 | 0.0077 | scalar transformer weights | budget=2016304 zero_fraction=0.09495789327739197 |
| `scalar_taylor2_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor2_abs_s32_tau1` | 2016304 | 9.50 | 0.9521 | 0.9439 | 0.0082 | scalar transformer weights | budget=2016304 zero_fraction=0.09495789327739197 |
| `scalar_weight_circuit_hybrid_protect_mag4_s32_full` | scalar-circuit | `scalar_weight_circuit_hybrid_protect_mag4_s32` | 2016304 | 9.50 | 0.9521 | 0.9429 | 0.0092 | scalar transformer weights | budget=2016304 zero_fraction=0.09495789327739197 |
| `scalar_weight_circuit_abs_wgrad_s32_full` | scalar-circuit | `scalar_weight_circuit_abs_wgrad_s32` | 5014736 | 23.62 | 0.9521 | 0.9410 | 0.0111 | scalar transformer weights | budget=5014736 zero_fraction=0.23616913218557098 |
| `weight_masked_value_budget_ladder_full` | structured-circuit | `stability_param` | 1518336 |  | 0.9521 | 0.9405 | 0.0116 | attn_v_head:1, head_channel_group:20, head_input_channel_group:15, mlp_group:12 | skip=0 max=0 budget=1500000 selected=48 |
| `scalar_taylor2_abs_s32_tau1_full` | scalar-circuit | `scalar_taylor2_abs_s32_tau1` | 5014736 | 23.62 | 0.9521 | 0.9400 | 0.0121 | scalar transformer weights | budget=5014736 zero_fraction=0.23616913218557098 |

- JSON: `/home/ubuntu/eval_outputs/pruning_granularity_audit.json`
- CSV: `/home/ubuntu/eval_outputs/pruning_granularity_audit.csv`
