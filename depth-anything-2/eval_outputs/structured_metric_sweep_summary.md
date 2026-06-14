# Structured Subcircuit Pruning Metric Sweep

Metrics tested over fine DAV2 subcircuits from `subcircuit_fine_all_zero_32_g32_h16`. Lower-scored nodes are jointly ablated and evaluated on DA-2K point-pair accuracy.

## first64_all_metrics

Baseline: `122/134 = 0.9104`

| metric | budget_nodes | correct_drop | accuracy | selected_param_est |
|---|---:|---:|---:|---:|
| ablation_correct | 5 | 4 | 0.8806 | 110816 |
| ablation_correct | 10 | 24 | 0.7313 | 221600 |
| ablation_correct | 25 | 41 | 0.6045 | 554144 |
| ablation_correct | 50 | 51 | 0.5299 | 1115712 |
| ablation_correct | 100 | 27 | 0.7090 | 2149504 |
| ablation_margin | 5 | 1 | 0.9030 | 123200 |
| ablation_margin | 10 | 6 | 0.8657 | 246400 |
| ablation_margin | 25 | 18 | 0.7761 | 578784 |
| ablation_margin | 50 | 13 | 0.8134 | 1080416 |
| ablation_margin | 100 | 30 | 0.6866 | 2054224 |
| magnitude | 5 | 122 | 0.0000 | 2336 |
| magnitude | 10 | 122 | 0.0000 | 6176 |
| magnitude | 25 | 122 | 0.0000 | 17696 |
| magnitude | 50 | 122 | 0.0000 | 54320 |
| magnitude | 100 | 122 | 0.0000 | 165216 |
| magnitude_circuit4 | 5 | 122 | 0.0000 | 2336 |
| magnitude_circuit4 | 10 | 122 | 0.0000 | 6176 |
| magnitude_circuit4 | 25 | 122 | 0.0000 | 17696 |
| magnitude_circuit4 | 50 | 122 | 0.0000 | 54832 |
| magnitude_circuit4 | 100 | 122 | 0.0000 | 171328 |
| param_eff_correct | 5 | -3 | 0.9328 | 44128 |
| param_eff_correct | 10 | 32 | 0.6716 | 142656 |
| param_eff_correct | 25 | 48 | 0.5522 | 499712 |
| param_eff_correct | 50 | 52 | 0.5224 | 1078752 |
| param_eff_correct | 100 | 37 | 0.6343 | 2235584 |
| param_eff_margin | 5 | -2 | 0.9254 | 52352 |
| param_eff_margin | 10 | 2 | 0.8955 | 80064 |
| param_eff_margin | 25 | 3 | 0.8881 | 304096 |
| param_eff_margin | 50 | 6 | 0.8657 | 618848 |
| param_eff_margin | 100 | 26 | 0.7164 | 1318848 |
| random | 5 | 2 | 0.8955 | 92320 |
| random | 10 | 1 | 0.9030 | 150768 |
| random | 25 | 11 | 0.8284 | 423152 |
| random | 50 | 24 | 0.7313 | 885664 |
| random | 100 | 41 | 0.6045 | 1976176 |
| stability | 5 | 0 | 0.9104 | 46096 |
| stability | 10 | 0 | 0.9104 | 92224 |
| stability | 25 | 0 | 0.9104 | 158336 |
| stability | 50 | 1 | 0.9030 | 192896 |
| stability | 100 | 6 | 0.8657 | 624096 |
| wanda | 5 | 122 | 0.0000 | 36992 |
| wanda | 10 | 122 | 0.0000 | 98592 |
| wanda | 25 | 122 | 0.0000 | 283392 |
| wanda | 50 | 122 | 0.0000 | 591392 |
| wanda | 100 | 122 | 0.0000 | 1207392 |
| wanda_anticircuit0p5 | 5 | 122 | 0.0000 | 36992 |
| wanda_anticircuit0p5 | 10 | 122 | 0.0000 | 98592 |
| wanda_anticircuit0p5 | 25 | 122 | 0.0000 | 283392 |
| wanda_anticircuit0p5 | 50 | 122 | 0.0000 | 591392 |
| wanda_anticircuit0p5 | 100 | 122 | 0.0000 | 1207392 |
| wanda_circuit4 | 5 | 122 | 0.0000 | 36992 |
| wanda_circuit4 | 10 | 122 | 0.0000 | 98592 |
| wanda_circuit4 | 25 | 122 | 0.0000 | 283392 |
| wanda_circuit4 | 50 | 122 | 0.0000 | 591392 |
| wanda_circuit4 | 100 | 122 | 0.0000 | 1207392 |

Best by correct drop:

| rank | metric | budget_nodes | correct_drop | accuracy | selected_param_est |
|---:|---|---:|---:|---:|---:|
| 1 | param_eff_correct | 5 | -3 | 0.9328 | 44128 |
| 2 | param_eff_margin | 5 | -2 | 0.9254 | 52352 |
| 3 | stability | 5 | 0 | 0.9104 | 46096 |
| 4 | stability | 10 | 0 | 0.9104 | 92224 |
| 5 | stability | 25 | 0 | 0.9104 | 158336 |
| 6 | ablation_margin | 5 | 1 | 0.9030 | 123200 |
| 7 | random | 10 | 1 | 0.9030 | 150768 |
| 8 | stability | 50 | 1 | 0.9030 | 192896 |
| 9 | param_eff_margin | 10 | 2 | 0.8955 | 80064 |
| 10 | random | 5 | 2 | 0.8955 | 92320 |
| 11 | param_eff_margin | 25 | 3 | 0.8881 | 304096 |
| 12 | ablation_correct | 5 | 4 | 0.8806 | 110816 |

## first64_gated_focus

Baseline: `122/134 = 0.9104`

| metric | budget_nodes | correct_drop | accuracy | selected_param_est |
|---|---:|---:|---:|---:|
| param_eff_margin | 25 | 3 | 0.8881 | 304096 |
| param_eff_margin | 50 | 6 | 0.8657 | 618848 |
| param_eff_margin | 100 | 26 | 0.7164 | 1318848 |
| param_eff_margin | 200 | 42 | 0.5970 | 3144000 |
| param_eff_margin | 400 | 51 | 0.5299 | 7446656 |
| safe_magnitude | 25 | -1 | 0.9179 | 18448 |
| safe_magnitude | 50 | 2 | 0.8955 | 55856 |
| safe_magnitude | 100 | 53 | 0.5149 | 270640 |
| safe_magnitude | 200 | 61 | 0.4552 | 1420448 |
| safe_magnitude | 400 | 108 | 0.1045 | 3484768 |
| safe_wanda | 25 | 19 | 0.7687 | 295696 |
| safe_wanda | 50 | 50 | 0.5373 | 603696 |
| safe_wanda | 100 | 41 | 0.6045 | 1174256 |
| safe_wanda | 200 | 45 | 0.5746 | 1737840 |
| safe_wanda | 400 | 49 | 0.5448 | 6190368 |
| stability | 25 | 0 | 0.9104 | 158336 |
| stability | 50 | 1 | 0.9030 | 192896 |
| stability | 100 | 6 | 0.8657 | 624096 |
| stability | 200 | 10 | 0.8358 | 2148944 |
| stability | 400 | 38 | 0.6269 | 6416576 |
| stability_param | 25 | 0 | 0.9104 | 614688 |
| stability_param | 50 | 1 | 0.9030 | 1628944 |
| stability_param | 100 | 1 | 0.9030 | 3135744 |
| stability_param | 200 | 48 | 0.5522 | 5737808 |
| stability_param | 400 | 51 | 0.5299 | 10726640 |
| stability_wanda | 25 | 0 | 0.9104 | 158336 |
| stability_wanda | 50 | 1 | 0.9030 | 192896 |
| stability_wanda | 100 | 6 | 0.8657 | 624096 |
| stability_wanda | 200 | 10 | 0.8358 | 2148944 |
| stability_wanda | 400 | 38 | 0.6269 | 6416576 |

Best by correct drop:

| rank | metric | budget_nodes | correct_drop | accuracy | selected_param_est |
|---:|---|---:|---:|---:|---:|
| 1 | safe_magnitude | 25 | -1 | 0.9179 | 18448 |
| 2 | stability | 25 | 0 | 0.9104 | 158336 |
| 3 | stability_param | 25 | 0 | 0.9104 | 614688 |
| 4 | stability_wanda | 25 | 0 | 0.9104 | 158336 |
| 5 | stability | 50 | 1 | 0.9030 | 192896 |
| 6 | stability_param | 50 | 1 | 0.9030 | 1628944 |
| 7 | stability_param | 100 | 1 | 0.9030 | 3135744 |
| 8 | stability_wanda | 50 | 1 | 0.9030 | 192896 |
| 9 | safe_magnitude | 50 | 2 | 0.8955 | 55856 |
| 10 | param_eff_margin | 25 | 3 | 0.8881 | 304096 |
| 11 | param_eff_margin | 50 | 6 | 0.8657 | 618848 |
| 12 | stability | 100 | 6 | 0.8657 | 624096 |

## holdout64_gated_focus

Baseline: `105/118 = 0.8898`

| metric | budget_nodes | correct_drop | accuracy | selected_param_est |
|---|---:|---:|---:|---:|
| param_eff_margin | 25 | 7 | 0.8305 | 304096 |
| param_eff_margin | 50 | 4 | 0.8559 | 618848 |
| param_eff_margin | 100 | 22 | 0.7034 | 1318848 |
| param_eff_margin | 200 | 33 | 0.6102 | 3144000 |
| safe_magnitude | 25 | 3 | 0.8644 | 18448 |
| safe_magnitude | 50 | 6 | 0.8390 | 55856 |
| safe_magnitude | 100 | 47 | 0.4915 | 270640 |
| safe_magnitude | 200 | 46 | 0.5000 | 1420448 |
| stability | 25 | 0 | 0.8898 | 158336 |
| stability | 50 | 0 | 0.8898 | 192896 |
| stability | 100 | 1 | 0.8814 | 624096 |
| stability | 200 | 7 | 0.8305 | 2148944 |
| stability_param | 25 | -1 | 0.8983 | 614688 |
| stability_param | 50 | -2 | 0.9068 | 1628944 |
| stability_param | 100 | 1 | 0.8814 | 3135744 |
| stability_param | 200 | 43 | 0.5254 | 5737808 |
| stability_wanda | 25 | 0 | 0.8898 | 158336 |
| stability_wanda | 50 | 0 | 0.8898 | 192896 |
| stability_wanda | 100 | 1 | 0.8814 | 624096 |
| stability_wanda | 200 | 7 | 0.8305 | 2148944 |

Best by correct drop:

| rank | metric | budget_nodes | correct_drop | accuracy | selected_param_est |
|---:|---|---:|---:|---:|---:|
| 1 | stability_param | 50 | -2 | 0.9068 | 1628944 |
| 2 | stability_param | 25 | -1 | 0.8983 | 614688 |
| 3 | stability | 25 | 0 | 0.8898 | 158336 |
| 4 | stability | 50 | 0 | 0.8898 | 192896 |
| 5 | stability_wanda | 25 | 0 | 0.8898 | 158336 |
| 6 | stability_wanda | 50 | 0 | 0.8898 | 192896 |
| 7 | stability | 100 | 1 | 0.8814 | 624096 |
| 8 | stability_param | 100 | 1 | 0.8814 | 3135744 |
| 9 | stability_wanda | 100 | 1 | 0.8814 | 624096 |
| 10 | safe_magnitude | 25 | 3 | 0.8644 | 18448 |
| 11 | param_eff_margin | 50 | 4 | 0.8559 | 618848 |
| 12 | safe_magnitude | 50 | 6 | 0.8390 | 55856 |

## Takeaways

- `stability` ranks by mean absolute pair-margin perturbation and is the most reliable ungated metric: zero drop at 25 nodes on first64 and holdout64, one-pair drop at 50-100 nodes depending on split.
- `stability_param` ranks mean absolute pair-margin perturbation per estimated parameter. It is best at moderate budgets: on holdout64, 50 nodes improves by 2 pairs and 100 nodes loses only 1 pair while selecting much larger parameter mass.
- Pure `magnitude` and pure `wanda` catastrophically fail for structured subcircuits because tiny final decoder channels are low saliency by weight/activation but causally mandatory.
- Soft circuit weighting (`wanda_circuit4`, `magnitude_circuit4`) does not fix that failure; hard causal/stability gating is needed before using weight/activation saliency.
- Raw `ablation_correct` overfits the 32-image circuit screen and composes badly; pair-margin detail is more useful than binary correctness for ranking prune candidates.
