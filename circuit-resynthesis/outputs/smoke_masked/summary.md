# MNIST Circuit Resynthesis Smoke

Source accuracy: `0.7520` with `407050` params.

## Final Results

| method | accuracy | params | note |
| --- | ---: | ---: | --- |
| `wanda_source_prune_0.50` | 0.7550 | 203264 | wanda no-retrain source pruning |
| `masked_mlp_local11_source_init_distilled` | 0.7540 | 67594 | decompiled local fc1 windows |
| `source_mlp` | 0.7520 | 407050 | bad substrate teacher |
| `magnitude_source_prune_0.50` | 0.7500 | 203264 | magnitude no-retrain source pruning |
| `magnitude_source_prune_0.80` | 0.7380 | 81306 | magnitude no-retrain source pruning |
| `wanda_source_prune_0.80` | 0.7340 | 81306 | wanda no-retrain source pruning |
| `masked_mlp_randomlocal11_source_init_distilled` | 0.7300 | 67594 | random local fc1 windows, source weights |
| `magnitude_source_prune_0.90` | 0.7110 | 40653 | magnitude no-retrain source pruning |
| `masked_mlp_local7_source_init_distilled` | 0.6920 | 30730 | decompiled local fc1 windows |
| `masked_mlp_randomlocal7_source_init_distilled` | 0.6900 | 30730 | random local fc1 windows, source weights |
| `magnitude_source_prune_0.95` | 0.6790 | 20327 | magnitude no-retrain source pruning |
| `masked_mlp_local11_random_init_distilled` | 0.6660 | 67594 | decompiled local fc1 mask, random weights |
| `wanda_source_prune_0.90` | 0.6600 | 40653 | wanda no-retrain source pruning |
| `masked_mlp_local7_random_init_distilled` | 0.6430 | 30730 | decompiled local fc1 mask, random weights |
| `wanda_source_prune_0.95` | 0.4980 | 20327 | wanda no-retrain source pruning |
| `tiny_conv_scratch` | 0.1550 | 3562 | same conv architecture, no teacher |
| `tiny_conv_distilled_random_init` | 0.1280 | 3562 | black-box distillation |
| `tiny_conv_distilled_circuit_init` | 0.1180 | 3562 | conv filters initialized from decompiled MLP patches |
| `fixed_stroke_bank_distilled` | 0.0920 | 2762 | decompiled filters frozen, train only head |

## Localizing Source FC1

| window | fc1 fraction kept | accuracy |
| ---: | ---: | ---: |
| 3 | 0.0115 | 0.5670 |
| 5 | 0.0319 | 0.6090 |
| 7 | 0.0625 | 0.6810 |
| 9 | 0.1033 | 0.7140 |
| 11 | 0.1543 | 0.7370 |
| 15 | 0.2870 | 0.7500 |

## Masked Local Resynthesis

| method | window | weight fraction left | active params | accuracy |
| --- | ---: | ---: | ---: | ---: |
| `masked_mlp_local7_source_init_distilled` | 7 | 0.0743 | 30208 | 0.6920 |
| `masked_mlp_randomlocal7_source_init_distilled` | 7 | 0.0743 | 30208 | 0.6900 |
| `masked_mlp_local7_random_init_distilled` | 7 | 0.0743 | 30208 | 0.6430 |
| `masked_mlp_local11_source_init_distilled` | 11 | 0.1650 | 67072 | 0.7540 |
| `masked_mlp_randomlocal11_source_init_distilled` | 11 | 0.1650 | 67072 | 0.7300 |
| `masked_mlp_local11_random_init_distilled` | 11 | 0.1650 | 67072 | 0.6660 |

## Read
- Decompiled circuit initialization did not beat plain distillation in this run: `0.1180` vs `0.1280`.
- Best local sparse resynthesis: `masked_mlp_local11_source_init_distilled` at `0.7540` accuracy with `0.1650` of weight params left.
- The source-local-window table tests whether the MLP first layer is actually local enough to recompile into conv filters.
- The masked MLP rows test a weaker target than conv: position-specific local circuits with the same hidden width, compared against random local masks.
- Fixed stroke-bank performance tests whether the recovered patches are sufficient features without learning the conv bank.
- Magnitude/Wanda pruning are included as dumb compression baselines; they do not change architecture.
