# MNIST Circuit Resynthesis

Source accuracy: `0.9797` with `407050` params.

## Final Results

| method | accuracy | params | note |
| --- | ---: | ---: | --- |
| `masked_mlp_local15_source_init_distilled` | 0.9820 | 120842 | decompiled local fc1 windows |
| `masked_mlp_randomlocal15_source_init_distilled` | 0.9815 | 120842 | random local fc1 windows, source weights |
| `masked_mlp_local11_source_init_distilled` | 0.9803 | 67594 | decompiled local fc1 windows |
| `wanda_source_prune_0.50` | 0.9798 | 203264 | wanda no-retrain source pruning |
| `source_mlp` | 0.9797 | 407050 | bad substrate teacher |
| `magnitude_source_prune_0.50` | 0.9792 | 203264 | magnitude no-retrain source pruning |
| `masked_mlp_local7_source_init_distilled` | 0.9781 | 30730 | decompiled local fc1 windows |
| `masked_mlp_randomlocal11_source_init_distilled` | 0.9773 | 67594 | random local fc1 windows, source weights |
| `masked_mlp_local11_random_init_distilled` | 0.9759 | 67594 | decompiled local fc1 mask, random weights |
| `masked_mlp_local15_random_init_distilled` | 0.9754 | 120842 | decompiled local fc1 mask, random weights |
| `masked_mlp_local7_random_init_distilled` | 0.9739 | 30730 | decompiled local fc1 mask, random weights |
| `masked_mlp_randomlocal7_source_init_distilled` | 0.9725 | 30730 | random local fc1 windows, source weights |
| `magnitude_source_prune_0.80` | 0.9429 | 81306 | magnitude no-retrain source pruning |
| `tiny_conv_scratch` | 0.9214 | 6410 | same conv architecture, no teacher |
| `wanda_source_prune_0.80` | 0.9154 | 81306 | wanda no-retrain source pruning |
| `tiny_conv_distilled_random_init` | 0.9133 | 6410 | black-box distillation |
| `tiny_conv_distilled_circuit_init` | 0.8858 | 6410 | conv filters initialized from decompiled MLP patches |
| `wanda_source_prune_0.90` | 0.8340 | 40653 | wanda no-retrain source pruning |
| `magnitude_source_prune_0.90` | 0.7259 | 40653 | magnitude no-retrain source pruning |
| `fixed_stroke_bank_distilled` | 0.6086 | 4810 | decompiled filters frozen, train only head |
| `wanda_source_prune_0.95` | 0.5920 | 20327 | wanda no-retrain source pruning |
| `magnitude_source_prune_0.95` | 0.5143 | 20327 | magnitude no-retrain source pruning |

## Localizing Source FC1

| window | fc1 fraction kept | accuracy |
| ---: | ---: | ---: |
| 3 | 0.0115 | 0.1601 |
| 5 | 0.0319 | 0.4594 |
| 7 | 0.0625 | 0.6845 |
| 9 | 0.1033 | 0.7827 |
| 11 | 0.1543 | 0.8339 |
| 15 | 0.2870 | 0.9123 |

## Masked Local Resynthesis

| method | window | weight fraction left | active params | accuracy |
| --- | ---: | ---: | ---: | ---: |
| `masked_mlp_local7_source_init_distilled` | 7 | 0.0743 | 30208 | 0.9781 |
| `masked_mlp_randomlocal7_source_init_distilled` | 7 | 0.0743 | 30208 | 0.9725 |
| `masked_mlp_local7_random_init_distilled` | 7 | 0.0743 | 30208 | 0.9739 |
| `masked_mlp_local11_source_init_distilled` | 11 | 0.1650 | 67072 | 0.9803 |
| `masked_mlp_randomlocal11_source_init_distilled` | 11 | 0.1650 | 67072 | 0.9773 |
| `masked_mlp_local11_random_init_distilled` | 11 | 0.1650 | 67072 | 0.9759 |
| `masked_mlp_local15_source_init_distilled` | 15 | 0.2960 | 120320 | 0.9820 |
| `masked_mlp_randomlocal15_source_init_distilled` | 15 | 0.2960 | 120320 | 0.9815 |
| `masked_mlp_local15_random_init_distilled` | 15 | 0.2960 | 120320 | 0.9754 |

## Read
- Decompiled circuit initialization did not beat plain distillation in this run: `0.8858` vs `0.9133`.
- Best local sparse resynthesis: `masked_mlp_local15_source_init_distilled` at `0.9820` accuracy with `0.2960` of weight params left.
- The source-local-window table tests whether the MLP first layer is actually local enough to recompile into conv filters.
- The masked MLP rows test a weaker target than conv: position-specific local circuits with the same hidden width, compared against random local masks.
- Fixed stroke-bank performance tests whether the recovered patches are sufficient features without learning the conv bank.
- Magnitude/Wanda pruning are included as dumb compression baselines; they do not change architecture.
