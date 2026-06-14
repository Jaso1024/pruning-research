# MNIST Circuit Resynthesis Smoke

Source accuracy: `0.9797` with `407050` params.

## Final Results

| method | accuracy | params | note |
| --- | ---: | ---: | --- |
| `wanda_source_prune_0.50` | 0.9798 | 203264 | wanda no-retrain source pruning |
| `source_mlp` | 0.9797 | 407050 | bad substrate teacher |
| `magnitude_source_prune_0.50` | 0.9792 | 203264 | magnitude no-retrain source pruning |
| `magnitude_source_prune_0.80` | 0.9429 | 81306 | magnitude no-retrain source pruning |
| `tiny_conv_scratch` | 0.9360 | 6410 | same conv architecture, no teacher |
| `tiny_conv_distilled_random_init` | 0.9264 | 6410 | black-box distillation |
| `wanda_source_prune_0.80` | 0.9154 | 81306 | wanda no-retrain source pruning |
| `tiny_conv_distilled_circuit_init` | 0.8439 | 6410 | conv filters initialized from decompiled MLP patches |
| `wanda_source_prune_0.90` | 0.8340 | 40653 | wanda no-retrain source pruning |
| `magnitude_source_prune_0.90` | 0.7259 | 40653 | magnitude no-retrain source pruning |
| `fixed_stroke_bank_distilled` | 0.6110 | 4810 | decompiled filters frozen, train only head |
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

## Read
- Decompiled circuit initialization did not beat plain distillation in this run: `0.8439` vs `0.9264`.
- The source-local-window table tests whether the MLP first layer is actually local enough to recompile into conv filters.
- Fixed stroke-bank performance tests whether the recovered patches are sufficient features without learning the conv bank.
- Magnitude/Wanda pruning are included as dumb compression baselines; they do not change architecture.
