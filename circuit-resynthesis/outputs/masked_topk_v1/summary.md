# MNIST Circuit Resynthesis

Source accuracy: `0.9797` with `407050` params.

## Final Results

| method | accuracy | params | note |
| --- | ---: | ---: | --- |
| `masked_mlp_local15_source_init_distilled` | 0.9814 | 120842 | decompiled local fc1 windows |
| `masked_mlp_randomlocal13_source_init_distilled` | 0.9811 | 92170 | random local fc1 windows, source weights |
| `masked_mlp_local13_source_init_distilled` | 0.9810 | 92170 | decompiled local fc1 windows |
| `masked_mlp_topk169_source_init_distilled` | 0.9809 | 92170 | per-neuron top-k fc1 weights, same density as local window |
| `masked_mlp_randomlocal15_source_init_distilled` | 0.9809 | 120842 | random local fc1 windows, source weights |
| `masked_mlp_topk81_source_init_distilled` | 0.9804 | 47114 | per-neuron top-k fc1 weights, same density as local window |
| `masked_mlp_local11_source_init_distilled` | 0.9804 | 67594 | decompiled local fc1 windows |
| `masked_mlp_local9_source_init_distilled` | 0.9802 | 47114 | decompiled local fc1 windows |
| `masked_mlp_topk225_source_init_distilled` | 0.9801 | 120842 | per-neuron top-k fc1 weights, same density as local window |
| `masked_mlp_topk121_source_init_distilled` | 0.9800 | 67594 | per-neuron top-k fc1 weights, same density as local window |
| `wanda_source_prune_0.50` | 0.9798 | 203264 | wanda no-retrain source pruning |
| `source_mlp` | 0.9797 | 407050 | bad substrate teacher |
| `masked_mlp_topk49_source_init_distilled` | 0.9794 | 30730 | per-neuron top-k fc1 weights, same density as local window |
| `magnitude_source_prune_0.50` | 0.9792 | 203264 | magnitude no-retrain source pruning |
| `masked_mlp_randomlocal11_source_init_distilled` | 0.9787 | 67594 | random local fc1 windows, source weights |
| `masked_mlp_local7_source_init_distilled` | 0.9783 | 30730 | decompiled local fc1 windows |
| `masked_mlp_local11_random_init_distilled` | 0.9775 | 67594 | decompiled local fc1 mask, random weights |
| `masked_mlp_local13_random_init_distilled` | 0.9771 | 92170 | decompiled local fc1 mask, random weights |
| `masked_mlp_randomlocal9_source_init_distilled` | 0.9765 | 47114 | random local fc1 windows, source weights |
| `masked_mlp_local9_random_init_distilled` | 0.9759 | 47114 | decompiled local fc1 mask, random weights |
| `masked_mlp_local15_random_init_distilled` | 0.9756 | 120842 | decompiled local fc1 mask, random weights |
| `masked_mlp_topk25_source_init_distilled` | 0.9752 | 18442 | per-neuron top-k fc1 weights, same density as local window |
| `masked_mlp_local7_random_init_distilled` | 0.9743 | 30730 | decompiled local fc1 mask, random weights |
| `masked_mlp_randomlocal7_source_init_distilled` | 0.9729 | 30730 | random local fc1 windows, source weights |
| `masked_mlp_local5_source_init_distilled` | 0.9708 | 18442 | decompiled local fc1 windows |
| `masked_mlp_local5_random_init_distilled` | 0.9658 | 18442 | decompiled local fc1 mask, random weights |
| `masked_mlp_randomlocal5_source_init_distilled` | 0.9616 | 18442 | random local fc1 windows, source weights |
| `magnitude_source_prune_0.80` | 0.9429 | 81306 | magnitude no-retrain source pruning |
| `wanda_source_prune_0.80` | 0.9154 | 81306 | wanda no-retrain source pruning |
| `wanda_source_prune_0.90` | 0.8340 | 40653 | wanda no-retrain source pruning |
| `magnitude_source_prune_0.90` | 0.7259 | 40653 | magnitude no-retrain source pruning |
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
| `masked_mlp_local5_source_init_distilled` | 5 | 0.0441 | 17920 | 0.9708 |
| `masked_mlp_randomlocal5_source_init_distilled` | 5 | 0.0441 | 17920 | 0.9616 |
| `masked_mlp_local5_random_init_distilled` | 5 | 0.0441 | 17920 | 0.9658 |
| `masked_mlp_topk25_source_init_distilled` | 5 | 0.0441 | 17920 | 0.9752 |
| `masked_mlp_local7_source_init_distilled` | 7 | 0.0743 | 30208 | 0.9783 |
| `masked_mlp_randomlocal7_source_init_distilled` | 7 | 0.0743 | 30208 | 0.9729 |
| `masked_mlp_local7_random_init_distilled` | 7 | 0.0743 | 30208 | 0.9743 |
| `masked_mlp_topk49_source_init_distilled` | 7 | 0.0743 | 30208 | 0.9794 |
| `masked_mlp_local9_source_init_distilled` | 9 | 0.1146 | 46592 | 0.9802 |
| `masked_mlp_randomlocal9_source_init_distilled` | 9 | 0.1146 | 46592 | 0.9765 |
| `masked_mlp_local9_random_init_distilled` | 9 | 0.1146 | 46592 | 0.9759 |
| `masked_mlp_topk81_source_init_distilled` | 9 | 0.1146 | 46592 | 0.9804 |
| `masked_mlp_local11_source_init_distilled` | 11 | 0.1650 | 67072 | 0.9804 |
| `masked_mlp_randomlocal11_source_init_distilled` | 11 | 0.1650 | 67072 | 0.9787 |
| `masked_mlp_local11_random_init_distilled` | 11 | 0.1650 | 67072 | 0.9775 |
| `masked_mlp_topk121_source_init_distilled` | 11 | 0.1650 | 67072 | 0.9800 |
| `masked_mlp_local13_source_init_distilled` | 13 | 0.2254 | 91648 | 0.9810 |
| `masked_mlp_randomlocal13_source_init_distilled` | 13 | 0.2254 | 91648 | 0.9811 |
| `masked_mlp_local13_random_init_distilled` | 13 | 0.2254 | 91648 | 0.9771 |
| `masked_mlp_topk169_source_init_distilled` | 13 | 0.2254 | 91648 | 0.9809 |
| `masked_mlp_local15_source_init_distilled` | 15 | 0.2960 | 120320 | 0.9814 |
| `masked_mlp_randomlocal15_source_init_distilled` | 15 | 0.2960 | 120320 | 0.9809 |
| `masked_mlp_local15_random_init_distilled` | 15 | 0.2960 | 120320 | 0.9756 |
| `masked_mlp_topk225_source_init_distilled` | 15 | 0.2960 | 120320 | 0.9801 |

## Read
- Best local sparse resynthesis: `masked_mlp_local15_source_init_distilled` at `0.9814` accuracy with `0.2960` of weight params left.
- The source-local-window table tests whether the MLP first layer is actually local enough to recompile into conv filters.
- The masked MLP rows test a weaker target than conv: position-specific local circuits with the same hidden width, compared against random local masks.
- Magnitude/Wanda pruning are included as dumb compression baselines; they do not change architecture.
