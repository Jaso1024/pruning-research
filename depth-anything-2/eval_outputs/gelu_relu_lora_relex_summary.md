# GELU to ReLU LoRA RELEX Summary

RELEX was adapted to the sandwich LoRA repair in two ways: folded effective weight deltas, and trainable LoRA factor deltas. Both used a 20-step observed window.

## Full DA-2K

| run | accuracy | correct | ties | note |
|---|---:|---:|---:|---|
| `baseline_dense_full` | 0.952128 | 1969/2068 |  | prior dense GELU |
| `best_200step_sandwich_radam` | 0.918279 | 1899/2068 | 0 | prior optimizer sweep |
| `raw_20step_sandwich_adagrad` | 0.888781 | 1838/2068 | 0 | prior 20-step optimizer sweep |
| `relex_folded_target20_full` | 0.882495 | 1825/2068 | 0 | full folded-delta RELEX target=20 |

## 32-image Target Sweep

| run | accuracy | correct | ties |
|---|---:|---:|---:|
| `relex_folded_target20_32img` | 0.901408 | 64/71 | 0 |
| `relex_folded_target40_32img` | 0.760563 | 54/71 | 1 |
| `relex_folded_target60_32img` | 0.028169 | 2/71 | 69 |
| `relex_folded_target100_32img` | 0.000000 | 0/71 | 71 |
| `relex_folded_target200_32img` | 0.000000 | 0/71 | 71 |
| `relex_factor_target20_32img` | 0.901408 | 64/71 | 0 |
| `relex_factor_target40_32img` | 0.000000 | 0/71 | 71 |
| `relex_factor_target100_32img` | 0.000000 | 0/71 | 71 |
| `relex_factor_target200_32img` | 0.000000 | 0/71 | 71 |

Conclusion: long-horizon RELEX extrapolation is not working in this LoRA repair regime; target 40+ usually collapses, and target 20 rank-1 reconstruction is worse than the raw 20-step Adagrad repair on full DA-2K.
