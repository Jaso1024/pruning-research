# DA-2K Post-Layer Token Pruning Summary

- Model: Depth Anything V2 Small (`vits`)
- Dataset: DA-2K, 1033 images / 2068 point pairs
- Dense baseline accuracy: `0.951644`
- Pruned DPT features are restored to the full patch grid with zero fill.
- Token-work ratio is an approximate transformer token-work estimate, not wall-clock speed.

## Hidden-State Diagonal Mahalanobis

Scores are computed per image from the hidden patch tokens after the listed block.
Higher Mahalanobis-distance tokens are kept.

| keep | after block 0 acc | after block 1 acc | block 1 token-work |
|---:|---:|---:|---:|
| 0.99 | 0.954062 | 0.955029 | 0.9919 |
| 0.98 | 0.954545 | 0.955029 | 0.9835 |
| 0.95 | 0.950193 | 0.953095 | 0.9586 |
| 0.90 | 0.945841 | 0.950677 | 0.9168 |
| 0.85 | 0.937137 | 0.944391 | 0.8753 |
| 0.80 | 0.934720 | 0.939555 | 0.8335 |
| 0.75 | 0.926983 | 0.933269 | 0.7918 |
| 0.70 | 0.917795 | 0.931335 | 0.7503 |
| 0.65 | 0.909574 | 0.923114 | 0.7085 |
| 0.60 | 0.894101 | 0.914894 | 0.6669 |
| 0.55 | 0.883946 | 0.901354 | 0.6252 |
| 0.50 | 0.868956 | 0.892166 | 0.5834 |
| 0.45 | 0.851064 | 0.869923 | 0.5419 |
| 0.40 | 0.823985 | 0.850097 | 0.5002 |
| 0.35 | 0.797389 | 0.833172 | 0.4586 |
| 0.30 | 0.768859 | 0.816248 | 0.4168 |
| 0.25 | 0.735493 | 0.769826 | 0.3752 |
| 0.20 | 0.695841 | 0.731625 | 0.3335 |

Block 1 is the better pruning point across the sweep. The useful aggressive range is around `0.70` to `0.55`: `0.70` keeps `0.931335` accuracy at about `0.7503` transformer token-work, while `0.55` is the lowest tested keep ratio that stays above `0.90` accuracy.

## Attention-Mass Pruning

Patch-query incoming attention mass is used to select the smallest token set whose cumulative mass reaches the threshold.

| prune after block | mass | accuracy | keep ratio | token-work |
|---:|---:|---:|---:|---:|
| 0 | 0.99 | 0.941489 | 0.9484 | 0.9527 |
| 0 | 0.95 | 0.887331 | 0.8177 | 0.8329 |
| 0 | 0.90 | 0.849130 | 0.6987 | 0.7238 |
| 2 | 0.95 | 0.946325 | 0.9205 | 0.9404 |
| 2 | 0.90 | 0.936170 | 0.8508 | 0.8881 |
| 2 | 0.80 | 0.922631 | 0.7230 | 0.7923 |
| 5 | 0.95 | 0.947292 | 0.8955 | 0.9478 |
| 5 | 0.90 | 0.943424 | 0.8135 | 0.9068 |
| 5 | 0.80 | 0.928433 | 0.6739 | 0.8369 |
| 8 | 0.95 | 0.952128 | 0.8675 | 0.9669 |
| 8 | 0.90 | 0.945358 | 0.7737 | 0.9434 |
| 8 | 0.80 | 0.931818 | 0.6241 | 0.9060 |

Attention-mass pruning after block 0 is destructive. It becomes much less destructive after a few dense blocks, but the best compute/accuracy tradeoffs from these runs are still weaker than hidden-state diagonal Mahalanobis after block 1.
