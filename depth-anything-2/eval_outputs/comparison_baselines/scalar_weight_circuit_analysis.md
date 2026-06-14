# Scalar Weight Circuit Analysis

## Method

We can estimate scalar-level circuits without brute-force ablating every weight. For each DA2K point-pair margin

`m = depth(point1) - depth(point2)`

and scalar weight `w`, zeroing that scalar gives a first-order margin change:

`delta_m ~= -w * grad_m(w)`.

So:

- `positive_wgrad = max(w * grad_m(w), 0)` marks scalars whose removal should lower the correct margin. These are scalar circuit-protection weights.
- `abs_wgrad = abs(w * grad_m(w))` is first-order scalar saliency magnitude. Low values are prune candidates.

The implementation is in `eval_scalar_weight_circuit_da2k.py`. It scores scalar weights on the first 32 stratified DA2K images, then evaluates global unstructured prefix pruning on full DA2K.

## Scalar Circuits Found

Top protected module-level circuits by `positive_wgrad`:

| rank | module | positive_wgrad | per weight |
|---:|---|---:|---:|
| 1 | `pretrained.blocks.8.mlp.fc2` | 1944.99 | 0.003298 |
| 2 | `pretrained.blocks.7.mlp.fc2` | 1920.03 | 0.003255 |
| 3 | `pretrained.blocks.9.mlp.fc2` | 1816.30 | 0.003079 |
| 4 | `pretrained.blocks.6.mlp.fc2` | 1810.58 | 0.003070 |
| 5 | `pretrained.blocks.5.mlp.fc2` | 1760.93 | 0.002986 |
| 6 | `pretrained.blocks.10.mlp.fc2` | 1656.96 | 0.002809 |
| 7 | `pretrained.blocks.11.attn.qkv` | 1449.01 | 0.003276 |
| 8 | `pretrained.blocks.10.attn.qkv` | 1448.70 | 0.003275 |
| 9 | `pretrained.blocks.8.attn.qkv` | 1436.92 | 0.003248 |
| 10 | `pretrained.blocks.9.attn.qkv` | 1435.52 | 0.003245 |

Top QKV head-level groups:

| rank | group | positive_wgrad | per weight |
|---:|---|---:|---:|
| 1 | `block11.v.head1` | 220.66 | 0.008979 |
| 2 | `block6.v.head1` | 172.20 | 0.007007 |
| 3 | `block9.v.head1` | 161.62 | 0.006577 |
| 4 | `block8.v.head3` | 158.61 | 0.006454 |
| 5 | `block9.v.head0` | 153.37 | 0.006240 |
| 6 | `block11.v.head3` | 151.74 | 0.006174 |
| 7 | `block9.v.head3` | 150.31 | 0.006116 |
| 8 | `block10.v.head2` | 149.98 | 0.006103 |
| 9 | `block10.v.head1` | 146.48 | 0.005960 |
| 10 | `block8.v.head2` | 143.61 | 0.005843 |

Top MLP hidden-channel groups:

| rank | group | positive_wgrad | per weight |
|---:|---|---:|---:|
| 1 | `block11.mlp.hidden52` | 17.17 | 0.022357 |
| 2 | `block8.mlp.hidden371` | 10.92 | 0.014215 |
| 3 | `block8.mlp.hidden107` | 10.78 | 0.014031 |
| 4 | `block5.mlp.hidden69` | 9.83 | 0.012801 |
| 5 | `block10.mlp.hidden568` | 9.43 | 0.012275 |
| 6 | `block8.mlp.hidden596` | 9.03 | 0.011753 |
| 7 | `block8.mlp.hidden443` | 8.99 | 0.011701 |
| 8 | `block9.mlp.hidden1457` | 8.81 | 0.011470 |
| 9 | `block5.mlp.hidden979` | 8.69 | 0.011310 |
| 10 | `block8.mlp.hidden113` | 8.51 | 0.011087 |

The scalar circuit is not uniformly the same as the earlier group-ablation result. Earlier group ablation found a removable `block11` QKV slice. Scalar protection says the important task-supporting attention circuit is concentrated in V heads, especially `block11.v.head1`, while much of the lowest-effect scalar mass sits in early transformer blocks.

## Pruning Results

Dense full DA2K reference: `1969 / 2068 = 0.9521276596`.

| method | masked values | correct | accuracy | drop vs dense |
|---|---:|---:|---:|---:|
| global magnitude | 534,768 | 1968 | 0.9516441006 | 1 |
| global Wanda | 534,768 | 1971 | 0.9530947776 | -2 |
| scalar `abs_wgrad` | 534,768 | 1964 | 0.9497098646 | 5 |
| scalar-protected magnitude alpha=4 | 534,768 | 1965 | 0.9501934236 | 4 |
| global magnitude | 1,020,352 | 1972 | 0.9535783366 | -3 |
| global Wanda | 1,020,352 | 1967 | 0.9511605416 | 2 |
| scalar `abs_wgrad` | 1,020,352 | 1965 | 0.9501934236 | 4 |
| scalar-protected magnitude alpha=4 | 1,020,352 | 1969 | 0.9521276596 | 0 |
| global magnitude | 1,518,336 | 1967 | 0.9511605416 | 2 |
| global Wanda | 1,518,336 | 1965 | 0.9501934236 | 4 |
| scalar `abs_wgrad` | 1,518,336 | 1958 | 0.9468085106 | 11 |
| scalar-protected magnitude alpha=4 | 1,518,336 | 1957 | 0.9463249516 | 12 |
| global magnitude | 2,016,304 | 1968 | 0.9516441006 | 1 |
| global Wanda | 2,016,304 | 1963 | 0.9492263056 | 6 |
| scalar `abs_wgrad` | 2,016,304 | 1953 | 0.9443907157 | 16 |
| scalar-protected magnitude alpha=4 | 2,016,304 | 1950 | 0.9429400387 | 19 |
| global magnitude | 3,006,528 | 1966 | 0.9506769826 | 3 |
| global Wanda | 3,006,528 | 1942 | 0.9390715667 | 27 |
| scalar `abs_wgrad` | 3,006,528 | 1954 | 0.9448742747 | 15 |
| scalar-protected magnitude alpha=4 | 3,006,528 | 1848 | 0.8936170213 | 121 |
| global magnitude | 5,014,736 | 1927 | 0.9318181818 | 42 |
| global Wanda | 5,014,736 | 1773 | 0.8573500967 | 196 |
| scalar `abs_wgrad` | 5,014,736 | 1946 | 0.9410058027 | 23 |
| scalar-protected magnitude alpha=4 | 5,014,736 | 1715 | 0.8293036750 | 254 |

## Read

Scalar-level circuits are feasible and useful, but not as a naive replacement for magnitude at all sparsity levels.

- `abs_wgrad` overfits less than Wanda at high masks and beats global magnitude at the hardest 5.0M-zero point by 19 DA2K decisions.
- At small and mid budgets, plain global magnitude remains stronger.
- Strong scalar protection on magnitude (`alpha=4`) is too aggressive. It preserves the 1.0M point, then forces pruning into weights that matter globally and collapses at 3.0M+.
- The scalar circuit structure says the protected circuit is mostly mid/late `fc2` MLP outputs plus V-projection attention heads. The removable scalar mass is mostly early-block transformer weights.

Next practical direction: use scalar circuits as a capped, layer-balanced term rather than a global multiplier. The failure mode is global ranking concentrating too much pruning in sequential block prefixes.

## Artifacts

- `eval_outputs/scalar_weight_circuit_abs_wgrad_s32_full/summary.json`
- `eval_outputs/scalar_weight_circuit_hybrid_protect_mag4_s32_full/summary.json`
- `eval_outputs/comparison_baselines/aggressive_masked_value_ladder.md`
