# Why Structured Circuit Masks Fall Behind Global Magnitude

Full DA-2K baseline: `1969 / 2068 = 0.9521276596`.

## Short Answer

Global magnitude survives high masked-value budgets because it removes many tiny individual weights spread across all transformer Linear layers. It does not remove whole representational channels.

The structured circuit masks remove entire rows/channels/heads/MLP groups. Those groups are individually low-impact on a 32-image probe, but their effects are not additive. Once enough whole groups are removed from the same decoder or transformer blocks, the representation loses bandwidth and accuracy drops sharply.

## Evidence

### 1. The aggressive structured selector mostly prunes decoder-head channels first

For `stability_param@500k`, selected groups were:

| Kind | Nodes | Params | Share |
|---|---:|---:|---:|
| `head_channel_group` | 15 | 461,040 | 86.2% |
| `head_input_channel_group` | 8 | 73,728 | 13.8% |

Top modules:

| Module | Nodes | Params | Share |
|---|---:|---:|---:|
| `depth_head.resize_layers.3` | 7 | 387,184 | 72.4% |
| `depth_head.scratch.refinenet4.resConfUnit1.conv1` | 8 | 73,792 | 13.8% |
| `depth_head.scratch.refinenet4.resConfUnit1.conv2` | 8 | 73,792 | 13.8% |

Every selected node had individual `correct_drop=0` on the 32-image circuit probe. Sixteen of them had exactly zero individual score.

### 2. The zero-score refinenet chunk is genuinely safe, but the later decoder channels accumulate damage

Diagnostic `stability_param` value-budget sweep:

| Target | Actual masked | Accuracy | Drop |
|---:|---:|---:|---:|
| 150k | 202,896 | 0.9521276596 | 0 |
| 250k | 258,208 | 0.9511605416 | 2 |
| 350k | 368,832 | 0.9501934236 | 4 |
| 450k | 479,456 | 0.9497098646 | 5 |
| 550k | 590,080 | 0.9482591876 | 8 |

The first free chunk is mostly all input/output groups for `refinenet4.resConfUnit1.conv1/conv2`. The drop begins as more `depth_head.resize_layers.3` channel groups are stacked.

### 3. Transformer-only structured pruning is worse, not better

When decoder-head candidates are excluded and only transformer groups are allowed:

| Target | Actual masked | Accuracy | Drop |
|---:|---:|---:|---:|
| 500k | 516,800 | 0.9366537718 | 32 |
| 1M | 1,008,960 | 0.9153771760 | 76 |
| 2M | 2,017,984 | 0.7379110251 | 443 |
| 3M | 3,002,432 | 0.6663442940 | 591 |

So the decoder-head groups were not the main problem. They were actually cheaper than whole transformer MLP/head groups. The deeper issue is coarse structured deletion.

### 4. Structured magnitude is not equivalent to global magnitude

Global magnitude is strong:

| Masked values | Accuracy | Drop |
|---:|---:|---:|
| 1,020,352 | 0.9535783366 | -3 |
| 2,016,304 | 0.9516441006 | 1 |
| 3,006,528 | 0.9506769826 | 3 |
| 5,014,736 | 0.9318181818 | 42 |

But structured magnitude collapses:

| Target | Actual masked | Accuracy | Drop |
|---:|---:|---:|---:|
| 500k | 503,648 | 0.0 | 1969 |
| 1M | 1,006,992 | 0.0 | 1969 |

At 500k, structured magnitude selected both input groups of `depth_head.scratch.output_conv2.2`, the final output conv. Those groups have low weight magnitude but nonzero circuit impact; removing both makes the depth output degenerate.

### 5. A simple circuit guard does not fix structured magnitude

`safe_magnitude` penalizes groups with positive individual circuit `correct_drop`. It avoids the final output-conv failure, but still performs badly:

| Target | Actual masked | Accuracy | Drop |
|---:|---:|---:|---:|
| 500k | 517,152 | 0.7886847195 | 338 |
| 1M | 1,009,312 | 0.7412959381 | 436 |
| 2M | 2,006,624 | 0.5807543520 | 768 |

The first 500k safe-magnitude mask is mostly block-0 attention/MLP groups, so this is a separate failure mode: many individually safe transformer groups interact badly when removed together.

## Mechanism

The comparison is not apples-to-apples in functional terms:

- **Global unstructured magnitude** zeroes small weights across many rows and matrices. At 3M zeros, it removes about `14.2%` of transformer Linear weights, but spread across all 48 transformer Linear matrices. It behaves like noise trimming.
- **Structured masks** delete whole channels/heads/MLP hidden groups. A 32-channel MLP group removes all `fc1` rows plus matching `fc2` columns. A decoder channel group removes an entire feature pathway. These are coherent representational units, so deleting many of them changes the computation much more than deleting the same number of tiny scalar weights.
- **Single-node circuit scores are not enough.** The selected groups often have `correct_drop=0` individually, but grouped deletions interact. The score estimates local marginal effect, while high-budget pruning needs cumulative conditional effect.
- **The 32-image probe is too small for high-budget selection.** It can identify obviously critical groups, but it does not reliably rank hundreds of individually harmless groups by joint replaceability.

## Implications

For structural compression, the next metric should not be plain circuit rank or plain group magnitude. It should be group-aware and cumulative:

1. Add per-module caps, e.g. no more than `5-10%` channel removal per module before re-scoring.
2. Use cumulative greedy validation on a held-out subset after each small batch of groups.
3. Score groups by reconstruction/activation error, not just individual DA-2K decision flips.
4. Prefer paired structural surgery: if removing an output channel, also account for downstream input-column removal in the score and parameter count.
5. Use circuit scores as hard protection for final-output and high-drop groups, but not as the only ranking signal.

The most likely next strong baseline is **group-structured magnitude or Wanda with module caps plus cumulative re-scoring**, not the current one-shot ranking.
