# Mirage difference-verification notes

Source: Yu et al., "Can Vision Models Truly Forget? Mirage: Representation-Level Certification of Visual Unlearning", arXiv:2605.20282, submitted 2026-05-19.

## Relevant method

Mirage is an auditing framework, not an optimizer. It tests whether a model variant that looks changed at the output level still preserves hidden representation structure.

Core diagnostics:

1. Linear Probe Recovery (LPR): freeze embeddings at a layer and train a regularized logistic probe to recover a target binary property. Mirage interprets absolute probe accuracy relative to a retrained baseline, not by itself.
2. Centered Kernel Alignment (CKA): compute linear CKA between representations from model variants on the same examples to measure global geometric alignment.
3. Feature separability: Fisher-style class separability, \|\mu_a-\mu_b\|^2 / (tr Sigma_a + tr Sigma_b), independent of probe optimization.
4. Layer-wise recovery: repeat probe recovery at early/middle/late layers to see where the residual structure lives.

Important framing:

- Output-level behavior can change while internal geometry remains close.
- Absolute verifier scores are misleading without a baseline/reference.
- Linear probes are intended as conservative, cheap lower bounds; nonlinear probes can reveal more.

## Adaptation to AirBench pruning scaffold

Our analogue of "unlearned vs original/retrained" is:

- original: 500-step full checkpoint
- pruned: fixed random 40% layer-1 mask
- recovered: pruned model after a proxy loss update

The faithful first step is to audit representation differences at the same layer sites already used by the recovery losses:

- layer-wise sites: layers.1.input, layers.1.output, layers.2.input, layers.2.output, layers.3.input, layers.3.output
- CKA(original, variant) at each site
- class separability at each site using CIFAR labels
- a linear probe at each site predicting CIFAR labels from frozen activations
- an origin verifier at each site predicting whether an activation came from original or variant; lower accuracy means the variant is harder to distinguish from original

For optimization, an origin-verifier loss is tempting but would be adversarial and easy to overfit. Safer first experiment: add fast post-hoc audits to every pruning-recovery result and compare the audit deltas against final validation accuracy.

