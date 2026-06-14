# Circuit LoRA Compensation Summary

Setup:

- Model: Depth Anything V2 ViT-S checkpoint at `/home/ubuntu/checkpoints/depth_anything_v2_vits.pth`
- Dataset: DA-2K at `/home/ubuntu/vision_token_tests/datasets/DA-2K/DA-2K`
- Circuit source: `/home/ubuntu/remote-work/depth-anything-2/eval_outputs/subcircuit_fine_stratified32_g32_h16/summary.json`
- Selection: `stability_param`
- Candidate kinds: `head_channel_group`, `head_input_channel_group`, `mlp_group`, `attn_v_head`, `attn_q_head`, `attn_k_head`, `attn_proj_group`
- Training: first 64 images, 5 epochs, no train/eval overlap

## LoRA Compensation

`removed values` is the number of tensor values zeroed by the circuit mask. `PEFT params` is temporary trainable adapter capacity before folding. `folded` is the merged model without reapplying a mask; `remasked` is after reapplying the mask as an integrity check.

| eval skip | placement | rank | removed values | PEFT params | dense | pruned | folded | remasked | folded-dense | folded-pruned | overlap |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | `head` | 4 | 534,768 | 103,364 | 0.9288 | 0.9288 | 0.9363 | 0.9363 | +0.0075 | +0.0075 | 0 |
| 64 | `head` | 4 | 1,020,352 | 103,364 | 0.9288 | 0.9213 | 0.9363 | 0.9363 | +0.0075 | +0.0150 | 0 |
| 64 | `head` | 4 | 1,518,336 | 103,364 | 0.9288 | 0.9288 | 0.9363 | 0.9363 | +0.0075 | +0.0075 | 0 |
| 64 | `head` | 4 | 2,016,304 | 103,364 | 0.9288 | 0.9213 | 0.9326 | 0.9326 | +0.0037 | +0.0112 | 0 |
| 64 | `head` | 4 | 3,006,528 | 103,364 | 0.9288 | 0.8914 | 0.8652 | 0.8652 | -0.0637 | -0.0262 | 0 |
| 64 | `head+masked` | 4 | 2,016,304 | 140,228 | 0.9288 | 0.9213 | 0.9288 | 0.9288 | +0.0000 | +0.0075 | 0 |
| 64 | `head+masked` | 4 | 3,006,528 | 155,588 | 0.9288 | 0.8914 | 0.8951 | 0.8951 | -0.0337 | +0.0037 | 0 |
| 64 | `head+masked` | 8 | 3,006,528 | 311,176 | 0.9288 | 0.8914 | 0.8989 | 0.8989 | -0.0300 | +0.0075 | 0 |
| 128 | `head` | 4 | 1,020,352 | 103,364 | 0.9450 | 0.9320 | 0.9450 | 0.9450 | +0.0000 | +0.0129 | 0 |
| 128 | `head` | 4 | 1,518,336 | 103,364 | 0.9450 | 0.9288 | 0.9417 | 0.9417 | -0.0032 | +0.0129 | 0 |
| 128 | `head` | 4 | 2,016,304 | 103,364 | 0.9450 | 0.9159 | 0.9353 | 0.9353 | -0.0097 | +0.0194 | 0 |

## No-Compensation Reference

Same-slice no-compensation references from `/home/ubuntu/eval_outputs/pruning_granularity_audit.md`:

| eval skip | removed values | dense | pruned | pruned-dense |
|---:|---:|---:|---:|---:|
| 64 | 534,768 | 0.9288 | 0.9288 | +0.0000 |
| 64 | 1,020,352 | 0.9288 | 0.9213 | -0.0075 |
| 128 | 534,768 | 0.9450 | 0.9385 | -0.0065 |
| 128 | 1,020,352 | 0.9450 | 0.9320 | -0.0129 |

## Readout

- The useful compensated budget is currently around 1.0M removed values. It exactly matches dense on the disjoint skip-128 holdout and beats the pruned model by +0.0129.
- The 1.5M and 2.0M masks still benefit from LoRA repair, but on the skip-128 holdout they land below dense by -0.0032 and -0.0097.
- The 3.0M mask is too destructive for these adapters. Adding masked-module LoRA and doubling rank improves over the pruned model, but remains far below dense.
- `folded` and `remasked` match in every run, so the folded adapter is not relying on reintroducing already pruned weights.
