# More-Params Compensation Summary

This pushes beyond the grouped depth-head circuit masks by using scalar-level transformer saliency masks, then repairing with folded LoRA.

Setup:

- Model: Depth Anything V2 ViT-S checkpoint at `/home/ubuntu/checkpoints/depth_anything_v2_vits.pth`
- Dataset: DA-2K at `/home/ubuntu/vision_token_tests/datasets/DA-2K/DA-2K`
- Scalar score images: 32
- Main scalar score: `taylor2_abs`
- Target: transformer linear weights
- Fold integrity: every reported `folded` value matched `remasked`, so the folded adapter did not refill pruned entries.

## Scalar-Mask LoRA Sweep

`removed` is the number of scalar transformer weight values zeroed.

| train | eval skip | score | removed | zero frac | method | rank | epochs | lr | PEFT params | dense | pruned | folded | folded-dense | folded-pruned |
|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 64 | `abs_wgrad` | 5,014,736 | 0.236 | `lora` | 8 | 1 | 0.0001 | 589,824 | 0.9288 | 0.9139 | 0.9101 | -0.0187 | -0.0037 |
| 64 | 64 | `taylor2_abs` | 5,014,736 | 0.236 | `lora` | 4 | 1 | 0.0001 | 294,912 | 0.9288 | 0.9064 | 0.9139 | -0.0150 | +0.0075 |
| 64 | 64 | `taylor2_abs` | 5,014,736 | 0.236 | `lora` | 8 | 1 | 0.0001 | 589,824 | 0.9288 | 0.9064 | 0.9176 | -0.0112 | +0.0112 |
| 64 | 64 | `taylor2_abs` | 5,014,736 | 0.236 | `lora` | 8 | 3 | 0.00003 | 589,824 | 0.9288 | 0.9064 | 0.9101 | -0.0187 | +0.0037 |
| 64 | 64 | `taylor2_abs` | 5,014,736 | 0.236 | `lora` | 8 | 5 | 0.001 | 589,824 | 0.9288 | 0.9064 | 0.8165 | -0.1124 | -0.0899 |
| 64 | 64 | `taylor2_abs` | 5,014,736 | 0.236 | `lora-bitfit` | 4 | 1 | 0.0001 | 336,384 | 0.9288 | 0.9064 | 0.9101 | -0.0187 | +0.0037 |
| 64 | 64 | `taylor2_abs` | 8,000,000 | 0.377 | `lora` | 4 | 1 | 0.00003 | 294,912 | 0.9288 | 0.8502 | 0.8614 | -0.0674 | +0.0112 |
| 64 | 64 | `taylor2_abs` | 10,000,000 | 0.471 | `lora` | 4 | 1 | 0.00003 | 294,912 | 0.9288 | 0.7528 | 0.7640 | -0.1648 | +0.0112 |
| 64 | 128 | `taylor2_abs` | 5,014,736 | 0.236 | `lora` | 8 | 1 | 0.0001 | 589,824 | 0.9450 | 0.9223 | 0.9223 | -0.0227 | +0.0000 |
| 128 | 128 | `taylor2_abs` | 5,014,736 | 0.236 | `lora` | 8 | 1 | 0.0001 | 589,824 | 0.9450 | 0.9223 | 0.9223 | -0.0227 | +0.0000 |
| 128 | 128 | `taylor2_abs` | 5,014,736 | 0.236 | `lora` | 8 | 2 | 0.00005 | 589,824 | 0.9450 | 0.9223 | 0.9256 | -0.0194 | +0.0032 |

## Readout

- The scalar path gets much more removal than the grouped circuit mask: 5,014,736 scalar weights removed, or 23.6% of transformer linear weights.
- Best same-slice 5M repair: `taylor2_abs`, rank-8 LoRA, 1 epoch, lr `1e-4`: 0.9064 -> 0.9176.
- Best disjoint holdout 5M repair: train 128 images, eval next 128 images, rank-8 LoRA, 2 epochs, lr `5e-5`: 0.9223 -> 0.9256.
- 8M and 10M removal still recover about +0.011 over their pruned starts, but the base masks are too damaged to be useful as quality-preserving compression.
- Aggressive LoRA training is actively harmful here: the 5M rank-8, 5 epoch, lr `1e-3` run collapsed from 0.9064 pruned to 0.8165 folded.

Current best high-removal setting:

```text
score=taylor2_abs
removed=5,014,736 transformer scalar weights
LoRA=rank 8, alpha 16, masked placement
train=128 images
epochs=2
lr=5e-5
holdout folded=0.9256 vs dense=0.9450
```
