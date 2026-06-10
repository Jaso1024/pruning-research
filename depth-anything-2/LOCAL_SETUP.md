# Local Depth Anything V2 Setup

This folder is a local clone of `https://github.com/DepthAnything/Depth-Anything-V2`.

## Environment

```bash
cd upstream/Depth-Anything-V2
source .venv/bin/activate
```

Installed from the upstream `requirements.txt` in `.venv`.

## Relative Depth Smoke Test

The small relative-depth checkpoint is present at:

```text
checkpoints/depth_anything_v2_vits.pth
```

Run the bundled examples:

```bash
python run.py \
  --encoder vits \
  --img-path assets/examples \
  --outdir depth_vis_vits_smoke \
  --pred-only \
  --grayscale
```

This should write 20 PNG files.

## DA-2K Evaluation

The DA-2K archive was downloaded and extracted to:

```text
datasets/DA-2K/extracted/DA-2K
```

Run the local evaluator:

```bash
python eval_da2k.py \
  --dataset-root datasets/DA-2K/extracted/DA-2K \
  --checkpoint checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --output-json eval_outputs/da2k_vits_full.json
```

The evaluator reports both conventions because Depth Anything V2 relative outputs are direction-convention dependent for this point-pair benchmark. For the `vits` checkpoint here, larger predicted values correspond to the closer point on DA-2K.

Latest local full run:

```text
pairs: 2068
best_direction: larger
best_accuracy: 0.9516441005802708
```

## 2:4 Layer Beam Search

Run a bounded beam search over DINOv2 transformer blocks, applying 2:4 structured sparsity to every linear module in each selected block and scoring each candidate on DA-2K:

```bash
python beam_sparse24_da2k.py \
  --dataset-root datasets/DA-2K/extracted/DA-2K \
  --checkpoint checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --method wanda \
  --beam-width 2 \
  --max-depth 2 \
  --max-images 25 \
  --calibration-images 8 \
  --output-dir beam_outputs/da2k_vits_sparse24_wanda_d2
```

Use `--max-images 0` for the full DA-2K split. The default ranking uses the verified `larger` relative-depth convention for this checkpoint.

## Patch-Embedding Norm Token Pruning

Depth Anything V2 does not have discrete input token IDs or an input embedding table like an LLM. The closest analogue scores each ViT patch token after `patch_embed` by its embedding norm, keeps a percentage of patch tokens, runs DINOv2 on the shorter token sequence, and scatters kept tokens back into the full patch grid for the DPT depth head.

```bash
python eval_patch_norm_da2k.py \
  --dataset-root datasets/DA-2K/extracted/DA-2K \
  --checkpoint checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --keep-percentage 0.5 \
  --norm l2 \
  --fill-mode zero \
  --output-json eval_outputs/da2k_vits_patch_norm_keep50.json
```

`--fill-mode input` keeps the unprocessed patch+position embedding for dropped grid positions instead of zero filling them.

## Iterative Unstructured WANDA

Run the refreshed WANDA pruning loop over Depth Anything V2 transformer linear weights:

```bash
python eval_wanda_unstructured_da2k.py \
  --dataset-root datasets/DA-2K/extracted/DA-2K \
  --checkpoint checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --target transformer \
  --pruning-scope per-matrix \
  --prune-fraction 0.35 \
  --prune-chunk-fraction 0.05 \
  --calibration-images 8 \
  --output-dir eval_outputs/da2k_vits_wanda_unstructured_pf35_full
```

This mirrors the original pruning-research iterative WANDA setup: recompute `abs(weight) * input_activation_rms` on the current pruned model, prune the next chunk, reapply masks, and repeat.

Global unstructured pruning is also supported:

```bash
python eval_wanda_unstructured_da2k.py \
  --dataset-root datasets/DA-2K/extracted/DA-2K \
  --checkpoint checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --target transformer \
  --pruning-scope global \
  --prune-fraction 0.50 \
  --prune-chunk-fraction 0.05 \
  --calibration-images 8 \
  --output-dir eval_outputs/da2k_vits_wanda_global_pf50_full
```

Local DA-2K results:

```text
per-matrix 35% full: 1896/2068 = 0.9168278529980658
per-matrix 40% full: 1851/2068 = 0.8950676982591876
raw global 50% full: 1188/2068 = 0.574468085106383
matrix-mean global 50% 25-image subset: 43/55 = 0.7818181818181819
```

Run matrix-mean global pruning with dense-output gradient repair while keeping the calibration images out of evaluation:

```bash
python eval_wanda_unstructured_da2k.py \
  --dataset-root datasets/DA-2K/extracted/DA-2K \
  --checkpoint checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --target transformer \
  --pruning-scope global \
  --score-normalization matrix-mean \
  --prune-fraction 0.50 \
  --prune-chunk-fraction 0.05 \
  --calibration-images 8 \
  --exclude-calibration-from-eval \
  --repair-steps 3 \
  --repair-lr 3e-5 \
  --output-dir eval_outputs/da2k_vits_wanda_global_mean_pf50_repair3_lr3e-5_full_heldout
```

The repair objective is MSE distillation against cached dense Depth Anything outputs on the calibration images. Local held-out DA-2K results with the first 8 images excluded from evaluation:

```text
dense held-out baseline: 1952/2050 = 0.9521951219512195
matrix-mean global 50% + 3 repair steps @ 3e-5: 1597/2050 = 0.7790243902439025
```

## Quantization Experiments

Run FP16, RTN INT8, GPTQ INT8, SmoothQuant INT8, and SmoothQuant plus Hessian-update INT8 experiments with:

```bash
python eval_quant_da2k.py \
  --dataset-root datasets/DA-2K/extracted/DA-2K \
  --checkpoint checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --method smoothquant-int8 \
  --base-dtype fp16 \
  --calibration-images 8 \
  --smooth-alpha 0.25 \
  --eval-batch-size 16 \
  --output-dir eval_outputs/quant_fullheldout_smoothquant_a0p25_int8
```

Supported `--method` values are:

```text
fp32
fp16
rtn-int8
gptq-int8
smoothquant-int8
smoothquant-hessian-int8
```

Implementation notes:

```text
INT runs target the 48 DINOv2 transformer Linear modules only.
DPT convolutional decoder layers remain floating point.
INT evaluation uses PyTorch fake quant/dequant modules for accuracy testing, not custom int GEMM kernels.
GPTQ is weight-only unless using smoothquant-hessian-int8.
SmoothQuant uses per-input-channel smoothing scales from calibration activation absmax statistics.
smoothquant-hessian-int8 applies GPTQ's Hessian correction to the smoothed weights and transformed input Hessian.
Recent-method knobs include --rotation hadamard, --weight-quant per-group, --act-quant per-token-group, and clipping ratios.
Use --eval-batch-size to bucket same-shaped transformed images and run batched model inference.
```

The 100-image held-out subset sweep used the first 8 selected images for calibration and evaluated on the remaining 92 images / 189 point pairs:

```text
FP16 subset: 170/189 = 0.8994708994708994
RTN INT8 subset: 171/189 = 0.9047619047619048
GPTQ INT8 subset: 170/189 = 0.8994708994708994
SmoothQuant alpha 0.25 subset: 172/189 = 0.91005291005291
SmoothQuant alpha 0.50 subset: 170/189 = 0.8994708994708994
SmoothQuant alpha 0.75 subset: 171/189 = 0.9047619047619048
SmoothQuant alpha 1.00 subset: 171/189 = 0.9047619047619048
SmoothQuant + Hessian alpha 0.25 subset: 170/189 = 0.8994708994708994
```

Full held-out DA-2K runs use the first 8 selected images for calibration and exclude them from evaluation, leaving 1025 images / 2050 point pairs:

```text
FP32 held-out baseline: 1951/2050 = 0.9517073170731707
FP16 held-out: 1950/2050 = 0.9512195121951219
RTN INT8 held-out: 1954/2050 = 0.953170731707317
GPTQ INT8 held-out: 1953/2050 = 0.9526829268292683
SmoothQuant alpha 0.25 INT8 held-out: 1956/2050 = 0.9541463414634146
SmoothQuant + Hessian alpha 0.25 INT8 held-out: 1952/2050 = 0.9521951219512195
```

SmoothQuant INT4 tests:

```text
W4A4 subset alpha sweep, 189 held-out pairs:
alpha 0.00: 125/189 = 0.6613756613756614
alpha 0.10: 106/189 = 0.5608465608465608
alpha 0.25: 106/189 = 0.5608465608465608
alpha 0.50: 116/189 = 0.6137566137566137
alpha 0.75: 121/189 = 0.6402116402116402
alpha 1.00: 125/189 = 0.6613756613756614

W4A4 SmoothQuant + Hessian subset alpha sweep, 189 held-out pairs:
alpha 0.00: 129/189 = 0.6825396825396826
alpha 0.10: 117/189 = 0.6190476190476191
alpha 0.25: 115/189 = 0.6084656084656085
alpha 0.50: 127/189 = 0.671957671957672
alpha 0.75: 115/189 = 0.6084656084656085
alpha 1.00: 121/189 = 0.6402116402116402

W4A8 subset alpha sweep, 189 held-out pairs:
alpha 0.00: 166/189 = 0.8783068783068783
alpha 0.25: 167/189 = 0.8835978835978836
alpha 0.50: 166/189 = 0.8783068783068783
alpha 0.75: 139/189 = 0.7354497354497355
alpha 1.00: 134/189 = 0.708994708994709

W4A16 subset alpha sweep, 189 held-out pairs:
alpha 0.00: 167/189 = 0.8835978835978836
alpha 0.25: 166/189 = 0.8783068783068783
alpha 0.50: 166/189 = 0.8783068783068783
alpha 0.75: 141/189 = 0.746031746031746
alpha 1.00: 133/189 = 0.7037037037037037

W4A4 SmoothQuant alpha 0.00 full held-out: 1216/2050 = 0.593170731707317
W4A4 SmoothQuant + Hessian alpha 0.00 full held-out, eval batch 64: 1278/2050 = 0.6234146341463415
W4A8 SmoothQuant alpha 0.25 full held-out: 1837/2050 = 0.8960975609756098
W4A8 SmoothQuant + Hessian alpha 0.25 full held-out, eval batch 16: 1907/2050 = 0.9302439024390244
W4A8 SmoothQuant + Hessian alpha 0.25 full held-out, eval batch 64: 1910/2050 = 0.9317073170731708
```

W4A4 activation-quantization-focused tests used SmoothQuant + Hessian/GPTQ with block-Hadamard rotation and group-wise activation scales:

```bash
python eval_quant_da2k.py \
  --dataset-root datasets/DA-2K/extracted/DA-2K \
  --checkpoint checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --method smoothquant-hessian-int8 \
  --base-dtype fp16 \
  --calibration-images 8 \
  --weight-bits 4 \
  --activation-bits 4 \
  --smooth-alpha 0.00 \
  --hessian-tokens-per-image 128 \
  --eval-batch-size 64 \
  --weight-quant per-group \
  --weight-group-size 128 \
  --act-quant per-token-group \
  --activation-group-size 128 \
  --rotation hadamard \
  --rotation-group-size 128 \
  --rotation-seed 0 \
  --output-dir eval_outputs/quant_lit_full_sqh_w4a4_rot_group128_seed0_a0
```

Subset sweep highlights, 189 held-out pairs:

```text
W4A4 SmoothQuant + Hessian, no rotation: 129/189 = 0.6825396825396826
Hadamard rotation only, alpha 0.50: 172/189 = 0.91005291005291
Hadamard rotation + per-group W128/A128, alpha 0.00: 173/189 = 0.9153439153439153
Hadamard rotation + per-group W128/A16, alpha 0.00: 173/189 = 0.9153439153439153
Hadamard rotation + per-group W128/A128, activation clip 0.95: 173/189 = 0.9153439153439153
Hadamard rotation group 32 + per-group W128/A64: 172/189 = 0.91005291005291
Hadamard rotation group 64 + per-group W128/A64: 165/189 = 0.873015873015873
```

Full held-out W4A4 results:

```text
W4A4 SmoothQuant + Hessian, no rotation: 1278/2050 = 0.6234146341463415
W4A4 SmoothQuant + Hessian + Hadamard rotation only, alpha 0.50: 1873/2050 = 0.9136585365853659
W4A4 SmoothQuant + Hessian + Hadamard rotation + per-group W128/A128, alpha 0.00: 1878/2050 = 0.9160975609756098
```

## Component Ablation Sweep

Run one-component-at-a-time ablations over transformer residual components:

```bash
python eval_component_ablation_da2k.py \
  --dataset-root datasets/DA-2K/extracted/DA-2K \
  --checkpoint checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --component-types block,attn,mlp \
  --max-images 100 \
  --output-dir eval_outputs/da2k_vits_component_ablation_residual_i100
```

`block` ablation returns the block input, effectively bypassing the block. `attn` and `mlp` ablations replace that residual branch output with zeros. Hooks are removed between runs.

Full DA-2K confirmation for selected candidates:

```bash
python eval_component_ablation_da2k.py \
  --dataset-root datasets/DA-2K/extracted/DA-2K \
  --checkpoint checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --component-types block,attn,mlp \
  --components block_10_mlp,block_11_mlp,block_10_attn,block_02_attn,block_10,block_01_mlp,block_05,block_11_attn \
  --max-images 0 \
  --output-dir eval_outputs/da2k_vits_component_ablation_selected_full
```

Selected full DA-2K results:

```text
dense baseline: 1969/2068 = 0.9521276595744681
without block_10_mlp: 1948/2068 = 0.941972920696325
without block_11_mlp: 1945/2068 = 0.940522243713733
without block_10_attn: 1930/2068 = 0.9332688588007737
without block_10: 1909/2068 = 0.9231141199226306
without block_02_attn: 1893/2068 = 0.9153771760154739
without block_11_attn: 1420/2068 = 0.6866537717601547
without block_05: 1265/2068 = 0.6117021276595744
without block_01_mlp: 1168/2068 = 0.5647969052224371
```

Run beam search over compounded component ablations:

```bash
python beam_component_ablation_da2k.py \
  --dataset-root datasets/DA-2K/extracted/DA-2K \
  --checkpoint checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --component-types block,attn,mlp \
  --components block_10_mlp,block_11_mlp,block_10_attn,block_02_attn,block_10,block_09_mlp,block_06,block_08_mlp,block_01_attn,block_04_attn,block_02,block_04_mlp \
  --beam-width 4 \
  --max-depth 4 \
  --max-images 100 \
  --output-dir eval_outputs/da2k_vits_component_ablation_beam_top12_i100_d4_bw4
```

Full DA-2K confirmation for selected compounded states:

```bash
python beam_component_ablation_da2k.py \
  --dataset-root datasets/DA-2K/extracted/DA-2K \
  --checkpoint checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --component-types block,attn,mlp \
  --components block_10_mlp,block_11_mlp,block_10_attn,block_02_attn,block_09_mlp \
  --eval-states 'block_10_mlp,block_11_mlp;block_10_attn,block_11_mlp;block_02_attn,block_10_mlp;block_09_mlp,block_10_attn;block_09_mlp,block_10_attn,block_10_mlp;block_09_mlp,block_10_attn,block_10_mlp,block_11_mlp' \
  --max-images 0 \
  --output-dir eval_outputs/da2k_vits_component_ablation_beam_selected_full
```

Selected compounded full DA-2K results:

```text
dense baseline: 1969/2068 = 0.9521276595744681
without block_10_mlp + block_11_mlp: 1924/2068 = 0.9303675048355899
without block_09_mlp + block_10_attn + block_10_mlp: 1905/2068 = 0.9211798839458414
without block_10_attn + block_11_mlp: 1901/2068 = 0.9192456479690522
without block_09_mlp + block_10_attn: 1885/2068 = 0.9115087040618955
without block_02_attn + block_10_mlp: 1877/2068 = 0.9076402321083172
without block_09_mlp + block_10_attn + block_10_mlp + block_11_mlp: 1858/2068 = 0.8984526112185687
```
