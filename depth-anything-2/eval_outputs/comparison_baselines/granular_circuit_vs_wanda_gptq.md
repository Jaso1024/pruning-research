# Granular Circuit Pruning vs Wanda / Magnitude / Quantization

Full DA-2K, 2068 point pairs. Dense FP32 reference: `1969 / 2068 = 0.9521276596`.

Parameter accounting:

- Dense VITS checkpoint tensor params: `24,785,089`
- Transformer Linear weights targeted by Wanda/magnitude/quantization: `21,233,664`
- Circuit pruning here masks rows in `pretrained.blocks.11.attn.qkv`; Wanda/magnitude masks unstructured transformer Linear weights.
- These pruning runs are still shape-preserving masks, so checkpoint file size does not shrink until structural compaction is implemented.

## Matched Zero-Count Pruning

| Method | Scope | Masked values | Active params left | Correct | Accuracy | Delta vs dense |
|---|---|---:|---:|---:|---:|---:|
| circuit `stability@16` | block 11 head 1 Q/K rows | 6,160 | 24,778,929 | 1974 | 0.9545454545 | +5 |
| global magnitude | all transformer Linears | 6,160 | 24,778,929 | 1969 | 0.9521276596 | 0 |
| global Wanda | all transformer Linears | 6,160 | 24,778,929 | 1968 | 0.9516441006 | -1 |
| circuit `stability@64` | block 11 head 1 Q/K rows | 24,640 | 24,760,449 | 1968 | 0.9516441006 | -1 |
| global magnitude | all transformer Linears | 24,640 | 24,760,449 | 1968 | 0.9516441006 | -1 |
| global Wanda | all transformer Linears | 24,640 | 24,760,449 | 1969 | 0.9521276596 | 0 |

Readout:

- At the very small budget, the granular circuit mask wins: `+5` decisions over dense, while magnitude is neutral and Wanda is `-1`.
- At the 24,640-zero budget, circuit pruning ties global magnitude and is one decision behind global Wanda.
- The circuit mask has a structural advantage that this table does not capture: it removes entire Q/K rows that can be made into qkv row surgery. Global Wanda/magnitude are unstructured zeros.

## Larger Existing Baselines

| Method | Masked values | Active params left | Correct | Accuracy | Delta vs dense | Note |
|---|---:|---:|---:|---:|---:|---|
| circuit `stability@192` | 73,920 | 24,711,169 | 1826 | 0.8829787234 | -143 | full block 11 head 1 Q/K/V slice is unsafe |
| per-matrix magnitude `pf=0.003875` | 82,260 | 24,702,829 | 1969 | 0.9521276596 | 0 | older unstructured baseline |
| per-matrix Wanda `pf=0.003875` | 82,260 | 24,702,829 | 1968 | 0.9516441006 | -1 | older unstructured baseline |
| structured circuit `stability@25` | 82,304 | 24,702,785 | 1967 | 0.9511605416 | -2 | older coarser circuit mask |
| per-matrix magnitude `pf=0.025477` | 540,936 | 24,244,153 | 1967 | 0.9511605416 | -2 | older unstructured baseline |
| per-matrix Wanda `pf=0.025477` | 540,936 | 24,244,153 | 1968 | 0.9516441006 | -1 | older unstructured baseline |
| per-matrix Wanda `pf=0.061239` | 1,300,320 | 23,484,769 | 1971 | 0.9530947776 | +2 | older unstructured baseline |

The old large-budget result still favors unstructured pruning on raw accuracy. The useful thing from the new granular circuit run is not raw zero count; it is that small row-structured masks can be nearly free or even slightly beneficial.

## Quantization Baselines

Quantization keeps the parameter count the same but stores transformer Linear weights at lower precision. Existing fake-quant full DA-2K runs:

| Method | Quantized transformer weights | Correct | Accuracy | Delta vs dense |
|---|---:|---:|---:|---:|
| FP16 | 0 | 1968 | 0.9516441006 | -1 |
| RTN W8 | 21,233,664 | 1971 | 0.9530947776 | +2 |
| GPTQ W8 | 21,233,664 | 1972 | 0.9535783366 | +3 |
| RTN W4 | 21,233,664 | 1802 | 0.8713733075 | -167 |
| GPTQ W4 | 21,233,664 | 1886 | 0.9119922631 | -83 |

Readout:

- W8 quantization is basically free and slightly above dense on this metric; GPTQ W8 is the best quant baseline.
- W4 is still bad for this model/eval, even with GPTQ.
- Quantization and circuit row pruning are complementary: W8 reduces storage/bandwidth for all transformer Linears, while the circuit mask points to row groups we can structurally remove.

## Files

- Granular circuit sweep: `eval_outputs/weight_masked_granular_block11_head1_qkv_full/summary.json`
- Matched magnitude 6,160: `eval_outputs/baseline_magnitude_global_z6160_full/summary.json`
- Matched Wanda 6,160: `eval_outputs/baseline_wanda_global_z6160_full/summary.json`
- Matched magnitude 24,640: `eval_outputs/baseline_magnitude_global_z24640_full/summary.json`
- Matched Wanda 24,640: `eval_outputs/baseline_wanda_global_z24640_full/summary.json`
- Existing broad comparison: `eval_outputs/comparison_baselines/compression_comparison.md`
