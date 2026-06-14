# DAV2 Compression Baseline Comparison

Full DA-2K, 2068 point pairs. Dense FP32 reference from the mask runs: `1969/2068 = 0.9521`.

## Pruning / Masking

| family | method | correct | accuracy | newly zeroed | nonzero params left | note |
|---|---|---:|---:|---:|---:|---|
| unstructured linear pruning | `magnitude pf=0.003875` | 1969 | 0.9521 | 82,260 | 24,702,445 | 48 transformer Linear weights only; no shape compaction |
| unstructured linear pruning | `wanda pf=0.003875` | 1968 | 0.9516 | 82,260 | 24,702,445 | 48 transformer Linear weights only; no shape compaction |
| structured circuit mask | `stability@25` | 1967 | 0.9512 | 82,304 | 24,702,401 | persistent structured zeros; shape not compacted yet |
| structured circuit mask | `stability@50` | 1957 | 0.9463 | 116,880 | 24,667,825 | persistent structured zeros; shape not compacted yet |
| structured circuit mask | `stability@100` | 1917 | 0.9270 | 364,144 | 24,420,561 | persistent structured zeros; shape not compacted yet |
| unstructured linear pruning | `magnitude pf=0.025477` | 1967 | 0.9512 | 540,936 | 24,243,769 | 48 transformer Linear weights only; no shape compaction |
| unstructured linear pruning | `wanda pf=0.025477` | 1968 | 0.9516 | 540,936 | 24,243,769 | 48 transformer Linear weights only; no shape compaction |
| structured circuit mask | `stability_param@25` | 1960 | 0.9478 | 540,960 | 24,243,745 | persistent structured zeros; shape not compacted yet |
| structured circuit mask | `stability_param@50` | 1937 | 0.9367 | 1,300,288 | 23,484,417 | persistent structured zeros; shape not compacted yet |
| unstructured linear pruning | `magnitude pf=0.061239` | 1968 | 0.9516 | 1,300,320 | 23,484,385 | 48 transformer Linear weights only; no shape compaction |
| unstructured linear pruning | `wanda pf=0.061239` | 1971 | 0.9531 | 1,300,320 | 23,484,385 | 48 transformer Linear weights only; no shape compaction |
| structured circuit mask | `stability_param@100` | 1836 | 0.8878 | 2,552,000 | 22,232,705 | persistent structured zeros; shape not compacted yet |

## Quantization

| method | correct | accuracy | quantized weights | rough fp16-equivalent MiB | note |
|---|---:|---:|---:|---:|---|
| `fp16` | 1968 | 0.9516 | 0 | 47.27 | fake-quant eval; transformer Linear weights quantized, other params fp16 |
| `RTN W8` | 1971 | 0.9531 | 21,233,664 | 27.02 | fake-quant eval; transformer Linear weights quantized, other params fp16 |
| `RTN W4` | 1802 | 0.8714 | 21,233,664 | 16.90 | fake-quant eval; transformer Linear weights quantized, other params fp16 |
| `GPTQ W8` | 1972 | 0.9536 | 21,233,664 | 27.02 | fake-quant eval; transformer Linear weights quantized, other params fp16 |
| `GPTQ W4` | 1886 | 0.9120 | 21,233,664 | 16.90 | fake-quant eval; transformer Linear weights quantized, other params fp16 |

## Readout

- At the same raw zero count, unstructured magnitude/Wanda preserve DA-2K better than structured circuit masks.
- The structured circuit masks are still the only entries here that correspond to removable channel groups after shape surgery.
- W8 quantization is basically free on DA-2K; W4 needs GPTQ and still drops hard (`0.9120`).
