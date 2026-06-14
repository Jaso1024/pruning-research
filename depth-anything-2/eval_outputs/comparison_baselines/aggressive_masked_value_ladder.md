# Aggressive Masked-Value Ladder

Full DA-2K, 2068 point pairs. Dense FP32 reference: `1969 / 2068 = 0.9521276596`.

Parameter accounting:

- Dense VITS checkpoint tensor params: `24,785,089`
- Transformer Linear weights targeted by global magnitude/Wanda: `21,233,664`
- Structured circuit masks target removable channel/row groups from `subcircuit_fine_stratified32_g32_h16`.
- All runs are still shape-preserving masks. "Active params left" subtracts masked tensor values from the dense checkpoint count; actual file size is unchanged until compaction.

## Structured Circuit Masks

Value-budget mode ranks circuit nodes by the selected metric and keeps adding nodes until the requested masked-value target is reached.

| Metric | Target values | Actual masked values | Active params left | Correct | Accuracy | Drop vs dense |
|---|---:|---:|---:|---:|---:|---:|
| stability_param | 500,000 | 534,768 | 24,250,321 | 1963 | 0.9492263056 | 6 |
| stability_param | 1,000,000 | 1,020,352 | 23,764,737 | 1941 | 0.9385880077 | 28 |
| stability_param | 1,500,000 | 1,518,336 | 23,266,753 | 1945 | 0.9405222437 | 24 |
| stability_param | 2,000,000 | 2,016,304 | 22,768,785 | 1931 | 0.9337524178 | 38 |
| stability_param | 3,000,000 | 3,006,528 | 21,778,561 | 1867 | 0.9028046422 | 102 |
| stability_param | 5,000,000 | 5,014,736 | 19,770,353 | 1391 | 0.6726305609 | 578 |
| stability | 500,000 | 506,496 | 24,278,593 | 1913 | 0.9250483559 | 56 |
| stability | 1,000,000 | 1,050,704 | 23,734,385 | 1870 | 0.9042553191 | 99 |
| stability | 1,500,000 | 1,502,848 | 23,282,241 | 1838 | 0.8887814313 | 131 |
| stability | 2,000,000 | 2,010,320 | 22,774,769 | 1807 | 0.8737911025 | 162 |
| stability | 3,000,000 | 3,052,144 | 21,732,945 | 1633 | 0.7896518375 | 336 |
| stability | 5,000,000 | 5,005,408 | 19,779,681 | 1297 | 0.6271760155 | 672 |

## Matched Global Unstructured Baselines

These runs prune exactly the same number of transformer Linear weights globally. They are not structurally compactable without sparse kernels, but they test the raw accuracy/zero-count frontier.

| Method | Masked values | Active params left | Correct | Accuracy | Drop vs dense |
|---|---:|---:|---:|---:|---:|
| global magnitude | 534,768 | 24,250,321 | 1968 | 0.9516441006 | 1 |
| global Wanda | 534,768 | 24,250,321 | 1971 | 0.9530947776 | -2 |
| global magnitude | 1,020,352 | 23,764,737 | 1972 | 0.9535783366 | -3 |
| global Wanda | 1,020,352 | 23,764,737 | 1967 | 0.9511605416 | 2 |
| global magnitude | 1,518,336 | 23,266,753 | 1967 | 0.9511605416 | 2 |
| global Wanda | 1,518,336 | 23,266,753 | 1965 | 0.9501934236 | 4 |
| global magnitude | 2,016,304 | 22,768,785 | 1968 | 0.9516441006 | 1 |
| global Wanda | 2,016,304 | 22,768,785 | 1963 | 0.9492263056 | 6 |
| global magnitude | 3,006,528 | 21,778,561 | 1966 | 0.9506769826 | 3 |
| global Wanda | 3,006,528 | 21,778,561 | 1942 | 0.9390715667 | 27 |
| global magnitude | 5,014,736 | 19,770,353 | 1927 | 0.9318181818 | 42 |
| global Wanda | 5,014,736 | 19,770,353 | 1773 | 0.8573500967 | 196 |

## Readout

- For raw high masked-value count, global magnitude is the best baseline. It stays near dense through `3.0M` zeros and still has `0.9318` at `5.0M` zeros.
- Wanda is not uniformly better here. It wins at `534k`, but magnitude beats it from `1.0M` onward, with a large gap at `3.0M` and `5.0M`.
- Current structured circuit pruning does not scale to high masked-value counts. The best aggressive structured point is `stability_param` at `534,768` masked values with `0.9492`; after `1M`, the drop is already substantial.
- The structural advantage is still real: circuit masks remove whole rows/channels and can become actual qkv/MLP/head surgery. But as a pure accuracy-vs-zero-count method, the current circuit ranking is behind global magnitude.

## Files

- Structured ladder: `eval_outputs/weight_masked_value_budget_ladder_full/summary.json`
- Matched global baselines:
  - `eval_outputs/baseline_magnitude_global_z534768_full/summary.json`
  - `eval_outputs/baseline_wanda_global_z534768_full/summary.json`
  - `eval_outputs/baseline_magnitude_global_z1020352_full/summary.json`
  - `eval_outputs/baseline_wanda_global_z1020352_full/summary.json`
  - `eval_outputs/baseline_magnitude_global_z1518336_full/summary.json`
  - `eval_outputs/baseline_wanda_global_z1518336_full/summary.json`
  - `eval_outputs/baseline_magnitude_global_z2016304_full/summary.json`
  - `eval_outputs/baseline_wanda_global_z2016304_full/summary.json`
  - `eval_outputs/baseline_magnitude_global_z3006528_full/summary.json`
  - `eval_outputs/baseline_wanda_global_z3006528_full/summary.json`
  - `eval_outputs/baseline_magnitude_global_z5014736_full/summary.json`
  - `eval_outputs/baseline_wanda_global_z5014736_full/summary.json`
