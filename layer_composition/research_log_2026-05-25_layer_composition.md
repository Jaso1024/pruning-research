# Layer Composition Distillation Research Log

## 2026-05-25 Plan

Goal: test whether Pythia-70M transformer block `middle` + `middle+1` can be distilled into one block initialized from the middle block.

Experiment constraints:
- Use `EleutherAI/pythia-70m`.
- Select the middle pair automatically from the model layer count.
- Teacher: frozen original layer `i` followed by frozen original layer `i+1`.
- Student: one layer initialized as a deep copy of original layer `i`.
- Optimize only student parameters.
- Use Muon for matrix-shaped parameters, AdamW for scalar/vector parameters if needed.
- Log every step as JSONL, including train loss, relative MSE, cosine, tokens/sec, GPU memory, and `nvidia-smi` utilization when available.
- Sweep learning rates without changing any other training knobs.
- Run on Modal H100 after local tests and smoke checks.

Test plan before coding:
- Layer-pair selection returns the expected middle and next layer for even and odd depths.
- Distillation config validates invalid layer pairs and learning-rate sweeps.
- Student initialization deep-copies the first teacher layer and does not alias its parameters.
- Muon performs finite updates on matrix parameters and preserves parameter shapes.
- JSONL step logging writes one structured record per step and flushes to disk.
- LR sweep expansion creates independent configs and result directories.
- Summary logic identifies the best run by final relative MSE.

## 2026-05-25 H100 Sweep

Implementation notes:
- Initial local tests passed with `uv run --with pytest pytest -q`.
- Local one-step Pythia-70M smoke passed with synthetic tokens.
- First H100 smoke exposed a Wikitext loader issue with the non-namespaced dataset ID; fixed by using `Salesforce/wikitext`.
- Larger H100 smoke exposed wasteful logits allocation from `AutoModelForCausalLM`; fixed by switching to `AutoModel`, since only hidden states are needed.
- Background `nvidia-smi --loop-ms=100` sampling was added because one-shot utilization checks after synchronization sampled idle GPU time.

Final run:
- Modal app: `pythia-layer-composition-distill`
- Modal workspace/profile: `jthomams477`
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-wxAJ704AWEZpKtfjwseRMb`
- Volume path: `/results/pythia70m_middle_pair_h100_sweep_20260525`
- Local copy: `runs/modal_h100_sweep_20260525`
- Model: `EleutherAI/pythia-70m`
- Layers: `2 -> 3` distilled into one copied layer `2`
- Data: `Salesforce/wikitext`, `wikitext-2-raw-v1`
- Batch/sequence: `1024 x 1024`
- Steps per LR: `100`
- LRs: `0.0003`, `0.001`, `0.003`, `0.01`
- Per-step logs: 400 total JSONL records
- Best student checkpoint downloaded: `runs/modal_h100_sweep_20260525/lr_0.01/student_layer.pt`

Results:

| lr | final relative MSE | final cosine | post-warmup tokens/sec | post-warmup avg GPU util | max GPU util |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.0003 | 0.609249 | 0.662890 | 2,838,095 | 99.2% | 100% |
| 0.001 | 0.443897 | 0.749750 | 2,829,730 | 99.1% | 100% |
| 0.003 | 0.249185 | 0.866519 | 2,826,781 | 99.2% | 100% |
| 0.01 | 0.181691 | 0.904641 | 2,831,202 | 99.2% | 100% |

Interpretation:
- The tested range was still improving at `lr=0.01`; no instability showed up over 100 steps.
- The single-layer student reaches cosine `0.9046` to the original two-layer composition after 100 Muon steps at `lr=0.01`.
- A next sweep should try `0.01`, `0.02`, `0.03`, and maybe longer runs, while watching for divergence.

## 2026-05-26 SLERP / Geometric Merge Sweep

Goal: test whether a no-training parameter merge of layer `2` and layer `3` can approximate the frozen two-block teacher `layer2(layer input) -> layer3`, using the same evaluation shape as the distillation run.

Plan:
- Add a merge evaluator that deep-copies layer `2`, merges matching parameters from layer `2` and layer `3`, and evaluates the single merged layer against the original two-layer teacher.
- Implement Family B methods: `slerp` and geometric-norm `geom_slerp`; include `linear` as a baseline.
- Use identical token windows for every merge setting by resetting RNG seeds per config.
- Log every step as JSONL with loss, relative MSE, cosine, tokens/sec, GPU memory, and sampled GPU utilization.

Test plan before coding:
- SLERP endpoints return exact source tensors.
- Orthogonal midpoint has the expected direction and norm behavior.
- Geometric SLERP interpolates tensor norms geometrically.
- Near-parallel tensors fall back to linear interpolation.
- Module merging deep-copies parameters and does not alias originals.
- Merge configs validate methods, `t` values, and sweep expansion.
- Summary selection picks the lowest final relative MSE.

Final run:
- Modal app: `pythia-layer-composition-distill`
- Modal workspace/profile: `jthomams477`
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-IyhF49DW0OLPzpUrONn4nB`
- Volume path: `/results/pythia70m_middle_pair_merge_h100_sweep_20260526`
- Local copy: `runs/modal_h100_merge_sweep_20260526`
- Model: `EleutherAI/pythia-70m`
- Layers: merge original layer `2` and original layer `3`, evaluate as one layer against teacher `2 -> 3`
- Data: `Salesforce/wikitext`, `wikitext-2-raw-v1`
- Batch/sequence: `1024 x 1024`
- Steps per merge point: `100`
- Methods: `linear`, `slerp`, `geom_slerp`
- `t` values: `0.0` through `1.0` in increments of `0.1`
- Per-step logs: 3300 total JSONL records
- Post-warmup average throughput across settings: `3,500,340` tokens/sec
- Post-warmup average sampled GPU utilization across settings: `99.4%`

Best per method:

| method | best t | final relative MSE | final cosine |
| --- | ---: | ---: | ---: |
| linear | 0.2 | 0.582679 | 0.654777 |
| slerp | 0.0 | 0.652916 | 0.643580 |
| geom_slerp | 0.0 | 0.652916 | 0.643580 |

Family B grid:

| method | t | final relative MSE | final cosine |
| --- | ---: | ---: | ---: |
| slerp | 0.0 | 0.652916 | 0.643580 |
| slerp | 0.1 | 0.689462 | 0.636887 |
| slerp | 0.2 | 0.744931 | 0.617251 |
| slerp | 0.3 | 0.802607 | 0.590958 |
| slerp | 0.4 | 0.847644 | 0.565795 |
| slerp | 0.5 | 0.879128 | 0.543906 |
| slerp | 0.6 | 0.895198 | 0.526905 |
| slerp | 0.7 | 0.890341 | 0.518341 |
| slerp | 0.8 | 0.869534 | 0.518969 |
| slerp | 0.9 | 0.831575 | 0.532454 |
| slerp | 1.0 | 0.838989 | 0.536456 |
| geom_slerp | 0.0 | 0.652916 | 0.643580 |
| geom_slerp | 0.1 | 0.786520 | 0.614733 |
| geom_slerp | 0.2 | 0.978488 | 0.578050 |
| geom_slerp | 0.3 | 1.199816 | 0.539543 |
| geom_slerp | 0.4 | 1.398325 | 0.504159 |
| geom_slerp | 0.5 | 1.520710 | 0.473578 |
| geom_slerp | 0.6 | 1.527245 | 0.450621 |
| geom_slerp | 0.7 | 1.397001 | 0.441367 |
| geom_slerp | 0.8 | 1.175699 | 0.450787 |
| geom_slerp | 0.9 | 0.953597 | 0.484052 |
| geom_slerp | 1.0 | 0.838989 | 0.536456 |

Interpretation:
- In this two-block setting, Family B did not improve over the `t=0` endpoint. Both SLERP variants got worse as they moved away from layer `2`, with geometric-norm SLERP especially poor near the midpoint.
- The simple linear baseline did improve over `t=0`, with best relative MSE `0.582679` at `t=0.2`, but it is still far behind the trained one-layer Muon distillation result from 2026-05-25 (`0.181691` relative MSE).
- The H100 was fully exercised for the evaluation sweep: sampled GPU utilization averaged `99.4%` after warmup and reached `100%` for every condition.

## 2026-05-26 Pythia Attention Distribution Probe

Goal: quickly compare post-softmax, pre-value attention distributions across heads and across Pythia sizes.

Plan:
- Use `EleutherAI/pythia-31m` as the next-smallest model below `EleutherAI/pythia-70m`.
- Run a couple short prompt/query environments through both models with `output_attentions=True` and eager attention.
- For each layer/head, log entropy, normalized entropy, max attention probability, diagonal mass, and local-window mass.
- For head-pair comparisons within the same model/layer, log Jensen-Shannon divergence, total variation, and cosine distance.
- For same layer/head comparisons between the small and big model, log the same distance metrics.

Test plan before coding:
- Identical attention distributions have zero Jensen-Shannon, total variation, and cosine distance.
- Disjoint distributions have large distance.
- Head summaries report entropy, diagonal mass, and local-window mass correctly.
- Within-model comparison emits one row per head pair.
- Cross-model comparison aligns same layer and same head.
- Config validation rejects empty prompt sets, invalid sequence lengths, and invalid dtypes.
- Run summary picks the closest cross-model head by Jensen-Shannon divergence.

Smoke run:
- Command output directory: `runs/attention_pythia31m_70m_smoke_20260526`
- Device/dtype: CPU, `fp32`
- Max sequence length: `48`
- Prompts: gridworld navigation prompt; simple key-counting question
- Cross-model rows: `96`
- Within-model rows: `672`
- Head-summary rows: `192`
- Closest same-layer/head cross-model match: prompt `1`, layer `1`, head `3`, JSD `0.033971`, total variation `0.197815`, cosine distance `0.073277`
- Worst same-layer/head cross-model match: prompt `1`, layer `4`, head `3`, JSD `0.623513`, total variation `0.932630`, cosine distance `0.908176`
- Closest within-model head pair: `pythia-31m`, prompt `0`, layer `4`, heads `1` and `6`, JSD `0.007008`
- Furthest within-model head pair: `pythia-70m`, prompt `1`, layer `5`, heads `6` and `7`, JSD `0.627435`

Mean cross-model JSD by layer:

| layer | mean JSD |
| ---: | ---: |
| 0 | 0.2021 |
| 1 | 0.2384 |
| 2 | 0.2045 |
| 3 | 0.3972 |
| 4 | 0.3916 |
| 5 | 0.4451 |

Mean within-model head-pair JSD by layer:

| model | layer | mean JSD |
| --- | ---: | ---: |
| pythia-31m | 0 | 0.2161 |
| pythia-31m | 1 | 0.2417 |
| pythia-31m | 2 | 0.2012 |
| pythia-31m | 3 | 0.2845 |
| pythia-31m | 4 | 0.2894 |
| pythia-31m | 5 | 0.3378 |
| pythia-70m | 0 | 0.2380 |
| pythia-70m | 1 | 0.1619 |
| pythia-70m | 2 | 0.2505 |
| pythia-70m | 3 | 0.2304 |
| pythia-70m | 4 | 0.2929 |
| pythia-70m | 5 | 0.4682 |

Interpretation:
- On this tiny prompt set, cross-model same-head attention similarity is much better in layers `0` and `2` than in layers `3` through `5`.
- The mean normalized entropy was higher for `pythia-31m` (`0.5016`) than `pythia-70m` (`0.4173`), so the smaller model's heads were more diffuse on these prompts.
- This is only a fast probe. A more serious pass should use a larger prompt set, aggregate confidence intervals, and add best-head matching across models instead of only same-index head matching.

## 2026-05-26 Pythia Attention Min-Diff Head Matching

Correction: the first attention probe compared same-index heads only. That is too strict because heads can be permuted across model sizes.

Added analytics:
- Full cross-model `8 x 8` head distance matrix per prompt and layer.
- `small_to_best_big`: best 70M head for each 31M head.
- `big_to_best_small`: best 31M head for each 70M head.
- `one_to_one`: minimum-cost one-to-one assignment across all heads in a layer.

Run:
- Command output directory: `runs/attention_pythia31m_70m_matching_20260526`
- Cross same-index rows: `96`
- Cross-head matching rows: `1056`
- Tests: `26 passed`

Mean cross-model JSD by layer:

| layer | same-index | small-to-best-big | one-to-one |
| ---: | ---: | ---: | ---: |
| 0 | 0.2021 | 0.0977 | 0.1212 |
| 1 | 0.2384 | 0.1516 | 0.1714 |
| 2 | 0.2045 | 0.1150 | 0.1213 |
| 3 | 0.3972 | 0.1664 | 0.3261 |
| 4 | 0.3916 | 0.2736 | 0.3176 |
| 5 | 0.4451 | 0.3494 | 0.4032 |

Interpretation:
- Head permutation was a real confound. Best-head matching makes layers `0`, `1`, and `2` substantially closer, and it cuts layer `3` from same-index JSD `0.3972` to best-match JSD `0.1664`.
- The one-to-one matching is the more conservative estimate. It still improves the same-index baseline, but late layers remain high: layer `3` one-to-one JSD `0.3261`, layer `4` `0.3176`, layer `5` `0.4032`.
- The gap between best-match and one-to-one is informative. In layer `3`, many 31M heads map to the same few 70M heads under unconstrained best-match, so unconstrained matching overstates similarity. This looks like many-to-one collapse rather than clean head permutation.
- Layers `4` and `5` remain divergent even after best matching. The best available 70M analogue for some 31M heads is still far away; worst best-match row is prompt `1`, layer `5`, 31M head `4` to 70M head `6`, JSD `0.5045`.
- Corrected takeaway: early attention motifs transfer across model size after head permutation; late attention does not reduce to a simple permutation.

## 2026-05-26 Pythia 31M to 70M Attention Linear-Combination Matching

Goal: test whether each 70M attention head can be better approximated as a convex linear combination of all 31M heads in the same layer, rather than by a single matched 31M head.

Implementation:
- Added `fit_head_linear_combinations` and `compare_cross_model_linear_combinations` in `layer_distill/attention_analysis.py`.
- For each prompt/layer, optimize one weight vector per 70M head over the 8 same-layer 31M heads.
- Weights are softmax-constrained, so each reconstructed head is a valid convex mixture of attention distributions.
- Objective is mean Jensen-Shannon divergence between the reconstructed distribution and the target 70M head.
- Output directory: `runs/attention_pythia31m_70m_linear_combo_20260526`
- Comparison artifact: `runs/attention_pythia31m_70m_linear_combo_20260526/linear_combo_comparison.json`
- Same prompts/config as `runs/attention_pythia31m_70m_matching_20260526`: CPU, fp32, max length `48`, two prompts.
- Tests after implementation: `49 passed in 1.63s`.

Results:

| layer | same-index JSD | small->best big JSD | big->best small JSD | injective JSD | linear-combo JSD | gain vs big->best small | effective heads | top weight |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.2021 | 0.0977 | 0.1123 | 0.1212 | 0.0887 | 21.1% | 2.83 | 0.57 |
| 1 | 0.2384 | 0.1516 | 0.1184 | 0.1714 | 0.0854 | 27.9% | 3.41 | 0.53 |
| 2 | 0.2045 | 0.1150 | 0.1130 | 0.1213 | 0.0980 | 13.3% | 2.49 | 0.67 |
| 3 | 0.3972 | 0.1664 | 0.2016 | 0.3261 | 0.1881 | 6.7% | 2.18 | 0.79 |
| 4 | 0.3916 | 0.2736 | 0.2499 | 0.3176 | 0.2430 | 2.8% | 1.88 | 0.76 |
| 5 | 0.4451 | 0.3494 | 0.3106 | 0.4032 | 0.3026 | 2.6% | 1.50 | 0.84 |

Overall means:
- Same-index JSD: `0.3132`
- Small-to-best-big JSD: `0.1923`
- Big-to-best-small JSD: `0.1843`
- Injective JSD: `0.2435`
- Linear-combo JSD: `0.1676`
- Linear-combo gain vs big-to-best-small: `9.0%`
- Mean effective heads in mixture: `2.38`
- Mean top weight: `0.693`

Interpretation:
- Linear combinations help, but they do not radically change the conclusion.
- The gains are meaningful in early layers: layers `0` and `1` improve by `21%` and `28%` versus the best single 31M head for each 70M head.
- Later layers barely improve: layers `4` and `5` only gain `2.8%` and `2.6%`.
- The mixtures become more one-hot with depth. Effective heads fall from `3.41` in layer `1` to `1.50` in layer `5`, and top weight rises to `0.84`.
- This suggests early 70M heads often are compositional mixtures of 31M attention motifs, but late 70M heads are not well explained by linear recombination of 31M heads. The late-layer mismatch is not just a missing basis rotation over heads.

## 2026-05-26 Pythia Attention Mass Divergence Examples

Goal: inspect concrete token-level examples where the 31M and 70M attention mass diverges after best-head matching.

Run:
- Output directory: `runs/attention_pythia31m_70m_divergence_examples_20260526`
- Files: `examples.json`, `query_category_rows.jsonl`
- Models/prompts: same two-prompt probe as the min-diff matching run.
- Examples selected from the worst best-match pairs plus one close control pair.

Concrete divergences:
- Prompt `Question: If Alice has three keys and gives Bob one key, how many keys does Alice still have? Answer:`
- Worst best-match pair: layer `5`, 31M head `4` to 70M head `6`, mean JSD `0.5045`.
  - Query token ` Alice`: 31M puts `0.978` on `:`, while 70M puts `1.000` on the earlier ` keys`.
  - Query token ` still`: 31M splits between `,` (`0.530`) and `:` (`0.458`), while 70M puts `1.000` on earlier ` keys`.
  - Query token ` keys`: 31M puts `0.682` on `:` and `0.260` on `,`, while 70M puts `1.000` on ` Bob`.
- Another late pair: layer `5`, 31M head `3` to 70M head `4`, mean JSD `0.4783`.
  - Query token `:`: 31M attends to repeated entity tokens ` Alice` (`0.892`) and earlier ` Alice` (`0.104`), while 70M attends to `?` (`0.849`) and current `:` (`0.149`).
  - Query token ` does`: 31M attends to earlier ` Alice` (`0.965`), while 70M attends to `,` (`1.000`).
  - Query token ` gives`: 31M attends to ` Alice` (`1.000`), while 70M attends to `Question` (`0.978`) and `and` (`0.022`).
- Layer `4`, 31M head `4` to 70M head `0`, mean JSD `0.4588`.
  - Query token ` has`: 31M is mostly self-attention (`has`, `0.938`), while 70M points to `Question` (`0.989`).
  - Query token ` gives`: 31M is mostly self-attention (`gives`, `0.951`), while 70M points to `keys` (`0.692`), `Alice` (`0.130`), and `three` (`0.100`).
  - Query token `:`: 31M attends to answer punctuation/current suffix (`:`, `?`, ` Answer`), while 70M attends to entity tokens (`Alice`, `Bob`) and `Question`.
- Gridworld prompt layer `3`, 31M head `5` to 70M head `4`, mean JSD `0.3909`.
  - Query token `world`: 31M puts `1.000` on itself, while 70M spreads to ` a` (`0.471`) and ` small` (`0.409`).
  - Query token ` square`: 31M attends to the two ` square` positions (`0.560`, `0.437`), while 70M attends to the phrase before it: ` the` (`0.540`), ` blue` (`0.239`), ` reach` (`0.113`).
  - Query token ` grid`: 31M puts `0.999` on itself, while 70M attends to ` small` (`0.504`) and ` a` (`0.371`).

Category-level mass shifts:

| pair | mean JSD | small first-token mass | big first-token mass | small self mass | big self mass | small local8 | big local8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| closest control L1 H3/H3 | 0.0340 | 0.092 | 0.096 | 0.265 | 0.218 | 0.924 | 0.956 |
| layer3 many-to-one | 0.3909 | 0.064 | 0.195 | 0.945 | 0.253 | 0.998 | 0.911 |
| layer4 bad match | 0.4588 | 0.074 | 0.308 | 0.474 | 0.064 | 0.979 | 0.736 |
| late layer bad match L5 | 0.4783 | 0.148 | 0.467 | 0.094 | 0.139 | 0.676 | 0.727 |
| worst best match L5 | 0.5045 | 0.205 | 0.534 | 0.085 | 0.043 | 0.531 | 0.633 |

Interpretation:
- The close control pair differs mostly in fine allocation inside the same local region; both models keep most mass in the local window.
- The bad layer `3` gridworld pair is almost a pure self-copy/local-copy head in 31M, while the closest 70M analogue points to phrase context before the current token. This is not just head permutation.
- In the late arithmetic prompt examples, 70M frequently uses sharp anchors: first token, punctuation, repeated entities, `keys`, or `Bob`. The 31M heads often point to punctuation, repeated surface tokens, or self/local positions instead.
- The late-layer divergence is therefore partly role splitting: 70M has sharper anchor heads, while the closest 31M heads often look like cruder local/self/repeated-token heads.

## 2026-05-26 Pythia 70M to 160M Attention Matching

Goal: repeat the attention-distribution analysis between `EleutherAI/pythia-70m` and the next larger Pythia checkpoint, `EleutherAI/pythia-160m`.

Implementation note:
- `pythia-70m` has 6 layers and 8 heads/layer.
- `pythia-160m` has 12 layers and 12 heads/layer.
- Matching compares the first 6 same-index layers.
- Added rectangular one-to-one matching: exact minimum-cost injective assignment from 8 small-model heads into 12 large-model heads.

Run:
- Matching output directory: `runs/attention_pythia70m_160m_matching_20260526`
- Divergence-example output directory: `runs/attention_pythia70m_160m_divergence_examples_20260526`
- Cross same-index rows: `96`
- Cross-head matching rows: `1488`
- Head-summary rows: `384`
- Within-model rows: `1920`
- Tests after rectangular matching: `27 passed`

Mean cross-model JSD by layer:

| layer | same-index | 70M-to-best-160M | injective 8-to-12 |
| ---: | ---: | ---: | ---: |
| 0 | 0.2098 | 0.1009 | 0.1039 |
| 1 | 0.1992 | 0.1074 | 0.1185 |
| 2 | 0.2012 | 0.1265 | 0.1428 |
| 3 | 0.3363 | 0.1837 | 0.2014 |
| 4 | 0.3190 | 0.1960 | 0.2096 |
| 5 | 0.4033 | 0.3173 | 0.3329 |

Comparison to 31M to 70M:
- 70M to 160M is substantially closer than 31M to 70M after matching, especially layers `3` and `4`.
- For 31M to 70M, the gap between unconstrained best-match and one-to-one was large in layer `3`, suggesting many-to-one collapse.
- For 70M to 160M, best-match and injective matching are close in layers `0` through `4`. That means the 70M heads mostly have distinct 160M analogues there, not just many 70M heads all mapping to one 160M head.
- Layer `5` remains the outlier: best-match JSD `0.3173`, injective JSD `0.3329`.

Head-summary trend:
- `pythia-160m` starts diffuse, then becomes very sharp in layers `8` through `11`.
- Mean normalized entropy for 160M by layer drops from about `0.68` in early layers to `0.05` by layer `11`.
- The comparable 70M final layer is already sharp: layer `5` normalized entropy `0.0970`.
- The first six 160M layers are not simply a sharper copy of 70M; 160M's extreme sharpening happens mostly after the layer range that 70M has.

Concrete divergence examples:
- Worst best-match pair: arithmetic prompt, layer `5`, 70M head `1` to 160M head `1`, mean JSD `0.4947`.
  - Query token ` key`: 70M puts `1.000` on ` If`, while 160M puts mass on local phrase tokens ` one` (`0.594`) and ` gives` (`0.152`) plus `Question` (`0.102`).
  - Query token ` one`: 70M puts `1.000` on ` If`, while 160M spreads across ` gives`, `Question`, `one`, and `Bob`.
  - Query token `:`: 70M puts `0.998` on `?`, while 160M puts `0.823` on `Question`.
- Second worst: arithmetic prompt, layer `5`, 70M head `2` to 160M head `1`, mean JSD `0.4707`.
  - Query token `?`: 70M self-attends to `?` (`1.000`), while 160M points to `Question` (`0.933`).
  - Query token ` If`: 70M self-attends to `If` (`1.000`), while 160M points to `Question` (`0.811`) and `:` (`0.163`).
  - Query token ` how`: 70M self-attends to `how` (`1.000`), while 160M points to `Question` (`0.913`).
- A close control: gridworld prompt, layer `3`, 70M head `2` to 160M head `6`, mean JSD `0.0377`.
  - The distributions are not identical, but both heads broadly mix first-token, previous-token, self, and local-context mass rather than disagreeing categorically.

Category-level mass shifts:

| pair | mean JSD | 70M first-token | 160M first-token | 70M self | 160M self | 70M local8 | 160M local8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| close control L3 | 0.0377 | 0.387 | 0.385 | 0.157 | 0.168 | 0.773 | 0.751 |
| good layer4 match | 0.0422 | 0.912 | 0.957 | 0.076 | 0.091 | 0.488 | 0.442 |
| layer4 bad match | 0.3988 | 0.335 | 0.498 | 0.087 | 0.139 | 0.867 | 0.609 |
| second worst L5 | 0.4707 | 0.044 | 0.557 | 0.548 | 0.157 | 0.997 | 0.661 |
| worst L5 | 0.4947 | 0.087 | 0.557 | 0.073 | 0.157 | 0.844 | 0.661 |

Interpretation:
- 70M to 160M preserves attention motifs better than 31M to 70M. Matching knocks early/mid-layer JSD to roughly `0.10` to `0.20`.
- The late layer still diverges. In layer `5`, some 70M heads are self/local heads or idiosyncratic anchor heads, while the nearest 160M heads often put much more mass on `Question` or other global anchors.
- The biggest structural difference is depth: 160M has six more layers. Its later layers become extremely sharp, but those do not have direct same-layer counterparts in 70M.

## 2026-05-26 Five-Pair Upward Pythia Attention Sweep

Goal: continue the same attention-distribution matching analysis for five adjacent upward Pythia transitions starting at `160m`.

Pairs:
- `EleutherAI/pythia-160m` -> `EleutherAI/pythia-410m`
- `EleutherAI/pythia-410m` -> `EleutherAI/pythia-1b`
- `EleutherAI/pythia-1b` -> `EleutherAI/pythia-1.4b`
- `EleutherAI/pythia-1.4b` -> `EleutherAI/pythia-2.8b`
- `EleutherAI/pythia-2.8b` -> `EleutherAI/pythia-6.9b`

Implementation notes:
- Added Modal H100 `mode=attention` to run multi-pair sweeps through the same harness.
- Added model-pair sweep helper and parser.
- Replaced bitmask assignment with Hungarian assignment so `16`, `32`, and rectangular head-count comparisons are practical.
- Rectangular one-to-one means minimum-cost injective matching over the smaller head set into the larger head set. For `410m -> 1b`, where head count decreases from `16` to `8`, this picks the best distinct 8-head subset from the 410M side.

Run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-dOlYKq4fjXDzDMnCmK1G88`
- Modal volume path: `/results/pythia_attention_5pair_h100_20260526`
- Local copy: `runs/modal_attention_5pair_h100_20260526`
- Derived summary: `runs/modal_attention_5pair_h100_20260526/analysis_summary.json`
- Device/dtype: H100, `bf16`
- Max sequence length: `48`
- Prompts: gridworld navigation; arithmetic/key-counting question
- Tests after implementation: `29 passed`

Scoreboard:

| pair | layers compared | same-index JSD | best-match JSD | injective JSD | worst best-match layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| 160m -> 410m | 12 | 0.2585 | 0.1612 | 0.1854 | 4 / 0.2313 |
| 410m -> 1b | 16 | 0.1714 | 0.0903 | 0.0954 | 4 / 0.2138 |
| 1b -> 1.4b | 16 | 0.1443 | 0.0660 | 0.0711 | 3 / 0.1111 |
| 1.4b -> 2.8b | 24 | 0.1621 | 0.0837 | 0.0892 | 22 / 0.2004 |
| 2.8b -> 6.9b | 32 | 0.1943 | 0.1025 | 0.1325 | 29 / 0.2065 |

Interpretation:
- Head permutation remains a major confound at every size. Best-match roughly halves the apparent same-index divergence in most transitions.
- The cleanest transition is `1b -> 1.4b`: best-match JSD `0.0660`, injective JSD `0.0711`. This looks like very strong preservation of attention motifs after head permutation.
- `410m -> 1b` is also surprisingly clean after matching despite head count dropping from `16` to `8`; injective JSD `0.0954`. This suggests many 410M heads have close analogues among the smaller head count of 1B, or are redundant under these prompts.
- `160m -> 410m` is rougher: best-match JSD `0.1612`, injective JSD `0.1854`. Layers `4`, `5`, and `8` are the worst by best-match JSD.
- `2.8b -> 6.9b` is not as clean as `1b -> 2.8b`, especially in later layers. Best-match JSD is `0.1025`, but injective rises to `0.1325`, with worst layers around `27` to `30` and layer `3`.
- Larger models do not monotonically become easier to align by same-index heads. But after head matching, adjacent scaling transitions mostly preserve a substantial amount of attention geometry.
- The biggest persistent failure mode is late-layer specialization: worst matched layers tend to be mid/late layers where heads are sharp and idiosyncratic.

## 2026-05-26 Low-QK Attention Distillation

Reminder: after the low-QK attention experiments reach a conclusion, return to the attention-distribution matching work and use the new evidence to decide whether head-level attention geometry is compressible by a smaller Q/K subspace.

Goal: test whether the `QK^T` dimension inside a Pythia attention block can be shrunk aggressively by distilling the original attention output into a replacement attention mechanism with `qk_dim=2`.

Implementation:
- Added `layer_distill/low_qk_attention.py`.
- Student attention keeps the same hidden output size and same number of heads by default, but projects Q and K to only `2` channels per head.
- Value dimension stays at the teacher head size (`64` for Pythia-70M), so this is specifically a low-QK-rank attention test rather than a value bottleneck test.
- Student initialization copies the teacher's first `qk_dim` Q/K channels per head, the full V projection, and the output projection from the GPT-NeoX attention block.
- Training target is the original Pythia-70M layer-2 attention output.
- Optimizer: Muon when available, with AdamW fallback.

Validation:
- Unit tests cover low-QK output shape, returned attention weights, causal masking, GPT-NeoX Q/K/V initialization, LR sweep config generation, and run-summary selection.
- Test command: `uv run --with pytest pytest -q`
- Result: `35 passed`

H100 run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-VXdBdRRxmhthIIJEKGjFgx`
- Modal volume path: `/pythia70m_low_qk2_attention_h100_seq256_20260526`
- Local copy: `runs/modal_low_qk2_attention_h100_seq256_20260526`
- Model: `EleutherAI/pythia-70m`
- Layer: `2`
- Data: `Salesforce/wikitext`, `wikitext-103-raw-v1`
- Batch size: `1024`
- Sequence length: `256`
- Tokens per step: `262144`
- Steps per LR: `100`
- Dtype: `bf16`
- Student params: `541728`

LR sweep:

| lr | start rel MSE | final rel MSE | final cosine | avg tok/s after warmup | avg sampled GPU util |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 0.116744 | 0.095564 | 0.951023 | 2.850M | 89.6% |
| 0.003 | 0.116744 | 0.074715 | 0.961919 | 2.848M | 91.0% |
| 0.010 | 0.116744 | 0.055587 | 0.971810 | 2.848M | 90.5% |
| 0.030 | 0.116744 | 0.051235 | 0.974091 | 2.848M | 90.4% |

Best run:
- `lr=0.03`
- Final rel MSE: `0.05123494192957878`
- Final cosine: `0.9740909337997437`
- Final loss: `0.0005174298421479762`
- Final throughput: `2849765` tokens/s

Seq-1024 attempt notes:
- Initial H100 seq-1024 runs OOMed.
- First issue was teacher attention materialization; loading the teacher with `attn_implementation="sdpa"` fixed that side.
- The student still OOMed at seq 1024 because the available SDPA path appears to materialize a large score tensor for asymmetric low-QK queries/keys with full-size values.
- Seq 256 is the current fast, healthy test. To push seq 1024, implement chunked distillation or a custom memory-efficient low-QK attention path rather than relying on the generic kernel choice.

Interpretation:
- The teacher-copy initialization is already meaningfully close: rel MSE starts at about `0.1167`, cosine about `0.9398`.
- Training the low-QK student for only 100 steps cuts rel MSE by more than half, to about `0.0512`, and reaches cosine `0.9741`.
- This is strong evidence that a layer-2 Pythia-70M attention output is at least locally distillable with only two Q/K channels per head when V and output projection capacity are preserved.
- This does not prove the attention distribution itself is preserved. It only says the attention module output can be matched well under this setup. The next useful check is to compare teacher and low-QK student attention maps directly, then test later layers where attention is sharper and more specialized.

## 2026-05-26 Low-QK Attention Distillation Across All Pythia-70M Layers

Follow-up: ran the same `qk_dim=2` low-QK attention distillation setup for all six Pythia-70M attention layers.

Run setup:
- Model: `EleutherAI/pythia-70m`
- Layers: `0,1,2,3,4,5`
- LR: `0.03`
- Steps per layer: `100`
- Batch size: `1024`
- Sequence length: `256`
- Tokens per step: `262144`
- Dtype: `bf16`
- Student heads: teacher head count (`8`)
- Student value dim: teacher head size (`64`)
- Optimizer: Muon when available, AdamW fallback for non-Muon params
- Local copy: `runs/modal_low_qk2_attention_all_layers_h100_20260526`
- Local aggregate: `runs/modal_low_qk2_attention_all_layers_h100_20260526/all_layers_summary.json`

Modal runs:
- Layer 0: `https://modal.com/apps/jthomams477/main/ap-osEZZWLPehdFvtuMDKnTTj`
- Layer 1: `https://modal.com/apps/jthomams477/main/ap-DgY2lOjWTF0iu2zf3Bvugd`
- Layer 2: `https://modal.com/apps/jthomams477/main/ap-HOJK5lquJZ5QZclVzyMiFd`
- Layer 3: `https://modal.com/apps/jthomams477/main/ap-wJLECccQdrFmLYz99DlIvf`
- Layer 4: `https://modal.com/apps/jthomams477/main/ap-fI0nbKDSTikB28v1sTKLVn`
- Layer 5: `https://modal.com/apps/jthomams477/main/ap-cqGTg3DK1z9Y5Q3PUPGN4R`

Results:

| layer | start rel MSE | final rel MSE | final cosine | best step | best rel MSE | best cosine | avg tok/s | avg sampled GPU util |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0000225 | 0.0000427 | 0.999979 | 1 | 0.0000225 | 0.999989 | 2.808M | 91.6% |
| 1 | 0.036879 | 0.016719 | 0.991620 | 100 | 0.016719 | 0.991620 | 2.822M | 91.2% |
| 2 | 0.116744 | 0.051235 | 0.974091 | 92 | 0.050402 | 0.974613 | 2.404M | 97.5% |
| 3 | 0.754365 | 0.024739 | 0.987568 | 98 | 0.023906 | 0.988499 | 2.816M | 93.4% |
| 4 | 0.705366 | 0.075618 | 0.965586 | 98 | 0.059868 | 0.974010 | 2.819M | 89.0% |
| 5 | 0.316007 | 0.052481 | 0.975801 | 96 | 0.051562 | 0.975804 | 2.812M | 89.8% |

Interpretation:
- Layer 0 is almost exactly preserved by the copied low-QK slice before training. Training at `lr=0.03` slightly worsened the final metric, so layer 0 should probably use a much smaller LR or no training.
- Layer 1 is easy: final rel MSE `0.0167`, cosine `0.9916`.
- Layer 2 matches the earlier LR sweep: final rel MSE `0.0512`, best-step rel MSE `0.0504`.
- Layer 3 is the most interesting positive case. It starts extremely bad (`0.7544` rel MSE) but trains down to best-step rel MSE `0.0239`, better than layers 2, 4, and 5.
- Layer 4 is the hardest under this setup. It starts bad, improves substantially, but oscillates late; best-step rel MSE is `0.0599`, while final regresses to `0.0756`.
- Layer 5 is similar to layer 2 at convergence, ending around rel MSE `0.0525`.
- `lr=0.03` is not uniformly stable. It works well enough for the fast sweep, but layer 4 in particular needs an LR sweep or decay/checkpoint selection.
- The across-layer result supports the hypothesis that Q/K can be compressed aggressively for output matching, but the failure mode is not monotonic with depth. Some late layers remain matchable, while layer 4 is the clearest bottleneck in this run.

## 2026-05-26 Full-Model Low-QK Attention Replacement

User clarification: the intended test was one Pythia-70M student model with every attention block replaced, not six separate per-layer modules trained independently.

Implementation:
- Added `layer_distill/low_qk_model.py`.
- Added `modal_layer_distill.py --mode low-qk-model`.
- Replaced all six GPT-NeoX attention modules with `qk_dim=2` low-QK adapters initialized from the original attentions.
- Froze the base student model and trained only the replacement attention adapters.
- Trained all replacement attentions jointly against the original teacher model's final hidden state.
- Used Muon for matrix parameters and AdamW for non-Muon parameters.
- Per-step logs include loss, relative MSE, cosine, grad norm, throughput, trainable parameter count, replaced layer count, and sampled GPU stats.

Validation:
- Added tests in `tests/test_low_qk_model.py`.
- Fixed a CUDA/CPU placement bug found on the first H100 attempt by moving replacement adapters to the source attention device and dtype during replacement.
- Full suite after fix: `uv run --with pytest pytest -q` -> `40 passed in 0.85s`.

H100 run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-IVLu7tln82UjxtE3NYYWzK`
- Local artifacts: `runs/modal_low_qk2_full_model_h100_devicefix_20260526`
- Model: `EleutherAI/pythia-70m`
- Replaced layers: `6`
- Trainable adapter params: `3250368`
- Batch size: `1024`
- Sequence length: `256`
- Tokens per step: `262144`
- Steps per LR: `100`
- Dtype: `bf16`
- Peak sampled H100 memory: `43811 MB`

Results:

| lr | start rel MSE | final rel MSE | final cosine | best step | best rel MSE | avg tok/s | avg sampled GPU util |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.003 | 0.019000 | 0.006611 | 0.996705 | 98 | 0.006602 | 932035 | 98.99% |
| 0.010 | 0.019000 | 0.006126 | 0.996947 | 100 | 0.006126 | 931598 | 98.99% |
| 0.030 | 0.019000 | 0.010810 | 0.994623 | 91 | 0.009353 | 931659 | 99.03% |

Interpretation:
- Best LR in this sweep was `0.01`, ending at final-hidden-state rel MSE `0.006126` and cosine `0.996947`.
- This full-model end-to-end target is much easier than standalone per-attention output matching: even the initialized low-QK model starts at final-hidden-state rel MSE `0.0190`, then trains down to `0.0061`.
- `lr=0.03` is too hot for the joint setting; it reached its best result at step 91 and regressed by step 100.
- GPU utilization was healthy: sampled steady-state utilization averaged about `99%` across all runs.
- Important caveat: this target was final hidden state, not logits, KL, perplexity, or generation quality. The result says the network can jointly absorb all six low-QK attention replacements well under a final-hidden distillation objective. The next stricter test is logits/perplexity evaluation, or training with a chunked logits/KL objective to avoid materializing huge vocab outputs at this batch size.

## 2026-05-26 Full-Model Low-QK Perplexity Evaluation

Follow-up: evaluated true next-token perplexity for the full-model low-QK attention checkpoints.

Implementation:
- Added `LowQKPerplexityEvalConfig` and `run_low_qk_perplexity_eval` in `layer_distill/low_qk_model.py`.
- Added `modal_layer_distill.py --mode low-qk-ppl`.
- Evaluation loads `AutoModelForCausalLM`, replaces all six attention blocks with low-QK adapters, loads saved adapter-only checkpoints, and computes causal LM NLL/PPL.
- Added chunked cross entropy over logits. Non-chunked batch-512 and batch-256 attempts OOMed inside the built-in Transformers loss due to full flattened vocab-loss allocation.

Validation:
- Local smoke evaluated teacher plus the `lr=0.01` adapter on a tiny Wikitext slice.
- Full suite after eval implementation: `uv run --with pytest pytest -q` -> `45 passed in 1.48s`.

Canonical H100 eval:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-0ai6jTsj6pKyuSuLbWX13i`
- Local artifacts: `runs/modal_low_qk2_full_model_ppl_wikitext_test_b512_chunked_20260526`
- Dataset: `Salesforce/wikitext`, `wikitext-2-raw-v1`, `test`
- Total next-token labels: `287693`
- Batch size: `512`
- Sequence length: `512`
- CE chunk size: `32768` tokens
- Dtype: `bf16`

Results:

| model | lr | loss | ppl | labels | steps | eval tok/s | peak sampled H100 mem | avg sampled GPU util |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher |  | 5.894453 | 363.02 | 287693 | 2 | 180379 | 68589 MB | 37.36% |
| low-QK adapters | 0.003 | 7.599376 | 1996.95 | 287693 | 2 | 587417 | 75761 MB | 52.00% |
| low-QK adapters | 0.010 | 7.261618 | 1424.56 | 287693 | 2 | 600144 | 75765 MB | 72.67% |
| low-QK adapters | 0.030 | 9.130232 | 9230.16 | 287693 | 2 | 598115 | 75765 MB | 66.67% |

Interpretation:
- Best adapter remains `lr=0.01`, matching the final-hidden-state distillation winner.
- The result is qualitatively bad for language modeling: `lr=0.01` has PPL `1424.56` versus teacher PPL `363.02`, despite final-hidden-state relative MSE `0.006126`.
- This means final hidden MSE alone is not a sufficient proxy for preserving the LM head distribution. Small hidden errors can be highly amplified by the unembedding, especially if they move along high-logit-sensitivity directions.
- `lr=0.03` was already unstable in final-hidden MSE and is catastrophic on PPL.
- Next experiment should train/evaluate against logits directly: chunked KL or CE distillation, likely with smaller per-step microbatches or chunked vocab/logit handling. The current low-QK architecture may still be viable, but the objective needs to see the LM head.

## 2026-05-26 Pythia 31M to 70M Exponential Attention Mixtures

Follow-up to the per-head distribution linear-combination experiment: fit geometric/exponential mixtures of smaller-model attention heads to approximate each larger-model attention head. For each 70M head, the composed distribution is:

`p_combo = softmax(sum_i w_i log p_31m_i)`

with simplex-constrained weights `w`. This tests whether combining heads in log-probability space is a better basis than arithmetic averaging of post-softmax distributions.

Implementation:
- Added `fit_head_exponential_combinations` and `compare_cross_model_exponential_combinations` in `layer_distill/attention_analysis.py`.
- Added `--exponential-combos` CLI path, writing `exponential_combo.jsonl` and `exponential_combo_summary.json`.
- Preserved causal support during exponential composition so masked future-token positions do not receive artificial mass.
- Added tests for exact geometric-mixture reconstruction and row emission.

Validation:
- Full suite after implementation: `uv run --with pytest pytest -q` -> `51 passed in 2.51s`.

Run:
- Command: `uv run python -m layer_distill.attention_analysis --exponential-combos --output-dir runs/attention_pythia31m_70m_exponential_combo_20260526 --small-model EleutherAI/pythia-31m --big-model EleutherAI/pythia-70m --prompts 'In a small gridworld, the agent starts at the red square and must reach the blue square.||Question: If Alice has three keys and gives Bob one key, how many keys does Alice still have? Answer:' --max-length 48 --dtype fp32 --device cpu --fit-steps 300 --fit-lr 0.2`
- Comparison baseline: `runs/attention_pythia31m_70m_linear_combo_20260526`.

Results:

| layer | linear JSD | exponential JSD | rel improvement | linear eff heads | exponential eff heads | linear top wt | exponential top wt |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0887 | 0.0874 | 1.5% | 2.83 | 2.87 | 0.57 | 0.57 |
| 1 | 0.0854 | 0.0873 | -2.2% | 3.41 | 2.85 | 0.53 | 0.58 |
| 2 | 0.0980 | 0.0986 | -0.6% | 2.49 | 2.30 | 0.67 | 0.69 |
| 3 | 0.1881 | 0.1885 | -0.2% | 2.18 | 1.93 | 0.79 | 0.79 |
| 4 | 0.2430 | 0.2346 | 3.4% | 1.88 | 2.38 | 0.76 | 0.62 |
| 5 | 0.3026 | 0.2788 | 7.9% | 1.50 | 1.89 | 0.84 | 0.76 |

Overall:
- Linear mean JSD: `0.167619`
- Exponential mean JSD: `0.162521`
- Relative JSD improvement: `3.04%`
- Exponential mean effective heads: `2.37`
- Exponential mean top weight: `0.667`

Interpretation:
- Exponential mixtures are modestly better overall, but the gain is concentrated in later layers, especially layer 5.
- Early/mid layers do not improve meaningfully; layers 1 to 3 are slightly worse than arithmetic mixtures.
- The late-layer improvement suggests some mismatches are better described as multiplicative/log-probability combinations of attention patterns rather than additive probability mixtures.
- The weights are still sparse-ish: mean effective heads is only about `2.37`, so this is not using all 31M heads diffusely.

## 2026-05-26 Pythia 31M to 70M Wasserstein Attention Mixtures

Follow-up: fit 1D Wasserstein-style mixtures of smaller-model attention heads to approximate each larger-model attention head. For each query row, each source attention distribution is treated as a distribution over ordered key positions. The mixture computes a discrete W2-style barycenter by averaging source quantile positions under simplex-constrained head weights, then linearly depositing the resulting quantile mass back onto adjacent integer key indices.

Important caveat:
- This uses token position geometry only. It is not an optimal transport solve over token semantics.
- The discrete projection back to key indices is approximate. It is useful for asking whether target heads look like positional mass shifts of source heads.

Implementation:
- Added `fit_head_wasserstein_combinations` and `compare_cross_model_wasserstein_combinations` in `layer_distill/attention_analysis.py`.
- Added `--wasserstein-combos` CLI path, writing `wasserstein_combo.jsonl` and `wasserstein_combo_summary.json`.
- Added tests for exact position-mixture reconstruction and row emission.

Validation:
- Full suite after implementation: `uv run --with pytest pytest -q` -> `53 passed in 2.68s`.

Run:
- Command: `uv run python -m layer_distill.attention_analysis --wasserstein-combos --output-dir runs/attention_pythia31m_70m_wasserstein_combo_20260526 --small-model EleutherAI/pythia-31m --big-model EleutherAI/pythia-70m --prompts 'In a small gridworld, the agent starts at the red square and must reach the blue square.||Question: If Alice has three keys and gives Bob one key, how many keys does Alice still have? Answer:' --max-length 48 --dtype fp32 --device cpu --fit-steps 300 --fit-lr 0.2`
- Comparison artifact: `runs/attention_pythia31m_70m_wasserstein_combo_20260526/wasserstein_vs_linear_exponential_comparison.json`

Results:

| layer | linear JSD | exponential JSD | wasserstein JSD | wasserstein - linear | wasserstein - exponential |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0887 | 0.0874 | 0.1118 | +0.0232 | +0.0245 |
| 1 | 0.0854 | 0.0873 | 0.1204 | +0.0350 | +0.0331 |
| 2 | 0.0980 | 0.0986 | 0.1118 | +0.0137 | +0.0132 |
| 3 | 0.1881 | 0.1885 | 0.2269 | +0.0389 | +0.0385 |
| 4 | 0.2430 | 0.2346 | 0.2581 | +0.0152 | +0.0235 |
| 5 | 0.3026 | 0.2788 | 0.3614 | +0.0588 | +0.0826 |

Overall:
- Linear mean JSD: `0.167619`
- Exponential mean JSD: `0.162521`
- Wasserstein mean JSD: `0.198415`
- Wasserstein mean effective heads: `2.28`
- Wasserstein mean top weight: `0.721`

Interpretation:
- Wasserstein interpolation is clearly worse here than both arithmetic and exponential mixtures.
- It loses in every layer and especially in layer 5.
- This argues against the mismatch being mostly a simple positional transport/shift of smaller-model heads along the key axis.
- Exponential remains the best of the tested distribution-composition families on this 31M to 70M two-prompt setup.

## 2026-05-26 Pythia 1.4B to 2.8B Exponential Attention Mixtures

Goal: run the exponential attention-head composition test on the cleanest larger adjacent transition from the previous sweep, `EleutherAI/pythia-1.4b -> EleutherAI/pythia-2.8b`.

Implementation:
- Added `run_attention_combo_analysis` dispatch helper in `layer_distill/attention_analysis.py`.
- Added Modal `--mode attention-combo` in `modal_layer_distill.py`.
- The Modal run must use the `jthomams477` profile. The local environment had `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` set for another workspace, so the successful run used `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET` after `modal profile activate jthomams477`.

Validation:
- Full suite after wrapper changes: `uv run --with pytest pytest -q` -> `55 passed in 2.02s`.
- Syntax check: `python3 -m py_compile modal_layer_distill.py layer_distill/attention_analysis.py`.

H100 run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-dQJ7ZktU6HX06Olfys9RED`
- Volume path: `/results/pythia_1_4b_to_2_8b_exponential_combo_20260526`
- Local copy: `runs/modal_pythia_1_4b_to_2_8b_exponential_combo_20260526/pythia_1_4b_to_2_8b_exponential_combo_20260526`
- Comparison artifact: `runs/modal_pythia_1_4b_to_2_8b_exponential_combo_20260526/pythia_1_4b_to_2_8b_exponential_combo_20260526/exponential_vs_matching_comparison.json`
- Prompts: gridworld navigation; arithmetic/key-counting question
- Max sequence length: `48`
- Fit steps: `300`
- Fit LR: `0.2`
- Dtype: `bf16`
- Rows: `1536` = 2 prompts x 24 layers x 32 target heads

Overall:
- Same-index JSD from previous matching sweep: `0.162091`
- Best single-head match JSD: `0.083667`
- Injective match JSD: `0.089161`
- Exponential-combo JSD: `0.057038`
- Improvement vs best single-head matching: `31.83%`
- Improvement vs injective matching: `36.03%`
- Mean effective 1.4B heads per 2.8B head: `4.46`
- Mean top weight: `0.438`

Layer results:

| layer | best-match JSD | exponential JSD | rel improvement | eff heads | top wt |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0811 | 0.0552 | 32.0% | 5.46 | 0.35 |
| 1 | 0.0749 | 0.0503 | 32.8% | 4.62 | 0.43 |
| 2 | 0.0775 | 0.0623 | 19.6% | 4.61 | 0.42 |
| 3 | 0.1458 | 0.1632 | -11.9% | 3.27 | 0.53 |
| 4 | 0.0584 | 0.0587 | -0.4% | 4.84 | 0.42 |
| 5 | 0.0692 | 0.0619 | 10.5% | 3.65 | 0.49 |
| 6 | 0.0671 | 0.0423 | 37.0% | 4.10 | 0.44 |
| 7 | 0.0554 | 0.0374 | 32.5% | 4.39 | 0.42 |
| 8 | 0.0664 | 0.0427 | 35.7% | 4.19 | 0.44 |
| 9 | 0.0482 | 0.0425 | 11.8% | 4.40 | 0.44 |
| 10 | 0.0464 | 0.0592 | -27.5% | 3.73 | 0.49 |
| 11 | 0.0502 | 0.0426 | 15.1% | 4.41 | 0.42 |
| 12 | 0.0542 | 0.0380 | 29.9% | 5.05 | 0.38 |
| 13 | 0.0500 | 0.0402 | 19.6% | 4.41 | 0.44 |
| 14 | 0.0626 | 0.0510 | 18.5% | 4.23 | 0.42 |
| 15 | 0.0600 | 0.0410 | 31.6% | 4.71 | 0.40 |
| 16 | 0.0505 | 0.0583 | -15.4% | 3.97 | 0.51 |
| 17 | 0.0462 | 0.0523 | -13.2% | 3.67 | 0.53 |
| 18 | 0.0735 | 0.0407 | 44.6% | 4.80 | 0.37 |
| 19 | 0.0583 | 0.0304 | 47.9% | 5.48 | 0.34 |
| 20 | 0.1683 | 0.0713 | 57.6% | 3.76 | 0.54 |
| 21 | 0.1530 | 0.0480 | 68.6% | 4.91 | 0.43 |
| 22 | 0.2004 | 0.0779 | 61.1% | 5.95 | 0.38 |
| 23 | 0.1903 | 0.1016 | 46.6% | 4.49 | 0.48 |

Interpretation:
- Exponential mixtures are very strong on this transition: `0.0570` mean JSD is lower than the already-clean `1b -> 1.4b` best-match result (`0.0660`) from the previous sweep.
- The gain is broad but not universal. Layers `3`, `10`, `16`, and `17` get worse than the best single-head match.
- Late layers `20` to `23` benefit heavily, which suggests many 2.8B late heads are compositional log-probability mixtures of 1.4B heads rather than one-to-one analogues.
- Compared with the 31M to 70M exponential run, this larger transition uses substantially more source heads per target head: mean effective heads `4.46` vs `2.37`.

## 2026-05-26 Pythia 1.4B to 2.8B Hybrid Attention-Distribution Transplant

Goal: evaluate a 2.8B causal LM that keeps the bigger model's FFNs, embeddings, LM head, attention V projections, and attention output projections, but replaces the attention distribution used before V with the 1.4B exponential-combo attention distributions.

Definition:
- Run the 1.4B model in parallel on the same token batch and collect post-softmax attention distributions.
- For each comparable layer/head, construct the 2.8B target-head distribution as `softmax(sum_i w_i log p_1.4B_i)`.
- Use those distributions inside the 2.8B model's attention blocks in place of native QK-derived weights.
- Keep 2.8B V and output projection (`dense`) from the original attention module.
- Keep 2.8B MLPs/FFNs, layer norms, embeddings, final norm, and LM head.
- Only the first 24 layers are replaced because 1.4B has 24 layers and 2.8B has 32 layers. The final 8 2.8B attention layers remain native.
- The exponential-combo weights are prompt-averaged from `pythia_1_4b_to_2_8b_exponential_combo_20260526`; they are not refit on Wikitext eval batches.

Implementation:
- Added `layer_distill/hybrid_attention.py`.
- Added `ExternalAttentionWeightsGPTNeoXAttention`, which ignores native Q/K scores but still computes native 2.8B V and applies native 2.8B output projection.
- Added `HybridAttentionEvalConfig` and `run_hybrid_attention_eval`.
- Added Modal `--mode hybrid-attn`.
- Added tests for external attention weights, combo-weight aggregation, combo distribution construction, and config validation.

Validation:
- Full suite after implementation/fix: `uv run --with pytest pytest -q` -> `60 passed in 2.97s`.
- Syntax check: `python3 -m py_compile layer_distill/hybrid_attention.py modal_layer_distill.py`.
- First H100 attempt failed after the baseline because this Transformers build's `GPTNeoXAttention` exposes `head_size` but not `num_attention_heads`; fixed by deriving head count from the dense projection size.

H100 run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-YsP7PFv0rDB8bnlRFfNgf5`
- Volume path: `/results/pythia_1_4b_to_2_8b_hybrid_expcombo_attn_eval_fix_20260526`
- Local copy: `runs/modal_pythia_1_4b_to_2_8b_hybrid_expcombo_attn_eval_fix_20260526/pythia_1_4b_to_2_8b_hybrid_expcombo_attn_eval_fix_20260526`
- Comparison artifact: `runs/modal_pythia_1_4b_to_2_8b_hybrid_expcombo_attn_eval_fix_20260526/pythia_1_4b_to_2_8b_hybrid_expcombo_attn_eval_fix_20260526/hybrid_vs_big_comparison.json`
- Dataset: Wikitext-2 raw test
- Eval labels: `16320`
- Batch size: `8`
- Sequence length: `256`
- Steps: `8`
- Dtype: `bf16`

Results:

| run | loss | ppl | labels | replaced layers | native tail layers |
| --- | ---: | ---: | ---: | ---: | ---: |
| native 2.8B | 2.9330 | 18.78 | 16320 | 0 | 32 |
| hybrid exp-combo QK + 2.8B V/O/FFN | 5.8196 | 336.85 | 16320 | 24 | 8 |

Derived:
- Hybrid loss delta vs native 2.8B: `+2.8866`
- Hybrid PPL ratio vs native 2.8B: `17.93x`
- Combo tensor shape: `[24, 32, 16]` = 24 layers, 32 target heads, 16 source heads

Interpretation:
- This transplant is not viable as-is for language modeling. Even though the exponential-combo distributions match 2.8B attention heads well in JSD, swapping them into the 2.8B forward pass causes a large PPL regression.
- The likely issue is activation coupling: the 1.4B attention distributions were generated from the 1.4B residual stream, while 2.8B V/O/MLP operate on a different residual stream. Good static distribution similarity on fixed prompts does not imply the distributions are interchangeable inside the larger model's recurrent computation.
- The result suggests that preserving V/O and FFNs is not enough; QK/attention distributions are coupled to the model's hidden-state geometry and to downstream layers.
- A less destructive follow-up would be partial interpolation inside the 2.8B model, e.g. `attn = (1-alpha) * native_2.8B_attn + alpha * expcombo_1.4B_attn`, or replacing only selected layers where exponential combo JSD was strongest.

## 2026-05-26 Reverse Hybrid Attention-Distribution Transplant

Goal: try the opposite direction. Fit exponential mixtures of 2.8B attention heads to approximate 1.4B attention heads, then run a 1.4B LM that keeps its own V projections, output projections, FFNs, norms, embeddings, and LM head while replacing its attention distributions with the 2.8B-derived exponential mixtures.

Reverse combo fit:
- Source model for attention distributions: `EleutherAI/pythia-2.8b`
- Target heads: `EleutherAI/pythia-1.4b`
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-EUsDrX8kXXRlCoB26lP0FP`
- Volume path: `/results/pythia_2_8b_to_1_4b_exponential_combo_20260526`
- Local copy: `runs/modal_pythia_2_8b_to_1_4b_exponential_combo_20260526/pythia_2_8b_to_1_4b_exponential_combo_20260526`
- Rows: `768` = 2 prompts x 24 layers x 16 target heads
- Combo shape for eval: `[24, 16, 32]`

Reverse combo results:
- Mean JSD: `0.063767`
- Mean effective 2.8B heads per 1.4B head: `4.80`
- Mean top weight: `0.407`
- Best row: layer `1`, target head `11`, JSD `0.00259`
- Worst row: layer `21`, target head `8`, JSD `0.26442`

Reverse hybrid eval:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-n34mKsj5mXreyQoAtpPqqL`
- Volume path: `/results/pythia_2_8b_to_1_4b_hybrid_expcombo_attn_eval_20260526`
- Local copy: `runs/modal_pythia_2_8b_to_1_4b_hybrid_expcombo_attn_eval_20260526/pythia_2_8b_to_1_4b_hybrid_expcombo_attn_eval_20260526`
- Comparison artifact: `runs/modal_pythia_2_8b_to_1_4b_hybrid_expcombo_attn_eval_20260526/pythia_2_8b_to_1_4b_hybrid_expcombo_attn_eval_20260526/hybrid_vs_native_1_4b_comparison.json`
- Dataset: Wikitext-2 raw test
- Eval labels: `16320`
- Batch size: `8`
- Sequence length: `256`
- Steps: `8`
- Dtype: `bf16`

Results:

| run | loss | ppl | labels | replaced layers | native tail layers |
| --- | ---: | ---: | ---: | ---: | ---: |
| native 1.4B | 3.0797 | 21.75 | 16320 | 0 | 24 |
| hybrid 2.8B exp-combo QK + 1.4B V/O/FFN | 5.0049 | 149.14 | 16320 | 24 | 0 |

Derived:
- Hybrid loss delta vs native 1.4B: `+1.9252`
- Hybrid PPL ratio vs native 1.4B: `6.86x`

Comparison to previous direction:
- 1.4B-derived attention into 2.8B target: PPL `336.85` vs native 2.8B `18.78` = `17.93x` worse.
- 2.8B-derived attention into 1.4B target: PPL `149.14` vs native 1.4B `21.75` = `6.86x` worse.

Interpretation:
- Reverse direction is less destructive than the previous transplant, but still clearly bad for language modeling.
- Static exponential-combo attention similarity is again insufficient for forward-pass interchangeability.
- The smaller 1.4B V/O/FFN stack tolerates externally supplied 2.8B-derived attention distributions better than the 2.8B stack tolerated 1.4B-derived distributions, but the residual-stream coupling problem remains large.
- Next reasonable test is not a full replacement, but an interpolation or layer subset: e.g. mix native and external attention distributions by alpha, or replace only layers with low combo JSD and avoid late layers with high reverse-combo JSD.

## 2026-05-26 Reverse Hybrid Single-Layer Sweep

Goal: rerun the reverse hybrid attention-distribution transplant independently for each 1.4B layer. Each run keeps native 1.4B weights everywhere except one target layer, whose post-softmax attention distribution is replaced by the fitted exponential combination of 2.8B attention heads.

Run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-NLxdT4WUCu4cLCKZmKymX6`
- Volume path: `/results/pythia_2_8b_to_1_4b_hybrid_expcombo_attn_per_layer_20260526`
- Local copy: `runs/modal_pythia_2_8b_to_1_4b_hybrid_expcombo_attn_per_layer_20260526/pythia_2_8b_to_1_4b_hybrid_expcombo_attn_per_layer_20260526`
- Comparison artifacts:
  - `per_layer_hybrid_vs_native_comparison.json`
  - `per_layer_hybrid_vs_native_comparison.csv`
  - `per_layer_hybrid_vs_native_comparison.md`
- Dataset: Wikitext-2 raw test
- Eval labels per run: `16320`
- Batch size: `8`
- Sequence length: `256`
- Steps: `8`
- Dtype: `bf16`

Native 1.4B baseline:
- Loss: `3.079706`
- PPL: `21.752004`

Per-layer results:

| layer | ppl | delta ppl | ratio | loss delta | mean combo JSD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 22.856 | +1.104 | 1.051x | +0.0495 | 0.0431 |
| 1 | 22.709 | +0.957 | 1.044x | +0.0430 | 0.0414 |
| 2 | 22.328 | +0.576 | 1.026x | +0.0262 | 0.0477 |
| 3 | 23.180 | +1.428 | 1.066x | +0.0636 | 0.1158 |
| 4 | 22.954 | +1.202 | 1.055x | +0.0538 | 0.0414 |
| 5 | 23.400 | +1.648 | 1.076x | +0.0730 | 0.0448 |
| 6 | 23.098 | +1.346 | 1.062x | +0.0601 | 0.0465 |
| 7 | 23.402 | +1.650 | 1.076x | +0.0731 | 0.0387 |
| 8 | 23.781 | +2.029 | 1.093x | +0.0892 | 0.0419 |
| 9 | 23.559 | +1.807 | 1.083x | +0.0798 | 0.0305 |
| 10 | 24.180 | +2.428 | 1.112x | +0.1058 | 0.0265 |
| 11 | 24.479 | +2.727 | 1.125x | +0.1181 | 0.0350 |
| 12 | 24.917 | +3.165 | 1.146x | +0.1359 | 0.0375 |
| 13 | 22.647 | +0.895 | 1.041x | +0.0403 | 0.0356 |
| 14 | 23.338 | +1.586 | 1.073x | +0.0704 | 0.0469 |
| 15 | 23.391 | +1.639 | 1.075x | +0.0726 | 0.0499 |
| 16 | 22.944 | +1.192 | 1.055x | +0.0533 | 0.0410 |
| 17 | 23.613 | +1.861 | 1.086x | +0.0821 | 0.0374 |
| 18 | 23.069 | +1.317 | 1.061x | +0.0588 | 0.0601 |
| 19 | 22.334 | +0.582 | 1.027x | +0.0264 | 0.0503 |
| 20 | 21.885 | +0.133 | 1.006x | +0.0061 | 0.1492 |
| 21 | 22.327 | +0.575 | 1.026x | +0.0261 | 0.1397 |
| 22 | 21.896 | +0.144 | 1.007x | +0.0066 | 0.1690 |
| 23 | 21.952 | +0.200 | 1.009x | +0.0092 | 0.1603 |

Derived:
- Best single-layer replacement: layer `20`, PPL `21.885`, only `1.006x` native.
- Worst single-layer replacement: layer `12`, PPL `24.917`, `1.146x` native.
- Mean single-layer PPL: `23.093`, mean ratio `1.062x` native.
- Full all-layer reverse replacement was much worse: PPL `149.14`, `6.86x` native.
- Pearson correlation between mean combo JSD and per-layer loss damage: `-0.668`.

Interpretation:
- Damage is strongly position-dependent. Middle layers, especially `10-12`, are most sensitive even though their exponential-combo JSD is low.
- Late layers `20-23` have the worst static combo JSD, but swapping them barely changes PPL. For this experiment, JSD is measuring distribution fit on fixed prompts, not causal importance in the LM forward pass.
- The all-layer failure is not explained by any one catastrophic layer. The error compounds across layers/residual states.

## 2026-05-26 Reverse Hybrid Greedy Layer Sweep

Goal: greedily add one replaced attention-distribution layer at a time. At each round, evaluate every remaining candidate layer on top of the current selected set, then keep the candidate with lowest PPL.

Run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-j1vOYKgQDATNu1iZ2E29md`
- Volume path: `/results/pythia_2_8b_to_1_4b_hybrid_expcombo_attn_greedy_20260526`
- Local copy: `runs/modal_pythia_2_8b_to_1_4b_hybrid_expcombo_attn_greedy_20260526/pythia_2_8b_to_1_4b_hybrid_expcombo_attn_greedy_20260526`
- Candidate evaluations: `300`
- Analysis artifacts:
  - `greedy_layer_path.json`
  - `greedy_layer_path.md`
  - `greedy_layer_path_analysis.json`
  - `greedy_layer_path_analysis.csv`
  - `greedy_layer_path_analysis.md`
- Dataset: Wikitext-2 raw test
- Eval labels per candidate: `16320`
- Batch size: `8`
- Sequence length: `256`
- Steps: `8`
- Dtype: `bf16`

Native 1.4B baseline:
- Loss: `3.079706`
- PPL: `21.752004`

Greedy order:

`20, 22, 23, 2, 19, 21, 13, 4, 16, 18, 0, 9, 17, 6, 1, 10, 14, 7, 11, 5, 15, 3, 8, 12`

Path:

| round | add | ppl | ratio | step delta ppl |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 21.885 | 1.006x | +0.133 |
| 2 | 22 | 22.061 | 1.014x | +0.176 |
| 3 | 23 | 22.311 | 1.026x | +0.250 |
| 4 | 2 | 22.816 | 1.049x | +0.505 |
| 5 | 19 | 23.386 | 1.075x | +0.571 |
| 6 | 21 | 24.022 | 1.104x | +0.636 |
| 7 | 13 | 24.735 | 1.137x | +0.713 |
| 8 | 4 | 25.671 | 1.180x | +0.935 |
| 9 | 16 | 26.789 | 1.232x | +1.118 |
| 10 | 18 | 28.250 | 1.299x | +1.461 |
| 11 | 0 | 29.779 | 1.369x | +1.529 |
| 12 | 9 | 31.284 | 1.438x | +1.505 |
| 13 | 17 | 33.099 | 1.522x | +1.815 |
| 14 | 6 | 35.680 | 1.640x | +2.581 |
| 15 | 1 | 38.899 | 1.788x | +3.219 |
| 16 | 10 | 42.533 | 1.955x | +3.633 |
| 17 | 14 | 46.769 | 2.150x | +4.236 |
| 18 | 7 | 52.350 | 2.407x | +5.582 |
| 19 | 11 | 59.789 | 2.749x | +7.438 |
| 20 | 5 | 68.753 | 3.161x | +8.964 |
| 21 | 15 | 81.374 | 3.741x | +12.621 |
| 22 | 3 | 95.976 | 4.412x | +14.602 |
| 23 | 8 | 115.691 | 5.319x | +19.715 |
| 24 | 12 | 149.137 | 6.856x | +33.446 |

Interpretation:
- Greedy confirms the low-damage subset is mostly late layers: `20,22,23`, then `2,19,21`.
- The first 6 replacements stay near native-ish performance, PPL `24.02` vs native `21.75`.
- Damage accelerates after roughly 10-12 replaced layers and becomes severe once the sensitive middle layers are unavoidable.
- Layer `12` is last in the greedy order and again looks like the most destructive layer under this setup.
- The final all-layer point exactly matches the prior full reverse hybrid run, PPL `149.14`, so the greedy path is internally consistent with the earlier endpoint.

## 2026-05-26 Greedy Full-Layer Removal Sweep

Goal: test full transformer-block removal with the same greedy protocol. At each round, skip every remaining candidate layer on top of the currently skipped set, then keep the candidate with lowest PPL.

Implementation detail: a removed layer is replaced by an identity transformer block that returns the incoming hidden states unchanged. The eval uses native `EleutherAI/pythia-1.4b` embeddings, final norm, and LM head.

Run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-yHrn9IleN9ymo8VucUmW9q`
- Volume path: `/results/pythia_1_4b_greedy_layer_removal_20260526`
- Local copy: `runs/modal_pythia_1_4b_greedy_layer_removal_20260526/pythia_1_4b_greedy_layer_removal_20260526`
- Candidate evaluations: `300`
- Analysis artifacts:
  - `greedy_removal_path.json`
  - `greedy_removal_path.md`
  - `greedy_removal_path_analysis.json`
  - `greedy_removal_path_analysis.csv`
  - `greedy_removal_path_analysis.md`
- Dataset: Wikitext-2 raw test
- Eval labels per candidate: `16320`
- Batch size: `8`
- Sequence length: `256`
- Steps: `8`
- Dtype: `bf16`

Native 1.4B baseline:
- Loss: `3.079706`
- PPL: `21.752004`

Greedy removal order:

`20, 9, 22, 17, 8, 21, 12, 2, 18, 23, 19, 16, 15, 14, 13, 7, 6, 10, 5, 4, 1, 3, 11, 0`

Path:

| round | remove | ppl | ratio | step delta ppl |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 24.196 | 1.112x | +2.444 |
| 2 | 9 | 27.201 | 1.251x | +3.005 |
| 3 | 22 | 31.753 | 1.460x | +4.551 |
| 4 | 17 | 37.683 | 1.732x | +5.930 |
| 5 | 8 | 45.611 | 2.097x | +7.928 |
| 6 | 21 | 59.465 | 2.734x | +13.854 |
| 7 | 12 | 78.104 | 3.591x | +18.640 |
| 8 | 2 | 115.352 | 5.303x | +37.248 |
| 9 | 18 | 165.880 | 7.626x | +50.528 |
| 10 | 23 | 236.229 | 10.860x | +70.349 |
| 11 | 19 | 235.114 | 10.809x | -1.115 |
| 12 | 16 | 306.294 | 14.081x | +71.180 |
| 13 | 15 | 433.689 | 19.938x | +127.396 |
| 14 | 14 | 562.654 | 25.867x | +128.965 |
| 15 | 13 | 769.047 | 35.356x | +206.393 |
| 16 | 7 | 1130.766 | 51.984x | +361.718 |
| 17 | 6 | 1676.413 | 77.070x | +545.648 |
| 18 | 10 | 2625.898 | 120.719x | +949.485 |
| 19 | 5 | 4321.007 | 198.649x | +1695.109 |
| 20 | 4 | 7335.195 | 337.218x | +3014.188 |
| 21 | 1 | 9437.089 | 433.844x | +2101.895 |
| 22 | 3 | 15166.569 | 697.258x | +5729.480 |
| 23 | 11 | 24001.536 | 1103.424x | +8834.967 |
| 24 | 0 | 371525.993 | 17080.082x | +347524.457 |

Interpretation:
- Full block removal is much more destructive than attention-distribution replacement. In the earlier greedy attention-replacement path, 6 replaced layers gave PPL `24.02`; here 6 removed layers already gives PPL `59.46`.
- The least-bad first removal is layer `20`, but even that is a `1.11x` PPL hit.
- Layer `0` is effectively indispensable under this identity-skip removal setup. Single removal of layer `0` produced very high PPL during round 1, and greedy left it for last; all-layer removal ends at PPL `371526`.
- The model tolerates removing a small number of late or mid-late blocks, but the degradation compounds quickly after about 4-6 removed layers.

## 2026-05-26 ResComp-Inspired 2:4 Sparsity Smoke

Reminder: come back to the attention replacement vs layer removal thread after this sparsity direction reaches a conclusion.

Goal: adapt the ResComp / compensation-aware residual idea from quantization to structured sparsity. First test uses `2:4` sparsity on `EleutherAI/pythia-31m`.

Implementation:
- Added a 2:4 projector over linear weight input columns: in each contiguous 4-column group, keep the top-2 magnitudes per output row.
- Compared three conversion methods:
  - `magnitude`: direct 2:4 magnitude pruning.
  - `sparsegpt`: GPTQ/SparseGPT-style group compensation using calibration Hessian.
  - `rescomp`: same group compensation plus the paper-inspired residual terms from quant-flow vs fp-flow inputs and compensation-aware weight discrepancy.
- Pruned all `torch.nn.Linear` modules with input dimension divisible by 4, including `embed_out`; total modules: `25`.
- This is a real model conversion / eval path, not a semantic shortcut. The final models preserve exact `0.5` density over pruned linear weights.

Run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-tT6UsaDUbBoO4ayOF0fnhZ`
- Volume path: `/results/pythia_31m_sparse24_rescomp_20260526`
- Local copy: `runs/modal_pythia_31m_sparse24_rescomp_20260526/pythia_31m_sparse24_rescomp_20260526`
- Calibration: Wikitext-2 train, `32768` tokens captured per linear.
- Eval: Wikitext-2 test, batch size `128`, seq len `256`; test split only yielded `9` eval batches / `287385` labels.
- Dtype: `bf16`

Results:

| method | ppl | ratio vs baseline | loss | density | mean module recon rel mse |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 170.861 | 1.000x | 5.1409 |  |  |
| magnitude | 8040.152 | 47.057x | 8.9922 | 0.500 | 0.0709 |
| sparsegpt | 692.653 | 4.054x | 6.5405 | 0.500 | 0.0241 |
| rescomp | 602.831 | 3.528x | 6.4016 | 0.500 | 0.0283 |

Interpretation:
- Compensation matters a lot for 2:4. Direct magnitude pruning is effectively unusable at this sparsity level on Pythia-31M.
- The ResComp-inspired term improved perplexity over SparseGPT-style compensation by about `13%` relative: `692.65 -> 602.83`.
- Reconstruction relative MSE is not perfectly predictive of final PPL here: `rescomp` has slightly worse mean module reconstruction than `sparsegpt`, but better end-to-end PPL. That suggests the asymmetric / fp-flow alignment term may be helping in a way the local quant-flow reconstruction metric does not capture.
- The run includes `embed_out`, which likely makes this harder than transformer-only pruning. A follow-up should rerun transformer-only, then compare whether `embed_out` dominates the PPL degradation.

## 2026-05-26 Faithful ResComp 2:4 Sparse Update

Goal: rerun the 2:4 sparsity test with a closer adaptation of ResComp Algorithm 1, rather than the first coarse group-update version.

Implementation delta:
- Keep a final sparse `Q` instead of re-projecting compensated weights at the end.
- Process contiguous 2:4 groups while applying column-wise lazy updates inside each block.
- Separate:
  - `sparsegpt`: standard Hessian compensation.
  - `gptq-cae`: add only the compensation-aware `P2` term.
  - `gptaq-cae`: add both fp-flow residual `P1` and compensation-aware `P2`.
- `rescomp` remains an alias for the full `gptaq-cae` behavior.

Run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-X6gmrQ26ZNM8wg43kEZT0F`
- Volume path: `/results/pythia_31m_sparse24_rescomp_exact_20260526`
- Local copy: `runs/modal_pythia_31m_sparse24_rescomp_exact_20260526/pythia_31m_sparse24_rescomp_exact_20260526`
- Same model/eval/calibration settings as the previous sparse smoke.

Results:

| method | ppl | ratio vs baseline | loss | density | mean module recon rel mse |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 170.861 | 1.000x | 5.1409 |  |  |
| magnitude | 8040.152 | 47.057x | 8.9922 | 0.500 | 0.0709 |
| sparsegpt | 707.345 | 4.140x | 6.5615 | 0.500 | 0.0242 |
| gptq-cae | 735.186 | 4.303x | 6.6001 | 0.500 | 0.0253 |
| gptaq-cae | 606.275 | 3.548x | 6.4073 | 0.500 | 0.0281 |

Comparison to first coarse run:
- Coarse `rescomp`: PPL `602.831`.
- Faithful `gptaq-cae`: PPL `606.275`.
- These are essentially the same conclusion within this small eval: full asymmetric + compensation-aware residual is clearly better than SparseGPT-style compensation, while `P2` alone is not enough.
- `gptq-cae` being worse than `sparsegpt` suggests the compensation-aware term is only useful here when paired with fp-flow/quant-flow residual alignment (`P1`), matching the paper's story that the precise residual has both inter-layer and intra-layer pieces.

## 2026-05-26 Qronos 2:4 Sparse Update

Goal: test the QRONOS idea in the same Pythia-31M 2:4 sparse-conversion harness. This is an adaptation of QRONOS' mismatched-input error correction and future-weight diffusion to structured sparsity, not an exact reproduction of the paper's quantization setting.

Implementation:
- Added `qronos` as a 2:4 method.
- Use quant-flow inputs `Xe` and fp-flow inputs `X`.
- Accumulate `H = Xe^T Xe` and `G = Xe^T X`.
- For each 4-column group, project the current group to exact 2:4 by keeping the top-2 entries per output row, then diffuse the induced error into future columns using the Cholesky factor of the damped inverse Hessian.
- The first group uses the QRONOS-style `G/H` correction before projection; later groups use round/project-then-diffuse.

Run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-sBhRUCAMXLYGZVCDnnjyWs`
- Volume path: `/results/pythia_31m_sparse24_qronos_20260526`
- Local copy: `runs/modal_pythia_31m_sparse24_qronos_20260526/pythia_31m_sparse24_qronos_20260526`
- Same model/eval/calibration settings as the faithful ResComp sparse run.

Results:

| method | ppl | ratio vs baseline | loss | density | mean module recon rel mse |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 170.861 | 1.000x | 5.1409 |  |  |
| magnitude | 8040.152 | 47.057x | 8.9922 | 0.500 | 0.0709 |
| sparsegpt | 707.345 | 4.140x | 6.5615 | 0.500 | 0.0242 |
| gptaq-cae | 606.275 | 3.548x | 6.4073 | 0.500 | 0.0281 |
| qronos | 977.594 | 5.722x | 6.8851 | 0.500 | 0.0295 |

Interpretation:
- QRONOS did not help in this first 2:4 adaptation. It is much better than direct magnitude pruning, but worse than both `sparsegpt` and `gptaq-cae`.
- Compared with `gptaq-cae`, QRONOS is `606.28 -> 977.59` PPL, about `1.61x` worse. Compared with `sparsegpt`, it is about `1.38x` worse.
- The lower weight error for QRONOS did not translate to better model PPL. Its mean weight relative MSE was `0.168`, lower than `sparsegpt` (`0.226`) and `gptaq-cae` (`0.252`), but local output reconstruction and end-to-end PPL were worse.
- Likely issue: QRONOS is designed around scalar quantization alphabet rounding. The 2:4 projection is a group constraint, so the "round one coordinate / diffuse to future coordinates" geometry is only approximate after adapting it to 4-column groups.
- Current conclusion: for this sparsity direction, the ResComp/GPTAQ-CAE-style residual update remains the best of the tested methods.

## 2026-05-26 ResComp 2:4 Sparse Pythia-1.4B

Goal: scale the ResComp/GPTAQ-CAE 2:4 sparsity test from `EleutherAI/pythia-31m` to `EleutherAI/pythia-1.4b`.

Run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-2ErMSphWo0VWEtDqsn5mxW`
- Volume path: `/results/pythia_1_4b_sparse24_rescomp_20260526`
- Local copy: `runs/modal_pythia_1_4b_sparse24_rescomp_20260526/pythia_1_4b_sparse24_rescomp_20260526`
- Methods: `magnitude`, `sparsegpt`, `gptaq-cae`.
- Calibration: Wikitext-2 train, `32768` tokens captured per linear.
- Eval: Wikitext-2 test, batch size `64`, seq len `256`, `16` batches / `261120` labels.
- Dtype: `bf16`.
- Pruned modules: `97` linear modules, exact `0.5` density.

Results:

| method | ppl | ratio vs baseline | loss | density | mean module recon rel mse | sparsify sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 20.771 | 1.000x | 3.0335 |  |  |  |
| magnitude | 578.124 | 27.834x | 6.3598 | 0.500 | 0.0847 | 69.0 |
| sparsegpt | 50.543 | 2.433x | 3.9228 | 0.500 | 0.0219 | 136.1 |
| gptaq-cae | 44.035 | 2.120x | 3.7850 | 0.500 | 0.0256 | 247.1 |

Interpretation:
- ResComp/GPTAQ-CAE still helps at 1.4B: `50.54 -> 44.03` PPL versus SparseGPT, a `12.9%` relative PPL reduction.
- The local reconstruction pattern repeated from 31M: `gptaq-cae` has slightly worse mean module reconstruction than `sparsegpt`, but better end-to-end PPL. That is more evidence that fp-flow alignment helps in a way local quant-flow MSE does not measure.
- Direct magnitude pruning is still bad, but less catastrophic at 1.4B than 31M: `27.8x` baseline PPL instead of `47.1x`.
- Cost: `gptaq-cae` conversion was about `1.82x` slower than `sparsegpt` on this run (`247s` vs `136s`) because it collects/uses both quant-flow and fp-flow information.

## 2026-05-26 Intra-Layer Attention Head Matching

Goal: extend the earlier cross-model Pythia attention-head JSD matching to also emit intra-layer head matching rows for each model.

Implementation:
- Added `compare_within_model_head_matching`.
- `within_model.jsonl` now mirrors `cross_head_matching.jsonl` style:
  - `match_type=all_pairs` for unordered intra-layer head pairs.
  - `match_type=head_to_best_head` for each head's nearest different head by JSD.
  - `match_type=one_to_one` for minimum-cost no-self assignment within a layer.
- Summary now separates raw pair count from best-match and one-to-one counts while preserving older `within_model.jsonl` files that lack `match_type`.

Smoke:
- Command: `uv run python -m layer_distill.attention_analysis --output-dir runs/attention_pythia31m_70m_intralayer_smoke_20260526 --small-model EleutherAI/pythia-31m --big-model EleutherAI/pythia-70m --prompts 'When Mary handed John the book, he thanked her because the story was exactly what he needed.' --max-length 32 --device auto --dtype fp32`
- Artifact: `runs/attention_pythia31m_70m_intralayer_smoke_20260526`.
- Counts: `within_model_pair_count=336`, `within_head_best_match_count=96`, `within_head_one_to_one_count=96`.
- Self-match check: `0` violations.

Modal run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-GMjKZYPKBEWEJtRwu1WOmS`
- Remote volume path: `/results/pythia_attention_5pair_intralayer_h100_20260526`
- Local copy: `runs/modal_attention_5pair_intralayer_h100_20260526/pythia_attention_5pair_intralayer_h100_20260526`
- Command used the `jthomams477` profile with local Modal token environment variables unset:
  `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_layer_distill.py --mode attention --run-name pythia_attention_5pair_intralayer_h100_20260526 --model-pairs 'EleutherAI/pythia-160m>EleutherAI/pythia-410m,EleutherAI/pythia-410m>EleutherAI/pythia-1b,EleutherAI/pythia-1b>EleutherAI/pythia-1.4b,EleutherAI/pythia-1.4b>EleutherAI/pythia-2.8b,EleutherAI/pythia-2.8b>EleutherAI/pythia-6.9b' --prompts 'In a small gridworld, the agent starts at the red square and must reach the blue square.||Question: If Alice has three keys and gives Bob one key, how many keys does Alice still have? Answer:' --max-length 48 --dtype bf16 --local-window 8`
- The Modal run completed successfully and wrote `sweep_summary.json` plus per-pair `within_model.jsonl` files.
- Per-pair intra-layer counts:
  - `pythia-160m_to_pythia-410m`: `all_pairs=7344`, `head_to_best_head=1056`, `one_to_one=1056`.
  - `pythia-410m_to_pythia-1b`: `all_pairs=6656`, `head_to_best_head=1024`, `one_to_one=1024`.
  - `pythia-1b_to_pythia-1.4b`: `all_pairs=6656`, `head_to_best_head=1024`, `one_to_one=1024`.
  - `pythia-1.4b_to_pythia-2.8b`: `all_pairs=37504`, `head_to_best_head=2816`, `one_to_one=2816`.
  - `pythia-2.8b_to_pythia-6.9b`: `all_pairs=63488`, `head_to_best_head=4096`, `one_to_one=4096`.
- Direct local artifact scan found `0` self-match violations across `head_to_best_head` and `one_to_one` rows.

Layer-level JSD aggregation:
- Generated `analysis/intralayer_one_to_one_jsd_by_pair_model_layer.csv`: one row per pair/model/layer for `match_type=one_to_one`, with `best_jsd`, `avg_jsd`, `worst_jsd`, and the corresponding best/worst head matches.
- Generated `analysis/intralayer_one_to_one_jsd_by_pair_model_layer.md`: Markdown tables for the same min-pairing results.
- Generated `analysis/intralayer_jsd_by_pair_model_layer.csv`: full aggregation for all `match_type` values (`all_pairs`, `head_to_best_head`, `one_to_one`).
- One-to-one aggregation produced `236` layer rows; full aggregation produced `708` rows.

Nearest-neighbor cluster aggregation:
- The one-to-one view was stricter than needed. Generated non-injective clusters by adding an edge from each head to its nearest same-layer head by JSD for each prompt, then taking connected components across the two-prompt nearest-neighbor graph.
- Generated `analysis/intralayer_nearest_neighbor_clusters_by_layer.csv`: one row per pair/model/layer with cluster count, cluster sizes, best/avg/worst within-cluster JSD, and best/worst cluster heads.
- Generated `analysis/intralayer_nearest_neighbor_clusters.csv`: one row per cluster with head membership and JSD stats.
- Generated `analysis/intralayer_nearest_neighbor_clusters_by_layer.md`: Markdown tables for the cluster view.
- The nearest-neighbor graph often collapses a layer into one connected component, so this view measures connected nearest-neighbor structure rather than threshold-separated communities.

## 2026-05-26 ResComp 2:4 Greedy Layer Sweep Pythia-1.4B

Goal: test ResComp/GPTAQ-CAE only, greedily over full transformer layers, evaluating after each selected layer.

Run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-4bqpyf46t45S9UXWSLVKg4`
- Remote volume path: `/results/pythia_1_4b_sparse24_rescomp_greedy_layers_20260526`
- Local copy: `runs/modal_pythia_1_4b_sparse24_rescomp_greedy_layers_20260526/pythia_1_4b_sparse24_rescomp_greedy_layers_20260526`
- Method: `gptaq-cae` only, with `rescomp` normalized to `gptaq-cae` in config.
- Model: `EleutherAI/pythia-1.4b`.
- Calibration: Wikitext-2 train, `32768` tokens captured per linear.
- Eval: Wikitext-2 test, batch size `64`, seq len `256`, `16` batches / `261120` labels.
- Dtype: `bf16`.
- Candidate search: true greedy layer search. At each round, each remaining transformer layer was temporarily sparsified, evaluated, restored, and the best-PPL candidate was permanently applied.
- Candidate count: `300` total layer candidates (`24 + 23 + ... + 1`).
- Layer groups: `24` transformer layers, `96` layer-owned linear modules. `embed_out` was excluded from the layer groups.

Results:

| round | selected layer | cumulative ppl | ratio vs baseline | candidates |
| ---: | ---: | ---: | ---: | ---: |
| 0 | baseline | 20.770710 | 1.0000x |  |
| 1 | 10 | 21.033471 | 1.0127x | 24 |
| 2 | 13 | 21.359867 | 1.0284x | 23 |
| 3 | 5 | 21.713043 | 1.0454x | 22 |
| 4 | 14 | 22.082411 | 1.0632x | 21 |
| 5 | 21 | 22.451140 | 1.0809x | 20 |
| 6 | 22 | 22.862684 | 1.1007x | 19 |
| 7 | 1 | 23.315006 | 1.1225x | 18 |
| 8 | 16 | 23.793806 | 1.1455x | 17 |
| 9 | 12 | 24.283215 | 1.1691x | 16 |
| 10 | 20 | 24.803722 | 1.1942x | 15 |
| 11 | 17 | 25.319732 | 1.2190x | 14 |
| 12 | 23 | 25.890175 | 1.2465x | 13 |
| 13 | 7 | 26.470735 | 1.2744x | 12 |
| 14 | 19 | 27.058039 | 1.3027x | 11 |
| 15 | 9 | 27.712211 | 1.3342x | 10 |
| 16 | 18 | 28.378132 | 1.3663x | 9 |
| 17 | 11 | 29.120403 | 1.4020x | 8 |
| 18 | 6 | 29.977013 | 1.4432x | 7 |
| 19 | 15 | 30.935942 | 1.4894x | 6 |
| 20 | 2 | 32.078205 | 1.5444x | 5 |
| 21 | 8 | 33.490631 | 1.6124x | 4 |
| 22 | 4 | 35.175507 | 1.6935x | 3 |
| 23 | 3 | 37.469095 | 1.8039x | 2 |
| 24 | 0 | 40.805462 | 1.9646x | 1 |

Interpretation:
- The greedy order is strongly non-monotonic in layer index: `10, 13, 5, 14, 21, 22, 1, 16, 12, 20, 17, 23, 7, 19, 9, 18, 11, 6, 15, 2, 8, 4, 3, 0`.
- The first half of the model is not uniformly more fragile, but the very earliest layers are selected last. Layer `0` is worst by the greedy criterion after all other transformer layers have already been converted.
- Full greedy transformer-layer ResComp ended at `40.805` PPL / `1.965x` baseline. The earlier flat all-linear ResComp run ended at `44.035` PPL / `2.120x`, but that flat run included `embed_out` while this greedy layer run excluded it, so the comparison is directionally useful but not exact apples-to-apples.
- The final result is still much better than the earlier flat `sparsegpt` run including `embed_out` (`50.543` PPL / `2.433x`) and far better than magnitude pruning (`578.124` PPL / `27.834x`).

## 2026-05-26 Wanda 2:4 Sparse Comparison Pythia-1.4B

Goal: compare the recorded 2:4 sparsity results against a cheap activation-aware post-training baseline from the 2:4 pruning literature.

Implementation:
- Added `wanda` to the sparse24 harness.
- Score: `abs(W_ij) * rms(X_j)` using captured calibration activations.
- Mask: legal N:M top-k within each contiguous group, no weight compensation.
- Added a focused test where Wanda selects a different legal mask than magnitude pruning.

Run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-PUKMYqc0ILiOvSy9txiJ8W`
- Remote volume path: `/results/pythia_1_4b_sparse24_wanda_compare_20260526`
- Method: `wanda`.
- Model: `EleutherAI/pythia-1.4b`.
- Calibration: Wikitext-2 train, `32768` tokens captured per linear.
- Eval: Wikitext-2 test, batch size `64`, seq len `256`, `16` batches / `261120` labels.
- Dtype: `bf16`.
- Pruned modules: `97` linear modules, exact `0.5` density.

Same-harness comparison:

| method | ppl | ratio vs baseline | loss | density | mean module recon rel mse | sparsify sec | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 20.771 | 1.000x | 3.0335 |  |  |  | dense |
| magnitude | 578.124 | 27.834x | 6.3598 | 0.500 | 0.0847 | 69.0 | flat all-linear run |
| wanda | 439.498 | 21.160x | 6.0856 | 0.500 | 0.0648 | 63.1 | activation-aware mask, no compensation |
| sparsegpt | 50.543 | 2.433x | 3.9228 | 0.500 | 0.0219 | 136.1 | compensation baseline |
| gptaq-cae / ResComp | 44.035 | 2.120x | 3.7850 | 0.500 | 0.0256 | 247.1 | flat all-linear ResComp |
| gptaq-cae / ResComp greedy | 40.805 | 1.965x |  | 0.500 |  |  | transformer layers only; excludes `embed_out` |

Interpretation:
- Wanda improves substantially over magnitude (`578.1 -> 439.5` PPL), but is still not close to compensation methods on Pythia-1.4B 2:4.
- SparseGPT and ResComp are the meaningful comparison class here: compensation is doing almost all of the quality recovery.
- The greedy ResComp result is best numerically, but it is not exact apples-to-apples with the flat all-linear table because it excludes `embed_out`.

## 2026-05-26 Intra-Layer Attention Basis Sweep

Question: can each layer's attention heads be reconstructed from a smaller learned basis than the number of heads in that layer?

Method:
- Per model/layer, collected attention distributions over two prompts (`43` query-token segments, `929` flattened attention features).
- Fit nonnegative NMF bases of size `1,2,4,8,16` where valid (`basis_size < heads`).
- Normalized each basis vector per query-token segment, refit nonnegative linear coefficients per original head, and evaluated segmented JSD/TV against the original head distributions.
- The basis vectors are learned distributions, not necessarily original heads.

Run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-80jXL5QcVIvAwuQgDe5Bab`
- Remote volume path: `/results/pythia_attention_basis_nmf_h100_20260526`
- Local copy: `runs/modal_attention_basis_nmf_h100_20260526/pythia_attention_basis_nmf_h100_20260526`
- Command used the `jthomams477` profile with local Modal token environment variables unset:
  `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_layer_distill.py --mode attention-basis --run-name pythia_attention_basis_nmf_h100_20260526 --model-pairs 'EleutherAI/pythia-160m>EleutherAI/pythia-410m,EleutherAI/pythia-410m>EleutherAI/pythia-1b,EleutherAI/pythia-1b>EleutherAI/pythia-1.4b,EleutherAI/pythia-1.4b>EleutherAI/pythia-2.8b,EleutherAI/pythia-2.8b>EleutherAI/pythia-6.9b' --prompts 'In a small gridworld, the agent starts at the red square and must reach the blue square.||Question: If Alice has three keys and gives Bob one key, how many keys does Alice still have? Answer:' --max-length 48 --dtype bf16 --basis-sizes 1,2,4,8,16 --steps 300 --fit-lr 0.2`

Artifacts:
- `analysis/attention_basis_by_model_layer_basis.csv`: one row per model/layer/basis size.
- `analysis/attention_basis_by_model_basis.csv`: model-level aggregation by basis size.
- `analysis/attention_basis_best_basis_by_layer.csv`: smallest basis meeting layer thresholds (`mean_jsd <= 0.02/0.05/0.10`) plus best available basis.
- `analysis/attention_basis_thresholds_by_model.csv`: model-level threshold summary.
- `analysis/attention_basis_summary.md`: Markdown summary tables.

Results at largest tested valid basis:

| model | heads | largest basis | avg layer mean JSD | layers with mean JSD <= 0.05 |
| --- | ---: | ---: | ---: | ---: |
| pythia-160m | 12 | 8 | 0.025433 | 11/12 |
| pythia-410m | 16 | 8 | 0.020973 | 23/24 |
| pythia-1b | 8 | 4 | 0.020964 | 16/16 |
| pythia-1.4b | 16 | 8 | 0.025510 | 20/24 |
| pythia-2.8b | 32 | 16 | 0.024545 | 26/32 |
| pythia-6.9b | 32 | 16 | 0.033861 | 22/32 |

Interpretation:
- Yes, on average a strictly smaller basis works: all tested models reach average layer mean JSD under `0.05` with basis size at most half the head count, except pythia-160m which needs `8/12` heads for the same average threshold.
- A strict all-layer `mean_jsd <= 0.05` criterion is not always met. The failures concentrate in late layers: pythia-1.4b layers `20-23`, pythia-2.8b layers `24-29`, and pythia-6.9b mostly layers `20-30`.
- No model reached average layer mean JSD under `0.02` at the tested basis sizes, though many individual layers did.

## 2026-05-26 Attention Basis Perplexity Check

Question: if attention heads can be reconstructed from a smaller basis distributionally, does replacing runtime attention with that smaller basis preserve language-model perplexity?

Important caveat:
- The previous NMF basis artifacts are tied to the two short prompts and their token positions, so they are not directly reusable for Wikitext sequences.
- This PPL check uses the operational analogue: for each forward pass and layer, compress the current attention tensor across heads into a nonnegative rank-`k` basis, reconstruct every head's attention distribution from that basis, renormalize over the causal support, and continue the model.

Implementation:
- Added `layer_distill/attention_basis_ppl.py`.
- Added Modal mode `attention-basis-ppl`.
- Added focused tests in `tests/test_attention_basis_ppl.py`.
- Full test suite: `92 passed, 1 skipped`.

Run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-wAYM9O3YuSnHWJ0yGs1sYf`
- Remote volume path: `/results/pythia_1_4b_attention_basis_ppl_initfix_20260526`
- Local copy: `runs/modal_pythia_1_4b_attention_basis_ppl_initfix_20260526/pythia_1_4b_attention_basis_ppl_initfix_20260526`
- Model: `EleutherAI/pythia-1.4b`
- Eval: Wikitext-2 test, `8` steps, batch size `4`, seq len `128`, `4064` labels.
- Dtype: `bf16`.
- Dynamic NMF iterations per layer/batch: `6`.

Results:

| run | basis | ppl | ratio vs baseline | loss |
| --- | ---: | ---: | ---: | ---: |
| baseline |  | 30.676112 | 1.000000 | 3.423484 |
| basis_2 | 2 | 301.645844 | 9.833249 | 5.709254 |
| basis_4 | 4 | 178.324264 | 5.813131 | 5.183604 |
| basis_8 | 8 | 69.394496 | 2.262167 | 4.239808 |

Interpretation:
- Even though basis `4` and `8` gave low attention-distribution JSD on short prompt captures, replacing live attention probabilities with dynamic NMF reconstructions causes large PPL degradation.
- Basis `8` is directionally much better than basis `4` and `2`, but still more than doubles perplexity on this short eval.
- The result suggests the attention distributions are low-dimensional in a reconstruction metric but the model is highly sensitive to reconstruction error when the approximation is placed in the forward path.

## 2026-05-26 Attention Basis Layer Subset Search

Question: since compressing attention heads in every layer hurts PPL badly, can we search for a small subset of transformer layers where dynamic attention-basis compression is tolerable?

Implementation:
- Added selected-layer patching to the dynamic attention-basis PPL path.
- Added a layer-group evaluator that evaluates baseline plus arbitrary layer subsets.
- Added a parallel tree/beam search over layer subsets. This is not a pure stochastic rollout MCTS; the objective is expensive and deterministic enough that expanding the best frontier directly gives a better subset-by-cardinality curve for the same H100 budget.
- Added Modal modes `attention-basis-layer-search` and `attention-basis-layer-groups`.

Tests:
- Focused suite: `uv run pytest -q tests/test_attention_basis_ppl.py` -> `7 passed`.
- Compile/import checks passed for `layer_distill/attention_basis_ppl.py` and `modal_layer_distill.py`.
- Full local suite: `uv run pytest -q` -> `98 passed, 1 skipped`.

Parallel search run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-harJfxNRjkaQBgNuHzNpnC`
- Remote volume path: `/results/pythia_1_4b_attention_basis_layer_search_b8_d4_20260526`
- Local copy: `runs/modal_pythia_1_4b_attention_basis_layer_search_b8_d4_20260526/pythia_1_4b_attention_basis_layer_search_b8_d4_20260526`
- Model: `EleutherAI/pythia-1.4b`
- Eval: Wikitext-2 test, `4` steps, batch size `4`, seq len `128`, `2032` labels.
- Dtype: `bf16`.
- Dynamic NMF: basis size `8`, `6` iterations.
- Search: depth `4`, beam width `4`, 24 candidate layers, 4 H100 worker calls per round.
- Baseline PPL on the 4-step search harness: `27.626584`.

Best result per depth:

| depth | best layers | ppl | ratio vs baseline | loss |
| ---: | --- | ---: | ---: | ---: |
| 1 | 4 | 27.608978 | 0.999363 | 3.318141 |
| 2 | 16,19 | 27.603963 | 0.999181 | 3.317959 |
| 3 | 16,19,23 | 27.668682 | 1.001524 | 3.320301 |
| 4 | 4,13,15,23 | 27.776657 | 1.005432 | 3.324196 |

Longer verification run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-bBKdjQAQWc7wr7v9Ipoh6f`
- Remote volume path: `/results/pythia_1_4b_attention_basis_layer_groups_b8_verify_20260526`
- Local copy: `runs/modal_pythia_1_4b_attention_basis_layer_groups_b8_verify_20260526/pythia_1_4b_attention_basis_layer_groups_b8_verify_20260526`
- Eval: same Wikitext-2 test harness as the all-layer PPL check, `8` steps, batch size `4`, seq len `128`, `4064` labels.

Verification results:

| layers | compressed layers | ppl | ratio vs baseline | loss |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0 | 30.676112 | 1.000000 | 3.423484 |
| 4 | 1 | 30.350164 | 0.989375 | 3.412802 |
| 16,19 | 2 | 30.810665 | 1.004386 | 3.427861 |
| 16,19,23 | 3 | 30.854104 | 1.005802 | 3.429270 |
| 4,13,15,23 | 4 | 30.620715 | 0.998194 | 3.421677 |

Interpretation:
- Layer selection changes the conclusion completely. Compressing all 24 layers with basis `8` gave `69.394496` PPL / `2.262167x` baseline, but selected subsets up to four layers stayed near baseline on the same 8-step harness.
- The best verified 4-layer subset, layers `4,13,15,23`, landed at `30.620715` PPL, slightly below the measured baseline on this short sample (`0.998194x`). Treat that as effectively baseline, not as a real improvement claim.
- The single-layer result for layer `4` also measured slightly below baseline (`0.989375x`), but the eval is still too short to call it beneficial.
- The practical result is that attention-basis compression should be scheduled by layer. Applying it everywhere is not viable, while a searched layer subset can be close to PPL-neutral.

Deeper search follow-up:
- Reason for the first depth-4 cap: it was a conservative first pass after the all-layer failure. After the depth-4 set verified close to baseline, that cap was too conservative.
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-mmvLWwgKBa9zHVtK7NAEwC`
- Remote volume path: `/results/pythia_1_4b_attention_basis_layer_search_b8_d8_bw8_20260526`
- Local copy: `runs/modal_pythia_1_4b_attention_basis_layer_search_b8_d8_bw8_20260526/pythia_1_4b_attention_basis_layer_search_b8_d8_bw8_20260526`
- Search: depth `8`, beam width `8`, 8 parallel H100 workers, same 4-step Wikitext-2 search objective.

Best 4-step search result per depth:

| depth | best layers | ppl | ratio vs baseline | loss |
| ---: | --- | ---: | ---: | ---: |
| 1 | 4 | 27.608978 | 0.999363 | 3.318141 |
| 2 | 16,19 | 27.603963 | 0.999181 | 3.317959 |
| 3 | 16,19,23 | 27.668682 | 1.001524 | 3.320301 |
| 4 | 3,13,15,23 | 27.752125 | 1.004544 | 3.323312 |
| 5 | 4,13,15,16,23 | 27.890498 | 1.009553 | 3.328286 |
| 6 | 4,14,15,16,22,23 | 27.999653 | 1.013504 | 3.332192 |
| 7 | 4,13,15,16,17,22,23 | 28.316640 | 1.024978 | 3.343450 |
| 8 | 4,13,15,16,17,20,22,23 | 28.533569 | 1.032830 | 3.351081 |

Longer verification of deeper path:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-Do4PG61sHcFjdIkPsPshnF`
- Remote volume path: `/results/pythia_1_4b_attention_basis_layer_groups_b8_d8_verify_20260526`
- Local copy: `runs/modal_pythia_1_4b_attention_basis_layer_groups_b8_d8_verify_20260526/pythia_1_4b_attention_basis_layer_groups_b8_d8_verify_20260526`
- Eval: Wikitext-2 test, `8` steps, batch size `4`, seq len `128`, `4064` labels.

Verification results:

| layers | compressed layers | ppl | ratio vs baseline | loss |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0 | 30.676112 | 1.000000 | 3.423484 |
| 3,13,15,23 | 4 | 31.089536 | 1.013477 | 3.436871 |
| 4,13,15,16,23 | 5 | 30.901428 | 1.007345 | 3.430802 |
| 4,14,15,16,22,23 | 6 | 31.152411 | 1.015527 | 3.438892 |
| 4,13,15,16,17,22,23 | 7 | 31.189107 | 1.016723 | 3.440069 |
| 4,13,15,16,17,20,22,23 | 8 | 31.555719 | 1.028674 | 3.451755 |

Updated interpretation:
- Deeper search does continue to find usable subsets, but the penalty starts accumulating after about four or five compressed layers.
- The best verified depth-5 set is still modest (`+0.734%` PPL), while depth-8 reaches `+2.867%` PPL on the longer 8-step harness.
- The earlier verified 4-layer set `4,13,15,23` remains the strongest near-neutral operating point from these runs, even though the deeper 4-step search found a slightly different depth-4 set that did not verify as well.

## 2026-05-26 Exponential Attention Basis Combination

Question: try the same layer-subset search, but reconstruct each head with an exponential/log-linear combination of basis attention distributions instead of the original arithmetic combination.

Implementation:
- Added `combine_mode={linear,exponential}` to the dynamic attention-basis PPL evaluator.
- Exponential mode uses the same dynamic per-batch NMF basis construction, normalizes each basis vector per query row, then fits each head with a log-linear/geometric mixture:
  `p(k) = softmax(sum_i w_i log b_i(k))`.
- Mixture weights are fit with a few manual gradient steps per batch/layer, avoiding autograd graph construction inside the model forward.
- Added Modal CLI parameter `--basis-combine-mode exponential` for `attention-basis-ppl`, `attention-basis-layer-search`, and `attention-basis-layer-groups`.

Tests:
- Focused suite: `uv run pytest -q tests/test_attention_basis_ppl.py` -> `9 passed`.
- Compile/import checks passed for `layer_distill/attention_basis_ppl.py` and `modal_layer_distill.py`.
- Full local suite: `uv run pytest -q` -> `100 passed, 1 skipped`.

Smoke:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-jA1Rbk971XoRvKBt6pGzxp`
- Model: `EleutherAI/pythia-31m`, basis size `2`, depth `1`, eval steps `1`.
- Purpose: verified remote `combine_mode=exponential` plumbing and forward pass.

Depth-8 search:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-dVvUvcLnHjbd1QV0VJMJnV`
- Remote volume path: `/results/pythia_1_4b_attention_basis_layer_search_b8_exp_d8_bw8_20260526`
- Local copy: `runs/modal_pythia_1_4b_attention_basis_layer_search_b8_exp_d8_bw8_20260526/pythia_1_4b_attention_basis_layer_search_b8_exp_d8_bw8_20260526`
- Model: `EleutherAI/pythia-1.4b`
- Eval: Wikitext-2 test, `4` steps, batch size `4`, seq len `128`, `2032` labels.
- Dynamic NMF: basis size `8`, `6` iterations, exponential combination.
- Search: depth `8`, beam width `8`, 8 parallel H100 workers.
- Baseline PPL on the 4-step search harness: `27.626584`.

Best 4-step search result per depth:

| depth | best layers | ppl | ratio vs baseline | loss |
| ---: | --- | ---: | ---: | ---: |
| 1 | 22 | 27.637879 | 1.000409 | 3.319187 |
| 2 | 20,22 | 27.640625 | 1.000508 | 3.319287 |
| 3 | 4,15,16 | 27.697833 | 1.002579 | 3.321354 |
| 4 | 4,15,16,23 | 27.735853 | 1.003955 | 3.322726 |
| 5 | 4,15,16,22,23 | 27.781116 | 1.005594 | 3.324356 |
| 6 | 4,15,16,21,22,23 | 27.987074 | 1.013049 | 3.331743 |
| 7 | 4,15,16,20,21,22,23 | 28.296911 | 1.024264 | 3.342753 |
| 8 | 4,13,15,16,19,21,22,23 | 28.617666 | 1.035874 | 3.354024 |

Longer verification:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-E8G34RATYhn0DiosMZ689V`
- Remote volume path: `/results/pythia_1_4b_attention_basis_layer_groups_b8_exp_d8_verify_20260526`
- Local copy: `runs/modal_pythia_1_4b_attention_basis_layer_groups_b8_exp_d8_verify_20260526/pythia_1_4b_attention_basis_layer_groups_b8_exp_d8_verify_20260526`
- Eval: Wikitext-2 test, `8` steps, batch size `4`, seq len `128`, `4064` labels.

Verification results:

| layers | compressed layers | ppl | ratio vs baseline | loss |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0 | 30.676112 | 1.000000 | 3.423484 |
| 22 | 1 | 30.761597 | 1.002787 | 3.426267 |
| 20,22 | 2 | 30.801894 | 1.004100 | 3.427576 |
| 4,15,16 | 3 | 30.865397 | 1.006170 | 3.429636 |
| 4,15,16,23 | 4 | 30.890191 | 1.006979 | 3.430439 |
| 4,15,16,22,23 | 5 | 30.897283 | 1.007210 | 3.430668 |
| 4,15,16,21,22,23 | 6 | 31.144486 | 1.015268 | 3.438637 |
| 4,15,16,20,21,22,23 | 7 | 31.386731 | 1.023165 | 3.446385 |
| 4,13,15,16,19,21,22,23 | 8 | 31.596865 | 1.030015 | 3.453058 |

Interpretation:
- Exponential combination changes the selected layers but does not improve the best verified operating point over the earlier linear-combination search.
- It is competitive around depth `4-5`: verified depth `5` is `+0.721%` PPL, close to the linear depth-5 verification at `+0.734%`.
- At depth `8`, exponential verified at `+3.002%` PPL, slightly worse than the linear depth-8 verification at `+2.867%`.
- The best result across both combination rules remains the earlier linear 4-layer set `4,13,15,23`, which verified at `0.998194x` baseline on this short 8-step harness.

## 2026-05-26 ResComp 1:2 Sparsity Greedy Layer Run

Question: rerun the ResComp layer-greedy sparsity experiment on `EleutherAI/pythia-1.4b`, but with `1:2` structured sparsity instead of the earlier `2:4` sparsity. User requested Modal only.

Run:
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-GeBHScRSICKigkSArp0X9H`
- Account/profile: `jthomams477`
- Run name: `pythia_1_4b_sparse12_rescomp_greedy_layers_20260526`
- Model: `EleutherAI/pythia-1.4b`
- Method: ResComp / `gptaq-cae`
- Sparsity: `1:2`, exact density `0.5`
- Eval: Wikitext-2 test, `16` steps, batch size `64`, seq len `256`, `261120` tokens
- Calibration: Wikitext-2 train, `32768` tokens
- Status: interrupted intentionally after round 11 completed and round 12 had started; no all-layer final result.

Baseline:
- PPL: `20.770710`
- Loss: `3.033544`

Completed greedy selections:

| greedy round | selected layer | selected layers so far | ppl | loss |
| ---: | ---: | --- | ---: | ---: |
| 1 | 10 | 10 | 21.157991 | - |
| 2 | 14 | 10,14 | 21.648773 | - |
| 3 | 5 | 10,14,5 | 22.197654 | - |
| 4 | 22 | 10,14,5,22 | 22.730864 | - |
| 5 | 13 | 10,14,5,22,13 | 23.371540 | - |
| 6 | 21 | 10,14,5,22,13,21 | 24.026612 | - |
| 7 | 9 | 10,14,5,22,13,21,9 | 24.741622 | - |
| 8 | 16 | 10,14,5,22,13,21,9,16 | 25.487163 | - |
| 9 | 20 | 10,14,5,22,13,21,9,16,20 | 26.280889 | - |
| 10 | 12 | 10,14,5,22,13,21,9,16,20,12 | 27.186795 | 3.302731 |
| 11 | 19 | 10,14,5,22,13,21,9,16,20,12,19 | 28.057774 | 3.334266 |

Partial round 12:
- The run was stopped while evaluating round 12 candidates.
- Visible completed round-12 candidates before stop included layers `0,1,2,3,4,6`, with best visible PPL `29.051025` for layer `1`.
- No round-12 layer should be treated as selected.

Interpretation:
- `1:2` and `2:4` have the same nominal density, but the greedy order changed substantially.
- Through 11 selected layers, `1:2` reached PPL `28.057774` versus baseline `20.770710`.
- Earlier `2:4` result was already recorded: final all-layer greedy PPL `40.805462`, ratio `1.9646x`, order `10,13,5,14,21,22,1,16,12,20,17,23,7,19,9,18,11,6,15,2,8,4,3,0`, Modal URL `https://modal.com/apps/jthomams477/main/ap-4bqpyf46t45S9UXWSLVKg4`.

## 2026-05-26 Smaller Linear Attention Bases

Question: rerun the same linear-combination layer-subset/MCTS-style search, but with smaller per-layer attention bases than the previous basis size `8`.

Setup:
- Model: `EleutherAI/pythia-1.4b`
- Method: dynamic attention basis with linear head reconstruction.
- Basis sizes: `4` and `2`.
- NMF iterations: `6`.
- Search eval: Wikitext-2 test, `4` steps, batch size `4`, seq len `128`, `2032` labels.
- Search: depth `8`, beam width `8`, 8 parallel H100 workers, all 24 layers.
- Verification eval: Wikitext-2 test, `8` steps, batch size `4`, seq len `128`, `4064` labels.

Runs:
- Basis `4` search: `pythia_1_4b_attention_basis_layer_search_b4_linear_d8_bw8_20260526`, Modal URL `https://modal.com/apps/jthomams477/main/ap-3Iygg3NSXLHtL2CjtHPWm2`
- Basis `4` search local copy: `runs/modal_pythia_1_4b_attention_basis_layer_search_b4_linear_d8_bw8_20260526/pythia_1_4b_attention_basis_layer_search_b4_linear_d8_bw8_20260526`
- Basis `4` verification: `pythia_1_4b_attention_basis_layer_groups_b4_linear_d8_verify_20260526`, Modal URL `https://modal.com/apps/jthomams477/main/ap-j4QSmVgbxDAEtpOOibd4gY`
- Basis `4` verification local copy: `runs/modal_pythia_1_4b_attention_basis_layer_groups_b4_linear_d8_verify_20260526/pythia_1_4b_attention_basis_layer_groups_b4_linear_d8_verify_20260526`
- Basis `2` search: `pythia_1_4b_attention_basis_layer_search_b2_linear_d8_bw8_20260526`, Modal URL `https://modal.com/apps/jthomams477/main/ap-SRbosRx0QwiQEi5WIuxjQC`
- Basis `2` search local copy: `runs/modal_pythia_1_4b_attention_basis_layer_search_b2_linear_d8_bw8_20260526/pythia_1_4b_attention_basis_layer_search_b2_linear_d8_bw8_20260526`
- Basis `2` verification: `pythia_1_4b_attention_basis_layer_groups_b2_linear_d8_verify_20260526`, Modal URL `https://modal.com/apps/jthomams477/main/ap-SMI5X6Dkxb3ZuPT7zFdaOn`
- Basis `2` verification local copy: `runs/modal_pythia_1_4b_attention_basis_layer_groups_b2_linear_d8_verify_20260526/pythia_1_4b_attention_basis_layer_groups_b2_linear_d8_verify_20260526`

Best 4-step search path, basis `4`:

| depth | best layers | ppl | ratio vs baseline | loss |
| ---: | --- | ---: | ---: | ---: |
| 1 | 20 | 27.755231 | 1.004657 | 3.323424 |
| 2 | 20,23 | 27.926217 | 1.010846 | 3.329566 |
| 3 | 19,20,23 | 28.203284 | 1.020875 | 3.339438 |
| 4 | 19,20,22,23 | 28.608353 | 1.035537 | 3.353699 |
| 5 | 16,19,20,22,23 | 28.967252 | 1.048528 | 3.366166 |
| 6 | 5,19,20,21,22,23 | 29.744399 | 1.076659 | 3.392641 |
| 7 | 13,16,19,20,21,22,23 | 30.295584 | 1.096610 | 3.411002 |
| 8 | 4,13,16,19,20,21,22,23 | 31.207796 | 1.129629 | 3.440668 |

Best 4-step search path, basis `2`:

| depth | best layers | ppl | ratio vs baseline | loss |
| ---: | --- | ---: | ---: | ---: |
| 1 | 20 | 27.781047 | 1.005591 | 3.324354 |
| 2 | 20,23 | 27.998802 | 1.013473 | 3.332162 |
| 3 | 20,22,23 | 28.355982 | 1.026402 | 3.344838 |
| 4 | 19,20,22,23 | 29.072574 | 1.052341 | 3.369795 |
| 5 | 13,19,20,22,23 | 29.666093 | 1.073824 | 3.390005 |
| 6 | 13,19,20,21,22,23 | 30.471244 | 1.102968 | 3.416783 |
| 7 | 4,13,19,20,21,22,23 | 31.349586 | 1.134762 | 3.445201 |
| 8 | 4,13,17,19,20,21,22,23 | 32.418381 | 1.173449 | 3.478726 |

Longer verification, basis `4`:

| layers | compressed layers | ppl | ratio vs baseline | loss |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0 | 30.676112 | 1.000000 | 3.423484 |
| 20 | 1 | 30.882643 | 1.006733 | 3.430194 |
| 20,23 | 2 | 31.011264 | 1.010926 | 3.434351 |
| 19,20,23 | 3 | 31.034908 | 1.011696 | 3.435113 |
| 19,20,22,23 | 4 | 31.399748 | 1.023590 | 3.446800 |
| 16,19,20,22,23 | 5 | 31.797944 | 1.036570 | 3.459402 |
| 5,19,20,21,22,23 | 6 | 32.949405 | 1.074106 | 3.494973 |
| 13,16,19,20,21,22,23 | 7 | 33.177719 | 1.081549 | 3.501879 |
| 4,13,16,19,20,21,22,23 | 8 | 33.922885 | 1.105840 | 3.524090 |

Longer verification, basis `2`:

| layers | compressed layers | ppl | ratio vs baseline | loss |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0 | 30.676112 | 1.000000 | 3.423484 |
| 20 | 1 | 30.816261 | 1.004569 | 3.428042 |
| 20,23 | 2 | 30.991139 | 1.010269 | 3.433701 |
| 20,22,23 | 3 | 31.414922 | 1.024084 | 3.447283 |
| 19,20,22,23 | 4 | 31.933039 | 1.040974 | 3.463641 |
| 13,19,20,22,23 | 5 | 32.703778 | 1.066099 | 3.487491 |
| 13,19,20,21,22,23 | 6 | 33.698552 | 1.098528 | 3.517455 |
| 4,13,19,20,21,22,23 | 7 | 34.236165 | 1.116053 | 3.533283 |
| 4,13,17,19,20,21,22,23 | 8 | 35.265512 | 1.149608 | 3.562905 |

Comparison to basis `8` linear:

| basis size | best verified 1-layer path | best verified 4-layer path | best verified 8-layer path |
| ---: | ---: | ---: | ---: |
| 8 | 1.001450 | 0.998194 | 1.028674 |
| 4 | 1.006733 | 1.023590 | 1.105840 |
| 2 | 1.004569 | 1.040974 | 1.149608 |

Interpretation:
- Smaller bases do not preserve attention well enough once many layers are compressed.
- Basis `4` is tolerable for shallow layer groups: depth `2` verifies at `+1.093%`, depth `3` at `+1.170%`, and depth `4` at `+2.359%`.
- Basis `2` is slightly better than basis `4` for a single layer in this short sample, but it falls behind by depth `3` and is clearly worse by depth `8`.
- The earlier basis `8` linear run remains the strongest option: its verified 4-layer set was effectively baseline (`0.998194x`), and even the 8-layer set was only `+2.867%`, compared with `+10.584%` for basis `4` and `+14.961%` for basis `2`.

## 2026-05-26 Quantized Linear Attention Basis

Question: test the same basis-8 linear 8-layer replacement at different quantization levels.

Implementation:
- Added `basis_quantization_bits` to attention-basis PPL configs, Modal layer-group evals, and the CLI.
- `0` means no quantization.
- Positive values quantize the reconstructed attention probability rows after the linear combination, using row-wise uniform bins over `[0, row_max]`; rows are then renormalized and causal mask zeros are preserved.
- This tests quantized reconstructed attention distributions, not offline-quantized model weights.

Tests:
- Focused suite: `uv run pytest -q tests/test_attention_basis_ppl.py` -> `10 passed`.
- Compile check: `python3 -m py_compile layer_distill/attention_basis_ppl.py modal_layer_distill.py` -> passed.
- Full local suite: `uv run pytest -q` -> `105 passed, 1 skipped`.

Setup:
- Model: `EleutherAI/pythia-1.4b`
- Layers: `4,13,15,16,17,20,22,23`
- Basis size: `8`
- NMF iterations: `6`
- Combine mode: `linear`
- Eval: Wikitext-2 test, `8` steps, batch size `4`, seq len `128`, `4064` labels.
- Baseline PPL in each run: `30.676112`

Runs:

| quant bits | Modal URL | local copy |
| ---: | --- | --- |
| 0 | `https://modal.com/apps/jthomams477/main/ap-YKv4F5zc3RStKUp4hrMxvK` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_q0_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_q0_20260526` |
| 8 | `https://modal.com/apps/jthomams477/main/ap-F3KIk74MvYGbpB4aqFcGsP` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_q8_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_q8_20260526` |
| 6 | `https://modal.com/apps/jthomams477/main/ap-q2UfQqlYrV2KTFEMtcVTfI` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_q6_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_q6_20260526` |
| 4 | `https://modal.com/apps/jthomams477/main/ap-oBUGGIqweTc127LcAuoMcJ` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_q4_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_q4_20260526` |
| 3 | `https://modal.com/apps/jthomams477/main/ap-0Qq35uwFRuurSkRhKhK11X` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_q3_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_q3_20260526` |
| 2 | `https://modal.com/apps/jthomams477/main/ap-AEIt6oICf1KnQX0D9e7cvW` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_q2_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_q2_20260526` |
| 1 | `https://modal.com/apps/jthomams477/main/ap-bRuyjCw9t4jXWnPW72Tcl5` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_q1_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_q1_20260526` |

Results:

| quant bits | ppl | ratio vs baseline | delta vs baseline | loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 31.483839 | 1.026331 | +2.633% | 3.449474 |
| 8 | 31.810978 | 1.036995 | +3.700% | 3.459811 |
| 6 | 31.795663 | 1.036496 | +3.650% | 3.459330 |
| 4 | 31.928630 | 1.040830 | +4.083% | 3.463503 |
| 3 | 31.986526 | 1.042718 | +4.272% | 3.465315 |
| 2 | 32.853665 | 1.070985 | +7.099% | 3.492063 |
| 1 | 34.149136 | 1.113216 | +11.322% | 3.530737 |

Interpretation:
- Mild quantization is not the main failure mode. Going from unquantized to 8/6 bits adds about `+1.0%` relative PPL on this short harness; 4/3 bits add about `+1.45-1.64%`.
- The cliff starts at 2 bits: 2-bit reconstructed attention reaches `+7.099%`, and 1-bit reaches `+11.322%`.
- For this layer set, 3-4 bit row-wise attention-probability quantization looks usable if the unquantized basis-8 replacement is already acceptable; 2-bit is probably too aggressive.

### Named Quantization Formats

Follow-up question: compare popular named formats: INT8, INT4, NVFP4, and adjacent common formats.

Implementation:
- Added `basis_quantization_format` alongside `basis_quantization_bits`; only one may be set.
- Supported formats: `int8`, `int4`, `fp8_e4m3`, `fp8_e5m2`, `nvfp4`, `mxfp4`, `nf4`.
- `int8` and `int4` use the same row-wise unsigned uniform attention-probability quantization as the earlier 8-bit and 4-bit runs.
- `fp8_e4m3` and `fp8_e5m2` use row-scaled floating-point codebooks.
- `nvfp4` uses E2M1 values with contiguous 16-value blocks and per-block scale, matching the key numerical shape described in NVIDIA's NVFP4 docs.
- `mxfp4` uses E2M1 values with contiguous 32-value blocks and power-of-two block scale.
- `nf4` uses a positive-side NF4/QLoRA-style codebook with 64-value blocks.
- These are dequantized numerical simulations for reconstructed attention probabilities, not hardware kernel or packed-storage benchmarks.

Source notes:
- NVIDIA describes NVFP4 as E2M1 values with 16-value blocks and E4M3 block scales, plus a second tensor-level FP32 scale.
- NVIDIA Transformer Engine docs describe NVFP4 E2M1 and block-scale layout details.

Tests:
- Focused suite: `uv run pytest -q tests/test_attention_basis_ppl.py` -> `11 passed`.
- Compile check: `python3 -m py_compile layer_distill/attention_basis_ppl.py modal_layer_distill.py` -> passed.
- Full local suite: `uv run pytest -q` -> `106 passed, 1 skipped`.

Setup:
- Model: `EleutherAI/pythia-1.4b`
- Layers: `4,13,15,16,17,20,22,23`
- Basis size: `8`
- NMF iterations: `6`
- Combine mode: `linear`
- Eval: Wikitext-2 test, `8` steps, batch size `4`, seq len `128`, `4064` labels.
- Baseline PPL in each run: `30.676112`

Runs:

| format | Modal URL | local copy |
| --- | --- | --- |
| int8 | `https://modal.com/apps/jthomams477/main/ap-ovWJE0aaMDFpMYiLu044dv` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_fmt_int8_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_fmt_int8_20260526` |
| int4 | `https://modal.com/apps/jthomams477/main/ap-TPoMIHcxGzxz8ND3ayeChO` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_fmt_int4_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_fmt_int4_20260526` |
| fp8_e4m3 | `https://modal.com/apps/jthomams477/main/ap-gYUtF41jon5dYbqsKlH2XQ` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_fmt_fp8_e4m3_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_fmt_fp8_e4m3_20260526` |
| fp8_e5m2 | `https://modal.com/apps/jthomams477/main/ap-wtYFaEizighogLVu6uupzu` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_fmt_fp8_e5m2_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_fmt_fp8_e5m2_20260526` |
| nvfp4 | `https://modal.com/apps/jthomams477/main/ap-UDx6oL5tAFoBnbkzgMv71w` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_fmt_nvfp4_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_fmt_nvfp4_20260526` |
| mxfp4 | `https://modal.com/apps/jthomams477/main/ap-Mi4NxJbTjhq6wTUGom6dGv` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_fmt_mxfp4_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_fmt_mxfp4_20260526` |
| nf4 | `https://modal.com/apps/jthomams477/main/ap-GD8LfFu67S6wCLWYXWsOeF` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_fmt_nf4_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_fmt_nf4_20260526` |

Results:

| format | ppl | ratio vs baseline | delta vs baseline | loss |
| --- | ---: | ---: | ---: | ---: |
| int8 | 31.810978 | 1.036995 | +3.700% | 3.459811 |
| int4 | 31.928630 | 1.040830 | +4.083% | 3.463503 |
| fp8_e4m3 | 31.706735 | 1.033597 | +3.360% | 3.456529 |
| fp8_e5m2 | 31.829669 | 1.037604 | +3.760% | 3.460399 |
| nvfp4 | 31.755414 | 1.035184 | +3.518% | 3.458063 |
| mxfp4 | 31.588538 | 1.029744 | +2.974% | 3.452794 |
| nf4 | 31.537312 | 1.028074 | +2.807% | 3.451171 |

Interpretation:
- On this short harness, all named formats stay near the unquantized basis-8 8-layer result (`31.483839`, `+2.633%`).
- The best named format here is `nf4` (`+2.807%`), then `mxfp4` (`+2.974%`), then `fp8_e4m3` (`+3.360%`), then `nvfp4` (`+3.518%`).
- `int8` and `int4` reproduce the earlier 8-bit and 4-bit uniform-row results exactly, as expected.
- The named floating/codebook formats mostly beat uniform `int4`; `fp8_e5m2` is close to uniform `int8` in this setup, likely because row scaling makes E5M2's wide dynamic range less useful than E4M3's denser mantissa.

### Factor Quantization Formats

Correction: the previous named-format section quantized the reconstructed attention probability rows. The requested surface is the attention-basis representation itself: the learned nonnegative `basis` rows and the per-head `coeffs` weights. This section quantizes those factors before reconstruction, then applies the usual causal mask and row renormalization.

Implementation:
- Added `basis_quantization_target`, with values `reconstructed` and `factors`.
- `factors` quantizes `basis` and `coeffs` after NMF iterations and before `coeffs @ basis`.
- Modal summaries now record `basis_quantization_target`.
- This remains a dequantized numerical simulation of factor formats, not a packed-kernel or storage-throughput benchmark.

Tests:
- Focused suite: `uv run pytest -q tests/test_attention_basis_ppl.py` -> `12 passed`.
- Compile check: `python3 -m py_compile layer_distill/attention_basis_ppl.py modal_layer_distill.py` -> passed.
- Full local suite: `uv run pytest -q` -> `108 passed, 1 skipped`.

Setup:
- Model: `EleutherAI/pythia-1.4b`
- Layers: `4,13,15,16,17,20,22,23`
- Basis size: `8`
- NMF iterations: `6`
- Combine mode: `linear`
- Quantization target: `factors`
- Eval: Wikitext-2 test, `8` steps, batch size `4`, seq len `128`, `4064` labels.
- Baseline PPL in each run: `30.676112`

Runs:

| format | Modal URL | local copy |
| --- | --- | --- |
| int8 | `https://modal.com/apps/jthomams477/main/ap-XPF6Zz5lDN88vMahSFGT80` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_factor_fmt_int8_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_factor_fmt_int8_20260526` |
| int4 | `https://modal.com/apps/jthomams477/main/ap-Pu9namEYRQZFANZrvNS6Pz` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_factor_fmt_int4_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_factor_fmt_int4_20260526` |
| fp8_e4m3 | `https://modal.com/apps/jthomams477/main/ap-NgV1zZZjQj3N5L0gySdUBX` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_factor_fmt_fp8_e4m3_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_factor_fmt_fp8_e4m3_20260526` |
| fp8_e5m2 | `https://modal.com/apps/jthomams477/main/ap-2qCy8GQDZiNKjkodoLBp2l` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_factor_fmt_fp8_e5m2_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_factor_fmt_fp8_e5m2_20260526` |
| nvfp4 | `https://modal.com/apps/jthomams477/main/ap-6qQQTxQXiXhMIaEedwWyxm` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_factor_fmt_nvfp4_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_factor_fmt_nvfp4_20260526` |
| mxfp4 | `https://modal.com/apps/jthomams477/main/ap-13tklBaln0mvVq2Yt9EUv1` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_factor_fmt_mxfp4_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_factor_fmt_mxfp4_20260526` |
| nf4 | `https://modal.com/apps/jthomams477/main/ap-hXo5P2LuxQI70PGo1j5QOS` | `runs/modal_pythia_1_4b_attention_basis_b8_linear_layers8_factor_fmt_nf4_20260526/pythia_1_4b_attention_basis_b8_linear_layers8_factor_fmt_nf4_20260526` |

Results, sorted best to worst:

| format | ppl | ratio vs baseline | delta vs baseline | loss |
| --- | ---: | ---: | ---: | ---: |
| unquantized | 31.483839 | 1.026331 | +2.633% | 3.449474 |
| fp8_e5m2 | 31.559508 | 1.028798 | +2.880% | 3.451875 |
| fp8_e4m3 | 31.577230 | 1.029375 | +2.938% | 3.452436 |
| int8 | 31.673071 | 1.032500 | +3.250% | 3.455467 |
| nf4 | 31.716561 | 1.033917 | +3.392% | 3.456839 |
| int4 | 31.742840 | 1.034774 | +3.477% | 3.457667 |
| nvfp4 | 31.779203 | 1.035959 | +3.596% | 3.458812 |
| mxfp4 | 31.930027 | 1.040876 | +4.088% | 3.463547 |

Interpretation:
- Factor quantization is milder than the replacement itself: unquantized basis-8 on these 8 layers is `+2.633%`, and the best quantized factor run is `fp8_e5m2` at `+2.880%`.
- The additional degradation from factor quantization is about `+0.25%` PPL for `fp8_e5m2`, `+0.30%` for `fp8_e4m3`, `+0.62%` for `int8`, and `+0.84%` for `int4`, all relative to the full baseline.
- `nvfp4` and `mxfp4` were worse than expected here (`+3.596%` and `+4.088%`). Since this simulation uses generic scaled codebooks on dynamic NMF factors rather than an NVFP4-trained/fused path, treat those as numerical format stress tests, not final hardware claims.

## 2026-05-26: 2:4 Layer-Wise MCTS Run Control

Goal:
- Run layer-wise MCTS for 2:4 sparsity methods on `EleutherAI/pythia-1.4b`.
- Keep Modal usage capped at 6 GPUs while actually providing enough rollout states to fill 6 workers.

Implementation/control notes:
- The first optimized full MCTS command used `parallel_workers=8`; stopped after the GPU cap was tightened.
- A second command used `parallel_workers=6` but only `mcts_iterations=4`, so Modal could only fan out 4 rollout states. Stopped this underfilled run.
- Relaunched with `parallel_workers=6` and `mcts_iterations=6` so rollout batches have enough states to occupy all 6 workers.
- Modal volume cleanup removed the partial/stale directories from the stopped runs:
  - `/pythia_1_4b_sparse24_mcts_all_methods_20260526`
  - `/pythia_1_4b_sparse24_mcts_all_methods_optimized_20260526`
  - `/pythia_1_4b_sparse24_mcts_all_methods_6gpu_20260526`
  - `/pythia_1_4b_sparse24_mcts_all_methods_6gpu_i6_20260526` partial before the final detached relaunch
  - smoke run directories for the MCTS code path

Active run:
- Modal app: `ap-Eixqx1SRpehUlDL27gGxLv`
- URL: `https://modal.com/apps/jthomams477/main/ap-Eixqx1SRpehUlDL27gGxLv`
- Modal `app list` showed `6` containers for this app after relaunch.
- Run name: `pythia_1_4b_sparse24_mcts_all_methods_6gpu_i6_20260526`
- Methods: `magnitude,wanda,sparsegpt,qronos,gptaq-cae`
- Model: `EleutherAI/pythia-1.4b`
- MCTS: `mcts_iterations=6`, `mcts_rollout_depth=2`, `parallel_workers=6`
- Search/eval: `greedy_max_layers=24`, `search_layer_count=24`, `eval_steps=16`, batch size `64`, seq len `256`, bf16, Wikitext train calibration and test eval.

Important caveat:
- The run is faithful to the implemented method set in this repo. Methods not implemented here, such as full RIA+channel permutation, SparseSwaps, FISTAPruner, Thanos, and OPTIMA, were not silently approximated.

## 2026-05-26: MCTS Resume Patch After Modal Map Stall

Observation:
- App `ap-8Bbac2Y8eAkYGfw364xGDL` stopped producing stdout after `2026-05-26 19:56:06-04:00`.
- Modal still showed 1 active container, but the local log and volume directory stopped advancing at `magnitude_depth_18_rollouts`.
- Depth-18 state summaries were committed and readable from the Modal volume, so the stall was in the map/return tail rather than in the actual state eval.

Fix:
- Stopped the stale app and killed the stale `tmux` session.
- Added summary-based resume for sparse24 MCTS state evals: before launching GPU workers, the driver reads already committed `summary.json` files for requested states and only schedules missing/mismatched states.
- Removed the redundant final `volume.commit()` from `run_h100_sparse24_layer_state_batch`; the worker still commits after every completed state.
- Replaced Python's randomized `hash((run_name, method))` MCTS seed with a stable `blake2s` seed for reproducible restarts.
- Added sparse24 tests for reading matching committed summaries and rejecting mismatched summaries.

Validation:
- `.venv/bin/python -m pytest -q tests/test_sparse24.py` passed: 23 tests.
- `.venv/bin/python -m py_compile modal_layer_distill.py layer_distill/sparse24.py` passed.

Relaunch:
- New app: `ap-A94GZM6WY94Q4Ggauz2mZk`
- Same run name: `pythia_1_4b_sparse24_mcts_all_methods_6gpu_i6_20260526`
- Local log: `runs/modal_logs/pythia_1_4b_sparse24_mcts_all_methods_6gpu_i6_20260526_resume1.log`
- Recent `gpu_mem_total_mb` logs showed exactly 6 unique GPU-reporting containers, so the H100 cap is respected even when Modal app list shows additional non-GPU/warm containers.

## 2026-05-26: Sparse24 MCTS Warm-Worker Cache Patch

Observation:
- The MCTS run was spending too much wall time in repeated cold state evaluation.
- Each sparse24 state batch worker loaded the tokenizer, Wikitext batches, an FP Pythia-1.4B model, a sparse Pythia-1.4B model, module lists, and a dense weight snapshot on every Modal call.
- Because `mcts_iterations=6` and `parallel_workers=6`, most depth rollouts were one state per worker, so warm Modal containers still repeated the same load path across later depths.

Fix:
- Added `sparse24_worker_cache_key` keyed only by model/data/eval shape settings, not run name, batch name, or pruning method.
- Added a Modal worker-local cache for the loaded FP model, sparse model, calibration/eval batches, module groups, and dense sparse-model snapshot.
- The cache preserves experiment semantics: every state still restores the sparse model to the dense snapshot before applying its requested layer set.
- First cached relaunch failed due to a missing `json` import in the remote function; fixed and relaunched.

Validation:
- `.venv/bin/python -m pytest -q tests/test_sparse24.py` passed: 24 tests.
- `.venv/bin/python -m py_compile modal_layer_distill.py layer_distill/sparse24.py` passed.

Relaunch:
- Current app after the fixed relaunch: `ap-q4sE70O9yybxKWG1jS7QlS`
- Same run name: `pythia_1_4b_sparse24_mcts_all_methods_6gpu_i6_20260526`
- Local log: `runs/modal_logs/pythia_1_4b_sparse24_mcts_all_methods_6gpu_i6_20260526_resume3_cached.log`
- Initial log showed `sparse24_worker_cache_miss`, expected for the first state on a fresh H100 container.

## 2026-05-26: Sparse24 MCTS Max-Container Guard and Partial Pull

Run-control update:
- Stopped app `ap-q4sE70O9yybxKWG1jS7QlS` after logs showed 8 distinct GPU-reporting containers in a short window, despite `parallel_workers=6`.
- Added `max_containers=6` to the Modal sparse24 layer-state worker function and relaunched as app `ap-1a487ezSGNyXxXzF59QzCf`.
- Local log: `runs/modal_logs/pythia_1_4b_sparse24_mcts_all_methods_6gpu_i6_20260526_resume4_cached_max6.log`

Validation:
- `.venv/bin/python -m pytest -q tests/test_sparse24.py` passed: 24 tests.
- `.venv/bin/python -m py_compile modal_layer_distill.py layer_distill/sparse24.py` passed.

Outcome:
- App `ap-1a487ezSGNyXxXzF59QzCf` was stopped when the live guard observed 7 unique GPU-reporting containers. This preserved the user's hard 6-GPU cap.
- Completed committed results: `magnitude` through selected depth 23.
- Partial committed results: `wanda` through selected depth 3.
- Not reached before stop: `sparsegpt`, `qronos`, `gptaq-cae`.
- Pulled summaries to `runs/partial_sparse24_mcts_i6/`.

Partial scoreboard:
- Baseline PPL: `20.7707`.
- Magnitude selected depths PPL: depth 1 `21.3980`, 3 `23.3348`, 5 `25.4058`, 7 `27.6371`, 9 `31.4241`, 11 `35.5750`, 13 `39.9973`, 15 `46.2454`, 17 `56.0579`, 19 `73.6551`, 21 `117.981`, 23 `344.524`.
- Magnitude depth-23 layer order: `[19, 20, 10, 21, 6, 14, 22, 11, 15, 13, 7, 2, 18, 16, 12, 17, 1, 4, 8, 5, 23, 3, 0]`.
- Wanda selected depths PPL: depth 1 `21.3182`, depth 3 `22.5679`.
- Wanda depth-3 layer order: `[6, 9, 14]`.

## 2026-05-26: ResComp GD-Step Round-8 Probe

Goal:
- Replace the ResComp/GPTAQ-CAE Newton/Cholesky sparse compensation step with a masked gradient-descent compensation step.
- Test the previously selected 2:4 greedy round-8 layer state on `EleutherAI/pythia-1.4b`.
- User typed layer order `10,13,5,14,21,22,1,1`; treated this as the known round-8 greedy state `10,13,5,14,21,22,1,16`, because the recorded round-8 2:4 ResComp layer order has layer `16` as the eighth selected layer.

Implementation:
- Added sparse24 method `gptaq-cae-gd` with alias `rescomp-gd`.
- The method builds an exact N:M mask, initializes sparse weights with the dense weights under that mask, then applies masked GD on the paired objective `x_quant @ sparse.T` versus `x_fp @ dense.T`.
- Added `sparse24-state` Modal mode for ordered fixed layer-state evaluation.
- GD settings for this probe: `gd_steps=1`, `gd_lr=0.25`, `gd_chunk_tokens=8192`.

Validation:
- `.venv/bin/python -m pytest -q tests/test_sparse24.py` passed: 25 tests.
- `.venv/bin/python -m py_compile modal_layer_distill.py layer_distill/sparse24.py` passed.

Run:
- Modal app: `ap-t12bIiVu0pkYJaVQYlWkkM`
- Run name: `pythia_1_4b_sparse24_rescomp_gd_round8_20260526`
- Local log: `runs/modal_logs/pythia_1_4b_sparse24_rescomp_gd_round8_20260526.log`
- Method: `gptaq-cae-gd`, 2:4, bf16, 4 calibration steps, 16 eval steps, batch size 64, sequence length 256.

Result:
- Layer state: `[10, 13, 5, 14, 21, 22, 1, 16]`
- PPL: `718482.960941`
- Loss: `13.484897`
- Mean module recon rel MSE: `274.259371`
- Mean weight rel MSE: `0.397447`
- Density: `0.5`
- Sparse modules: `32`
- Sparsify elapsed: `43.65s`

Comparison:
- Earlier Newton/Cholesky ResComp 2:4 greedy round-8 PPL: `23.793806`.
- Baseline dense PPL from the same sparse24 harness: `20.770710`.
- Conclusion: this one-step GD replacement diverged badly. The per-module logs show the error explosion begins around later selected layers, especially MLP output projections. A future GD variant needs a much smaller step, per-module line search, or a safer normalized update; `gd_lr=0.25` is not viable.

## 2026-05-26: Corrected Sequential ResComp GD-Step Round-8 Probe

Issue with previous probe:
- The first `gptaq-cae-gd` implementation was mask-first GD and removed the sequential ResComp compensation path. That was not the requested test.

Fix:
- Reworked `gptaq-cae-gd` to keep the ResComp/GPTAQ-CAE sequential 2:4 projection loop.
- Replaced the compensation update with one first-order residual-gradient step on unprocessed columns.
- Kept the same block/lazy update structure for performance.
- Step size is `gd_lr / lambda_max(H_damped)` using power iteration on the damped Hessian.
- Restricted this method to exactly one local GD step for now; multi-step local GD needs residual recomputation.

Validation:
- `.venv/bin/python -m pytest -q tests/test_sparse24.py` passed: 26 tests.
- `.venv/bin/python -m py_compile modal_layer_distill.py layer_distill/sparse24.py` passed.

Run:
- Modal app: `ap-gmD7OMLl1zfMR4XzZEpnUm`
- Run name: `pythia_1_4b_sparse24_rescomp_gdseq_round8_20260526`
- Local log: `runs/modal_logs/pythia_1_4b_sparse24_rescomp_gdseq_round8_20260526.log`
- Local summary: `runs/modal_pythia_1_4b_sparse24_rescomp_gdseq_round8_20260526/summary.json`
- Method: `gptaq-cae-gd`, 2:4, bf16, `gd_steps=1`, `gd_lr=0.25`, 4 calibration steps, 16 eval steps.

Result:
- Layer state: `[10, 13, 5, 14, 21, 22, 1, 16]`
- PPL: `27.794873`
- Loss: `3.324852`
- Mean module recon rel MSE: `0.069208`
- Mean weight rel MSE: `0.135706`
- Density: `0.5`
- Sparse modules: `32`
- Sparsify elapsed: `51.63s`

Comparison:
- Corrected sequential GD is sane and far better than the bad mask-first probe: `718482.96 -> 27.79` PPL.
- It is still worse than Newton/Cholesky ResComp round 8: `27.79` vs `23.793806` PPL.
- This is the expected ordering for one conservative first-order step versus the inverse-Hessian compensation update.

## 2026-05-26: Corrected Sequential ResComp GD-Step LR Sweep

Goal:
- Sanity-check the corrected sequential ResComp-GD result across multiple step sizes, because a single `gd_lr=0.25` point was not trustworthy enough.

Setup:
- Model: `EleutherAI/pythia-1.4b`
- Layer state: `[10, 13, 5, 14, 21, 22, 1, 16]`
- Method: `gptaq-cae-gd`
- Sparsity: `2:4`
- `gd_steps=1`
- Eval: same 16-step Wikitext test eval as prior sparse24 round-8 probes.
- Launched Modal runs under `jthomams477`; max concurrent GPU jobs was 6.
- The first `0.0625` app stalled before weight loading and was stopped; reran as `lr0p0625b`.

Artifacts:
- Summary directory: `runs/modal_pythia_1_4b_sparse24_rescomp_gdseq_round8_lr_sweep_20260526/`
- Logs: `runs/modal_logs/pythia_1_4b_sparse24_rescomp_gdseq_round8_lr*_20260526.log`

Results:

| gd lr | ppl | loss | mean recon rel MSE | mean weight rel MSE |
|---:|---:|---:|---:|---:|
| 0.03125 | 28.224028 | 3.340174 | 0.072482 | 0.135629 |
| 0.0625 | 28.222310 | 3.340113 | 0.072021 | 0.135639 |
| 0.125 | 28.027059 | 3.333170 | 0.071100 | 0.135655 |
| 0.25 | 27.794873 | 3.324852 | 0.069208 | 0.135706 |
| 0.5 | 27.402206 | 3.310624 | 0.065409 | 0.135827 |
| 1.0 | 27.043095 | 3.297432 | 0.059113 | 0.136141 |
| 2.0 | 27.230760 | 3.304347 | 0.053451 | 0.137007 |
| 4.0 | 34.629132 | 3.544695 | 0.079650 | 0.139201 |

Interpretation:
- The corrected GD update is stable across a broad range and has a clear LR curve.
- Best tested LR is `1.0`, with PPL `27.043095`.
- `2.0` improves reconstruction MSE further but slightly worsens language-model PPL, so local reconstruction is not perfectly aligned with downstream PPL.
- `4.0` overshoots and clearly hurts PPL.
- Even the best GD point remains worse than Newton/Cholesky ResComp round 8 PPL `23.793806`.

## 2026-05-26: Diagonal-Hessian ResComp Round-8 Probe

Goal:
- Replace the corrected GD-step approximation with a diagonal-Hessian approximation for ResComp on the same 2:4 round-8 state.

Setup:
- Model: `EleutherAI/pythia-1.4b`
- Layer state: `[10, 13, 5, 14, 21, 22, 1, 16]`
- Method: `gptaq-cae-diag`
- Sparsity: `2:4`
- Eval: same 16-step Wikitext test eval as prior sparse24 round-8 probes.
- Modal profile: `jthomams477`

Implementation notes:
- Added `gptaq-cae-diag` / `rescomp-diag` as a sparse24 method.
- The method uses the same sequential 2:4 projection and CAE residual terms as ResComp, but replaces the full inverse-Hessian Cholesky factor with `diag(rsqrt(diag(H)))`.
- A first run was stopped and ignored because logs showed `has_fp_inputs=false`; fixed paired FP/sparse input collection and reran.

Validation:
- `.venv/bin/python -m pytest -q tests/test_sparse24.py` -> `27 passed`
- `.venv/bin/python -m py_compile modal_layer_distill.py layer_distill/sparse24.py`

Artifacts:
- Modal app: `ap-zBxWjtgwtaEIH2ceBOBOSb`
- Run name: `pythia_1_4b_sparse24_rescomp_diag_round8_paired_20260526`
- Local log: `runs/modal_logs/pythia_1_4b_sparse24_rescomp_diag_round8_paired_20260526.log`
- Local summary: `runs/modal_pythia_1_4b_sparse24_rescomp_diag_round8_paired_20260526/summary.json`

Result:

| method | ppl | loss | mean recon rel MSE | mean weight rel MSE | sparsify sec |
|---|---:|---:|---:|---:|---:|
| diagonal-Hessian ResComp | 24.119414 | 3.183017 | 0.048111 | 0.152269 | 50.74 |

Comparison:

| method | ppl | note |
|---|---:|---|
| dense baseline | 20.770710 | no 2:4 sparsity |
| full Newton/Cholesky ResComp | 23.793806 | best ResComp round-8 result so far |
| diagonal-Hessian ResComp | 24.119414 | close to full ResComp, much better than GD |
| best corrected GD-step ResComp | 27.043095 | best LR tested was `1.0` |
| invalid mask-first GD proxy | 718482.960900 | invalid earlier proxy, not comparable |

Interpretation:
- Diagonal-Hessian ResComp recovers most of the gap between the first-order GD probe and full Cholesky ResComp.
- It is still worse than the full inverse-Hessian update by about `0.326` PPL, so off-diagonal Hessian coupling is helping, but not nearly as much as the jump from GD to diagonal scaling.
- Diagonal recon rel MSE `0.048111` is better than the best GD recon rel MSE `0.059113`, matching the PPL ordering here.
