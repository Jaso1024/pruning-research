# 2026-05-26 First Attention and First FFN Norm Accuracy

## Goal

Try token pruning by hidden-state norm immediately after the first attention module and immediately after the first FFN/MLP module.

## Implementation

- Extended `middle_layer_scorer.py` with `activation_target`:
  - `layer`: post transformer block, existing behavior.
  - `attn`: hook the target layer's first attention module, e.g. Qwen `self_attn`.
  - `ffn`: hook the target layer's FFN/MLP module, e.g. Qwen `mlp`.
- Added DeepSeek accuracy methods:
  - `first_attn_norm`
  - `first_ffn_norm`
- Both use `Qwen/Qwen3-0.6B`, `layer_index=1`, keep rate `0.3`, chunk size `32`.

## Accuracy Subset

- Artifact: `local/deepseek_accuracy_first_attn_ffn/modal_longbench_deepseek_subset_20260526T144506`
- Datasets: `triviaqa,passage_retrieval_en,qasper,gov_report`
- Limit: first 10 samples per dataset
- Target: DeepSeek `deepseek-chat`
- Compared against the prior run:
  - `local/deepseek_accuracy_middle_layer_valid/modal_longbench_deepseek_subset_20260526T141900`

| method | macro | triviaqa | passage_retrieval_en | qasper | gov_report |
|---|---:|---:|---:|---:|---:|
| baseline | 63.08 | 100.00 | 100.00 | 39.72 | 12.60 |
| first_attn_norm | 53.37 | 100.00 | 70.00 | 30.28 | 13.20 |
| first_ffn_norm | 40.91 | 94.00 | 40.00 | 16.33 | 13.32 |
| cross_family_spec_prefill | 52.50 | 100.00 | 80.00 | 17.39 | 12.61 |
| embedding_norm | 47.60 | 100.00 | 70.00 | 7.23 | 13.17 |
| middle_layer_norm | 43.83 | 94.00 | 50.00 | 18.26 | 13.04 |

The first-attention variant is the best pruned method on this subset by macro score, slightly above cross-family SpecPrefill. The first-FFN variant is worse than midpoint and embedding norm.

## Compression Time

| method | avg compression_s | p50 compression_s |
|---|---:|---:|
| first_attn_norm | 0.062480 | 0.042908 |
| first_ffn_norm | 0.040424 | 0.035596 |
| middle_layer_norm | 0.097525 | 0.095189 |
| cross_family_spec_prefill | 0.167609 | 0.157736 |
| embedding_norm | 0.029104 | 0.028100 |

First-attention norm is slower than embedding norm but faster than midpoint norm and cross-family SpecPrefill on this subset.

## Validation

- `python3 -m unittest discover -s tests`: 54 tests passed.
- `python3 -m py_compile speculative_prefill/vllm_patch/middle_layer_scorer.py speculative_prefill/vllm_patch/worker/middle_layer_norm_worker.py speculative_prefill/api_eval/pruning.py eval/deepseek_longbench_subset.py modal/run_deepseek_accuracy.py speculative_prefill/api_eval/modal_commands.py`: passed.
