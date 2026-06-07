# 2026-05-26 Larger Parallel Accuracy Subset

## Goal

Run a larger DeepSeek accuracy subset on Modal in parallel after first-attention and first-FFN norm variants looked promising on the 10-sample subset.

## Run Shape

- Modal profile/workspace: `jthomams477`
- Parallelism: one Modal run per dataset, four runs launched concurrently.
- Target API/model: DeepSeek `deepseek-chat`
- Datasets:
  - `triviaqa`
  - `passage_retrieval_en`
  - `qasper`
  - `gov_report`
- Limit: first 50 samples per dataset.
- Total samples: 200.
- Methods:
  - baseline
  - cross-family SpecPrefill, `Qwen/Qwen3-0.6B`
  - embedding norm, `Qwen/Qwen3-0.6B`
  - first-attention norm, `Qwen/Qwen3-0.6B`
  - first-FFN norm, `Qwen/Qwen3-0.6B`
  - middle-layer norm, `Qwen/Qwen3-0.6B`
- Keep rate: `0.3`
- Chunk size: `32`

## Artifacts

- `local/deepseek_accuracy_larger_parallel/modal_longbench_deepseek_subset_20260526T145058`
- `local/deepseek_accuracy_larger_parallel/modal_longbench_deepseek_subset_20260526T145055`
- `local/deepseek_accuracy_larger_parallel/modal_longbench_deepseek_subset_20260526T145039`
- `local/deepseek_accuracy_larger_parallel/modal_longbench_deepseek_subset_20260526T145140`
- Aggregate: `local/deepseek_accuracy_larger_parallel/aggregate_50x4_20260526T145229`

## Results

| method | macro | triviaqa | passage_retrieval_en | qasper | gov_report | avg comp s | p50 comp s | api errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 62.66 | 91.27 | 100.00 | 47.61 | 11.77 | 0.0000 | 0.0000 | 0 |
| cross_family_spec_prefill | 50.10 | 94.93 | 76.00 | 17.82 | 11.66 | 0.1851 | 0.1779 | 0 |
| embedding_norm | 48.17 | 93.27 | 62.00 | 26.17 | 11.25 | 0.0338 | 0.0318 | 0 |
| first_attn_norm | 50.99 | 94.20 | 64.00 | 34.45 | 11.30 | 0.0501 | 0.0432 | 0 |
| first_ffn_norm | 41.14 | 89.33 | 40.00 | 23.53 | 11.71 | 0.0516 | 0.0423 | 0 |
| middle_layer_norm | 44.71 | 91.07 | 52.00 | 24.33 | 11.45 | 0.1048 | 0.1015 | 0 |

First-attention norm remains the strongest pruned method by macro score on this larger subset and is materially faster than cross-family SpecPrefill in local compression time. SpecPrefill is better on passage retrieval, while first-attention norm is much better on Qasper.

## Validation

- `python3 -m unittest discover -s tests`: 54 tests passed.
- `python3 -m py_compile speculative_prefill/vllm_patch/middle_layer_scorer.py speculative_prefill/vllm_patch/worker/middle_layer_norm_worker.py speculative_prefill/api_eval/pruning.py eval/deepseek_longbench_subset.py modal/run_deepseek_accuracy.py speculative_prefill/api_eval/modal_commands.py`: passed.
