# 2026-05-26 Middle-Layer Norm Optimization and Accuracy

## Optimization

The first middle-layer norm implementation was too slow because it ran a full sidecar HF model forward with `output_hidden_states=True`, then read the midpoint hidden state. That computes every layer and stores all hidden states.

The optimized implementation:

- Adds `speculative_prefill/vllm_patch/middle_layer_scorer.py`.
- Locates transformer blocks, including direct `.layers` used by Qwen `AutoModel`.
- Registers a forward hook on the target midpoint block.
- Captures that block output and raises an internal exception to stop the forward immediately.
- Does not pass `output_hidden_states=True`.

This keeps the method contextual while avoiding the second half of the scorer model and all hidden-state materialization.

## Throughput Sanity Checks

- Optimized smoke:
  - Artifact: `local/qps_benchmark_middle_layer_optimized_smoke/qps_throughput_20260526T180630Z_middle_layer_norm`
  - FSL 0.2 QPS, 1 sample per dataset: `4/4` success, avg latency `0.981s`
  - Prior unoptimized smoke was `1.375s`
- Old failure point probe:
  - Artifact: `local/qps_benchmark_middle_layer_optimized_probe/qps_throughput_20260526T181921Z_middle_layer_norm`
  - MDQA 3.8 QPS, 32 samples per dataset: `128/128` success, avg latency `1.956s`
  - Prior unoptimized run failed at 3.8 QPS with `24/128` timeouts.

## Accuracy Subset

Initial run with `Downloads/Credentials/env` was invalid because that file only had placeholder `DEEPSEEK_API_KEY=your_deepseek_api_key_here`.

Valid-key run:

- Artifact: `local/deepseek_accuracy_middle_layer_valid/modal_longbench_deepseek_subset_20260526T141900`
- Datasets: `triviaqa,passage_retrieval_en,qasper,gov_report`
- Limit: first 10 samples per dataset
- Target: DeepSeek `deepseek-chat`
- Local methods:
  - baseline
  - embedding norm, `Qwen/Qwen3-0.6B`
  - cross-family SpecPrefill, `Qwen/Qwen3-0.6B`
  - middle-layer norm, `Qwen/Qwen3-0.6B`
- Keep rate: `0.3`
- Chunk size: `32`

| method | macro | triviaqa | passage_retrieval_en | qasper | gov_report |
|---|---:|---:|---:|---:|---:|
| baseline | 63.17 | 100.00 | 100.00 | 39.72 | 12.97 |
| cross_family_spec_prefill | 52.50 | 100.00 | 80.00 | 17.39 | 12.61 |
| embedding_norm | 47.60 | 100.00 | 70.00 | 7.23 | 13.17 |
| middle_layer_norm | 43.83 | 94.00 | 50.00 | 18.26 | 13.04 |

Compression timing from the same subset:

| method | avg compression_s | p50 compression_s |
|---|---:|---:|
| baseline | 0.000009 | 0.000008 |
| embedding_norm | 0.029104 | 0.028100 |
| cross_family_spec_prefill | 0.167609 | 0.157736 |
| middle_layer_norm | 0.097525 | 0.095189 |

Middle-layer norm is now faster than cross-family SpecPrefill in local compression time on this subset, but accuracy is worse than both baseline and SpecPrefill on this small sample.

## Validation

- `python3 -m unittest discover -s tests`: 52 tests passed.
- `python3 -m py_compile speculative_prefill/vllm_patch/middle_layer_scorer.py speculative_prefill/vllm_patch/worker/middle_layer_norm_worker.py speculative_prefill/api_eval/pruning.py eval/deepseek_longbench_subset.py modal/run_deepseek_accuracy.py speculative_prefill/api_eval/modal_commands.py modal/run_qps_benchmark.py`: passed.
