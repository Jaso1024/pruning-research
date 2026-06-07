# 2026-05-26 Middle-Layer Norm Prefill Pruning

## Goal

Test a stricter variant of embedding-norm pruning: instead of scoring tokens by input embedding vector norm, score each prompt token by the norm of its contextual hidden state at the model midpoint, then prune with the same chunked top-norm selection.

## Implementation

- Added `keep_strategy: middle_layer_norm`.
- Added `enable_middle_layer_norm_prefill()` and `ENABLE_MIDDLE_LAYER_NORM_PREFILL=1`.
- Added `MiddleLayerNormPrefillWorker`.
- The worker loads the vLLM-served model normally, then loads a sidecar HF `AutoModel` for contextual scoring.
- For each prefill prompt:
  - Run the full unpruned prompt through the sidecar model with `output_hidden_states=True`.
  - Use hidden state index `round(num_hidden_layers * layer_fraction)`, with `layer_fraction: 0.5`.
  - Score tokens by L2 hidden-state norm.
  - Apply the same chunked top-score selector as embedding norm.
  - Rewrite prompt token ids and position ids before vLLM prefill.

This is intentionally not a vocabulary-level cache. It is contextual and pays an extra scorer forward pass per prompt.

## Config

- `configs/config_middle_layer_norm_p3.yaml`
- `percentage: 0.3`
- `chunk: true`
- `chunk_size: 32`
- `norm: l2`
- `keep_high: true`
- `layer_fraction: 0.5`

## Validation

- `python3 -m unittest discover -s tests`: 49 tests passed.
- `python3 -m py_compile speculative_prefill/vllm_patch/worker/middle_layer_norm_worker.py speculative_prefill/vllm_patch/selector.py speculative_prefill/vllm_patch/config.py speculative_prefill/vllm_patch/__init__.py speculative_prefill/scripts.py modal/run_qps_benchmark.py`: passed.

## Smoke

- Modal workspace/profile: `jthomams477`
- Hardware: H100
- Model: `Qwen/Qwen3-1.7B`
- Command shape: FSL, 0.2 QPS, 1 sample per dataset, `gpu_memory_utilization=0.6`
- Artifact: `local/qps_benchmark_middle_layer_smoke/qps_throughput_20260526T154808Z_middle_layer_norm`
- Result: `4/4` requests succeeded, avg latency `1.375s`.

## Full Throughput Run

- Artifact: `local/qps_benchmark_middle_layer/qps_throughput_20260526T155158Z_middle_layer_norm`
- Aggregate with prior methods: `local/qps_benchmark_middle_layer/aggregate_with_middle_20260526T134258`
- Charts:
  - `local/qps_benchmark_middle_layer/aggregate_with_middle_20260526T134258/charts/few-shot-learning.svg`
  - `local/qps_benchmark_middle_layer/aggregate_with_middle_20260526T134258/charts/multi-doc-qa.svg`
  - `local/qps_benchmark_middle_layer/aggregate_with_middle_20260526T134258/charts/summarization.svg`

## Result Summary

| method | category | tested points | max ok QPS | paper grid completed | latency at max ok | first failed QPS | first failed timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | few-shot-learning | 16 | 3.2 | true | 1.517 |  |  |
| baseline | multi-doc-qa | 13 | 5.0 | true | 1.534 |  |  |
| baseline | summarization | 11 | 2.2 | true | 10.628 |  |  |
| embedding_norm | few-shot-learning | 16 | 3.2 | true | 2.145 |  |  |
| embedding_norm | multi-doc-qa | 13 | 5.0 | true | 0.836 |  |  |
| embedding_norm | summarization | 11 | 2.2 | true | 7.318 |  |  |
| middle_layer_norm | few-shot-learning | 16 | 3.2 | true | 4.403 |  |  |
| middle_layer_norm | multi-doc-qa | 10 | 3.4 | false | 3.798 | 3.8 | 24 |
| middle_layer_norm | summarization | 11 | 2.2 | true | 15.049 |  |  |
| spec_prefill | few-shot-learning | 16 | 3.2 | true | 3.283 |  |  |
| spec_prefill | multi-doc-qa | 13 | 5.0 | true | 3.173 |  |  |
| spec_prefill | summarization | 11 | 2.2 | true | 9.588 |  |  |

## Takeaway

Middle-layer norm works functionally, but the contextual scoring pass is expensive. It completes FSL and summarization paper grids, but fails multi-doc QA at 3.8 QPS, while baseline, embedding norm, and SpecPrefill all complete the 5.0 QPS MDQA grid under this harness.
