# 2026-05-26 QPS Throughput Benchmark

## Goal

Run paper-comparable QPS throughput experiments for baseline, embedding-norm prefill pruning, and SpecPrefill on Modal H100 using the existing LongBench QPS harness.

## Benchmark Scope

- Model: `Qwen/Qwen3-1.7B`
- SpecPrefill draft model: `Qwen/Qwen3-0.6B`
- Hardware: Modal `jthomams477`, H100
- Serving stack: vLLM `0.9.1`, `VLLM_USE_V1=0`
- Max model length: `40960`, matching Qwen3-1.7B config. vLLM rejected `65536` without an override.
- Categories and grids match repo paper scripts:
  - `few-shot-learning`: QPS `0.2..3.2` step `0.2`, 32 samples per dataset
  - `multi-doc-qa`: QPS `0.2,0.6,..,5.0`, 32 samples per dataset
  - `summarization`: QPS `0.2..2.2` step `0.2`, 32 samples per dataset
- Request timeouts match scripts after their `TIMEOUT + 5` behavior:
  - FSL: 25s
  - MDQA: 15s
  - Summarization: 45s

## Harness Notes

- Added structured QPS client JSON output while preserving legacy stdout markers.
- Switched QPS dataset loading to direct `THUDM/LongBench` `data.zip` reads to avoid dataset-script interactivity and preserve deterministic seeded samples.
- Added `--disable-log-requests` to vLLM serving. Without it, vLLM logs full prompts, making multi-doc throughput measure server log IO rather than method throughput.
- Parallelized at the process level by launching one Modal run per method, with method-specific output directories.

## Smoke Results

- Baseline smoke:
  - Artifact: `local/qps_benchmark_smoke/qps_throughput_20260526T051709Z_baseline`
  - FSL, 1 sample per dataset, 0.2 QPS: `4/4` success, avg latency `1.863s`
- Earlier smoke confirmed embedding-norm and SpecPrefill server paths before the request-log patch:
  - Artifact: `local/qps_benchmark_smoke/qps_throughput_20260526T040020Z`
  - embedding-norm FSL 0.2 QPS: `4/4` success, avg latency `1.868s`
  - SpecPrefill FSL 0.2 QPS: `4/4` success, avg latency `1.896s`

## Full Run Artifacts

- Baseline: `local/qps_benchmark_parallel/qps_throughput_20260526T052657Z_baseline`
- Embedding norm: `local/qps_benchmark_parallel/qps_throughput_20260526T052657Z_embedding_norm`
- SpecPrefill: `local/qps_benchmark_parallel/qps_throughput_20260526T052657Z_spec_prefill`
- Combined aggregate: `local/qps_benchmark_parallel/aggregate_20260526T032252`
- Charts:
  - `local/qps_benchmark_parallel/aggregate_20260526T032252/charts/few-shot-learning.svg`
  - `local/qps_benchmark_parallel/aggregate_20260526T032252/charts/multi-doc-qa.svg`
  - `local/qps_benchmark_parallel/aggregate_20260526T032252/charts/summarization.svg`

## Full Run Summary

| method | category | tested points | max ok QPS | paper grid completed | latency at max ok | timeouts at max ok |
|---|---:|---:|---:|---:|---:|---:|
| baseline | few-shot-learning | 16 | 3.2 | true | 1.517 | 0 |
| baseline | multi-doc-qa | 13 | 5.0 | true | 1.534 | 0 |
| baseline | summarization | 11 | 2.2 | true | 10.628 | 0 |
| embedding_norm | few-shot-learning | 16 | 3.2 | true | 2.145 | 0 |
| embedding_norm | multi-doc-qa | 13 | 5.0 | true | 0.836 | 0 |
| embedding_norm | summarization | 11 | 2.2 | true | 7.318 | 0 |
| spec_prefill | few-shot-learning | 16 | 3.2 | true | 3.283 | 0 |
| spec_prefill | multi-doc-qa | 13 | 5.0 | true | 3.173 | 1 |
| spec_prefill | summarization | 11 | 2.2 | true | 9.588 | 0 |

## Validation

- `python3 -m unittest discover -s tests`: 46 tests passed before full run.
- `python3 -m unittest tests.test_qps_benchmark_utils`: 7 tests passed after QPS runner edits.
- `python3 -m py_compile modal/run_qps_benchmark.py eval/qps_client.py speculative_prefill/qps_benchmarks/utils.py`: passed.
