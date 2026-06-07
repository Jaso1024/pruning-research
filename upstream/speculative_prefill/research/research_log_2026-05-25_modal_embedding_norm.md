# Research Log: Modal Runner for Embedding-Norm Prefill

Date: 2026-05-25

## Sources

- Modal GPU docs: https://modal.com/docs/guide/gpu
- Modal image docs: https://modal.com/docs/reference/modal.Image
- Modal secrets docs: https://modal.com/docs/guide/secrets
- Modal run CLI docs: https://modal.com/docs/reference/cli/run

## Notes

- Modal functions request GPUs with the `gpu` argument, e.g. `gpu="L40S"` or `gpu="H100"`. Multi-GPU syntax is `gpu="H100:8"`, but the embedding-norm worker intentionally starts with `tensor_parallel_size=1` because tensor-parallel embedding tables may be sharded.
- Local source can be added to the Modal image with `Image.add_local_dir(...)` or package-focused helpers. This repo is easiest to run by copying the checkout into a fixed remote path and running commands with that as `cwd`.
- Hugging Face access should be supplied through a Modal secret or local environment forwarding. The runner uses `Secret.from_local_environ(["HF_TOKEN"])`, so gated Llama models require `HF_TOKEN` in the local environment before `modal run`.
- The first remote validation target should be synthetic latency, not LongBench, because it avoids dataset download/scoring complexity and proves the vLLM worker path actually runs on GPU.

## Intended Validation

Run three comparable modes at the same model, input length, output length, batch size, and iteration count:

1. `baseline`: no prefill pruning.
2. `embedding_norm`: base-model-only embedding-norm pruning.
3. `spec_prefill`: original draft-model SpecPrefill.

The first meaningful check is not whether embedding norm wins quality; it is whether it avoids draft-model overhead and reduces prefill latency at the requested keep rate.

## H100 Synthetic Runs: Paper Config Sweep

Date: 2026-05-25

Environment:
- Modal profile: `jthomams477`
- GPU request: `H100`
- Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Spec model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Shape: `input_len=1536`, `output_len=1`, `batch_size=4`
- Iterations: `warmup_iters=1`, `iters=3`

The paper efficiency-search script sweeps `p1`, `p3`, `p5`, `p7`, and the matching `*_full_lah8` variants. `p9` appears in downstream/LongBench-style scripts, so it was included here as an additional paper config.

| mode/config | avg latency seconds |
| --- | ---: |
| baseline | 0.044330972 |
| embedding_norm p30 | 0.038096050 |
| spec_prefill p1 | 0.111692416 |
| spec_prefill p3 | 0.122640643 |
| spec_prefill p5 | 0.123500984 |
| spec_prefill p7 | 0.192960208 |
| spec_prefill p9 | 0.196242619 |
| spec_prefill p1_full_lah8 | 0.423162476 |
| spec_prefill p3_full_lah8 | 0.345398059 |
| spec_prefill p5_full_lah8 | 0.408646859 |
| spec_prefill p7_full_lah8 | 0.338602334 |
| spec_prefill p9_full_lah8 | 0.286932657 |

Interpretation:
- These runs validate the paper config paths on Modal/H100, but they do not reproduce the paper speedup regime because the draft model is the same size as the base model.
- Plain percentage configs are much cheaper than `*_full_lah8` in this TinyLlama/TinyLlama setup.
- All SpecPrefill configs are slower than baseline here; embedding-norm p30 remains faster than baseline on this shape.

## Qwen/Qwen3.5-2B Smoke Test

Date: 2026-05-25

- `requirements.txt` pins `vllm==0.6.3.post1` and `transformers==4.50.2`.
- `Qwen/Qwen3.5-2B` declares `model_type=qwen3_5` and `transformers_version=4.57.0.dev0`.
- Baseline, embedding-norm, and SpecPrefill smoke tests all failed before model load with `KeyError: 'qwen3_5'` / Transformers does not recognize the architecture.
- A temporary current-vLLM Qwen baseline runner was tried and then deleted at user request. No current-vLLM runner remains in the repo.

## Qwen/Qwen3-1.7B Smoke Test

Date: 2026-05-25

- `Qwen/Qwen3-1.7B` declares `model_type=qwen3`, architecture `Qwen3ForCausalLM`, and `transformers_version=4.51.0`.
- Baseline smoke on the existing pinned stack failed before model load with `KeyError: 'qwen3'`; `transformers==4.50.2` does not recognize the architecture.
- A temporary Modal smoke app upgraded only Transformers to `4.51.3` while keeping `vllm==0.6.3.post1`. That got past config loading, but old vLLM then failed with `ValueError: Model architectures ['Qwen3ForCausalLM'] are not supported for now`.
- Result: Qwen3-1.7B does not work on this repo's pinned vLLM stack without a vLLM upgrade or a nontrivial compatibility patch.

## vLLM Upgrade Attempt for Qwen3

Date: 2026-05-25

- Official vLLM docs list `Qwen3ForCausalLM` as supported in vLLM `0.9.1`; vLLM `0.8.3` docs do not list Qwen3. The minimal practical target chosen here is therefore `vllm==0.9.1`.
- vLLM `0.9.1` source pins `torch==2.7.0`, so `requirements.txt` was moved to `torch==2.7.0`, `transformers==4.51.3`, and `vllm==0.9.1`.
- The Modal latency image was moved from CUDA `12.1.1` / Python `3.10` to CUDA `12.8.1` / Python `3.11`.
- vLLM `0.9.1` no longer exposes the old `vllm.executor.gpu_executor.GPUExecutor` module this repo patched. The embedding-norm path was adapted to route through `parallel_config.worker_cls` on vLLM `0.9.1`, with the old GPUExecutor monkey patch retained only as a legacy fallback.
- Embedding-norm mode forces `VLLM_USE_V1=0` because the repo's pruning path mutates V0 `SequenceData` / `SequenceGroupMetadata` internals.
- Local validation passed:
  - `/Users/jaso1024/.local/bin/python3.11 -m unittest discover -s tests -p 'test_vllm*_*.py'`
  - `python3 -m unittest tests/test_embedding_norm_selector.py`
  - `python3 -m py_compile $(find speculative_prefill eval examples modal -name '*.py' | tr '\n' ' ')`
- H100 Modal smoke could not be run after the upgrade because every configured workspace (`jthomams477`, `adola2048`, `fjosca`) failed app creation with `workspace billing cycle spend limit reached`.

## Proper Modal Profile Retry

Date: 2026-05-25

- The local shell had `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` set. Modal reported `Using adola2048 workspace based on environment variables`, so earlier `MODAL_PROFILE=...` attempts were not actually selecting each profile.
- Retried with `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=<profile> ...`.
- Results:
  - `jthomams477`: app creation worked; H100 runs completed.
  - `adola2048`: app creation failed with `workspace billing cycle spend limit reached`.
  - `fjosca`: app creation failed with `workspace billing cycle spend limit reached`.
- vLLM `0.9.1` CLI changed boolean flag parsing, so the Modal runner now uses `--no-enable-chunked-prefill` instead of `--enable-chunked-prefill False`.
- For comparable baseline and embedding-norm measurements, the Modal runner now sets `VLLM_USE_V1=0` for all modes.
- Qwen3 H100 smoke on `jthomams477`, model `Qwen/Qwen3-1.7B`, shape `input_len=128`, `output_len=1`, `batch_size=1`, `warmup_iters=0`, `iters=1`:

| mode | engine | avg latency seconds |
| --- | --- | ---: |
| baseline | vLLM V0 | 0.058244328 |
| embedding_norm p30 | vLLM V0 | 0.130930205 |

- A prior baseline smoke before forcing V0 used vLLM V1 and completed at `0.039941709` seconds. That number should not be compared against embedding-norm because the engine path differed.

## Qwen3 Full Synthetic Latency Pass

Date: 2026-05-25

Environment:
- Modal profile: `jthomams477`
- GPU request: `H100`
- Engine: vLLM V0, `vllm==0.9.1`, `VLLM_USE_V1=0`
- Base model: `Qwen/Qwen3-1.7B`
- Shape: `input_len=2048`, `output_len=1`, `batch_size=4`
- Iterations: `warmup_iters=1`, `iters=3`

| mode/config | avg latency seconds | status |
| --- | ---: | --- |
| baseline | 0.062673486 | completed |
| embedding_norm p30 | 0.105140494 | completed |
| spec_prefill p30, draft `Qwen/Qwen3-1.7B` | n/a | failed while loading the second Qwen model: `ValueError: Duplicate layer name: model.layers.0.self_attn.attn` |
| spec_prefill p30, draft `Qwen/Qwen3-0.6B` | n/a | failed with the same duplicate attention-layer name error while loading the draft model |

Notes:
- The SpecPrefill worker was patched enough for vLLM `0.9.1` to get past the old `LoraNotSupportedWorkerBase` import and old worker-constructor kwargs.
- The remaining SpecPrefill blocker is deeper: this repo's original SpecPrefill path loads a second model inside the same worker process and assumes Llama internals in `look_ahead_spec_worker.py`. On Qwen/vLLM `0.9.1`, it fails before reaching the Llama-specific assertion because vLLM's attention layer registry rejects duplicate layer prefixes for the second Qwen model.

## Qwen3 SpecPrefill vLLM 0.9.1 Port

Date: 2026-05-25

Changes:
- Load the draft model through vLLM's model-loader utilities with `prefix="spec_prefill_draft"` so base and draft attention layers have distinct registry names.
- Add Qwen3 model/layer/attention support in `LookAheadSpecWorker`.
- Capture draft attention queries through vLLM `get_forward_context()` instead of the old vLLM 0.6 attention-forward signature.
- Port the SpecPrefill scheduler monkey patch to vLLM 0.9.1's `_get_num_new_uncached_and_cached_tokens(...)`, partial-prefill metadata, and `SchedulingBudget.add_num_batched_tokens(...)` API.

Local validation:
- `/Users/jaso1024/.local/bin/python3.11 -m unittest discover -s tests -p 'test_vllm*_*.py'`: passed, 10 tests.
- `python3 -m unittest tests/test_embedding_norm_selector.py`: passed, 9 tests.
- `python3 -m py_compile $(rg --files -g '*.py' speculative_prefill eval examples modal tests)`: passed.

H100 Modal validation:

| mode/config | model | draft | shape | avg latency seconds | status |
| --- | --- | --- | --- | ---: | --- |
| spec_prefill p30 | `Qwen/Qwen3-1.7B` | `Qwen/Qwen3-0.6B` | `input_len=2048`, `output_len=1`, `batch_size=4`, `warmup_iters=1`, `iters=3` | 0.161013405 | completed |
| spec_prefill p30 smoke | `Qwen/Qwen3-1.7B` | `Qwen/Qwen3-1.7B` | `input_len=128`, `output_len=1`, `batch_size=1`, `warmup_iters=0`, `iters=1` | 0.220486311 | completed |

Commands:

```bash
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 \
  modal run modal/run_embedding_norm_latency.py \
  --mode spec_prefill \
  --model Qwen/Qwen3-1.7B \
  --spec-model Qwen/Qwen3-0.6B \
  --config configs/config_p3.yaml \
  --input-len 2048 \
  --output-len 1 \
  --batch-size 4 \
  --warmup-iters 1 \
  --iters 3
```

```bash
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 \
  modal run modal/run_embedding_norm_latency.py \
  --mode spec_prefill \
  --model Qwen/Qwen3-1.7B \
  --spec-model Qwen/Qwen3-1.7B \
  --config configs/config_p3.yaml \
  --input-len 128 \
  --output-len 1 \
  --batch-size 1 \
  --warmup-iters 0 \
  --iters 1
```

Interpretation:
- The previous duplicate attention-layer failure is fixed for both cross-size and same-model Qwen3 draft loading.
- The vLLM 0.9.1 scheduler incompatibility is fixed for the synthetic latency path.
- On the comparable full Qwen3 shape already measured above, SpecPrefill p30 with a Qwen3-0.6B draft is slower than baseline and embedding-norm p30. That is expected for this small H100 synthetic shape because SpecPrefill pays a second-model draft pass before pruning.

## Embedding-Norm Selector Optimization

Date: 2026-05-25

Changes:
- Keep the embedding norm table on CPU after the one-time model-load calculation, avoiding per-prefill GPU scoring plus CPU synchronization during request rewriting.
- Avoid transferring the full prompt-token tensor back from GPU when rebuilding `AugmentedSequenceData`.
- Replace full stable `argsort` selection with thresholded `topk` selection that keeps deterministic earlier-token tie behavior at the cutoff.
- Avoid padding allocation in chunk mode by computing full chunk means plus the tail chunk directly.

Local validation:
- `python3 -m unittest tests/test_embedding_norm_selector.py`: passed, 11 tests.
- `/Users/jaso1024/.local/bin/python3.11 -m unittest discover -s tests -p 'test_vllm*_*.py'`: passed, 10 tests.
- `python3 -m py_compile $(rg --files -g '*.py' speculative_prefill eval examples modal tests)`: passed.

Local selector microbenchmark:

| selector | 2048-token p30 per iteration |
| --- | ---: |
| old full stable argsort | 56.44 us |
| new thresholded topk | 40.31 us |

H100 Modal validation:

| mode/config | model | shape | avg latency seconds | status |
| --- | --- | --- | ---: | --- |
| embedding_norm p30 optimized | `Qwen/Qwen3-1.7B` | `input_len=2048`, `output_len=1`, `batch_size=4`, `warmup_iters=1`, `iters=3` | 0.071130286 | completed |

Prior comparable result before this optimization was `0.105140494` seconds. The same baseline from the full synthetic pass was `0.062673486` seconds.

## Qwen3 Broad Synthetic Shape Sweep

Date: 2026-05-25

Command:

```bash
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 \
  modal run modal/run_latency_sweep.py \
  --modes baseline,embedding_norm,spec_prefill \
  --model Qwen/Qwen3-1.7B \
  --spec-model Qwen/Qwen3-0.6B \
  --config configs/config_embedding_norm_p3.yaml \
  --spec-config configs/config_p3.yaml \
  --input-lens 256,512,1024,2048,4096 \
  --batch-sizes 1,2,4,8 \
  --output-len 1 \
  --warmup-iters 1 \
  --iters 3
```

Artifacts:
- CSV: `benchmark_results/qwen3_latency_sweep_20260525T232254Z/latency_sweep.csv`
- JSONL: `benchmark_results/qwen3_latency_sweep_20260525T232254Z/latency_sweep.jsonl`
- Raw per-mode JSON: `benchmark_results/qwen3_latency_sweep_20260525T232254Z/raw/`
- SVG charts: `benchmark_results/qwen3_latency_sweep_20260525T232254Z/charts/`

Matrix:
- Base model: `Qwen/Qwen3-1.7B`
- Spec draft: `Qwen/Qwen3-0.6B`
- Methods: baseline, embedding-norm p30, SpecPrefill p30
- Shapes: 5 input lengths x 4 batch sizes = 20 shapes per method, 60 measured rows total.
- The sweep benchmark loads one vLLM engine per method at max requested context length, then runs all shapes for that method.

Average latency seconds:

| batch | input | baseline | embedding_norm | spec_prefill |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 256 | 0.016204 | 0.017149 | 0.041662 |
| 1 | 512 | 0.016170 | 0.016182 | 0.042904 |
| 1 | 1024 | 0.016275 | 0.016038 | 0.044281 |
| 1 | 2048 | 0.016398 | 0.017048 | 0.046612 |
| 1 | 4096 | 0.030288 | 0.017721 | 0.158973 |
| 2 | 256 | 0.015411 | 0.016216 | 0.049399 |
| 2 | 512 | 0.015652 | 0.017603 | 0.045716 |
| 2 | 1024 | 0.016620 | 0.017700 | 0.048414 |
| 2 | 2048 | 0.028819 | 0.017579 | 0.057077 |
| 2 | 4096 | 0.061168 | 0.034076 | 0.102327 |
| 4 | 256 | 0.016481 | 0.016560 | 0.052922 |
| 4 | 512 | 0.017624 | 0.017970 | 0.063477 |
| 4 | 1024 | 0.028495 | 0.018103 | 0.078589 |
| 4 | 2048 | 0.057871 | 0.036339 | 0.154737 |
| 4 | 4096 | 0.121944 | 0.069148 | 0.261481 |
| 8 | 256 | 0.017371 | 0.018528 | 0.089085 |
| 8 | 512 | 0.028578 | 0.019444 | 0.103069 |
| 8 | 1024 | 0.057083 | 0.036334 | 0.172872 |
| 8 | 2048 | 0.116445 | 0.070711 | 0.312241 |
| 8 | 4096 | 0.247273 | 0.136891 | 0.563343 |

Takeaways:
- Embedding-norm p30 is close to baseline at small shapes and becomes faster once prefill cost dominates: the crossover appears around batch/input token products above roughly 2k-4k tokens.
- SpecPrefill p30 remains slower than both baseline and embedding-norm in every measured Qwen3 shape here.
- The `batch=1,input=4096` SpecPrefill average includes high variance; p50 was much lower than avg. The raw JSON should be used before drawing strong conclusions from that single cell.

## Qwen3 Longer Synthetic Inputs

Date: 2026-05-25

Commands:

```bash
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 \
  modal run modal/run_latency_sweep.py \
  --modes baseline,embedding_norm,spec_prefill \
  --model Qwen/Qwen3-1.7B \
  --spec-model Qwen/Qwen3-0.6B \
  --config configs/config_embedding_norm_p3.yaml \
  --spec-config configs/config_p3.yaml \
  --input-lens 8192,16384 \
  --batch-sizes 1,2,4 \
  --output-len 1 \
  --warmup-iters 1 \
  --iters 3
```

```bash
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 \
  modal run modal/run_latency_sweep.py \
  --modes baseline,embedding_norm,spec_prefill \
  --model Qwen/Qwen3-1.7B \
  --spec-model Qwen/Qwen3-0.6B \
  --config configs/config_embedding_norm_p3.yaml \
  --spec-config configs/config_p3.yaml \
  --input-lens 32768 \
  --batch-sizes 1,2 \
  --output-len 1 \
  --warmup-iters 1 \
  --iters 3
```

Artifacts:
- 8k/16k CSV: `benchmark_results/qwen3_latency_sweep_20260525T232947Z/latency_sweep.csv`
- 8k/16k charts: `benchmark_results/qwen3_latency_sweep_20260525T232947Z/charts/`
- 32k CSV: `benchmark_results/qwen3_latency_sweep_20260525T233330Z/latency_sweep.csv`
- 32k charts: `benchmark_results/qwen3_latency_sweep_20260525T233330Z/charts/`

Average latency seconds:

| batch | input | baseline | embedding_norm | spec_prefill |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8192 | 0.066232 | 0.024261 | 0.166329 |
| 1 | 16384 | 0.151923 | 0.043807 | 0.163935 |
| 2 | 8192 | 0.130371 | 0.040230 | 0.141247 |
| 2 | 16384 | 0.306080 | 0.085146 | 0.329542 |
| 4 | 8192 | 0.258051 | 0.081838 | 0.281082 |
| 4 | 16384 | 0.612920 | 0.171728 | 0.656727 |
| 1 | 32768 | 0.399389 | 0.102790 | 0.461696 |
| 2 | 32768 | 0.796591 | 0.203890 | 0.913239 |

Ratios vs baseline:

| batch | input | embedding_norm/base | spec_prefill/base |
| ---: | ---: | ---: | ---: |
| 1 | 8192 | 0.366 | 2.511 |
| 1 | 16384 | 0.288 | 1.079 |
| 2 | 8192 | 0.309 | 1.083 |
| 2 | 16384 | 0.278 | 1.077 |
| 4 | 8192 | 0.317 | 1.089 |
| 4 | 16384 | 0.280 | 1.071 |
| 1 | 32768 | 0.257 | 1.156 |
| 2 | 32768 | 0.256 | 1.146 |

Takeaways:
- Embedding-norm p30 gets more favorable as context grows, landing around 25-37% of baseline latency from 8k through 32k.
- SpecPrefill p30 becomes much closer to baseline at >=16k, but remains slower than baseline and much slower than embedding-norm in these Qwen3 synthetic runs.

## DeepSeek API Accuracy Subset on Modal/H100

Date: 2026-05-25

Changes:
- Added `eval/deepseek_longbench_subset.py` for API-target LongBench evaluation with baseline, embedding-norm text compression, and cross-family SpecPrefill-style text-span compression.
- Added `modal/run_deepseek_accuracy.py` so local compressors run on Modal H100 while DeepSeek is used only as the target API.
- LongBench is loaded from `THUDM/LongBench` `data.zip` directly to avoid noninteractive `datasets` trust prompts and version drift.
- DeepSeek calls are parallelized by default through the async client.

Validation:
- `python3 -m unittest tests/test_deepseek_accuracy_utils.py`: passed, 8 tests.
- `python3 -m py_compile speculative_prefill/api_eval/*.py eval/deepseek_longbench_subset.py modal/run_deepseek_accuracy.py`: passed.

Modal dry-run command:

```bash
set -a
. /Users/jaso1024/Documents/compressioncompany/research/.env
set +a
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 \
  uv tool run modal run modal/run_deepseek_accuracy.py \
  --datasets passage_retrieval_en \
  --limit 1 \
  --deepseek-model deepseek-chat \
  --scorer-models Qwen/Qwen3-0.6B \
  --draft-models Qwen/Qwen3-0.6B \
  --keep-rate 0.3 \
  --chunk-size 32 \
  --lookahead 1 \
  --max-tokens 32 \
  --concurrency 2 \
  --dry-run
```

Dry-run artifact:
- `local/deepseek_accuracy/modal_longbench_deepseek_subset_20260525T200818/`

Dry-run token/compression timings:

| method | local model | original local tokens | kept local tokens | compression seconds |
| --- | --- | ---: | ---: | ---: |
| baseline | none | 0 | 0 | 0.000 |
| embedding_norm | `Qwen/Qwen3-0.6B` | 11304 | 3400 | 0.038 |
| cross_family_spec_prefill | `Qwen/Qwen3-0.6B` | 11304 | 3400 | 0.578 |

Live subset command:

```bash
set -a
. /Users/jaso1024/Documents/compressioncompany/research/.env
set +a
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 \
  uv tool run modal run modal/run_deepseek_accuracy.py \
  --datasets passage_retrieval_en,triviaqa \
  --limit 1 \
  --deepseek-model deepseek-chat \
  --scorer-models Qwen/Qwen3-0.6B,HuggingFaceTB/SmolLM2-360M-Instruct \
  --draft-models Qwen/Qwen3-0.6B,HuggingFaceTB/SmolLM2-360M-Instruct \
  --keep-rate 0.3 \
  --chunk-size 32 \
  --lookahead 1 \
  --max-tokens 64 \
  --concurrency 5
```

Live artifact:
- `local/deepseek_accuracy/modal_longbench_deepseek_subset_20260525T201032/`

Scores by method and local model:

| method | local model | passage_retrieval_en | triviaqa |
| --- | --- | ---: | ---: |
| baseline | none | 100.0 | 100.0 |
| embedding_norm | `Qwen/Qwen3-0.6B` | 100.0 | 100.0 |
| embedding_norm | `HuggingFaceTB/SmolLM2-360M-Instruct` | 0.0 | 100.0 |
| cross_family_spec_prefill | `Qwen/Qwen3-0.6B` | 100.0 | 100.0 |
| cross_family_spec_prefill | `HuggingFaceTB/SmolLM2-360M-Instruct` | 0.0 | 100.0 |

Prediction details:
- `passage_retrieval_en`: baseline and Qwen-pruned prompts answered `Paragraph 15`; SmolLM embedding-norm answered `Paragraph 14`, and SmolLM cross-family SpecPrefill answered `Paragraph 7`.
- `triviaqa`: every method answered `United States`.

Interpretation:
- This is only an initial two-dataset, one-sample-per-dataset subset; it is not a full accuracy claim.
- Qwen3-0.6B preserved both initial examples at 30% keep rate for both pruning methods.
- SmolLM2-360M was too weak for the retrieval example at this keep rate in both pruning methods, even though it preserved the TriviaQA case.

## Full DeepSeek API LongBench Accuracy Run

Date: 2026-05-25

Scope:
- Paper-comparable LongBench full split from this repo's `eval/long_bench/pred_vllm.py` non-`_e` dataset list.
- Target model/API: DeepSeek `deepseek-chat`.
- Local scorer/draft model: `Qwen/Qwen3-0.6B` for both embedding-norm and cross-family SpecPrefill-style text-span compression.
- Methods: baseline, embedding-norm p30, cross-family SpecPrefill p30.
- Keep rate: `0.3`; chunk size: `32`; cross-family lookahead: `1`.
- Samples: all 4,750 LongBench samples; rows: 14,250 DeepSeek generations.
- Generation limits: original LongBench `dataset2maxlen.json` values, cap `512`, with newline stop for `samsum`.

Commands:

```bash
set -a
. /Users/jaso1024/Documents/compressioncompany/research/.env
set +a
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 \
  uv tool run modal run modal/run_deepseek_accuracy.py \
  --datasets all \
  --limit all \
  --deepseek-model deepseek-chat \
  --scorer-models Qwen/Qwen3-0.6B \
  --draft-models Qwen/Qwen3-0.6B \
  --keep-rate 0.3 \
  --chunk-size 32 \
  --lookahead 1 \
  --max-tokens 512 \
  --concurrency 8 \
  --sample-block-size 25
```

The first long run exited after 145/190 shards. It was resumed with:

```bash
env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 \
  uv tool run modal run modal/run_deepseek_accuracy.py \
  --datasets all \
  --limit all \
  --deepseek-model deepseek-chat \
  --scorer-models Qwen/Qwen3-0.6B \
  --draft-models Qwen/Qwen3-0.6B \
  --keep-rate 0.3 \
  --chunk-size 32 \
  --lookahead 1 \
  --max-tokens 512 \
  --concurrency 8 \
  --sample-block-size 25 \
  --skip-existing \
  --continue-on-api-error
```

Notes:
- `--skip-existing` skips shard directories already present locally.
- `--continue-on-api-error` records DeepSeek API refusals as `api_error` rows with empty predictions, which score as zero. This preserves benchmark coverage without filtering or rewriting benchmark content.
- One transient `lcc` baseline API failure was rerun by moving its old shard into `local/deepseek_accuracy/superseded_api_error_shards/` and rerunning the missing shard with `--skip-existing`.

Artifacts:
- Final aggregate: `local/deepseek_accuracy/aggregates/deepseek_accuracy_aggregate_20260525T232958.json`
- Stable latest aggregate: `local/deepseek_accuracy/aggregates/latest.json`
- Shards: `local/deepseek_accuracy/modal_longbench_deepseek_*_s*_n25_*`
- Superseded transient-error shard: `local/deepseek_accuracy/superseded_api_error_shards/modal_longbench_deepseek_lcc_s125_n25_20260525T230428/`

Validation:
- `python3 -m unittest discover -s tests`: passed, 39 tests.
- `python3 -m py_compile speculative_prefill/api_eval/*.py eval/deepseek_longbench_subset.py eval/aggregate_deepseek_accuracy.py modal/run_deepseek_accuracy.py`: passed.
- Coverage: 190/190 shards, 14,250/14,250 rows, 4,750/4,750 samples.
- API errors retained in final aggregate: 3 rows total: baseline 2, cross-family SpecPrefill 1. These were DeepSeek `Content Exists Risk` refusals in `passage_retrieval_zh`.

Macro average across the 21 LongBench datasets:

| method | macro score |
| --- | ---: |
| baseline | 38.34 |
| cross-family SpecPrefill p30 | 31.63 |
| embedding-norm p30 | 28.24 |

Per-dataset scores:

| dataset | n | baseline | cross-family SpecPrefill | embedding norm |
| --- | ---: | ---: | ---: | ---: |
| narrativeqa | 200 | 37.40 | 33.62 | 28.85 |
| qasper | 200 | 49.22 | 27.12 | 29.99 |
| multifieldqa_en | 150 | 57.99 | 33.85 | 32.61 |
| multifieldqa_zh | 200 | 1.25 | 1.25 | 0.50 |
| hotpotqa | 200 | 71.42 | 59.69 | 54.71 |
| 2wikimqa | 200 | 69.43 | 49.38 | 50.78 |
| musique | 200 | 51.33 | 39.01 | 37.66 |
| dureader | 200 | 0.40 | 0.13 | 0.15 |
| gov_report | 200 | 42.64 | 42.17 | 40.15 |
| qmsum | 200 | 27.89 | 22.93 | 22.47 |
| multi_news | 200 | 34.05 | 30.71 | 31.09 |
| vcsum | 200 | 0.06 | 0.06 | 0.09 |
| trec | 200 | 34.00 | 29.00 | 24.50 |
| triviaqa | 200 | 92.84 | 93.21 | 91.80 |
| samsum | 200 | 48.56 | 47.65 | 43.67 |
| lsht | 200 | 51.50 | 54.00 | 38.50 |
| passage_count | 200 | 24.00 | 5.50 | 3.50 |
| passage_retrieval_en | 200 | 100.00 | 83.50 | 56.50 |
| passage_retrieval_zh | 200 | 0.00 | 0.00 | 0.00 |
| lcc | 500 | 7.18 | 6.21 | 1.70 |
| repobench-p | 500 | 4.00 | 5.23 | 3.84 |

Interpretation:
- On this DeepSeek-target/API-retokenized setup, 30% text-span pruning loses substantial accuracy versus full-context baseline.
- Cross-family SpecPrefill p30 is better than embedding-norm p30 on macro average and on many QA/retrieval tasks, but it still trails baseline materially.
- The result is not a direct reproduction of the original paper's target model stack; it is paper-comparable in dataset coverage, prompt/scoring surface, and method/keep-rate structure, with DeepSeek substituted as the target API per the experiment plan.
