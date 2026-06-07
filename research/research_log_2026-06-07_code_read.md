# 2026-06-07 Code Read Log

## Scope

- Repository: `/Users/jaso1024/Documents/code/spec-prefill`
- Goal: read current project source and tests so future work in this thread can use live repo context.
- Included: project Python, shell/config/Markdown docs, tests, Modal harnesses, and upstream SpecPrefill source.
- Excluded from full read: `.venv`, `.git`, caches, generated `runs/`, generated `benchmark_results/`, and vendored `external/`.

## Initial Inventory

- Active Python files after exclusions: 117
- Active Python LOC after exclusions: 39,051
- Main areas:
  - `upstream/speculative_prefill`: vLLM SpecPrefill patch, benchmarks, evals, tests.
  - `unstructured-to-structured`: saliency/pruning research harnesses and CIFAR AirBench Modal work.
  - `layer_composition`: layer merge/distill/removal, low-rank attention, sparse 2:4 experiments.

## Read Progress

- Started with repo shape, git status, AGENTS instructions, and prior-memory orientation.
- Completed a structural pass over every active Python file after exclusions.
- Read the upstream SpecPrefill implementation, DeepSeek/LongBench eval path, QPS/latency harnesses, and upstream tests inventory.
- Read the unstructured-to-structured saliency/pruning implementation, AirBench Modal harness, GPTQ/Qronos/affine harnesses, and tests inventory.
- Read the layer-composition implementation: attention analysis, attention-basis PPL patching, hybrid attention, layer removal, low-QK distillation/adapters, layer merge, Muon optimizer, sparse 2:4, Modal launchers, and tests inventory.
- Read the root shell helper `superwhisper-entitlement-snapshot.sh`.

## Architecture Notes

- `upstream/speculative_prefill` is an upstream-style vLLM monkey patch. `enable_prefill_spec()` configures env/state, patches executor, scheduler, input building, and registers cleanup. Runtime token pruning is driven by `SpecPrefillWorker` plus `LookAheadSpecWorker`; API-eval compression is a separate text-compressor path.
- `unstructured-to-structured` is a research harness collection for saliency, WANDA variants, N:M pruning, GPTQ/Qronos quantization/pruning, affine toy checks, and CIFAR AirBench recovery experiments. Most heavy work is Modal/H100-oriented and writes JSON/JSONL artifacts.
- `layer_composition` is a second research track around layer behavior and model surgery: merge/distill adjacent layers, compress or replace attention, evaluate attention-head basis approximations, remove layers greedily, train low-QK adapters, and apply/evaluate structured 2:4 sparsity.
- The shared experiment style is explicit config dataclasses, deterministic seeds where practical, full-sequence `use_cache=False` eval for patched modules, JSONL step logging, summary JSON/Markdown, and Modal volume commits for remote runs.

## Caveats Found While Reading

- Several replacement modules intentionally do not implement KV cache paths: hybrid external attention, head-basis compression, low-QK replacement, and layer-removal eval.
- Upstream SpecPrefill has hard support limits around non-Llama/Qwen3 draft models, LoRA/prompt adapters, beam search, and some cache ops.
- The upstream SCROLLS eval patch contains a legacy comment saying `let's cheat a bit` for NarrativeQA pruning. I did not see that phrase in the active local research harnesses.
- All top-level project directories are currently untracked in git, so future edits should avoid assuming git can distinguish local generated/source state cleanly.
