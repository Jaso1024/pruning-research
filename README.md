# Spec Prefill Research Workspace

This repository is a snapshot of messy preliminary research code around speculative prefill, token pruning, structured pruning, and layer-composition experiments. It is not a polished library, not a stable benchmark package, and not organized as one clean product. The goal is to preserve the current working code and notes so the experiments can be inspected, resumed, or refactored later.

## What Is Here

- `upstream/speculative_prefill/`: a modified upstream SpecPrefill/vLLM monkey-patch tree with API eval, DeepSeek/LongBench helpers, QPS/latency harnesses, and tests.
- `unstructured-to-structured/`: saliency and pruning experiments, including WANDA-style scoring, N:M pruning, GPTQ/Qronos experiments, affine toy checks, CIFAR AirBench Modal runs, and related tests.
- `layer_composition/`: layer merge/distillation/removal experiments, attention-head analysis and basis compression, hybrid attention replacement, low-QK adapters, structured 2:4 sparsity, Modal launchers, and tests.
- `depth-anything-2/`: Depth Anything V2 compression research on DA-2K, including token merging/pruning, WANDA/GPTQ-style baselines, circuit discovery and ablation, scalar saliency, activation replacement, MoE-style sparsity, and LoRA/PEFT repair sweeps. See `depth-anything-2/COMPRESSION_RESEARCH.md`.
- `circuit-resynthesis/`: MNIST prototype for decompiling dense MLP structure into local/circuit masks and resynthesizing smaller students.
- `research/` and top-level research notes: logs and notes from the exploratory work.
- `superwhisper-entitlement-snapshot.sh`: a separate local research helper that redacts sensitive-looking values before printing Superwhisper entitlement/cache metadata.

## Current State

Expect rough edges:

- Multiple subprojects have separate `pyproject.toml` or setup conventions.
- Much of the heavy evaluation path assumes Modal/H100 or local CUDA.
- Generated outputs, local virtualenvs, caches, and large tensors are intentionally ignored.
- Several model replacement paths are eval-only and intentionally do not implement KV-cache inference.
- Some files are modified upstream research code rather than clean upstream source.

The code has tests in the subproject `tests/` directories, but this snapshot should be treated as research scaffolding first and production code second.

## Useful Starting Points

```bash
# Layer-composition unit tests
cd layer_composition
pytest

# Unstructured-to-structured unit tests
cd unstructured-to-structured
pytest

# Upstream SpecPrefill tests
cd upstream/speculative_prefill
python -m unittest discover -s tests
```

Modal launchers:

- `layer_composition/modal_layer_distill.py`
- `unstructured-to-structured/modal_pythia_saliency.py`
- `unstructured-to-structured/modal_cifar_airbench.py`
- `upstream/speculative_prefill/modal/run_deepseek_accuracy.py`
- `upstream/speculative_prefill/modal/run_qps_benchmark.py`

## Repository Hygiene

The commit excludes local environments, caches, generated run outputs, benchmark outputs, and large model/tensor artifacts. If a future result matters, promote it into a small summary file under `research/` rather than committing raw `runs/` contents.
