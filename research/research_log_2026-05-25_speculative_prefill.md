# Research Log: Speculative Prefill

Date: 2026-05-25

## Sources

- Liu, Jingyu; Chen, Beidi; Zhang, Ce. "Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation." arXiv:2502.02789, v2, 2025-05-19. https://arxiv.org/abs/2502.02789
- Upasani, Shubhangi; Raju, Ravi Shanker; Li, Bo; Ji, Mengmeng; Long, John; Wu, Chen; Thakker, Urmish; Wang, Guangtao. "Cross-Family Speculative Prefill: Training-Free Long-Context Compression with Small Draft Models." arXiv:2603.02631, v3, 2026-03-13. https://arxiv.org/abs/2603.02631
- Reference implementation for Liu et al.: https://github.com/Jingyu6/speculative_prefill

## Core Idea

Prefill is often compute-bound, and for medium/long prompts the target model spends most of its first-token latency doing full forward passes over tokens that may not be needed for the final answer. Speculative prefill uses a smaller draft/speculator model to score prompt-token importance, keeps the highest-scoring chunks, and sends only those chunks to the target model. Unlike speculative decoding, this targets TTFT/prefill rather than autoregressive decode throughput.

The mechanism is training-free:

1. Run a draft model on the full prompt.
2. Collect attention from the last prompt token plus several lookahead generated tokens.
3. Aggregate attention into one scalar importance score per prompt token.
4. Smooth scores, chunk the prompt, and select top chunks under a keep-rate budget.
5. Run the target model on the selected content.

## SpecPrefill: Original Same-Family Form

Paper: Liu et al. 2025.

Important implementation details:

- Draft and target are expected to be in the same model family/tokenizer regime.
- The paper uses Llama-3.1-8B-Instruct as the speculator for Llama-3.1-70B-Instruct and Llama-3.1-405B-Instruct-FP8.
- Attention tensor shape is `[N, L, S, H]`, where `N` is lookahead steps, `L` layers, `S` sequence length, `H` heads.
- Importance aggregation is max over layer/head dimensions, then mean over lookahead.
- They smooth token scores with 1D average pooling, then partition into contiguous chunks and select top-K chunks.
- They preserve original token position IDs for selected tokens. This is critical in their formulation, especially for synthetic position-sensitive tasks like retrieval/counting.
- Decoding position IDs are offset to the original full prompt length, even though fewer prompt tokens are passed to the target.
- vLLM integration is a monkey patch. The README says it must run before constructing vLLM `LLM`.
- Reported vLLM version in the paper appendix: `0.6.3.post1`, with `enforce_eager=True` and `enable_chunked_prefill=False`.
- Their "full LAH" variant uses 8 lookahead steps; beyond 16 lookahead gave little additional gain.

Reported results:

- Up to 7x maximum end-to-end QPS improvement for Llama-3.1-405B-Instruct-FP8 on real downstream tasks.
- Up to 7.66x TTFT improvement in synthetic efficiency benchmarking at 10% keep rate.
- On LongBench, multi-doc QA / single-doc QA / few-shot tasks can often preserve quality at low keep rates; summarization degrades more; some code tasks improve for smaller target models, plausibly from denoising.
- On RULER with 10% keep rate, performance is preserved on retrieval, multi-hop tracking, and QA, but aggregation tasks degrade because every word can matter.
- Compared with MInference, SpecPrefill is strongest for large batch and short-to-medium context lengths under 128k. MInference becomes more competitive as context length grows and batch size shrinks.

Limitations:

- Token-dropping methods cannot output logits for all original input tokens.
- Multi-turn conversation may fail without full context recomputation or storing all KV caches.
- Fixed keep rate is crude; adaptive keep-rate selection is an open problem.
- Pure attention-derived importance may be weaker than more principled saliency estimators.

## Cross-Family Speculative Prefill

Paper: Upasani et al. 2026.

This paper asks whether the same attention-based saliency signal transfers when draft and target models come from different families. Their answer is mostly yes across Qwen, LLaMA, and DeepSeek pairings.

Key modification versus original SpecPrefill:

- Chunk scoring and top-K selection stay the same.
- Selected draft-token chunks are mapped back to text spans.
- Adjacent selected chunks are merged.
- Non-contiguous spans are concatenated with a delimiter.
- The resulting compressed text is re-tokenized with the target tokenizer.
- Position IDs are reset to contiguous target-side positions instead of restoring original positions.

This avoids trying to align original position IDs across incompatible tokenizers and architectures.

Keep-rate handling:

- Define keep rate relative to target tokenization: `rho = compressed_target_length / original_target_length`.
- Apply that fraction in the draft domain when selecting chunks.
- Re-tokenization means the final target-side length can drift slightly.
- Their hardware-friendly compressed lengths are 8k, 16k, 24k, and 32k.

Constants from appendix:

- Lookahead steps: `N = 8`.
- LongBench/RULER chunk size: 32.
- LongBench/RULER smoothing: 1D average pooling kernel size 13.
- LongBench/RULER delimiter: `[...]`.
- Code Debug delimiter: `// omitted`.
- Code Debug chunk size: 128.
- Decoding: greedy for reported results.

Reported results:

- LongBench v2:
  - Llama-3.1-8B full prompt: 31.2 average. Cross-family Qwen3-1.7B draft at 25% keep: 29.6; at 50% keep: 31.2.
  - Qwen3-8B full prompt: 29.2 average. Llama-3.2-1B draft at 25% keep: 28.4; Qwen3-1.7B draft at 25% keep: 31.8.
  - DeepSeek-R1 full prompt: 58.3 average. Qwen3-4B draft at 6% keep: 53.3; Llama-3.1-8B draft at 6% keep: 54.1.
- Code Debug:
  - DeepSeek-V3.1 full prompt: 67.51 accuracy. Llama-3.1-8B draft: 64.72 at 20% keep, 59.13 at 15% keep.
  - DeepSeek-R1 full prompt: 74.37 accuracy. Llama-3.1-8B draft: 70.30 at 30% keep, 68.02 at 25%, 62.44 at 15%.
- RULER:
  - Compressing 128k prompts to 16k with a lightweight draft reduced TTFT from 46s to about 2.5s, roughly 18x.
  - Compressing to 32k produced about 4.3s TTFT.

Interpretation:

- Attention-based importance appears substantially model-family agnostic for many long-context tasks.
- Denoising can make compressed prompts outperform full prompts on some retrieval-heavy tasks.
- Larger/longer-context draft models help mostly when the draft needs to score very long inputs. In the DeepSeek-R1 LongBench v2 setup, Qwen3-4B with 262k context beat Qwen3-1.7B with 32k context.
- Code debugging is a weak spot. Fine-grained dependencies across files/functions can be damaged by aggressive span dropping. A code-focused implementation should probably add structural constraints, not rely on pure top-K attention chunks.

## Implementation-Relevant Takeaways

For a first local prototype, cross-family text-level reconstruction is simpler and more portable than original-position-ID restoration:

- It can work with any target tokenizer.
- It avoids model-specific position ID surgery in the target engine.
- It can run as a preprocessor before a normal target model call.
- It will not get the full systems benefit unless the target model truly avoids processing dropped tokens, but it is enough to validate quality/selection.

For a performance-oriented vLLM path, original SpecPrefill is more invasive:

- Need access to draft-model attentions and lookahead queries/KV.
- Need request scheduling around draft prefill plus target prefill.
- Need position-ID control for same-family selected-token forwarding.
- Need to disable incompatible vLLM features if following the paper's implementation baseline.

Suggested reproduction order:

1. Implement offline compressor that takes prompt text, draft model, target tokenizer, keep target length, chunk size, pooling kernel, delimiter.
2. Test deterministic invariants first: order preservation, no duplicate spans, target-token budget approximation, delimiter insertion only between non-contiguous spans.
3. Validate saliency on synthetic prompts where a unique answer-bearing span is embedded in distractors.
4. Benchmark quality on a small LongBench/RULER subset with a cheap target model.
5. Only after quality is credible, integrate with a serving stack for TTFT/QPS measurement.

Do not claim true TTFT gains from a prototype unless the target-model forward actually skips the dropped tokens and draft overhead is measured end-to-end.
