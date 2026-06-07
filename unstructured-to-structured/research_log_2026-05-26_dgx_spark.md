# DGX Spark Investigation Log - 2026-05-26

Target: `100.87.8.90`

## Local Context

- Working directory: `/Users/jaso1024/Documents/code/spec-prefill/unstructured-to-structured`
- Repo/memory search found SpecPrefill benchmark context, but no prior notes for `100.87.8.90` or this Spark.

## Reachability

- ICMP reachable: 3/3 packets received, ~8.8 ms average RTT.
- Reverse DNS: `spark-1095.tail095c86.ts.net.`
- Local machine has a Tailscale-style `100.110.136.76` interface.
- `tailscale` CLI is not on the local PATH, but the macOS app exists at `/Applications/Tailscale.app`.

## Exposed Services From Local Machine

Checked ports: `22 80 443 8000 8080 8888 3000 5000 5001 6006 7860 8265 9090 11434`

- Open: `22/tcp`, SSH banner `SSH-2.0-Tailscale`
- Refused: all other checked ports

## SSH Access

- `root@100.87.8.90` works in batch mode.
- Tested usernames `jaso1024`, `ubuntu`, `nvidia`, `dgx`, `spark`, `jason`, `jasont`, `jaso`, and `admin`; each failed because the remote local user was not present.

## Host Summary

- Hostname: `spark-1095`
- OS: Ubuntu 24.04.4 LTS
- Kernel: `6.17.0-1014-nvidia`
- Architecture: `aarch64`
- Uptime at investigation: 10 days, 19 hours
- Load average: `0.04 0.11 0.09`
- Users: `root`, `opt32`

## Hardware

- CPU: NVIDIA GB10 Spark CPU, ARM Cortex-X925, 20 logical CPUs
- Memory: 121 GiB total, 118 GiB available at inspection time
- Swap: 15 GiB
- GPU: NVIDIA GB10
- Driver: 580.142
- CUDA runtime reported by `nvidia-smi`: 13.0
- GPU utilization: 0%
- GPU processes: only desktop display processes (`Xorg`, `gnome-shell`)

`nvidia-smi` reports GPU memory accounting as `Not Supported`, which appears consistent with this GB10/unified-memory setup rather than a normal discrete GPU memory report.

## Storage

- Root filesystem: `/dev/nvme0n1p2`, ext4, 3.7 TiB total, 775 GiB used, 2.8 TiB available
- EFI partition: `/dev/nvme0n1p1`

## Services

- No failed systemd units.
- Docker is active; no running containers.
- Tailscale is active.
- NVIDIA persistence daemon is active.
- DGX Dashboard is active and local-only:
  - `dgx-dashboard.service`
  - `/opt/nvidia/dgx-dashboard-service/dashboard-service -port 11000 serve`
  - Listens on `127.0.0.1:11000`
  - `curl http://127.0.0.1:11000/` returns the DGX Dashboard SPA HTML
- `dgx-dashboard-admin.service` is active.

Remote listeners of note:

- `0.0.0.0:22` and `[::]:22`
- `127.0.0.1:11000` for DGX Dashboard
- `127.0.0.1:631` for CUPS
- Tailscale service ports

## Tooling

- Python: `/usr/bin/python3`, Python 3.12.3
- Root/global Python modules checked: `torch`, `vllm`, `transformers`, `numpy` all missing
- `opt32` login Python modules checked: `torch`, `vllm`, `transformers`, `numpy` all missing
- CUDA toolkit is installed under `/usr/local/cuda-13.0`
- `nvcc` exists at `/usr/local/cuda/bin/nvcc`, but is not on the default PATH in the root SSH session.
- Docker: version 29.2.1
- NVIDIA container tooling: `nvidia-container-cli` 1.19.0
- Docker images include CUDA 13 Ubuntu images, TensorRT-LLM 1.3.0rc14, TVM/compiler/benchmark images, and local `*-dgxspark` compiler/benchmarker images.

## Filesystem/Projects

Top-level user account:

- `/home/opt32`

Git/project directories found at shallow depth:

- `/home/opt32/opt32/.git`
- `/home/opt32/test/autokernel/.git`
- `/home/opt32/.codex/memories/.git`
- `/home/opt32/.nvm/.git`
- `/home/opt32/.config/nvim/.git`

Notable project-looking directories under `/home/opt32/opt32`:

- `compiler-experiments`
- `alphakernel`
- `models`
- `tools`
- `papers`
- `graphs`

## Network Notes

- Remote outbound HTTPS to GitHub works.
- Remote Tailscale health check reports: `Tailscale can't reach the configured DNS servers. Internet connectivity may be affected.`
- The default network interface is Wi-Fi: `wlP9s9`, address `10.105.47.34/20`.

## Immediate Next Steps

- For the dashboard: use an SSH tunnel, e.g. `ssh -L 11000:127.0.0.1:11000 root@100.87.8.90`, then open `http://127.0.0.1:11000/`.
- For ML work: create a dedicated Python environment; the global Python is bare despite CUDA/toolkit availability.
- For containerized work: inspect or reuse the existing DGX Spark-specific Docker images before installing large host-level packages.
- If DNS-dependent Tailscale behavior matters, inspect Tailscale DNS configuration on the Spark.

## Pythia 31M GSM8K Parameter Saliency - 2026-05-26

- Added a modular local package under `saliency/` for GSM8K calibration batching, per-parameter saliency accumulation, experiment execution, and artifact writing.
- Saliency method: `abs(parameter * gradient)` accumulated over answer-token language-model loss batches and normalized by supervised answer tokens. This gives a parameter-level gradient-weighted saliency tensor for every trainable parameter.
- GSM8K loader uses `openai/gsm8k` with config `main`; current HF Hub tooling rejects legacy bare `gsm8k`.
- Tests: `uv run pytest -q` -> 14 passed.
- Local smoke: `uv run python -m saliency.cli --output-dir runs/local_pythia31m_gsm8k_smoke --model-name EleutherAI/pythia-31m --max-examples 1 --batch-size 1 --max-length 256 --dtype fp32 --device cpu --top-k 5`.
- Modal key handling: direct `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` env overrides initially hit a billing-cap workspace; unsetting them and using `MODAL_PROFILE=jthomams477` launched under the correct account.
- Modal smoke: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --max-examples 1 --batch-size 1 --max-length 256 --dtype bf16 --run-name pythia31m_gsm8k_modal_smoke_20260526`.
- Full Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --max-examples 0 --batch-size 32 --max-length 512 --dtype fp32 --run-name pythia31m_gsm8k_full_saliency_20260526`.
- Full run results: 7,473 GSM8K train examples, 234 batches, 721,858 supervised answer tokens, loss/token 3.613166570621836, 76 parameter tensors, 30,494,720 scalar saliency values, total saliency 538.2901029959321.
- Local artifacts: `runs/modal_pythia31m_gsm8k_full_saliency_20260526_files/saliency.pt`, `summary.json`, and `parameter_summary.jsonl`.
- Top saliency tensor by total score: `gpt_neox.layers.4.attention.query_key_value.weight` with sum 44.39107131958008.

## Pythia 31M GSM8K 50% Matrix Pruning PPL - 2026-05-26

- Added `saliency/prune_eval.py` and `saliency/prune_cli.py`.
- Pruning rule: for each 2D weight matrix, zero the lowest-saliency 50% of entries using the saved GSM8K saliency tensor; leave biases and 1D parameters unchanged.
- Tests: `uv run pytest -q` -> 18 passed.
- Local smoke: `uv run python -m saliency.prune_cli --output-dir runs/local_prune_ppl_smoke --saliency-path runs/modal_pythia31m_gsm8k_full_saliency_20260526_files/saliency.pt --model-name EleutherAI/pythia-31m --max-examples 4 --batch-size 2 --max-length 256 --dtype fp32 --device cpu --prune-fraction 0.5`.
- Full Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --max-examples 0 --batch-size 32 --max-length 512 --dtype fp32 --saliency-run-name pythia31m_gsm8k_full_saliency_20260526 --prune-fraction 0.5 --run-name pythia31m_gsm8k_prune50_ppl_20260526`.
- Full run results: 26 matrix tensors pruned, 15,237,120 / 30,474,240 matrix weights zeroed, exact zero fraction 0.5, no missing saliency tensors.
- Baseline GSM8K answer-token PPL: 37.08338807512854, loss/token 3.613169108560392.
- Pruned GSM8K answer-token PPL: 139.06676297670938, loss/token 4.934954127004117.
- Change: +101.98337490158085 PPL, 3.7501094208266283x PPL ratio, +1.321785018443725 loss/token.
- Local artifacts: `runs/modal_pythia31m_gsm8k_prune50_ppl_20260526/prune_ppl_summary.json` and `pruned_tensors.jsonl`.

## Pythia 31M GSM8K 25% Matrix Pruning PPL - 2026-05-26

- Pruning rule: for each 2D weight matrix, zero the lowest-saliency 25% of entries using the saved GSM8K saliency tensor; leave biases and 1D parameters unchanged.
- Local smoke: `uv run python -m saliency.prune_cli --output-dir runs/local_prune25_ppl_smoke --saliency-path runs/modal_pythia31m_gsm8k_full_saliency_20260526_files/saliency.pt --model-name EleutherAI/pythia-31m --max-examples 4 --batch-size 2 --max-length 256 --dtype fp32 --device cpu --prune-fraction 0.25`.
- Full Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --max-examples 0 --batch-size 32 --max-length 512 --dtype fp32 --saliency-run-name pythia31m_gsm8k_full_saliency_20260526 --prune-fraction 0.25 --run-name pythia31m_gsm8k_prune25_ppl_20260526`.
- Full run results: 26 matrix tensors pruned, 7,618,560 / 30,474,240 matrix weights zeroed, exact zero fraction 0.25, no missing saliency tensors.
- Baseline GSM8K answer-token PPL: 37.08338807512854, loss/token 3.613169108560392.
- Pruned GSM8K answer-token PPL: 40.40890439444631, loss/token 3.69905016648444.
- Change: +3.325516319317771 PPL, 1.0896767121866129x PPL ratio, +0.08588105792404832 loss/token.
- Local artifacts: `runs/modal_pythia31m_gsm8k_prune25_ppl_20260526/prune_ppl_summary.json` and `pruned_tensors.jsonl`.

## Pythia 1.4B GSM8K 100-Sample bf16 Saliency + 25% Matrix Pruning PPL - 2026-05-26

- User requested the Pythia 1.4B version in bf16 and limited to 100 GSM8K samples.
- A prior fp32 smoke with 4 samples completed but was superseded by the bf16 request.
- Saliency run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode saliency --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 4 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_saliency_bf16_20260526`.
- Saliency results: 100 train examples, 25 batches, 9,848 supervised answer tokens, loss/token 2.0426482534524775, 292 parameter tensors, 1,414,647,808 scalar saliency values, total saliency 4365.340932162479.
- Top saliency tensor by total score: `gpt_neox.layers.0.mlp.dense_4h_to_h.weight` with sum 88.86261749267578.
- PPL run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_saliency_bf16_20260526 --prune-fraction 0.25 --run-name pythia14b_gsm8k_100_prune25_ppl_bf16_20260526`.
- Pruning results: 98 matrix tensors pruned, 353,501,184 / 1,414,004,736 matrix weights zeroed, exact zero fraction 0.25, no missing saliency tensors.
- Baseline 100-sample GSM8K answer-token PPL: 7.720404623083257, loss/token 2.0438667749796915.
- Pruned 100-sample GSM8K answer-token PPL: 7.698485014076171, loss/token 2.0410235580828595.
- Change: -0.021919609007086116 PPL, 0.9971608212163455x PPL ratio, -0.0028432168968319793 loss/token.
- Small local artifacts downloaded: `runs/modal_pythia14b_gsm8k_100_saliency_bf16_20260526/summary.json`, `parameter_summary.jsonl`, `runs/modal_pythia14b_gsm8k_100_prune25_ppl_bf16_20260526/prune_ppl_summary.json`, and `pruned_tensors.jsonl`.
- The full 1.4B `saliency.pt` artifact was left on the Modal volume because it is multi-GB and the local run only needs summaries unless further offline analysis is requested.

## Pythia 1.4B GSM8K 100-Sample bf16 50% Matrix Pruning PPL - 2026-05-26

- Reused saliency artifact: `/results/pythia14b_gsm8k_100_saliency_bf16_20260526/saliency.pt`.
- Full Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_saliency_bf16_20260526 --prune-fraction 0.5 --run-name pythia14b_gsm8k_100_prune50_ppl_bf16_20260526`.
- Pruning results: 98 matrix tensors pruned, 707,002,368 / 1,414,004,736 matrix weights zeroed, exact zero fraction 0.5, no missing saliency tensors.
- Baseline 100-sample GSM8K answer-token PPL: 7.720404623083257, loss/token 2.0438667749796915.
- Pruned 100-sample GSM8K answer-token PPL: 9.455036120335519, loss/token 2.2465475223395615.
- Change: +1.7346314972522618 PPL, 1.2246814230520875x PPL ratio, +0.20268074735986996 loss/token.
- Local artifacts: `runs/modal_pythia14b_gsm8k_100_prune50_ppl_bf16_20260526/prune_ppl_summary.json` and `pruned_tensors.jsonl`.

## Pythia 1.4B GSM8K 100-Sample bf16 Global 50% Matrix Pruning PPL - 2026-05-26

- User clarified the desired pruning rule: pool saliency scores across the whole LLM, then prune least-salient to most-salient globally, rather than enforcing the same fraction per matrix.
- Implemented global pruning over all 2D weight tensors via `pruning_scope="global"`; non-matrix parameters remain unchanged.
- Scoped tests: `uv run pytest tests/test_prune_eval.py -q` -> 6 passed. Full collection still has an unrelated GPTQ import failure.
- Full Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_saliency_bf16_20260526 --prune-fraction 0.5 --pruning-scope global --run-name pythia14b_gsm8k_100_global_prune50_ppl_bf16_20260526`.
- Pruning results: 98 matrix tensors touched, 707,002,368 / 1,414,004,736 matrix weights zeroed globally, exact zero fraction 0.5, global threshold 1.6066993566710153e-06, no missing saliency tensors.
- Baseline 100-sample GSM8K answer-token PPL: 7.720404623083257, loss/token 2.0438667749796915.
- Global-pruned PPL: 12.87961483880283, loss/token 2.5556458164094233.
- Change: +5.159210215719573 PPL, 1.6682564538514002x PPL ratio, +0.5117790414297319 loss/token.
- Most-pruned tensors by fraction: `gpt_neox.embed_in.weight` 99.11383514792561%, `embed_out.weight` 98.36159043639671%, `gpt_neox.layers.21.attention.dense.weight` 79.31520938873291%.
- Least-pruned tensors by fraction: `gpt_neox.layers.8.attention.dense.weight` 15.584540367126465%, `gpt_neox.layers.7.attention.dense.weight` 17.131423950195312%, `gpt_neox.layers.9.attention.dense.weight` 17.3184871673584%.
- Local artifacts: `runs/modal_pythia14b_gsm8k_100_global_prune50_ppl_bf16_20260526/prune_ppl_summary.json` and `pruned_tensors.jsonl`.

## Pythia 1.4B GSM8K 100-Sample bf16 Global 25% Matrix Pruning PPL - 2026-05-26

- Reused saliency artifact: `/results/pythia14b_gsm8k_100_saliency_bf16_20260526/saliency.pt`.
- Full Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_saliency_bf16_20260526 --prune-fraction 0.25 --pruning-scope global --run-name pythia14b_gsm8k_100_global_prune25_ppl_bf16_20260526`.
- Pruning results: 98 matrix tensors touched, 353,501,184 / 1,414,004,736 matrix weights zeroed globally, exact zero fraction 0.25, global threshold 4.026218789476843e-07, no missing saliency tensors.
- Baseline 100-sample GSM8K answer-token PPL: 7.720404623083257, loss/token 2.0438667749796915.
- Global-pruned PPL: 9.28756002744153, loss/token 2.2286758732737613.
- Change: +1.5671554043582736 PPL, 1.2029887655981957x PPL ratio, +0.18480909829406977 loss/token.
- Most-pruned tensors by fraction: `gpt_neox.embed_in.weight` 98.05340754773477%, `embed_out.weight` 96.83811100384662%, `gpt_neox.layers.21.attention.query_key_value.weight` 39.73712126413981%.
- Least-pruned tensors by fraction: `gpt_neox.layers.8.attention.dense.weight` 3.9548873901367188%, `gpt_neox.layers.7.attention.dense.weight` 4.352450370788574%, `gpt_neox.layers.9.attention.dense.weight` 4.384160041809082%.
- Local artifacts: `runs/modal_pythia14b_gsm8k_100_global_prune25_ppl_bf16_20260526/prune_ppl_summary.json` and `pruned_tensors.jsonl`.

## Pythia 31M GSM8K Test GPTQ FP8 - 2026-05-26

- Added `saliency/gptq_eval.py` and `saliency/gptq_cli.py`.
- Method: collect activation Hessians for every `torch.nn.Linear` module on GSM8K train, then GPTQ-quantize each linear weight matrix to per-row scaled `torch.float8_e4m3fn`; leave biases and embeddings unchanged. Pythia-31M has 25 linear modules under this rule, including `embed_out`.
- Tests: `uv run pytest -q` -> 21 passed.
- Local smoke: `uv run python -m saliency.gptq_cli --output-dir runs/local_gptq_fp8_smoke --model-name EleutherAI/pythia-31m --max-calibration-examples 1 --max-eval-examples 1 --calibration-batch-size 1 --eval-batch-size 1 --max-length 128 --dtype fp32 --device cpu --blocksize 64`.
- Full Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode gptq-fp8 --model-name EleutherAI/pythia-31m --max-calibration-examples 0 --max-eval-examples 0 --calibration-batch-size 32 --eval-batch-size 32 --max-length 512 --dtype fp32 --run-name pythia31m_gsm8k_test_gptq_fp8_20260526`.
- Modal run URL: `https://modal.com/apps/jthomams477/main/ap-PSY2vD6fQckXmWpXFlxx8d`.
- Calibration split: GSM8K train, 7,473 examples, 234 batches, 2,248,349 activation rows per linear module including padding positions.
- Eval split: GSM8K test, 1,319 examples, 42 batches, 130,710 supervised answer tokens.
- Baseline GSM8K test answer-token PPL: 35.80871058660608, loss/token 3.5781911762860052.
- GPTQ FP8 GSM8K test answer-token PPL: 35.889853855497044, loss/token 3.5804546331394405.
- Change: +0.0811432688909619 PPL, 1.0022660204056975x PPL ratio, +0.0022634568534352084 loss/token.
- Quantized weights: 17,596,416 linear weights; weighted mean absolute quantization error 0.0007105088381710665.
- Local artifacts: `runs/modal_pythia31m_gsm8k_test_gptq_fp8_20260526/pythia31m_gsm8k_test_gptq_fp8_20260526/gptq_fp8_summary.json` and `gptq_layers.jsonl`.

## Pythia 31M GSM8K Test Multi-Step GPTQ FP8 - 2026-05-26

- Added multi-step GPTQ controls to `saliency/gptq_eval.py`, `saliency/gptq_cli.py`, and `modal_pythia_saliency.py`.
- Multi-step method: keep a CPU snapshot of original FP32 linear weights; at each step, collect fresh linear activation Hessians from the current quantized model on GSM8K train, then GPTQ-quantize the original FP32 linear weights using that step's Hessians. This avoids the degenerate no-op of re-quantizing already-FP8 weights and keeps each evaluated model as one FP8 quantized weight tensor per linear module.
- Tests: `uv run pytest -q` -> 23 passed.
- Local smoke: `uv run python -m saliency.gptq_cli --output-dir runs/local_gptq_fp8_multistep_smoke --model-name EleutherAI/pythia-31m --max-calibration-examples 1 --max-eval-examples 1 --calibration-batch-size 1 --eval-batch-size 1 --max-length 128 --dtype fp32 --device cpu --blocksize 64 --gptq-steps 2 --eval-steps 1,2`.
- Full Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode gptq-fp8 --model-name EleutherAI/pythia-31m --max-calibration-examples 0 --max-eval-examples 0 --calibration-batch-size 32 --eval-batch-size 32 --max-length 512 --dtype fp32 --gptq-steps 8 --eval-steps 1,2,4,8 --run-name pythia31m_gsm8k_test_gptq_fp8_multistep_20260526`.
- Modal run URL: `https://modal.com/apps/jthomams477/main/ap-tyWR9XXOQK57grexlGHpPa`.
- Calibration split: GSM8K train, 7,473 examples per step, 234 batches per step, 2,248,349 activation rows per linear module including padding positions.
- Eval split: GSM8K test, 1,319 examples, 42 batches, 130,710 supervised answer tokens.
- Baseline GSM8K test answer-token PPL: 35.80871058660608, loss/token 3.5781911762860052.
- Step 1 GPTQ FP8: PPL 35.889853855497044, loss/token 3.5804546331394405, ratio 1.0022660204056975, delta PPL +0.0811432688909619.

## Qronos Paper-Faithful Weight-Only PTQ - 2026-05-27

- Paper/code target: Qronos, arXiv 2505.11695, with the official Brevitas Qronos implementation in `external/brevitas-qronos/src/brevitas/graph/qronos.py` and Brevitas docs at `https://xilinx.github.io/brevitas/dev/papers/qronos.html`.
- Implemented `saliency/qronos_eval.py` for the existing Pythia/GSM8K harness. The core update follows Brevitas Qronos: collect quantized-input covariance `H = X_tilde^T X_tilde`, cross covariance `G = X^T X_tilde`, normalize by batch examples, use optional activation order by `diag(H)`, compute the first proxy column from `G` and `triu(H, 1)`, apply the Sherman-Morrison tail inverse update, and diffuse later column errors with the upper Cholesky factor in blocks.
- Matched the Brevitas LLM defaults where relevant: asymmetric per-row weight-only quantization, `percdamp=1e-6`, `num_blocks=100`, activation order on, skip `embed_out`/`lm_head` unless explicitly requested. The paper-code-faithful default does not mask padding; `use_attention_mask` remains an ablation flag for the GSM8K padded harness.
- Added tests in `tests/test_qronos_eval.py` for endpoint-preserving asymmetric quantization, identity matched-input RTN behavior, use of cross covariance in the first Qronos column, hook snapshotting, and CPU/CUDA parity when CUDA is available.
- Critical CUDA fix: `torch.set_float32_matmul_precision("high")` allowed TF32 on H100 and made Qronos covariance/inverse math non-faithful. The Pythia-31M CUDA smoke before the fix had PPL ratio `138.84357377573969`. Disabling TF32 for Qronos changed the same smoke to baseline PPL `70.91240000261821`, Qronos PPL `70.42093996498193`, ratio `0.9930694767400604`.
- Local tests after fixes: `uv run pytest tests/test_qronos_eval.py tests/test_gptq_eval.py -q` -> 17 passed. Compile check: `uv run python -m py_compile saliency/qronos_eval.py modal_pythia_saliency.py` -> passed.
- Local CPU Pythia-31M smoke: baseline PPL `70.91183734542656`, Qronos W4 PPL `71.2720693607482`, ratio `1.005079998330418`, 24 linear layers quantized.
- Pythia-1.4B 4-example H100 large-matrix smoke: run `pythia14b_qronos_w4_fp32_4ex_smoke_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-eAffuaQApZUMuB1H1nFQu6`, baseline PPL `12.167617453807862`, Qronos W4 PPL `14.531224233485958`, ratio `1.194253870049013`, 96 layers quantized, elapsed `90.50689125061035s`.
- Parallel H100 grid launched with 8 clients under `MODAL_PROFILE=jthomams477` and env tokens unset. Seven completed; `pythia14b_gsm8k100_qronos_w4_fp32_blocks50_20260527` was stopped before model load after hanging before a summary. Local parsed grid summary: `runs/qronos_modal_grid_summary_20260527.json`.
- Pythia-1.4B GSM8K 100-example Qronos grid results:
  - W4 FP32 paper-faithful: baseline PPL `7.655819188354375`, Qronos PPL `7.687473183795954`, ratio `1.0041346320573674`, elapsed `351.80858540534973s`.
  - W4 FP32 no activation order: baseline PPL `7.655819188354375`, Qronos PPL `7.674180970232209`, ratio `1.0023984085081012`, elapsed `347.15629982948303s`.
  - W4 FP32 masked tokens only: baseline PPL `7.655819188354375`, Qronos PPL `7.683617398694524`, ratio `1.0036309909699062`, elapsed `340.4537663459778s`.
  - W4 FP32 `percdamp=1e-5`: baseline PPL `7.655819188354375`, Qronos PPL `7.683921702378474`, ratio `1.0036707389937902`, elapsed `352.00488471984863s`.
  - W4 FP32 quantize last layer: baseline PPL `7.655819188354375`, Qronos PPL `7.685475410511221`, ratio `1.0038736837204774`, elapsed `360.73456287384033s`.
  - W4 BF16 paper-faithful: baseline PPL `7.711002873131888`, Qronos PPL `7.7360996870533025`, ratio `1.0032546757321101`, elapsed `139.5677354335785s`.
  - W3 FP32 paper-faithful: baseline PPL `7.655819188354375`, Qronos PPL `7.883586279355673`, ratio `1.029750845128078`, elapsed `386.42413568496704s`.
- Step 2 GPTQ FP8: PPL 36.05388307361149, loss/token 3.5850145714839865, ratio 1.0068467275975335, delta PPL +0.24517248700540506.
- Step 4 GPTQ FP8: PPL 36.16874532836692, loss/token 3.5881953571778653, ratio 1.0100543900035177, delta PPL +0.3600347417608347.
- Step 8 GPTQ FP8: PPL 36.1771871800154, loss/token 3.5884287317821895, ratio 1.0102901385549232, delta PPL +0.36847659340931926.
- Result: extra recomputed-Hessian GPTQ steps were worse than one step under this setup; degradation mostly saturated by step 4.
- Local artifacts: `runs/modal_pythia31m_gsm8k_test_gptq_fp8_multistep_20260526/pythia31m_gsm8k_test_gptq_fp8_multistep_20260526/gptq_fp8_summary.json` and `gptq_layers.jsonl`.

## Pythia 31M GSM8K Test Staged Path to GPTQ FP8 Target - 2026-05-26

- User clarified the desired "multiple steps" as breaking the movement toward `Wq` into steps, not repeated full requantization.
- Added `--staged-to-wq` to the GPTQ runner. Method: compute the one-step GPTQ FP8 target `Wq` once from GSM8K train Hessians, then evaluate `W_alpha = W + alpha * (Wq - W)` for `alpha = step / gptq_steps`. The final `alpha=1.0` model exactly matches the one-step GPTQ FP8 endpoint; intermediate models are fp32/bf16 staged weights, not deployable FP8-only weights.
- Tests: `uv run pytest -q` -> 24 passed.
- Local smoke: `uv run python -m saliency.gptq_cli --output-dir runs/local_gptq_fp8_staged_smoke --model-name EleutherAI/pythia-31m --max-calibration-examples 1 --max-eval-examples 1 --calibration-batch-size 1 --eval-batch-size 1 --max-length 128 --dtype fp32 --device cpu --blocksize 64 --gptq-steps 2 --eval-steps 1,2 --staged-to-wq`.
- Full Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode gptq-fp8 --model-name EleutherAI/pythia-31m --max-calibration-examples 0 --max-eval-examples 0 --calibration-batch-size 32 --eval-batch-size 32 --max-length 512 --dtype fp32 --gptq-steps 8 --eval-steps 1,2,3,4,5,6,7,8 --staged-to-wq --run-name pythia31m_gsm8k_test_gptq_fp8_staged_to_wq_20260526`.
- Modal run URL: `https://modal.com/apps/jthomams477/main/ap-GNPgET5E6eQAi8MWimRMrq`.
- Baseline GSM8K test answer-token PPL: 35.80871058660608, loss/token 3.5781911762860052.
- Alpha 0.125: PPL 35.76881613307254, loss/token 3.577076456080015, ratio 0.9988859008637841, delta PPL -0.03989445353354171.
- Alpha 0.25: PPL 35.73753308494338, loss/token 3.5762014834246614, ratio 0.9980122852653251, delta PPL -0.07117750166270298.
- Alpha 0.375: PPL 35.739905817247404, loss/token 3.5762678745080954, ratio 0.9980785465817802, delta PPL -0.06880476935867819.
- Alpha 0.5: PPL 35.74528047652462, loss/token 3.5764182457826488, ratio 0.9982286402095364, delta PPL -0.06343011008146249.
- Alpha 0.625: PPL 35.74949520945331, loss/token 3.5765361490250362, ratio 0.9983463415414091, delta PPL -0.059215377152774806.
- Alpha 0.75: PPL 35.79262995299177, loss/token 3.57774200497822, ratio 0.9995509295545445, delta PPL -0.01608063361431533.
- Alpha 0.875: PPL 35.83186884163424, loss/token 3.578837688528637, ratio 1.000646721276717, delta PPL +0.02315825502815727.
- Alpha 1.0: PPL 35.889853855497044, loss/token 3.5804546331394405, ratio 1.0022660204056975, delta PPL +0.0811432688909619.
- Best point in this grid: alpha 0.25 with PPL 35.73753308494338, about 0.20% lower PPL than baseline. This is a fractional movement toward the quantized target, not a pure FP8 deployment format.
- Local artifacts: `runs/modal_pythia31m_gsm8k_test_gptq_fp8_staged_to_wq_20260526/pythia31m_gsm8k_test_gptq_fp8_staged_to_wq_20260526/gptq_fp8_summary.json` and `gptq_layers.jsonl`.

## Pythia 31M GSM8K Test Iterative Damped GPTQ FP8 - 2026-05-26

- User clarified the desired multi-step GPTQ experiment as doing a GPTQ/Newton-style target computation at each step and moving partway toward that step's `Wq`, not only interpolating once toward a fixed first target.
- Added `--iterative-damped-gptq` to the GPTQ runner. Method per step: snapshot current weights, collect fresh GSM8K train activation Hessians on the current model, compute that step's GPTQ FP8 target, then apply `W <- W + eta * (Wq_step - W)`. For this run, `eta = 1 / 8 = 0.125`.
- Tests: `uv run pytest -q` -> 26 passed.
- Local smoke: `uv run python -m saliency.gptq_cli --output-dir runs/local_gptq_fp8_iterative_damped_smoke --model-name EleutherAI/pythia-31m --max-calibration-examples 1 --max-eval-examples 1 --calibration-batch-size 1 --eval-batch-size 1 --max-length 128 --dtype fp32 --device cpu --blocksize 64 --gptq-steps 2 --eval-steps 1,2 --iterative-damped-gptq`.
- Full Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode gptq-fp8 --model-name EleutherAI/pythia-31m --max-calibration-examples 0 --max-eval-examples 0 --calibration-batch-size 32 --eval-batch-size 32 --max-length 512 --dtype fp32 --gptq-steps 8 --eval-steps 1,2,3,4,5,6,7,8 --iterative-damped-gptq --run-name pythia31m_gsm8k_test_gptq_fp8_iterative_damped_20260526`.
- Modal run URL: `https://modal.com/apps/jthomams477/main/ap-E6FZGX9LLyrElURamMGXbI`.
- Baseline GSM8K test answer-token PPL: 35.80871058660608, loss/token 3.5781911762860052.
- Step 1, cumulative nominal alpha 0.125: PPL 35.76881613307254, loss/token 3.577076456080015, ratio 0.9988859008637841, delta PPL -0.03989445353354171.
- Step 2, cumulative nominal alpha 0.25: PPL 35.74769585484638, loss/token 3.5764858154502335, ratio 0.9982960924657664, delta PPL -0.0610147317597054.
- Step 3, cumulative nominal alpha 0.375: PPL 35.730273949688915, loss/token 3.575998339223639, ratio 0.9978095654483994, delta PPL -0.07843663691716785.
- Step 4, cumulative nominal alpha 0.5: PPL 35.72902018105991, loss/token 3.5759632487950426, ratio 0.9977745524974032, delta PPL -0.07969040554617379.
- Step 5, cumulative nominal alpha 0.625: PPL 35.74160671165771, loss/token 3.576315464279225, ratio 0.9981260460416167, delta PPL -0.06710387494837278.
- Step 6, cumulative nominal alpha 0.75: PPL 35.74342632236778, loss/token 3.5763663731375757, ratio 0.9981768607925604, delta PPL -0.0652842642383007.
- Step 7, cumulative nominal alpha 0.875: PPL 35.75609366244897, loss/token 3.5767207067111833, ratio 0.9985306110358301, delta PPL -0.05261692415711394.
- Step 8, cumulative nominal alpha 1.0: PPL 35.76411869090474, loss/token 3.576945119593231, ratio 0.9987547193135174, delta PPL -0.0445918957013447.
- Best point in this grid: step 4 with PPL 35.72902018105991, about 0.22% lower PPL than baseline. Every evaluated damped iterative step beat baseline under this setup.
- Runtime: 111.04140377044678 seconds.
- Timing breakdown from Modal logs: baseline eval 1.68s; iterative step inner loop 102.47s total, split into 70.98s calibration/Hessian collection, 23.64s GPTQ target construction, and 7.85s per-step eval. Model/tokenizer/dataset setup and Python overhead account for about 6.89s inside the recorded 111.04s runner time. Image build was an additional 5.11s outside the runner timer.
- Per-step timing from tqdm rates: step 1 12.48s (8.63s calib, 2.87s target, 0.98s eval), step 2 12.66s (8.56s, 3.11s, 0.99s), step 3 12.64s (8.70s, 2.94s, 1.00s), step 4 13.29s (9.23s, 3.07s, 0.99s), step 5 13.05s (9.08s, 2.98s, 0.98s), step 6 12.95s (9.08s, 2.90s, 0.97s), step 7 12.92s (9.03s, 2.92s, 0.97s), step 8 12.48s (8.65s, 2.86s, 0.96s).
- Main cost: repeated full GSM8K-train calibration every damped step, about 69% of the iterative loop and about 64% of total recorded runner time.
- Local artifacts: `runs/modal_pythia31m_gsm8k_test_gptq_fp8_iterative_damped_20260526/pythia31m_gsm8k_test_gptq_fp8_iterative_damped_20260526/gptq_fp8_summary.json` and `gptq_layers.jsonl`.

## Pythia 31M GSM8K Test Diagonal Activation-Hessian GPTQ FP8 - 2026-05-26

- Added `--hessian-approximation {full,diagonal}` to the GPTQ runner. `full` keeps per-layer `X.T @ X`; `diagonal` stores only `sum(X ** 2, dim=0)` per linear input feature.
- For FP8 GPTQ quantization, the diagonal Hessian removes off-diagonal error compensation. With the current per-row FP8 scale, the one-step diagonal target is equivalent to independent per-row FP8 quantization; the diagonal `h_j` affects stored curvature/damping stats, not the quantized target choice.
- Tests: `uv run pytest -q` -> 30 passed.
- Local smoke: `uv run python -m saliency.gptq_cli --output-dir runs/local_gptq_fp8_diagonal_smoke --model-name EleutherAI/pythia-31m --max-calibration-examples 1 --max-eval-examples 1 --calibration-batch-size 1 --eval-batch-size 1 --max-length 128 --dtype fp32 --device cpu --blocksize 64 --hessian-approximation diagonal`.
- One-step Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode gptq-fp8 --model-name EleutherAI/pythia-31m --max-calibration-examples 0 --max-eval-examples 0 --calibration-batch-size 32 --eval-batch-size 32 --max-length 512 --dtype fp32 --hessian-approximation diagonal --run-name pythia31m_gsm8k_test_gptq_fp8_diag_20260526`.
- One-step Modal URL: `https://modal.com/apps/jthomams477/main/ap-7LAATqab5dgO6q6wRLzXnE`.
- One-step baseline GSM8K test answer-token PPL: 35.80871058660608, loss/token 3.5781911762860052.
- One-step diagonal FP8 PPL: 36.87940633991842, loss/token 3.6076533014850383, ratio 1.0299004274594803, delta PPL +1.0706957533123358.
- One-step runtime: 11.903778076171875 seconds; weighted mean absolute quantization error 0.0005523165613343484.
- 8-step damped diagonal Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode gptq-fp8 --model-name EleutherAI/pythia-31m --max-calibration-examples 0 --max-eval-examples 0 --calibration-batch-size 32 --eval-batch-size 32 --max-length 512 --dtype fp32 --gptq-steps 8 --eval-steps 1,2,3,4,5,6,7,8 --iterative-damped-gptq --hessian-approximation diagonal --run-name pythia31m_gsm8k_test_gptq_fp8_iterative_damped_diag_20260526`.
- 8-step damped diagonal Modal URL: `https://modal.com/apps/jthomams477/main/ap-pA6HwFbTGqCHDIRsApvgGn`.
- 8-step damped diagonal runtime: 58.609626054763794 seconds.
- 8-step diagonal timing from Modal logs: baseline eval 1.61s; iterative step inner loop 49.21s total, split into 39.28s calibration/Hessian collection, 2.24s target construction, and 7.69s per-step eval.
- 8-step diagonal per-step PPL: step 1 35.845543343943966, step 2 35.89637784506962, step 3 35.95565163153889, step 4 36.02762296175696, step 5 36.10257378098183, step 6 36.16893104061009, step 7 36.23863349673691, step 8 36.30164628250991.
- Best 8-step diagonal point: step 1 with PPL 35.845543343943966, still worse than baseline by +0.03683275733788349 PPL. Final step 8 ratio was 1.0137658041249942, delta PPL +0.49293569590382447.
- Compared to 8-step full-Hessian damped GPTQ, diagonal reduced recorded runner time from 111.04s to 58.61s, but lost the quality improvement: full-Hessian best was PPL 35.72902018105991, diagonal best was PPL 35.845543343943966.
- Local artifacts: `runs/modal_pythia31m_gsm8k_test_gptq_fp8_diag_20260526/pythia31m_gsm8k_test_gptq_fp8_diag_20260526/gptq_fp8_summary.json`, `runs/modal_pythia31m_gsm8k_test_gptq_fp8_iterative_damped_diag_20260526/pythia31m_gsm8k_test_gptq_fp8_iterative_damped_diag_20260526/gptq_fp8_summary.json`, and corresponding `gptq_layers.jsonl` files.

## Pythia 31M GSM8K Test Gradient-Descent GPTQ FP8 - 2026-05-26

- Added `--gradient-descent-gptq` as a third stepped update mode. Per step: collect the current activation Hessian, compute the current GPTQ FP8 target `Wq`, then apply one gradient descent update on the local quadratic `0.5 * (W - Wq) H (W - Wq)^T`.
- Step size: per-layer `gradient_step_scale / L`, with `gradient_step_scale = 1.0` and `L` estimated by a cheap row-sum Lipschitz bound of the damped Hessian. This is intentionally stable and conservative.
- Tests: `uv run pytest -q` -> 35 passed.
- Local smoke: `uv run python -m saliency.gptq_cli --output-dir runs/local_gptq_fp8_gradient_smoke --model-name EleutherAI/pythia-31m --max-calibration-examples 1 --max-eval-examples 1 --calibration-batch-size 1 --eval-batch-size 1 --max-length 128 --dtype fp32 --device cpu --blocksize 64 --gptq-steps 2 --eval-steps 1,2 --gradient-descent-gptq`.
- 1-step Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode gptq-fp8 --model-name EleutherAI/pythia-31m --max-calibration-examples 0 --max-eval-examples 0 --calibration-batch-size 32 --eval-batch-size 32 --max-length 512 --dtype fp32 --gradient-descent-gptq --run-name pythia31m_gsm8k_test_gptq_fp8_gd1_20260526`.
- 1-step Modal URL: `https://modal.com/apps/jthomams477/main/ap-XOBbjP7nWuASxGyccBV5qJ`.
- 1-step GD result: baseline PPL 35.80871058660608; GD PPL 35.76645718055127, ratio 0.9988200243638313, delta PPL -0.04225340605481165, loss/token 3.577010503930457.
- 1-step GD runtime: 19.628123998641968 seconds. Timing from logs: baseline eval 1.62s; step loop 12.91s, split into 9.06s calibration/Hessian, 2.85s target construction, 1.00s eval.
- 8-step Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode gptq-fp8 --model-name EleutherAI/pythia-31m --max-calibration-examples 0 --max-eval-examples 0 --calibration-batch-size 32 --eval-batch-size 32 --max-length 512 --dtype fp32 --gptq-steps 8 --eval-steps 1,2,3,4,5,6,7,8 --gradient-descent-gptq --run-name pythia31m_gsm8k_test_gptq_fp8_gd8_20260526`.
- 8-step Modal URL: `https://modal.com/apps/jthomams477/main/ap-jek1pHIeGSKLQQMqGXgng7`.
- 8-step GD runtime: 107.42492914199829 seconds. Timing from logs: baseline eval 1.63s; iterative step loop 100.42s, split into 70.12s calibration/Hessian, 22.43s target construction, and 7.88s per-step eval.
- 8-step GD per-step PPL: step 1 35.76645718055127, step 2 35.73473140099505, step 3 35.666475346382875, step 4 35.620222391445616, step 5 35.62005686124899, step 6 35.600658365528126, step 7 35.57543164110867, step 8 35.554529612462744.
- 8-step GD final result: PPL 35.554529612462744, loss/token 3.571067563391764, ratio 0.9929016998942595, delta PPL -0.2541809741433383, delta loss/token -0.007123612894241127.
- Current comparison: full-Hessian damped GPTQ best PPL 35.72902018105991 in 111.04s; diagonal damped best PPL 35.845543343943966 in 58.61s; full-Hessian GD best/final PPL 35.554529612462744 in 107.42s. Under this setup, 8-step GD gives the best PPL so far at roughly the same cost as 8-step full-Hessian damped GPTQ.
- Local artifacts: `runs/modal_pythia31m_gsm8k_test_gptq_fp8_gd1_20260526/pythia31m_gsm8k_test_gptq_fp8_gd1_20260526/gptq_fp8_summary.json`, `runs/modal_pythia31m_gsm8k_test_gptq_fp8_gd8_20260526/pythia31m_gsm8k_test_gptq_fp8_gd8_20260526/gptq_fp8_summary.json`, and corresponding `gptq_layers.jsonl` files.

## Pythia 31M GSM8K One-Step GD Step-Size Sweep - 2026-05-26

- Added a one-run `--gradient-step-scales` sweep for `--gradient-descent-gptq`. It computes the full-Hessian calibration and GPTQ FP8 target once, restores the original weights for each candidate scale, applies one GD update, then evaluates each candidate on GSM8K test.
- Step-size meaning: multiplier on the per-layer `1 / row_sum_bound(H + damp I)` learning rate used by the full-batch quadratic GD update.
- Tests: `uv run pytest -q` -> 37 passed.
- Local smoke: `uv run python -m saliency.gptq_cli --output-dir runs/local_gptq_fp8_gradient_scale_smoke --model-name EleutherAI/pythia-31m --max-calibration-examples 1 --max-eval-examples 1 --calibration-batch-size 1 --eval-batch-size 1 --max-length 128 --dtype fp32 --device cpu --blocksize 64 --gradient-descent-gptq --gradient-step-scales 0.5,1.0,2.0`.
- Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode gptq-fp8 --model-name EleutherAI/pythia-31m --max-calibration-examples 0 --max-eval-examples 0 --calibration-batch-size 32 --eval-batch-size 32 --max-length 512 --dtype fp32 --gradient-descent-gptq --gradient-step-scales 0.25,0.5,1,1.5,2,3,4,6,8,12,16 --run-name pythia31m_gsm8k_test_gptq_fp8_gd_scale_sweep_20260526`.
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-O7gKYu70DL1g4F7TsHWDuT`.
- Runtime: 31.476970434188843 seconds for baseline plus 11 one-step scale evaluations.
- Baseline GSM8K test answer-token PPL: 35.80871058660608, loss/token 3.5781911762860052.
- Scale 0.25: PPL 35.79718611574764, ratio 0.9996781657124859, delta PPL -0.011524470858439884.
- Scale 0.5: PPL 35.78487183106484, ratio 0.9993342749529173, delta PPL -0.023838755541241596.
- Scale 1.0: PPL 35.76645718055127, ratio 0.9988200243638313, delta PPL -0.04225340605481165.
- Scale 1.5: PPL 35.746237501254704, ratio 0.9982553662411194, delta PPL -0.06247308535137819.
- Scale 2.0: PPL 35.73513955184104, ratio 0.9979454430623211, delta PPL -0.07357103476504534.
- Scale 3.0: PPL 35.69211355783267, ratio 0.9967438920066275, delta PPL -0.1165970287734126.
- Scale 4.0: PPL 35.664672309832675, ratio 0.9959775631567341, delta PPL -0.14403827677340786.
- Scale 6.0: PPL 35.633595886853946, ratio 0.9951097178065486, delta PPL -0.175114699752136.
- Scale 8.0: PPL 35.616888583255445, ratio 0.994643146871019, delta PPL -0.19182200335063726.
- Scale 12.0: PPL 35.67534376763064, ratio 0.9962755760598282, delta PPL -0.13336681897543912.
- Scale 16.0: PPL 35.83732841767337, ratio 1.0007991863040717, delta PPL +0.02861783106728666.
- Best one-step scale in this grid: 8.0 with PPL 35.616888583255445. This beats scale 1.0 GD (35.76645718055127) and the best full-Hessian damped GPTQ point (35.72902018105991), but remains worse than 8-step GD (35.554529612462744).
- Local artifacts: `runs/modal_pythia31m_gsm8k_test_gptq_fp8_gd_scale_sweep_20260526/pythia31m_gsm8k_test_gptq_fp8_gd_scale_sweep_20260526/gptq_fp8_summary.json` and `gptq_layers.jsonl`.

## Pythia 1.4B Cheap Saliency Approximations - 2026-05-26

- Added `saliency/approx.py` and `saliency/approx_cli.py` for saliency artifacts compatible with the existing prune-PPL evaluator.
- Implemented cheap methods:
  - `magnitude`: `abs(weight)` for trainable 2D parameters, no calibration forward pass.
  - `wanda`: `abs(weight) * input_activation_rms` from GSM8K calibration forward hooks.
  - `dfa_gradcam`: `abs(weight * direct_feedback_alignment_gradient)` using output CE residuals, hash-projected random feedback by output width, and captured linear activations; no backprop.
- Added Modal mode `approx-saliency` with `--approx-method`.
- Tests: `uv run pytest tests/test_approx_saliency.py tests/test_prune_eval.py tests/test_saliency.py -q` -> 14 passed.
- Local smoke: `uv run python -m saliency.approx_cli --output-dir runs/local_pythia31m_dfa_gradcam_smoke2 --model-name EleutherAI/pythia-31m --method dfa_gradcam --max-examples 1 --batch-size 1 --max-length 96 --device cpu --dtype fp32 --top-k 5`.
- WANDA Modal saliency run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method wanda --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_wanda_saliency_bf16_20260526`.
- WANDA saliency runtime: 28.121448516845703 seconds, 100 examples, 13 batches, 98 matrix tensors, 1,414,004,736 scalar scores.
- WANDA 25% per-matrix prune-PPL run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_wanda_saliency_bf16_20260526 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_wanda_prune25_ppl_bf16_20260526`.
- WANDA 25% per-matrix result: baseline PPL 7.720404623083257, pruned PPL 7.805537898214442, ratio 1.011027048359181, delta loss/token +0.010966693744922651.
- Magnitude Modal saliency run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method magnitude --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_magnitude_saliency_bf16_20260526`.
- Magnitude saliency runtime: 22.188782215118408 seconds, no calibration forward pass.
- Magnitude 25% per-matrix prune-PPL run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_magnitude_saliency_bf16_20260526 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_magnitude_prune25_ppl_bf16_20260526`.
- Magnitude 25% per-matrix result: baseline PPL 7.720404623083257, pruned PPL 8.583812377432562, ratio 1.1118345211814158, delta loss/token +0.1060113728675871.
- DFA Modal saliency run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method dfa_gradcam --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 4 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_dfa_gradcam_saliency_bf16_20260526`.
- DFA saliency runtime: 114.41106271743774 seconds, 100 examples, 25 batches, 9,848 supervised tokens, 98 matrix tensors, 1,414,004,736 scalar scores.
- DFA 25% per-matrix prune-PPL run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_dfa_gradcam_saliency_bf16_20260526 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_dfa_gradcam_prune25_ppl_bf16_20260526`.
- DFA 25% per-matrix result: baseline PPL 7.720404623083257, pruned PPL 57.185697192386684, ratio 7.407085507073946, delta loss/token +2.002437043054427.
- Interpretation: WANDA is the better first cheap approximation: about 4x faster saliency generation than exact 1.4B backprop saliency (28.1s vs 110.5s) with modest 25% pruning damage. The current DFA implementation is not useful yet; CPU activation capture and dense residual projection remove the expected speed advantage, and the random direct-feedback score is much worse for pruning.

## Pythia 1.4B Row-Conditioned WANDA Saliency - 2026-05-27

- Goal: test a WANDA variant where the activation multiplier is no longer one scalar per input column. Plain per-token `|x_j|` is still shared by every row in column `j`, so to satisfy the row-different requirement the implemented score conditions input-dimension magnitude by output-row activity.
- Score definition for each linear weight matrix: `score[o, i] = abs(W[o, i]) * (sum_t abs(y[t, o]) * abs(x[t, i]) / sum_t abs(y[t, o]))`, where `x` is the linear input and `y` is the linear output from calibration forward passes. Trainable 2D non-Linear weights fall back to `abs(weight)`.
- Added `RowConditionedWandaAccumulator`, `_row_conditioned_wanda_scores`, and `--approx-method row_wanda` with aliases `token_wanda` and `row_conditioned_wanda`.
- Tests: `uv run pytest tests/test_approx_saliency.py tests/test_prune_eval.py -q` -> 22 passed. Compile check: `uv run python -m py_compile saliency/approx.py modal_pythia_saliency.py`.
- Local smoke: `uv run python -m saliency.approx_cli --output-dir runs/local_pythia31m_row_wanda_smoke --model-name EleutherAI/pythia-31m --method row_wanda --max-examples 1 --batch-size 1 --max-length 128 --dtype fp32 --device cpu --top-k 5`.
- Modal saliency run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method row_wanda --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 4 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_row_wanda_saliency_bf16_20260527`.
- Saliency Modal URL: `https://modal.com/apps/jthomams477/main/ap-3oy5PdzhD2jmyeVNfM8Q72`.
- Saliency runtime: 35.57713532447815 seconds, 100 examples, 25 batches, 98 matrix tensors, 1,414,004,736 scalar scores, 5,656,018,944 score bytes.
- Top saliency tensors by total score: `embed_out.weight` sum 2665688.5, `gpt_neox.embed_in.weight` sum 1771506.75, `gpt_neox.layers.22.attention.query_key_value.weight` sum 211324.0625.
- 25% per-matrix prune-PPL run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_row_wanda_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_row_wanda_prune25_ppl_bf16_20260527`.
- Prune-PPL Modal URL: `https://modal.com/apps/jthomams477/main/ap-Yoois5QjeJf3383uztvbwQ`.
- Result: baseline PPL 7.720404623083257, row-WANDA pruned PPL 7.85642957115383, ratio 1.017618888479481, delta loss/token +0.017465475223395366.
- Comparison on the same 25% per-matrix setup: exact backprop GradCAM-style saliency PPL 7.698485014076171, standard WANDA PPL 7.805537898214442, row-WANDA PPL 7.85642957115383, magnitude PPL 8.583812377432562, DFA GradCAM approximation PPL 57.185697192386684. Row-WANDA is better than magnitude but worse than standard WANDA and exact backprop saliency here.
- Local artifacts: `runs/modal_pythia14b_gsm8k_100_row_wanda_saliency_bf16_20260527/summary.json`, `parameter_summary.jsonl`, `runs/modal_pythia14b_gsm8k_100_row_wanda_prune25_ppl_bf16_20260527/pythia14b_gsm8k_100_row_wanda_prune25_ppl_bf16_20260527/prune_ppl_summary.json`, and `pruned_tensors.jsonl`.

## Pythia 1.4B Row-Conditioned WANDA with 500 Calibration Examples - 2026-05-27

- Goal: check whether row-WANDA was worse than standard WANDA because its row-conditioned activation estimates were noisier at 100 calibration examples.
- Kept the PPL evaluation harness comparable to the earlier saliency comparison: saliency/calibration uses GSM8K train-500, but prune-PPL evaluation uses GSM8K train-100 with 25% per-matrix pruning.
- Modal saliency run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method row_wanda --model-name EleutherAI/pythia-1.4b --max-examples 500 --batch-size 4 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_500_row_wanda_saliency_bf16_20260527`.
- Saliency Modal URL: `https://modal.com/apps/jthomams477/main/ap-KmH1eSC4uTh5s8ugYjfs4b`.
- Saliency runtime: 38.180331230163574 seconds, 500 examples, 125 batches, 98 matrix tensors, 1,414,004,736 scalar scores, 5,656,018,944 score bytes.
- Top saliency tensors by total score: `embed_out.weight` sum 2643835.0, `gpt_neox.embed_in.weight` sum 1771506.75, `gpt_neox.layers.22.attention.query_key_value.weight` sum 208751.75.
- 25% per-matrix prune-PPL run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_500_row_wanda_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_500rowwanda_eval100_prune25_ppl_bf16_20260527`.
- Prune-PPL Modal URL: `https://modal.com/apps/jthomams477/main/ap-sbrONAzLZDZMY9iOo7CDOQ`.
- Result: baseline PPL 7.720404623083257, pruned PPL 7.815054911340014, ratio 1.01225975747097, delta loss/token +0.01218521527213623.
- Comparison: increasing row-WANDA calibration from 100 to 500 examples improved PPL from 7.85642957115383 to 7.815054911340014, nearly matching standard WANDA 100-example PPL 7.805537898214442, but still behind exact backprop GradCAM-style saliency PPL 7.698485014076171.
- Local artifacts: `runs/modal_pythia14b_gsm8k_500_row_wanda_saliency_bf16_20260527/summary.json`, `parameter_summary.jsonl`, `runs/modal_pythia14b_gsm8k_500rowwanda_eval100_prune25_ppl_bf16_20260527/pythia14b_gsm8k_500rowwanda_eval100_prune25_ppl_bf16_20260527/prune_ppl_summary.json`, and `pruned_tensors.jsonl`.

## Pythia 1.4B Iterative WANDA + One-Step GPTQ-GD Repair - 2026-05-26

- Goal: combine iterative WANDA pruning with the one-step GPTQ-style GD update used in the FP8 experiments. For each pruning chunk, recompute WANDA on the current model, prune the next least-salient weights per matrix, apply one gradient-descent update toward per-layer FP8 GPTQ targets, then reapply the prune masks so repaired weights cannot unzero pruned entries.
- Implementation: added opt-in repair to `run_iterative_approx_prune_ppl_experiment` via `--repair-with-gptq-gd`, `--gradient-step-scale`, `--hessian-approximation`, `--damp-percent`, and `--blocksize`. The 1.4B run uses diagonal Hessians reusing WANDA activation sumsq statistics; full Hessians are still available but are likely too expensive for 1.4B.
- Important caveat: repair applies to `torch.nn.Linear` modules. The input embedding is pruned but not GPTQ-GD repaired; the tied/output linear `embed_out` is repaired and mask-reapplied.
- Tests: `uv run pytest tests/test_approx_saliency.py tests/test_gptq_eval.py -q` -> 20 passed.
- Local smoke: Pythia-31M, 2 GSM8K examples, 10% pruning in 5% chunks, diagonal one-step repair, scale 1.0 -> completed with masks preserved and PPL ratio 1.0441724312893044.
- Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode iterative-approx-prune-ppl --approx-method wanda --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --prune-fraction 0.25 --prune-chunk-fraction 0.05 --repair-with-gptq-gd --hessian-approximation diagonal --gradient-step-scale 8.0 --run-name pythia14b_gsm8k_100_iterative_wanda_prune25_chunk5_gptqgd_diag_s8_bf16_20260526`.
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-xaHP7uOIG7epI00hUOW80O`.
- Runtime: 306.72233486175537 seconds.
- Pruning: 25% per matrix over 98 matrix tensors, 1,414,004,736 weights seen, 353,501,184 weights zeroed. The run used 5% chunks plus a tiny integer-remainder cleanup step, so 6 repair/prune steps total.
- Repair summaries: 6 one-step repairs, 97 linear layers repaired each step. Final repair reapplied 327,745,536 zero masks to linear weights; final weighted mean absolute GD step delta was 1.3979177700743327e-05 and final weighted mean absolute remaining distance to FP8 target was 0.0003069473847989317.
- Result: baseline PPL 7.720404623083257, pruned+repaired PPL 8.622249901848443, ratio 1.1168132141764637, delta loss/token +0.11047928513403704.
- Comparison: this is much worse than one-shot WANDA 25% per-matrix pruning (PPL 7.805537898214442, ratio 1.011027048359181) and worse than iterative WANDA without repair (PPL 7.80871-ish, ratio 1.01144-ish). The diagonal FP8-target GD repair seems to move surviving weights in a direction that hurts the pruned 1.4B model under this setup; scale 8.0 from the 31M full-Hessian sweep does not transfer.
- Local artifacts: `runs/modal_pythia14b_gsm8k_100_iterative_wanda_prune25_chunk5_gptqgd_diag_s8_bf16_20260526_artifacts/pythia14b_gsm8k_100_iterative_wanda_prune25_chunk5_gptqgd_diag_s8_bf16_20260526/iterative_prune_ppl_summary.json`, `iterative_pruning_steps.jsonl`, `repair_steps.jsonl`, and `pruned_tensors.jsonl`.

## Correction: Pythia 1.4B Iterative WANDA + Loss-GD Pruning Repair - 2026-05-27

- Correction: the prior `repair_with_gptq_gd` run was the wrong objective for pruning. It stepped toward an FP8 quantization target, not toward better pruning recovery. The corrected implementation does not use GPTQ targets, FP8, or Hessians.
- Correct algorithm now implemented: recompute WANDA on the current model, prune the next least-salient chunk to zero, run one standard backprop gradient-descent step on GSM8K calibration LM loss over trainable 2D weights only, mask pruned gradients/weights so pruned entries stay exactly zero, then recompute saliency.
- Added flags: `--repair-with-loss-gd` and `--repair-learning-rate`. The old `repair_with_gptq_gd` path now raises because it was the wrong objective for this pruning experiment.
- Added heldout eval support for iterative pruning: calibration can use GSM8K train while PPL is measured on GSM8K test.
- Tests: `uv run pytest tests/test_approx_saliency.py tests/test_gptq_eval.py -q` -> 22 passed. Compile check: `uv run python -m py_compile saliency/approx.py modal_pythia_saliency.py`.
- Local smoke: Pythia-31M CPU, 1 GSM8K example, 2% WANDA pruning, one loss-GD repair step with lr 1e-3 -> completed with method `one_step_loss_gradient_descent` and nonzero update summary.
- In-sample Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode iterative-approx-prune-ppl --approx-method wanda --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --prune-fraction 0.25 --prune-chunk-fraction 0.05 --repair-with-loss-gd --repair-learning-rate 0.001 --run-name pythia14b_gsm8k_100_iterative_wanda_prune25_chunk5_lossgd_lr1e3_bf16_20260527`.
- In-sample Modal URL: `https://modal.com/apps/jthomams477/main/ap-Nm4iEi59NvhUmW3SEgmdFj`.
- In-sample result: baseline PPL 7.720404623083257, pruned+loss-GD PPL 5.765130455180887, ratio 0.7467394180278724, delta loss/token -0.2920389926888711. This directly optimizes and evaluates the same train-100 examples, so it is a sanity check rather than the clean comparison.
- Heldout Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode iterative-approx-prune-ppl --approx-method wanda --model-name EleutherAI/pythia-1.4b --max-examples 100 --eval-split test --max-eval-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --prune-fraction 0.25 --prune-chunk-fraction 0.05 --repair-with-loss-gd --repair-learning-rate 0.001 --run-name pythia14b_gsm8k_train100_test100_iterative_wanda_prune25_chunk5_lossgd_lr1e3_bf16_20260527`.
- Heldout Modal URL: `https://modal.com/apps/jthomams477/main/ap-NrdwzErQOXL3qvquhwUfjp`.
- Heldout result: train-100 calibration/repair, test-100 PPL eval. Baseline PPL 8.30205273392189, pruned+loss-GD PPL 6.190560858736999, ratio 0.745666289668649, delta loss/token -0.29347711174542734.
- Runtime: 181.37535881996155 seconds for the heldout run, versus 306.72233486175537 seconds for the wrong diagonal FP8/GPTQ repair run.
- Repair details: 6 prune/repair steps because of the same 5% chunks plus tiny integer-remainder cleanup. Each step updated 98 trainable 2D tensors and reapplied masks. Final repair mean absolute step delta was 6.933407281273234e-08, max step delta 0.0003789067268371582, grad L2 norm 8.472995424417787.
- Local artifacts: `runs/modal_pythia14b_gsm8k_100_iterative_wanda_prune25_lossgd_lr1e3_bf16_20260527/pythia14b_gsm8k_100_iterative_wanda_prune25_chunk5_lossgd_lr1e3_bf16_20260527/iterative_prune_ppl_summary.json` and `runs/modal_pythia14b_gsm8k_train100_test100_iterative_wanda_prune25_lossgd_lr1e3_bf16_20260527/pythia14b_gsm8k_train100_test100_iterative_wanda_prune25_chunk5_lossgd_lr1e3_bf16_20260527/iterative_prune_ppl_summary.json`, with corresponding `iterative_pruning_steps.jsonl`, `repair_steps.jsonl`, and `pruned_tensors.jsonl`.

## Pythia 1.4B Iterative WANDA + Loss-GD Repair at 50% Pruning - 2026-05-27

- Ran the same corrected pruning repair setup as above, but with 50% per-matrix pruning. Calibration/repair used GSM8K train-100; PPL eval used GSM8K test-100. WANDA was recomputed every 5% chunk and one standard masked loss-GD step was applied after each chunk with lr 1e-3.
- Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode iterative-approx-prune-ppl --approx-method wanda --model-name EleutherAI/pythia-1.4b --max-examples 100 --eval-split test --max-eval-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --prune-fraction 0.50 --prune-chunk-fraction 0.05 --repair-with-loss-gd --repair-learning-rate 0.001 --run-name pythia14b_gsm8k_train100_test100_iterative_wanda_prune50_chunk5_lossgd_lr1e3_bf16_20260527`.
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-wTklyfNGqiOLYyKVSGNhIO`.
- Runtime: 528.9659698009491 seconds.
- Pruning: 50% per matrix over 98 trainable 2D tensors, 1,414,004,736 weights seen, 707,002,368 weights zeroed. The run used 10 full 5% chunks plus a tiny integer-remainder cleanup, so 11 prune/repair steps total.
- Result on GSM8K test-100: baseline PPL 8.30205273392189, pruned+loss-GD PPL 8.624110205374885, ratio 1.0387925109337213, delta loss/token +0.03805899143672731.
- Comparison to the 25% heldout run: 25% pruning improved test-100 PPL from 8.30205273392189 to 6.190560858736999 (ratio 0.745666289668649), while 50% pruning degrades to 8.624110205374885 (ratio 1.0387925109337213). The loss-GD repair is useful at 25% under this setup but does not fully recover 50% sparsity.
- Final repair step: mean absolute step delta 6.453741670865733e-08, max step delta 0.00040039047598838806, grad L2 norm 11.398701991909304.
- Local artifacts: `runs/modal_pythia14b_gsm8k_train100_test100_iterative_wanda_prune50_lossgd_lr1e3_bf16_20260527/pythia14b_gsm8k_train100_test100_iterative_wanda_prune50_chunk5_lossgd_lr1e3_bf16_20260527/iterative_prune_ppl_summary.json`, `iterative_pruning_steps.jsonl`, `repair_steps.jsonl`, and `pruned_tensors.jsonl`.

## Pythia 1.4B Direct 2:4 WANDA + Loss-GD Repair Baseline - 2026-05-27

- Implemented direct native row-wise 2:4 WANDA pruning with no column permutation. For every trainable 2D weight matrix, each contiguous input-dimension quartet gets one lowest-WANDA entry zeroed in step 1, then WANDA is recomputed and the next lowest remaining entry is zeroed in step 2. After each structured pruning step, one standard masked loss-GD repair step is applied on GSM8K train-100 with lr 1e-3, and pruned masks are reapplied.
- Added `--pruning-structure`, `--structured-n`, `--structured-m`, and `--structured-group-dim` to the iterative pruning Modal path. Existing unstructured pruning behavior is unchanged.
- Tests: `uv run pytest tests/test_approx_saliency.py tests/test_gptq_eval.py -q` -> 25 passed. Compile check: `uv run python -m py_compile saliency/approx.py modal_pythia_saliency.py`.
- Local smoke: Pythia-31M CPU, GSM8K train/test 1 example, `--pruning-structure 2:4`, two repair steps, final zero fraction 0.5. A too-short `max_length=64` smoke had no usable answer tokens; rerun with `max_length=256` completed.
- Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode iterative-approx-prune-ppl --approx-method wanda --model-name EleutherAI/pythia-1.4b --max-examples 100 --eval-split test --max-eval-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --pruning-structure 2:4 --structured-n 2 --structured-m 4 --structured-group-dim 1 --repair-with-loss-gd --repair-learning-rate 0.001 --run-name pythia14b_gsm8k_train100_test100_2to4_wanda_lossgd_lr1e3_bf16_20260527`.
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-hGJ6ycqsOKixQJva60H6n2`.
- Runtime: 88.46253991127014 seconds.
- Pruning: 2 structured WANDA/recompute chunks over 98 trainable 2D tensors, 1,414,004,736 weights seen, 707,002,368 weights zeroed, final zero fraction 0.5. Step 1 zeroed 353,501,184 weights, exactly one per quartet. Step 2 zeroed another 353,501,184 weights, exactly two per quartet total.
- Repair details: step 1 reapplied 353,501,184 zero masks, loss/token 2.122905988406593, grad L2 norm 13.2336144293521, mean abs step delta 9.872365841874988e-08. Step 2 reapplied 707,002,368 zero masks, loss/token 3.4458676918457427, grad L2 norm 30.16972354170038, mean abs step delta 1.480887506869629e-07.
- Result on GSM8K test-100: baseline PPL 8.30205273392189, direct 2:4 WANDA + loss-GD repair PPL 21.67180839154333, ratio 2.6104156509381222, delta loss/token +0.9595094618881492.
- Comparison: this is much worse than the unstructured 50% WANDA + loss-GD repair run on the same train/test split, which reached PPL 8.624110205374885. The native quartet constraint is the damaging factor here, not merely the 50% zero count.
- Local artifacts: `runs/modal_pythia14b_gsm8k_train100_test100_2to4_wanda_lossgd_lr1e3_bf16_20260527/pythia14b_gsm8k_train100_test100_2to4_wanda_lossgd_lr1e3_bf16_20260527/iterative_prune_ppl_summary.json`, `iterative_pruning_steps.jsonl`, `repair_steps.jsonl`, and `pruned_tensors.jsonl`.

## Pythia 1.4B 2:4 WANDA Per-Matrix Attribution - 2026-05-27

- Goal: store and evaluate after each trainable 2D weight matrix to identify which matrices cause the largest marginal PPL changes under native row-wise 2:4 WANDA pruning with loss-GD repair.
- Attribution semantics: cumulative in model parameter order, not independent leave-one-matrix ablation. For each matrix, WANDA is recomputed on the current model, one lowest-saliency entry per native quartet is pruned, one masked loss-GD repair step is run, WANDA is recomputed, the second entry per quartet is pruned, another masked loss-GD repair step is run, then GSM8K test-100 PPL is evaluated and written to `matrix_attribution.jsonl`.
- Important caveat: this does not reproduce the direct 2:4 baseline failure because it runs repair after every matrix, for 196 masked loss-GD repair steps total. The direct 2:4 baseline ran only two repair steps, one after each global structured pruning pass.
- Added `--mode nm-matrix-attribution` plus `--matrix-limit` and the helper `run_nm_matrix_attribution_experiment`. Rows are streamed to disk after every matrix; summary stores compact rows and top marginal PPL increases.
- Tests: `uv run pytest tests/test_approx_saliency.py tests/test_gptq_eval.py -q` -> 26 passed. Compile check: `uv run python -m py_compile saliency/approx.py modal_pythia_saliency.py`.
- Local smoke: Pythia-31M CPU, GSM8K train/test 1 example, `--pruning-structure 2:4`, `--matrix-limit 1`, loss-GD repair enabled -> completed and wrote `matrix_attribution_summary.json`.
- Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode nm-matrix-attribution --approx-method wanda --model-name EleutherAI/pythia-1.4b --max-examples 100 --eval-split test --max-eval-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --pruning-structure 2:4 --structured-n 2 --structured-m 4 --structured-group-dim 1 --repair-with-loss-gd --repair-learning-rate 0.001 --run-name pythia14b_gsm8k_train100_test100_2to4_matrix_attr_lossgd_lr1e3_bf16_20260527`.
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-sfXYlIehsld8xhLGKBFqf6`.
- Runtime: 1572.3538439273834 seconds for 98 matrices.
- Final cumulative result under this per-matrix repair schedule: baseline PPL 8.30205273392189, final PPL 5.262681352456852, ratio 0.6339012195084867, delta loss/token -0.45586214187546226, final zero fraction 0.5.
- Largest positive marginal PPL changes: `embed_out.weight` +0.585677108116796; `gpt_neox.layers.9.attention.query_key_value.weight` +0.11052745532639552; `gpt_neox.layers.9.mlp.dense_h_to_4h.weight` +0.101792455868992; `gpt_neox.layers.23.mlp.dense_h_to_4h.weight` +0.09748019897407278; `gpt_neox.layers.10.mlp.dense_h_to_4h.weight` +0.09683502488218476; `gpt_neox.layers.12.attention.query_key_value.weight` +0.08796719858123492; `gpt_neox.layers.10.attention.query_key_value.weight` +0.08396898097248151.
- Module-type sums of marginal PPL changes: `embed_out` +0.585677108116796, `mlp_up` +0.1685388159801562, `attn_qkv` -0.6969707377785008, `mlp_down` -0.7725601923308822, `embed_in` -1.1118138944232214, `attn_out` -1.212242481029386.
- Local artifacts: `runs/modal_pythia14b_gsm8k_train100_test100_2to4_matrix_attr_lossgd_lr1e3_bf16_20260527/pythia14b_gsm8k_train100_test100_2to4_matrix_attr_lossgd_lr1e3_bf16_20260527/matrix_attribution_summary.json` and `matrix_attribution.jsonl`.

## Pythia 1.4B 2:4 WANDA Original-Cadence Matrix Attribution - 2026-05-27

- Correction to the attribution above: to identify the cause of the direct 2:4 baseline failure, attribution must preserve the original repair cadence. This run recomputes WANDA once for pass 1, prunes one entry per quartet one matrix at a time with PPL eval after every matrix, repairs once after the full pass, recomputes WANDA once for pass 2, prunes the second entry per quartet one matrix at a time with PPL eval after every matrix, then repairs once at the end.
- Added `--mode nm-global-pass-matrix-attribution`, `run_nm_global_pass_matrix_attribution_experiment`, streamed `global_pass_matrix_attribution.jsonl`, and `global_pass_repair_checkpoints.jsonl`.
- Tests: `uv run pytest tests/test_approx_saliency.py tests/test_gptq_eval.py -q` -> 27 passed. Compile check: `uv run python -m py_compile saliency/approx.py modal_pythia_saliency.py`.
- Local smoke: Pythia-31M CPU, GSM8K train/test 1 example, `--pruning-structure 2:4`, `--matrix-limit 1`, loss-GD repair enabled -> 2 per-matrix rows and 2 repair checkpoints written.
- Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode nm-global-pass-matrix-attribution --approx-method wanda --model-name EleutherAI/pythia-1.4b --max-examples 100 --eval-split test --max-eval-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --pruning-structure 2:4 --structured-n 2 --structured-m 4 --structured-group-dim 1 --repair-with-loss-gd --repair-learning-rate 0.001 --run-name pythia14b_gsm8k_train100_test100_2to4_global_pass_matrix_attr_lossgd_lr1e3_bf16_20260527`.
- Modal URL: `https://modal.com/apps/jthomams477/main/ap-nZWMpoIbWUUzoKEINrQT85`.
- Runtime: 645.135502576828 seconds.
- This reproduces the bad direct 2:4 result: baseline PPL 8.30205273392189, final repaired PPL 21.63518128140729, ratio 2.6060038372204883, delta loss/token +0.9578179511576277. The earlier direct 2:4 run was PPL 21.67180839154333.
- Repair checkpoints: after pass 1 and repair, PPL 8.281014917550118 at 25% zero fraction; after pass 2 and repair, PPL 21.63518128140729 at 50% zero fraction. Immediately before the final repair, after pass 2 `embed_out.weight`, PPL was 33.842768435284626.
- Largest marginal PPL increases: pass 2 `embed_out.weight` +8.777524412231429; pass 2 `gpt_neox.layers.15.attention.query_key_value.weight` +1.0826945416529519; pass 2 `gpt_neox.layers.11.attention.query_key_value.weight` +1.0571783637626169; pass 2 `gpt_neox.layers.8.attention.query_key_value.weight` +0.9823573428307579; pass 2 `gpt_neox.layers.9.mlp.dense_h_to_4h.weight` +0.9305309381026987; pass 2 `gpt_neox.layers.10.attention.query_key_value.weight` +0.8114360501315225; pass 2 `gpt_neox.layers.7.attention.query_key_value.weight` +0.7788934657558837.
- Pass 1 was not the main issue: after pass 1 `embed_out.weight`, PPL was 9.019481304350982, and the pass-1 repair brought it back to 8.281014917550118. The failure happens during pass 2, especially the output embedding / LM head and mid-layer QKV matrices.
- Module-type sums over both passes: `attn_qkv` +10.207238128214483, `embed_out` +8.940050177019772, `mlp_up` +3.0427445322893067, `mlp_down` +2.0483282430553587, `attn_out` +1.9666804866841368, `embed_in` +0.07414052090054213.
- Local artifacts: `runs/modal_pythia14b_gsm8k_train100_test100_2to4_global_pass_matrix_attr_lossgd_lr1e3_bf16_20260527/pythia14b_gsm8k_train100_test100_2to4_global_pass_matrix_attr_lossgd_lr1e3_bf16_20260527/global_pass_matrix_attribution_summary.json`, `global_pass_matrix_attribution.jsonl`, and `global_pass_repair_checkpoints.jsonl`.

## Pythia 1.4B Standard WANDA with 500 Calibration Examples - 2026-05-27

- Goal: run standard WANDA with the same larger 500-example GSM8K calibration set used for the row-conditioned WANDA follow-up, then evaluate 25% per-matrix pruning on the existing 100-example answer-token PPL harness.
- Saliency Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method wanda --model-name EleutherAI/pythia-1.4b --max-examples 500 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_500_wanda_saliency_bf16_20260527`.
- Saliency Modal URL: `https://modal.com/apps/jthomams477/main/ap-fDvcHIyd7W6FBteLe8v72I`.
- Saliency runtime: 30.08710026741028 seconds over 500 train examples / 63 batches. Total saliency sum was 14374647.001953125 and score storage was 5656018944 bytes.
- Top saliency tensors by sum: `embed_out.weight` 3120768.0, `gpt_neox.embed_in.weight` 1771506.75, `gpt_neox.layers.22.attention.query_key_value.weight` 231221.234375, `gpt_neox.layers.23.attention.query_key_value.weight` 223685.140625, `gpt_neox.layers.20.attention.query_key_value.weight` 203182.375.
- Prune/eval Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_500_wanda_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_500wanda_eval100_prune25_ppl_bf16_20260527`.
- Prune/eval Modal URL: `https://modal.com/apps/jthomams477/main/ap-YsUGSxmb9HCj1YAHECD38q`.
- Prune/eval runtime: 61.1260290145874 seconds. Pruning was 25% per matrix over 98 matrices: 1,414,004,736 weights seen, 353,501,184 zeroed, actual zero fraction 0.25.
- Result: baseline PPL 7.720404623083257, pruned PPL 7.7612781829659365, ratio 1.0052942250928756, delta PPL +0.04087355988267927, delta loss/token +0.005280259951259136.
- Comparison: this improves over standard WANDA with 100 calibration examples (PPL 7.805537898214442, ratio 1.011027048359181) and over row-conditioned WANDA with 500 calibration examples (PPL 7.815054911340014, ratio 1.01225975747097). It remains slightly worse than the exact backprop GradCAM-style saliency 100-example run (PPL 7.698485014076171, ratio 0.9971608212163455).
- Local artifacts: `runs/modal_pythia14b_gsm8k_500_wanda_saliency_bf16_20260527/summary.json`, `runs/modal_pythia14b_gsm8k_500_wanda_saliency_bf16_20260527/parameter_summary.jsonl`, and `runs/modal_pythia14b_gsm8k_500wanda_eval100_prune25_ppl_bf16_20260527/pythia14b_gsm8k_500wanda_eval100_prune25_ppl_bf16_20260527/prune_ppl_summary.json`.

## Pythia 1.4B RIA Saliency with 100 Calibration Examples - 2026-05-27

- Goal: test RIA as a channel-balanced WANDA-like saliency baseline on the original 100-example GSM8K calibration setup.
- Implemented `--approx-method ria` with score `(|w_ij| / sum_i |w_ij| + |w_ij| / sum_j |w_ij|) * activation_rms_j^0.5`, where high score means keep. The activation term reuses the same input-column RMS collected by WANDA. Trainable 2D non-Linear weights use the relative-importance term without activation scaling.
- Tests: `uv run pytest tests/test_approx_saliency.py tests/test_prune_eval.py -q` -> 24 passed. Compile check: `uv run python -m py_compile saliency/approx.py saliency/approx_cli.py modal_pythia_saliency.py`.
- Local smoke: `uv run python -m saliency.approx_cli --output-dir runs/local_pythia31m_ria_smoke --model-name EleutherAI/pythia-31m --method ria --max-examples 1 --batch-size 1 --max-length 128 --dtype fp32 --device cpu --top-k 5`.
- Saliency Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method ria --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_ria_saliency_bf16_20260527`.
- Saliency Modal URL: `https://modal.com/apps/jthomams477/main/ap-i8jOcLvVVgRaA3hBQpK1Ns`.
- Saliency runtime: 34.823912620544434 seconds over 100 train examples / 13 batches. Total saliency sum was 614519.6011962891 and score storage was 5656018944 bytes.
- Top saliency tensors by sum: `embed_out.weight` 68882.703125, `gpt_neox.embed_in.weight` 52352.0, `gpt_neox.layers.15.mlp.dense_h_to_4h.weight` 8334.05078125, `gpt_neox.layers.16.mlp.dense_h_to_4h.weight` 8226.4208984375, `gpt_neox.layers.14.mlp.dense_h_to_4h.weight` 8211.875.
- Prune/eval Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_ria_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_ria_prune25_ppl_bf16_20260527`.
- Prune/eval Modal URL: `https://modal.com/apps/jthomams477/main/ap-jkrZUZhoX32z2M4PiGnsXG`.
- Prune/eval runtime: 39.27290368080139 seconds. Pruning was 25% per matrix over 98 matrices: 1,414,004,736 weights seen, 353,501,184 zeroed, actual zero fraction 0.25.
- Result: baseline PPL 7.720404623083257, RIA-pruned PPL 7.837306425464192, ratio 1.0151419268922006, delta PPL +0.1169018023809345, delta loss/token +0.015028432168968209.
- Comparison on the same 100-calibration, 25% per-matrix setup: exact backprop GradCAM-style saliency PPL 7.698485014076171, standard WANDA PPL 7.805537898214442, RIA PPL 7.837306425464192, row-WANDA PPL 7.85642957115383, magnitude PPL 8.583812377432562, DFA GradCAM approximation PPL 57.185697192386684. RIA beats row-WANDA and magnitude, but is worse than standard WANDA and exact backprop saliency here.
- Local artifacts: `runs/modal_pythia14b_gsm8k_100_ria_saliency_bf16_20260527/summary.json`, `runs/modal_pythia14b_gsm8k_100_ria_saliency_bf16_20260527/parameter_summary.jsonl`, and `runs/modal_pythia14b_gsm8k_100_ria_prune25_ppl_bf16_20260527/pythia14b_gsm8k_100_ria_prune25_ppl_bf16_20260527/prune_ppl_summary.json`.

## Pythia 1.4B Output-Perturbation Activation-Stat Saliency Sweep - 2026-05-27

- Goal: test local reconstruction / output-perturbation saliency variants on the original Pythia-1.4B bf16, GSM8K train-100 calibration, 25% per-matrix pruning setup, using parallel Modal GPU jobs where possible.
- Implemented new approximation methods:
  - `output_l2`: `w_ij^2 * sum_t x_tj^2`, matching squared local output damage. Non-Linear 2D weights fall back to `w^2`.
  - `mean_abs_wanda`: `abs(w_ij) * mean_t abs(x_tj)`.
  - `var_output`: `w_ij^2 * Var_t(x_tj)`.
  - `q95_wanda`: `abs(w_ij) * nearest_rank_quantile_0.95(abs(x_tj))`, using streaming top-tail retention to avoid materializing all activations.
  - `max_wanda`: `abs(w_ij) * max_t abs(x_tj)`.
- Tests: `uv run pytest tests/test_approx_saliency.py tests/test_prune_eval.py -q` -> 26 passed. Compile check: `uv run python -m py_compile saliency/approx.py saliency/approx_cli.py modal_pythia_saliency.py`.
- Local smokes: `output_l2` and `q95_wanda` on Pythia-31M CPU with 1 GSM8K example both completed and wrote saliency artifacts.
- Saliency Modal runs launched in parallel:
  - `output_l2`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method output_l2 --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_output_l2_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-pueandf535973yN4MV77Is`.
  - `mean_abs_wanda`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method mean_abs_wanda --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_mean_abs_wanda_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-w4dEm5aYH2vMFERo9MPt1y`.
  - `var_output`: first client `ap-T4bg2hB4DFYLb2XQ4LWr0D` stalled after CUDA startup and was killed locally after no progress; clean rerun command used `--run-name pythia14b_gsm8k_100_var_output_saliency_bf16_20260527_rerun`, URL `https://modal.com/apps/jthomams477/main/ap-mX6RF9cKpOH5ioAYsLLRg2`.
  - `q95_wanda`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method q95_wanda --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_q95_wanda_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-u2KIv4lkwor0jByUdllvZ8`.
  - `max_wanda`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method max_wanda --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_max_wanda_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-fY9TgfDcAeCnfWYX6h4vwU`.
- Results on 25% per-matrix pruning, GSM8K train-100 answer-token PPL:
  - `output_l2`: saliency 31.11368751525879s, baseline PPL 7.720404623083257, pruned PPL 7.796032474711701, ratio 1.0097958404151932, delta loss/token +0.009748172217709072.
  - `mean_abs_wanda`: saliency 29.314494371414185s, pruned PPL 7.824583528219231, ratio 1.0134939695808802, delta loss/token +0.013403736799349808.
  - `var_output`: saliency 38.80336022377014s, pruned PPL 7.939837190750913, ratio 1.0284224180442012, delta loss/token +0.02802599512591364.
  - `q95_wanda`: saliency 722.5027174949646s, pruned PPL 8.542075932901668, ratio 1.1064285293236695, delta loss/token +0.10113728675873235.
  - `max_wanda`: saliency 29.177189826965332s, pruned PPL 7.881999701820093, ratio 1.0209309079803515, delta loss/token +0.020714865962631723.
- Comparison to existing baselines on this same one-shot 25% per-matrix harness: exact backprop GradCAM-style saliency PPL 7.698485014076171, standard WANDA PPL 7.805537898214442, output_l2 PPL 7.796032474711701, mean_abs_wanda PPL 7.824583528219231, RIA PPL 7.837306425464192, row-WANDA PPL 7.85642957115383, max_wanda PPL 7.881999701820093, var_output PPL 7.939837190750913, magnitude PPL 8.583812377432562, q95_wanda PPL 8.542075932901668, DFA PPL 57.185697192386684.
- Interpretation: the squared local output-damage score is the only new variant that beat standard WANDA here, and it did so slightly. Mean-abs is close but worse than WANDA. Outlier-aware q95/max did not help; q95 was both very slow under exact streaming top-tail retention and nearly as bad as magnitude-only pruning.
- Local artifacts:
  - Saliency summaries: `runs/modal_pythia14b_gsm8k_100_output_l2_saliency_bf16_20260527/summary.json`, `runs/modal_pythia14b_gsm8k_100_mean_abs_wanda_saliency_bf16_20260527/summary.json`, `runs/modal_pythia14b_gsm8k_100_var_output_saliency_bf16_20260527_rerun/summary.json`, `runs/modal_pythia14b_gsm8k_100_q95_wanda_saliency_bf16_20260527/summary.json`, `runs/modal_pythia14b_gsm8k_100_max_wanda_saliency_bf16_20260527/summary.json`.
  - Prune summaries: `runs/modal_pythia14b_gsm8k_100_output_l2_prune25_ppl_bf16_20260527/pythia14b_gsm8k_100_output_l2_prune25_ppl_bf16_20260527/prune_ppl_summary.json`, `runs/modal_pythia14b_gsm8k_100_mean_abs_wanda_prune25_ppl_bf16_20260527/pythia14b_gsm8k_100_mean_abs_wanda_prune25_ppl_bf16_20260527/prune_ppl_summary.json`, `runs/modal_pythia14b_gsm8k_100_var_output_prune25_ppl_bf16_20260527/pythia14b_gsm8k_100_var_output_prune25_ppl_bf16_20260527/prune_ppl_summary.json`, `runs/modal_pythia14b_gsm8k_100_q95_wanda_prune25_ppl_bf16_20260527/pythia14b_gsm8k_100_q95_wanda_prune25_ppl_bf16_20260527/prune_ppl_summary.json`, and `runs/modal_pythia14b_gsm8k_100_max_wanda_prune25_ppl_bf16_20260527/pythia14b_gsm8k_100_max_wanda_prune25_ppl_bf16_20260527/prune_ppl_summary.json`.

## Pythia 1.4B Angular Saliency Sweep - 2026-05-27

- Goal: test three angular output-change saliency variants on the original Pythia-1.4B bf16, GSM8K train-100 calibration, 25% per-matrix pruning harness, running the methods in parallel on Modal.
- Implemented `AngularActivationAccumulator`, collecting per-Linear `sum_t x_tj^2`, `sum_t y_ti^2`, and `sum_t y_ti x_tj`. Non-Linear trainable 2D weights fall back to `weight^2`.
- Methods:
  - `angular_exact`: `1 - cos(y_i, y_i - w_ij x_j)`.
  - `angular_approx`: `w_ij^2 / ||y_i||^2 * (||x_j||^2 - (y_i^T x_j)^2 / ||y_i||^2)`.
  - `angular_hybrid`: `w_ij^2 / ||y_i||^2 * (||x_j||^2 - 0.5 * (y_i^T x_j)^2 / ||y_i||^2)`.
- Tests: `uv run pytest tests/test_approx_saliency.py tests/test_prune_eval.py -q` -> 27 passed. Compile check: `uv run python -m py_compile saliency/approx.py saliency/approx_cli.py modal_pythia_saliency.py`.
- Local smokes: Pythia-31M CPU, 1 GSM8K example, `angular_exact`, `angular_approx`, and `angular_hybrid` all completed and wrote saliency artifacts.
- Saliency Modal runs launched in parallel:
  - `angular_exact`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method angular_exact --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_angular_exact_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-PGxbTkItXr2lrc0hVujqFF`.
  - `angular_approx`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method angular_approx --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_angular_approx_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-4vhUvgZs88lFnwHkEsR37u`.
  - `angular_hybrid`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method angular_hybrid --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_angular_hybrid_l05_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-0KWItrEYFUz2xu0VWuCLk8`.
- Saliency runtimes: `angular_exact` 26.765637397766113s, `angular_approx` 26.322875261306763s, `angular_hybrid` 25.690593004226685s. Each wrote 5,656,018,944 bytes of float32 scores.
- Prune/eval Modal runs launched in parallel:
  - `angular_exact`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_angular_exact_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_angular_exact_prune25_ppl_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-DKdF1SkVlhwc4P7kgV66Lq`.
  - `angular_approx`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_angular_approx_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_angular_approx_prune25_ppl_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-IPiZxpGoWCzWRkE9N0dc9G`.
  - `angular_hybrid`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_angular_hybrid_l05_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_angular_hybrid_l05_prune25_ppl_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-mFZNCiJRSa2pZynOJqlVmE`.
- Results on 25% per-matrix pruning, GSM8K train-100 answer-token PPL:
  - `angular_exact`: baseline PPL 7.720404623083257, pruned PPL 2328.6554125258267, ratio 301.62349335465836, delta PPL +2320.9350079027436, delta loss/token +5.709179528838343.
  - `angular_approx`: baseline PPL 7.720404623083257, pruned PPL 2485.0153575264576, ratio 321.8763107436758, delta PPL +2477.2949529033745, delta loss/token +5.774167343623071.
  - `angular_hybrid`: baseline PPL 7.720404623083257, pruned PPL 1472.7434156203167, ratio 190.75987432277284, delta PPL +1465.0230109972333, delta loss/token +5.251015434606011.
- Interpretation: these angular variants are not viable as direct low-score pruning metrics in this harness. Exact angular likely under-penalizes removals that preserve direction but destroy magnitude; the normalized approximate/hybrid forms still perform far worse than WANDA, output_l2, magnitude, and even the DFA approximation.
- Local downloaded summaries: `runs/modal_angular_20260527/angular_exact_saliency_summary.json`, `runs/modal_angular_20260527/angular_approx_saliency_summary.json`, `runs/modal_angular_20260527/angular_hybrid_l05_saliency_summary.json`, `runs/modal_angular_20260527/angular_exact_prune25_ppl_summary.json`, `runs/modal_angular_20260527/angular_approx_prune25_ppl_summary.json`, and `runs/modal_angular_20260527/angular_hybrid_l05_prune25_ppl_summary.json`.

## Pythia 1.4B RI-Only and Angular Hybrid Lambda Follow-Up - 2026-05-27

- Goal: finish the RI-only baseline and test angular hybrid with progressively less angular emphasis than lambda 0.5.
- RI-only saliency definition: `abs(w_ij) / input-channel L1 + abs(w_ij) / output-channel L1`, with no activation term and no calibration forward pass.
- RI-only saliency Modal run was already present on the volume from the interrupted run: `pythia14b_gsm8k_100_ri_saliency_bf16_20260527`. Saliency runtime 21.29844379425049s, `num_examples` 0, score storage 5,656,018,944 bytes.
- RI-only prune/eval Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_ri_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_ri_prune25_ppl_bf16_20260527`.
- RI-only prune/eval Modal URL: `https://modal.com/apps/jthomams477/main/ap-8vPX35ymDbk0AfPSHQ9lGa`.
- RI-only result: baseline PPL 7.720404623083257, pruned PPL 8.272330972835158, ratio 1.0714893035659938, delta PPL +0.5519263497519011, delta loss/token +0.06904955320877315.
- Added `ApproxSaliencyConfig.angular_hybrid_lambda`, `--angular-hybrid-lambda` to the local CLI and Modal entrypoint, and recorded the lambda in saliency metadata. Tests: `uv run pytest tests/test_approx_saliency.py tests/test_prune_eval.py -q` -> 28 passed. Compile check: `uv run python -m py_compile modal_pythia_saliency.py saliency/approx.py saliency/approx_cli.py`.
- Local smoke: `uv run python -m saliency.approx_cli --output-dir runs/local_pythia31m_angular_hybrid_l025_smoke --model-name EleutherAI/pythia-31m --method angular_hybrid --angular-hybrid-lambda 0.25 --max-examples 1 --batch-size 1 --max-length 128 --dtype fp32 --device cpu --top-k 5`.
- Modal flag-binding note: first lambda sweep used the new flag but the Modal local entrypoint accidentally passed it to the wrong branch, so the runs named without `_fixed_` (`l025`, `l01`, `l0`) all duplicated lambda 0.5 and should be ignored. The corrected runs below use `_fixed_` names and have verified metadata lambdas.
- Corrected saliency Modal runs launched in parallel:
  - lambda 0.25: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method angular_hybrid --angular-hybrid-lambda 0.25 --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_angular_hybrid_l025_fixed_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-E0rSZhWvuwyoh8TQHbYLQy`.
  - lambda 0.1: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method angular_hybrid --angular-hybrid-lambda 0.1 --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_angular_hybrid_l01_fixed_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-radnZWMhBjSy0gIxnrMbFn`.
  - lambda 0.0: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method angular_hybrid --angular-hybrid-lambda 0.0 --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_angular_hybrid_l0_fixed_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-M6z9WQrfoWaNPnfwSczhVE`.
- Corrected saliency runtimes: lambda 0.25 25.708048582077026s, lambda 0.1 26.94409441947937s, lambda 0.0 28.886515617370605s.
- Corrected prune/eval Modal runs launched in parallel:
  - lambda 0.25: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_angular_hybrid_l025_fixed_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_angular_hybrid_l025_fixed_prune25_ppl_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-YAp5HM3YtbXWUKfGmuOeSG`.
  - lambda 0.1: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_angular_hybrid_l01_fixed_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_angular_hybrid_l01_fixed_prune25_ppl_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-3a7tF4V5I4d0TG9ITe4xo4`.
  - lambda 0.0: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_angular_hybrid_l0_fixed_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_angular_hybrid_l0_fixed_prune25_ppl_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-5wY6lOTeZfXdCJTyVSIYsD`.
- Angular hybrid lambda results on 25% per-matrix pruning:
  - lambda 0.5, from prior run: pruned PPL 1472.7434156203167, ratio 190.75987432277284.
  - lambda 0.25: baseline PPL 7.720404623083257, pruned PPL 1143.02065205968, ratio 148.051910212343, delta PPL +1135.3002474365967.
  - lambda 0.1: baseline PPL 7.720404623083257, pruned PPL 1005.3377138473169, ratio 130.21826742622468, delta PPL +997.6173092242336.
  - lambda 0.0: baseline PPL 7.720404623083257, pruned PPL 928.405617035533, ratio 120.25349218869835, delta PPL +920.6852124124498.
- Interpretation: reducing the angular component helps monotonically in this small sweep, but even lambda 0.0 is catastrophically worse than standard WANDA, output_l2, RI-only, magnitude, and GradCAM-style saliency. The normalized `w^2 ||x||^2 / ||y||^2` energy term appears to be a bad pruning score on this harness.
- Local downloaded summaries:
  - RI: `runs/modal_ri_20260527/ri_saliency_summary.json` and `runs/modal_ri_20260527/ri_prune25_ppl_summary.json`.
  - Angular lambda: `runs/modal_angular_lambda_20260527/angular_hybrid_l025_saliency_summary.json`, `angular_hybrid_l01_saliency_summary.json`, `angular_hybrid_l0_saliency_summary.json`, `angular_hybrid_l025_prune25_ppl_summary.json`, `angular_hybrid_l01_prune25_ppl_summary.json`, and `angular_hybrid_l0_prune25_ppl_summary.json`.

## Pythia 1.4B Feature-WANDA Plus Cosine Nudge - 2026-05-27

- Goal: try a metric that keeps feature-WANDA as the base score while integrating a small amount of exact cosine-damage saliency.
- Implemented `feature_cosine_wanda`, with aliases `feature_wanda_cosine`, `row_wanda_cosine`, and `cosine_feature_wanda`.
- Score definition: base is row/feature-WANDA `abs(w_ij) * (|Y|^T |X| / sum_t |Y_ti|)_ij`. Exact cosine damage is `1 - cos(y_i, y_i - w_ij x_j)`. Final score is `base * (1 + alpha * clamp(cos_damage / mean_matrix(cos_damage), max=10))`.
- Implementation note: `FeatureCosineWandaAccumulator` collects both row-conditioned absolute-input statistics and angular dot/norm statistics in one forward-hook pass, avoiding two calibration passes.
- Tests: `uv run pytest tests/test_approx_saliency.py tests/test_prune_eval.py -q` -> 30 passed. Compile check: `uv run python -m py_compile saliency/approx.py saliency/approx_cli.py modal_pythia_saliency.py`.
- Local smoke: `uv run python -m saliency.approx_cli --output-dir runs/local_pythia31m_feature_cosine_wanda_a005_smoke --model-name EleutherAI/pythia-31m --method feature_cosine_wanda --feature-cosine-alpha 0.05 --max-examples 1 --batch-size 1 --max-length 128 --dtype fp32 --device cpu --top-k 5`.
- Saliency Modal runs launched in parallel:
  - alpha 0.02: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method feature_cosine_wanda --feature-cosine-alpha 0.02 --feature-cosine-clip 10.0 --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_feature_cosine_wanda_a002_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-fyJNZ325SgAkNsxM08Gfzf`.
  - alpha 0.05: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method feature_cosine_wanda --feature-cosine-alpha 0.05 --feature-cosine-clip 10.0 --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_feature_cosine_wanda_a005_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-LMsuKDtGhuf7QsytLKdSwg`.
  - alpha 0.1: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method feature_cosine_wanda --feature-cosine-alpha 0.1 --feature-cosine-clip 10.0 --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_feature_cosine_wanda_a01_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-Gkgn9DI92yye2sd9edD6uI`.
- Saliency runtimes: alpha 0.02 31.497380018234253s, alpha 0.05 31.647844552993774s, alpha 0.1 28.74028778076172s. Each wrote 5,656,018,944 bytes of float32 scores.
- Prune/eval Modal runs:
  - alpha 0.02: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_feature_cosine_wanda_a002_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_feature_cosine_wanda_a002_prune25_ppl_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-nsWXp3PNxK2jyfqNaloSv8`.
  - alpha 0.05: first client `ap-LYbo6cUR4c7XMBCGR5aw7y` was locally killed after several silent minutes before useful logs; clean rerun used `--run-name pythia14b_gsm8k_100_feature_cosine_wanda_a005_prune25_ppl_bf16_20260527_rerun`, URL `https://modal.com/apps/jthomams477/main/ap-xfh8iNLZq4KXQ5OUz9UoZ0`.
  - alpha 0.1: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_feature_cosine_wanda_a01_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_feature_cosine_wanda_a01_prune25_ppl_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-Nf41Ecdw12rsGfofTjJnQ6`.
- Results on 25% per-matrix pruning, GSM8K train-100 answer-token PPL:
  - alpha 0.02: baseline PPL 7.720404623083257, pruned PPL 7.7991996623483075, ratio 1.0102060763796576, delta loss/token +0.01015434606011345.
  - alpha 0.05: baseline PPL 7.720404623083257, pruned PPL 7.7991996623483075, ratio 1.0102060763796576, delta loss/token +0.01015434606011345.
  - alpha 0.1: baseline PPL 7.720404623083257, pruned PPL 7.811881285023703, ratio 1.011848687524348, delta loss/token +0.011779041429731851.
- Comparison to same harness baselines from earlier sections: standard WANDA PPL 7.805537898214442, row/feature-WANDA PPL 7.85642957115383, output_l2 PPL 7.796032474711701, exact backprop GradCAM-style PPL 7.698485014076171. The small cosine nudge improves substantially over feature-WANDA and slightly beats standard WANDA, but it is still just behind output_l2 and far behind exact GradCAM-style saliency.
- Local downloaded summaries: `runs/modal_feature_cosine_wanda_20260527/a002_saliency_summary.json`, `a005_saliency_summary.json`, `a01_saliency_summary.json`, `a002_prune25_ppl_summary.json`, `a005_prune25_ppl_summary.json`, and `a01_prune25_ppl_summary.json`.

## Pythia 1.4B Graph-Propagated Saliency as Unstructured Scores - 2026-05-27

- Goal correction: do not test 2:4 or structured candidate selection. Test the graph-propagated idea only as ordinary per-weight saliency tensors in the same Pythia-1.4B bf16, GSM8K train-100 calibration/eval, 25% per-matrix unstructured pruning harness.
- Implemented GPT-NeoX graph-propagated saliency methods in `saliency/approx.py`, exposed through `saliency/approx_cli.py` and `modal_pythia_saliency.py`.
- Scope of implementation: for residual-stream output matrices (`attention.dense.weight` and `mlp.dense_4h_to_h.weight`), score `w_ij^2` times input energy propagated through a small downstream graph. Other trainable 2D weights use the existing `output_l2` fallback (`w_ij^2 * sum_t x_tj^2`) because their downstream intervention graph is not the residual-output graph.
- Methods:
  - `graph_norm`: exact LayerNorm input-Jacobian column norm squared through the next block's input/post-attention LayerNorm branches.
  - `graph_qkv`: Hutchinson/VJP estimate through next input LayerNorm plus next QKV projection, 4 random probes.
  - `graph_mlp`: Hutchinson/VJP estimate through next post-attention LayerNorm plus next MLP up projection, 4 random probes.
  - `graph_qkv_mlp`: sum of the QKV and MLP propagated gains, 4 random probes each branch.
- Tests added before implementation for exact LayerNorm Jacobian column gains, projected VJP gains against autograd, and graph-score fallback behavior. Final validation: `uv run pytest -q` -> 59 passed. Compile check: `uv run python -m py_compile saliency/approx.py saliency/approx_cli.py modal_pythia_saliency.py`.
- Local smokes: Pythia-31M CPU `graph_norm` and `graph_qkv --graph-num-probes 2` both completed.
- Saliency Modal runs:
  - `graph_norm`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method graph_norm --graph-num-probes 4 --graph-seed 17 --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_graph_norm_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-aLTjM7qwPolJEWACSR27Q9`, runtime 47.44259762763977s.
  - `graph_qkv`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method graph_qkv --graph-num-probes 4 --graph-seed 17 --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_graph_qkv_p4_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-A7rwWsHveAPhW3Cm3sJl4V`, runtime 44.317726612091064s.
  - `graph_mlp`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method graph_mlp --graph-num-probes 4 --graph-seed 17 --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_graph_mlp_p4_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-pWgRL9QG4NcolQbWm7JYu1`, runtime 41.125574588775635s.
  - `graph_qkv_mlp`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method graph_qkv_mlp --graph-num-probes 4 --graph-seed 17 --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_graph_qkv_mlp_p4_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-55qNoPmuPqN4na4jkkHAHR`, runtime 49.458088636398315s.
- Each saliency run wrote 5,656,018,944 bytes of float32 scores.
- Prune/eval Modal runs:
  - `graph_norm`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_graph_norm_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_graph_norm_prune25_ppl_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-Hh0368qzws6a4TIV3xLn1K`.
  - `graph_qkv`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_graph_qkv_p4_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_graph_qkv_p4_prune25_ppl_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-56dccaLd5x3YwXVAk9vpoE`.
  - `graph_mlp`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_graph_mlp_p4_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_graph_mlp_p4_prune25_ppl_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-ZK8fvRcKhxJkKJDFQjK6Oh`.
  - `graph_qkv_mlp`: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_graph_qkv_mlp_p4_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_graph_qkv_mlp_p4_prune25_ppl_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-cZTqM5lY4veuZgVmZ7TqKe`.
- Results on 25% per-matrix pruning, GSM8K train-100 answer-token PPL:
  - `graph_norm`: baseline PPL 7.720404623083257, pruned PPL 7.885201814187934, ratio 1.0213456676366353, delta loss/token +0.021121039805036546.
  - `graph_qkv`: baseline PPL 7.720404623083257, pruned PPL 8.027390159417351, ratio 1.039762881781641, delta loss/token +0.038992688870836734.
  - `graph_mlp`: baseline PPL 7.720404623083257, pruned PPL 8.152250873031077, ratio 1.05593570169323, delta loss/token +0.05442729488220932.
  - `graph_qkv_mlp`: baseline PPL 7.720404623083257, pruned PPL 7.930168220332082, ratio 1.0271700263768109, delta loss/token +0.02680747359870006.
- Interpretation: in this first unstructured-saliency harness, graph-propagated variants do not beat the simpler baselines. Best graph variant is `graph_norm` at PPL 7.8852, worse than standard WANDA 7.8055, output_l2 7.7960, feature-cosine-WANDA 7.7992, and exact backprop GradCAM-style saliency 7.6985. The likely issue is that the downstream local endpoint gain is not aligned enough with final answer-token PPL, and only residual-output matrices receive graph propagation while the rest use fallback scoring.
- Local downloaded summaries: `runs/modal_graph_propagated_20260527/graph_norm_saliency_summary.json`, `graph_qkv_saliency_summary.json`, `graph_mlp_saliency_summary.json`, `graph_qkv_mlp_saliency_summary.json`, `graph_norm_prune25_ppl_summary.json`, `graph_qkv_prune25_ppl_summary.json`, `graph_mlp_prune25_ppl_summary.json`, and `graph_qkv_mlp_prune25_ppl_summary.json`.

## Pythia 1.4B Corrected All-Matrix Logits-VJP Graph Saliency - 2026-05-27

- Correction to prior graph-propagated saliency run: the `graph_norm/qkv/mlp` methods above were mixed metrics because only residual-output matrices received graph propagation while other matrices used `output_l2` fallback. That caveat likely contaminated the comparison. This section uses an all-matrix graph saliency method.
- Implemented `graph_vjp_logits` in `saliency/approx.py`, with CLI aliases `graph_logits`, `subgraph_logits`, and `hutchinson_logits`.
- Score definition: for each random Rademacher projection `r` at answer-token logits, run a VJP through the actual downstream model graph and accumulate `(W_ij * d(r^T logits_answer)/dW_ij)^2`, averaged over probes and answer tokens. This is a Hutchinson estimate of full logits-endpoint perturbation damage for every trainable 2D parameter. There is no Linear-weight `output_l2` fallback.
- Tests: added `test_accumulate_vjp_parameter_scores_uses_weight_times_endpoint_vjp_squared`. Final validation: `uv run pytest -q` -> 60 passed. Compile check: `uv run python -m py_compile saliency/approx.py saliency/approx_cli.py modal_pythia_saliency.py`.
- Local smoke: `uv run python -m saliency.approx_cli --output-dir runs/local_pythia31m_graph_vjp_logits_smoke --model-name EleutherAI/pythia-31m --method graph_vjp_logits --graph-num-probes 1 --graph-seed 17 --max-examples 1 --batch-size 1 --max-length 128 --dtype fp32 --device cpu --top-k 5`.
- Saliency Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method graph_vjp_logits --graph-num-probes 4 --graph-seed 17 --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_graph_vjp_logits_p4_saliency_bf16_20260527`.
- Saliency Modal URL: `https://modal.com/apps/jthomams477/main/ap-7ylk6aIm0DqSIGBEnjThUd`.
- Saliency runtime: 187.27437615394592s, supervised answer tokens 9848, score storage 5,656,018,944 bytes, parameters scored 98.
- Prune/eval Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_graph_vjp_logits_p4_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_graph_vjp_logits_p4_prune25_ppl_bf16_20260527`.
- Prune/eval Modal URL: `https://modal.com/apps/jthomams477/main/ap-38E4sq1Sxz8daX44UOCCJl`.
- Pruning summary: 98 matrix tensors seen/pruned, 1,414,004,736 matrix weights seen, 353,501,184 zeroed, actual zero fraction 0.25, missing saliency `[]`.
- Result on 25% per-matrix unstructured pruning, GSM8K train-100 answer-token PPL: baseline PPL 7.720404623083257, pruned PPL 7.58981694803833, ratio 0.9830853845853516, delta PPL -0.13058767504492685, delta loss/token -0.017059301380991432.
- Interpretation: the user's concern was right. The mixed fallback caveat did materially change the experiment. The corrected all-matrix logits-VJP graph saliency is now the best result in this 100-example, 25% per-matrix pruning harness, beating the earlier exact backprop GradCAM-style saliency PPL 7.6985 and the cheaper baselines around 7.80.
- Local downloaded summaries: `runs/modal_graph_vjp_logits_20260527/graph_vjp_logits_p4_saliency_summary.json` and `runs/modal_graph_vjp_logits_20260527/graph_vjp_logits_p4_prune25_ppl_summary.json`.

## Pythia 1.4B Clean Local-Subgraph VJP Saliency - 2026-05-27

- Goal: implement the user's requested local-subgraph versions, not the full logits-to-end graph. These methods still cover every trainable 2D matrix and do not fall back to `output_l2`.
- Implemented `local_subgraph_vjp` / `local_graph_vjp` / `local_vjp` and all-token aliases `local_subgraph_vjp_all_tokens` / `local_graph_vjp_all_tokens` / `local_vjp_all_tokens`.
- Local graph definition:
  - `attn_context_l` endpoint, the input to `attention.dense`, scores same-layer `attention.query_key_value.weight` and previous residual-output matrices (`attention.dense.weight`, `mlp.dense_4h_to_h.weight`), or `embed_in.weight` for layer 0.
  - `mlp_activation_l` endpoint, the input to `mlp.dense_4h_to_h`, scores same-layer `mlp.dense_h_to_4h.weight` and previous residual-output matrices, or `embed_in.weight` for layer 0.
  - `final_norm` endpoint scores final-layer residual-output matrices.
  - `logits` endpoint scores only `embed_out.weight`.
- Score definition: for each local endpoint and random Rademacher projection `r`, accumulate `(W_ij * d(r^T endpoint)/dW_ij)^2`, averaged over probes and local endpoint positions. This is local one-block Hutchinson VJP damage, not final-model logits damage except for `embed_out`.
- Coverage: 50 local endpoint groups cover all 98 trainable 2D matrices on Pythia-1.4B; prune/eval summaries have `missing_saliency: []`.
- Tests: added endpoint-group coverage and VJP-gradient accumulator tests. Final validation: `uv run pytest -q` -> 62 passed. Compile check: `uv run python -m py_compile saliency/approx.py saliency/approx_cli.py modal_pythia_saliency.py`.
- Local smoke: `uv run python -m saliency.approx_cli --output-dir runs/local_pythia31m_local_vjp_smoke2 --model-name EleutherAI/pythia-31m --method local_vjp --graph-num-probes 1 --graph-seed 17 --max-examples 1 --batch-size 1 --max-length 128 --dtype fp32 --device cpu --top-k 3`.
- Saliency Modal runs:
  - Answer-token aligned endpoints: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method local_vjp --graph-num-probes 4 --graph-seed 17 --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_local_vjp_p4_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-k1SwVAPJb3nF8IcO6Yu1bL`, runtime 577.1618943214417s, supervised answer tokens 9848, endpoint tokens 492400.
  - All non-padding next-token endpoints: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method local_vjp_all_tokens --graph-num-probes 4 --graph-seed 17 --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_local_vjp_alltok_p4_saliency_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-B6WAZ6SmKL5DV696l9fPyc`, runtime 249.96892762184143s, supervised answer tokens 9848, endpoint tokens 806250.
- Prune/eval Modal runs:
  - Answer-token aligned endpoints: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_local_vjp_p4_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_local_vjp_p4_prune25_ppl_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-8fsJuZ6tYlAu1wEm8CKCGs`.
  - All-token endpoints: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_local_vjp_alltok_p4_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_local_vjp_alltok_p4_prune25_ppl_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-0wsjL1Jgye4vlgFFCYfjGA`.
- Results on 25% per-matrix unstructured pruning, GSM8K train-100 answer-token PPL:
  - `local_vjp`: baseline PPL 7.720404623083257, pruned PPL 8.22876603321869, ratio 1.0658464724265204, delta loss/token +0.06376929325751401.
  - `local_vjp_all_tokens`: baseline PPL 7.720404623083257, pruned PPL 7.7360996870533025, ratio 1.0020329328236397, delta loss/token +0.002030869212022779.
- Interpretation: the all-token local subgraph version is the useful local result. It is much better than the earlier mixed local graph runs and slightly worse than exact backprop GradCAM-style saliency (7.6985), while the answer-token-aligned endpoint variant is poor. Likely reason: local endpoints need prompt/non-answer positions because Q/K/V and residual-stream perturbations at prompt positions affect later answer-token behavior; answer-only local endpoints under-protect those positions.
- Local downloaded summaries: `runs/modal_local_subgraph_vjp_20260527/local_vjp_p4_saliency_summary.json`, `local_vjp_alltok_p4_saliency_summary.json`, `local_vjp_p4_prune25_ppl_summary.json`, and `local_vjp_alltok_p4_prune25_ppl_summary.json`.

## Pythia 1.4B Local Forward-Diff Wanda Saliency - 2026-05-27

- Goal: implement the user's requested "Wanda diff at slightly larger local subgraphs" as a forward-only saliency method, not a VJP/gradient method and not full-model logits propagation.
- Implemented `local_forward_wanda` with aliases `forward_subgraph_wanda`, `local_wanda_diff`, and `subgraph_wanda_diff`.
- Score form: use local forward endpoint damage in the spirit of `w_ij^2 * x_j^2 * downstream_gain_i`, where the downstream gain is measured by finite forward differences instead of backprop.
- Endpoint coverage:
  - `mlp.dense_h_to_4h.weight`: finite forward unit-gain through GELU.
  - `attention.dense.weight` and `mlp.dense_4h_to_h.weight`: finite coordinate perturbation through the next local LayerNorm endpoint; final-layer residual-output matrices use final LayerNorm.
  - `gpt_neox.embed_in.weight`: finite coordinate perturbation through layer 0 input/post-attention LayerNorm endpoints, accumulated by token id.
  - `attention.query_key_value.weight` and `embed_out.weight`: immediate linear endpoint damage. Important caveat: this is not exact replay through attention softmax for Q/K/V; it is the performant first local-forward version with no `output_l2` fallback.
- Coverage: all 98 trainable 2D matrices are scored; prune/eval had `missing_saliency: []`.
- Tests added before implementation for LayerNorm forward-diff gains and activation forward-diff scores. Final validation: `uv run pytest -q` -> 64 passed.
- Local smoke: `uv run python -m saliency.approx_cli --output-dir runs/local_pythia31m_local_forward_wanda_smoke --model-name EleutherAI/pythia-31m --method local_forward_wanda --local-forward-eps 0.001 --max-examples 1 --batch-size 1 --max-length 128 --dtype fp32 --device cpu --top-k 5`.
- Saliency Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method local_forward_wanda --local-forward-eps 0.001 --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_local_forward_wanda_saliency_bf16_20260527`.
- Saliency Modal URL: `https://modal.com/apps/jthomams477/main/ap-EVSWoJvLBTfoHtxJyp8Ypj`.
- Saliency runtime: 59.02458953857422s, score storage 5,656,018,944 bytes, parameters scored 98.
- Prune/eval Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_local_forward_wanda_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_local_forward_wanda_prune25_ppl_bf16_20260527`.
- Prune/eval Modal URL: `https://modal.com/apps/jthomams477/main/ap-R7oeJrKONjZGIqM2aN2eKC`.
- Pruning summary: 98 matrix tensors seen/pruned, 1,414,004,736 matrix weights seen, 353,501,184 zeroed, actual zero fraction 0.25, missing saliency `[]`.
- Result on 25% per-matrix unstructured pruning, GSM8K train-100 answer-token PPL: baseline PPL 7.720404623083257, pruned PPL 7.985115162355309, ratio 1.0342871328894594, delta PPL +0.2647105392720519, delta loss/token +0.0337124289195776.
- Interpretation: this is faster than the local VJP methods and uses forward local endpoint differences, but it does not beat standard WANDA or feature-cosine WANDA in this harness. It lands worse than standard WANDA (7.8055), feature-cosine WANDA alpha 0.02/0.05 (7.7992), output_l2 (7.7960), exact GradCAM-style saliency (7.6985), and local VJP all-token (7.7361). The likely issue is that the finite local gains are too coarse, especially for Q/K/V where this implementation still uses immediate projection damage instead of replaying attention.
- Local downloaded summaries: `runs/modal_local_forward_wanda_20260527/local_forward_wanda_saliency_summary.json` and `runs/modal_local_forward_wanda_20260527/local_forward_wanda_prune25_ppl_summary.json`.

## Pythia 1.4B Masked Standard WANDA Correctness Rerun - 2026-05-27

- Goal: rerun standard WANDA after fixing the activation accumulator to ignore calibration padding positions. The old WANDA implementation flattened all `[batch, seq]` activations, so right-padding tokens contributed to the per-input-column activation norm.
- Corrected score definition:
  \[
  S_{i,j}=|W_{i,j}|\sqrt{\frac{1}{|\mathcal{T}|}\sum_{(b,t)\in\mathcal{T}} X_{b,t,j}^2}
  \]
  where \(\mathcal{T}=\{(b,t): \mathrm{attention\_mask}_{b,t}=1\}\). This is still dense-model one-shot scoring followed by per-matrix pruning, not official sequential layerwise WANDA.
- Code/test validation before Modal: added `test_wanda_scores_ignore_padding_positions_with_attention_mask`; ran `uv run pytest tests/test_approx_saliency.py::test_wanda_scores_scale_weight_magnitude_by_input_rms_and_fallback_to_magnitude tests/test_approx_saliency.py::test_wanda_scores_ignore_padding_positions_with_attention_mask -q` -> 2 passed; ran `uv run python -m py_compile saliency/approx.py`; ran `uv run pytest tests/test_approx_saliency.py tests/test_prune_eval.py -q` -> 39 passed.
- Saliency Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method wanda --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_wanda_masked_saliency_bf16_20260527`.
- Saliency Modal URL: `https://modal.com/apps/jthomams477/main/ap-QslIEyZ8PMyIhAyyYkO7bV`.
- Saliency runtime: 28.01578187942505s, 13 batches, score storage 5,656,018,944 bytes, parameters scored 98, total elements 1,414,004,736, total saliency 15381870.364257812.
- Top saliency sums: `embed_out.weight` 3411604.75, `gpt_neox.embed_in.weight` 1771506.75, `gpt_neox.layers.22.attention.query_key_value.weight` 258231.34375, `gpt_neox.layers.23.attention.query_key_value.weight` 249172.375, `gpt_neox.layers.20.attention.query_key_value.weight` 227022.53125.
- Prune/eval Modal run: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_wanda_masked_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_wanda_masked_prune25_ppl_bf16_20260527`.
- Prune/eval Modal URL: `https://modal.com/apps/jthomams477/main/ap-aJqn9QK1f6QbHTK8XLcmdS`.
- Pruning summary: 98 matrix tensors seen/pruned, 1,414,004,736 matrix weights seen, 353,501,184 zeroed, actual zero fraction 0.25, missing saliency `[]`.
- Result on 25% per-matrix unstructured pruning, GSM8K train-100 answer-token PPL: baseline PPL 7.720404623083257, pruned PPL 7.834123763000133, ratio 1.0147296865214637, delta PPL +0.11371913991687599, delta loss/token +0.014622258326563387.
- Comparison: the old unmasked standard WANDA entry had pruned PPL 7.805537898214442 and ratio 1.011027048359181. The masked correctness fix is slightly worse in this 100-example harness, but it is the correct WANDA activation statistic because padding tokens no longer enter \(\|X_{:,j}\|_2\). It remains better than magnitude pruning (8.5838), worse than output_l2 (7.7960), exact backprop GradCAM-style saliency (7.6985), graph_vjp_logits (7.5898), and local_vjp_all_tokens (7.7361).
- Local downloaded summaries: `runs/modal_pythia14b_gsm8k_100_wanda_masked_20260527/wanda_masked_saliency_summary.json` and `runs/modal_pythia14b_gsm8k_100_wanda_masked_20260527/wanda_masked_prune25_ppl_summary.json`.

## Pythia 1.4B Masked WANDA Per-Output-Row Pruning Rerun - 2026-05-27

- Goal: undo the earlier stronger counterfactual in point (2), where `per_matrix` flattened each matrix and pruned the lowest 25% scores over the whole 2D tensor. This rerun keeps the same masked WANDA saliency artifact and changes only the pruning rule to prune the lowest fraction independently within each output row.
- Implemented `per_output_row` pruning scope in `saliency/prune_eval.py`: for a 2D weight matrix \(W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\), each output row prunes
  \[
  \left\lfloor s\,d_{\mathrm{in}}\right\rfloor
  \]
  entries with the lowest saliency in that row. This is closer to the usual WANDA pruning rule than the matrix-wide threshold.
- Tests: added `test_apply_row_saliency_pruning_zeroes_fraction_within_each_output_row`. Validation: `uv run pytest tests/test_prune_eval.py::test_apply_row_saliency_pruning_zeroes_fraction_within_each_output_row -q` -> 1 passed; `uv run python -m py_compile saliency/prune_eval.py saliency/prune_cli.py modal_pythia_saliency.py`; `uv run pytest tests/test_prune_eval.py tests/test_approx_saliency.py -q` -> 40 passed.
- Prune/eval Modal run reusing masked WANDA saliency: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_wanda_masked_saliency_bf16_20260527 --prune-fraction 0.25 --pruning-scope per_output_row --run-name pythia14b_gsm8k_100_wanda_masked_row_prune25_ppl_bf16_20260527`.
- Prune/eval Modal URL: `https://modal.com/apps/jthomams477/main/ap-Jh7Vvm9qM1WALyvFDlUeiY`.
- Pruning summary: 98 matrix tensors seen/pruned, 1,414,004,736 matrix weights seen, 353,501,184 zeroed, actual zero fraction 0.25, missing saliency `[]`.
- Result on 25% per-output-row unstructured pruning, GSM8K train-100 answer-token PPL: baseline PPL 7.720404623083257, pruned PPL 7.76758560089034, ratio 1.0061112053202517, delta PPL +0.047180977807082414, delta loss/token +0.006092607636067893.
- Comparison: row-wise pruning improves masked WANDA from PPL 7.834123763000133 to 7.76758560089034. It also beats the old unmasked matrix-wide WANDA result of 7.805537898214442. This suggests point (2) was not a stronger counterfactual in practice for WANDA on this harness; the row-wise constraint appears to protect each output channel from pathological uneven sparsity.
- Local downloaded summary: `runs/modal_pythia14b_gsm8k_100_wanda_masked_row_prune_20260527/wanda_masked_row_prune25_ppl_summary.json`.

## Pythia 1.4B Original-vs-Proper WANDA 2x2 Ablation - 2026-05-27

- Goal: ablate the whole gap between the original WANDA implementation and proper WANDA in parallel on Modal. The two axes are:
  - Activation statistic: original/unmasked includes padding positions in the per-input-column RMS; proper/masked excludes padding with `attention_mask`.
  - Pruning rule: matrix-wide threshold prunes the lowest 25% scores across each whole 2D matrix; row-wise threshold prunes the lowest 25% independently inside each output row.
- Implemented explicit `original_wanda` / `wanda_unmasked` / `legacy_wanda` aliases, leaving `wanda` as the masked implementation. Added `test_original_wanda_alias_disables_attention_mask`.
- Validation before Modal: `uv run pytest tests/test_approx_saliency.py::test_original_wanda_alias_disables_attention_mask tests/test_approx_saliency.py::test_wanda_scores_ignore_padding_positions_with_attention_mask tests/test_prune_eval.py::test_apply_row_saliency_pruning_zeroes_fraction_within_each_output_row -q` -> 3 passed; `uv run python -m py_compile saliency/approx.py saliency/approx_cli.py saliency/prune_eval.py saliency/prune_cli.py modal_pythia_saliency.py`; `uv run pytest tests/test_approx_saliency.py tests/test_prune_eval.py -q` -> 41 passed.
- Saliency Modal runs:
  - Original/unmasked: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method original_wanda --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_original_wanda_saliency_bf16_20260527_ablate`, URL `https://modal.com/apps/jthomams477/main/ap-v7u26jH6pdWCNXo5tnlO7B`, runtime 49.33236598968506s, total saliency 14450700.425292969.
  - Proper/masked: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode approx-saliency --approx-method wanda --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --run-name pythia14b_gsm8k_100_proper_wanda_saliency_bf16_20260527_ablate`, URL `https://modal.com/apps/jthomams477/main/ap-I7Ajjh4T1eEgcZ8XmiRiPL`, runtime 25.254256010055542s, total saliency 15381870.364257812.
- Prune/eval Modal runs:
  - Original/unmasked + matrix-wide: first local client wedged, relaunched as `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_original_wanda_saliency_bf16_20260527_ablate --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_original_wanda_per_matrix_prune25_bf16_20260527_ablate_retry`, URL `https://modal.com/apps/jthomams477/main/ap-l1bqW6TlDuINXSz47LQhTj`.
  - Original/unmasked + row-wise: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_original_wanda_saliency_bf16_20260527_ablate --prune-fraction 0.25 --pruning-scope per_output_row --run-name pythia14b_gsm8k_100_original_wanda_per_row_prune25_bf16_20260527_ablate`, URL `https://modal.com/apps/jthomams477/main/ap-OzMELo42cFJe7dfXYECQDQ`.
  - Proper/masked + matrix-wide: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_proper_wanda_saliency_bf16_20260527_ablate --prune-fraction 0.25 --pruning-scope per_matrix --run-name pythia14b_gsm8k_100_proper_wanda_per_matrix_prune25_bf16_20260527_ablate`, URL `https://modal.com/apps/jthomams477/main/ap-HQe7p2EVdT1TnuDToZvKiP`.
  - Proper/masked + row-wise: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 modal run modal_pythia_saliency.py --mode prune-ppl --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --saliency-run-name pythia14b_gsm8k_100_proper_wanda_saliency_bf16_20260527_ablate --prune-fraction 0.25 --pruning-scope per_output_row --run-name pythia14b_gsm8k_100_proper_wanda_per_row_prune25_bf16_20260527_ablate`, URL `https://modal.com/apps/jthomams477/main/ap-BzHbqYr3QgW3NawW9qqf35`.
- Results on 25% unstructured pruning, GSM8K train-100 answer-token PPL:

| Activation RMS | Pruning rule | Pruned PPL | Ratio | Delta loss/token |
| --- | --- | ---: | ---: | ---: |
| original/unmasked | matrix-wide | 7.805537898214442 | 1.011027048359181 | +0.010966693744922651 |
| original/unmasked | per-output-row | 7.701612572438232 | 0.997565924124127 | -0.002437043054427601 |
| proper/masked | matrix-wide | 7.834123763000133 | 1.0147296865214637 | +0.014622258326563387 |
| proper/masked | per-output-row | 7.76758560089034 | 1.0061112053202517 | +0.006092607636067893 |

- Interpretation: row-wise pruning is the dominant win in this 100-example harness. The best cell is surprisingly the original/unmasked activation statistic with row-wise pruning, which slightly beats the unpruned baseline PPL. Masking padding improves the definition but not this metric here. This may be because including pad positions gives a regularizing bias toward columns active in actual embedding/eos-style padded tails, or because the train-100 answer-token evaluation is noisy enough that the cleaner RMS is not monotonically better.
- Local downloaded summaries: `runs/modal_wanda_original_vs_proper_ablation_20260527/original_wanda_saliency_summary.json`, `proper_wanda_saliency_summary.json`, `original_wanda_per_matrix_prune25_summary.json`, `original_wanda_per_row_prune25_summary.json`, `proper_wanda_per_matrix_prune25_summary.json`, and `proper_wanda_per_row_prune25_summary.json`.

## Pythia 1.4B Full WANDA Properness Ablation - 2026-05-27

- Goal: ablate all implementation changes between the original local WANDA variant and the paper-style WANDA path, using up to 8 H100 Modal runs concurrently.
- Axes:
  - Activation statistic: `original` includes padding positions in the per-input-column RMS; `masked` excludes padding positions using `attention_mask`.
  - Pruning scope: `per_matrix` prunes the lowest 25% scores across each whole 2D matrix; `per_output_row` prunes the lowest 25% independently inside each output row.
  - Schedule: `one_shot` computes dense-model WANDA scores once before pruning; `sequential` recomputes WANDA on the current pruned model before each embedding/layer/head group and prunes that group immediately.
- Implemented `WandaAblationPruneConfig`, target-filtered WANDA activation hooks, one-shot and sequential WANDA prune/eval paths, Pythia grouping as embeddings -> transformer layers -> output head, and Modal mode `wanda-ablation-prune-ppl`.
- Tests before Modal: `uv run pytest tests/test_approx_saliency.py::test_sequential_wanda_parameter_groups_order_embeddings_layers_and_head tests/test_approx_saliency.py::test_apply_sequential_wanda_pruning_recomputes_each_group -q` -> 2 passed; `uv run python -m py_compile saliency/approx.py modal_pythia_saliency.py`; `uv run pytest tests/test_approx_saliency.py tests/test_prune_eval.py -q` -> 43 passed.
- Modal runs:
  - Original + matrix + one-shot: `pythia14b_gsm8k_100_wanda_ablate_original_matrix_oneshot_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-b5Kd2YbWJTyAct5O6TDmur`.
  - Original + row + one-shot: `pythia14b_gsm8k_100_wanda_ablate_original_row_oneshot_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-8WfGUinEohZgJUZ5jFL9Y6`.
  - Masked + matrix + one-shot: `pythia14b_gsm8k_100_wanda_ablate_masked_matrix_oneshot_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-bSpGNDl1yKSjek9K8P974H`.
  - Masked + row + one-shot: `pythia14b_gsm8k_100_wanda_ablate_masked_row_oneshot_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-vvKcsHpZkCVc57YmH4SMUV`.
  - Original + matrix + sequential: `pythia14b_gsm8k_100_wanda_ablate_original_matrix_seq_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-k797quqHcpb2SH8gBWNPIF`.
  - Original + row + sequential: `pythia14b_gsm8k_100_wanda_ablate_original_row_seq_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-JQJ5mamo8OqPDHnjfUDoAg`.
  - Masked + matrix + sequential: `pythia14b_gsm8k_100_wanda_ablate_masked_matrix_seq_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-gkrQzxRfGZgPPc7qaPP3ns`.
  - Masked + row + sequential: `pythia14b_gsm8k_100_wanda_ablate_masked_row_seq_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-0Hg5tFy20TCw2xjyyKmbVz`.
- Results on 25% unstructured pruning, GSM8K train-100 answer-token PPL. Baseline PPL was 7.720404623083257 for every run.

| Activation RMS | Pruning rule | Schedule | Pruned PPL | Ratio | Delta loss/token |
| --- | --- | --- | ---: | ---: | ---: |
| original/unmasked | matrix-wide | one-shot | 7.805537898214442 | 1.011027048359181 | +0.010966693744922651 |
| original/unmasked | per-output-row | one-shot | 7.701612572438232 | 0.997565924124127 | -0.002437043054427601 |
| proper/masked | matrix-wide | one-shot | 7.834123763000133 | 1.0147296865214637 | +0.014622258326563387 |
| proper/masked | per-output-row | one-shot | 7.76758560089034 | 1.0061112053202517 | +0.006092607636067893 |
| original/unmasked | matrix-wide | sequential | 7.904441825347251 | 1.0238377664447458 | +0.023558082859463703 |
| original/unmasked | per-output-row | sequential | 7.667279199517633 | 0.9931188291081553 | -0.006904955320877537 |
| proper/masked | matrix-wide | sequential | 7.881999701820093 | 1.0209309079803515 | +0.020714865962631723 |
| proper/masked | per-output-row | sequential | 7.692233707059613 | 0.9963511088603548 | -0.0036555645816411797 |

- Interpretation: row-wise pruning dominates the result. Sequential recomputation helps the row-wise cells but hurts the matrix-wide cells. The best cell is still not the paper-pure one: original/unmasked + per-output-row + sequential gets PPL 7.667279199517633, while proper/masked + per-output-row + sequential gets 7.692233707059613. Masking padding is the cleaner statistic, but in this GSM8K train-100 answer-token harness it is not the best measured variant.
- Local downloaded summaries: `runs/modal_wanda_full_ablation_20260527/*.json`.

## Pythia 1.4B Exact Matrix-Sequential WANDA Follow-Up - 2026-05-27

- Clarification: the previous `sequential` rows were embedding/layer/head group-sequential, not exact paper matrix-sequential. They propagated pruning effects between layers, but not between individual matrices inside a layer.
- Goal: test exact matrix-by-matrix WANDA only for the promising row-wise cells. This schedule scores and prunes one trainable 2D matrix, then reruns calibration through the modified model before scoring the next matrix. Fused QKV remains one matrix.
- Implemented `matrix_sequential` WANDA schedule with 98 single-matrix steps for Pythia-1.4B: `gpt_neox.embed_in.weight`, each layer's QKV/attention dense/MLP matrices in parameter order, then `embed_out.weight`.
- Tests before Modal: added `test_sequential_wanda_matrix_parameter_groups_order_every_matrix` and `test_apply_matrix_sequential_wanda_pruning_recomputes_each_matrix_inside_layer`. Validation: targeted tests -> 2 passed; `uv run python -m py_compile saliency/approx.py modal_pythia_saliency.py`; `uv run pytest tests/test_approx_saliency.py tests/test_prune_eval.py -q` -> 45 passed.
- Modal runs:
  - Original/unmasked + per-output-row + exact matrix-sequential: `pythia14b_gsm8k_100_wanda_ablate_original_row_matrixseq_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-dxy9GMsmOVeUrF5SgMKAIr`.
  - Proper/masked + per-output-row + exact matrix-sequential: `pythia14b_gsm8k_100_wanda_ablate_masked_row_matrixseq_bf16_20260527`, URL `https://modal.com/apps/jthomams477/main/ap-5xLr4jPFCYGhwqWvJpDE0k`.
- Results on 25% per-output-row unstructured pruning, GSM8K train-100 answer-token PPL. Baseline PPL was 7.720404623083257 for both runs.

| Activation RMS | Schedule | Steps | Pruned PPL | Ratio | Delta loss/token |
| --- | --- | ---: | ---: | ---: | ---: |
| original/unmasked | exact matrix-sequential | 98 | 7.661053232178578 | 0.9923123989217839 | -0.0077173030056867375 |
| proper/masked | exact matrix-sequential | 98 | 7.717269433431697 | 0.9995939086350233 | -0.00040617384240482224 |

- Comparison to group-sequential row-wise: original/unmasked improves from 7.667279199517633 to 7.661053232178578; proper/masked worsens from 7.692233707059613 to 7.717269433431697. The exact schedule helps the already-best unmasked row-wise variant slightly, but it does not rescue the masked/paper-pure variant in this harness.
- Local downloaded summaries: `runs/modal_wanda_matrix_sequential_promising_20260527/original_row_matrixseq_summary.json` and `runs/modal_wanda_matrix_sequential_promising_20260527/masked_row_matrixseq_summary.json`.

## Pythia 1.4B Superset Subgraph WANDA - 2026-05-27

- Goal: implement the closed-form superset-subgraph WANDA score and test it using the best WANDA harness variant we had measured: unmasked activation positions, per-output-row pruning, exact matrix-sequential recomputation.
- Implemented `superset_wanda` / `closed_form_superset_wanda` / `superset_subgraph_wanda` aliases. The score uses the local closed-form approximation
  `W_ij^2 * sum_t x_tj^2 * ||J_F(y_t) e_i||_2^2`, where the local subgraph factor is propagated through the next local modules when available. GELU derivatives and LayerNorm input-Jacobian column norms use closed-form paths; the method falls back to existing local gain approximations where needed.
- Added routing through one-shot, group-sequential, and exact matrix-sequential WANDA prune/eval paths via `wanda_method`, plus Modal CLI support through `--wanda-method`.
- Tests before Modal:
  - `uv run pytest tests/test_approx_saliency.py::test_apply_matrix_sequential_wanda_can_route_to_superset_scorer tests/test_approx_saliency.py::test_activation_local_jacobian_square_matches_gelu_autograd -q` -> 2 passed.
  - `uv run python -m py_compile saliency/approx.py saliency/approx_cli.py modal_pythia_saliency.py` -> passed.
  - `uv run pytest tests/test_approx_saliency.py tests/test_prune_eval.py -q` -> 47 passed.
- Local smoke:
  - `uv run python -m saliency.approx_cli --output-dir runs/local_pythia31m_superset_wanda_smoke --model-name EleutherAI/pythia-31m --method superset_wanda --max-examples 1 --batch-size 1 --max-length 64 --dtype fp32 --device cpu --top-k 3` completed and scored 26 tensors / 30,474,240 elements.
  - One-example Pythia-31M matrix-sequential prune/eval smoke completed with baseline PPL 77.65652564675415, pruned PPL 93.22261976608188, ratio 1.2004479854035082, and `score_method=superset_wanda`.
- Modal run:
  - Command: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 uv run modal run modal_pythia_saliency.py --mode wanda-ablation-prune-ppl --approx-method superset_wanda --wanda-method superset_wanda --wanda-activation unmasked --wanda-schedule matrix_sequential --pruning-scope per_output_row --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --prune-fraction 0.25 --run-name pythia14b_gsm8k_100_superset_wanda_unmasked_row_matrixseq_bf16_20260527`
  - URL: `https://modal.com/apps/jthomams477/main/ap-Y5xxwb5gpMIzubq5JIvMDd`
- Result on 25% per-output-row unstructured pruning, exact matrix-sequential schedule, GSM8K train-100 answer-token PPL:

| Method | Activation positions | Schedule | Steps | Baseline PPL | Pruned PPL | Ratio | Delta loss/token |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| superset WANDA | unmasked | exact matrix-sequential | 98 | 7.720404623083257 | 7.805537898214442 | 1.011027048359181 | +0.010966693744922651 |

- Comparison: this exactly matches the earlier original/unmasked matrix-wide one-shot PPL and is worse than the best measured WANDA variant, original/unmasked + per-output-row + exact matrix-sequential, which got PPL 7.661053232178578 / ratio 0.9923123989217839. The superset factor did not improve this harness in the current implementation.

## Correction: Superset Subgraph WANDA Closed-Form Factors - 2026-05-27

- Issue with the first `superset_wanda` implementation: it used a local output-coordinate gain but did not implement the two-matrix closed form. In particular, for `W_1 -> activation -> W_2`, the MLP-up score omitted the downstream `||W_2[:, i]||_2^2` column-norm factor.
- Corrected verified pieces:
  - Added `linear_superset_wanda_scores(weight, inputs, output_gain)`, implementing `W_ij^2 * sum_t x_tj^2 * g_ti`.
  - Added `layernorm_input_downstream_colnorm_squares(inputs, norm, downstream_weight)`, an exact closed form for `diag(J_LN^T W_down^T W_down J_LN)`.
  - MLP-up weights now use `activation_local_jacobian_square(preactivation) * ||dense_4h_to_h[:, i]||_2^2`.
  - Residual-output and `embed_in` weights now use exact LayerNorm-to-next-linear column-norm gains, instead of LayerNorm endpoint column norms.
  - `embed_out` and QKV currently use immediate linear endpoint damage; QKV softmax propagation is still not the full special-case softmax closed form.
- New correctness tests:
  - `test_linear_superset_wanda_scores_match_two_matrix_counterfactual`: exact match to brute-force counterfactual in the two-linear-matrix case.
  - `test_linear_superset_wanda_scores_include_activation_and_downstream_columns`: verifies activation gain includes downstream column norms and differs from the earlier incorrect local-only gain.
  - `test_layernorm_input_downstream_colnorm_squares_matches_autograd_columns`: exact match to autograd Jacobian column norms for LayerNorm followed by a downstream linear map.
  - `test_superset_wanda_accumulator_uses_mlp_downstream_column_norm`: verifies the actual accumulator path uses the MLP down-projection column norm.
- Validation:
  - Targeted new tests passed.
  - `uv run python -m py_compile saliency/approx.py saliency/approx_cli.py modal_pythia_saliency.py` passed.
  - `uv run pytest tests/test_approx_saliency.py tests/test_prune_eval.py -q` -> 51 passed.
  - Pythia-31M local scoring smoke completed: `runs/local_pythia31m_superset_wanda_corrected_smoke`.
  - Pythia-31M local matrix-sequential prune/eval smoke completed: baseline PPL 77.65652564675415, pruned PPL 85.55649809849493, ratio 1.1017296664503942, `score_method=superset_wanda`.
- Corrected Modal run:
  - Command: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 uv run modal run modal_pythia_saliency.py --mode wanda-ablation-prune-ppl --approx-method superset_wanda --wanda-method superset_wanda --wanda-activation unmasked --wanda-schedule matrix_sequential --pruning-scope per_output_row --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --prune-fraction 0.25 --run-name pythia14b_gsm8k_100_superset_wanda_corrected_unmasked_row_matrixseq_bf16_20260527`
  - URL: `https://modal.com/apps/jthomams477/main/ap-XkrklPjKNkYTRGd5vnn9In`
  - Runtime inside experiment: 89.76928997039795s.
- Corrected result on 25% per-output-row unstructured pruning, exact matrix-sequential schedule, GSM8K train-100 answer-token PPL:

| Method | Activation positions | Schedule | Steps | Baseline PPL | Pruned PPL | Ratio | Delta loss/token |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| corrected superset WANDA | unmasked | exact matrix-sequential | 98 | 7.720404623083257 | 7.792866573242642 | 1.009385771044011 | +0.00934199837530425 |

- Comparison: the corrected implementation improves over the first bad superset run, 7.805537898214442 -> 7.792866573242642, but remains worse than the best prior WANDA variant, original/unmasked + per-output-row + exact matrix-sequential, which got 7.661053232178578. This corrected run verifies the linear/injective closed-form factors but should not be treated as a complete softmax-aware QKV implementation.

## Correction: Superset Subgraph WANDA Softmax-Aware QKV - 2026-05-28

- Issue with the corrected 2026-05-27 run: QKV still used immediate linear endpoint damage. That was not the actual local superset subgraph for attention because Q and K must pass through rotary embedding, attention logits, causal softmax, and the attention output projection; V must pass through attention mixing and the output projection.
- Implemented the QKV closed-form local gain path:
  - Q gains use the softmax derivative `scale * sum_s A_as * (K_sd - mean_K_ad) * V_s`, pushed through the attention output projection Gram.
  - K gains use `scale * A_as * (V_s - O_a) * Q_ad`, also pushed through the output projection Gram.
  - V gains use attention-column energy through the same output projection Gram.
  - Rotary embeddings are handled in pre-rotary Q/K coordinates so the score is applied to the fused QKV linear output coordinates actually produced by `query_key_value.weight`.
- Existing closed-form factors retained:
  - MLP-up uses activation derivative squared times downstream MLP-down column norms.
  - Residual-output and `embed_in` use exact LayerNorm-to-next-linear gains.
  - `embed_out` remains immediate endpoint damage.
- Added direct verification tests:
  - `test_attention_qkv_superset_output_gains_match_autograd_endpoint_columns`: no-rotary causal attention helper matches autograd endpoint Jacobian column norms.
  - `test_attention_qkv_superset_output_gains_push_through_rotary_embedding`: rotary path matches autograd for pre-rotary Q/K coordinates.
  - `test_superset_wanda_accumulator_uses_attention_qkv_softmax_gain`: actual accumulator route for `attention.query_key_value.weight` uses the softmax-aware QKV gain.
- Validation:
  - Targeted QKV tests passed: 3 passed.
  - `uv run pytest tests/test_approx_saliency.py tests/test_prune_eval.py -q` -> 54 passed.
  - `uv run python -m py_compile saliency/approx.py saliency/approx_cli.py modal_pythia_saliency.py` passed.
  - Pythia-31M local scoring smoke completed: `runs/local_pythia31m_superset_wanda_qkv_softmax_smoke`.
  - Pythia-31M local matrix-sequential prune/eval smoke completed: baseline PPL 77.65652564675415, pruned PPL 87.12803463232599, ratio 1.1219666847916436, `score_method=superset_wanda`.
- Modal run:
  - Command: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 uv run modal run modal_pythia_saliency.py --mode wanda-ablation-prune-ppl --approx-method superset_wanda --wanda-method superset_wanda --wanda-activation unmasked --wanda-schedule matrix_sequential --pruning-scope per_output_row --model-name EleutherAI/pythia-1.4b --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --prune-fraction 0.25 --run-name pythia14b_gsm8k_100_superset_wanda_qkvsoftmax_unmasked_row_matrixseq_bf16_20260528`
  - URL: `https://modal.com/apps/jthomams477/main/ap-cu3G930jo8KW2MZQKQsPaB`
  - Runtime inside experiment: 134.93630123138428s.
- Result on 25% per-output-row unstructured pruning, exact matrix-sequential schedule, GSM8K train-100 answer-token PPL:

| Method | Activation positions | Schedule | Steps | Baseline PPL | Pruned PPL | Ratio | Delta loss/token |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| superset WANDA + softmax-aware QKV | unmasked | exact matrix-sequential | 98 | 7.720404623083257 | 7.773898144712818 | 1.0069288494892639 | +0.006904955320877093 |

- Comparison: the softmax-aware QKV correction improves the corrected-but-QKV-local run, 7.792866573242642 -> 7.773898144712818, and improves the first bad superset run, 7.805537898214442 -> 7.773898144712818. It is still worse than the best measured plain WANDA variant, original/unmasked + per-output-row + exact matrix-sequential, which got PPL 7.661053232178578 / ratio 0.9923123989217839 in the same 100-example Pythia-1.4B GSM8K harness.

## Qwen3-0.6B GSM8K Calibration Transfer Check - 2026-05-28

- Goal: test the top 3 previous GSM8K-100 pruning results, current superset/subgraph WANDA, and weight magnitude on `Qwen/Qwen3-0.6B`.
- Top-3 interpretation from prior Pythia-1.4B GSM8K-100 runs:
  - `graph_vjp_logits`, 4 probes, per-matrix pruning.
  - `original_wanda`, unmasked activations, per-output-row pruning, exact matrix-sequential schedule.
  - Exact gradient saliency, per-matrix pruning.
- Added Qwen3 support for superset WANDA:
  - Qwen MLP gate/up/down closed-form gains, including SwiGLU factors and downstream down-projection column norms.
  - RMSNorm-to-downstream-linear exact input-coordinate metric.
  - Qwen attention Q/K/V gains through q/k RMSNorm, RoPE, causal softmax, GQA cross-head K/V terms, and `o_proj` Gram.
  - Qwen hook coverage for `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, embeddings, and final RMSNorm.
- Local verification:
  - `uv run pytest tests/test_approx_saliency.py::test_rmsnorm_input_downstream_colnorm_squares_matches_autograd_columns tests/test_approx_saliency.py::test_qwen_mlp_superset_output_gains_match_autograd_endpoint_columns tests/test_approx_saliency.py::test_qwen_attention_qkv_superset_output_gains_match_autograd_endpoint_columns -q` -> 3 passed.
  - `uv run pytest tests/test_approx_saliency.py tests/test_prune_eval.py -q` -> 57 passed.
  - `uv run python -m py_compile saliency/approx.py saliency/approx_cli.py modal_pythia_saliency.py` passed.
  - Tiny randomly initialized `Qwen3ForCausalLM` hook smoke produced a finite `model.layers.0.self_attn.q_proj.weight` saliency tensor with shape `(16, 16)`.
- Modal profile: `jthomams477`.
- Shared calibration/eval setup:
  - Model: `Qwen/Qwen3-0.6B`.
  - Dataset slice: GSM8K train, 100 examples.
  - Max length: 512.
  - Batch size: 8.
  - Dtype: bf16.
  - Prune fraction: 25%.
  - Baseline answer-token PPL: 3.5501763575451273 on 12,502 supervised tokens.

| Method | Scope / schedule | Pruned PPL | Ratio | Delta loss/token | Modal URL |
| --- | --- | ---: | ---: | ---: | --- |
| superset WANDA | unmasked, per-output-row, matrix-sequential | 3.5592749945255875 | 1.0025628690138515 | +0.002559590465525563 | https://modal.com/apps/jthomams477/main/ap-3OF6ldePj7cVdwNoPHYLtc |
| graph VJP logits p4 | per-matrix | 3.5615532949724016 | 1.0032046119069817 | +0.0031994880819068428 | https://modal.com/apps/jthomams477/main/ap-RENOGPuXvXBT1NX8gBOxk8 |
| original WANDA | unmasked, per-output-row, matrix-sequential | 3.6236217320493602 | 1.020687810155724 | +0.02047672372420406 | https://modal.com/apps/jthomams477/main/ap-RWLYdglesBKtnLoeTPIo96 |
| exact gradient saliency | per-matrix | 3.6375609554077677 | 1.0246141568930578 | +0.024316109422492405 | https://modal.com/apps/jthomams477/main/ap-475CTMSfUpgU6ANVxN3X92 |
| weight magnitude | per-matrix | 4.31667747027037 | 1.2159050806296456 | +0.19548872180451138 | https://modal.com/apps/jthomams477/main/ap-HaW1geJp4Fq4wPZyjKAkwT |

- Saliency-generation runs:
  - `graph_vjp_logits` p4: `https://modal.com/apps/jthomams477/main/ap-umZ8nPJ4U2kujvKUCK3qVq`, runtime 73.07612419128418s.
  - Exact gradient saliency: `https://modal.com/apps/jthomams477/main/ap-t1EV2OAbgkf1ROzxa7NTYD`, runtime 44.42585849761963s.
  - Magnitude saliency: `https://modal.com/apps/jthomams477/main/ap-M3hujsLzUUzbVb9IBzfevm`, runtime 11.309767246246338s.
- Main readout: on this Qwen3-0.6B/GSM8K-100 slice, the Qwen-aware superset WANDA variant is best by a small margin, followed very closely by graph-VJP. Original WANDA and exact gradient are clearly behind, and magnitude is much weaker.

## Qwen3-0.6B GSM8K QRONOS Check - 2026-05-28

- Goal: test QRONOS on the same `Qwen/Qwen3-0.6B` GSM8K train-100 calibration/eval slice.
- QRONOS is a W4 weight-only quantization run, not a 25% pruning run, so it is comparable by PPL movement but not by sparsity/compression mechanism.
- Existing QRONOS implementation was reused without code changes; it is model-agnostic over `torch.nn.Linear`, and Qwen exposed 196 quantized linear layers when skipping `lm_head`.
- Shared setup:
  - Model: `Qwen/Qwen3-0.6B`.
  - Dataset slice: GSM8K train, 100 examples.
  - Max length: 512.
  - Batch size: 8.
  - Dtype: fp32.
  - Weight bits: 4.
  - `percdamp=1e-6`, `num_blocks=100`, `quantize_last_layer=false`.
  - Paper-code-faithful padding behavior: `use_attention_mask=false`.
  - Baseline answer-token PPL: 3.5468403234210304 on 12,502 supervised tokens.

| Method | Activation order | Quantized PPL | Ratio | Delta loss/token | Modal URL |
| --- | --- | ---: | ---: | ---: | --- |
| QRONOS W4 FP32 paper-faithful | true | 3.5100753388950956 | 0.9896344404671497 | -0.010419656097211849 | https://modal.com/apps/jthomams477/main/ap-C6bYMvBBRIK71lDVYILPgF |
| QRONOS W4 FP32 no activation order | false | 3.529433178598824 | 0.9950922107467707 | -0.0049198720001675245 | https://modal.com/apps/jthomams477/main/ap-j8kyVyprrYcAhBb56Td5mM |

- Commands:
  - Paper-faithful: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 uv run modal run modal_pythia_saliency.py --mode qronos-weight-only-ppl --model-name Qwen/Qwen3-0.6B --max-examples 100 --batch-size 8 --max-length 512 --dtype fp32 --weight-bits 4 --percdamp 1e-6 --num-blocks 100 --no-use-attention-mask --run-name qwen3_06b_gsm8k100_qronos_w4_fp32_paper_20260528`
  - No activation order: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 uv run modal run modal_pythia_saliency.py --mode qronos-weight-only-ppl --model-name Qwen/Qwen3-0.6B --max-examples 100 --batch-size 8 --max-length 512 --dtype fp32 --weight-bits 4 --percdamp 1e-6 --num-blocks 100 --no-use-activation-order --no-use-attention-mask --run-name qwen3_06b_gsm8k100_qronos_w4_fp32_noact_20260528`
- Main readout: QRONOS improved PPL on this 100-example Qwen3/GSM8K slice, with the paper-faithful activation-order run best. This is directionally stronger than the pruning methods above, but it is a W4 quantization result rather than a 25% sparse-pruning result.

## Correction: Qwen3-0.6B GSM8K QRONOS Base-Precision Pruning - 2026-05-28

- Correction: the QRONOS section immediately above was the wrong comparison target for WANDA-style pruning. It tested W4 weight-only quantization in fp32. The pruning comparison should use sparse pruning in the same base dtype as WANDA, bf16.
- Implemented a QRONOS-style base-precision pruning path:
  - `qronos_prune_weight`: QRONOS sequential covariance / cross-covariance mechanics, but with a base-precision sparsifying operator instead of an integer quantizer.
  - `run_qronos_prune_experiment`: original/pruned model pair, QRONOS pair-stat collection, layerwise pruning, then answer-token PPL eval.
  - Modal mode: `qronos-prune-ppl`.
- Local verification:
  - `uv run pytest tests/test_qronos_eval.py tests/test_gptq_eval.py -q` -> 19 passed.
  - `uv run python -m py_compile saliency/qronos_eval.py modal_pythia_saliency.py` passed.
- Shared setup:
  - Model: `Qwen/Qwen3-0.6B`.
  - Dataset slice: GSM8K train, 100 examples.
  - Max length: 512.
  - Batch size: 8.
  - Dtype: bf16.
  - Prune fraction: 25%.
  - Pruning scope: per output row.
  - `percdamp=1e-6`, `num_blocks=100`, `quantize_last_layer=false`.
  - Padding behavior matched the QRONOS/WANDA unmasked comparisons: `use_attention_mask=false`.
  - Baseline answer-token PPL: 3.5501763575451273 on 12,502 supervised tokens.

| Method | Activation order | Pruned PPL | Ratio | Delta loss/token | Zero fraction | Modal URL |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| QRONOS-style bf16 pruning | false | 3.5185139029626584 | 0.9910814417669204 | -0.008958566629339249 | 0.25 | https://modal.com/apps/jthomams477/main/ap-eJwSFI1lCCsiMw8xKNernn |
| QRONOS-style bf16 pruning | true | 3.527531393219367 | 0.993621453684228 | -0.0063989761638136855 | 0.25 | https://modal.com/apps/jthomams477/main/ap-NSdGp2zeEu8ANUUQjAlO0W |

- Commands:
  - Activation order: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 uv run modal run modal_pythia_saliency.py --mode qronos-prune-ppl --model-name Qwen/Qwen3-0.6B --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --prune-fraction 0.25 --pruning-scope per_output_row --percdamp 1e-6 --num-blocks 100 --no-use-attention-mask --run-name qwen3_06b_gsm8k100_qronos_prune25_bf16_row_actorder_20260528`
  - No activation order: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 uv run modal run modal_pythia_saliency.py --mode qronos-prune-ppl --model-name Qwen/Qwen3-0.6B --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --prune-fraction 0.25 --pruning-scope per_output_row --percdamp 1e-6 --num-blocks 100 --no-use-activation-order --no-use-attention-mask --run-name qwen3_06b_gsm8k100_qronos_prune25_bf16_row_noact_20260528`
- Coverage caveat: this QRONOS pruning path is `torch.nn.Linear`-only and skips `lm_head`, so Qwen exposed 196 pruned linear layers / 440,401,920 weights. The earlier WANDA Qwen table counted 197 two-dimensional tensors / 595,984,384 weights. This is a pruning result in the same base precision, but not identical tensor coverage to the all-2D WANDA harness.
- Main readout: on this exact Qwen3-0.6B/GSM8K-100 bf16 pruning slice, QRONOS-style pruning with no activation order is best among the pruning methods tested so far, improving answer-token PPL from 3.5501763575451273 to 3.5185139029626584. The prior W4 QRONOS numbers should not be used as WANDA-pruning comparisons.

## Correction: Qwen3-0.6B QRONOS Pruning With WANDA-Equal Tensor Coverage - 2026-05-28

- Correction to the QRONOS pruning section above: the first bf16 pruning implementation only covered `torch.nn.Linear` tensors. For an apples-to-apples WANDA pruning comparison, QRONOS pruning must target the same WANDA matrix-sequential 2D parameter list.
- Implementation update:
  - QRONOS pruning target names now come from WANDA's matrix-sequential target grouping.
  - Linear weights use QRONOS sequential covariance / cross-covariance pruning.
  - Non-Linear 2D parameters use the same base-precision magnitude fallback WANDA uses when no activation statistic exists. On Qwen3-0.6B this adds `model.embed_tokens.weight`.
- Local verification:
  - Added tests that QRONOS target names match WANDA matrix-sequential target names and that embedding fallback pruning is applied.
  - `uv run pytest tests/test_qronos_eval.py tests/test_gptq_eval.py -q` -> 21 passed.
  - `uv run python -m py_compile saliency/qronos_eval.py modal_pythia_saliency.py` passed.
- Shared setup:
  - Model: `Qwen/Qwen3-0.6B`.
  - Dataset slice: GSM8K train, 100 examples.
  - Max length: 512.
  - Batch size: 8.
  - Dtype: bf16.
  - Prune fraction: 25%.
  - Pruning scope: per output row.
  - `percdamp=1e-6`, `num_blocks=100`.
  - Padding behavior: `use_attention_mask=false`.
  - Baseline answer-token PPL: 3.5501763575451273 on 12,502 supervised tokens.

| Method | Activation order | Matrix tensors | Weights seen | Weights zeroed | Pruned PPL | Ratio | Delta loss/token | Modal URL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| QRONOS-style bf16 pruning, WANDA-equal coverage | true | 197 | 595,984,384 | 148,996,096 | 3.5547227649376967 | 1.0012806145201512 | +0.0012797952327627815 | https://modal.com/apps/jthomams477/main/ap-uJfnOgejLYJ2Rcvbg68tEQ |
| QRONOS-style bf16 pruning, WANDA-equal coverage | false | 197 | 595,984,384 | 148,996,096 | 3.5626929920170687 | 1.003525637380053 | +0.0035194368900974826 | https://modal.com/apps/jthomams477/main/ap-eHjcZj7ddyikPVtYi2jvIB |

- Commands:
  - Activation order: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 uv run modal run modal_pythia_saliency.py --mode qronos-prune-ppl --model-name Qwen/Qwen3-0.6B --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --prune-fraction 0.25 --pruning-scope per_output_row --percdamp 1e-6 --num-blocks 100 --no-use-attention-mask --run-name qwen3_06b_gsm8k100_qronos_prune25_bf16_row_equal_wanda_actorder_20260528`
  - No activation order: `env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET MODAL_PROFILE=jthomams477 uv run modal run modal_pythia_saliency.py --mode qronos-prune-ppl --model-name Qwen/Qwen3-0.6B --max-examples 100 --batch-size 8 --max-length 512 --dtype bf16 --prune-fraction 0.25 --pruning-scope per_output_row --percdamp 1e-6 --num-blocks 100 --no-use-activation-order --no-use-attention-mask --run-name qwen3_06b_gsm8k100_qronos_prune25_bf16_row_equal_wanda_noact_20260528`
- Main readout: after matching WANDA tensor coverage, QRONOS-style pruning no longer shows the large Linear-only gain. The activation-order variant is best and slightly beats the prior Qwen3 superset WANDA result, 3.5547227649376967 vs 3.5592749945255875, but the margin is small. The no-activation-order variant is worse than the activation-order variant once embeddings are included.

## Four-Family LLM 25% Pruning Grid - 2026-05-28

- Goal: scale beyond Qwen3-0.6B and test five generally applicable pruning approaches on four LLM families.
- Method choice correction: `superset_wanda` is not a safe cross-family method in this codebase because its closed-form hooks are Qwen/GPT-NeoX shaped and can fail for Llama/OPT-style models. The cross-family five-method grid therefore used:
  - QRONOS-style bf16 pruning with WANDA-equal tensor coverage, per-output-row.
  - Original WANDA, unmasked activations, per-output-row, matrix-sequential.
  - Graph-VJP logits p4 saliency, per-matrix pruning.
  - Exact backprop saliency, per-matrix pruning.
  - Weight magnitude, per-matrix pruning.
- Shared setup:
  - Dataset slice: GSM8K train, 100 examples.
  - Max length: 512.
  - Batch size: 8.
  - Dtype: bf16.
  - Prune fraction: 25%.
  - Modal profile: `jthomams477`.
  - Max concurrency: 8 H100 jobs.
- Local artifacts:
  - Orchestrator results: `runs/qwen_series_4families_5methods_20260528/orchestrator_results.json`.
  - Parsed scoreboard: `runs/qwen_series_4families_5methods_20260528/parsed_results.json`.
  - Logs: `runs/qwen_series_4families_5methods_20260528/logs/`.
  - TinyLlama QRONOS retry summary: `runs/qwen_series_4families_5methods_20260528/modal_downloads/tinyllama11b_qronos_prune_summary.json`.
- Execution note: the first TinyLlama QRONOS local client hung before useful progress and was killed; the retry completed normally. The retry result is used below.

| Model | Family | Method | Baseline PPL | Pruned PPL | Ratio | Delta loss/token | Tensors | Weights |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-1.7B` | Qwen | magnitude | 4.7561118257795965 | 4.445628672476948 | 0.9347191225362419 | -0.06750919852823545 | 197 | 1,720,451,072 |
| `Qwen/Qwen3-1.7B` | Qwen | QRONOS | 4.7561118257795965 | 4.643341837065238 | 0.9762894581024966 | -0.023996160614301543 | 197 | 1,720,451,072 |
| `Qwen/Qwen3-1.7B` | Qwen | exact grad | 4.7561118257795965 | 4.7106788486874 | 0.9904474539799641 | -0.009598464245720528 | 197 | 1,720,451,072 |
| `Qwen/Qwen3-1.7B` | Qwen | graph-VJP logits p4 | 4.7561118257795965 | 5.021967496608533 | 1.05589769134273 | +0.054391297392417215 | 197 | 1,720,451,072 |
| `Qwen/Qwen3-1.7B` | Qwen | original WANDA | 4.7561118257795965 | 5.240280251446194 | 1.1017992098171987 | +0.09694448888177898 | 197 | 1,720,451,072 |
| `EleutherAI/pythia-1.4b` | GPT-NeoX | graph-VJP logits p4 | 7.720404623083257 | 7.592900359309375 | 0.9834847692577334 | -0.01665312753858661 | 98 | 1,414,004,736 |
| `EleutherAI/pythia-1.4b` | GPT-NeoX | original WANDA | 7.720404623083257 | 7.661053232178578 | 0.9923123989217839 | -0.0077173030056867375 | 98 | 1,414,004,736 |
| `EleutherAI/pythia-1.4b` | GPT-NeoX | QRONOS | 7.720404623083257 | 7.689109957373796 | 0.995946499278562 | -0.004061738424045558 | 98 | 1,414,004,736 |
| `EleutherAI/pythia-1.4b` | GPT-NeoX | exact grad | 7.720404623083257 | 7.751826658034058 | 1.0040699984631443 | +0.004061738424045114 | 98 | 1,414,004,736 |
| `EleutherAI/pythia-1.4b` | GPT-NeoX | magnitude | 7.720404623083257 | 8.583812377432562 | 1.1118345211814158 | +0.1060113728675871 | 98 | 1,414,004,736 |
| `facebook/opt-1.3b` | OPT | graph-VJP logits p4 | 9.28464958590473 | 9.45214183753043 | 1.0180396955292716 | +0.01787891101178385 | 146 | 1,315,115,008 |
| `facebook/opt-1.3b` | OPT | QRONOS | 9.28464958590473 | 9.58752933431831 | 1.032621559447261 | +0.03210077204388462 | 146 | 1,315,115,008 |
| `facebook/opt-1.3b` | OPT | original WANDA | 9.28464958590473 | 9.626566452053401 | 1.036826038827329 | +0.036164160910198806 | 146 | 1,315,115,008 |
| `facebook/opt-1.3b` | OPT | exact grad | 9.28464958590473 | 10.161130311751792 | 1.094401056037448 | +0.0902072328321819 | 146 | 1,315,115,008 |
| `facebook/opt-1.3b` | OPT | magnitude | 9.28464958590473 | 11.901155711541293 | 1.2818098950776504 | +0.24827305973181613 | 146 | 1,315,115,008 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Llama | QRONOS | 4.044530399706655 | 4.042095585783848 | 0.9993979983626816 | -0.0006021829130598011 | 156 | 1,099,956,224 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Llama | original WANDA | 4.044530399706655 | 4.09970313626253 | 1.013641320772339 | +0.013549115543846524 | 156 | 1,099,956,224 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Llama | exact grad | 4.044530399706655 | 4.100937707701297 | 1.0139465654649877 | +0.013850207000376313 | 156 | 1,099,956,224 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Llama | graph-VJP logits p4 | 4.044530399706655 | 4.11950095502454 | 1.018536281819843 | +0.018366578848325155 | 156 | 1,099,956,224 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Llama | magnitude | 4.044530399706655 | 4.36203811342476 | 1.078502986092312 | +0.07557395558901026 | 156 | 1,099,956,224 |

- Main readout:
  - QRONOS is the most stable cross-family method: it is best on TinyLlama, second on Qwen3-1.7B and OPT, and third on Pythia.
  - Graph-VJP is strongest on Pythia and OPT, but not on Qwen3-1.7B or TinyLlama.
  - Magnitude is unexpectedly best on Qwen3-1.7B on this small GSM8K-100 answer-token slice, but is very poor on the other three models.
  - Original WANDA is solid on Pythia and TinyLlama, weak on Qwen3-1.7B and behind QRONOS/graph-VJP on OPT.
  - Because all evaluations reuse the 100-example calibration/eval slice, PPL improvements after pruning should be treated as slice-local and noisy, not as generalization claims.

## Superset WANDA Cross-Family Correction - 2026-05-28

- User correction: the four-family comparison should include `superset_wanda`; excluding it because the implementation was architecture-specific was the wrong endpoint. I changed the implementation to make the closed-form superset scorer work across the tested families.
- Implementation changes:
  - `qwen_attention_qkv_superset_output_gains` now handles split Q/K/V attention without Q/K RMSNorm and without rotary position embeddings. This covers Llama/TinyLlama-style and OPT-style split projections in addition to Qwen with Q/K norms.
  - `LocalForwardWandaAccumulator` now covers OPT decoder blocks under `model.decoder.layers.*`, including `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`, `embed_tokens`, `embed_positions`, final layer norm, and `lm_head` downstream gains.
  - Llama-style `model.layers.*.self_attn.{q,k,v,o}_proj` now uses the same split-attention superset path even when `q_norm` and `k_norm` are absent.
- Local verification:
  - Added tests for split attention without Q/K norms matching autograd endpoint column gains.
  - Added tests that Llama-style split-QKV gets the expected superset WANDA score.
  - Added tests that OPT-style decoder targets, including embeddings, are covered.
  - `uv run pytest tests/test_approx_saliency.py -q` -> 53 passed.
  - `uv run pytest tests/test_approx_saliency.py tests/test_qronos_eval.py tests/test_gptq_eval.py -q` -> 74 passed.
  - `uv run python -m py_compile saliency/approx.py modal_pythia_saliency.py` passed.
- Modal runs:
  - Shared setup: GSM8K train 100 examples, max length 512, batch size 8, bf16, 25% pruning, per-output-row, matrix-sequential, unmasked activations, `wanda_method=superset_wanda`.
  - Local parsed artifact: `runs/qwen_series_4families_superset_wanda_20260528/parsed_results.json`.

| Model | Family | Baseline PPL | Superset WANDA PPL | Ratio | Delta loss/token | Tensors | Weights | Modal URL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `facebook/opt-1.3b` | OPT | 9.28464958590473 | 8.969443633741983 | 0.966050851004515 | -0.034538805363673486 | 146 | 1,315,115,008 | https://modal.com/apps/jthomams477/main/ap-nwkgZOZONH2uelmdVvlrPx |
| `EleutherAI/pythia-1.4b` | GPT-NeoX | 7.720404623083257 | 7.773898144712818 | 1.0069288494892639 | +0.006904955320877093 | 98 | 1,414,004,736 | https://modal.com/apps/jthomams477/main/ap-gkkrecJ4dL0lnuRTdUrStG |
| `Qwen/Qwen3-1.7B` | Qwen | 4.7561118257795965 | 4.988338376068391 | 1.0488269743848442 | +0.04767237242041289 | 197 | 1,720,451,072 | https://modal.com/apps/jthomams477/main/ap-vLGL1zcHUTs6aNU2LP9vIp |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Llama | 4.044530399706655 | 4.125707358303488 | 1.0200707994685168 | +0.01987203613097477 | 156 | 1,099,956,224 | https://modal.com/apps/jthomams477/main/ap-CQGNAhXMFTWEF4fF2fopHS |

- Readout versus the previous four-family grid:
  - Superset WANDA is now the best method on OPT-1.3B in this slice: 8.9694 PPL vs graph-VJP's previous best 9.4521 and QRONOS 9.5875.
  - Superset WANDA is not competitive on Qwen3-1.7B here: 4.9883, behind magnitude 4.4456, QRONOS 4.6433, and exact grad 4.7107.
  - Superset WANDA is also behind the prior bests on Pythia-1.4B and TinyLlama-1.1B.
  - This changes the cross-family story: `superset_wanda` should be included in the method set, and it materially matters for OPT.

## Working Theory for Cross-Family Method Differences - 2026-05-28

- Goal: develop a falsifiable theory for why pruning methods rank differently across Qwen3-1.7B, Pythia-1.4B, OPT-1.3B, and TinyLlama-1.1B on the GSM8K-100 answer-token PPL slice.
- Important confound: the methods are not fully matched on schedule/scope.
  - `magnitude`, `exact_grad`, and `graph_vjp_logits_p4` were one-shot saliency artifacts pruned per matrix.
  - `original_wanda` and `superset_wanda` were unmasked, per-output-row, matrix-sequential.
  - `qronos` was per-output-row with WANDA-equal tensor coverage and a sequential module pass.
  - Therefore method comparisons mix saliency objective with pruning geometry. Any theory below should be tested with matched geometry.

### Method Objectives

- Magnitude estimates no data-dependent objective: keep large `|W_ij|`.
- Original WANDA estimates immediate local linear output damage:
  - Score is approximately `|W_ij| * rms(x_j)`.
  - It ignores downstream mixing, residual cancellation, and output-channel curvature.
- Superset WANDA estimates a local downstream endpoint quadratic:
  - Score is `W_ij^2 * sum_t x_tj^2 * ||J_F e_i||^2`.
  - It adds exact local gains through activation functions, attention softmax, output projections, and next-normalization/projection branches.
  - It is still local: it is not the final answer-token loss unless the chosen endpoint is a faithful surrogate.
- Exact grad estimates first-order answer-loss sensitivity:
  - Current implementation accumulates `|W_ij * dL/dW_ij|`.
  - This can be unstable for pruning because a weight can have low first-order gradient at the current optimum but high second-order removal damage.
- Graph-VJP logits estimates full downstream logits endpoint damage:
  - Score is a Hutchinson estimate of `(W_ij * d(r^T logits)/dW_ij)^2` on answer-token positions.
  - This is closer to final output geometry, but p4 can be noisy.
- QRONOS-style pruning is different: it does not just rank weights. It uses input Hessian/cross-covariance and future-column compensation while pruning in base precision.
  - This should help when feature correlations are strong or when pruning one column changes the effective future input distribution.

### Model-Specific Hypotheses

- Qwen3-1.7B:
  - Observed ranking: magnitude > QRONOS > exact_grad > superset_wanda > graph-VJP > original WANDA.
  - Hypothesis: low-magnitude weights are genuinely a large dead/redundant tail on this GSM8K slice, while activation/logit-conditioned criteria overfit the small calibration distribution.
  - Qwen's RMSNorm + SwiGLU + GQA/rotary path may make local gain estimates sharp and outlier-sensitive. Superset WANDA may preserve high local-Jacobian channels that do not matter for answer-token PPL, while pruning small but globally useful weights elsewhere.
  - QRONOS survives because covariance/future compensation is less dependent on a single local endpoint ranking.
  - Falsifiable prediction: clipping or tempering superset gains on Qwen should improve it; increasing graph probes alone may not close the gap if the issue is objective mismatch rather than estimator noise.

- Pythia-1.4B:
  - Observed ranking: graph-VJP > original WANDA > QRONOS > exact_grad > superset_wanda > magnitude.
  - Hypothesis: Pythia's GPT-NeoX stack has enough activation-scale heterogeneity that magnitude-only pruning is bad, but final-logits endpoint geometry is informative.
  - Original WANDA works because activation RMS is a reasonable proxy for local damage in this architecture. Graph-VJP improves because it sees answer-token logits and downstream residual mixing.
  - Superset WANDA underperforms because its local endpoint is too myopic: it preserves local reconstruction through next-block gains rather than final answer-token behavior.
  - Falsifiable prediction: graph-VJP p16/p32 should be at least as good as p4 on Pythia; superset WANDA should improve if the endpoint is moved closer to logits or if its local gains are blended with WANDA.

- OPT-1.3B:
  - Observed ranking: superset_wanda > graph-VJP > QRONOS > original WANDA > exact_grad > magnitude.
  - Hypothesis: OPT is the architecture where the superset approximation is best matched. Its pre-LN decoder and simpler split-attention path make exact local LayerNorm-to-next-projection gains a faithful proxy for downstream residual-stream damage.
  - Magnitude fails badly because raw weight scale is poorly calibrated across OPT matrices/channels.
  - Exact grad also fails because first-order loss sensitivity at the dense point is not removal damage.
  - Superset WANDA's gain over graph-VJP suggests the local deterministic second-order proxy has lower variance than p4 logits Hutchinson here.
  - Falsifiable prediction: ablate OPT superset components; most gain should come from residual-output and embedding/next-LayerNorm propagation, not from attention Q/K/V softmax gain.

- TinyLlama-1.1B:
  - Observed ranking: QRONOS > original WANDA > exact_grad > graph-VJP > superset_wanda > magnitude.
  - Hypothesis: TinyLlama is highly normalized and compact; many methods are close, but raw magnitude still removes important small weights.
  - QRONOS is best because it handles correlated features and compensates pruning effects without relying on the local endpoint being a faithful final-loss proxy.
  - Superset WANDA may be harmed by GQA/RMSNorm/SwiGLU local gain sharpness, similar to Qwen, but the effect is smaller.
  - Falsifiable prediction: QRONOS should remain strong under larger eval slices; superset WANDA should improve if gains are clipped/temperature-scaled or if pruning is per-matrix instead of per-row.

### Cross-Cutting Theory

- The best method depends on which approximation error dominates:
  - If raw weight scale is already a good redundancy signal, magnitude can win. This happened only for Qwen3-1.7B.
  - If output-token endpoint geometry matters and Hutchinson variance is manageable, graph-VJP wins. This happened for Pythia.
  - If local residual-stream Jacobians are faithful, superset WANDA wins. This happened for OPT.
  - If feature covariance/compensation dominates, QRONOS wins or stays near the top. This happened for TinyLlama and was stable across models.
- Improvements after pruning are likely slice-local regularization effects, not evidence that pruning generally improves the model.

### Next Tests

1. Match pruning geometry: rerun all six methods with per-output-row matrix-sequential where possible, and separately per-matrix one-shot where possible.
2. Increase graph probes: p4 vs p16/p32 on Pythia and OPT to separate estimator variance from objective quality.
3. Superset gain ablations:
   - No attention-softmax gain.
   - Residual-output/next-norm only.
   - MLP-up downstream-column norm only.
   - Gain clipping or power tempering, e.g. `gain^alpha` for `alpha in {0.25, 0.5, 0.75}`.
4. Rank-correlation diagnostics: compare per-matrix Spearman/Kendall between magnitude, WANDA, superset WANDA, graph-VJP, and QRONOS pruning masks; focus on winner-vs-loser disagreement matrices.
5. Evaluate on a disjoint set or another corpus slice to distinguish robust pruning from GSM8K-100 overfitting.

## 2026-05-28 Rigorous Theory Follow-Up

The prior section was hypothesis-level and not sufficient. New plan:

1. Compute model-weight tail statistics and saliency/mask agreement diagnostics for the four-family grid.
2. Run causal ablations that modify the suspicious superset-WANDA local gains:
   - gain tempering with `gain^0.5`
   - empirical high-quantile clipping if tempering helps or exposes outlier sensitivity
3. Run matched-geometry magnitude through the same matrix-sequential, per-output-row pruning harness so magnitude is not advantaged or disadvantaged by a different pruning schedule.
4. Explain model-family differences only after comparing weight tails, score-weight alignment, mask overlap, and ablation PPL deltas.

### Evidence Collected

All runs below use GSM8K-100 calibration/eval, bf16 base precision, 25% unstructured pruning, per-output-row matrix-sequential pruning unless noted.

Superset-WANDA gain ablations:

| Model | Baseline PPL | Superset PPL | `gain^0.5` PPL | clip-95 PPL |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 4.7561 | 4.9883 | 4.9645 | 4.8374 |
| Pythia-1.4B | 7.7204 | 7.7739 | 7.6766 | 7.7235 |
| OPT-1.3B | 9.2846 | 8.9694 | 9.0279 | 10.2565 |
| TinyLlama-1.1B | 4.0445 | 4.1257 | 4.1207 | 4.0886 |

Matched-geometry magnitude:

| Model | Previous magnitude PPL | Matrix-seq row magnitude PPL | Interpretation |
|---|---:|---:|---|
| Qwen3-1.7B | 4.4456 | 4.5376 | Still improves over baseline; Qwen magnitude win is not just schedule/scope artifact. |
| Pythia-1.4B | 8.5838 | 8.1721 | Geometry helps but magnitude remains bad. |
| OPT-1.3B | 11.9012 | 13.1149 | Magnitude is structurally wrong for OPT. |
| TinyLlama-1.1B | 4.3620 | 4.2403 | Geometry helps but remains worse than QRONOS/WANDA variants. |

Weight/saliency diagnostics from streamed Modal jobs:

| Model | Bottom 25% weight-L2 fraction | Top 25% weight-L2 fraction | Exact-grad pruned weight-L2 fraction | Graph-VJP pruned weight-L2 fraction |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 0.0069 | 0.7524 | 0.0374 | 0.0172 |
| Pythia-1.4B | 0.0077 | 0.7398 | 0.0464 | 0.0348 |
| OPT-1.3B | 0.0065 | 0.7660 | 0.0787 | 0.0617 |
| TinyLlama-1.1B | 0.0075 | 0.7423 | 0.0369 | 0.0247 |

The raw weight tails are very similar across families: the bottom 25% of weights only carries about 0.65-0.77% of sampled weight L2, while the top 25% carries about 74-77%. So the Qwen magnitude result is not explained by Qwen having a uniquely larger dead-weight tail.

### Revised Theory

The strongest causal signal is from superset gain modification. Clipping high local gains helps Qwen, Pythia, and TinyLlama, but destroys OPT. Tempering with `gain^0.5` similarly helps Pythia and mildly helps Qwen/TinyLlama while hurting OPT relative to unmodified superset WANDA. That suggests:

1. OPT is the model where the exact local superset endpoint is actually aligned with pruning damage. Its large local LayerNorm/next-projection gains are real fragile directions. Clipping removes useful signal.
2. Qwen and TinyLlama have RMSNorm/SwiGLU/GQA paths where local closed-form gains are too sharp. The largest local gains over-protect channels that do not proportionally matter for final answer-token PPL. Clipping those gains improves the method.
3. Pythia is intermediate: graph-VJP is still best, but gain tempering makes superset WANDA cross below baseline PPL. Its issue is not that local propagation is useless; it is that the untempered local endpoint overweighted a subset of channels.

Magnitude is a separate phenomenon. Matched matrix-sequential per-row magnitude still helps only Qwen. Since the raw weight-tail statistics are similar across all four models, Qwen's magnitude win is probably not just "more tiny weights." A better explanation is that Qwen's trained parameterization makes small weights especially disposable on this slice, while activation/logit-conditioned criteria overfit calibration-token geometry. The fact that graph-VJP strongly aligns with absolute weight on Qwen yet underperforms magnitude reinforces that endpoint conditioning is not always beneficial.

QRONOS remains plausibly strong for TinyLlama and competitive elsewhere because it attacks a different error term: feature covariance and compensation after pruning, not just weight or endpoint ranking. This predicts QRONOS should look best when the ranking methods disagree but no single ranker has a clean architectural match.

### Next Falsification Tests

1. Run superset `gain^0.25` and `gain^0.75` on Pythia/Qwen/TinyLlama to estimate the best gain temperature and whether it is monotone.
2. Run clip-99 on OPT to see whether only extreme outliers matter or whether any clipping hurts.
3. Compute QRONOS mask overlap and block covariance summaries; current diagnostics cover magnitude/exact-grad/graph-VJP but not QRONOS.
4. Repeat the Qwen magnitude-vs-graph-vs-superset comparison on a non-GSM8K calibration/eval slice to test whether magnitude is a GSM8K-100 regularization artifact.
