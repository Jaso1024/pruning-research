# Scalar Taylor Saliency

## Setup

Taylor scores were added to `eval_scalar_weight_circuit_da2k.py`.

For DA2K margin `m = depth(point1) - depth(point2)`, use pair loss:

`L(m) = softplus(-m / tau)`, with `tau = 1`.

For scalar weight pruning `w -> 0`, use the local-linear margin change:

`delta_m ~= -w * grad_m(w)`.

Then approximate loss change with:

- first order: `delta_L1 = L'(m) * delta_m`
- second order: `delta_L2 = delta_L1 + 0.5 * L''(m) * delta_m^2`
- third order: `delta_L3 = delta_L2 + (1/6) * L'''(m) * delta_m^3`

The pruning scores below rank by absolute predicted loss change: `abs(delta_Lk)`, accumulated over the first 32 stratified DA2K images, then evaluated on full DA2K.

Dense full DA2K reference: `1969 / 2068 = 0.9521276596`.

## Results

| method | masked values | correct | accuracy | drop vs dense |
|---|---:|---:|---:|---:|
| global magnitude | 534,768 | 1968 | 0.9516441006 | 1 |
| scalar `abs_wgrad` | 534,768 | 1964 | 0.9497098646 | 5 |
| Taylor-1 `abs(delta_L1)` | 534,768 | 1962 | 0.9487427466 | 7 |
| Taylor-2 `abs(delta_L2)` | 534,768 | 1963 | 0.9492263056 | 6 |
| Taylor-3 `abs(delta_L3)` | 534,768 | 1962 | 0.9487427466 | 7 |
| global magnitude | 1,020,352 | 1972 | 0.9535783366 | -3 |
| scalar `abs_wgrad` | 1,020,352 | 1965 | 0.9501934236 | 4 |
| Taylor-1 `abs(delta_L1)` | 1,020,352 | 1965 | 0.9501934236 | 4 |
| Taylor-2 `abs(delta_L2)` | 1,020,352 | 1965 | 0.9501934236 | 4 |
| Taylor-3 `abs(delta_L3)` | 1,020,352 | 1965 | 0.9501934236 | 4 |
| global magnitude | 1,518,336 | 1967 | 0.9511605416 | 2 |
| scalar `abs_wgrad` | 1,518,336 | 1958 | 0.9468085106 | 11 |
| Taylor-1 `abs(delta_L1)` | 1,518,336 | 1956 | 0.9458413926 | 13 |
| Taylor-2 `abs(delta_L2)` | 1,518,336 | 1956 | 0.9458413926 | 13 |
| Taylor-3 `abs(delta_L3)` | 1,518,336 | 1956 | 0.9458413926 | 13 |
| global magnitude | 2,016,304 | 1968 | 0.9516441006 | 1 |
| scalar `abs_wgrad` | 2,016,304 | 1953 | 0.9443907157 | 16 |
| Taylor-1 `abs(delta_L1)` | 2,016,304 | 1953 | 0.9443907157 | 16 |
| Taylor-2 `abs(delta_L2)` | 2,016,304 | 1952 | 0.9439071567 | 17 |
| Taylor-3 `abs(delta_L3)` | 2,016,304 | 1953 | 0.9443907157 | 16 |
| global magnitude | 3,006,528 | 1966 | 0.9506769826 | 3 |
| scalar `abs_wgrad` | 3,006,528 | 1954 | 0.9448742747 | 15 |
| Taylor-1 `abs(delta_L1)` | 3,006,528 | 1954 | 0.9448742747 | 15 |
| Taylor-2 `abs(delta_L2)` | 3,006,528 | 1955 | 0.9453578337 | 14 |
| Taylor-3 `abs(delta_L3)` | 3,006,528 | 1954 | 0.9448742747 | 15 |
| global magnitude | 5,014,736 | 1927 | 0.9318181818 | 42 |
| scalar `abs_wgrad` | 5,014,736 | 1946 | 0.9410058027 | 23 |
| Taylor-1 `abs(delta_L1)` | 5,014,736 | 1943 | 0.9395551257 | 26 |
| Taylor-2 `abs(delta_L2)` | 5,014,736 | 1944 | 0.9400386847 | 25 |
| Taylor-3 `abs(delta_L3)` | 5,014,736 | 1943 | 0.9395551257 | 26 |

## Read

- Higher-order Taylor terms did not beat the simpler scalar `abs_wgrad` score in this run.
- Taylor-2 is slightly better than Taylor-1/3 at `3.0M` and `5.0M` masks, but the gain is small.
- Taylor-1 and Taylor-3 are effectively identical at these budgets, which suggests the cubic term is too small/noisy under the local-linear margin approximation and `tau=1`.
- Plain global magnitude is still best until the hardest `5.0M` budget. Scalar `abs_wgrad` remains the best high-mask scalar method tested so far.

The next thing worth testing is not deeper Taylor order, but layer-balanced Taylor or Taylor with a calibrated `tau` chosen from the observed margin distribution, so easy pairs do not saturate away.

## Artifacts

- `eval_outputs/scalar_taylor1_abs_s32_tau1_full/summary.json`
- `eval_outputs/scalar_taylor2_abs_s32_tau1_full/summary.json`
- `eval_outputs/scalar_taylor3_abs_s32_tau1_full/summary.json`
- `eval_outputs/scalar_weight_circuit_abs_wgrad_s32_full/summary.json`
