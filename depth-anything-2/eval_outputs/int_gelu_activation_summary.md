# Integer-friendly GELU replacement sweep

Depth Anything V2 Small on DA-2K. Replaces `nn.GELU` only; no repair, no calibration fitting, no finetuning.

`pwl...` variants quantize fixed-point coefficients but keep activation values in float. `intpwl...` variants additionally fake-quantize activation values and use integer affine segment math before dequantizing back into the surrounding float model.

## Full DA-2K, exact fake-int PWL

| activation | correct / pairs | accuracy | notes |
|---|---:|---:|---|
| `dense` | 1968 / 2068 | 0.951644 | original GELU baseline |
| `intpwl13_xq10_cq8` | 1955 / 2068 | 0.945358 | exact fake-int: Q10 activation, Q8 segment coefficients |
| `intpwl13_xq12_cq8` | 1956 / 2068 | 0.945841 | exact fake-int: Q12 activation, Q8 segment coefficients |
| `intpwl13_xq8_cq8` | 1955 / 2068 | 0.945358 | exact fake-int: Q8 activation, Q8 segment coefficients |
| `pwl13_q8` | 1957 / 2068 | 0.946325 | 12 affine segments, q8 coefficients, float activation values |

## Full DA-2K, coefficient-quantized PWL

| activation | correct / pairs | accuracy | notes |
|---|---:|---:|---|
| `dense` | 1968 / 2068 | 0.951644 | original GELU baseline |
| `hardgelu_s0_3125` | 1834 / 2068 | 0.886847 | cheap hard gate, slope 5/16; failed full eval |
| `pwl13_q12` | 1954 / 2068 | 0.944874 | 12 affine segments, q12 coefficients, float activation values |
| `pwl13_q8` | 1957 / 2068 | 0.946325 | 12 affine segments, q8 coefficients, float activation values |
| `pwl17_q12` | 1954 / 2068 | 0.944874 | 16 affine segments, q12 coefficients, float activation values |
| `pwl17_q8` | 1957 / 2068 | 0.946325 | 16 affine segments, q8 coefficients, float activation values |

## 32-image exact fake-int screen

| activation | correct / pairs | accuracy | notes |
|---|---:|---:|---|
| `dense` | 66 / 71 | 0.929577 | original GELU baseline |
| `intpwl13_xq10_cq8` | 66 / 71 | 0.929577 | exact fake-int: Q10 activation, Q8 segment coefficients |
| `intpwl13_xq12_cq8` | 66 / 71 | 0.929577 | exact fake-int: Q12 activation, Q8 segment coefficients |
| `intpwl13_xq8_cq8` | 66 / 71 | 0.929577 | exact fake-int: Q8 activation, Q8 segment coefficients |
| `intpwl17_xq10_cq8` | 66 / 71 | 0.929577 |  |
| `intpwl17_xq12_cq8` | 66 / 71 | 0.929577 |  |
| `pwl13_q8` | 66 / 71 | 0.929577 | 12 affine segments, q8 coefficients, float activation values |

## 32-image broad screen

| activation | correct / pairs | accuracy | notes |
|---|---:|---:|---|
| `dense` | 66 / 71 | 0.929577 | original GELU baseline |
| `hardgelu_s0_25` | 52 / 71 | 0.732394 |  |
| `hardgelu_s0_3125` | 66 / 71 | 0.929577 | cheap hard gate, slope 5/16; failed full eval |
| `hardgelu_s0_375` | 56 / 71 | 0.788732 |  |
| `hardswish` | 46 / 71 | 0.647887 |  |
| `pwl13_q12` | 66 / 71 | 0.929577 | 12 affine segments, q12 coefficients, float activation values |
| `pwl13_q8` | 66 / 71 | 0.929577 | 12 affine segments, q8 coefficients, float activation values |
| `pwl17_q12` | 66 / 71 | 0.929577 | 16 affine segments, q12 coefficients, float activation values |
| `pwl17_q8` | 66 / 71 | 0.929577 | 16 affine segments, q8 coefficients, float activation values |
| `pwl7_q8` | 62 / 71 | 0.873239 |  |
| `pwl9_q8` | 62 / 71 | 0.873239 |  |
| `relu` | 45 / 71 | 0.633803 |  |
| `shifted_square_r2` | 52 / 71 | 0.732394 |  |

## Takeaway

Dense GELU baseline: `1968/2068 = 0.951644`.
Best coefficient-only PWL: `pwl13_q8` and `pwl17_q8`, both `1957/2068 = 0.946325`.
Best exact fake-int PWL: `intpwl13_xq12_cq8`, `1956/2068 = 0.945841`; Q8/Q10 activation precision are one pair lower at `1955/2068 = 0.945358`.
The cheap hard-gate `x * clamp(0.5 + 5*x/16, 0, 1)` is not good enough: `1834/2068 = 0.886847` full DA-2K.
Current practical candidate: 12-segment PWL GELU with Q8 coefficients; use Q8 activation precision if lowest integer cost matters, Q12 if the extra single DA-2K pair decision matters.
