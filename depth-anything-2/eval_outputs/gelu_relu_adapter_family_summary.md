# GELU to ReLU Adapter Family Summary

Full rows use direct `fc1`/`fc2` adapter repairs, rank 32, alpha 32, cal32/tok8192, 200 RAdam steps, folded into dense weights before DA-2K evaluation.

## Full DA-2K

| method | accuracy | correct | trainable params | note |
|---|---:|---:|---:|---|
| `dense_gelu_baseline` | 0.952128 | 1969/2068 |  | prior dense GELU |
| `hidden_sandwich_lora_radam` | 0.918279 | 1899/2068 |  | prior best: hidden sandwich LoRA, RAdam |
| `dora` | 0.914894 | 1892/2068 | 1497600 | direct fc1/fc2 |
| `glora` | 0.914894 | 1892/2068 | 2949120 | direct fc1/fc2 |
| `fact_tucker` | 0.903288 | 1868/2068 | 1499136 | direct fc1/fc2 |
| `lora` | 0.897485 | 1856/2068 | 1474560 | direct fc1/fc2 |
| `loha` | 0.896518 | 1854/2068 | 2949120 | direct fc1/fc2 |
| `lokr` | 0.719052 | 1487/2068 | 369024 | direct fc1/fc2 |
| `vera` | 0.628143 | 1299/2068 | 768 | direct fc1/fc2 |
| `ia3` | 0.611219 | 1264/2068 | 36864 | direct fc1/fc2 |

## 32-image sweep, 50 steps

| method | accuracy | correct | ties |
|---|---:|---:|---:|
| `glora` | 0.887324 | 63/71 | 0 |
| `dora` | 0.788732 | 56/71 | 0 |
| `lora` | 0.760563 | 54/71 | 0 |
| `vera` | 0.647887 | 46/71 | 0 |
| `ia3` | 0.633803 | 45/71 | 0 |
| `loha` | 0.633803 | 45/71 | 0 |
| `lokr` | 0.633803 | 45/71 | 0 |
| `fact_tucker` | 0.605634 | 43/71 | 0 |
