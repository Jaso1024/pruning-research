# GELU to ReLU LoRA Optimizer Summary

All rows are full DA-2K evaluations of `lora_sandwich` with rank 32, alpha 32, cal32/tok8192, 200 steps, batch size 2048, and lr 0.003.

| optimizer | accuracy | correct | ties | direction | note |
|---|---:|---:|---:|---|---|
| `radam` | 0.918279 | 1899/2068 | 0 | larger | best/tied |
| `adagrad` | 0.918279 | 1899/2068 | 1 | larger | best/tied |
| `adamw` | 0.916828 | 1896/2068 | 2 | larger |  |
| `adam` | 0.916828 | 1896/2068 | 2 | larger |  |
| `adamax` | 0.914894 | 1892/2068 | 1 | larger |  |
| `sgd_momentum` | 0.624758 | 1292/2068 | 0 | larger |  |
| `sgd` | 0.617988 | 1278/2068 | 0 | larger |  |
| `nadam` | 0.553191 | 1144/2068 | 160 | larger |  |
| `rmsprop` | 0.503868 | 1042/2068 | 0 | smaller |  |
