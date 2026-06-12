# GELU to ReLU Identity LoRA Repair Summary

| run | accuracy | correct | note |
|---|---:|---:|---|
| `dense_baseline` | 0.952128 | 1969/2068 | prior full GELU baseline |
| `lora_rank32_cal32_tok8192` | 0.911992 | 1886/2068 | identity LoRA repair, 32 calibration images |
| `lora_rank64_cal32_tok8192` | 0.910058 | 1882/2068 | identity LoRA repair, 32 calibration images |
| `lora_rank16_cal32_tok8192` | 0.905706 | 1873/2068 | identity LoRA repair, 32 calibration images |
| `lora_rank16_cal8_tok4096` | 0.892650 | 1846/2068 | identity LoRA repair, 8 calibration images |
| `newton_ls_cal32_tok8192` | 0.785783 | 1625/2068 | full fc2 least-squares repair |
| `relu_plain` | 0.619439 | 1281/2068 | plain GELU->ReLU full control |
