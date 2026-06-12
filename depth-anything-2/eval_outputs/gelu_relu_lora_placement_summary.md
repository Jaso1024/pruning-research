# GELU to ReLU LoRA Placement Summary

| run | accuracy | correct | note |
|---|---:|---:|---|
| `dense_baseline` | 0.952128 | 1969/2068 | prior full GELU baseline |
| `lora_hidden_rank32_cal32` | 0.911992 | 1886/2068 | hidden identity LoRA folded into fc2; current best |
| `lora_fc2_rank32_cal32` | 0.906673 | 1875/2068 | standard fc2 LoRA folded into fc2 |
| `lora_output_rank32_cal32` | 0.905706 | 1873/2068 | parallel MLP-output LoRA residual |
| `newton_ls_cal32_tok8192` | 0.785783 | 1625/2068 | full fc2 least-squares repair |
| `lora_fc1_rank32_cal32` | 0.654255 | 1353/2068 | fc1 pre-activation LoRA folded into fc1 |
| `relu_plain` | 0.619439 | 1281/2068 | plain GELU->ReLU full control |
