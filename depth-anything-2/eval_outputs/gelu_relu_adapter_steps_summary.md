# GELU to ReLU Adapter Step-Count Summary

Rows compare 200 vs 500 steps for the strongest adapter candidates. All trainable rows use rank 32, alpha 32, cal32/tok8192, RAdam, lr 0.003.

| method | steps | accuracy | correct | note |
|---|---:|---:|---:|---|
| `dense_gelu_baseline` |  | 0.952128 | 1969/2068 | original GELU |
| `dora_direct_fc1_fc2` | 500 | 0.932302 | 1928/2068 | direct adapter, more steps |
| `glora_direct_fc1_fc2` | 500 | 0.926015 | 1915/2068 | direct adapter, more steps |
| `lora_direct_fc1_fc2` | 500 | 0.919246 | 1901/2068 | direct adapter, more steps |
| `hidden_sandwich_lora_radam` | 200 | 0.918279 | 1899/2068 | prior best |
| `hidden_sandwich_lora_radam` | 500 | 0.916344 | 1895/2068 | more steps |
| `fact_tucker_direct_fc1_fc2` | 500 | 0.915861 | 1894/2068 | direct adapter, more steps |
| `dora_direct_fc1_fc2` | 200 | 0.914894 | 1892/2068 | direct adapter |
| `glora_direct_fc1_fc2` | 200 | 0.914894 | 1892/2068 | direct adapter |
| `fact_tucker_direct_fc1_fc2` | 200 | 0.903288 | 1868/2068 | direct adapter |
| `lora_direct_fc1_fc2` | 200 | 0.897485 | 1856/2068 | direct adapter |
