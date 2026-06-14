# Circuit LoRA Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9288 | 248 | 267 |
| `pruned_student` | 0.9326 | 249 | 267 |
| `lora_repaired_unmerged` | 0.9288 | 248 | 267 |
| `folded_lora_unmasked` | 0.9288 | 248 | 267 |
| `folded_lora_remasked` | 0.9438 | 252 | 267 |

## Selected Circuits

| name | kind | module | range | correct_drop | abs_margin_delta | params |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `block_00_mlp_group_46_1472_1504` | `mlp_group` | `pretrained.blocks.0.mlp` | 1472:1504 | 0 | 0.0119 | 24608 |
| `block_00_v_head_05` | `attn_v_head` | `pretrained.blocks.0.attn` | 320:384 | 0 | 0.0136 | 24640 |
| `block_00_mlp_group_22_704_736` | `mlp_group` | `pretrained.blocks.0.mlp` | 704:736 | 0 | 0.0141 | 24608 |
| `block_11_mlp_group_37_1184_1216` | `mlp_group` | `pretrained.blocks.11.mlp` | 1184:1216 | 0 | 0.0150 | 24608 |
| `block_00_mlp_group_29_928_960` | `mlp_group` | `pretrained.blocks.0.mlp` | 928:960 | 0 | 0.0153 | 24608 |
| `block_00_mlp_group_40_1280_1312` | `mlp_group` | `pretrained.blocks.0.mlp` | 1280:1312 | 0 | 0.0154 | 24608 |
| `block_00_mlp_group_06_192_224` | `mlp_group` | `pretrained.blocks.0.mlp` | 192:224 | 0 | 0.0164 | 24608 |
| `block_00_mlp_group_09_288_320` | `mlp_group` | `pretrained.blocks.0.mlp` | 288:320 | 0 | 0.0167 | 24608 |
| `block_00_mlp_group_05_160_192` | `mlp_group` | `pretrained.blocks.0.mlp` | 160:192 | 0 | 0.0167 | 24608 |
| `block_11_mlp_group_08_256_288` | `mlp_group` | `pretrained.blocks.11.mlp` | 256:288 | 0 | 0.0167 | 24608 |

## Repair

- Masked tensor values: `246112`
- LoRA trainable params: `147456`
- LoRA modules: `5`
- Merge RMS deltas: `{'pretrained.blocks.0.attn.qkv': 0.004490847699344158, 'pretrained.blocks.0.mlp.fc1': 0.0035691247321665287, 'pretrained.blocks.0.mlp.fc2': 0.0021257514599710703, 'pretrained.blocks.11.mlp.fc1': 0.0030876665841788054, 'pretrained.blocks.11.mlp.fc2': 0.0011704027419909835}`
