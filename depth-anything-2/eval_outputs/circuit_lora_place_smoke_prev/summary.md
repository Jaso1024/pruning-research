# Circuit LoRA Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 1.0000 | 19 | 19 |
| `pruned_student` | 1.0000 | 19 | 19 |
| `lora_repaired_unmerged` | 1.0000 | 19 | 19 |
| `folded_lora_unmasked` | 1.0000 | 19 | 19 |
| `folded_lora_remasked` | 1.0000 | 19 | 19 |

## Selected Circuits

| name | kind | module | range | correct_drop | abs_margin_delta | params |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `block_03_mlp_group_09_288_320` | `mlp_group` | `pretrained.blocks.3.mlp` | 288:320 | -3 | 0.0689 | 24608 |
| `block_05_attn_proj_group_08_256_288` | `attn_proj_group` | `pretrained.blocks.5.attn` | 256:288 | -3 | 0.1614 | 12320 |
| `block_05_v_head_02` | `attn_v_head` | `pretrained.blocks.5.attn` | 128:192 | -3 | 0.2519 | 24640 |

## Repair

- Masked tensor values: `61568`
- LoRA trainable params: `49152`
- LoRA modules: `8`
- Merge RMS deltas: `{'pretrained.blocks.2.attn.qkv': 0.0011365555692464113, 'pretrained.blocks.2.attn.proj': 0.001113878795877099, 'pretrained.blocks.2.mlp.fc1': 0.0010906432289630175, 'pretrained.blocks.2.mlp.fc2': 0.0005762128275819123, 'pretrained.blocks.4.attn.qkv': 0.001098816515877843, 'pretrained.blocks.4.attn.proj': 0.0010216227965429425, 'pretrained.blocks.4.mlp.fc1': 0.0011187008349224925, 'pretrained.blocks.4.mlp.fc2': 0.0005361250368878245}`
