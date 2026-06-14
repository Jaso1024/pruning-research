# Circuit LoRA Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 1.0000 | 19 | 19 |
| `pruned_student` | 1.0000 | 19 | 19 |
| `lora_repaired_unmerged` | 0.9474 | 18 | 19 |
| `folded_lora_unmasked` | 0.9474 | 18 | 19 |
| `folded_lora_remasked` | 0.9474 | 18 | 19 |

## Selected Circuits

| name | kind | module | range | correct_drop | abs_margin_delta | params |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `block_03_mlp_group_09_288_320` | `mlp_group` | `pretrained.blocks.3.mlp` | 288:320 | -3 | 0.0689 | 24608 |
| `block_05_attn_proj_group_08_256_288` | `attn_proj_group` | `pretrained.blocks.5.attn` | 256:288 | -3 | 0.1614 | 12320 |
| `block_05_v_head_02` | `attn_v_head` | `pretrained.blocks.5.attn` | 128:192 | -3 | 0.2519 | 24640 |

## Repair

- Masked tensor values: `61568`
- LoRA trainable params: `24576`
- LoRA modules: `4`
- Merge RMS deltas: `{'pretrained.blocks.3.mlp.fc1': 0.0010692481882870197, 'pretrained.blocks.3.mlp.fc2': 0.0005258211749605834, 'pretrained.blocks.5.attn.qkv': 0.001042557181790471, 'pretrained.blocks.5.attn.proj': 0.0010727191111072898}`
