# Circuit PEFT Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.8333 | 5 | 6 |
| `pruned_student` | 0.8333 | 5 | 6 |
| `peft_repaired_unmerged` | 0.8333 | 5 | 6 |
| `folded_peft_unmasked` | 0.8333 | 5 | 6 |
| `folded_peft_remasked` | 0.8333 | 5 | 6 |

## Selected Circuits

| name | kind | module | range | correct_drop | abs_margin_delta | params |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `block_03_mlp_group_09_288_320` | `mlp_group` | `pretrained.blocks.3.mlp` | 288:320 | -3 | 0.0689 | 24608 |
| `block_05_attn_proj_group_08_256_288` | `attn_proj_group` | `pretrained.blocks.5.attn` | 256:288 | -3 | 0.1614 | 12320 |

## Repair

- PEFT method: `lora`
- Masked tensor values: `36928`
- PEFT trainable params: `13824`
- PEFT modules: `6`
- Merge RMS deltas: `{'pretrained.blocks.2.attn.qkv': 0.00013388243678491563, 'pretrained.blocks.2.attn.proj': 0.00013453148130793124, 'pretrained.blocks.4.attn.qkv': 0.00013178632070776075, 'pretrained.blocks.4.attn.proj': 0.00012808502651751041, 'pretrained.blocks.6.attn.qkv': 0.00013385136844590306, 'pretrained.blocks.6.attn.proj': 0.00012986446381546557}`
