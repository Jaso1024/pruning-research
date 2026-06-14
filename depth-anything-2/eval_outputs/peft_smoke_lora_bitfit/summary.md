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

- PEFT method: `lora-bitfit`
- Masked tensor values: `36928`
- PEFT trainable params: `12288`
- PEFT modules: `4`
- Merge RMS deltas: `{'pretrained.blocks.2.attn.qkv': 0.00013402204785961658, 'pretrained.blocks.2.attn.proj': 0.00013449459220282733, 'pretrained.blocks.4.attn.qkv': 0.00013166229473426938, 'pretrained.blocks.4.attn.proj': 0.00012821391283068806}`
