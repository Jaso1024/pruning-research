# Circuit LoRA Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9288 | 248 | 267 |
| `pruned_student` | 0.8352 | 223 | 267 |
| `lora_repaired_unmerged` | 0.6217 | 166 | 267 |
| `folded_lora_unmasked` | 0.6217 | 166 | 267 |
| `folded_lora_remasked` | 0.6217 | 166 | 267 |

## Selected Circuits

| name | kind | module | range | correct_drop | abs_margin_delta | params |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `block_03_mlp_group_09_288_320` | `mlp_group` | `pretrained.blocks.3.mlp` | 288:320 | -3 | 0.0689 | 24608 |
| `block_05_attn_proj_group_08_256_288` | `attn_proj_group` | `pretrained.blocks.5.attn` | 256:288 | -3 | 0.1614 | 12320 |
| `block_05_v_head_02` | `attn_v_head` | `pretrained.blocks.5.attn` | 128:192 | -3 | 0.2519 | 24640 |
| `block_08_v_head_05` | `attn_v_head` | `pretrained.blocks.8.attn` | 320:384 | -3 | 0.4155 | 24640 |
| `block_02_mlp_group_38_1216_1248` | `mlp_group` | `pretrained.blocks.2.mlp` | 1216:1248 | -2 | 0.0394 | 24608 |
| `block_03_mlp_group_27_864_896` | `mlp_group` | `pretrained.blocks.3.mlp` | 864:896 | -2 | 0.0410 | 24608 |
| `block_01_mlp_group_22_704_736` | `mlp_group` | `pretrained.blocks.1.mlp` | 704:736 | -2 | 0.0438 | 24608 |
| `block_03_mlp_group_15_480_512` | `mlp_group` | `pretrained.blocks.3.mlp` | 480:512 | -2 | 0.0512 | 24608 |
| `block_02_mlp_group_15_480_512` | `mlp_group` | `pretrained.blocks.2.mlp` | 480:512 | -2 | 0.0529 | 24608 |
| `block_05_mlp_group_33_1056_1088` | `mlp_group` | `pretrained.blocks.5.mlp` | 1056:1088 | -2 | 0.0701 | 24608 |

## Repair

- Masked tensor values: `233856`
- LoRA trainable params: `688128`
- LoRA modules: `28`
- Merge RMS deltas: `{'pretrained.blocks.0.attn.qkv': 0.012638846412301064, 'pretrained.blocks.0.attn.proj': 0.009924318641424179, 'pretrained.blocks.0.mlp.fc1': 0.005493371747434139, 'pretrained.blocks.0.mlp.fc2': 0.003507760586217046, 'pretrained.blocks.1.attn.qkv': 0.00525273522362113, 'pretrained.blocks.1.attn.proj': 0.004688142798841, 'pretrained.blocks.1.mlp.fc1': 0.004301480948925018, 'pretrained.blocks.1.mlp.fc2': 0.002791011705994606, 'pretrained.blocks.2.attn.qkv': 0.003934058360755444, 'pretrained.blocks.2.attn.proj': 0.0037554814480245113, 'pretrained.blocks.2.mlp.fc1': 0.0034182562958449125, 'pretrained.blocks.2.mlp.fc2': 0.00308608147315681, 'pretrained.blocks.3.attn.qkv': 0.0048582120798528194, 'pretrained.blocks.3.attn.proj': 0.003978237509727478, 'pretrained.blocks.3.mlp.fc1': 0.0030299080535769463, 'pretrained.blocks.3.mlp.fc2': 0.002026531845331192, 'pretrained.blocks.4.attn.qkv': 0.002787657780572772, 'pretrained.blocks.4.attn.proj': 0.0032311405520886183, 'pretrained.blocks.4.mlp.fc1': 0.0029335827566683292, 'pretrained.blocks.4.mlp.fc2': 0.0015934386756271124, 'pretrained.blocks.6.attn.qkv': 0.0029768028762191534, 'pretrained.blocks.6.attn.proj': 0.0031797580886632204, 'pretrained.blocks.6.mlp.fc1': 0.0035881763324141502, 'pretrained.blocks.6.mlp.fc2': 0.0031069410033524036, 'pretrained.blocks.7.attn.qkv': 0.0029919533990323544, 'pretrained.blocks.7.attn.proj': 0.005632875487208366, 'pretrained.blocks.7.mlp.fc1': 0.003525020554661751, 'pretrained.blocks.7.mlp.fc2': 0.002562552457675338}`
