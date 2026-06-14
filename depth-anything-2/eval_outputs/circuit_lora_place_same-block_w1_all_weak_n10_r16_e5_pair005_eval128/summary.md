# Circuit LoRA Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9288 | 248 | 267 |
| `pruned_student` | 0.8352 | 223 | 267 |
| `lora_repaired_unmerged` | 0.7640 | 204 | 267 |
| `folded_lora_unmasked` | 0.7640 | 204 | 267 |
| `folded_lora_remasked` | 0.7640 | 204 | 267 |

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
- LoRA trainable params: `491520`
- LoRA modules: `20`
- Merge RMS deltas: `{'pretrained.blocks.1.attn.qkv': 0.00380338984541595, 'pretrained.blocks.1.attn.proj': 0.003699099412187934, 'pretrained.blocks.1.mlp.fc1': 0.0033239983022212982, 'pretrained.blocks.1.mlp.fc2': 0.0018452279036864638, 'pretrained.blocks.2.attn.qkv': 0.003573063528165221, 'pretrained.blocks.2.attn.proj': 0.003649400547146797, 'pretrained.blocks.2.mlp.fc1': 0.0031485927756875753, 'pretrained.blocks.2.mlp.fc2': 0.001755037810653448, 'pretrained.blocks.3.attn.qkv': 0.003495696932077408, 'pretrained.blocks.3.attn.proj': 0.003384567331522703, 'pretrained.blocks.3.mlp.fc1': 0.003102616872638464, 'pretrained.blocks.3.mlp.fc2': 0.0016842253971844912, 'pretrained.blocks.5.attn.qkv': 0.00369621766731143, 'pretrained.blocks.5.attn.proj': 0.0033705015666782856, 'pretrained.blocks.5.mlp.fc1': 0.003528023837134242, 'pretrained.blocks.5.mlp.fc2': 0.0018254733877256513, 'pretrained.blocks.8.attn.qkv': 0.0038349786773324013, 'pretrained.blocks.8.attn.proj': 0.0032093538902699947, 'pretrained.blocks.8.mlp.fc1': 0.0037365190219134092, 'pretrained.blocks.8.mlp.fc2': 0.0018059442518278956}`
