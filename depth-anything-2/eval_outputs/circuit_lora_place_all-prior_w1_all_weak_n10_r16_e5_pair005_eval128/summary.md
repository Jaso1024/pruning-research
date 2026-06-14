# Circuit LoRA Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9288 | 248 | 267 |
| `pruned_student` | 0.8352 | 223 | 267 |
| `lora_repaired_unmerged` | 0.5993 | 160 | 267 |
| `folded_lora_unmasked` | 0.6030 | 161 | 267 |
| `folded_lora_remasked` | 0.5993 | 160 | 267 |

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
- LoRA trainable params: `786432`
- LoRA modules: `32`
- Merge RMS deltas: `{'pretrained.blocks.0.attn.qkv': 0.008373171091079712, 'pretrained.blocks.0.attn.proj': 0.006532470230013132, 'pretrained.blocks.0.mlp.fc1': 0.006415668874979019, 'pretrained.blocks.0.mlp.fc2': 0.005645494908094406, 'pretrained.blocks.1.attn.qkv': 0.007847139611840248, 'pretrained.blocks.1.attn.proj': 0.005268893204629421, 'pretrained.blocks.1.mlp.fc1': 0.004561132751405239, 'pretrained.blocks.1.mlp.fc2': 0.0032307435758411884, 'pretrained.blocks.2.attn.qkv': 0.0038244975730776787, 'pretrained.blocks.2.attn.proj': 0.003456431208178401, 'pretrained.blocks.2.mlp.fc1': 0.0030232523567974567, 'pretrained.blocks.2.mlp.fc2': 0.0017628022469580173, 'pretrained.blocks.3.attn.qkv': 0.003934867214411497, 'pretrained.blocks.3.attn.proj': 0.0031517420429736376, 'pretrained.blocks.3.mlp.fc1': 0.0025430195964872837, 'pretrained.blocks.3.mlp.fc2': 0.0015233347658067942, 'pretrained.blocks.4.attn.qkv': 0.002344404812902212, 'pretrained.blocks.4.attn.proj': 0.0026683255564421415, 'pretrained.blocks.4.mlp.fc1': 0.0028876157011836767, 'pretrained.blocks.4.mlp.fc2': 0.0016853323904797435, 'pretrained.blocks.5.attn.qkv': 0.0026703234761953354, 'pretrained.blocks.5.attn.proj': 0.002897405531257391, 'pretrained.blocks.5.mlp.fc1': 0.0026239107828587294, 'pretrained.blocks.5.mlp.fc2': 0.002288553398102522, 'pretrained.blocks.6.attn.qkv': 0.0022673835046589375, 'pretrained.blocks.6.attn.proj': 0.002697937423363328, 'pretrained.blocks.6.mlp.fc1': 0.0026849471032619476, 'pretrained.blocks.6.mlp.fc2': 0.0028283654246479273, 'pretrained.blocks.7.attn.qkv': 0.002482497598975897, 'pretrained.blocks.7.attn.proj': 0.0036566692870110273, 'pretrained.blocks.7.mlp.fc1': 0.002798945875838399, 'pretrained.blocks.7.mlp.fc2': 0.005179269704967737}`
