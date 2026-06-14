# Circuit PEFT Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9288 | 248 | 267 |
| `pruned_student` | 0.8352 | 223 | 267 |
| `peft_repaired_unmerged` | 0.8951 | 239 | 267 |
| `folded_peft_unmasked` | 0.8951 | 239 | 267 |
| `folded_peft_remasked` | 0.8951 | 239 | 267 |

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

- PEFT method: `lora`
- Masked tensor values: `233856`
- PEFT trainable params: `147456`
- PEFT modules: `16`
- Merge RMS deltas: `{'pretrained.blocks.0.attn.qkv': 0.0028246883302927017, 'pretrained.blocks.0.attn.proj': 0.0028087012469768524, 'pretrained.blocks.1.attn.qkv': 0.002683427417650819, 'pretrained.blocks.1.attn.proj': 0.0028243844863027334, 'pretrained.blocks.2.attn.qkv': 0.0025332621298730373, 'pretrained.blocks.2.attn.proj': 0.002604431239888072, 'pretrained.blocks.3.attn.qkv': 0.002289363881573081, 'pretrained.blocks.3.attn.proj': 0.0023519406095147133, 'pretrained.blocks.4.attn.qkv': 0.0024172523990273476, 'pretrained.blocks.4.attn.proj': 0.002418805379420519, 'pretrained.blocks.6.attn.qkv': 0.0022646072320640087, 'pretrained.blocks.6.attn.proj': 0.002091522328555584, 'pretrained.blocks.7.attn.qkv': 0.002288280986249447, 'pretrained.blocks.7.attn.proj': 0.0022707050666213036, 'pretrained.blocks.9.attn.qkv': 0.0020994790829718113, 'pretrained.blocks.9.attn.proj': 0.002315841382369399}`
