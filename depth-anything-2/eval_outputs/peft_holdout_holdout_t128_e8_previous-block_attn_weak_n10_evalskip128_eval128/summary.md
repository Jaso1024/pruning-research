# Circuit PEFT Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9450 | 292 | 309 |
| `pruned_student` | 0.8511 | 263 | 309 |
| `peft_repaired_unmerged` | 0.8964 | 277 | 309 |
| `folded_peft_unmasked` | 0.8964 | 277 | 309 |
| `folded_peft_remasked` | 0.8964 | 277 | 309 |

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
- PEFT trainable params: `92160`
- PEFT modules: `10`
- Merge RMS deltas: `{'pretrained.blocks.0.attn.qkv': 0.005311538465321064, 'pretrained.blocks.0.attn.proj': 0.003997946158051491, 'pretrained.blocks.1.attn.qkv': 0.004939124453812838, 'pretrained.blocks.1.attn.proj': 0.0041537098586559296, 'pretrained.blocks.2.attn.qkv': 0.004430096596479416, 'pretrained.blocks.2.attn.proj': 0.0038109850138425827, 'pretrained.blocks.4.attn.qkv': 0.004206192214041948, 'pretrained.blocks.4.attn.proj': 0.003742275992408395, 'pretrained.blocks.7.attn.qkv': 0.003970644902437925, 'pretrained.blocks.7.attn.proj': 0.003838073695078492}`
