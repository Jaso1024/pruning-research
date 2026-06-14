# Circuit PEFT Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9288 | 248 | 267 |
| `pruned_student` | 0.8352 | 223 | 267 |
| `peft_repaired_unmerged` | 0.8689 | 232 | 267 |
| `folded_peft_unmasked` | 0.8689 | 232 | 267 |
| `folded_peft_remasked` | 0.8689 | 232 | 267 |

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
- PEFT trainable params: `184320`
- PEFT modules: `20`
- Merge RMS deltas: `{'pretrained.blocks.0.attn.qkv': 0.003276947420090437, 'pretrained.blocks.0.attn.proj': 0.0029563908465206623, 'pretrained.blocks.1.attn.qkv': 0.0028238899540156126, 'pretrained.blocks.1.attn.proj': 0.002848116448149085, 'pretrained.blocks.2.attn.qkv': 0.002542200032621622, 'pretrained.blocks.2.attn.proj': 0.0025703187566250563, 'pretrained.blocks.3.attn.qkv': 0.0023941381368786097, 'pretrained.blocks.3.attn.proj': 0.0023946180008351803, 'pretrained.blocks.4.attn.qkv': 0.002419832395389676, 'pretrained.blocks.4.attn.proj': 0.002377672353759408, 'pretrained.blocks.5.attn.qkv': 0.0022304817102849483, 'pretrained.blocks.5.attn.proj': 0.0022013757843524218, 'pretrained.blocks.6.attn.qkv': 0.0023598410189151764, 'pretrained.blocks.6.attn.proj': 0.002147001912817359, 'pretrained.blocks.7.attn.qkv': 0.002299314131960273, 'pretrained.blocks.7.attn.proj': 0.0021619333419948816, 'pretrained.blocks.8.attn.qkv': 0.0021365166176110506, 'pretrained.blocks.8.attn.proj': 0.0021145965438336134, 'pretrained.blocks.9.attn.qkv': 0.0021649568807333708, 'pretrained.blocks.9.attn.proj': 0.0022152590099722147}`
