# Circuit PEFT Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9288 | 248 | 267 |
| `pruned_student` | 0.8352 | 223 | 267 |
| `peft_repaired_unmerged` | 0.8652 | 231 | 267 |
| `folded_peft_unmasked` | 0.8652 | 231 | 267 |
| `folded_peft_remasked` | 0.8652 | 231 | 267 |

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
- PEFT trainable params: `393216`
- PEFT modules: `32`
- Merge RMS deltas: `{'pretrained.blocks.0.attn.qkv': 0.0028154596220701933, 'pretrained.blocks.0.attn.proj': 0.0026017678901553154, 'pretrained.blocks.0.mlp.fc1': 0.0026478085201233625, 'pretrained.blocks.0.mlp.fc2': 0.001430598320439458, 'pretrained.blocks.1.attn.qkv': 0.002738220850005746, 'pretrained.blocks.1.attn.proj': 0.0026521815452724695, 'pretrained.blocks.1.mlp.fc1': 0.0022493479773402214, 'pretrained.blocks.1.mlp.fc2': 0.0013812502147629857, 'pretrained.blocks.2.attn.qkv': 0.002583190565928817, 'pretrained.blocks.2.attn.proj': 0.0026414955500513315, 'pretrained.blocks.2.mlp.fc1': 0.002190368017181754, 'pretrained.blocks.2.mlp.fc2': 0.001275387010537088, 'pretrained.blocks.3.attn.qkv': 0.0024636711459606886, 'pretrained.blocks.3.attn.proj': 0.0023791687563061714, 'pretrained.blocks.3.mlp.fc1': 0.002115172566846013, 'pretrained.blocks.3.mlp.fc2': 0.001258114818483591, 'pretrained.blocks.4.attn.qkv': 0.0024442870635539293, 'pretrained.blocks.4.attn.proj': 0.0023608659394085407, 'pretrained.blocks.4.mlp.fc1': 0.0023189156781882048, 'pretrained.blocks.4.mlp.fc2': 0.0012733620824292302, 'pretrained.blocks.6.attn.qkv': 0.002577198902145028, 'pretrained.blocks.6.attn.proj': 0.002190985716879368, 'pretrained.blocks.6.mlp.fc1': 0.0024769676383584738, 'pretrained.blocks.6.mlp.fc2': 0.0011490787146613002, 'pretrained.blocks.7.attn.qkv': 0.0023506907746195793, 'pretrained.blocks.7.attn.proj': 0.002244594506919384, 'pretrained.blocks.7.mlp.fc1': 0.0024926725309342146, 'pretrained.blocks.7.mlp.fc2': 0.001169137074612081, 'pretrained.blocks.9.attn.qkv': 0.002302236622199416, 'pretrained.blocks.9.attn.proj': 0.0021850739140063524, 'pretrained.blocks.9.mlp.fc1': 0.0025522205978631973, 'pretrained.blocks.9.mlp.fc2': 0.0012187737738713622}`
