# Circuit LoRA Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9288 | 248 | 267 |
| `pruned_student` | 0.8352 | 223 | 267 |
| `lora_repaired_unmerged` | 0.6816 | 182 | 267 |
| `folded_lora_unmasked` | 0.6816 | 182 | 267 |
| `folded_lora_remasked` | 0.6854 | 183 | 267 |

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
- LoRA trainable params: `675840`
- LoRA modules: `27`
- Merge RMS deltas: `{'pretrained.blocks.0.attn.qkv': 0.004957873839884996, 'pretrained.blocks.0.attn.proj': 0.004102169536054134, 'pretrained.blocks.0.mlp.fc1': 0.004527805373072624, 'pretrained.blocks.0.mlp.fc2': 0.002182144671678543, 'pretrained.blocks.1.attn.qkv': 0.004110859241336584, 'pretrained.blocks.1.attn.proj': 0.003628920763731003, 'pretrained.blocks.1.mlp.fc1': 0.003392455168068409, 'pretrained.blocks.1.mlp.fc2': 0.0020559525582939386, 'pretrained.blocks.2.attn.qkv': 0.003958433400839567, 'pretrained.blocks.2.attn.proj': 0.0038069915026426315, 'pretrained.blocks.2.mlp.fc1': 0.003210717113688588, 'pretrained.blocks.2.mlp.fc2': 0.00184139225166291, 'pretrained.blocks.3.mlp.fc1': 0.003208961570635438, 'pretrained.blocks.3.mlp.fc2': 0.001751960488036275, 'pretrained.blocks.4.attn.qkv': 0.003584169549867511, 'pretrained.blocks.4.attn.proj': 0.0034143440425395966, 'pretrained.blocks.4.mlp.fc1': 0.003239769721403718, 'pretrained.blocks.4.mlp.fc2': 0.0018147601513192058, 'pretrained.blocks.5.attn.qkv': 0.004008668474853039, 'pretrained.blocks.5.attn.proj': 0.003506358712911606, 'pretrained.blocks.5.mlp.fc1': 0.0035646436735987663, 'pretrained.blocks.5.mlp.fc2': 0.001951349782757461, 'pretrained.blocks.7.attn.qkv': 0.0037282658740878105, 'pretrained.blocks.7.attn.proj': 0.0033284896053373814, 'pretrained.blocks.7.mlp.fc1': 0.003931277897208929, 'pretrained.blocks.7.mlp.fc2': 0.0018316751811653376, 'pretrained.blocks.8.attn.qkv': 0.00365169788710773}`
