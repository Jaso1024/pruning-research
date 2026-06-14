# Circuit LoRA Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9288 | 248 | 267 |
| `pruned_student` | 0.8352 | 223 | 267 |
| `lora_repaired_unmerged` | 0.0000 | 0 | 267 |
| `folded_lora_unmasked` | 0.0000 | 0 | 267 |
| `folded_lora_remasked` | 0.0000 | 0 | 267 |

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
- LoRA trainable params: `983040`
- LoRA modules: `40`
- Merge RMS deltas: `{'pretrained.blocks.0.attn.qkv': 0.007755848579108715, 'pretrained.blocks.0.attn.proj': 0.004720910452306271, 'pretrained.blocks.0.mlp.fc1': 0.005501448176801205, 'pretrained.blocks.0.mlp.fc2': 0.0044610751792788506, 'pretrained.blocks.1.attn.qkv': 0.006473192945122719, 'pretrained.blocks.1.attn.proj': 0.0071790642105042934, 'pretrained.blocks.1.mlp.fc1': 0.00721804378554225, 'pretrained.blocks.1.mlp.fc2': 0.003860806580632925, 'pretrained.blocks.2.attn.qkv': 0.0037794241216033697, 'pretrained.blocks.2.attn.proj': 0.0049231271259486675, 'pretrained.blocks.2.mlp.fc1': 0.002866024151444435, 'pretrained.blocks.2.mlp.fc2': 0.0022201607935130596, 'pretrained.blocks.3.attn.qkv': 0.004235528875142336, 'pretrained.blocks.3.attn.proj': 0.0028856294229626656, 'pretrained.blocks.3.mlp.fc1': 0.0023137619718909264, 'pretrained.blocks.3.mlp.fc2': 0.0018140040338039398, 'pretrained.blocks.4.attn.qkv': 0.0022363862954080105, 'pretrained.blocks.4.attn.proj': 0.0022386168129742146, 'pretrained.blocks.4.mlp.fc1': 0.0026135784573853016, 'pretrained.blocks.4.mlp.fc2': 0.0016495701856911182, 'pretrained.blocks.5.attn.qkv': 0.0022305708844214678, 'pretrained.blocks.5.attn.proj': 0.0024793879128992558, 'pretrained.blocks.5.mlp.fc1': 0.002644595690071583, 'pretrained.blocks.5.mlp.fc2': 0.0013977824710309505, 'pretrained.blocks.6.attn.qkv': 0.002065974986180663, 'pretrained.blocks.6.attn.proj': 0.0021418528631329536, 'pretrained.blocks.6.mlp.fc1': 0.0022583589889109135, 'pretrained.blocks.6.mlp.fc2': 0.0015427402686327696, 'pretrained.blocks.7.attn.qkv': 0.002118876902386546, 'pretrained.blocks.7.attn.proj': 0.002467017387971282, 'pretrained.blocks.7.mlp.fc1': 0.0025348523631691933, 'pretrained.blocks.7.mlp.fc2': 0.0021946607157588005, 'pretrained.blocks.8.attn.qkv': 0.0022108189295977354, 'pretrained.blocks.8.attn.proj': 0.00291509204544127, 'pretrained.blocks.8.mlp.fc1': 0.003325200406834483, 'pretrained.blocks.8.mlp.fc2': 0.0035221753641963005, 'pretrained.blocks.9.attn.qkv': 0.002275418723002076, 'pretrained.blocks.9.attn.proj': 0.0036901915445923805, 'pretrained.blocks.9.mlp.fc1': 0.002948659472167492, 'pretrained.blocks.9.mlp.fc2': 0.002874810481444001}`
