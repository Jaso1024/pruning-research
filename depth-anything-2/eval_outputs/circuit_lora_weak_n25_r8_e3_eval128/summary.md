# Circuit LoRA Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9066 | 233 | 257 |
| `pruned_student` | 0.6304 | 162 | 257 |
| `lora_repaired_unmerged` | 0.6070 | 156 | 257 |
| `folded_lora_unmasked` | 0.6070 | 156 | 257 |
| `folded_lora_remasked` | 0.5837 | 150 | 257 |

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
| `block_04_attn_proj_group_04_128_160` | `attn_proj_group` | `pretrained.blocks.4.attn` | 128:160 | -2 | 0.0774 | 12320 |
| `block_05_mlp_group_26_832_864` | `mlp_group` | `pretrained.blocks.5.mlp` | 832:864 | -2 | 0.0893 | 24608 |
| `block_04_v_head_02` | `attn_v_head` | `pretrained.blocks.4.attn` | 128:192 | -2 | 0.1526 | 24640 |
| `block_07_k_head_01` | `attn_k_head` | `pretrained.blocks.7.attn` | 64:128 | -2 | 0.1633 | 24640 |
| `block_07_q_head_01` | `attn_q_head` | `pretrained.blocks.7.attn` | 64:128 | -2 | 0.1633 | 24640 |
| `block_06_attn_proj_group_08_256_288` | `attn_proj_group` | `pretrained.blocks.6.attn` | 256:288 | -2 | 0.1751 | 12320 |
| `block_03_attn_proj_group_00_0_32` | `attn_proj_group` | `pretrained.blocks.3.attn` | 0:32 | -2 | 0.2189 | 12320 |
| `block_07_attn_proj_group_08_256_288` | `attn_proj_group` | `pretrained.blocks.7.attn` | 256:288 | -2 | 0.2954 | 12320 |
| `block_06_v_head_00` | `attn_v_head` | `pretrained.blocks.6.attn` | 0:64 | -2 | 0.2972 | 24640 |
| `block_05_k_head_05` | `attn_k_head` | `pretrained.blocks.5.attn` | 320:384 | -2 | 0.3674 | 24640 |
| `block_05_q_head_05` | `attn_q_head` | `pretrained.blocks.5.attn` | 320:384 | -2 | 0.3674 | 24640 |
| `block_06_k_head_00` | `attn_k_head` | `pretrained.blocks.6.attn` | 0:64 | -2 | 0.4142 | 24640 |
| `block_06_q_head_00` | `attn_q_head` | `pretrained.blocks.6.attn` | 0:64 | -2 | 0.4142 | 24640 |
| `block_03_attn_proj_group_10_320_352` | `attn_proj_group` | `pretrained.blocks.3.attn` | 320:352 | -1 | 0.0535 | 12320 |
| `block_06_v_head_04` | `attn_v_head` | `pretrained.blocks.6.attn` | 256:320 | -1 | 0.1234 | 24640 |

## Repair

- Masked tensor values: `541824`
- LoRA trainable params: `215040`
- LoRA modules: `18`
- Merge RMS deltas: `{'pretrained.blocks.1.mlp.fc1': 0.00640473235398531, 'pretrained.blocks.1.mlp.fc2': 0.0020722292829304934, 'pretrained.blocks.2.mlp.fc1': 0.005158089101314545, 'pretrained.blocks.2.mlp.fc2': 0.00344369956292212, 'pretrained.blocks.3.attn.proj': 0.0041136606596410275, 'pretrained.blocks.3.mlp.fc1': 0.0033597168512642384, 'pretrained.blocks.3.mlp.fc2': 0.0028472067788243294, 'pretrained.blocks.4.attn.qkv': 0.003900353331118822, 'pretrained.blocks.4.attn.proj': 0.0037896011490374804, 'pretrained.blocks.5.attn.qkv': 0.0035630916245281696, 'pretrained.blocks.5.attn.proj': 0.004295293241739273, 'pretrained.blocks.5.mlp.fc1': 0.002864174311980605, 'pretrained.blocks.5.mlp.fc2': 0.0025549072306603193, 'pretrained.blocks.6.attn.qkv': 0.002974133938550949, 'pretrained.blocks.6.attn.proj': 0.003499084385111928, 'pretrained.blocks.7.attn.qkv': 0.002613142365589738, 'pretrained.blocks.7.attn.proj': 0.0030194988939911127, 'pretrained.blocks.8.attn.qkv': 0.004164490848779678}`
