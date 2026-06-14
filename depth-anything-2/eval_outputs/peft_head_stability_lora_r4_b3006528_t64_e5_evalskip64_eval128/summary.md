# Circuit PEFT Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9288 | 248 | 267 |
| `pruned_student` | 0.8914 | 238 | 267 |
| `peft_repaired_unmerged` | 0.8652 | 231 | 267 |
| `folded_peft_unmasked` | 0.8652 | 231 | 267 |
| `folded_peft_remasked` | 0.8652 | 231 | 267 |

## Selected Circuits

| name | kind | module | range | correct_drop | abs_margin_delta | params |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `depth_head_scratch_refinenet4_resConfUnit1_conv1_in_group_00_0_16` | `head_input_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv1` | 0:16 | 0 | 0.0000 | 9216 |
| `depth_head_scratch_refinenet4_resConfUnit1_conv1_in_group_01_16_32` | `head_input_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv1` | 16:32 | 0 | 0.0000 | 9216 |
| `depth_head_scratch_refinenet4_resConfUnit1_conv1_in_group_02_32_48` | `head_input_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv1` | 32:48 | 0 | 0.0000 | 9216 |
| `depth_head_scratch_refinenet4_resConfUnit1_conv1_in_group_03_48_64` | `head_input_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv1` | 48:64 | 0 | 0.0000 | 9216 |
| `depth_head_scratch_refinenet4_resConfUnit1_conv2_in_group_00_0_16` | `head_input_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv2` | 0:16 | 0 | 0.0000 | 9216 |
| `depth_head_scratch_refinenet4_resConfUnit1_conv2_in_group_01_16_32` | `head_input_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv2` | 16:32 | 0 | 0.0000 | 9216 |
| `depth_head_scratch_refinenet4_resConfUnit1_conv2_in_group_02_32_48` | `head_input_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv2` | 32:48 | 0 | 0.0000 | 9216 |
| `depth_head_scratch_refinenet4_resConfUnit1_conv2_in_group_03_48_64` | `head_input_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv2` | 48:64 | 0 | 0.0000 | 9216 |
| `depth_head_scratch_refinenet4_resConfUnit1_conv1_out_group_00_0_16` | `head_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv1` | 0:16 | 0 | 0.0000 | 9232 |
| `depth_head_scratch_refinenet4_resConfUnit1_conv1_out_group_01_16_32` | `head_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv1` | 16:32 | 0 | 0.0000 | 9232 |
| `depth_head_scratch_refinenet4_resConfUnit1_conv1_out_group_02_32_48` | `head_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv1` | 32:48 | 0 | 0.0000 | 9232 |
| `depth_head_scratch_refinenet4_resConfUnit1_conv1_out_group_03_48_64` | `head_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv1` | 48:64 | 0 | 0.0000 | 9232 |
| `depth_head_scratch_refinenet4_resConfUnit1_conv2_out_group_00_0_16` | `head_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv2` | 0:16 | 0 | 0.0000 | 9232 |
| `depth_head_scratch_refinenet4_resConfUnit1_conv2_out_group_01_16_32` | `head_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv2` | 16:32 | 0 | 0.0000 | 9232 |
| `depth_head_scratch_refinenet4_resConfUnit1_conv2_out_group_02_32_48` | `head_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv2` | 32:48 | 0 | 0.0000 | 9232 |
| `depth_head_scratch_refinenet4_resConfUnit1_conv2_out_group_03_48_64` | `head_channel_group` | `depth_head.scratch.refinenet4.resConfUnit1.conv2` | 48:64 | 0 | 0.0000 | 9232 |
| `depth_head_resize_layers_3_out_group_10_160_176` | `head_channel_group` | `depth_head.resize_layers.3` | 160:176 | 0 | 0.0134 | 55312 |
| `depth_head_resize_layers_3_out_group_04_64_80` | `head_channel_group` | `depth_head.resize_layers.3` | 64:80 | 0 | 0.0148 | 55312 |
| `depth_head_resize_layers_3_out_group_01_16_32` | `head_channel_group` | `depth_head.resize_layers.3` | 16:32 | 0 | 0.0204 | 55312 |
| `depth_head_resize_layers_3_out_group_05_80_96` | `head_channel_group` | `depth_head.resize_layers.3` | 80:96 | 0 | 0.0214 | 55312 |
| `depth_head_resize_layers_3_out_group_18_288_304` | `head_channel_group` | `depth_head.resize_layers.3` | 288:304 | 0 | 0.0215 | 55312 |
| `depth_head_resize_layers_3_out_group_06_96_112` | `head_channel_group` | `depth_head.resize_layers.3` | 96:112 | 0 | 0.0222 | 55312 |
| `depth_head_resize_layers_3_out_group_15_240_256` | `head_channel_group` | `depth_head.resize_layers.3` | 240:256 | 0 | 0.0230 | 55312 |
| `depth_head_resize_layers_3_out_group_13_208_224` | `head_channel_group` | `depth_head.resize_layers.3` | 208:224 | 0 | 0.0251 | 55312 |
| `block_00_mlp_group_06_192_224` | `mlp_group` | `pretrained.blocks.0.mlp` | 192:224 | 0 | 0.0119 | 24608 |
| `depth_head_resize_layers_3_out_group_20_320_336` | `head_channel_group` | `depth_head.resize_layers.3` | 320:336 | 0 | 0.0270 | 55312 |
| `depth_head_resize_layers_3_out_group_02_32_48` | `head_channel_group` | `depth_head.resize_layers.3` | 32:48 | 0 | 0.0277 | 55312 |
| `block_00_mlp_group_36_1152_1184` | `mlp_group` | `pretrained.blocks.0.mlp` | 1152:1184 | 0 | 0.0126 | 24608 |
| `block_00_mlp_group_25_800_832` | `mlp_group` | `pretrained.blocks.0.mlp` | 800:832 | 0 | 0.0128 | 24608 |
| `depth_head_resize_layers_3_out_group_11_176_192` | `head_channel_group` | `depth_head.resize_layers.3` | 176:192 | 0 | 0.0296 | 55312 |
| `depth_head_resize_layers_3_out_group_16_256_272` | `head_channel_group` | `depth_head.resize_layers.3` | 256:272 | 0 | 0.0306 | 55312 |
| `depth_head_resize_layers_3_in_group_14_224_240` | `head_input_channel_group` | `depth_head.resize_layers.3` | 224:240 | 0 | 0.0307 | 55296 |
| `depth_head_resize_layers_3_in_group_12_192_208` | `head_input_channel_group` | `depth_head.resize_layers.3` | 192:208 | 0 | 0.0310 | 55296 |
| `block_00_mlp_group_05_160_192` | `mlp_group` | `pretrained.blocks.0.mlp` | 160:192 | 0 | 0.0140 | 24608 |
| `block_00_mlp_group_18_576_608` | `mlp_group` | `pretrained.blocks.0.mlp` | 576:608 | 0 | 0.0141 | 24608 |
| `depth_head_resize_layers_3_in_group_01_16_32` | `head_input_channel_group` | `depth_head.resize_layers.3` | 16:32 | 0 | 0.0318 | 55296 |
| `depth_head_resize_layers_3_in_group_03_48_64` | `head_input_channel_group` | `depth_head.resize_layers.3` | 48:64 | 0 | 0.0322 | 55296 |
| `block_00_mlp_group_38_1216_1248` | `mlp_group` | `pretrained.blocks.0.mlp` | 1216:1248 | 0 | 0.0144 | 24608 |
| `block_00_mlp_group_10_320_352` | `mlp_group` | `pretrained.blocks.0.mlp` | 320:352 | 0 | 0.0145 | 24608 |
| `depth_head_resize_layers_3_in_group_00_0_16` | `head_input_channel_group` | `depth_head.resize_layers.3` | 0:16 | 0 | 0.0327 | 55296 |
| `block_00_mlp_group_28_896_928` | `mlp_group` | `pretrained.blocks.0.mlp` | 896:928 | 0 | 0.0148 | 24608 |
| `block_11_mlp_group_41_1312_1344` | `mlp_group` | `pretrained.blocks.11.mlp` | 1312:1344 | 0 | 0.0148 | 24608 |
| `block_11_mlp_group_37_1184_1216` | `mlp_group` | `pretrained.blocks.11.mlp` | 1184:1216 | 0 | 0.0148 | 24608 |
| `depth_head_resize_layers_3_in_group_02_32_48` | `head_input_channel_group` | `depth_head.resize_layers.3` | 32:48 | 0 | 0.0333 | 55296 |
| `depth_head_resize_layers_3_in_group_19_304_320` | `head_input_channel_group` | `depth_head.resize_layers.3` | 304:320 | 0 | 0.0334 | 55296 |
| `block_00_v_head_05` | `attn_v_head` | `pretrained.blocks.0.attn` | 320:384 | 0 | 0.0153 | 24640 |
| `block_11_mlp_group_45_1440_1472` | `mlp_group` | `pretrained.blocks.11.mlp` | 1440:1472 | 0 | 0.0154 | 24608 |
| `block_00_mlp_group_35_1120_1152` | `mlp_group` | `pretrained.blocks.0.mlp` | 1120:1152 | 0 | 0.0156 | 24608 |
| `block_00_mlp_group_46_1472_1504` | `mlp_group` | `pretrained.blocks.0.mlp` | 1472:1504 | 0 | 0.0158 | 24608 |
| `block_00_mlp_group_23_736_768` | `mlp_group` | `pretrained.blocks.0.mlp` | 736:768 | 0 | 0.0161 | 24608 |
| `depth_head_resize_layers_3_out_group_14_224_240` | `head_channel_group` | `depth_head.resize_layers.3` | 224:240 | 0 | 0.0363 | 55312 |
| `depth_head_resize_layers_3_in_group_15_240_256` | `head_input_channel_group` | `depth_head.resize_layers.3` | 240:256 | 0 | 0.0364 | 55296 |
| `block_00_mlp_group_16_512_544` | `mlp_group` | `pretrained.blocks.0.mlp` | 512:544 | 0 | 0.0163 | 24608 |
| `block_00_mlp_group_07_224_256` | `mlp_group` | `pretrained.blocks.0.mlp` | 224:256 | 0 | 0.0166 | 24608 |
| `block_00_mlp_group_22_704_736` | `mlp_group` | `pretrained.blocks.0.mlp` | 704:736 | 0 | 0.0169 | 24608 |
| `block_11_mlp_group_11_352_384` | `mlp_group` | `pretrained.blocks.11.mlp` | 352:384 | 0 | 0.0169 | 24608 |
| `depth_head_resize_layers_3_in_group_13_208_224` | `head_input_channel_group` | `depth_head.resize_layers.3` | 208:224 | 0 | 0.0380 | 55296 |
| `depth_head_resize_layers_3_in_group_23_368_384` | `head_input_channel_group` | `depth_head.resize_layers.3` | 368:384 | 0 | 0.0381 | 55296 |
| `block_00_mlp_group_45_1440_1472` | `mlp_group` | `pretrained.blocks.0.mlp` | 1440:1472 | 0 | 0.0171 | 24608 |
| `depth_head_resize_layers_3_in_group_07_112_128` | `head_input_channel_group` | `depth_head.resize_layers.3` | 112:128 | 0 | 0.0386 | 55296 |
| `block_00_mlp_group_31_992_1024` | `mlp_group` | `pretrained.blocks.0.mlp` | 992:1024 | 0 | 0.0173 | 24608 |
| `block_00_mlp_group_30_960_992` | `mlp_group` | `pretrained.blocks.0.mlp` | 960:992 | 0 | 0.0173 | 24608 |
| `block_00_mlp_group_34_1088_1120` | `mlp_group` | `pretrained.blocks.0.mlp` | 1088:1120 | 0 | 0.0173 | 24608 |
| `depth_head_resize_layers_3_out_group_23_368_384` | `head_channel_group` | `depth_head.resize_layers.3` | 368:384 | 0 | 0.0390 | 55312 |
| `depth_head_resize_layers_3_out_group_00_0_16` | `head_channel_group` | `depth_head.resize_layers.3` | 0:16 | 0 | 0.0391 | 55312 |
| `block_00_mlp_group_17_544_576` | `mlp_group` | `pretrained.blocks.0.mlp` | 544:576 | 0 | 0.0176 | 24608 |
| `block_00_mlp_group_02_64_96` | `mlp_group` | `pretrained.blocks.0.mlp` | 64:96 | 0 | 0.0177 | 24608 |
| `block_11_mlp_group_04_128_160` | `mlp_group` | `pretrained.blocks.11.mlp` | 128:160 | 0 | 0.0178 | 24608 |
| `depth_head_resize_layers_3_in_group_18_288_304` | `head_input_channel_group` | `depth_head.resize_layers.3` | 288:304 | 0 | 0.0402 | 55296 |
| `block_11_mlp_group_20_640_672` | `mlp_group` | `pretrained.blocks.11.mlp` | 640:672 | 0 | 0.0180 | 24608 |
| `block_00_mlp_group_14_448_480` | `mlp_group` | `pretrained.blocks.0.mlp` | 448:480 | 0 | 0.0180 | 24608 |
| `block_00_mlp_group_09_288_320` | `mlp_group` | `pretrained.blocks.0.mlp` | 288:320 | 0 | 0.0181 | 24608 |
| `block_00_mlp_group_37_1184_1216` | `mlp_group` | `pretrained.blocks.0.mlp` | 1184:1216 | 0 | 0.0181 | 24608 |
| `depth_head_resize_layers_3_in_group_10_160_176` | `head_input_channel_group` | `depth_head.resize_layers.3` | 160:176 | 0 | 0.0410 | 55296 |
| `block_00_mlp_group_39_1248_1280` | `mlp_group` | `pretrained.blocks.0.mlp` | 1248:1280 | 0 | 0.0184 | 24608 |
| `block_00_mlp_group_33_1056_1088` | `mlp_group` | `pretrained.blocks.0.mlp` | 1056:1088 | 0 | 0.0187 | 24608 |
| `block_00_mlp_group_44_1408_1440` | `mlp_group` | `pretrained.blocks.0.mlp` | 1408:1440 | 0 | 0.0188 | 24608 |
| `block_01_mlp_group_12_384_416` | `mlp_group` | `pretrained.blocks.1.mlp` | 384:416 | 0 | 0.0189 | 24608 |
| `block_01_mlp_group_10_320_352` | `mlp_group` | `pretrained.blocks.1.mlp` | 320:352 | 0 | 0.0190 | 24608 |
| `block_11_mlp_group_00_0_32` | `mlp_group` | `pretrained.blocks.11.mlp` | 0:32 | 0 | 0.0191 | 24608 |
| `block_00_mlp_group_43_1376_1408` | `mlp_group` | `pretrained.blocks.0.mlp` | 1376:1408 | 0 | 0.0194 | 24608 |
| `block_00_mlp_group_40_1280_1312` | `mlp_group` | `pretrained.blocks.0.mlp` | 1280:1312 | 0 | 0.0194 | 24608 |
| `block_11_mlp_group_40_1280_1312` | `mlp_group` | `pretrained.blocks.11.mlp` | 1280:1312 | 0 | 0.0195 | 24608 |
| `block_00_mlp_group_47_1504_1536` | `mlp_group` | `pretrained.blocks.0.mlp` | 1504:1536 | 0 | 0.0195 | 24608 |
| `block_00_mlp_group_24_768_800` | `mlp_group` | `pretrained.blocks.0.mlp` | 768:800 | 0 | 0.0196 | 24608 |
| `block_00_mlp_group_20_640_672` | `mlp_group` | `pretrained.blocks.0.mlp` | 640:672 | 0 | 0.0198 | 24608 |
| `block_01_mlp_group_37_1184_1216` | `mlp_group` | `pretrained.blocks.1.mlp` | 1184:1216 | 0 | 0.0198 | 24608 |
| `block_01_mlp_group_30_960_992` | `mlp_group` | `pretrained.blocks.1.mlp` | 960:992 | 0 | 0.0200 | 24608 |
| `block_00_mlp_group_03_96_128` | `mlp_group` | `pretrained.blocks.0.mlp` | 96:128 | 0 | 0.0201 | 24608 |
| `block_00_mlp_group_29_928_960` | `mlp_group` | `pretrained.blocks.0.mlp` | 928:960 | 0 | 0.0201 | 24608 |
| `block_00_k_head_05` | `attn_k_head` | `pretrained.blocks.0.attn` | 320:384 | 0 | 0.0202 | 24640 |
| `block_00_q_head_05` | `attn_q_head` | `pretrained.blocks.0.attn` | 320:384 | 0 | 0.0202 | 24640 |
| `block_11_mlp_group_05_160_192` | `mlp_group` | `pretrained.blocks.11.mlp` | 160:192 | 0 | 0.0202 | 24608 |
| `block_00_mlp_group_11_352_384` | `mlp_group` | `pretrained.blocks.0.mlp` | 352:384 | 0 | 0.0203 | 24608 |
| `block_01_mlp_group_32_1024_1056` | `mlp_group` | `pretrained.blocks.1.mlp` | 1024:1056 | 0 | 0.0203 | 24608 |
| `depth_head_resize_layers_3_out_group_03_48_64` | `head_channel_group` | `depth_head.resize_layers.3` | 48:64 | 0 | 0.0461 | 55312 |

## Repair

- PEFT method: `lora`
- Train/eval overlap images: `0`
- Masked tensor values: `3006528`
- PEFT trainable params: `103364`
- PEFT modules: `34`
- Merge RMS deltas: `{'depth_head.projects.0': 0.002893665339797735, 'depth_head.projects.1': 0.0033191198017448187, 'depth_head.projects.2': 0.003605773439630866, 'depth_head.projects.3': 0.0026334819849580526, 'depth_head.resize_layers.0': 0.0037417460698634386, 'depth_head.resize_layers.1': 0.0036298090126365423, 'depth_head.resize_layers.3': 0.0011137158144265413, 'depth_head.scratch.layer1_rn': 0.0012535860296338797, 'depth_head.scratch.layer2_rn': 0.001360538532026112, 'depth_head.scratch.layer3_rn': 0.002080586040392518, 'depth_head.scratch.layer4_rn': 0.0014695758000016212, 'depth_head.scratch.refinenet1.out_conv': 0.002341608749702573, 'depth_head.scratch.refinenet1.resConfUnit1.conv1': 0.0024601437617093325, 'depth_head.scratch.refinenet1.resConfUnit1.conv2': 0.0008098558755591512, 'depth_head.scratch.refinenet1.resConfUnit2.conv1': 0.0011264616623520851, 'depth_head.scratch.refinenet1.resConfUnit2.conv2': 0.0007060224306769669, 'depth_head.scratch.refinenet2.out_conv': 0.002202581847086549, 'depth_head.scratch.refinenet2.resConfUnit1.conv1': 0.0018770396709442139, 'depth_head.scratch.refinenet2.resConfUnit1.conv2': 0.0010847491212189198, 'depth_head.scratch.refinenet2.resConfUnit2.conv1': 0.001162672066129744, 'depth_head.scratch.refinenet2.resConfUnit2.conv2': 0.0007241471321322024, 'depth_head.scratch.refinenet3.out_conv': 0.0032810659613460302, 'depth_head.scratch.refinenet3.resConfUnit1.conv1': 0.0024024583399295807, 'depth_head.scratch.refinenet3.resConfUnit1.conv2': 0.001555121736600995, 'depth_head.scratch.refinenet3.resConfUnit2.conv1': 0.0011285434011369944, 'depth_head.scratch.refinenet3.resConfUnit2.conv2': 0.001341828377917409, 'depth_head.scratch.refinenet4.out_conv': 0.002832122379913926, 'depth_head.scratch.refinenet4.resConfUnit1.conv1': 0.0, 'depth_head.scratch.refinenet4.resConfUnit1.conv2': 0.0, 'depth_head.scratch.refinenet4.resConfUnit2.conv1': 0.001952220220118761, 'depth_head.scratch.refinenet4.resConfUnit2.conv2': 0.0025173951871693134, 'depth_head.scratch.output_conv1': 0.0006328127929009497, 'depth_head.scratch.output_conv2.0': 0.0009304271079599857, 'depth_head.scratch.output_conv2.2': 0.0005277377786114812}`
