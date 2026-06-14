# Circuit PEFT Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9450 | 292 | 309 |
| `pruned_student` | 0.9159 | 283 | 309 |
| `peft_repaired_unmerged` | 0.9353 | 289 | 309 |
| `folded_peft_unmasked` | 0.9353 | 289 | 309 |
| `folded_peft_remasked` | 0.9353 | 289 | 309 |

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

## Repair

- PEFT method: `lora`
- Train/eval overlap images: `0`
- Masked tensor values: `2016304`
- PEFT trainable params: `103364`
- PEFT modules: `34`
- Merge RMS deltas: `{'depth_head.projects.0': 0.002020350657403469, 'depth_head.projects.1': 0.002299706917256117, 'depth_head.projects.2': 0.0025707310996949673, 'depth_head.projects.3': 0.001850705360993743, 'depth_head.resize_layers.0': 0.0022779987193644047, 'depth_head.resize_layers.1': 0.0027002375572919846, 'depth_head.resize_layers.3': 0.0010733945528045297, 'depth_head.scratch.layer1_rn': 0.0009821817511692643, 'depth_head.scratch.layer2_rn': 0.0009102577459998429, 'depth_head.scratch.layer3_rn': 0.0017314094584435225, 'depth_head.scratch.layer4_rn': 0.0010272037470713258, 'depth_head.scratch.refinenet1.out_conv': 0.002890585921704769, 'depth_head.scratch.refinenet1.resConfUnit1.conv1': 0.001720395521260798, 'depth_head.scratch.refinenet1.resConfUnit1.conv2': 0.000685522856656462, 'depth_head.scratch.refinenet1.resConfUnit2.conv1': 0.0006549849640578032, 'depth_head.scratch.refinenet1.resConfUnit2.conv2': 0.00046608122647739947, 'depth_head.scratch.refinenet2.out_conv': 0.0019351685186848044, 'depth_head.scratch.refinenet2.resConfUnit1.conv1': 0.0013771557714790106, 'depth_head.scratch.refinenet2.resConfUnit1.conv2': 0.0007899607880972326, 'depth_head.scratch.refinenet2.resConfUnit2.conv1': 0.0009564361535012722, 'depth_head.scratch.refinenet2.resConfUnit2.conv2': 0.0006547983502969146, 'depth_head.scratch.refinenet3.out_conv': 0.003023243974894285, 'depth_head.scratch.refinenet3.resConfUnit1.conv1': 0.0014685705536976457, 'depth_head.scratch.refinenet3.resConfUnit1.conv2': 0.0011878981022164226, 'depth_head.scratch.refinenet3.resConfUnit2.conv1': 0.001466779736801982, 'depth_head.scratch.refinenet3.resConfUnit2.conv2': 0.0011395341716706753, 'depth_head.scratch.refinenet4.out_conv': 0.0027292303275316954, 'depth_head.scratch.refinenet4.resConfUnit1.conv1': 0.0, 'depth_head.scratch.refinenet4.resConfUnit1.conv2': 0.0, 'depth_head.scratch.refinenet4.resConfUnit2.conv1': 0.0018309142906218767, 'depth_head.scratch.refinenet4.resConfUnit2.conv2': 0.0022139272186905146, 'depth_head.scratch.output_conv1': 0.0008494801004417241, 'depth_head.scratch.output_conv2.0': 0.000972039473708719, 'depth_head.scratch.output_conv2.2': 0.0006370286573655903}`
