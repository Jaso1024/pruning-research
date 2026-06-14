# Circuit PEFT Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9288 | 248 | 267 |
| `pruned_student` | 0.9288 | 248 | 267 |
| `peft_repaired_unmerged` | 0.9363 | 250 | 267 |
| `folded_peft_unmasked` | 0.9363 | 250 | 267 |
| `folded_peft_remasked` | 0.9363 | 250 | 267 |

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

## Repair

- PEFT method: `lora`
- Train/eval overlap images: `0`
- Masked tensor values: `534768`
- PEFT trainable params: `103364`
- PEFT modules: `34`
- Merge RMS deltas: `{'depth_head.projects.0': 0.0005582374869845808, 'depth_head.projects.1': 0.000807658419944346, 'depth_head.projects.2': 0.0012382370186969638, 'depth_head.projects.3': 0.0011478678788989782, 'depth_head.resize_layers.0': 0.0006527705118060112, 'depth_head.resize_layers.1': 0.0010292675578966737, 'depth_head.resize_layers.3': 0.0010845421347767115, 'depth_head.scratch.layer1_rn': 0.0007575156632810831, 'depth_head.scratch.layer2_rn': 0.0006197149050422013, 'depth_head.scratch.layer3_rn': 0.0007483637309633195, 'depth_head.scratch.layer4_rn': 0.0008324604132212698, 'depth_head.scratch.refinenet1.out_conv': 0.0008100330014713109, 'depth_head.scratch.refinenet1.resConfUnit1.conv1': 0.000828743155580014, 'depth_head.scratch.refinenet1.resConfUnit1.conv2': 0.00047521639498881996, 'depth_head.scratch.refinenet1.resConfUnit2.conv1': 0.0004705564060714096, 'depth_head.scratch.refinenet1.resConfUnit2.conv2': 0.0003533443377818912, 'depth_head.scratch.refinenet2.out_conv': 0.0010590320453047752, 'depth_head.scratch.refinenet2.resConfUnit1.conv1': 0.0008015644270926714, 'depth_head.scratch.refinenet2.resConfUnit1.conv2': 0.0006674809264950454, 'depth_head.scratch.refinenet2.resConfUnit2.conv1': 0.0006087010842747986, 'depth_head.scratch.refinenet2.resConfUnit2.conv2': 0.000496032414957881, 'depth_head.scratch.refinenet3.out_conv': 0.001640035305172205, 'depth_head.scratch.refinenet3.resConfUnit1.conv1': 0.000914071046281606, 'depth_head.scratch.refinenet3.resConfUnit1.conv2': 0.0008509040926583111, 'depth_head.scratch.refinenet3.resConfUnit2.conv1': 0.0008360622450709343, 'depth_head.scratch.refinenet3.resConfUnit2.conv2': 0.0006767968297936022, 'depth_head.scratch.refinenet4.out_conv': 0.0016657504020258784, 'depth_head.scratch.refinenet4.resConfUnit1.conv1': 0.0, 'depth_head.scratch.refinenet4.resConfUnit1.conv2': 0.0, 'depth_head.scratch.refinenet4.resConfUnit2.conv1': 0.00105638790410012, 'depth_head.scratch.refinenet4.resConfUnit2.conv2': 0.0008910073665902019, 'depth_head.scratch.output_conv1': 0.00036672496935352683, 'depth_head.scratch.output_conv2.0': 0.0007507689879275858, 'depth_head.scratch.output_conv2.2': 0.0002155498950742185}`
