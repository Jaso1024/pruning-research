# Circuit PEFT Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.8333 | 5 | 6 |
| `pruned_student` | 0.8333 | 5 | 6 |
| `peft_repaired_unmerged` | 0.8333 | 5 | 6 |
| `folded_peft_unmasked` | 0.8333 | 5 | 6 |
| `folded_peft_remasked` | 0.8333 | 5 | 6 |

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

## Repair

- PEFT method: `lora`
- Train/eval overlap images: `0`
- Masked tensor values: `202896`
- PEFT trainable params: `51682`
- PEFT modules: `34`
- Merge RMS deltas: `{'depth_head.projects.0': 0.0001305303449044004, 'depth_head.projects.1': 0.00013036145537625998, 'depth_head.projects.2': 0.00014078144158702344, 'depth_head.projects.3': 0.00013411811960395426, 'depth_head.resize_layers.0': 8.21956418803893e-05, 'depth_head.resize_layers.1': 0.0001353020779788494, 'depth_head.resize_layers.3': 4.4172054913360626e-05, 'depth_head.scratch.layer1_rn': 0.0001192826239275746, 'depth_head.scratch.layer2_rn': 8.979858102975413e-05, 'depth_head.scratch.layer3_rn': 6.424156163120642e-05, 'depth_head.scratch.layer4_rn': 4.3874202674487606e-05, 'depth_head.scratch.refinenet1.out_conv': 0.00024849464534781873, 'depth_head.scratch.refinenet1.resConfUnit1.conv1': 0.0001110123994294554, 'depth_head.scratch.refinenet1.resConfUnit1.conv2': 0.00010390797251602635, 'depth_head.scratch.refinenet1.resConfUnit2.conv1': 9.459644934395328e-05, 'depth_head.scratch.refinenet1.resConfUnit2.conv2': 0.00010085286339744925, 'depth_head.scratch.refinenet2.out_conv': 0.00029649503994733095, 'depth_head.scratch.refinenet2.resConfUnit1.conv1': 0.00010722934530349448, 'depth_head.scratch.refinenet2.resConfUnit1.conv2': 0.00011218928557354957, 'depth_head.scratch.refinenet2.resConfUnit2.conv1': 0.00010266032768413424, 'depth_head.scratch.refinenet2.resConfUnit2.conv2': 0.00010681038111215457, 'depth_head.scratch.refinenet3.out_conv': 0.0003394676314201206, 'depth_head.scratch.refinenet3.resConfUnit1.conv1': 0.00011162422742927447, 'depth_head.scratch.refinenet3.resConfUnit1.conv2': 0.00011407333659008145, 'depth_head.scratch.refinenet3.resConfUnit2.conv1': 0.00010787424980662763, 'depth_head.scratch.refinenet3.resConfUnit2.conv2': 0.00010759931319626048, 'depth_head.scratch.refinenet4.out_conv': 0.0003631824511103332, 'depth_head.scratch.refinenet4.resConfUnit1.conv1': 0.0, 'depth_head.scratch.refinenet4.resConfUnit1.conv2': 0.0, 'depth_head.scratch.refinenet4.resConfUnit2.conv1': 0.00010746171756181866, 'depth_head.scratch.refinenet4.resConfUnit2.conv2': 0.00010641603876138106, 'depth_head.scratch.output_conv1': 8.799388160696253e-05, 'depth_head.scratch.output_conv2.0': 0.0001302184391533956, 'depth_head.scratch.output_conv2.2': 0.00035633466904982924}`
