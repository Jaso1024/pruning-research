# Scalar Mask LoRA Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.8333 | 5 | 6 |
| `pruned_student` | 0.8333 | 5 | 6 |
| `peft_repaired_unmerged` | 0.8333 | 5 | 6 |
| `folded_peft_unmasked` | 0.8333 | 5 | 6 |
| `folded_peft_remasked` | 0.8333 | 5 | 6 |

## Repair

- Prune score: `taylor2_abs`
- Masked tensor values: `6160`
- Target zero fraction: `0.0003`
- PEFT method: `lora`
- LoRA placement: `masked`
- PEFT trainable params: `52992`
- Train/eval overlap images: `0`
