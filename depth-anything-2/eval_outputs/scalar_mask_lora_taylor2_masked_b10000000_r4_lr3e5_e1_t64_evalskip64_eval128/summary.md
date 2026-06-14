# Scalar Mask LoRA Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9288 | 248 | 267 |
| `pruned_student` | 0.7528 | 201 | 267 |
| `peft_repaired_unmerged` | 0.7640 | 204 | 267 |
| `folded_peft_unmasked` | 0.7640 | 204 | 267 |
| `folded_peft_remasked` | 0.7640 | 204 | 267 |

## Repair

- Prune score: `taylor2_abs`
- Masked tensor values: `10000000`
- Target zero fraction: `0.4710`
- PEFT method: `lora`
- LoRA placement: `masked`
- PEFT trainable params: `294912`
- Train/eval overlap images: `0`
