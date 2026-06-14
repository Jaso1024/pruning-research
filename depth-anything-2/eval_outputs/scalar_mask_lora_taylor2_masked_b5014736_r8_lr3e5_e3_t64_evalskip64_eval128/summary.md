# Scalar Mask LoRA Repair DA2K

## Results

| model | accuracy | correct | pairs |
| --- | ---: | ---: | ---: |
| `dense_teacher` | 0.9288 | 248 | 267 |
| `pruned_student` | 0.9064 | 242 | 267 |
| `peft_repaired_unmerged` | 0.9101 | 243 | 267 |
| `folded_peft_unmasked` | 0.9101 | 243 | 267 |
| `folded_peft_remasked` | 0.9101 | 243 | 267 |

## Repair

- Prune score: `taylor2_abs`
- Masked tensor values: `5014736`
- Target zero fraction: `0.2362`
- PEFT method: `lora`
- LoRA placement: `masked`
- PEFT trainable params: `589824`
- Train/eval overlap images: `0`
