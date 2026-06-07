from __future__ import annotations

import torch

from saliency.calibration import build_causal_lm_batch, format_gsm8k_prompt_answer


class TinyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token = "<pad>"
    pad_token_id = 99

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        tokens = [tok for tok in text.replace("\n", " \n ").split(" ") if tok]
        ids = []
        for token in tokens:
            if token == self.eos_token:
                ids.append(self.eos_token_id)
            else:
                ids.append(abs(hash(token)) % 10_000 + 1)
        return {"input_ids": ids}


def test_format_gsm8k_prompt_answer_uses_question_and_answer_fields():
    prompt, answer = format_gsm8k_prompt_answer({"question": "What is 2+2?", "answer": "4"})

    assert prompt == "Question: What is 2+2?\nAnswer:"
    assert answer == "4"


def test_build_causal_lm_batch_masks_prompt_tokens_and_pads():
    tokenizer = TinyTokenizer()
    records = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "What is 1+1?", "answer": "2"},
    ]

    batch = build_causal_lm_batch(tokenizer, records, max_length=32, device=torch.device("cpu"))

    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[0] == 2
    assert batch["supervised_tokens"] > 0
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)

    prompt_len = len(tokenizer("Question: What is 2+2?\nAnswer:", add_special_tokens=False)["input_ids"])
    assert torch.all(batch["labels"][0, :prompt_len] == -100)
    assert torch.any(batch["labels"][0, prompt_len:] != -100)


def test_build_causal_lm_batch_can_train_on_full_sequence():
    tokenizer = TinyTokenizer()
    batch = build_causal_lm_batch(
        tokenizer,
        [{"question": "What is 2+2?", "answer": "4"}],
        max_length=32,
        answer_only_loss=False,
        device=torch.device("cpu"),
    )

    non_pad = batch["attention_mask"].bool()
    assert torch.all(batch["labels"][non_pad] != -100)
