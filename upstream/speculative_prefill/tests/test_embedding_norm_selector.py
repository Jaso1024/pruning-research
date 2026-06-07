import tempfile
import unittest
from pathlib import Path

import torch

from speculative_prefill.vllm_patch.config import SpecConfig
from speculative_prefill.vllm_patch.selector import (
    embedding_norms_from_weight,
    hidden_state_norms,
    select_kept_indices_from_scores,
    token_scores_from_embedding_norms,
)


class EmbeddingNormSelectorTest(unittest.TestCase):
    def test_selects_highest_scores_and_restores_original_order(self):
        scores = torch.tensor([0.1, 9.0, 0.2, 8.0, 0.3])

        kept = select_kept_indices_from_scores(scores, percentage=0.4)

        self.assertEqual(kept.tolist(), [1, 3])

    def test_selects_lowest_scores_when_requested(self):
        scores = torch.tensor([0.1, 9.0, 0.2, 8.0, 0.3])

        kept = select_kept_indices_from_scores(
            scores, percentage=0.4, keep_high=False)

        self.assertEqual(kept.tolist(), [0, 2])

    def test_stable_tie_break_keeps_earlier_tokens(self):
        scores = torch.tensor([1.0, 1.0, 1.0, 1.0])

        kept = select_kept_indices_from_scores(scores, percentage=0.5)

        self.assertEqual(kept.tolist(), [0, 1])

    def test_stable_tie_break_at_cutoff_keeps_earlier_tokens(self):
        scores = torch.tensor([9.0, 7.0, 7.0, 7.0, 1.0])

        kept = select_kept_indices_from_scores(scores, percentage=0.6)

        self.assertEqual(kept.tolist(), [0, 1, 2])

    def test_low_score_tie_break_at_cutoff_keeps_earlier_tokens(self):
        scores = torch.tensor([1.0, 7.0, 7.0, 7.0, 9.0])

        kept = select_kept_indices_from_scores(
            scores, percentage=0.6, keep_high=False)

        self.assertEqual(kept.tolist(), [0, 1, 2])

    def test_chunk_mode_selects_whole_chunks(self):
        scores = torch.tensor([1.0, 1.0, 9.0, 9.0, 2.0])

        kept = select_kept_indices_from_scores(
            scores, percentage=0.3, chunk=True, chunk_size=2)

        self.assertEqual(kept.tolist(), [2, 3])

    def test_percentage_one_keeps_everything(self):
        scores = torch.tensor([3.0, 1.0, 2.0])

        kept = select_kept_indices_from_scores(scores, percentage=1.0)

        self.assertEqual(kept.tolist(), [0, 1, 2])

    def test_rejects_invalid_percentage(self):
        with self.assertRaises(ValueError):
            select_kept_indices_from_scores(torch.tensor([1.0]), percentage=0)

        with self.assertRaises(ValueError):
            select_kept_indices_from_scores(torch.tensor([1.0]), percentage=1.1)

    def test_scores_prompt_tokens_from_embedding_norm_vector(self):
        embedding_norms = torch.tensor([0.5, 2.0, 4.0, 8.0])
        prompt_token_ids = torch.tensor([3, 1, 3, 0])

        scores = token_scores_from_embedding_norms(
            prompt_token_ids, embedding_norms)

        self.assertEqual(scores.tolist(), [8.0, 2.0, 8.0, 0.5])

    def test_embedding_norms_support_l1_and_l2(self):
        weight = torch.tensor([[3.0, 4.0], [1.0, -2.0]])

        self.assertEqual(
            embedding_norms_from_weight(weight, norm="l2").tolist(),
            [5.0, torch.sqrt(torch.tensor(5.0)).item()],
        )
        self.assertEqual(
            embedding_norms_from_weight(weight, norm="l1").tolist(),
            [7.0, 3.0],
        )

    def test_config_accepts_embedding_norm_strategy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join([
                    "keep_strategy: embedding_norm",
                    "keep_kwargs:",
                    "  percentage: 0.3",
                    "  chunk: true",
                    "  chunk_size: 32",
                    "  norm: l2",
                    "  keep_high: true",
                ]),
                encoding="utf-8",
            )

            config = SpecConfig.from_path(str(config_path))

        self.assertEqual(config.keep_strategy, "embedding_norm")
        self.assertEqual(config.keep_kwargs["percentage"], 0.3)

    def test_hidden_state_norms_support_l1_and_l2(self):
        hidden_states = torch.tensor([[[3.0, 4.0], [1.0, -2.0]]])

        self.assertEqual(
            hidden_state_norms(hidden_states, norm="l2").tolist(),
            [5.0, torch.sqrt(torch.tensor(5.0)).item()],
        )
        self.assertEqual(
            hidden_state_norms(hidden_states, norm="l1").tolist(),
            [7.0, 3.0],
        )

    def test_config_accepts_middle_layer_norm_strategy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join([
                    "keep_strategy: middle_layer_norm",
                    "keep_kwargs:",
                    "  percentage: 0.3",
                    "  chunk: true",
                    "  chunk_size: 32",
                    "  norm: l2",
                    "  keep_high: true",
                    "  layer_fraction: 0.5",
                ]),
                encoding="utf-8",
            )

            config = SpecConfig.from_path(str(config_path))

        self.assertEqual(config.keep_strategy, "middle_layer_norm")
        self.assertEqual(config.keep_kwargs["layer_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
