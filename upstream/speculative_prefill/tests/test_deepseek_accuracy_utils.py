import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from speculative_prefill.api_eval.pruning import (
    TextSpan,
    merge_adjacent_spans,
    spans_to_text,
    token_chunk_spans_from_offsets,
    token_scores_from_query_attentions,
)
from speculative_prefill.api_eval.results import group_scores, score_longbench_predictions
from speculative_prefill.api_eval.aggregate import collect_shards, summarize_shards
from speculative_prefill.api_eval.modal_commands import (
    build_deepseek_longbench_command,
    completed_shard_exists,
    expand_dataset_spec,
    output_run_name,
    sample_block_shards,
)
from eval.deepseek_longbench_subset import apply_deepseek_result, parse_limit


class DeepSeekAccuracyUtilsTest(unittest.TestCase):
    def test_token_chunk_spans_ignores_empty_offsets_and_merges_chunks(self):
        offsets = [(0, 0), (0, 4), (5, 10), (11, 15), (16, 20)]
        kept = torch.tensor([0, 1])

        spans = token_chunk_spans_from_offsets(offsets, kept, chunk_size=2)

        self.assertEqual(spans, [TextSpan(0, 15)])

    def test_merge_adjacent_spans_respects_gap(self):
        spans = [TextSpan(0, 5), TextSpan(6, 10), TextSpan(20, 25)]

        merged = merge_adjacent_spans(spans, max_gap=1)

        self.assertEqual(merged, [TextSpan(0, 10), TextSpan(20, 25)])

    def test_spans_to_text_uses_delimiter(self):
        text = "alpha beta gamma"
        spans = [TextSpan(0, 5), TextSpan(11, 16)]

        compressed = spans_to_text(text, spans, delimiter="\n...\n")

        self.assertEqual(compressed, "alpha\n...\ngamma")

    def test_token_scores_from_query_attentions_takes_prompt_slice(self):
        step0 = (
            torch.tensor([[[[0.1, 0.4, 0.2, 0.3]]]]),
            torch.tensor([[[[0.9, 0.1, 0.0, 0.0]]]]),
        )
        step1 = (
            torch.tensor([[[[0.2, 0.7, 0.1, 0.0, 1.0]]]]),
            torch.tensor([[[[0.0, 0.2, 0.8, 0.0, 1.0]]]]),
        )

        scores = token_scores_from_query_attentions([step0, step1], seq_len=4)

        self.assertTrue(torch.allclose(scores, torch.tensor([0.9, 0.7, 0.8, 0.3])))

    def test_token_scores_from_query_attentions_can_select_one_based_layer(self):
        step0 = (
            torch.tensor([[[[0.1, 0.4, 0.2, 0.3]]]]),
            torch.tensor([[[[0.9, 0.1, 0.0, 0.0]]]]),
        )
        step1 = (
            torch.tensor([[[[0.2, 0.7, 0.1, 0.0, 1.0]]]]),
            torch.tensor([[[[0.0, 0.2, 0.8, 0.0, 1.0]]]]),
        )

        scores = token_scores_from_query_attentions(
            [step0, step1],
            seq_len=4,
            layer_index=1,
        )

        self.assertTrue(torch.allclose(scores, torch.tensor([0.2, 0.7, 0.2, 0.3])))

    def test_token_scores_from_query_attentions_rejects_missing_layer(self):
        with self.assertRaises(IndexError):
            token_scores_from_query_attentions(
                [(torch.tensor([[[[1.0]]]]),)],
                seq_len=1,
                layer_index=2,
            )

    def test_score_longbench_predictions(self):
        rows = [
            {
                "dataset": "triviaqa",
                "pred": "Paris",
                "answers": ["Paris"],
                "all_classes": [],
            },
            {
                "dataset": "triviaqa",
                "pred": "London",
                "answers": ["Paris"],
                "all_classes": [],
            },
        ]

        scores = score_longbench_predictions(rows)

        self.assertEqual(scores["triviaqa"]["score"], 50.0)
        self.assertEqual(scores["triviaqa"]["n"], 2)

    def test_group_scores(self):
        rows = [
            {
                "dataset": "triviaqa",
                "method": "embedding_norm",
                "local_model": "a",
                "pred": "Paris",
                "answers": ["Paris"],
                "all_classes": [],
            },
            {
                "dataset": "triviaqa",
                "method": "embedding_norm",
                "local_model": "b",
                "pred": "London",
                "answers": ["Paris"],
                "all_classes": [],
            },
        ]

        scores = group_scores(rows, ("method", "local_model"))

        self.assertEqual(scores["embedding_norm | a"]["triviaqa"]["score"], 100.0)
        self.assertEqual(scores["embedding_norm | b"]["triviaqa"]["score"], 0.0)

    def test_deepseek_longbench_script_help_runs(self):
        result = subprocess.run(
            [sys.executable, "eval/deepseek_longbench_subset.py", "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_build_deepseek_longbench_command(self):
        command = build_deepseek_longbench_command(
            datasets="trec",
            limit=1,
            deepseek_model="deepseek-chat",
            scorer_models="Qwen/Qwen3-0.6B",
            draft_models="Qwen/Qwen3-0.6B",
            first_layer_draft_models="Qwen/Qwen3-0.6B",
            middle_layer_models="Qwen/Qwen3-0.6B",
            first_attn_models="Qwen/Qwen3-0.6B",
            first_ffn_models="Qwen/Qwen3-0.6B",
            keep_rate=0.3,
            chunk_size=32,
            lookahead=1,
            max_tokens=16,
            concurrency=2,
            dry_run=False,
            sample_start=0,
            sample_count=None,
            continue_on_api_error=False,
        )

        self.assertEqual(command[:2], ["python", "eval/deepseek_longbench_subset.py"])
        self.assertIn("--deepseek-model", command)
        self.assertIn("deepseek-chat", command)
        self.assertIn("--first-layer-draft-models", command)
        self.assertIn("--middle-layer-models", command)
        self.assertIn("--first-attn-models", command)
        self.assertIn("--first-ffn-models", command)
        self.assertNotIn("--dry-run", command)

    def test_parse_limit_all(self):
        self.assertIsNone(parse_limit("all"))
        self.assertIsNone(parse_limit("none"))
        self.assertEqual(parse_limit("7"), 7)

    def test_expand_dataset_spec_all(self):
        datasets = expand_dataset_spec("all")

        self.assertIn("narrativeqa", datasets)
        self.assertIn("repobench-p", datasets)
        self.assertEqual(len(datasets), 21)

    def test_output_run_name_marks_dataset(self):
        name = output_run_name("20260525T000000", "passage_retrieval_en", True, 25, 50)

        self.assertEqual(name, "modal_longbench_deepseek_passage_retrieval_en_s25_n50_20260525T000000")

    def test_sample_block_shards(self):
        shards = sample_block_shards(["qasper"], limit="55", block_size=25)

        self.assertEqual(shards, [("qasper", 0, 25), ("qasper", 25, 25), ("qasper", 50, 5)])

    def test_completed_shard_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "modal_longbench_deepseek_triviaqa_s0_n25_20260525T000000"
            shard.mkdir()
            (shard / "run_config.json").write_text(
                '{"requested_datasets":"all","datasets":"triviaqa","sample_start":0,'
                '"sample_count":25,"dry_run":false}',
                encoding="utf-8",
            )
            (shard / "predictions.jsonl").write_text("{}\n", encoding="utf-8")

            self.assertTrue(completed_shard_exists(root, "all", "triviaqa", 0, 25, False))
            self.assertFalse(completed_shard_exists(root, "all", "triviaqa", 25, 25, False))

    def test_collect_and_summarize_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "modal_longbench_deepseek_triviaqa_s0_n25_20260525T000000"
            shard.mkdir()
            (shard / "run_config.json").write_text(
                '{"requested_datasets":"all","datasets":"triviaqa","sample_start":0,'
                '"sample_count":25,"dry_run":false}',
                encoding="utf-8",
            )
            rows = [
                {
                    "dataset": "triviaqa",
                    "sample_idx": 0,
                    "method": "baseline",
                    "local_model": "none",
                    "pred": "Paris",
                    "answers": ["Paris"],
                    "all_classes": [],
                },
                {
                    "dataset": "triviaqa",
                    "sample_idx": 0,
                    "method": "embedding_norm",
                    "local_model": "Qwen/Qwen3-0.6B",
                    "pred": "London",
                    "api_error": "refused",
                    "answers": ["Paris"],
                    "all_classes": [],
                },
            ]
            (shard / "predictions.jsonl").write_text(
                "\n".join(__import__("json").dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            shards = collect_shards(root, requested_datasets="all")
            summary = summarize_shards(shards)

            self.assertEqual(len(shards), 1)
            self.assertEqual(summary["coverage"]["triviaqa"]["completed_blocks"], 1)
            self.assertEqual(summary["row_count"], 2)
            self.assertEqual(
                summary["scores_by_method_model"]["baseline | none"]["triviaqa"]["score"],
                100.0,
            )
            self.assertEqual(summary["api_error_count"], 1)
            self.assertEqual(summary["macro_scores_by_method_model"]["baseline | none"], 100.0)

    def test_apply_deepseek_result_records_error_when_allowed(self):
        row = {}

        apply_deepseek_result(row, RuntimeError("400 Bad Request"), continue_on_error=True)

        self.assertEqual(row["pred"], "")
        self.assertIn("400 Bad Request", row["api_error"])


if __name__ == "__main__":
    unittest.main()
