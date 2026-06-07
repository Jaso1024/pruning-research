import tempfile
import unittest
from pathlib import Path

from speculative_prefill.vllm_benchmarks.sweep_utils import (
    BenchmarkRow,
    parse_int_csv,
    rows_to_csv,
    svg_latency_by_input_len,
)


class LatencySweepUtilsTest(unittest.TestCase):
    def test_parse_int_csv(self):
        self.assertEqual(parse_int_csv("256, 512,1024"), [256, 512, 1024])

    def test_rows_to_csv_writes_stable_columns(self):
        rows = [
            BenchmarkRow(
                mode="baseline",
                model="Qwen/Qwen3-1.7B",
                spec_model="",
                config="",
                input_len=256,
                output_len=1,
                batch_size=4,
                warmup_iters=1,
                iters=3,
                avg_latency=0.02,
                p50_latency=0.02,
                p90_latency=0.03,
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.csv"
            rows_to_csv(rows, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("mode,model,spec_model,config,input_len", text)
        self.assertIn("baseline,Qwen/Qwen3-1.7B,,", text)

    def test_svg_latency_by_input_len_contains_methods(self):
        rows = [
            BenchmarkRow("baseline", "m", "", "", 256, 1, 4, 1, 3, 0.02, 0.02, 0.03),
            BenchmarkRow("embedding_norm", "m", "", "c", 256, 1, 4, 1, 3, 0.03, 0.03, 0.04),
        ]

        svg = svg_latency_by_input_len(rows, batch_size=4)

        self.assertIn("<svg", svg)
        self.assertIn("baseline", svg)
        self.assertIn("embedding_norm", svg)


if __name__ == "__main__":
    unittest.main()
