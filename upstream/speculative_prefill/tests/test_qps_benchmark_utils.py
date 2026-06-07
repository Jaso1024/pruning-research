import tempfile
import unittest
from pathlib import Path

from speculative_prefill.qps_benchmarks.utils import (
    DEFAULT_NUM_SAMPLES,
    PAPER_QPS_GRIDS,
    QPSResult,
    build_client_command,
    build_mode_env,
    build_server_command,
    parse_float_csv,
    parse_qps_client_output,
    rows_to_csv,
    rows_to_jsonl,
)


class QPSBenchmarkUtilsTest(unittest.TestCase):
    def test_paper_grids_match_scripts(self):
        self.assertEqual(DEFAULT_NUM_SAMPLES, 32)
        self.assertEqual(PAPER_QPS_GRIDS["few-shot-learning"][0], 0.2)
        self.assertEqual(PAPER_QPS_GRIDS["few-shot-learning"][-1], 3.2)
        self.assertEqual(len(PAPER_QPS_GRIDS["few-shot-learning"]), 16)
        self.assertEqual(PAPER_QPS_GRIDS["multi-doc-qa"][0], 0.2)
        self.assertEqual(PAPER_QPS_GRIDS["multi-doc-qa"][-1], 5.0)
        self.assertEqual(len(PAPER_QPS_GRIDS["multi-doc-qa"]), 13)
        self.assertEqual(PAPER_QPS_GRIDS["summarization"][0], 0.2)
        self.assertEqual(PAPER_QPS_GRIDS["summarization"][-1], 2.2)
        self.assertEqual(len(PAPER_QPS_GRIDS["summarization"]), 11)

    def test_parse_qps_client_output_latency_and_timeout(self):
        latency = parse_qps_client_output("Profiling\nAverage latency: 1.234s\n")
        timeout = parse_qps_client_output("Found timeout in queries\n")

        self.assertFalse(latency.timed_out)
        self.assertEqual(latency.avg_latency_s, 1.234)
        self.assertTrue(timeout.timed_out)
        self.assertIsNone(timeout.avg_latency_s)

    def test_parse_float_csv(self):
        self.assertEqual(parse_float_csv("0.2, 0.6,1.0"), [0.2, 0.6, 1.0])

    def test_build_mode_env_sets_only_requested_patch(self):
        baseline = build_mode_env("baseline", spec_model="draft", config="cfg.yaml")
        embedding = build_mode_env("embedding_norm", spec_model="draft", config="cfg.yaml")
        middle = build_mode_env("middle_layer_norm", spec_model="draft", config="cfg.yaml")
        spec = build_mode_env("spec_prefill", spec_model="draft", config="cfg.yaml")

        self.assertEqual(baseline, {"VLLM_USE_V1": "0"})
        self.assertEqual(embedding["ENABLE_EMBEDDING_NORM_PREFILL"], "1")
        self.assertNotIn("ENABLE_SP", embedding)
        self.assertEqual(middle["ENABLE_MIDDLE_LAYER_NORM_PREFILL"], "1")
        self.assertNotIn("ENABLE_EMBEDDING_NORM_PREFILL", middle)
        self.assertEqual(middle["SPEC_CONFIG_PATH"], "cfg.yaml")
        self.assertEqual(spec["ENABLE_SP"], "draft")
        self.assertEqual(spec["SPEC_CONFIG_PATH"], "cfg.yaml")

    def test_build_server_command_uses_vllm_openai_server(self):
        cmd = build_server_command(
            model="Qwen/Qwen3-1.7B",
            port=8888,
            max_model_len=65536,
            max_num_seqs=256,
        )

        self.assertEqual(cmd[:4], ["python", "-m", "speculative_prefill.scripts", "serve"])
        self.assertIn("--no-enable-chunked-prefill", cmd)
        self.assertIn("--disable-log-requests", cmd)
        self.assertIn("--api-key", cmd)
        self.assertIn("local_server", cmd)

    def test_build_client_command_records_json(self):
        cmd = build_client_command(
            model="Qwen/Qwen3-1.7B",
            category="summarization",
            qps=0.4,
            timeout_s=45.0,
            num_samples=32,
            output_json="/tmp/result.json",
            max_tokens=8,
            max_tolerance=2,
        )

        self.assertEqual(cmd[:2], ["python", "eval/qps_client.py"])
        self.assertIn("--output-json", cmd)
        self.assertIn("/tmp/result.json", cmd)
        self.assertIn("--max-tokens", cmd)
        self.assertIn("--max-tolerance", cmd)

    def test_rows_write_stable_artifacts(self):
        row = QPSResult(
            mode="baseline",
            category="multi-doc-qa",
            qps=0.2,
            status="ok",
            avg_latency_s=1.0,
            timed_out=False,
            num_requests=128,
            num_success=128,
            num_timeout=0,
            timeout_s=15.0,
            num_samples_per_dataset=32,
            model="Qwen/Qwen3-1.7B",
            spec_model="",
            config="",
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "results.csv"
            jsonl_path = root / "results.jsonl"
            rows_to_csv([row], csv_path)
            rows_to_jsonl([row], jsonl_path)
            csv_text = csv_path.read_text(encoding="utf-8")
            jsonl_text = jsonl_path.read_text(encoding="utf-8")

        self.assertIn("mode,category,qps,status", csv_text)
        self.assertIn("baseline,multi-doc-qa,0.2,ok", csv_text)
        self.assertIn('"num_requests": 128', jsonl_text)


if __name__ == "__main__":
    unittest.main()
