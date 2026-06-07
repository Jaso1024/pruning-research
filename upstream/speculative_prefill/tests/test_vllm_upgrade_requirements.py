import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _requirements() -> dict[str, str]:
    pins = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        package, version = line.split("==", 1)
        pins[package] = version
    return pins


class VllmUpgradeRequirementsTest(unittest.TestCase):
    def test_requirements_target_qwen3_capable_vllm_stack(self):
        pins = _requirements()

        self.assertEqual(pins["vllm"], "0.9.1")
        self.assertEqual(pins["torch"], "2.7.0")
        self.assertRegex(pins["transformers"], r"^4\.5[1-9]\.")

    def test_modal_cuda_image_matches_vllm_091_stack(self):
        modal_source = (ROOT / "modal" / "run_embedding_norm_latency.py").read_text(
            encoding="utf-8"
        )

        match = re.search(r"nvidia/cuda:(\d+)\.(\d+)\.", modal_source)
        self.assertIsNotNone(match)
        major, minor = map(int, match.groups())
        self.assertGreaterEqual((major, minor), (12, 8))

    def test_modal_runner_uses_vllm_091_boolean_cli_shape(self):
        modal_source = (ROOT / "modal" / "run_embedding_norm_latency.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('"--no-enable-chunked-prefill"', modal_source)
        self.assertNotIn('"--enable-chunked-prefill",\n        "False"', modal_source)

    def test_modal_runner_forces_v0_engine_for_comparable_modes(self):
        modal_source = (ROOT / "modal" / "run_embedding_norm_latency.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('"VLLM_USE_V1": "0"', modal_source)


if __name__ == "__main__":
    unittest.main()
