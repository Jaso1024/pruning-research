import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Vllm091PatchContractTest(unittest.TestCase):
    def test_executor_patch_uses_worker_cls_on_new_vllm(self):
        source = (
            ROOT / "speculative_prefill" / "vllm_patch" / "executor" / "__init__.py"
        ).read_text(encoding="utf-8")

        self.assertIn("parallel_config.worker_cls", source)
        self.assertIn("create_embedding_norm_worker", source)
        self.assertIn("create_middle_layer_norm_worker", source)
        module_preamble = source.split("def _patch_legacy_gpu_executor", 1)[0]
        self.assertNotIn(
            "from speculative_prefill.vllm_patch.executor.gpu_executor",
            module_preamble,
        )

    def test_embedding_norm_worker_accepts_vllm_config_kwargs(self):
        source = (
            ROOT
            / "speculative_prefill"
            / "vllm_patch"
            / "worker"
            / "embedding_norm_worker.py"
        ).read_text(encoding="utf-8")

        self.assertIn('kwargs["vllm_config"]', source)
        self.assertNotIn('kwargs["scheduler_config"]', source)
        self.assertNotIn('kwargs["model_runner_cls"] = ModelRunner', source)

    def test_middle_layer_norm_worker_accepts_vllm_config_kwargs(self):
        source = (
            ROOT
            / "speculative_prefill"
            / "vllm_patch"
            / "worker"
            / "middle_layer_norm_worker.py"
        ).read_text(encoding="utf-8")

        self.assertIn('kwargs["vllm_config"]', source)
        self.assertIn("AutoModel", source)
        self.assertNotIn('kwargs["scheduler_config"]', source)
        self.assertNotIn('kwargs["model_runner_cls"] = ModelRunner', source)

    def test_spec_prefill_worker_accepts_vllm091_worker_surface(self):
        source = (
            ROOT
            / "speculative_prefill"
            / "vllm_patch"
            / "worker"
            / "spec_prefill_worker.py"
        ).read_text(encoding="utf-8")

        self.assertIn('kwargs["vllm_config"]', source)
        self.assertIn("LoRANotSupportedWorkerBase", source)
        self.assertNotIn('kwargs["scheduler_config"]', source)
        self.assertNotIn('kwargs["model_runner_cls"] = ModelRunner', source)

    def test_lookahead_worker_uses_prefixed_draft_model_and_qwen_attention(self):
        source = (
            ROOT
            / "speculative_prefill"
            / "vllm_patch"
            / "worker"
            / "look_ahead_spec_worker.py"
        ).read_text(encoding="utf-8")

        self.assertIn('prefix="spec_prefill_draft"', source)
        self.assertIn("Qwen3Attention", source)
        self.assertIn("get_forward_context", source)

    def test_scheduler_patch_uses_vllm091_prefill_api(self):
        source = (
            ROOT / "speculative_prefill" / "vllm_patch" / "scheduler.py"
        ).read_text(encoding="utf-8")

        self.assertIn("_get_num_new_uncached_and_cached_tokens", source)
        self.assertIn("partial_prefill_metadata", source)
        self.assertIn("num_batched_tokens=num_new_tokens_uncached", source)
        self.assertNotIn("_get_num_new_tokens", source)

    def test_embedding_norm_does_not_patch_spec_scheduler(self):
        source = (ROOT / "speculative_prefill" / "vllm_patch" / "__init__.py").read_text(
            encoding="utf-8"
        )
        embedding_fn = source.split("def enable_embedding_norm_prefill", 1)[1]
        embedding_fn = embedding_fn.split("atexit.register(clean_up_fn)", 1)[0]

        self.assertNotIn("patch_scheduler", embedding_fn)


if __name__ == "__main__":
    unittest.main()
