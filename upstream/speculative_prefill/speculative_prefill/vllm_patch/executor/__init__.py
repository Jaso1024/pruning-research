import os


def patch_executor():
    if not _patch_legacy_gpu_executor():
        _patch_platform_worker_cls()


def _worker_cls_for_mode() -> str | None:
    pruning_mode = os.environ.get("PREFILL_PRUNING_MODE")
    if pruning_mode == "embedding_norm":
        return (
            "speculative_prefill.vllm_patch.worker.embedding_norm_worker."
            "create_embedding_norm_worker"
        )
    if pruning_mode == "middle_layer_norm":
        return (
            "speculative_prefill.vllm_patch.worker.middle_layer_norm_worker."
            "create_middle_layer_norm_worker"
        )
    if os.environ.get("SPEC_MODEL"):
        return (
            "speculative_prefill.vllm_patch.worker.spec_prefill_worker."
            "create_spec_worker"
        )
    return None


def _patch_legacy_gpu_executor() -> bool:
    try:
        from vllm.executor import gpu_executor
        from speculative_prefill.vllm_patch.executor.gpu_executor import (
            PatchedGPUExecutor,
            PatchedGPUExecutorAsync,
        )
    except (ImportError, ModuleNotFoundError):
        return False

    gpu_executor.GPUExecutor = PatchedGPUExecutor
    gpu_executor.GPUExecutorAsync = PatchedGPUExecutorAsync
    return True


def _patch_platform_worker_cls() -> None:
    from vllm.platforms.cuda import CudaPlatform

    if getattr(CudaPlatform.check_and_update_config, "_spec_prefill_patched", False):
        return

    original = CudaPlatform.check_and_update_config.__func__

    def check_and_update_config(cls, vllm_config):
        original(cls, vllm_config)
        worker_cls = _worker_cls_for_mode()
        if worker_cls is not None:
            parallel_config = vllm_config.parallel_config
            parallel_config.worker_cls = worker_cls

    check_and_update_config._spec_prefill_patched = True
    CudaPlatform.check_and_update_config = classmethod(check_and_update_config)
