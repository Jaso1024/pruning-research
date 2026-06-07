def enable_prefill_spec(*args, **kwargs):
    from speculative_prefill.vllm_patch import enable_prefill_spec as _enable
    return _enable(*args, **kwargs)


def enable_embedding_norm_prefill(*args, **kwargs):
    from speculative_prefill.vllm_patch import (
        enable_embedding_norm_prefill as _enable,
    )
    return _enable(*args, **kwargs)


def enable_middle_layer_norm_prefill(*args, **kwargs):
    from speculative_prefill.vllm_patch import (
        enable_middle_layer_norm_prefill as _enable,
    )
    return _enable(*args, **kwargs)
