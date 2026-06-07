import atexit
import os
from typing import Optional

import torch
import torch.distributed

from speculative_prefill.vllm_patch.config import init_spec_config

_TITLE = """
|=========================================================================================|
|                                                                                         |
|  ███████╗██████╗ ███████╗ ██████╗██╗   ██╗██╗      █████╗ ████████╗██╗██╗   ██╗███████╗ |
|  ██╔════╝██╔══██╗██╔════╝██╔════╝██║   ██║██║     ██╔══██╗╚══██╔══╝██║██║   ██║██╔════╝ |
|  ███████╗██████╔╝█████╗  ██║     ██║   ██║██║     ███████║   ██║   ██║██║   ██║█████╗   |
|  ╚════██║██╔═══╝ ██╔══╝  ██║     ██║   ██║██║     ██╔══██║   ██║   ██║╚██╗ ██╔╝██╔══╝   |
|  ███████║██║     ███████╗╚██████╗╚██████╔╝███████╗██║  ██║   ██║   ██║ ╚████╔╝ ███████╗ |
|  ╚══════╝╚═╝     ╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═══╝  ╚══════╝ |
|      ██████╗ ██████╗ ███████╗███████╗██╗██╗     ██╗     ██╗███╗   ██╗ ██████╗           |
|      ██╔══██╗██╔══██╗██╔════╝██╔════╝██║██║     ██║     ██║████╗  ██║██╔════╝           |
|      ██████╔╝██████╔╝█████╗  █████╗  ██║██║     ██║     ██║██╔██╗ ██║██║  ███╗          |
|      ██╔═══╝ ██╔══██╗██╔══╝  ██╔══╝  ██║██║     ██║     ██║██║╚██╗██║██║   ██║          |
|      ██║     ██║  ██║███████╗██║     ██║███████╗███████╗██║██║ ╚████║╚██████╔╝          |
|      ╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝           |
|                                                                                         |
|=========================================================================================|
| Notes:                                                                                  |
|    - Currently only support Llama model as the base and speculator.                     |
|    - Currently does not support chunked prefill, use enable_chunked_prefill=False       |
|    - Recommend to set gpu_memory_utilization when using tensor_parallel_size > 1        |
|    - Please use enforce_eager=True, which makes long context task correct.              |
|=========================================================================================|
"""


def clean_up_fn():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def enable_prefill_spec(
    spec_model: str = 'meta-llama/Llama-3.2-1B-Instruct', 
    spec_config_path: Optional[str] = None
):
    print(_TITLE)
    print("Setting up environment vars...")
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("PREFILL_PRUNING_MODE", "spec_model")
    os.environ.setdefault("SPEC_MODEL", spec_model)
    if spec_config_path is not None:
        os.environ.setdefault("SPEC_CONFIG_PATH", spec_config_path)

    init_spec_config()

    print("Applying speculative prefill vllm monkey patch...")
    from speculative_prefill.vllm_patch.data import patch_data
    from speculative_prefill.vllm_patch.executor import patch_executor
    from speculative_prefill.vllm_patch.scheduler import patch_scheduler

    patch_executor()
    patch_scheduler()
    patch_data()

    atexit.register(clean_up_fn)


def enable_embedding_norm_prefill(
    spec_config_path: Optional[str] = None
):
    print(_TITLE)
    print("Setting up environment vars for embedding-norm prefill pruning...")
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("PREFILL_PRUNING_MODE", "embedding_norm")
    if spec_config_path is not None:
        os.environ.setdefault("SPEC_CONFIG_PATH", spec_config_path)

    init_spec_config()

    print("Applying embedding-norm prefill vllm monkey patch...")
    from speculative_prefill.vllm_patch.data import patch_data
    from speculative_prefill.vllm_patch.executor import patch_executor

    patch_executor()
    patch_data()

    atexit.register(clean_up_fn)


def enable_middle_layer_norm_prefill(
    spec_config_path: Optional[str] = None
):
    print(_TITLE)
    print("Setting up environment vars for middle-layer norm prefill pruning...")
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("PREFILL_PRUNING_MODE", "middle_layer_norm")
    if spec_config_path is not None:
        os.environ.setdefault("SPEC_CONFIG_PATH", spec_config_path)

    init_spec_config()

    print("Applying middle-layer norm prefill vllm monkey patch...")
    from speculative_prefill.vllm_patch.data import patch_data
    from speculative_prefill.vllm_patch.executor import patch_executor

    patch_executor()
    patch_data()

    atexit.register(clean_up_fn)
