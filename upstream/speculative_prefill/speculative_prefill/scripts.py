import os

from speculative_prefill import (
    enable_embedding_norm_prefill,
    enable_middle_layer_norm_prefill,
    enable_prefill_spec,
)

spec_model = os.environ.get(
    "ENABLE_SP", None)
embedding_norm_prefill = os.environ.get(
    "ENABLE_EMBEDDING_NORM_PREFILL", None)
middle_layer_norm_prefill = os.environ.get(
    "ENABLE_MIDDLE_LAYER_NORM_PREFILL", None)

if middle_layer_norm_prefill:
    enable_middle_layer_norm_prefill(
        spec_config_path='./configs/config.yaml'
    )
elif embedding_norm_prefill:
    enable_embedding_norm_prefill(
        spec_config_path='./configs/config.yaml'
    )
elif spec_model:
    enable_prefill_spec(
        spec_model=spec_model, 
        spec_config_path='./configs/config.yaml'
    )

from vllm.scripts import main

if __name__ == "__main__":    
    main()
