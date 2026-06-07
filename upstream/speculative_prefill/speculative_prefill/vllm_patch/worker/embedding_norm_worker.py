from typing import Dict, List

import torch
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.worker.worker import Worker
try:
    from vllm.worker.worker_base import LoRANotSupportedWorkerBase
except ImportError:
    from vllm.worker.worker_base import LoraNotSupportedWorkerBase as LoRANotSupportedWorkerBase

from speculative_prefill.vllm_patch.config import get_spec_config
from speculative_prefill.vllm_patch.data.input_builder import (
    AugmentedModelInputForGPUBuilder,
)
from speculative_prefill.vllm_patch.data.sequence import AugmentedSequenceData
from speculative_prefill.vllm_patch.selector import (
    embedding_norms_from_weight,
    select_kept_indices_from_scores,
    token_scores_from_embedding_norms,
)


def create_embedding_norm_worker(*args, **kwargs) -> "EmbeddingNormPrefillWorker":
    vllm_config = kwargs["vllm_config"]
    scheduler_config = vllm_config.scheduler_config
    parallel_config = vllm_config.parallel_config

    assert scheduler_config.chunked_prefill_enabled == False, \
        "Please set --enable-chunked-prefill=False or enable_chunked_prefill=False. "

    tensor_parallel_size = getattr(parallel_config, "tensor_parallel_size", 1)
    if tensor_parallel_size != 1:
        raise NotImplementedError(
            "embedding_norm prefill currently supports tensor_parallel_size=1. "
            "Tensor-parallel embedding tables may be sharded, so scoring them "
            "needs a separate gather path before it is safe."
        )

    base_model_worker = Worker(*args, **kwargs)
    return EmbeddingNormPrefillWorker(base_model_worker=base_model_worker)


class EmbeddingNormPrefillWorker(LoRANotSupportedWorkerBase):
    def __init__(self, base_model_worker: Worker):
        self.base_model_worker = base_model_worker
        self.vllm_config = base_model_worker.vllm_config
        self.model_config = base_model_worker.model_config
        self.cache_config = base_model_worker.cache_config
        self.parallel_config = base_model_worker.parallel_config
        self.scheduler_config = base_model_worker.scheduler_config
        self.spec_config = get_spec_config()
        if self.spec_config.keep_strategy != "embedding_norm":
            raise ValueError(
                "EmbeddingNormPrefillWorker requires keep_strategy=embedding_norm."
            )
        keep_kwargs = self.spec_config.keep_kwargs
        self._norm = keep_kwargs.get("norm", "l2")
        self._percentage = keep_kwargs.get("percentage", 1.0)
        self._chunk = keep_kwargs.get("chunk", False)
        self._chunk_size = keep_kwargs.get("chunk_size", 32)
        self._keep_high = keep_kwargs.get("keep_high", True)

        self.base_model_worker.model_runner._builder_cls = (
            AugmentedModelInputForGPUBuilder
        )
        self.id_to_context_len: Dict[str, int] = {}
        self.embedding_norms = None

    def init_device(self) -> None:
        self.base_model_worker.init_device()
        self.base_model_worker.load_model()
        self._prepare_embedding_norms()

    def load_model(self, *args, **kwargs):
        pass

    def determine_num_available_blocks(self):
        return self.base_model_worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.base_model_worker.initialize_cache(
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
        )

    @torch.inference_mode
    def execute_model(
        self,
        execute_model_req: ExecuteModelRequest | None = None,
    ) -> List[SamplerOutput] | None:
        if execute_model_req is None:
            return self.base_model_worker.execute_model(execute_model_req)

        has_prefill = any(
            sgm.is_prompt for sgm in execute_model_req.seq_group_metadata_list
        )
        if has_prefill:
            execute_model_req = self._rewrite_prefill_requests(execute_model_req)

        execute_model_req = self._record_and_update_requests(execute_model_req)
        return self.base_model_worker.execute_model(execute_model_req)

    @torch.inference_mode()
    def start_worker_execution_loop(self) -> None:
        self.base_model_worker.start_worker_execution_loop()

    def _prepare_embedding_norms(self) -> None:
        model = self.base_model_worker.model_runner.model
        weight = self._find_embedding_weight(model)
        self.embedding_norms = (
            embedding_norms_from_weight(weight, norm=self._norm)
            .to(device="cpu")
            .contiguous()
        )

    def _find_embedding_weight(self, model) -> torch.Tensor:
        candidates = [
            ("model", "embed_tokens", "weight"),
            ("model", "model", "embed_tokens", "weight"),
            ("transformer", "wte", "weight"),
        ]
        for path in candidates:
            current = model
            for attr in path:
                current = getattr(current, attr, None)
                if current is None:
                    break
            if isinstance(current, torch.Tensor):
                return current

        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            weight = getattr(embeddings, "weight", None)
            if isinstance(weight, torch.Tensor):
                return weight

        raise AttributeError(
            "Could not locate input embedding weight on the vLLM model."
        )

    def _rewrite_prefill_requests(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> ExecuteModelRequest:
        assert self.embedding_norms is not None

        new_seq_group_metadata_list = []
        for metadata in execute_model_req.seq_group_metadata_list:
            if metadata.is_prompt:
                assert len(metadata.seq_data) == 1
                seq_id = metadata.get_first_seq_id()
                seq_data = metadata.seq_data[seq_id]
                prompt_token_ids = torch.as_tensor(
                    seq_data._prompt_token_ids,
                    dtype=torch.long,
                )
                scores = token_scores_from_embedding_norms(
                    prompt_token_ids,
                    self.embedding_norms,
                )
                kept_indices = self._select_kept_indices(scores)

                new_seq_data = AugmentedSequenceData.from_seqs_and_pos_ids(
                    prompt_token_ids=prompt_token_ids[kept_indices].tolist(),
                    position_ids=kept_indices.tolist(),
                    output_token_ids=seq_data._output_token_ids,
                )
                metadata.seq_data[seq_id] = new_seq_data

            new_seq_group_metadata_list.append(metadata)

        return execute_model_req.clone(
            seq_group_metadata_list=new_seq_group_metadata_list
        )

    def _select_kept_indices(self, scores: torch.Tensor) -> torch.LongTensor:
        return select_kept_indices_from_scores(
            scores,
            percentage=self._percentage,
            chunk=self._chunk,
            chunk_size=self._chunk_size,
            keep_high=self._keep_high,
        )

    def _record_and_update_requests(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> ExecuteModelRequest:
        for metadata in execute_model_req.seq_group_metadata_list:
            assert len(metadata.seq_data) == 1
            request_id = metadata.request_id
            seq_id = metadata.get_first_seq_id()
            request_seq_id = f"{request_id}_{seq_id}"
            seq_data: AugmentedSequenceData = metadata.seq_data[seq_id]

            if metadata.is_prompt:
                self.id_to_context_len[request_seq_id] = seq_data.get_prompt_len()
            else:
                seq_data._context_len = self.id_to_context_len[request_seq_id]
                metadata.seq_data[seq_id] = seq_data
                self.id_to_context_len[request_seq_id] += 1

        return execute_model_req

    def get_cache_block_size_bytes(self) -> int:
        return self.base_model_worker.get_cache_block_size_bytes()

    def __getattr__(self, attr):
        return getattr(self.base_model_worker, attr)

    @property
    def rank(self):
        return self.base_model_worker.rank

    @property
    def device(self):
        return self.base_model_worker.device

    @property
    def _driver_rank(self) -> int:
        return 0
