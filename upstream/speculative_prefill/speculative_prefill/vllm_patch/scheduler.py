import time
from collections import deque
from typing import Deque, List, Optional, Set, Tuple

from vllm.core import scheduler
from vllm.core.interfaces import AllocStatus
from vllm.core.scheduler import (PartialPrefillMetadata, ScheduledSequenceGroup,
                                 SchedulerPrefillOutputs, SchedulingBudget)
from vllm.logger import init_logger
from vllm.sequence import SequenceGroup, SequenceStatus

from speculative_prefill.vllm_patch.config import get_spec_config

logger = init_logger(__name__)

def _schedule_prefills(
        self: scheduler.Scheduler,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> SchedulerPrefillOutputs:
        if budget.remaining_token_budget() == 0:
            return SchedulerPrefillOutputs(
                seq_groups=[],
                ignored_seq_groups=[],
                num_lookahead_slots=self._get_num_lookahead_slots(
                    is_prefill=True, enable_chunking=enable_chunking),
            )

        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []
        using_prompt_embeds: bool = False

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            if (partial_prefill_metadata is not None
                    and not partial_prefill_metadata.can_schedule(seq_group)):
                leftover_waiting_sequences.appendleft(seq_group)
                waiting_queue.popleft()
                continue

            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group,
                    SequenceStatus.WAITING,
                    enable_chunking,
                    budget,
                    partial_prefill_metadata=partial_prefill_metadata,
                ))
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached

            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d",
                    num_new_tokens,
                    prompt_limit,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots = self._get_num_lookahead_slots(
                True, enable_chunking)

            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens,
                    num_lookahead_slots,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            if len(seq_groups) == 0:
                using_prompt_embeds = seq_group.uses_prompt_embeds()
            if using_prompt_embeds != seq_group.uses_prompt_embeds():
                leftover_waiting_sequences.appendleft(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            if (budget.num_batched_tokens
                    >= self.scheduler_config.max_num_batched_tokens):
                break

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            blocks_to_copy: List[Tuple[int, int]] = []
            self._append_slots(seq_group, blocks_to_copy, enable_chunking)
            assert not blocks_to_copy

            if partial_prefill_metadata is not None:
                partial_prefill_metadata.maybe_increment_partial_prefills(
                    seq_group)

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking),
        )


def _get_num_lookahead_slots(
    self: scheduler.Scheduler,
    is_prefill: bool,
    enable_chunking: bool
) -> int:
    assert not enable_chunking
    assert not self.scheduler_config.is_multi_step

    if is_prefill:
        return get_spec_config().look_ahead_cnt
    else:
        return self.scheduler_config.num_lookahead_slots


def patch_scheduler():
    scheduler.Scheduler._get_num_lookahead_slots = _get_num_lookahead_slots
    scheduler.Scheduler._schedule_prefills = _schedule_prefills
