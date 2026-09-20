# SPDX-License-Identifier: Apache-2.0
"""Handle failed KV reads across all hybrid cache groups."""

from vllm.v1.core.sched.scheduler import Scheduler

_original_update_requests_with_invalid_blocks = Scheduler._update_requests_with_invalid_blocks


def _update_requests_with_invalid_blocks(self, requests, invalid_block_ids, num_scheduled_tokens, evict_blocks=True):
    affected = set()
    total_tokens = 0
    evicted = set()
    ordinary_requests = []
    for request in requests:
        groups = self.kv_cache_manager.get_block_ids(request.request_id)
        if len(groups) <= 1:
            ordinary_requests.append(request)
            continue
        block_ids = {block for group in groups for block in group if block != 0}
        if block_ids.isdisjoint(invalid_block_ids):
            continue
        # A recurrent state cannot resume at an arbitrary attention-block
        # boundary. Conservatively invalidate the complete hybrid prefix.
        # The caller retains the configured fail/recompute policy.
        affected.add(request.request_id)
        total_tokens += max(0, request.num_computed_tokens - num_scheduled_tokens.get(request.request_id, 0))
        request.num_computed_tokens = 0
        if evict_blocks:
            evicted.update(block_ids)
    if ordinary_requests:
        reqs, tokens, blocks = _original_update_requests_with_invalid_blocks(
            self, ordinary_requests, invalid_block_ids, num_scheduled_tokens, evict_blocks
        )
        affected.update(reqs)
        total_tokens += tokens
        evicted.update(blocks)
    return affected, total_tokens, evicted


Scheduler._update_requests_with_invalid_blocks = _update_requests_with_invalid_blocks
