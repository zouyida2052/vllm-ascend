# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.patch.platform.patch_hybrid_kv_load import _update_requests_with_invalid_blocks


@pytest.mark.parametrize("evict", [False, True])
@pytest.mark.parametrize("invalid", [{7}, {2}, {99}])
def test_failure_in_any_hybrid_group_invalidates_the_request(evict, invalid):
    request = SimpleNamespace(request_id="request", num_computed_tokens=2304)
    scheduler = SimpleNamespace(kv_cache_manager=SimpleNamespace(get_block_ids=Mock(return_value=([1, 2], [0, 7, 8]))))
    result = _update_requests_with_invalid_blocks(scheduler, [request], invalid, {"request": 1}, evict)
    if invalid == {99}:
        assert result == (set(), 0, set())
        assert request.num_computed_tokens == 2304
    else:
        assert result == ({"request"}, 2303, {1, 2, 7, 8} if evict else set())
        assert request.num_computed_tokens == 0


def test_real_scheduler_fail_policy_reports_hybrid_request_without_recompute():
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    scheduler = object.__new__(Scheduler)
    request = SimpleNamespace(
        request_id="remote-request", num_computed_tokens=24, status=RequestStatus.WAITING_FOR_REMOTE_KVS
    )
    scheduler.recompute_kv_load_failures = False
    scheduler.skipped_waiting = [request]
    scheduler.running = []
    scheduler.kv_cache_manager = SimpleNamespace(get_block_ids=Mock(return_value=([1], [2], [3], [4, 5, 6, 7])))
    assert scheduler._handle_invalid_blocks({4}, {}) == {"remote-request"}


def test_single_group_requests_keep_upstream_shared_block_handling(monkeypatch):
    import vllm_ascend.patch.platform.patch_hybrid_kv_load as module

    requests = [SimpleNamespace(request_id="one"), SimpleNamespace(request_id="two")]
    scheduler = SimpleNamespace(kv_cache_manager=SimpleNamespace(get_block_ids=Mock(return_value=([1],))))
    original = Mock(return_value=({"one", "two"}, 128, {1}))
    monkeypatch.setattr(module, "_original_update_requests_with_invalid_blocks", original)
    assert module._update_requests_with_invalid_blocks(scheduler, requests, {1}, {}, True) == ({"one", "two"}, 128, {1})
    original.assert_called_once_with(scheduler, requests, {1}, {}, True)
