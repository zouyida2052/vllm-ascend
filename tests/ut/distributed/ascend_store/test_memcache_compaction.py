# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

import copy
import unittest
from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
    MmcDirect,
    _compact_group0_transfer_args,
)

_KEY = "DeepSeek-V4-Pro-0813@tp:0@group:0@cache_role:kv@cache_family:default@hash"


def _layer(block_size, base):
    scale = block_size // 32
    return [base, base + 100000, base + 100000 + 4096 * scale, base + 100000], [
        size * scale for size in (19456, 4096, 128, 4224)
    ]


def _coverage(addrs, sizes):
    """Independently compute the union of half-open byte ranges."""
    merged = []
    for start, end in sorted((addr, addr + size) for addr, size in zip(addrs, sizes, strict=True)):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


class TestMemcacheGroup0Compaction(unittest.TestCase):
    def test_verified_layout_preserves_coverage_and_order(self):
        for block_size in (32, 128):
            for num_layers in (1, 29, 32, 61):
                with self.subTest(block_size=block_size, num_layers=num_layers):
                    rows = [_layer(block_size, layer * 200000) for layer in range(num_layers)]
                    addr = [value for row, _ in rows for value in row]
                    size = [value for _, row in rows for value in row]
                    original = copy.deepcopy(([addr], [size]))
                    compact_addr, compact_size = _compact_group0_transfer_args([_KEY], [addr], [size])
                    self.assertEqual(compact_addr[0], [value for row, _ in rows for value in (row[0], row[3])])
                    self.assertEqual(compact_size[0], [value for _, row in rows for value in (row[0], row[3])])
                    self.assertEqual(_coverage(addr, size), _coverage(compact_addr[0], compact_size[0]))
                    self.assertEqual(len(compact_size[0]), num_layers * 2)
                    self.assertEqual(sum(size) - sum(compact_size[0]), num_layers * 4224 * (block_size // 32))
                    self.assertEqual(([addr], [size]), original)

    def test_other_model_group_role_family_unchanged(self):
        addr, size = _layer(32, 0)
        for key in (
            _KEY.replace("DeepSeek-V4-Pro-0813", "another-model"),
            _KEY.replace("group:0", "group:1"),
            _KEY.replace("cache_role:kv", "cache_role:state"),
            _KEY.replace("cache_family:default", "cache_family:c4"),
            _KEY.replace("DeepSeek-V4-Pro-0813@", "DeepSeek-V4-Pro-0813-other@"),
        ):
            with self.subTest(key=key):
                actual = _compact_group0_transfer_args([key], [addr], [size])
                self.assertEqual(actual, ([addr], [size]))
                self.assertIs(actual[0][0], addr)

    def test_unknown_or_partial_layout_unchanged(self):
        addr, size = _layer(32, 0)
        wrong_size = size.copy()
        wrong_size[-1] -= 1
        separate_buffer = addr.copy()
        separate_buffer[-1] += 1
        scale_gap = addr.copy()
        scale_gap[2] += 1
        for row, counts in (
            ([], []),
            (addr[:3], size[:3]),
            (addr[:3], size),
            (addr, wrong_size),
            (separate_buffer, size),
            (scale_gap, size),
            (addr + separate_buffer, size + size),
            (addr + addr[:2], size + size[:2]),
        ):
            with self.subTest(row=row, sizes=counts):
                original = copy.deepcopy(([row], [counts]))
                self.assertEqual(_compact_group0_transfer_args([_KEY], [row], [counts]), original)

    def test_mixed_batch_compacts_only_verified_object(self):
        addr, size = _layer(128, 0)
        addresses, sizes = [addr, [10], addr], [size, [7], size]
        original = copy.deepcopy((addresses, sizes))
        actual = _compact_group0_transfer_args([_KEY, _KEY, "other"], addresses, sizes)
        self.assertEqual(actual, ([[addr[0], addr[3]], [10], addr], [[size[0], size[3]], [7], size]))
        self.assertEqual((addresses, sizes), original)

    def test_empty_batch(self):
        self.assertEqual(_compact_group0_transfer_args([], [], []), ([], []))

    def test_mismatched_batch_lengths_rejected(self):
        for args in (([_KEY], [], []), ([], [[0]], []), ([_KEY], [[0]], [])):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "batch lengths"):
                _compact_group0_transfer_args(*args)

    def test_repeated_compaction_is_idempotent(self):
        addr, size = _layer(128, 0)
        actual = _compact_group0_transfer_args([_KEY], [addr], [size])
        self.assertEqual(_compact_group0_transfer_args([_KEY], *actual), actual)

    def test_backend_put_get_round_trip(self):
        # Exercise the real GET/PUT integration with a packed byte store.
        # Different source/destination pointers ensure no address is serialized.
        for block_size in (32, 128):
            with self.subTest(block_size=block_size):
                self._check_round_trip(block_size)

    def _check_round_trip(self, block_size):
        source = bytearray((index * 17 + 13) % 256 for index in range(400000))
        destination = bytearray([255]) * 400000
        rows = [_layer(block_size, base) for base in (0, 200000)]
        addr = [value for row, _ in rows for value in row]
        sizes = [value for _, row in rows for value in row]
        objects = {}

        def put(keys, addresses, counts, direction):
            self.assertEqual(direction, MmcDirect.COPY_L2G.value)
            for key, pointers, lengths in zip(keys, addresses, counts, strict=True):
                objects[key] = b"".join(
                    source[pointer : pointer + length] for pointer, length in zip(pointers, lengths, strict=True)
                )
            return [0] * len(keys)

        def get(keys, addresses, counts, direction):
            self.assertEqual(direction, MmcDirect.COPY_G2L.value)
            for key, pointers, lengths in zip(keys, addresses, counts, strict=True):
                offset = 0
                for pointer, length in zip(pointers, lengths, strict=True):
                    target = pointer - len(source)
                    destination[target : target + length] = objects[key][offset : offset + length]
                    offset += length
                self.assertEqual(offset, len(objects[key]))
            return [0] * len(keys)

        backend = MemcacheBackend.__new__(MemcacheBackend)
        backend._lazy_init = False
        backend._store_initialized = True
        backend.store = MagicMock()
        backend.store.batch_put_from_layers.side_effect = put
        backend.store.batch_get_into_layers.side_effect = get
        self.assertTrue(backend.put([_KEY], [addr], [sizes]))
        target_addr = [pointer + len(source) for pointer in addr]
        self.assertEqual(backend.get([_KEY], [target_addr], [sizes]), [0])
        self.assertEqual(len(backend.store.batch_put_from_layers.call_args.args[1][0]), 4)
        self.assertEqual(len(backend.store.batch_get_into_layers.call_args.args[1][0]), 4)
        for start, end in _coverage(addr, sizes):
            self.assertEqual(destination[start:end], source[start:end])
        self.assertEqual(destination[90000:100000], bytearray([255]) * 10000)
