"""页分配器单测：页对齐、跨页、回收与完整性。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minisgl.core import Req, SamplingParams  # noqa: E402
from minisgl.kvcache.naive_cache import NaiveCacheHandle  # noqa: E402
from minisgl.scheduler.cache import CacheManager  # noqa: E402


def _make_req(table_idx: int, length: int, output_len: int = 1) -> Req:
    return Req(
        input_ids=torch.arange(1, length + 1, dtype=torch.long),
        table_idx=table_idx,
        cached_len=0,
        output_len=output_len,
        uid=table_idx,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=NaiveCacheHandle(),
    )


def _make_manager(
    num_pages: int, page_size: int, table_rows: int = 4
) -> tuple[CacheManager, torch.Tensor]:
    page_table = torch.zeros((table_rows, 128), dtype=torch.int32)
    manager = CacheManager(num_pages, page_size, page_table)
    return manager, page_table


@pytest.mark.parametrize("page_size", [1, 4, 16])
def test_allocate_is_page_aligned_and_disjoint(page_size: int) -> None:
    manager, page_table = _make_manager(num_pages=16, page_size=page_size)
    reqs = [_make_req(0, 5), _make_req(1, 10)]

    manager.allocate_paged(reqs)

    slots0 = page_table[0, :5].tolist()
    slots1 = page_table[1, :10].tolist()
    assert len(set(slots0) | set(slots1)) == len(slots0) + len(slots1)
    assert all(slot % page_size == 0 for slot in slots0[::page_size])
    # 同一请求的槽位按页连续展开
    assert slots0 == list(range(slots0[0], slots0[0] + 5))

    # 释放后必须完整回收
    for req in reqs:
        req.cached_len = req.device_len
        manager.cache_req(req, finished=True)
    manager.check_integrity()
    assert len(manager.free_slots) == 16


@pytest.mark.parametrize("page_size", [4, 16])
def test_decode_crosses_page_boundary(page_size: int) -> None:
    manager, page_table = _make_manager(num_pages=16, page_size=page_size)
    req = _make_req(0, page_size, output_len=2)
    manager.allocate_paged([req])
    first_page_slots = page_table[0, :page_size].tolist()
    assert first_page_slots == list(range(first_page_slots[0], first_page_slots[0] + page_size))

    # 模拟 decode 向前推进到跨越页边界的位置
    req.complete_one()  # cached_len=page_size, device_len=page_size+1
    manager.allocate_paged([req])
    next_slot = int(page_table[0, page_size])
    assert next_slot % page_size == 0
    assert next_slot >= page_size

    req.cached_len = req.device_len
    manager.cache_req(req, finished=True)
    manager.check_integrity()


def test_free_slots_remain_page_aligned_after_partial_request() -> None:
    manager, page_table = _make_manager(num_pages=8, page_size=4)
    req = _make_req(0, 6)
    manager.allocate_paged([req])
    # 请求只用了 6 个 token，但占用了 2 个完整页
    req.cached_len = 6
    manager.cache_req(req, finished=True)
    manager.check_integrity()
    assert torch.all(manager.free_slots % 4 == 0)
    assert len(manager.free_slots) == 8


def test_insufficient_pages_raises_clear_error() -> None:
    manager, _ = _make_manager(num_pages=1, page_size=4)
    req = _make_req(0, 12)
    with pytest.raises((RuntimeError, NotImplementedError)):
        manager.allocate_paged([req])
