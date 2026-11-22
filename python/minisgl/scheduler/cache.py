"""页分配器：页表写入、分配、释放与完整性检查。

``free_slots`` 里存的是页起始 token 槽位（page_size=2 时形如 ``[0, 2, 4, ...]``）。
全局页表始终按 page_size=1 记录原始 token 槽位，attention 后端在边界处换算页号。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import torch

from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl.kvcache.base import BasePrefixCache

if TYPE_CHECKING:
    from minisgl.core import Req


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


class CacheManager:
    def __init__(
        self,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor,
        type: str = "naive",
    ) -> None:
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache: BasePrefixCache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        # 匹配前缀时去掉最后一个 token：它还没有被计入 KV
        return self.prefix_cache.match_prefix(input_ids[:-1])

    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: list[Req]) -> None:
        needed_pages = 0
        allocation_info: list[tuple[int, int, int]] = []
        for req in reqs:
            first_page = _ceil_div(req.cached_len, self.page_size)
            last_page = _ceil_div(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            self._write_page_table(allocated, allocation_info)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        # 有效缓存区间：[0, req.cached_len)
        # [0, old_handle.cached_len)                prefill 前已在缓存里
        # [old_handle.cached_len, req.cached_len)  本次为请求分配、尚未插入缓存
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        self.unlock(old_handle)
        # 这段已经由前缀缓存接管，释放本请求重复分配的部分
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:
            # 请求结束，尾部未插入缓存的页也要释放
            self._free(page_indices[new_handle.cached_len :])
        else:
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed: "
                f"free_pages({len(self.free_slots)}) + "
                f"cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1 and not bool(torch.all(self.free_slots % self.page_size == 0)):
            raise RuntimeError("free slots are not page aligned")

    @contextmanager
    def lazy_free_region(self) -> Iterator[None]:
        """延迟归还页，给重叠调度留接口（阶段 9）。"""

        lazy_free_list: list[torch.Tensor] = []

        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices[:: self.page_size])

        original = self._free
        self._free = lazy_free  # type: ignore[method-assign]
        try:
            yield
        finally:
            self._free = original  # type: ignore[method-assign]
            if lazy_free_list:
                self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            if len(self.free_slots) < needed_pages:
                raise RuntimeError(
                    f"not enough KV pages: need {needed_pages}, have {len(self.free_slots)}"
                )
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()

    def _write_page_table(
        self,
        allocated: torch.Tensor,
        allocation_info: list[tuple[int, int, int]],
    ) -> None:
        offset = 0
        for table_idx, first_page, last_page in allocation_info:
            first_pos = first_page * self.page_size
            last_pos = last_page * self.page_size
            length = last_pos - first_pos
            self.page_table[table_idx, first_pos:last_pos] = allocated[offset : offset + length]
            offset += length
        assert offset == len(allocated), "allocated token count does not match page table writes"
