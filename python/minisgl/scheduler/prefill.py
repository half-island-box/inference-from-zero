"""Prefill 管理：pending 队列、准入控制与组 batch（阶段 3 不分块）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from minisgl.core import Batch, Req

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager


class ChunkedReq(Req):
    """阶段 4 启用；阶段 3 只保留类型定义。"""

    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    max_seq_len: int

    def _estimate(self, pending: PendingReq, cached_len: int) -> tuple[int, int]:
        """返回 (输出长度, 需预留的 token 数)，预留里包含页尾浪费。"""
        output_len = min(pending.output_len, self.max_seq_len - pending.input_len)
        extend_len = pending.input_len - cached_len
        page_waste = self.cache_manager.page_size - 1
        return output_len, extend_len + output_len + page_waste

    def _try_allocate_one(self, pending: PendingReq) -> tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None

        handle = self.cache_manager.match_prefix(pending.input_ids).cuda_handle
        cached_len = handle.cached_len
        output_len, estimated_len = self._estimate(pending, cached_len)
        if output_len < 0:
            return None

        # 准入时同时预留输入与未来输出，避免 decode 中途 OOM
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            self.cache_manager.unlock(handle)
            return None

        table_idx = self.table_manager.allocate()
        token_pool = self.table_manager.token_pool
        token_pool[table_idx, cached_len : pending.input_len] = pending.input_ids[cached_len:].to(
            token_pool.device
        )
        if cached_len > 0:
            self.table_manager.page_table[table_idx, :cached_len] = handle.get_matched_indices()
        return handle, table_idx

    def _make_req(
        self,
        pending: PendingReq,
        handle: BaseCacheHandle,
        table_idx: int,
    ) -> Req:
        cached_len = handle.cached_len
        output_len, estimated_len = self._estimate(pending, cached_len)
        # 阶段 3 不切块：单个请求可以超过 max_extend_tokens，避免活锁
        self.token_budget -= pending.input_len - cached_len
        self.reserved_size += estimated_len
        return Req(
            input_ids=pending.input_ids.clone(),
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=output_len,
            uid=pending.uid,
            sampling_params=pending.sampling_params,
            cache_handle=handle,
        )

    def try_add_one(self, pending: PendingReq) -> Req | None:
        resource = self._try_allocate_one(pending)
        if resource is None:
            return None
        handle, table_idx = resource
        return self._make_req(pending, handle, table_idx)


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    max_seq_len: int
    max_extend_tokens: int = 8192
    pending_list: list[PendingReq] = field(default_factory=list)

    def add_req(self, uid: int, input_ids: torch.Tensor, sampling_params) -> None:  # noqa: ANN001
        self.pending_list.append(
            PendingReq(uid=uid, input_ids=input_ids, sampling_params=sampling_params)
        )

    def schedule_next_batch(self) -> Batch | None:
        if not self.pending_list:
            return None

        adder = PrefillAdder(
            token_budget=self.max_extend_tokens,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            max_seq_len=self.max_seq_len,
        )
        reqs: list[Req] = []
        for pending in self.pending_list:
            if reqs and adder.token_budget <= 0:
                break
            req = adder.try_add_one(pending)
            if req is None:
                break  # FCFS：队首放不下就等下一轮
            reqs.append(req)

        if not reqs:
            return None
        self.pending_list = self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        for index, pending in enumerate(self.pending_list):
            if pending.uid == uid:
                self.pending_list.pop(index)
                return pending.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
