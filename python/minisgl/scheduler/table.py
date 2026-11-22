"""请求槽位与 token 池。"""

from __future__ import annotations

import torch


class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        # dummy 请求也从这个池子取输入，因此初始化为合法的 token id 0
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        if not self._free_slots:
            raise RuntimeError("no free request slot; increase max_running_req")
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)
