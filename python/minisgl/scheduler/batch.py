"""把 Batch 组装成模型 forward 需要的扁平输入。"""

from __future__ import annotations

import torch

from minisgl.core import Batch


def prepare_batch(batch: Batch, page_table: torch.Tensor, token_pool: torch.Tensor) -> None:
    """填充 ``input_ids`` / ``positions`` / ``out_loc``。

    - positions：每个 token 的绝对位置，prefill 为 ``cached_len..device_len-1``；
    - input_ids：从 token_pool 取出本轮要计算的 token；
    - out_loc：本轮 K/V 要写入的物理 token 槽位。
    """
    batch.padded_reqs = batch.reqs
    positions: list[torch.Tensor] = []
    input_ids: list[torch.Tensor] = []
    out_loc: list[torch.Tensor] = []

    for req in batch.padded_reqs:
        start, end = req.cached_len, req.device_len
        positions.append(torch.arange(start, end, dtype=torch.long, device=page_table.device))
        input_ids.append(token_pool[req.table_idx, start:end].to(torch.long))
        out_loc.append(page_table[req.table_idx, start:end].to(torch.long))

    batch.positions = torch.cat(positions)
    batch.input_ids = torch.cat(input_ids)
    batch.out_loc = torch.cat(out_loc)
