"""RoPE（旋转位置编码）。

采用 HF 的 ``rotate_half`` 约定：把 head_dim 分成前后两半旋转，而不是相邻元素
交错。cos/sin 预计算成表，按绝对 positions 取值。
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from .base import StateLessOP


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        device: torch.device,
    ) -> None:
        if rotary_dim != head_dim:
            raise ValueError(f"partial rotary is not supported: {rotary_dim=} {head_dim=}")
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
        )
        positions = torch.arange(max_position, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        # 每个频率覆盖前后两半，拼成完整 head_dim
        cos = freqs.cos()
        sin = freqs.sin()
        # 注册成普通属性会进入 state_dict，因此用下划线前缀的 buffer
        self._cos = torch.cat((cos, cos), dim=-1)
        self._sin = torch.cat((sin, sin), dim=-1)

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self._cos.index_select(0, positions.reshape(-1)).to(query.dtype)
        sin = self._sin.index_select(0, positions.reshape(-1)).to(query.dtype)
        cos = cos.reshape(*positions.shape, 1, -1)  # [..., T, 1, D]
        sin = sin.reshape(*positions.shape, 1, -1)
        return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    scaling: tuple[tuple[str, Any], ...] | None,
    device: torch.device,
) -> RotaryEmbedding:
    """按配置与设备缓存 RoPE 表，避免每层重复分配 cos/sin。"""
    if scaling is not None:
        rope_type = dict(scaling).get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(f"stage 1 only supports default RoPE, got {rope_type!r}")
    return RotaryEmbedding(head_dim, rotary_dim, max_position, base, device=device)
