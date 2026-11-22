"""RoPE（旋转位置编码）。

采用 HF 的 ``rotate_half`` 约定：把 head_dim 分成前后两半旋转，而不是相邻元素
交错。cos/sin 预计算成表，按绝对 positions 取值。
"""

from __future__ import annotations

import functools
from typing import Any

import torch
from torch import nn

# meta 构建模型时，RoPE buffer 必须显式放到目标设备，不能留在 meta
_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: str | torch.device | None) -> torch.device | None:
    """设置 RoPE buffer 的构造设备，返回旧值方便恢复。"""
    global _ROPE_DEVICE
    old = _ROPE_DEVICE
    _ROPE_DEVICE = None if device is None else torch.device(device)
    return old


def resolve_rope_device(default: torch.device) -> torch.device:
    return _ROPE_DEVICE if _ROPE_DEVICE is not None else default


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    _cos: torch.Tensor
    _sin: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        if rotary_dim != head_dim:
            raise ValueError(f"partial rotary is not supported: {rotary_dim=} {head_dim=}")
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
        )
        positions = torch.arange(max_position, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # non-persistent buffer：不进入 state_dict，但会跟随 .to() / assign 之外的生命周期
        self.register_buffer("_cos", torch.cat((cos, cos), dim=-1), persistent=False)
        self.register_buffer("_sin", torch.cat((sin, sin), dim=-1), persistent=False)

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
            raise ValueError(f"stage 2 only supports default RoPE, got {rope_type!r}")
    return RotaryEmbedding(head_dim, rotary_dim, max_position, base, device=device)
