"""RMSNorm 的纯 PyTorch 实现。"""

from __future__ import annotations

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.weight = torch.empty(size)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 与 HF 对齐：fp32 归一化后乘 weight，再转回输入 dtype
        dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * x_fp32.to(dtype)
