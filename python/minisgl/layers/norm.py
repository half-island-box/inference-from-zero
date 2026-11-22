"""RMSNorm 的纯 PyTorch 实现。"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        # 推理不需要梯度，权重固定
        self.weight = nn.Parameter(torch.empty(size), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 与 HF 对齐：fp32 归一化后乘 weight，再转回输入 dtype
        dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * x_fp32.to(dtype)
