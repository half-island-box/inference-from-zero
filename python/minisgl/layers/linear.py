"""普通 Linear 层（阶段 1 不做张量并行）。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseOP


class Linear(BaseOP):
    def __init__(self, in_features: int, out_features: int, has_bias: bool = False) -> None:
        self.weight = torch.empty(out_features, in_features)
        self.bias = torch.empty(out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
