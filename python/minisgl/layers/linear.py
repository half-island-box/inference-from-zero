"""普通 Linear 层（阶段 2 不做张量并行）。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, has_bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if has_bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
