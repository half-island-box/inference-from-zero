"""门控 MLP 的激活函数。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _and_mul(x: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    return act * x[..., x.shape[-1] // 2 :]


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    return _and_mul(x, F.silu(x[..., : x.shape[-1] // 2]))


def gelu_and_mul(x: torch.Tensor) -> torch.Tensor:
    return _and_mul(x, F.gelu(x[..., : x.shape[-1] // 2]))
