"""采样。阶段 1 只实现 greedy。"""

from __future__ import annotations

import torch


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    # 与 HF 基线一致：先转 fp32 再 argmax
    return logits.float().argmax(dim=-1)
