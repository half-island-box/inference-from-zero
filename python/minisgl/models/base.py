"""因果语言模型接口。"""

from __future__ import annotations

from abc import abstractmethod

import torch

from minisgl.layers import BaseOP


class BaseLLMModel(BaseOP):
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        """``input_ids``: [B, T]；返回 logits [B, keep, vocab]（keep=None 时保留全序列）。"""

    @staticmethod
    def _positions(input_ids: torch.Tensor, positions: torch.Tensor | None) -> torch.Tensor:
        if positions is not None:
            return positions
        return torch.arange(input_ids.shape[1], device=input_ids.device)
