"""因果语言模型接口与严格权重加载。"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from minisgl.core import get_active_batch


class BaseLLMModel(nn.Module):
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        """``input_ids``: [B, T]（阶段 1）或 [T]（阶段 2 batch）；返回 logits。"""

    def _resolve_inputs(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = get_active_batch()
        if input_ids is None:
            if batch is None:
                raise RuntimeError("input_ids is required outside of a batch forward")
            input_ids = batch.input_ids
            if positions is None:
                positions = batch.positions
        elif positions is None:
            length = input_ids.shape[0] if input_ids.dim() == 1 else input_ids.shape[1]
            positions = torch.arange(length, device=input_ids.device)
        return input_ids, positions

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """保留阶段 1 的严格性：key 集合、shape、dtype 都在安装前检查。"""
        merged = dict(state_dict)
        if strict:
            expected = set(self.state_dict().keys())
            given = set(merged.keys())
            missing = sorted(expected - given)
            unexpected = sorted(given - expected)
            if missing:
                raise RuntimeError(f"Missing key in state_dict: {missing[0]}")
            if unexpected:
                raise RuntimeError(f"Unexpected keys in state_dict: {unexpected}")

            current = self.state_dict()
            for key in sorted(expected):
                want, got = current[key], merged[key]
                if want.shape != got.shape or want.dtype != got.dtype:
                    raise RuntimeError(
                        f"Shape/dtype mismatch for {key}: "
                        f"expected {tuple(want.shape)} {want.dtype}, "
                        f"got {tuple(got.shape)} {got.dtype}"
                    )
        return super().load_state_dict(merged, strict=strict, assign=assign)
