"""attention backend 接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from minisgl.core import Batch


@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor:
        """每个请求最后一个 query token 在扁平 logits 中的下标。"""


class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        """``q``: [T, Hq, D]，``k``/``v``: [T, Hkv, D]；返回 [T, Hq, D]。"""

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    # CUDA Graph 相关接口留到阶段 8，阶段 2 的参考后端不实现
    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        raise NotImplementedError

    def prepare_for_capture(self, batch: Batch) -> None:
        raise NotImplementedError

    def prepare_for_replay(self, batch: Batch) -> None:
        raise NotImplementedError
