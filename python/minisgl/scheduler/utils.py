"""调度器的请求包装与状态。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from minisgl.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor  # CPU tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class RequestState:
    uid: int
    input_ids: list[int]
    sampling_params: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    stop_reason: str | None = None
    arrival_time: float = field(default_factory=time.perf_counter)
    token_times: list[float] = field(default_factory=list)
    prefill_logits: torch.Tensor | None = None
    finished: bool = False
    finished_time: float | None = None

    @property
    def ttft_ms(self) -> float | None:
        if not self.token_times:
            return None
        return (self.token_times[0] - self.arrival_time) * 1e3

    @property
    def itl_ms(self) -> list[float]:
        return [
            (later - earlier) * 1e3
            for earlier, later in zip(self.token_times, self.token_times[1:])
        ]
