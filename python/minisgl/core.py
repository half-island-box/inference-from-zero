"""引擎的基础数据类型。

阶段 1 只定义采样参数；``Req`` / ``Batch`` / ``Context`` 从阶段 2 开始引入。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

    @property
    def is_greedy(self) -> bool:
        # temperature=0 或 top_k=1 且不做 nucleus 截断时退化为 argmax
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
