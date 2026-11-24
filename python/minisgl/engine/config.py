"""引擎配置。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# 引擎配置类
@dataclass(frozen=True)
class EngineConfig:
    model_path: str = ""
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    max_running_req: int = 256
    page_size: int = 1
    memory_ratio: float = 0.9 # 显存占用
    max_seq_len_override: int | None = None
    num_page_override: int | None = None
    attention_backend: str = "torch"
    seed: int = 0
