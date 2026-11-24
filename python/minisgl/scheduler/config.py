"""调度器配置。"""

from __future__ import annotations

from dataclasses import dataclass

from minisgl.engine import EngineConfig


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "naive"
    eos_token_id: int | None = None
    record_prefill_logits: bool = False
