"""阶段 1 的 naive 生成：prefill 后每步重算整个序列。

没有 KV Cache，所以第 t 个输出 token 需要一次长度 t 的完整前向。这正是阶段 2
要解决的瓶颈。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from minisgl.core import SamplingParams
from minisgl.models import BaseLLMModel

from .sample import greedy_sample


@dataclass
class GenerationStats:
    prefill_ms: float = 0.0
    decode_ms: list[float] = field(default_factory=list)

    @property
    def mean_decode_ms(self) -> float | None:
        return sum(self.decode_ms) / len(self.decode_ms) if self.decode_ms else None


@dataclass
class GenerationResult:
    output_ids: list[int]
    stop_reason: str
    stats: GenerationStats
    prefill_logits: torch.Tensor | None = None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def naive_generate(
    model: BaseLLMModel,
    input_ids: list[int],
    sampling_params: SamplingParams | None = None,
    *,
    eos_token_id: int | None = None,
    device: str | torch.device = "cpu",
) -> GenerationResult:
    params = sampling_params or SamplingParams()
    if not input_ids:
        raise ValueError("input_ids must not be empty")
    if not params.is_greedy:
        raise NotImplementedError("stage 1 only supports greedy sampling")
    if params.max_tokens == 0:
        return GenerationResult([], "length", GenerationStats())

    device = torch.device(device)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    stats = GenerationStats()
    generated: list[int] = []
    stop_reason = "length"

    _sync(device)
    start = time.perf_counter()
    logits = model.forward(ids, logits_to_keep=1)
    _sync(device)
    stats.prefill_ms = (time.perf_counter() - start) * 1e3
    last_logits = logits[0, -1]
    prefill_logits = last_logits

    while len(generated) < params.max_tokens:
        token = int(greedy_sample(last_logits).item())
        if not params.ignore_eos and eos_token_id is not None and token == eos_token_id:
            stop_reason = "eos"
            break

        generated.append(token)
        if len(generated) == params.max_tokens:
            break

        next_id = torch.tensor([[token]], dtype=torch.long, device=device)
        ids = torch.cat([ids, next_id], dim=1)
        _sync(device)
        start = time.perf_counter()
        logits = model.forward(ids, logits_to_keep=1)
        _sync(device)
        stats.decode_ms.append((time.perf_counter() - start) * 1e3)
        last_logits = logits[0, -1]

    return GenerationResult(generated, stop_reason, stats, prefill_logits)
