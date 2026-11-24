"""采样：greedy 快路径 + 每请求 temperature / top_k / top_p。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from minisgl.core import Batch


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    # 与 HF 基线一致：先转 fp32 再 argmax
    return logits.float().argmax(dim=-1)


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None = None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None

    @property
    def is_greedy(self) -> bool:
        return self.temperatures is None


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [req.sampling_params for req in batch.reqs]
        if all(param.is_greedy for param in params):
            # 全 greedy 时不构造任何采样张量
            return BatchSamplingArgs()

        min_t = 1e-6
        min_p = 1e-6
        temperatures = [
            max(0.0 if param.is_greedy else param.temperature, min_t) for param in params
        ]
        top_ks = [param.top_k if param.top_k >= 1 else self.vocab_size for param in params]
        top_ps = [min(max(param.top_p, min_p), 1.0) for param in params]

        top_k = None
        if any(k != self.vocab_size for k in top_ks):
            top_k = torch.tensor(top_ks, dtype=torch.int32, device=self.device)
        top_p = None
        if any(p < 1.0 for p in top_ps):
            top_p = torch.tensor(top_ps, dtype=torch.float32, device=self.device)
        return BatchSamplingArgs(
            temperatures=torch.tensor(temperatures, dtype=torch.float32, device=self.device),
            top_k=top_k,
            top_p=top_p,
        )

    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs | None) -> torch.Tensor:
        if args is None or args.is_greedy:
            return greedy_sample(logits).to(torch.int32)

        batch_size = logits.shape[0]
        logits = logits.float()
        assert args.temperatures is not None
        temperatures = args.temperatures.to(logits.device).to(logits.dtype).clamp_min(1e-6)
        processed = logits / temperatures.unsqueeze(-1)

        for index in range(batch_size):
            row = processed[index]
            if args.top_k is not None:
                k = int(args.top_k[index].item())
                if 0 < k < row.numel():
                    threshold = row.topk(k).values[-1]
                    row[row < threshold] = float("-inf")
            if args.top_p is not None:
                top_p = float(args.top_p[index].item())
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(row, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative = probs.cumsum(dim=-1)
                    # 保留累计概率不超过 top_p 的 token，并至少保留第一个
                    remove = (cumulative - probs) > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    row.fill_(float("-inf"))
                    row.scatter_(0, sorted_indices, sorted_logits)

        probs = torch.softmax(processed, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1).to(torch.int32)
