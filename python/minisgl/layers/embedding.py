"""词嵌入与 LM Head（阶段 2 不做词表并行）。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim), requires_grad=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


# 最后一个 layer 的 hidden state -> logits
class LMHead(nn.Module):
    weight: nn.Parameter
    _tied_embedding: Embedding

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tie_word_embeddings: bool = False,
        tied_embedding: Embedding | None = None,
    ) -> None:
        super().__init__()
        if tie_word_embeddings != (tied_embedding is not None):
            raise ValueError("tie_word_embeddings requires a matching tied_embedding")
        self.tie_word_embeddings = tie_word_embeddings
        if tie_word_embeddings:
            # 只保存非注册引用：既不出现在 state_dict，也不会被重复注册为子模块
            assert tied_embedding is not None
            object.__setattr__(self, "_tied_embedding", tied_embedding)
        else:
            self.weight = nn.Parameter(
                torch.empty(num_embeddings, embedding_dim), requires_grad=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tie_word_embeddings:
            weight = self._tied_embedding.weight
        else:
            weight = self.weight
        return F.linear(x, weight)
