"""词嵌入与 LM Head（阶段 1 不做词表并行）。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseOP


class Embedding(BaseOP):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.weight = torch.empty(num_embeddings, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


class LMHead(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tie_word_embeddings: bool = False,
        tied_embedding: Embedding | None = None,
    ) -> None:
        if tie_word_embeddings != (tied_embedding is not None):
            raise ValueError("tie_word_embeddings requires a matching tied_embedding")
        self.tie_word_embeddings = tie_word_embeddings
        self.weight = None if tie_word_embeddings else torch.empty(num_embeddings, embedding_dim)
        self._tied_embedding = tied_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._tied_embedding.weight if self._tied_embedding is not None else self.weight
        return F.linear(x, weight)

    def state_dict(
        self, *, prefix: str = "", result: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        if self.tie_word_embeddings:
            return {} if result is None else result
        return super().state_dict(prefix=prefix, result=result)

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if self.tie_word_embeddings:
            # 共享权重时 checkpoint 里可能仍带一份 lm_head.weight
            state_dict.pop(f"{prefix}.weight" if prefix else "weight", None)
            return
        super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
