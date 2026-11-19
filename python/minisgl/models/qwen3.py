"""Qwen3 结构：在 Llama 的基础上给每个 head 加 QK RMSNorm。"""

from __future__ import annotations

import torch

from minisgl.layers import BaseOP, Embedding, LMHead, OPList, RMSNorm

from .base import BaseLLMModel
from .config import ModelConfig
from .utils import GatedMLP, RopeAttn


class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self.self_attn = RopeAttn(config, layer_id, has_qk_norm=True)
        self.mlp = GatedMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self._layer_id = layer_id

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), positions)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers.op_list:
            x = layer(x, positions)
        return self.norm(x)


class Qwen3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self.model = Qwen3Model(config)
        self.lm_head = LMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        hidden = self.model.forward(input_ids, self._positions(input_ids, positions))
        if logits_to_keep is not None:
            hidden = hidden[:, -logits_to_keep:, :]
        return self.lm_head(hidden)


__all__ = ["Qwen3ForCausalLM"]
