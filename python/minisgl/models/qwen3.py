"""Qwen3 结构：在 Llama 的基础上给每个 head 加 QK RMSNorm。"""

from __future__ import annotations

import torch
from torch import nn

from minisgl.layers import Embedding, LMHead, RMSNorm

from .base import BaseLLMModel
from .config import ModelConfig
from .utils import GatedMLP, RopeAttn


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.self_attn = RopeAttn(config, layer_id, has_qk_norm=True)
        self.mlp = GatedMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self._layer_id = layer_id

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn.forward(self.input_layernorm.forward(x), positions)
        return x + self.mlp.forward(self.post_attention_layernorm.forward(x))


class Qwen3Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        for layer in self.layers:
            x = layer.forward(x, positions)
        return self.norm.forward(x)


class Qwen3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = LMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        input_ids, positions = self._resolve_inputs(input_ids, positions)
        hidden = self.model.forward(input_ids, positions)
        if logits_to_keep is not None:
            hidden = hidden[:, -logits_to_keep:, :]
        return self.lm_head.forward(hidden)


__all__ = ["Qwen3ForCausalLM"]
