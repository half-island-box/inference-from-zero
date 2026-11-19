"""单请求全序列注意力：QKV 拆分、QK Norm、RoPE、SDPA。

阶段 1 每次都重算整段序列，因此 Q/K/V 长度相同，直接使用 causal SDPA。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import StateLessOP
from .norm import RMSNorm
from .rotary import RotaryEmbedding


class AttentionLayer(StateLessOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary: RotaryEmbedding,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ) -> None:
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError(f"{num_qo_heads=} must be divisible by {num_kv_heads=}")
        self.layer_id = layer_id
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.qo_attn_dim = num_qo_heads * head_dim
        self.kv_attn_dim = num_kv_heads * head_dim
        self.rotary = rotary
        # 仅持有引用，q_norm/k_norm 的权重由上层模块负责加载
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = qkv.shape
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        q = q.view(batch, seq_len, self.num_qo_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q, k = self.rotary(positions, q, k)

        # [B, T, H, D] -> [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.num_kv_heads != self.num_qo_heads:
            group = self.num_qo_heads // self.num_kv_heads
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.transpose(1, 2).reshape(batch, seq_len, self.qo_attn_dim)
