"""注意力层：QKV 拆分、QK Norm、RoPE。

两种运行模式：

- 阶段 1 路径（没有全局 batch）：输入 ``qkv`` 为 ``[B, T, ...]``，直接做 causal SDPA；
- 阶段 2 路径（存在全局 batch）：输入为扁平 ``[T, ...]``，QK Norm + RoPE 后交给
  attention backend，由 backend 负责写 KV cache 和读取历史 KV。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..core import get_active_batch
from .norm import RMSNorm
from .rotary import RotaryEmbedding


class AttentionLayer(nn.Module):
    _rotary: RotaryEmbedding

    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary: RotaryEmbedding,
        qk_norm_eps: float | None = None,
    ) -> None:
        super().__init__()
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError(f"{num_qo_heads=} must be divisible by {num_kv_heads=}")
        self.layer_id = layer_id
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.qo_attn_dim = num_qo_heads * head_dim
        self.kv_attn_dim = num_kv_heads * head_dim
        # RoPE 由模型层共享，这里只保存非注册引用
        object.__setattr__(self, "_rotary", rotary)
        # QK Norm 只在这里注册一次，避免与 RopeAttn 重复
        self.q_norm = RMSNorm(head_dim, qk_norm_eps) if qk_norm_eps is not None else None
        self.k_norm = RMSNorm(head_dim, qk_norm_eps) if qk_norm_eps is not None else None

    def forward(self, qkv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch = get_active_batch()
        if batch is not None:
            return self._forward_batch(qkv, positions, batch)

        # 阶段 1：整段序列重算
        bsz, seq_len, _ = qkv.shape
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        q = q.view(bsz, seq_len, self.num_qo_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q, k = self._rotary.forward(positions, q, k)

        # [B, T, H, D] -> [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.num_kv_heads != self.num_qo_heads:
            group = self.num_qo_heads // self.num_kv_heads
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.transpose(1, 2).reshape(bsz, seq_len, self.qo_attn_dim)

    def _forward_batch(
        self, qkv: torch.Tensor, positions: torch.Tensor, batch: object
    ) -> torch.Tensor:
        # 扁平 token 布局：[T, H, D]
        num_tokens = qkv.shape[0]
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        q = q.view(num_tokens, self.num_qo_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q, k = self._rotary.forward(positions, q, k)
        # 具体 backend 由全局 Context 持有；这里延迟导入避免循环依赖
        from ..core import get_global_ctx

        ctx = get_global_ctx()
        out = ctx.attn_backend.forward(q, k, v, self.layer_id, batch)  # type: ignore[arg-type]
        return out.reshape(num_tokens, self.qo_attn_dim)
