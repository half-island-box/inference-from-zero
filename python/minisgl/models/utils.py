"""Llama/Qwen3 共用的模型组件。"""

from __future__ import annotations

import torch

from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    Linear,
    RMSNorm,
    gelu_and_mul,
    get_rope,
    silu_and_mul,
)

from .config import ModelConfig

_ACT_FNS = {"silu": silu_and_mul, "gelu": gelu_and_mul}


class GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self.gate_up_proj = Linear(config.hidden_size, 2 * config.intermediate_size, has_bias=False)
        try:
            self.act_fn = _ACT_FNS[config.hidden_act]
        except KeyError:
            raise ValueError(f"unsupported activation: {config.hidden_act!r}") from None
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class RopeAttn(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, has_qk_norm: bool = False) -> None:
        rotary_config = config.rotary_config
        self.qkv_proj = Linear(
            config.hidden_size,
            (config.num_qo_heads + 2 * config.num_kv_heads) * config.head_dim,
            has_bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps) if has_qk_norm else None
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps) if has_qk_norm else None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            rotary=get_rope(
                rotary_config.head_dim,
                rotary_config.rotary_dim,
                rotary_config.max_position,
                rotary_config.base,
                tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
                str(torch.get_default_device()),
            ),
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = Linear(
            config.num_qo_heads * config.head_dim, config.hidden_size, has_bias=False
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.attn(self.qkv_proj(x), positions))
