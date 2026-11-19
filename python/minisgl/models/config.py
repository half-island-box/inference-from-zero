"""从 HuggingFace config 提取引擎需要的模型参数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    attention_bias: bool
    model_type: str
    architectures: tuple[str, ...]

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        # 多模态模型的文本塔挂在 text_config 下
        if getattr(config, "text_config", None) is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if getattr(config, attr, None) is None and getattr(top, attr, None) is not None:
                    setattr(config, attr, getattr(top, attr))

        num_qo_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_qo_heads) or num_qo_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // num_qo_heads
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = (rope_scaling or {}).get("rope_theta", 10000.0)

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            rms_norm_eps=config.rms_norm_eps,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=float(rope_theta),
                scaling=rope_scaling,
            ),
            hidden_act=config.hidden_act,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
            attention_bias=bool(getattr(config, "attention_bias", False)),
            model_type=getattr(config, "model_type", "llama"),
            architectures=tuple(getattr(config, "architectures", None) or ()),
        )
