"""模型注册、构建与权重加载。"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import cast

import torch

from minisgl.layers import set_rope_device

from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .weight import convert_hf_state_dict, load_weight, resolve_model_path

_ARCH_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
}
_MODEL_TYPE_DEFAULTS = {"llama": "LlamaForCausalLM", "qwen3": "Qwen3ForCausalLM"}


@contextmanager
def _default_device_dtype(device: str | torch.device, dtype: torch.dtype) -> Iterator[None]:
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            yield
    finally:
        torch.set_default_dtype(old_dtype)


def create_model(
    config: ModelConfig,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    architecture: str | None = None,
    rope_device: str | torch.device | None = None,
) -> BaseLLMModel:
    """构造模型。

    ``device="meta"`` 用于大模型的流式加载；此时必须通过 ``rope_device`` 指定真实的
    目标设备，否则 RoPE 的 cos/sin buffer 会留在 meta 或错误的设备上。
    """
    arch = architecture or (config.architectures[0] if config.architectures else None)
    if arch is None:
        arch = _MODEL_TYPE_DEFAULTS.get(config.model_type)
    if arch not in _ARCH_REGISTRY:
        raise ValueError(f"unsupported architecture: {arch!r}")

    module_name, class_name = _ARCH_REGISTRY[arch]
    module = importlib.import_module(module_name, package=__package__)
    constructor = cast("Callable[[ModelConfig], BaseLLMModel]", getattr(module, class_name))

    target = torch.device(device)
    rope_target = torch.device(rope_device) if rope_device is not None else target
    if target.type == "meta" and rope_target.type == "meta":
        raise ValueError("rope_device must be a real device when constructing on meta")

    old_rope_device = set_rope_device(rope_target)
    try:
        with _default_device_dtype(target, dtype):
            return constructor(config)
    finally:
        set_rope_device(old_rope_device)


def load_model_weights(
    model: BaseLLMModel,
    state_dict: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype,
    assign: bool = False,
) -> None:
    """安装流式权重；dtype 不匹配直接报错，禁止静默 cast。"""
    merged = dict(state_dict)
    # tied embedding：checkpoint 里可能仍带一份 lm_head.weight，直接过滤
    lm_head = getattr(model, "lm_head", None)
    if getattr(lm_head, "tie_word_embeddings", False):
        merged = {key: value for key, value in merged.items() if not key.startswith("lm_head.")}
    for key, tensor in merged.items():
        if tensor.dtype != dtype:
            raise RuntimeError(
                f"dtype mismatch for {key}: checkpoint has {tensor.dtype}, expected {dtype}"
            )
    model.load_state_dict(merged, strict=True, assign=assign)


__all__ = [
    "BaseLLMModel",
    "ModelConfig",
    "RotaryConfig",
    "convert_hf_state_dict",
    "create_model",
    "load_model_weights",
    "load_weight",
    "resolve_model_path",
]
