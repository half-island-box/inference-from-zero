"""模型注册、构建与权重加载。"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from contextlib import contextmanager
from typing import Iterator, cast

import torch

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
) -> BaseLLMModel:
    arch = architecture or (config.architectures[0] if config.architectures else None)
    if arch is None:
        arch = _MODEL_TYPE_DEFAULTS.get(config.model_type)
    if arch not in _ARCH_REGISTRY:
        raise ValueError(f"unsupported architecture: {arch!r}")

    module_name, class_name = _ARCH_REGISTRY[arch]
    module = importlib.import_module(module_name, package=__package__)
    constructor = cast("Callable[[ModelConfig], BaseLLMModel]", getattr(module, class_name))
    with _default_device_dtype(device, dtype):
        return constructor(config)


__all__ = [
    "BaseLLMModel",
    "ModelConfig",
    "RotaryConfig",
    "convert_hf_state_dict",
    "create_model",
    "load_weight",
    "resolve_model_path",
]
