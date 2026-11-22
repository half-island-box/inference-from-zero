"""attention backend 构造入口。"""

from __future__ import annotations

from .base import BaseAttnBackend, BaseAttnMetadata
from .torch_backend import TorchAttnBackend, TorchAttnMetadata


def create_attention_backend(backend: str = "torch") -> BaseAttnBackend:
    if backend == "torch":
        return TorchAttnBackend()
    raise ValueError(f"unsupported attention backend: {backend!r}")


__all__ = [
    "BaseAttnBackend",
    "BaseAttnMetadata",
    "TorchAttnBackend",
    "TorchAttnMetadata",
    "create_attention_backend",
]
