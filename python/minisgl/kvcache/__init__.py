"""KV Cache 与前缀缓存的构造入口。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    InsertResult,
    MatchResult,
    SizeInfo,
)

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> BaseKVCachePool:
    from .mha_pool import MHAKVCache

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        num_pages=num_pages,
        page_size=page_size,
        dtype=dtype,
        device=device,
    )


def create_prefix_cache(
    device: torch.device, type: str = "naive", page_size: int = 1
) -> BasePrefixCache:
    if type == "naive":
        from .naive_cache import NaivePrefixCache

        return NaivePrefixCache(device=device)
    if type == "radix":
        from .radix_cache import RadixPrefixCache

        return RadixPrefixCache(device=device, page_size=page_size)
    raise ValueError(f"unsupported prefix cache type: {type!r}")


__all__ = [
    "BaseCacheHandle",
    "BaseKVCachePool",
    "BasePrefixCache",
    "InsertResult",
    "MatchResult",
    "SizeInfo",
    "create_kvcache_pool",
    "create_prefix_cache",
]
