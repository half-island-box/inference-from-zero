"""MHA KV Cache 池：``(2, num_layers, num_pages + 1, page_size, kv_heads, head_dim)``。

``+1`` 是 dummy page，留给后续 CUDA Graph 填充请求使用。阶段 2 的 ``store_kv`` 用
torch 的 ``index_copy_`` 作为参考实现，后续阶段再替换为自定义 kernel。
"""

from __future__ import annotations

import torch

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._num_layers = num_layers
        self._num_pages = num_pages + 1  # dummy page
        self._page_size = page_size
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._device = device
        self._kv_buffer = torch.empty(
            (2, num_layers, self._num_pages, page_size, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        # 先扁平化成按物理 token 槽位索引的视图，再写入本轮 K/V
        shape = (-1, self._num_kv_heads, self._head_dim)
        k_cache = self._k_buffer[layer_id].view(shape)
        v_cache = self._v_buffer[layer_id].view(shape)
        indices = out_loc.to(torch.long)
        k_cache.index_copy_(0, indices, k.reshape(shape).to(k_cache.dtype))
        v_cache.index_copy_(0, indices, v.reshape(shape).to(v_cache.dtype))

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_pages(self) -> int:
        """包含 dummy page 的总页数。"""
        return self._num_pages

    @property
    def page_size(self) -> int:
        return self._page_size
