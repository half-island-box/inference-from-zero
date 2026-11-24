"""Radix 前缀缓存：按页对齐的 token 前缀树，带引用计数与 LRU 淘汰。

节点保存一段连续 token（key）和对应的物理 KV 槽位（value），长度始终按
``page_size`` 对齐；叶子按时间戳做 LRU 淘汰，``ref_count > 0`` 的节点受保护。
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass

import torch

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KeyFn = Callable[[torch.Tensor], Hashable]


def _align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def _compare_key(key: torch.Tensor, other: torch.Tensor) -> int:
    """返回第一个不同 token 的位置；完全相同则返回较短长度。"""
    length = min(len(key), len(other))
    diff = key[:length] != other[:length]
    if not bool(diff.any()):
        return length
    return int(diff.nonzero()[0].item())


class RadixTreeNode:
    _counter = 0

    def __init__(self, key_fn: KeyFn, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: dict[Hashable, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count = 0
        self.uuid = RadixTreeNode._counter
        RadixTreeNode._counter += 1
        self.timestamp = time.monotonic_ns() if tic is None else tic

        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def key(self) -> torch.Tensor:
        return self._key

    @property
    def value(self) -> torch.Tensor:
        return self._value

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        return _compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        """把当前节点在 ``pos`` 处一分为二，返回前缀部分（新父节点）。"""
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)
        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode
    empty_tensor: torch.Tensor

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        values: list[torch.Tensor] = []
        while not node.is_root():
            values.append(node.value)
            node = node.parent
        if not values:
            return self.empty_tensor
        values.reverse()
        return torch.cat(values)


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device, page_size: int) -> None:
        super().__init__()
        self.device = device
        self.page_size = page_size
        self.key_fn: KeyFn = _make_key_fn(page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root 始终受保护

    # ------------------------------------------------------------------ 锁
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    # ------------------------------------------------------------------ 匹配 / 插入
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node, self.empty_tensor))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = _align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node, self.empty_tensor))

    # ------------------------------------------------------------------ 淘汰
    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        if size > self.evictable_size:
            raise RuntimeError(f"cannot evict {size} tokens, only {self.evictable_size} evictable")

        leaves = self._collect_evictable_leaves()
        heapq.heapify(leaves)
        evicted_indices: list[torch.Tensor] = []
        evicted_size = 0
        while evicted_size < size:
            if not leaves:
                raise RuntimeError(f"cannot evict enough cache: need {size}, got {evicted_size}")
            node = heapq.heappop(leaves)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node.key)]
            # root 始终受保护；父节点变成无引用的叶子后才能继续淘汰
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leaves, parent)
        return torch.cat(evicted_indices)

    def reset(self) -> None:
        raise NotImplementedError("RadixPrefixCache.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=self.evictable_size, protected_size=self.protected_size)

    def check_integrity(self) -> None:
        evictable = protected = 0
        stack = list(self.root_node.children.values())
        while stack:
            node = stack.pop()
            if node.ref_count > 0:
                protected += node.length
            else:
                evictable += node.length
            stack.extend(node.children.values())
        if (evictable, protected) != (self.evictable_size, self.protected_size):
            raise RuntimeError(
                "RadixPrefixCache integrity failed: "
                f"walk=({evictable}, {protected}) != "
                f"counters=({self.evictable_size}, {self.protected_size})"
            )

    # ------------------------------------------------------------------ 内部
    def _collect_evictable_leaves(self) -> list[RadixTreeNode]:
        leaves: list[RadixTreeNode] = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leaves.append(node)
            else:
                stack.extend(node.children.values())
        return leaves

    def _tree_walk(self, input_ids: torch.Tensor) -> tuple[RadixTreeNode, int]:
        prefix_len = 0
        total_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < total_len:
            child = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child is None:
                return node, prefix_len
            node = child

            # key 按页对齐，至少能匹配一页
            match_len = _align_down(node.get_match_len(input_ids[prefix_len:]), self.page_size)
            prefix_len += match_len
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len
            node.timestamp = tic
        return node, prefix_len


def _make_key_fn(page_size: int) -> KeyFn:
    if page_size == 1:
        return lambda x: int(x[0].item())
    return lambda x: tuple(int(v) for v in x[:page_size].tolist())
