"""模块基类与严格 state_dict 协议。

约定：张量参数直接作为普通属性保存，嵌套模块也是普通属性。
``state_dict`` 按属性名递归拼接 key，因此属性名就是权重 key 的一部分。

加载结束时若仍有未消费的 key 直接报错，用来尽早暴露权重映射错误。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

import torch

_STATE_DICT = dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name

# 基础操作类
class BaseOP:
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = {} if result is None else result
        for name, value in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(value, torch.Tensor):
                result[_concat_prefix(prefix, name)] = value
            elif isinstance(value, BaseOP):
                value.state_dict(prefix=_concat_prefix(prefix, name), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for name, value in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(value, torch.Tensor):
                key = _concat_prefix(prefix, name)
                if key not in state_dict:
                    raise RuntimeError(f"Missing key in state_dict: {key}")
                item = state_dict.pop(key)
                if value.shape != item.shape or value.dtype != item.dtype:
                    raise RuntimeError(
                        f"Shape/dtype mismatch for {key}: "
                        f"expected {tuple(value.shape)} {value.dtype}, "
                        f"got {tuple(item.shape)} {item.dtype}"
                    )
                setattr(self, name, item)
            elif isinstance(value, BaseOP):
                value.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {sorted(state_dict)}")


class StateLessOP(BaseOP):
    """不持有参数、也不参与 state_dict 的模块（如 RoPE、注意力封装）。"""

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        return {} if result is None else result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {sorted(state_dict)}")


T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    """按顺序组合一组模块，key 使用数字下标（``layers.0``）。"""

    def __init__(self, ops: list[T]) -> None:
        self.op_list = ops

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("OPList is a container; iterate over op_list")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = {} if result is None else result
        for index, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(index)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for index, op in enumerate(self.op_list):
            op.load_state_dict(
                state_dict, prefix=_concat_prefix(prefix, str(index)), _internal=True
            )
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {sorted(state_dict)}")
