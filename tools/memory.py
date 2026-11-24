"""CUDA 显存统计：显式绑定设备。

``torch.cuda.max_memory_allocated()`` / ``reset_peak_memory_stats()`` 不传设备时
统计的是 ``torch.cuda.current_device()``（默认 0 号卡）。用 ``--device cuda:N``
在非 0 卡上跑时，必须把设备传进去，否则会读到 0 号卡的空统计（峰值恒为 0）。
"""

from __future__ import annotations

from typing import Any


def reset_peak_memory(device: Any) -> None:
    import torch

    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)


def peak_memory_gb(device: Any) -> float | None:
    import torch

    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(dev) / 1e9
