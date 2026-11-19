"""pytest 公共配置：路径设置与基于 GPU 的自动跳过。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: test requires a CUDA device")
    config.addinivalue_line("markers", "slow: end-to-end test that loads model weights")


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _cuda_available():
        return
    skip = pytest.mark.skip(reason="requires CUDA")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
