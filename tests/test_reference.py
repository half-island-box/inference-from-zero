"""与落盘黄金基线做端到端回归（路线图阶段 0）。

需要 CUDA 设备和模型权重，因此标记为 ``gpu`` 与 ``slow``。
运行前先在 GPU 主机上生成一次基线::

    python tools/reference_hf.py --model Qwen/Qwen3-0.6B
    pytest tests/test_reference.py -m gpu
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "out" / "golden" / "Qwen__Qwen3-0.6B" / "golden.json"
MODEL = os.environ.get("GOLDEN_MODEL", "Qwen/Qwen3-0.6B")


def _golden_path() -> Path:
    return Path(os.environ.get("GOLDEN_PATH", str(DEFAULT_GOLDEN)))


@pytest.mark.gpu
@pytest.mark.slow
def test_reference_matches_golden(tmp_path: Path) -> None:
    golden = _golden_path()
    if not golden.exists():
        pytest.skip(f"golden baseline not found at {golden}; run tools/reference_hf.py first")

    out_dir = tmp_path / "golden"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "reference_hf.py"),
            "--model",
            MODEL,
            "--output",
            str(out_dir),
        ],
        check=True,
    )
    compare = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "compare.py"),
            "--baseline",
            str(golden),
            "--actual",
            str(out_dir / "golden.json"),
        ]
    )
    assert compare.returncode == 0, "greedy outputs or logits diverged from the golden baseline"


@pytest.mark.gpu
@pytest.mark.slow
def test_reference_is_deterministic(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "reference_hf.py"),
            "--model",
            MODEL,
            "--output",
            str(tmp_path / "golden"),
            "--limit",
            "3",
            "--check-determinism",
        ],
        check=True,
    )
    assert proc.returncode == 0
