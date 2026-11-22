"""generate_paged.py 日志记录格式的 CPU 单测。"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

pytest.importorskip("torch")

from scripts.generate_paged import format_bench_entry, format_generate_entry  # noqa: E402

META: dict[str, Any] = {
    "model": "Qwen/Qwen3-0.6B",
    "model_source": "modelscope",
    "dtype": "bfloat16",
    "torch_version": "2.4.0",
    "transformers_version": "4.56.0",
    "cuda_device": None,
}


def _args(**overrides: Any) -> argparse.Namespace:
    values = {
        "log_stage": 2,
        "log_note": "note",
        "device": "cpu",
        "page_size": 16,
        "max_running_req": 8,
        "max_tokens": 32,
        "ignore_eos": False,
        "seed": 0,
        "bench_token_len": 256,
        "bench_batch": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_format_generate_entry_for_stage2() -> None:
    stats = {
        "num_prompts": 20,
        "num_batches": 3,
        "prefill_mean_ms": 8.5,
        "decode_mean_ms": 4.25,
        "peak_memory_gb": 1.5,
    }
    entry = format_generate_entry(stats, args=_args(ignore_eos=True), meta=META)

    assert "paged KV Cache 批量生成 （阶段 2）" in entry
    assert "`Qwen/Qwen3-0.6B` / bfloat16（modelscope）" in entry
    assert "KV cache: page_size=16, max_running_req=8" in entry
    assert "max_tokens=32, ignore_eos=True, batch≤8" in entry
    assert "prefill 均值 = 8.50 ms" in entry
    assert "单步 decode 均值 = 4.25 ms" in entry
    assert entry.rstrip().endswith("- 结论 / 下一步: note")


@pytest.mark.parametrize("kind", ["batch", "seq"])
def test_format_bench_entry_table(kind: str) -> None:
    rows = [
        {"key": 1, "prefill_ms": 5.0, "decode_mean_ms": 1.5, "peak_memory_gb": 1.0},
        {"key": 8, "prefill_ms": 20.0, "decode_mean_ms": 6.0, "peak_memory_gb": 1.4},
    ]
    entry = format_bench_entry(rows, args=_args(), meta=META, kind=kind)

    if kind == "batch":
        assert "paged decode 随 batch size 扩展" in entry
        assert "| batch size | prefill (ms) | decode 均值 (ms) | 峰值显存 (GB) |" in entry
    else:
        assert "paged decode 随序列长度扩展" in entry
        assert "| prompt 长度 | prefill (ms) | decode 均值 (ms) | 峰值显存 (GB) |" in entry
    assert "| 1 | 5.00 | 1.50 | 1.00 |" in entry
    assert "| 8 | 20.00 | 6.00 | 1.40 |" in entry


def test_missing_values_render_as_dash() -> None:
    rows = [{"key": 1, "prefill_ms": None, "decode_mean_ms": None, "peak_memory_gb": None}]
    entry = format_bench_entry(rows, args=_args(), meta=META, kind="batch")
    assert "| 1 | - | - | - |" in entry
