"""generate_naive.py 日志记录格式的 CPU 单测。"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from scripts.generate_naive import format_bench_entry, format_generate_entry  # noqa: E402

META: dict[str, Any] = {
    "model": "Qwen/Qwen3-0.6B",
    "model_source": "modelscope",
    "dtype": "bfloat16",
    "torch_version": "2.4.0",
    "transformers_version": "4.44.2",
    "cuda_device": None,
}


def _args(**overrides: Any) -> argparse.Namespace:
    values = {
        "log_stage": 1,
        "log_note": "note",
        "device": "cpu",
        "max_tokens": 32,
        "ignore_eos": False,
        "seed": 0,
        "bench_repeats": 3,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_format_bench_entry_matches_log_table() -> None:
    results = [
        {
            "prompt_len": 128,
            "prefill_ms": 12.3,
            "decode_ms": [5.6, 5.7],
            "decode_mean_ms": 5.65,
        },
        {
            "prompt_len": 2048,
            "prefill_ms": 100.0,
            "decode_ms": [50.0],
            "decode_mean_ms": 50.0,
        },
    ]
    entry = format_bench_entry(results, args=_args(), meta=META)

    assert "naive decode 长度扩展 （阶段 1）" in entry
    assert "`Qwen/Qwen3-0.6B` / bfloat16（modelscope）" in entry
    assert "bench_repeats=3" in entry
    assert "| prompt 长度 | prefill 耗时 (ms) | 单步 decode（长度 L+1）(ms) |" in entry
    assert "| 128 | 12.30 | 5.65 |" in entry
    assert "| 2048 | 100.00 | 50.00 |" in entry
    assert entry.rstrip().endswith("- 结论 / 下一步: note")


def test_format_generate_entry_reports_means() -> None:
    stats = {
        "num_prompts": 20,
        "prefill_mean_ms": 8.5,
        "decode_mean_ms": 4.25,
        "peak_memory_gb": 1.5,
    }
    entry = format_generate_entry(stats, args=_args(ignore_eos=True), meta=META)

    assert "naive 生成（无 KV Cache） （阶段 1）" in entry
    assert "max_tokens=32, ignore_eos=True, seed=0, 20 条固定 prompt" in entry
    assert "prefill 均值 = 8.50 ms" in entry
    assert "单步 decode 均值 = 4.25 ms" in entry
    assert "峰值显存 = 1.50 GB" in entry


def test_missing_peak_memory_is_dash() -> None:
    stats = {
        "num_prompts": 1,
        "prefill_mean_ms": None,
        "decode_mean_ms": None,
        "peak_memory_gb": None,
    }
    entry = format_generate_entry(stats, args=_args(), meta=META)
    assert "prefill 均值 = - ms" in entry
    assert "峰值显存 = - GB" in entry
