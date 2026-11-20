"""bench.py 实验日志追加功能的纯 CPU 单测。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.bench import _length_span, format_log_entry, insert_log_entry

SUMMARY: dict[str, Any] = {
    "load": {
        "name": "short_32",
        "input_lens": (32,),
        "output_lens": (32,),
        "shared_prefix": 0,
        "description": "single short request",
    },
    "requests": 1,
    "total_output_tokens": 32,
    "wall_s": 1.0,
    "ttft_ms": {"p50": 12.0, "p95": 15.0},
    "itl_ms": {"p50": 1.5, "p95": 2.0, "p99": 2.5},
    "throughput_output_tokens_per_s": 100.0,
    "peak_memory_gb": 1.2,
    "per_request": [],
}


def _entry(stage: int = 3, name: str = "test") -> str:
    return format_log_entry(
        [SUMMARY],
        stage=stage,
        meta={"model": "Qwen/Qwen3-0.6B", "dtype": "bfloat16"},
        config={"model_source": "modelscope", "backend": "hf", "seed": 0, "warmup": 1},
        env="NVIDIA H100 / driver 535.104.05 / torch 2.4.0 / transformers 4.44.2",
        commit="abc1234",
        date="2026-11-20",
        note=name,
    )


def test_length_span() -> None:
    assert _length_span((32,)) == "32"
    assert _length_span([32, 4096, 128]) == "32–4096"


def test_format_log_entry_contains_required_fields() -> None:
    entry = _entry()
    assert "### 2026-11-20 — bench: short_32 （阶段 3）" in entry
    assert "commit: abc1234" in entry
    assert "`Qwen/Qwen3-0.6B` / bfloat16（modelscope）" in entry
    assert "backend=hf, seed=0, warmup=1" in entry
    assert "| short_32 | 1 | 32 | 32 | 12.00 | 1.50 | 2.00 | 2.50 | 100.00 | 1.20 |" in entry
    assert "- 结论 / 下一步: test" in entry


def _write_log(path: Path) -> None:
    path.write_text(
        "# 实验日志\n\n"
        "## 阶段 2：KV Cache 与批量前向\n\n"
        "（待开始）\n\n"
        "## 阶段 3：连续批处理调度器\n\n"
        "（待开始）\n",
        encoding="utf-8",
    )


def test_insert_replaces_placeholder_and_keeps_order(tmp_path: Path) -> None:
    log = tmp_path / "log.md"
    _write_log(log)

    insert_log_entry(log, 3, _entry(stage=3, name="first"))

    text = log.read_text(encoding="utf-8")
    assert "## 阶段 3：连续批处理调度器\n\n### 2026-11-20" in text
    assert text.index("阶段 2") < text.index("阶段 3") < text.index("### 2026-11-20")
    # 阶段 2 的占位符不受影响
    assert text.count("（待开始）") == 1
    assert "## 阶段 2：KV Cache 与批量前向\n\n（待开始）" in text


def test_insert_appends_multiple_entries_in_stage(tmp_path: Path) -> None:
    log = tmp_path / "log.md"
    _write_log(log)

    insert_log_entry(log, 2, _entry(stage=2, name="first"))
    insert_log_entry(log, 2, _entry(stage=2, name="second"))

    text = log.read_text(encoding="utf-8")
    assert text.index("结论 / 下一步: first") < text.index("结论 / 下一步: second")
    assert text.index("结论 / 下一步: second") < text.index("## 阶段 3")


def test_stage_heading_does_not_confuse_prefix(tmp_path: Path) -> None:
    log = tmp_path / "log.md"
    log.write_text("# 日志\n\n## 阶段 10：张量并行\n\n（待开始）\n", encoding="utf-8")

    insert_log_entry(log, 1, _entry(stage=1, name="stage1"))

    text = log.read_text(encoding="utf-8")
    # 阶段 1 小节不存在，应在文末新建，不能写进阶段 10
    assert text.index("## 阶段 10") < text.index("## 阶段 1\n")
    assert "阶段 10：张量并行\n\n（待开始）\n\n## 阶段 1" in text


def test_insert_creates_missing_stage_section(tmp_path: Path) -> None:
    log = tmp_path / "log.md"
    _write_log(log)

    insert_log_entry(log, 7, _entry(stage=7, name="new"))

    text = log.read_text(encoding="utf-8")
    assert text.endswith("## 阶段 7\n\n" + _entry(stage=7, name="new").rstrip() + "\n")
