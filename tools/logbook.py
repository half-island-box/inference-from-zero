"""实验日志（experiments/log.md）的公共写入逻辑。

bench.py 与 scripts/generate_naive.py 共用：收集运行环境、生成记录片段，
并插入到对应阶段小节。仅依赖标准库，CPU 环境可直接导入。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def git_commit() -> str:
    """返回 short commit；工作区有改动时追加 ``(dirty)``。"""
    sha = _run(["git", "rev-parse", "--short", "HEAD"])
    if sha is None:
        return "unknown"
    return f"{sha} (dirty)" if _run(["git", "status", "--porcelain"]) else sha


def gpu_env(meta: dict[str, Any], device: str) -> str:
    """组装「GPU / 驱动 / torch / transformers」单行环境信息。"""
    if device.startswith("cuda"):
        name = meta.get("cuda_device") or device
        target = meta.get("device") or device
        driver_raw = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        driver = driver_raw.splitlines()[0].strip() if driver_raw else "?"
        gpu = f"{name} ({target}) / driver {driver}"
    else:
        gpu = f"device {device}（非 CUDA，仅 CPU 参考）"
    return f"{gpu} / torch {meta['torch_version']} / transformers {meta['transformers_version']}"


def _is_stage_heading(line: str, stage: int) -> bool:
    prefix = f"## 阶段 {stage}"
    if not line.startswith(prefix):
        return False
    rest = line[len(prefix) :]
    # 避免 "## 阶段 1" 误匹配 "## 阶段 10"
    return not rest or not rest[0].isdigit()


def insert_log_entry(path: Path, stage: int, entry: str) -> None:
    """把记录插到 ``## 阶段 N`` 小节末尾；小节不存在时在文末新建。"""
    text = path.read_text(encoding="utf-8") if path.exists() else "# 实验日志\n"
    lines = text.splitlines(keepends=True)

    start = next((i for i, line in enumerate(lines) if _is_stage_heading(line, stage)), None)
    if start is None:
        separator = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
        path.write_text(
            text + separator + f"## 阶段 {stage}\n\n{entry.rstrip()}\n", encoding="utf-8"
        )
        return

    end = next((j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")), len(lines))
    # 去掉占位符，避免出现「（待开始）」压在记录上面
    placeholders = {"（待开始）", "（待填）"}
    body = [line for line in lines[start + 1 : end] if line.strip() not in placeholders]
    section = "".join(body).strip("\n")
    merged = (f"{section}\n\n" if section else "") + entry.rstrip() + "\n\n"
    lines[start + 1 : end] = [f"\n{merged}"]
    path.write_text("".join(lines), encoding="utf-8")
