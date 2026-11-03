"""NVTX 打点 + 分阶段计时。

设计原则：
- **默认关闭时零开销**：只判断一个模块级 bool，不建 range、不调 perf_counter。
- 打开方式：`bench.py --profile`，或环境变量 `IFTZ_NVTX=1`（`IFTZ_NVTX_DEEP=1` 打开逐层打点）。
- 关闭时不依赖 CUDA：在 CPU/无 GPU 环境也能 import。

两个入口：
- `prof.phase("name")`：既推 NVTX range（在 Nsight 时间线上能看到），又累计该阶段的
  CPU 墙钟耗时，`prof.report()` 打印汇总表。
- `prof.nvtx("name", deep=True)`：只推 NVTX range，用于高频位置（每层 attention 之类），
  必须配合 `deep=True` 才会生效，避免默认把 trace 撑爆。
"""

from __future__ import annotations

import contextlib
import functools
import os
from time import perf_counter

import torch

_disabled_env = ("", "0", "false", "False")
_started = os.environ.get("IFTZ_NVTX", "0") not in _disabled_env
_deep = os.environ.get("IFTZ_NVTX_DEEP", "0") not in _disabled_env

# CPU-only 的 torch 编译时 torch.cuda.nvtx 会直接 raise（"NVTX functions not installed"），
# 所以先探一次：能推 range 才推，否则只做 CPU 侧计时。
_nvtx_ok = False

# name -> [calls, 累计 CPU 秒]
_stats: dict[str, list] = {}


def _probe_nvtx() -> bool:
    global _nvtx_ok
    try:
        torch.cuda.nvtx.range_push("iftz/probe")
        torch.cuda.nvtx.range_pop()
        _nvtx_ok = True
    except Exception:  # noqa: BLE001 - CPU-only torch / 旧版本没有 NVTX
        _nvtx_ok = False
    return _nvtx_ok


if _started:
    _probe_nvtx()


def enable(deep: bool = False) -> None:
    """在进程内打开打点（bench.py --profile 会调用）。"""
    global _started, _deep
    _started = True
    if deep:
        _deep = True
    if _started and not _nvtx_ok:
        _probe_nvtx()
        if not _nvtx_ok:
            print("[profiler] 当前 torch 不支持 NVTX（无 CUDA build），只统计 CPU 阶段耗时")


def nvtx_available() -> bool:
    return _nvtx_ok


def disable() -> None:
    global _started, _deep
    _started = False
    _deep = False


def enabled() -> bool:
    return _started


def deep_enabled() -> bool:
    return _started and _deep


@contextlib.contextmanager
def nvtx(name: str, deep: bool = False):
    """只推 NVTX range，不统计 CPU 时间。"""
    if not _started or not _nvtx_ok or (deep and not _deep):
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@contextlib.contextmanager
def phase(name: str, deep: bool = False):
    """NVTX range + CPU 墙钟累计，结束时由 report() 汇总。"""
    if not _started or (deep and not _deep):
        yield
        return
    push = _nvtx_ok
    if push:
        torch.cuda.nvtx.range_push(name)
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        if push:
            torch.cuda.nvtx.range_pop()
        slot = _stats.get(name)
        if slot is None:
            _stats[name] = [1, dt]
        else:
            slot[0] += 1
            slot[1] += dt


def nvtx_fn(name: str | None = None, deep: bool = False):
    """给函数套一层 range，调试时临时加在任意函数上。"""

    def decorator(fn):
        label = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with nvtx(label, deep=deep):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def under_nsys() -> bool:
    """当前进程是不是被 `nsys profile` 启动的（靠 nsys 注入的 LD_PRELOAD 判断）。

    用途：`--profile` 只是"打开打点"，不负责录制；没有 nsys 父进程时
    NVTX range 和 cudaProfilerStart 都是空转，不会产生 .nsys-rep。
    这个判断用来在跑完时给出准确提示，而不是让人以为已经有 trace 了。
    """
    preload = os.environ.get("LD_PRELOAD", "").lower()
    if "nsys" in preload:
        return True
    return any(k.startswith("NSYS_") or k.startswith("NSIGHT_") for k in os.environ)


def start_capture() -> None:
    """对齐 nsys 的 --capture-range=cudaProfilerApi。

    先 sync，保证 start 之前的工作不落进 trace；没装 nsys 时这行是 no-op。
    """
    if not _started or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.profiler.start()


def stop_capture() -> None:
    if not _started or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def report(title: str = "phase timing", sort_by: str = "total") -> str:
    """打印分阶段 CPU 耗时汇总（降序）。返回文本，方便写进日志。"""
    if not _stats:
        return ""

    key = (lambda kv: kv[1][1]) if sort_by == "total" else (lambda kv: kv[1][1] / kv[1][0])
    rows = sorted(_stats.items(), key=key, reverse=True)
    total = sum(v[1] for v in _stats.values())
    width = max(len(name) for name in _stats)

    lines = [
        f"===== {title} =====",
        f"{'phase':<{width}} {'calls':>8} {'cpu_s':>9} {'ms/call':>9} {'share':>7}",
    ]
    for name, (calls, sec) in rows:
        lines.append(
            f"{name:<{width}} {calls:>8d} {sec:>9.3f} {sec / calls * 1e3:>9.2f} {sec / total * 100:>6.1f}%"
        )
    lines.append(f"{'TOTAL':<{width}} {sum(v[0] for v in _stats.values()):>8d} {total:>9.3f}")
    text = "\n".join(lines)
    print(text)
    return text


def reset() -> None:
    _stats.clear()
