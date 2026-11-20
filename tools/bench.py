#!/usr/bin/env python3
"""在固定负载Load上测量 TTFT / ITL / 吞吐（路线图阶段 0）。

阶段 0 的 benchmark 是串行的（一次一个请求），作为 batch=1 基线。后续阶段会
增加 ``engine`` 后端，用同一组负载测量真实调度器的表现。

用户传入 list[Load]，遍历每个load，搭配runtime进行bench_load
bench_load便利load中每个seq进行bench_request
以一个load为单元返回指标

用法::


    python tools/bench.py --model Qwen/Qwen3-0.6B --load short_32
    python tools/bench.py --model Qwen/Qwen3-0.6B --load all
    python tools/bench.py --model Qwen/Qwen3-0.6B --load all --append-log --log-stage 2
    （--log-stage 支持 0–12 任一阶段，记录插入 log.md 对应小节）
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.hf_runtime import GreedyDecoder, HFRuntime, load_hf_runtime
from tools.workloads import LOADS, Load, get_load, make_input_ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--model-source", choices=("huggingface", "modelscope"), default="modelscope"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument(
        "--backend",
        default="hf",
        choices=("hf",),
        help="hf = sequential HF baseline; engine backend is added in stage 3",
    )
    parser.add_argument(
        "--load", default="short_32", help="load name, comma-separated list, or 'all'"
    )
    parser.add_argument("--list", action="store_true", help="list available loads and exit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1, help="number of small warmup requests")
    parser.add_argument("--output-dir", type=Path, default=Path("out/bench"))
    parser.add_argument(
        "--append-log",
        action="store_true",
        help="把本轮结果追加到实验日志（默认 experiments/log.md）",
    )
    parser.add_argument("--log-path", type=Path, default=Path("experiments/log.md"))
    parser.add_argument(
        "--log-stage",
        type=int,
        default=None,
        help="记录所属阶段编号（0–12），必须与 --append-log 一起使用",
    )
    parser.add_argument("--log-note", default="待填", help="记录中的“结论 / 下一步”")
    return parser.parse_args(argv)


def _sync(runtime: HFRuntime) -> None:
    if runtime.device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def bench_request(runtime: HFRuntime, input_ids: list[int], max_tokens: int) -> dict[str, Any]:
    """对一个使用 KV cache 的 greedy 请求计时；benchmark 忽略 EOS。"""
    decoder = GreedyDecoder(runtime, input_ids, max_tokens, ignore_eos=True)

    _sync(runtime)
    start = time.perf_counter()
    decoder.prefill()
    token, reason = decoder.next_token()
    _sync(runtime)

    ttft_ms = (time.perf_counter() - start) * 1e3  # prefill统计ttft

    itl_ms: list[float] = []  # token间时延，衡量decode性能
    output_tokens = 0
    while token is not None:
        output_tokens += 1
        step_start = time.perf_counter()
        token, reason = decoder.next_token()
        _sync(runtime)
        if token is not None:
            itl_ms.append((time.perf_counter() - step_start) * 1e3)

    return {
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
        "output_tokens": output_tokens,
        "stop_reason": reason,
    }


# 装载runtime和一个load，遍历load中的seq进行bench_request
def bench_load(runtime: HFRuntime, load: Load, seed: int = 0, warmup: int = 1) -> dict[str, Any]:
    import torch

    inputs = make_input_ids(load, seed=seed)

    # 预热
    if warmup > 0:
        warm_ids = make_input_ids(Load("warmup", (32,), (2,)), seed=seed)[0]
        for _ in range(warmup):
            bench_request(runtime, warm_ids, 2)
        _sync(runtime)

    if runtime.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    wall_start = time.perf_counter()
    results = [
        bench_request(runtime, input_ids, max_tokens)
        for input_ids, max_tokens in zip(inputs, load.output_lens)
    ]
    _sync(runtime)
    wall_s = time.perf_counter() - wall_start

    peak_gb = None
    if runtime.device.startswith("cuda"):
        peak_gb = torch.cuda.max_memory_allocated() / 1e9

    ttfts = [result["ttft_ms"] for result in results]
    itls = [ms for result in results for ms in result["itl_ms"]]
    total_tokens = sum(result["output_tokens"] for result in results)

    return {
        "load": dataclasses.asdict(load),
        "requests": len(load.input_lens),
        "total_output_tokens": total_tokens,
        "wall_s": wall_s,
        "ttft_ms": {"p50": _percentile(ttfts, 50), "p95": _percentile(ttfts, 95)},
        "itl_ms": {
            "p50": _percentile(itls, 50),
            "p95": _percentile(itls, 95),
            "p99": _percentile(itls, 99),
        },
        "throughput_output_tokens_per_s": total_tokens / wall_s if wall_s > 0 else None,
        "peak_memory_gb": peak_gb,
        "per_request": results,
    }


def _fmt(value: float | None, unit: str = "") -> str:
    return "-" if value is None else f"{value:.2f}{unit}"


def _print_summary(summary: dict[str, Any]) -> None:
    ttft = summary["ttft_ms"]
    itl = summary["itl_ms"]
    print(
        f"\n[{summary['load']['name']}] requests={summary['requests']} "
        f"tokens={summary['total_output_tokens']} wall={summary['wall_s']:.2f}s"
    )
    print(
        f"  TTFT  p50={_fmt(ttft['p50'], 'ms')} p95={_fmt(ttft['p95'], 'ms')} | "
        f"ITL p50={_fmt(itl['p50'], 'ms')} p95={_fmt(itl['p95'], 'ms')} "
        f"p99={_fmt(itl['p99'], 'ms')}"
    )
    print(
        f"  throughput={_fmt(summary['throughput_output_tokens_per_s'], ' tok/s')} "
        f"peak_mem={_fmt(summary['peak_memory_gb'], ' GB')}"
    )


def _run(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def _git_commit() -> str:
    sha = _run(["git", "rev-parse", "--short", "HEAD"])
    if sha is None:
        return "unknown"
    return f"{sha} (dirty)" if _run(["git", "status", "--porcelain"]) else sha


def _gpu_summary(runtime: HFRuntime) -> str:
    """组装「GPU / 驱动 / torch / transformers」单行环境信息。"""
    meta = runtime.meta
    if runtime.device.startswith("cuda"):
        name = meta.get("cuda_device") or runtime.device
        driver_raw = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        driver = driver_raw.splitlines()[0].strip() if driver_raw else "?"
        gpu = f"{name} / driver {driver}"
    else:
        gpu = f"device {runtime.device}（非 CUDA，仅 CPU 参考）"
    return f"{gpu} / torch {meta['torch_version']} / transformers {meta['transformers_version']}"


def _length_span(lengths: list[int] | tuple[int, ...]) -> str:
    values = list(lengths)
    lo, hi = min(values), max(values)
    return str(lo) if lo == hi else f"{lo}–{hi}"


def format_log_entry(
    summaries: list[dict[str, Any]],
    *,
    stage: int,
    meta: dict[str, Any],
    config: dict[str, Any],
    env: str,
    commit: str,
    date: str | None = None,
    note: str = "待填",
) -> str:
    """按 experiments/log.md 的格式生成一条 Markdown 记录。"""
    date = date or datetime.date.today().isoformat()
    names = ", ".join(summary["load"]["name"] for summary in summaries)
    lines = [
        f"### {date} — bench: {names} （阶段 {stage}）",
        "",
        f"- commit: {commit}",
        f"- GPU / 驱动 / torch / transformers: {env}",
        f"- 模型 / dtype: `{meta['model']}` / {meta['dtype']}（{config['model_source']}）",
        f"- 配置: backend={config['backend']}, seed={config['seed']}, warmup={config['warmup']}",
        f"- 负载: {names}",
        "",
        "| 负载 | 请求数 | 输入长度 | 输出长度 | TTFT p50 (ms) | ITL p50 (ms) "
        "| ITL p95 (ms) | ITL p99 (ms) | tok/s | 峰值显存 (GB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        load = summary["load"]
        lines.append(
            f"| {load['name']} | {summary['requests']} "
            f"| {_length_span(load['input_lens'])} | {_length_span(load['output_lens'])} "
            f"| {_fmt(summary['ttft_ms']['p50'])} | {_fmt(summary['itl_ms']['p50'])} "
            f"| {_fmt(summary['itl_ms']['p95'])} | {_fmt(summary['itl_ms']['p99'])} "
            f"| {_fmt(summary['throughput_output_tokens_per_s'])} "
            f"| {_fmt(summary['peak_memory_gb'])} |"
        )
    lines += ["", f"- 结论 / 下一步: {note}", ""]
    return "\n".join(lines)


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        for name, load in LOADS.items():
            print(f"{name:18s} requests={load.num_requests:<3d} {load.description}")
        return 0

    if args.append_log and args.log_stage is None:
        raise SystemExit("--append-log requires --log-stage N")

    load_names = list(LOADS) if args.load == "all" else [n.strip() for n in args.load.split(",")]
    loads = [get_load(name) for name in load_names]

    runtime = load_hf_runtime(
        args.model,
        dtype=args.dtype,
        device=args.device,
        model_source=args.model_source,
    )
    print(f"backend=hf model={args.model} dtype={args.dtype} device={args.device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for load in loads:
        summary = bench_load(runtime, load, seed=args.seed, warmup=args.warmup)
        summaries.append(summary)
        payload = {
            "kind": "bench",
            "backend": args.backend,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "meta": runtime.meta,
            "config": {"seed": args.seed, "warmup": args.warmup},
            "summary": {key: value for key, value in summary.items() if key != "per_request"},
            "per_request": summary["per_request"],
        }
        out_path = args.output_dir / f"{load.name}_{args.backend}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _print_summary(summary)
        print(f"  wrote {out_path}")

    if args.append_log:
        entry = format_log_entry(
            summaries,
            stage=args.log_stage,
            meta=runtime.meta,
            config={
                "model_source": args.model_source,
                "backend": args.backend,
                "seed": args.seed,
                "warmup": args.warmup,
            },
            env=_gpu_summary(runtime),
            commit=_git_commit(),
            note=args.log_note,
        )
        insert_log_entry(args.log_path, args.log_stage, entry)
        print(f"\nappended log entry -> {args.log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
