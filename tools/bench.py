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
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import time
from pathlib import Path
from typing import Any

import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.hf_runtime import GreedyDecoder, HFRuntime, load_hf_runtime
from tools.workloads import LOADS, Load, get_load, make_input_ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-source", choices=("huggingface", "modelscope"), default="modelscope")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument(
        "--backend",
        default="hf",
        choices=("hf",),
        help="hf = sequential HF baseline; engine backend is added in stage 3",
    )
    parser.add_argument("--load", default="short_32", help="load name, comma-separated list, or 'all'")
    parser.add_argument("--list", action="store_true", help="list available loads and exit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1, help="number of small warmup requests")
    parser.add_argument("--output-dir", type=Path, default=Path("out/bench"))
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

    ttft_ms = (time.perf_counter() - start) * 1e3 #prefill统计ttft

    itl_ms: list[float] = [] # token间时延，衡量decode性能
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
def bench_load(
    runtime: HFRuntime, load: Load, seed: int = 0, warmup: int = 1
) -> dict[str, Any]:
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        for name, load in LOADS.items():
            print(f"{name:18s} requests={load.num_requests:<3d} {load.description}")
        return 0

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
    for load in loads:
        summary = bench_load(runtime, load, seed=args.seed, warmup=args.warmup)
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
