#!/usr/bin/env python3
"""阶段 3/4：连续批处理 + 分块预填充的生成与压测。

生成模式输出与阶段 0/1/2 相同的 JSON/NPZ，可直接用 ``tools/compare.py`` 对比::

    python scripts/generate_continuous.py --model Qwen/Qwen3-0.6B
    python tools/compare.py --baseline out/golden/Qwen__Qwen3-0.6B/golden.json \
                            --actual   out/continuous/Qwen__Qwen3-0.6B/golden.json

压测模式测量并发吞吐、TTFT / ITL 与峰值显存，并写入实验日志。阶段 4 用
``--max-prefill-length`` 控制 chunk 大小，``--sched-policy interleave`` 让 decode
插在 chunk 之间::

    python scripts/generate_continuous.py --model Qwen/Qwen3-0.6B \
        --max-prefill-length 2048 --sched-policy interleave \
        --bench-long-input-len 32768 --bench-num-requests 9 --bench-input-len 64 \
        --append-log --log-stage 4
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "python")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import transformers
from minisgl.core import SamplingParams
from minisgl.llm import LLM

from tools.logbook import git_commit, gpu_env, insert_log_entry
from tools.memory import peak_memory_gb, reset_peak_memory
from tools.workloads import PROMPTS


def model_slug(model_name: str) -> str:
    return model_name.rstrip("/").replace("/", "__")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="model id or local path")
    parser.add_argument(
        "--model-source", choices=("modelscope", "huggingface"), default="modelscope"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--num-pages", type=int, default=0, help="0 = auto from free memory")
    parser.add_argument("--max-running-req", type=int, default=16)
    parser.add_argument(
        "--max-extend-tokens",
        "--max-prefill-length",
        dest="max_extend_tokens",
        type=int,
        default=8192,
        help="每次 iteration 的 prefill token 预算（chunk 大小）",
    )
    parser.add_argument(
        "--sched-policy",
        choices=("prefill_first", "decode_first", "interleave"),
        default="prefill_first",
        help="prefill / decode 均有请求时的调度策略",
    )
    parser.add_argument("--max-seq-len", type=int, default=0, help="0 = model max position")
    parser.add_argument("--memory-ratio", type=float, default=0.9)
    parser.add_argument("--cache-type", default="naive", choices=("naive", "radix"))
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1, help="计时前的预热请求次数")
    parser.add_argument("--bench-num-requests", type=int, default=0)
    parser.add_argument("--bench-input-len", type=int, default=256)
    parser.add_argument(
        "--bench-long-input-len",
        type=int,
        default=0,
        help=">0 时第一条请求用该输入长度，其余用 --bench-input-len（测长 prompt + 短请求并发）",
    )
    parser.add_argument(
        "--bench-shared-prefix",
        type=int,
        default=0,
        help=">0 时所有 bench 请求共享一段该长度的前缀（测前缀缓存命中）",
    )
    parser.add_argument("--bench-output-len", type=int, default=32)
    parser.add_argument("--bench-mixed", action="store_true", help="输入长度在 1/4..全长度间随机")
    parser.add_argument("--bench-serial", action="store_true", help="额外测逐条串行吞吐")
    parser.add_argument("--append-log", action="store_true")
    parser.add_argument("--log-path", type=Path, default=Path("experiments/log.md"))
    parser.add_argument("--log-stage", type=int, default=3)
    parser.add_argument("--log-note", default="待填")
    args = parser.parse_args(argv)
    if args.page_size < 1:
        parser.error("--page-size must be >= 1")
    if (
        args.max_tokens < 0
        or args.bench_output_len < 0
        or args.bench_long_input_len < 0
        or args.bench_shared_prefix < 0
    ):
        parser.error("token counts must be >= 0")
    if args.max_running_req < 1:
        parser.error("--max-running-req must be >= 1")
    if args.max_extend_tokens < 1:
        parser.error("--max-prefill-length must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    return args


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _meta(args: argparse.Namespace, llm: LLM) -> dict[str, Any]:
    return {
        "model": args.model,
        "model_source": args.model_source,
        "dtype": args.dtype,
        "device": args.device,
        "page_size": args.page_size,
        "num_pages": llm.engine.num_pages,
        "max_running_req": llm.engine.max_running_req,
        "max_extend_tokens": llm.scheduler.prefill_manager.max_extend_tokens,
        "sched_policy": llm.scheduler.sched_policy,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(torch.device(args.device))
            if args.device.startswith("cuda")
            else None
        ),
    }


def _log_header(args: argparse.Namespace, meta: dict[str, Any], title: str) -> list[str]:
    return [
        f"### {datetime.date.today().isoformat()} — {title} （阶段 {args.log_stage}）",
        "",
        f"- commit: {git_commit()}",
        f"- GPU / 驱动 / torch / transformers: {gpu_env(meta, args.device)}",
        f"- 模型 / dtype: `{meta['model']}` / {meta['dtype']}（{meta['model_source']}）",
        f"- KV cache / 调度: page_size={meta['page_size']}, num_pages={meta['num_pages']}, "
        f"max_running_req={meta['max_running_req']}, "
        f"max_extend_tokens={meta['max_extend_tokens']}, "
        f"sched_policy={meta['sched_policy']}, cache={args.cache_type}",
        f"- warmup: {args.warmup} 次（不计时）",
    ]


def _warmup(llm: LLM, args: argparse.Namespace, token_id: int) -> None:
    if args.warmup <= 0:
        return
    length = min(32, max(1, llm.engine.max_seq_len - 1))
    for _ in range(args.warmup):
        llm.generate(
            [[token_id] * length],
            [SamplingParams(max_tokens=1, ignore_eos=True)],
        )


def run_generation(args: argparse.Namespace, llm: LLM, meta: dict[str, Any]) -> str:
    tokenizer = llm.tokenizer
    assert tokenizer is not None
    eos_token_id = tokenizer.eos_token_id
    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    input_id_lists = [
        tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts
    ]
    if any(not ids for ids in input_id_lists):
        raise SystemExit("tokenizer produced an empty sequence")

    _warmup(llm, args, min(1000, llm.engine.model_config.vocab_size - 1))
    reset_peak_memory(llm.engine.device)

    params = SamplingParams(max_tokens=args.max_tokens, ignore_eos=args.ignore_eos)
    start = time.perf_counter()
    results = llm.generate(input_id_lists, [params] * len(input_id_lists))
    _sync(llm.engine.device)
    wall_s = time.perf_counter() - start
    llm.scheduler.check_integrity()

    out_dir = args.output or Path("out") / "continuous" / model_slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    logits_by_id: dict[str, np.ndarray] = {}
    ttfts, itls = [], []
    for index, (prompt, ids, result) in enumerate(zip(prompts, input_id_lists, results)):
        case_id = f"p{index:02d}"
        if result["prefill_logits"] is not None:
            logits_by_id[case_id] = (
                result["prefill_logits"].float().cpu().numpy().astype(np.float32)
            )
        if result["ttft_ms"] is not None:
            ttfts.append(result["ttft_ms"])
        itls.extend(result["itl_ms"])
        cases.append(
            {
                "id": case_id,
                "prompt": prompt,
                "input_ids": list(ids),
                "output_ids": result["token_ids"],
                "stop_reason": result["stop_reason"],
                "timing": {"ttft_ms": result["ttft_ms"], "itl_ms": result["itl_ms"]},
            }
        )

    payload = {
        "kind": "continuous",
        "meta": meta,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sampling": {
            "max_tokens": args.max_tokens,
            "ignore_eos": args.ignore_eos,
            "eos_token_id": eos_token_id,
            "add_special_tokens": False,
            "seed": args.seed,
        },
        "prompt_source": "tools/workloads.py:PROMPTS",
        "logits_file": "logits.npz",
        "cases": cases,
    }
    (out_dir / "golden.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(out_dir / "logits.npz", **logits_by_id)

    peak = peak_memory_gb(llm.engine.device)
    stats = {
        "num_requests": len(cases),
        "wall_s": wall_s,
        "ttft_mean_ms": _mean(ttfts),
        "itl_mean_ms": _mean(itls),
        "peak_memory_gb": peak,
    }
    print(
        f"generated {len(cases)} requests in {wall_s:.2f}s, "
        f"TTFT mean={_fmt(stats['ttft_mean_ms'])}ms, "
        f"ITL mean={_fmt(stats['itl_mean_ms'])}ms, peak={_fmt(peak)}GB"
    )
    return format_generate_entry(stats, args=args, meta=meta)


def _make_bench_requests(
    args: argparse.Namespace, llm: LLM
) -> tuple[list[list[int]], list[SamplingParams]]:
    token_id = min(1000, llm.engine.model_config.vocab_size - 1)
    tail_id = max(1, token_id - 1)
    rng = random.Random(args.seed)
    low = max(1, args.bench_input_len // 4)
    shared = [token_id] * args.bench_shared_prefix
    prompts: list[list[int]] = []
    if args.bench_long_input_len > 0:
        # 第一条请求承担长 prompt，其余保持短输入，用来观察 chunk 与 decode 交错
        prompts.append(shared + [token_id] * args.bench_long_input_len)
    for _ in range(args.bench_num_requests - len(prompts)):
        length = (
            rng.randint(low, args.bench_input_len) if args.bench_mixed else args.bench_input_len
        )
        prompts.append(shared + [tail_id] * length)
    params = [SamplingParams(max_tokens=args.bench_output_len, ignore_eos=True) for _ in prompts]
    return prompts, params


def _summarize(results: list[dict[str, Any]], wall_s: float, device: torch.device) -> dict:
    ttfts = [result["ttft_ms"] for result in results if result["ttft_ms"] is not None]
    itls = [value for result in results for value in result["itl_ms"]]
    total_tokens = sum(len(result["token_ids"]) for result in results)
    return {
        "wall_s": wall_s,
        "total_tokens": total_tokens,
        "ttft_p50": _percentile(ttfts, 50),
        "ttft_p95": _percentile(ttfts, 95),
        "itl_p50": _percentile(itls, 50),
        "itl_p99": _percentile(itls, 99),
        "throughput": total_tokens / wall_s if wall_s > 0 else None,
        "peak_memory_gb": peak_memory_gb(device),
    }


def run_bench(args: argparse.Namespace, llm: LLM, meta: dict[str, Any]) -> str:
    prompts, params = _make_bench_requests(args, llm)
    _warmup(llm, args, min(1000, llm.engine.model_config.vocab_size - 1))
    reset_peak_memory(llm.engine.device)

    start = time.perf_counter()
    results = llm.generate(prompts, params)
    _sync(llm.engine.device)
    wall_s = time.perf_counter() - start
    llm.scheduler.check_integrity()
    stats = _summarize(results, wall_s, llm.engine.device)

    serial = None
    if args.bench_serial:
        reset_peak_memory(llm.engine.device)
        start = time.perf_counter()
        serial_results: list[dict[str, Any]] = []
        for prompt, param in zip(prompts, params):
            serial_results.extend(llm.generate([prompt], [param]))
        _sync(llm.engine.device)
        serial_wall = time.perf_counter() - start
        serial = _summarize(serial_results, serial_wall, llm.engine.device)

    print(
        f"concurrent: {len(prompts)} reqs, wall={wall_s:.2f}s, "
        f"TTFT p50/p95={_fmt(stats['ttft_p50'])}/{_fmt(stats['ttft_p95'])}ms, "
        f"ITL p50/p99={_fmt(stats['itl_p50'])}/{_fmt(stats['itl_p99'])}ms, "
        f"throughput={_fmt(stats['throughput'])} tok/s, peak={_fmt(stats['peak_memory_gb'])}GB"
    )
    if serial is not None:
        print(
            f"serial: wall={serial['wall_s']:.2f}s, "
            f"throughput={_fmt(serial['throughput'])} tok/s"
        )
    return format_bench_entry(stats, serial=serial, args=args, meta=meta)


def format_generate_entry(stats: dict[str, Any], *, args: argparse.Namespace, meta: dict) -> str:
    lines = _log_header(args, meta, "连续批处理生成")
    lines += [
        f"- 配置: max_tokens={args.max_tokens}, ignore_eos={args.ignore_eos}, "
        f"{stats['num_requests']} 条固定 prompt",
        f"- 结果: wall = {stats['wall_s']:.2f} s；TTFT 均值 = {_fmt(stats['ttft_mean_ms'])} ms；"
        f"ITL 均值 = {_fmt(stats['itl_mean_ms'])} ms；"
        f"峰值显存 = {_fmt(stats['peak_memory_gb'])} GB",
        "",
        f"- 结论 / 下一步: {args.log_note}",
        "",
    ]
    return "\n".join(lines)


def format_bench_entry(
    stats: dict[str, Any],
    *,
    serial: dict[str, Any] | None,
    args: argparse.Namespace,
    meta: dict,
) -> str:
    lines = _log_header(args, meta, "连续批处理并发压测")
    shared_note = f"共享前缀 {args.bench_shared_prefix} + " if args.bench_shared_prefix else ""
    long_note = f"首条 {args.bench_long_input_len} + " if args.bench_long_input_len else ""
    load = (
        f"{args.bench_num_requests} 条请求，输入长度 {shared_note}{long_note}"
        f"{'1/4..' if args.bench_mixed else ''}{args.bench_input_len}，"
        f"输出 {args.bench_output_len}"
    )
    lines += [
        f"- 配置: {load}；greedy，ignore_eos=True",
        "",
        "| 模式 | 请求数 | wall (s) | TTFT p50 (ms) | TTFT p95 (ms) "
        "| ITL p50 (ms) | ITL p99 (ms) | tok/s | 峰值显存 (GB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| 并发 | {args.bench_num_requests} | {stats['wall_s']:.2f} "
        f"| {_fmt(stats['ttft_p50'])} | {_fmt(stats['ttft_p95'])} "
        f"| {_fmt(stats['itl_p50'])} | {_fmt(stats['itl_p99'])} "
        f"| {_fmt(stats['throughput'])} | {_fmt(stats['peak_memory_gb'])} |",
    ]
    if serial is not None:
        lines.append(
            f"| 串行 | {args.bench_num_requests} | {serial['wall_s']:.2f} "
            f"| {_fmt(serial['ttft_p50'])} | {_fmt(serial['ttft_p95'])} "
            f"| {_fmt(serial['itl_p50'])} | {_fmt(serial['itl_p99'])} "
            f"| {_fmt(serial['throughput'])} | {_fmt(serial['peak_memory_gb'])} |"
        )
    if serial is not None and serial["throughput"] and stats["throughput"]:
        lines.append(f"- 相对串行加速: {stats['throughput'] / serial['throughput']:.2f}×")
    lines += ["", f"- 结论 / 下一步: {args.log_note}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run this script on the GPU host")

    bench = args.bench_num_requests > 0
    llm = LLM(
        args.model,
        model_source=args.model_source,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        page_size=args.page_size,
        max_running_req=args.max_running_req,
        num_pages=args.num_pages or None,
        max_seq_len=args.max_seq_len or None,
        max_extend_tokens=args.max_extend_tokens,
        sched_policy=args.sched_policy,
        memory_ratio=args.memory_ratio,
        cache_type=args.cache_type,
        record_prefill_logits=not bench,
    )
    meta = _meta(args, llm)

    if bench:
        entry = run_bench(args, llm, meta)
    else:
        entry = run_generation(args, llm, meta)

    if args.append_log:
        insert_log_entry(args.log_path, args.log_stage, entry)
        print(f"\nappended log entry -> {args.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
