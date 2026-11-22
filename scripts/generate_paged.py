#!/usr/bin/env python3
"""阶段 2：带 paged KV Cache 的批量 greedy 生成。

与阶段 1 的 ``generate_naive.py`` 输出相同格式的 JSON/NPZ，可直接用
``tools/compare.py`` 对比::

    python scripts/generate_paged.py --model Qwen/Qwen3-0.6B --page-size 16
    python tools/compare.py --baseline out/golden/Qwen__Qwen3-0.6B/golden.json \
                            --actual   out/paged/Qwen__Qwen3-0.6B/golden.json

还可测量不同 batch size / 序列长度下的 prefill 与 decode 延迟::

    python scripts/generate_paged.py --model Qwen/Qwen3-0.6B \
        --bench-batch-sizes 1,2,4,8 --append-log --log-stage 2
    python scripts/generate_paged.py --model Qwen/Qwen3-0.6B \
        --bench-seq-lens 128,256,512,1024 --append-log --log-stage 2

每个配置计时前默认跑 1 次同形状预热（``--warmup``），避免首个 batch 的
CUDA kernel 加载 / allocator 初始化开销污染 prefill 数据。
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
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
from minisgl.models import (
    BaseLLMModel,
    ModelConfig,
    create_model,
    load_model_weights,
    load_weight,
    resolve_model_path,
)
from minisgl.scheduler import BatchRequest, PagedEngine
from transformers import AutoConfig, AutoTokenizer

from tools.logbook import git_commit, gpu_env, insert_log_entry
from tools.workloads import PROMPTS


def model_slug(model_name: str) -> str:
    return model_name.rstrip("/").replace("/", "__")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="model id or local path")
    parser.add_argument(
        "--model-source", choices=("modelscope", "huggingface"), default="modelscope"
    )
    parser.add_argument("--output", type=Path, default=None, help="output directory")
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--page-size", type=int, default=16, help="KV cache page size")
    parser.add_argument("--num-pages", type=int, default=0, help="total pages (0 = auto)")
    parser.add_argument("--max-running-req", type=int, default=8, help="request slots per batch")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="only use the first N prompts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bench-batch-sizes",
        default="",
        help="comma-separated batch sizes; measure prefill/decode latency for each",
    )
    parser.add_argument(
        "--bench-seq-lens",
        default="",
        help="comma-separated prompt lengths; measure with --bench-batch",
    )
    parser.add_argument(
        "--bench-batch", type=int, default=1, help="batch size for --bench-seq-lens"
    )
    parser.add_argument(
        "--bench-token-len", type=int, default=256, help="prompt length for batch bench"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="每个配置计时前的不计时预热次数（默认 1）",
    )
    parser.add_argument(
        "--append-log",
        action="store_true",
        help="把本轮结果追加到实验日志（默认 experiments/log.md）",
    )
    parser.add_argument("--log-path", type=Path, default=Path("experiments/log.md"))
    parser.add_argument("--log-stage", type=int, default=2, help="记录所属阶段编号（默认 2）")
    parser.add_argument("--log-note", default="待填", help="记录中的“结论 / 下一步”")
    args = parser.parse_args(argv)
    if args.page_size < 1:
        parser.error("--page-size must be >= 1")
    if args.max_tokens < 0:
        parser.error("--max-tokens must be >= 0")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.bench_batch < 1:
        parser.error("--bench-batch must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    return args


def load_model(
    args: argparse.Namespace,
) -> tuple[BaseLLMModel, ModelConfig, str, Any]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run this script on the GPU host")
    dtype = getattr(torch, args.dtype)
    path = resolve_model_path(args.model, args.model_source)
    hf_config = AutoConfig.from_pretrained(path)
    config = ModelConfig.from_hf(hf_config)
    # meta 构建 + assign 安装，避免在 GPU 上同时驻留两份权重
    model = create_model(config, device="meta", dtype=dtype, rope_device=args.device)
    load_model_weights(
        model,
        load_weight(
            path,
            device=args.device,
            dtype=dtype,
            tie_word_embeddings=config.tie_word_embeddings,
        ),
        dtype=dtype,
        assign=True,
    )
    return model, config, path, hf_config


def _auto_num_pages(args: argparse.Namespace, config: ModelConfig, needed_tokens: int) -> int:
    if args.num_pages > 0:
        return args.num_pages
    pages = math.ceil(needed_tokens / args.page_size) + 8
    if not args.device.startswith("cuda"):
        return max(pages, 256)
    free_bytes, _ = torch.cuda.mem_get_info()
    per_page = (
        2
        * config.num_layers
        * config.num_kv_heads
        * config.head_dim
        * args.page_size
        * getattr(torch, args.dtype).itemsize
    )
    max_pages = max(int(free_bytes * 0.4) // per_page, 1)
    return min(max(pages * 2, 512), max_pages)


def _make_engine(
    args: argparse.Namespace,
    model: BaseLLMModel,
    config: ModelConfig,
    num_pages: int,
    max_seq_len: int,
) -> PagedEngine:
    return PagedEngine(
        model,
        config,
        num_pages=num_pages,
        page_size=args.page_size,
        max_running_req=args.max_running_req,
        max_seq_len=max_seq_len,
        device=args.device,
        dtype=getattr(torch, args.dtype),
    )


def _meta(args: argparse.Namespace, model_path: str, hf_config: Any) -> dict[str, Any]:
    return {
        "model": args.model,
        "model_path": model_path,
        "model_source": args.model_source,
        "dtype": args.dtype,
        "device": args.device,
        "page_size": args.page_size,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "architectures": getattr(hf_config, "architectures", None),
    }


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _run_warmup(
    engine: PagedEngine,
    requests: list[BatchRequest],
    rounds: int,
    eos_token_id: int | None = None,
) -> None:
    """用同形状请求预热 kernel / allocator，结果不计入统计。"""
    for _ in range(rounds):
        engine.generate(requests, eos_token_id=eos_token_id)


def _log_header(args: argparse.Namespace, meta: dict[str, Any], title: str) -> list[str]:
    return [
        f"### {datetime.date.today().isoformat()} — {title} （阶段 {args.log_stage}）",
        "",
        f"- commit: {git_commit()}",
        f"- GPU / 驱动 / torch / transformers: {gpu_env(meta, args.device)}",
        f"- 模型 / dtype: `{meta['model']}` / {meta['dtype']}（{meta['model_source']}）",
        f"- KV cache: page_size={args.page_size}, max_running_req={args.max_running_req}",
        f"- warmup: {args.warmup} 次/配置（不计时）",
    ]


def format_generate_entry(stats: dict[str, Any], *, args: argparse.Namespace, meta: dict) -> str:
    lines = _log_header(args, meta, "paged KV Cache 批量生成")
    lines += [
        f"- 配置: max_tokens={args.max_tokens}, ignore_eos={args.ignore_eos}, "
        f"batch≤{args.max_running_req}, {stats['num_prompts']} 条固定 prompt"
        f"（{stats['num_batches']} 个 batch）",
        f"- 结果: prefill 均值 = {_fmt(stats['prefill_mean_ms'])} ms；"
        f"单步 decode 均值 = {_fmt(stats['decode_mean_ms'])} ms；"
        f"峰值显存 = {_fmt(stats['peak_memory_gb'])} GB",
        "",
        f"- 结论 / 下一步: {args.log_note}",
        "",
    ]
    return "\n".join(lines)


def format_bench_entry(
    rows: list[dict[str, Any]], *, args: argparse.Namespace, meta: dict, kind: str
) -> str:
    if kind == "batch":
        title = "paged decode 随 batch size 扩展"
        first_col = "batch size"
        config = f"固定 prompt 长度 {args.bench_token_len}，max_tokens={args.max_tokens}"
    else:
        title = "paged decode 随序列长度扩展"
        first_col = "prompt 长度"
        config = f"batch size={args.bench_batch}，max_tokens={args.max_tokens}"
    lines = _log_header(args, meta, title)
    lines += [
        f"- 配置: {config}；greedy，ignore_eos=True",
        "",
        f"| {first_col} | prefill (ms) | decode 均值 (ms) | 峰值显存 (GB) |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['key']} | {_fmt(row['prefill_ms'])} "
            f"| {_fmt(row['decode_mean_ms'])} | {_fmt(row['peak_memory_gb'])} |"
        )
    lines += ["", f"- 结论 / 下一步: {args.log_note}", ""]
    return "\n".join(lines)


def run_generation(
    args: argparse.Namespace,
    model: BaseLLMModel,
    config: ModelConfig,
    model_path: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    eos_token_id = tokenizer.eos_token_id
    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    input_id_lists = [
        tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts
    ]
    if any(not ids for ids in input_id_lists):
        raise SystemExit("tokenizer produced an empty sequence")
    max_seq_len = max(len(ids) for ids in input_id_lists) + args.max_tokens
    num_pages = _auto_num_pages(
        args, config, sum(len(ids) + args.max_tokens for ids in input_id_lists)
    )
    engine = _make_engine(args, model, config, num_pages, max_seq_len)

    out_dir = args.output or Path("out") / "paged" / model_slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    logits_by_id: dict[str, np.ndarray] = {}
    prefill_ms: list[float] = []
    decode_means: list[float] = []
    for start in range(0, len(prompts), args.max_running_req):
        end = start + args.max_running_req
        chunk = list(zip(prompts[start:end], input_id_lists[start:end]))
        requests = [
            BatchRequest(
                ids,
                SamplingParams(max_tokens=args.max_tokens, ignore_eos=args.ignore_eos),
            )
            for _, ids in chunk
        ]
        if start == 0:
            # 首个 batch 先打满同形状的预热请求，再重置显存统计
            warm_requests = [
                BatchRequest(ids, SamplingParams(max_tokens=1, ignore_eos=True)) for _, ids in chunk
            ]
            _run_warmup(engine, warm_requests, args.warmup, eos_token_id)
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
        run = engine.generate(requests, eos_token_id=eos_token_id)
        prefill_ms.append(run.prefill_ms)
        if run.decode_ms:
            decode_means.append(sum(run.decode_ms) / len(run.decode_ms))
        for offset, ((prompt, ids), result) in enumerate(zip(chunk, run.results)):
            case_id = f"p{start + offset:02d}"
            if result.prefill_logits is not None:
                logits_by_id[case_id] = (
                    result.prefill_logits.float().cpu().numpy().astype(np.float32)
                )
            cases.append(
                {
                    "id": case_id,
                    "prompt": prompt,
                    "input_ids": list(ids),
                    "output_ids": result.output_ids,
                    "stop_reason": result.stop_reason,
                    "timing": {"decode_ms": run.decode_ms},
                }
            )
        engine.check_integrity()
        print(
            f"batch {start // args.max_running_req + 1}: "
            f"size={len(chunk)} prefill={run.prefill_ms:.1f}ms "
            f"decode_mean={_fmt(_mean(run.decode_ms))}ms"
        )

    payload = {
        "kind": "paged",
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
    golden_json = out_dir / "golden.json"
    golden_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "logits.npz", **logits_by_id)

    peak_gb = None
    if args.device.startswith("cuda"):
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak memory: {peak_gb:.2f} GB")
    print(f"wrote {golden_json} and {out_dir / 'logits.npz'}")
    return {
        "num_prompts": len(cases),
        "num_batches": math.ceil(len(cases) / args.max_running_req),
        "prefill_mean_ms": _mean(prefill_ms),
        "decode_mean_ms": _mean(decode_means),
        "peak_memory_gb": peak_gb,
    }


def _bench_one(
    engine: PagedEngine,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    max_tokens: int,
    warmup: int,
) -> dict[str, Any]:
    token_id = min(1000, vocab_size - 1)
    if warmup > 0:
        warm_requests = [
            BatchRequest([token_id] * seq_len, SamplingParams(max_tokens=1, ignore_eos=True))
            for _ in range(batch_size)
        ]
        _run_warmup(engine, warm_requests, warmup)

    requests = [
        BatchRequest([token_id] * seq_len, SamplingParams(max_tokens=max_tokens, ignore_eos=True))
        for _ in range(batch_size)
    ]
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run = engine.generate(requests, eos_token_id=None)
    engine.check_integrity()
    peak = torch.cuda.max_memory_allocated() / 1e9 if engine.device.type == "cuda" else None
    return {
        "prefill_ms": run.prefill_ms,
        "decode_mean_ms": _mean(run.decode_ms),
        "peak_memory_gb": peak,
    }


def run_bench(
    args: argparse.Namespace,
    model: BaseLLMModel,
    config: ModelConfig,
    meta: dict[str, Any],
    *,
    kind: str,
) -> list[dict[str, Any]]:
    if kind == "batch":
        keys = [int(item) for item in args.bench_batch_sizes.split(",") if item.strip()]
        seq_lens = [args.bench_token_len] * len(keys)
        batch_sizes = keys
    else:
        keys = [int(item) for item in args.bench_seq_lens.split(",") if item.strip()]
        seq_lens = keys
        batch_sizes = [args.bench_batch] * len(keys)
    if not keys:
        raise SystemExit(f"--bench-{'batch-sizes' if kind == 'batch' else 'seq-lens'} is empty")

    max_seq_len = max(seq_lens) + args.max_tokens
    max_batch = max(batch_sizes)
    needed = sum(batch * (length + args.max_tokens) for batch, length in zip(batch_sizes, seq_lens))
    num_pages = _auto_num_pages(args, config, needed)

    # 基准需要更大的请求槽位；同时让日志记录真实的 max_running_req
    args.max_running_req = max(args.max_running_req, max_batch)
    engine = _make_engine(args, model, config, num_pages, max_seq_len)
    rows = []
    for key, batch_size, seq_len in zip(keys, batch_sizes, seq_lens):
        stats = _bench_one(
            engine, config.vocab_size, batch_size, seq_len, args.max_tokens, args.warmup
        )
        row = {"key": key, **stats}
        rows.append(row)
        print(
            f"{kind}={key:>6} prefill={stats['prefill_ms']:8.2f}ms "
            f"decode_mean={_fmt(stats['decode_mean_ms'])}ms "
            f"peak={_fmt(stats['peak_memory_gb'])}GB"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    model, config, model_path, hf_config = load_model(args)
    meta = _meta(args, model_path, hf_config)

    entry = ""
    if args.bench_batch_sizes or args.bench_seq_lens:
        if args.bench_batch_sizes:
            rows = run_bench(args, model, config, meta, kind="batch")
            entry = format_bench_entry(rows, args=args, meta=meta, kind="batch")
        if args.bench_seq_lens:
            rows = run_bench(args, model, config, meta, kind="seq")
            entry += format_bench_entry(rows, args=args, meta=meta, kind="seq")
    else:
        stats = run_generation(args, model, config, model_path, meta)
        entry = format_generate_entry(stats, args=args, meta=meta)

    if args.append_log:
        insert_log_entry(args.log_path, args.log_stage, entry)
        print(f"\nappended log entry -> {args.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
