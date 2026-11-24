#!/usr/bin/env python3
"""阶段 1：用纯 PyTorch 模型做 greedy 生成（无 KV Cache）。

生成结果写成与阶段 0 黄金基线相同的 JSON/NPZ 格式，可直接用
``tools/compare.py`` 对比::

    python scripts/generate_naive.py --model Qwen/Qwen3-0.6B
    python tools/compare.py --baseline out/golden/Qwen__Qwen3-0.6B/golden.json \
                            --actual   out/naive/Qwen__Qwen3-0.6B/golden.json

还可测量序列长度增长时单步 decode 的耗时（阶段 2 的动机）::

    python scripts/generate_naive.py --model Qwen/Qwen3-0.6B \
        --bench-lengths 128,256,512,1024,2048

加 ``--append-log`` 可把本轮结果写入 ``experiments/log.md`` 的阶段小节::

    python scripts/generate_naive.py --model Qwen/Qwen3-0.6B \
        --bench-lengths 128,256,512,1024,2048 --append-log
"""

from __future__ import annotations

import argparse
import datetime
import json
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
from minisgl.engine import naive_generate
from minisgl.models import (
    BaseLLMModel,
    ModelConfig,
    create_model,
    load_weight,
    resolve_model_path,
)
from transformers import AutoConfig, AutoTokenizer

from tools.logbook import git_commit, gpu_env, insert_log_entry
from tools.memory import peak_memory_gb, reset_peak_memory
from tools.workloads import PROMPTS


def model_slug(model_name: str) -> str:
    return model_name.rstrip("/").replace("/", "__")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="model id or local path")
    parser.add_argument(
        "--model-source",
        choices=("modelscope", "huggingface"),
        default="modelscope",
        help="where to download the model from",
    )
    parser.add_argument("--output", type=Path, default=None, help="output directory")
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="only use the first N prompts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bench-lengths",
        default="",
        help="comma-separated prompt lengths; measure one naive decode step for each",
    )
    parser.add_argument("--bench-repeats", type=int, default=3)
    parser.add_argument(
        "--append-log",
        action="store_true",
        help="把本轮结果追加到实验日志（默认 experiments/log.md）",
    )
    parser.add_argument("--log-path", type=Path, default=Path("experiments/log.md"))
    parser.add_argument("--log-stage", type=int, default=1, help="记录所属阶段编号（默认 1）")
    parser.add_argument("--log-note", default="待填", help="记录中的“结论 / 下一步”")
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.bench_repeats < 1:
        parser.error("--bench-repeats must be >= 1")
    return args


def load_model(args: argparse.Namespace) -> tuple[BaseLLMModel, ModelConfig, str, Any]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run this script on the GPU host")
    dtype = getattr(torch, args.dtype)
    path = resolve_model_path(args.model, args.model_source)
    hf_config = AutoConfig.from_pretrained(path)
    config = ModelConfig.from_hf(hf_config)
    model = create_model(config, device=args.device, dtype=dtype)
    state = dict(
        load_weight(
            path,
            device=args.device,
            dtype=dtype,
            tie_word_embeddings=config.tie_word_embeddings,
        )
    )
    model.load_state_dict(state)
    return model, config, path, hf_config


def _meta(args: argparse.Namespace, model_path: str, hf_config: Any) -> dict:
    return {
        "model": args.model,
        "model_path": model_path,
        "model_source": args.model_source,
        "dtype": args.dtype,
        "device": args.device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(torch.device(args.device))
            if args.device.startswith("cuda")
            else None
        ),
        "architectures": getattr(hf_config, "architectures", None),
    }


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _log_header(args: argparse.Namespace, meta: dict, title: str) -> list[str]:
    return [
        f"### {datetime.date.today().isoformat()} — {title} （阶段 {args.log_stage}）",
        "",
        f"- commit: {git_commit()}",
        f"- GPU / 驱动 / torch / transformers: {gpu_env(meta, args.device)}",
        f"- 模型 / dtype: `{meta['model']}` / {meta['dtype']}（{meta['model_source']}）",
    ]


def format_generate_entry(stats: dict[str, Any], *, args: argparse.Namespace, meta: dict) -> str:
    lines = _log_header(args, meta, "naive 生成（无 KV Cache）")
    lines += [
        f"- 配置: max_tokens={args.max_tokens}, ignore_eos={args.ignore_eos}, "
        f"seed={args.seed}, {stats['num_prompts']} 条固定 prompt（raw token ids）",
        f"- 结果: prefill 均值 = {_fmt(stats['prefill_mean_ms'])} ms；"
        f"单步 decode 均值 = {_fmt(stats['decode_mean_ms'])} ms；"
        f"峰值显存 = {_fmt(stats['peak_memory_gb'])} GB",
        "",
        f"- 结论 / 下一步: {args.log_note}",
        "",
    ]
    return "\n".join(lines)


def format_bench_entry(
    results: list[dict[str, Any]], *, args: argparse.Namespace, meta: dict
) -> str:
    lengths = ", ".join(str(item["prompt_len"]) for item in results)
    lines = _log_header(args, meta, "naive decode 长度扩展")
    lines += [
        f"- 配置: 无 KV Cache，每步重算全序列；bench_repeats={args.bench_repeats}",
        f"- 负载: prompt 长度 {lengths}",
        "",
        "| prompt 长度 | prefill 耗时 (ms) | 单步 decode（长度 L+1）(ms) |",
        "| --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item['prompt_len']} | {_fmt(item['prefill_ms'])} "
            f"| {_fmt(item['decode_mean_ms'])} |"
        )
    lines += ["", f"- 结论 / 下一步: {args.log_note}", ""]
    return "\n".join(lines)


def run_generation(
    args: argparse.Namespace, model: BaseLLMModel, model_path: str, meta: dict
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    eos_token_id = tokenizer.eos_token_id
    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    out_dir = args.output or Path("out") / "naive" / model_slug(args.model)

    cases: list[dict] = []
    logits_by_id: dict[str, np.ndarray] = {}
    reset_peak_memory(args.device)

    for index, prompt in enumerate(prompts):
        case_id = f"p{index:02d}"
        input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not input_ids:
            raise SystemExit(f"{case_id}: tokenizer produced an empty sequence")
        params = SamplingParams(max_tokens=args.max_tokens, ignore_eos=args.ignore_eos)
        result = naive_generate(
            model, input_ids, params, eos_token_id=eos_token_id, device=args.device
        )
        logits_by_id[case_id] = result.prefill_logits.float().cpu().numpy().astype(np.float32)
        cases.append(
            {
                "id": case_id,
                "prompt": prompt,
                "input_ids": list(input_ids),
                "output_ids": result.output_ids,
                "stop_reason": result.stop_reason,
                "timing": {
                    "prefill_ms": result.stats.prefill_ms,
                    "decode_ms": result.stats.decode_ms,
                },
            }
        )
        print(
            f"[{index + 1}/{len(prompts)}] {case_id} in={len(input_ids)} "
            f"out={len(result.output_ids)} stop={result.stop_reason} "
            f"prefill={result.stats.prefill_ms:.1f}ms "
            f"decode_mean={_mean(result.stats.decode_ms) or 0.0:.1f}ms"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "naive",
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

    peak_gb = peak_memory_gb(args.device)
    if peak_gb is not None:
        print(f"peak memory: {peak_gb:.2f} GB")
    print(f"wrote {golden_json} and {out_dir / 'logits.npz'}")

    decode_means: list[float] = []
    for case in cases:
        mean = _mean(case["timing"]["decode_ms"])
        if mean is not None:
            decode_means.append(mean)
    return {
        "num_prompts": len(cases),
        "prefill_mean_ms": _mean([case["timing"]["prefill_ms"] for case in cases]),
        "decode_mean_ms": _mean(decode_means),
        "peak_memory_gb": peak_gb,
    }


def run_bench(
    args: argparse.Namespace,
    model: BaseLLMModel,
    out_dir: Path,
    meta: dict,
    vocab_size: int,
) -> list[dict[str, Any]]:
    device = torch.device(args.device)
    lengths = [int(item) for item in args.bench_lengths.split(",") if item.strip()]
    if not lengths:
        raise SystemExit("--bench-lengths is empty")
    token_id = min(1000, vocab_size - 1)
    results = []
    for length in lengths:
        ids = torch.full((1, length), token_id, dtype=torch.long, device=device)
        _sync(device)
        start = time.perf_counter()
        model.forward(ids, logits_to_keep=1)
        _sync(device)
        prefill_ms = (time.perf_counter() - start) * 1e3

        step_ids = torch.cat(
            [ids, torch.full((1, 1), token_id, dtype=torch.long, device=device)], dim=1
        )
        decode = []
        for _ in range(args.bench_repeats):
            _sync(device)
            start = time.perf_counter()
            model.forward(step_ids, logits_to_keep=1)
            _sync(device)
            decode.append((time.perf_counter() - start) * 1e3)
        item = {
            "prompt_len": length,
            "prefill_ms": prefill_ms,
            "decode_ms": decode,
            "decode_mean_ms": _mean(decode),
        }
        results.append(item)
        print(
            f"len={length:>6} prefill={prefill_ms:8.2f}ms "
            f"decode(L+1)={item['decode_mean_ms']:8.2f}ms"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bench_decode.json"
    path.write_text(
        json.dumps(
            {"meta": meta, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {path}")
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    model, config, model_path, hf_config = load_model(args)
    out_dir = args.output or Path("out") / "naive" / model_slug(args.model)
    meta = _meta(args, model_path, hf_config)

    if args.bench_lengths:
        results = run_bench(args, model, out_dir, meta, config.vocab_size)
        entry = format_bench_entry(results, args=args, meta=meta)
    else:
        stats = run_generation(args, model, model_path, meta)
        entry = format_generate_entry(stats, args=args, meta=meta)

    if args.append_log:
        insert_log_entry(args.log_path, args.log_stage, entry)
        print(f"\nappended log entry -> {args.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
