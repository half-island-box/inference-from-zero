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
        "cuda_device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "architectures": getattr(hf_config, "architectures", None),
    }


def run_generation(
    args: argparse.Namespace, model: BaseLLMModel, model_path: str, hf_config: Any
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    eos_token_id = tokenizer.eos_token_id
    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    out_dir = args.output or Path("out") / "naive" / model_slug(args.model)

    cases: list[dict] = []
    logits_by_id: dict[str, np.ndarray] = {}
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

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
        "meta": _meta(args, model_path, hf_config),
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
    if args.device.startswith("cuda"):
        print(f"peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"wrote {golden_json} and {out_dir / 'logits.npz'}")


def run_bench(
    args: argparse.Namespace,
    model: BaseLLMModel,
    out_dir: Path,
    model_path: str,
    hf_config: Any,
    vocab_size: int,
) -> None:
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
            {"meta": _meta(args, model_path, hf_config), "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    model, config, model_path, hf_config = load_model(args)
    out_dir = args.output or Path("out") / "naive" / model_slug(args.model)

    if args.bench_lengths:
        run_bench(args, model, out_dir, model_path, hf_config, config.vocab_size)
    else:
        run_generation(args, model, model_path, hf_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
