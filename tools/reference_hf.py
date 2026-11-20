#!/usr/bin/env python3
"""生成 HuggingFace 黄金基线（路线图阶段 0）。

对每条固定 prompt 记录：
  * tokenize 后的 ``input_ids``（原始 prompt，``add_special_tokens=False``）；
  * greedy ``output_ids`` 以及停止原因；
  * 一次无 cache 前向后最后一个位置的 logits（float32）。

输出布局::

    out/golden/<model-slug>/golden.json
    out/golden/<model-slug>/logits.npz

在 GPU 主机上运行::

    python tools/reference_hf.py --model Qwen/Qwen3-0.6B
    python tools/reference_hf.py --model Qwen/Qwen3-0.6B --check-determinism
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.hf_runtime import forward_last_logits, greedy_generate, load_hf_runtime
from tools.workloads import PROMPTS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="model id or local path")
    parser.add_argument(
        "--model-source",
        choices=("modelscope", "huggingface"),
        default="modelscope",
        help="where to download the model from (default: modelscope)",
    )
    parser.add_argument("--output", type=Path, default=None, help="output directory")
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="optional HF attention backend, e.g. sdpa / flash_attention_2 / eager",
    )
    parser.add_argument("--max-tokens", type=int, default=32, help="greedy tokens per prompt")
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="generate exactly --max-tokens tokens even if EOS is produced",
    )
    parser.add_argument("--limit", type=int, default=0, help="only use the first N prompts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="regenerate everything and verify the tokens/logits are reproducible",
    )
    parser.add_argument(
        "--determinism-tolerance",
        type=float,
        default=1e-4,
        help="max logits diff allowed when --check-determinism is set",
    )
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    return args


def model_slug(model_name: str) -> str:
    return model_name.rstrip("/").replace("/", "__")


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    import torch

    args = parse_args(argv)
    torch.manual_seed(args.seed)

    runtime = load_hf_runtime(
        args.model,
        dtype=args.dtype,
        device=args.device,
        model_source=args.model_source,
        attn_implementation=args.attn_implementation,
    )
    print(f"loaded {args.model} ({runtime.meta['num_parameters']:,} params) on {args.device}")

    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    cases: list[dict] = []
    logits_by_id: dict[str, "np.ndarray"] = {}

    for index, prompt in enumerate(prompts):
        case_id = f"p{index:02d}"
        input_ids = runtime.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not input_ids:
            raise SystemExit(f"{case_id}: tokenizer produced an empty sequence")
        output_ids, stop_reason = greedy_generate(
            runtime, input_ids, args.max_tokens, ignore_eos=args.ignore_eos
        )
        logits = forward_last_logits(runtime, input_ids)
        logits_by_id[case_id] = logits
        cases.append(
            {
                "id": case_id,
                "prompt": prompt,
                "input_ids": list(input_ids),
                "output_ids": list(output_ids),
                "stop_reason": stop_reason,
            }
        )
        print(
            f"[{index + 1}/{len(prompts)}] {case_id} "
            f"in={len(input_ids)} out={len(output_ids)} stop={stop_reason}"
        )

    out_dir = args.output or Path("out") / "golden" / model_slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "kind": "golden",
        "meta": runtime.meta,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sampling": {
            "max_tokens": args.max_tokens,
            "ignore_eos": args.ignore_eos,
            "eos_token_id": runtime.eos_token_id,
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
    print(f"wrote {golden_json} and {out_dir / 'logits.npz'}")

    if args.check_determinism:
        failures: list[str] = []
        for case in cases:
            again, reason = greedy_generate(
                runtime, case["input_ids"], args.max_tokens, ignore_eos=args.ignore_eos
            )
            if again != case["output_ids"] or reason != case["stop_reason"]:
                failures.append(f"{case['id']}: tokens differ")
                continue
            diff = float(
                np.max(np.abs(forward_last_logits(runtime, case["input_ids"]) - logits_by_id[case["id"]]))
            )
            if diff > args.determinism_tolerance:
                failures.append(f"{case['id']}: logits max diff {diff:.3e}")
        if failures:
            print("determinism check FAILED:")
            for failure in failures:
                print(f"  {failure}")
            return 1
        print(f"determinism check PASSED ({len(cases)} cases)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
