"""阶段 3 GPU 验收：连续批处理与黄金基线 / 串行结果对比。"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from minisgl.core import SamplingParams  # noqa: E402
from minisgl.llm import LLM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "out" / "golden" / "Qwen__Qwen3-0.6B" / "golden.json"


def _load_payload() -> tuple[dict, Path]:
    golden_path = Path(os.environ.get("GOLDEN_PATH", str(DEFAULT_GOLDEN)))
    if not golden_path.exists():
        pytest.skip(f"golden baseline not found at {golden_path}; run tools/reference_hf.py first")
    return json.loads(golden_path.read_text(encoding="utf-8")), golden_path


def _build_llm(
    payload: dict,
    *,
    page_size: int,
    max_running_req: int,
    needed_tokens: int,
    max_seq_len: int,
    record_prefill_logits: bool,
    max_extend_tokens: int = 8192,
) -> LLM:
    model_name = os.environ.get("GOLDEN_MODEL", payload["meta"]["model"])
    model_source = os.environ.get("MODEL_SOURCE", "modelscope")
    dtype = getattr(torch, payload["meta"].get("dtype", "bfloat16"))

    # 按负载估算页数，避免占满整张卡
    hf_config = None
    path = None
    from minisgl.models import ModelConfig, resolve_model_path

    path = resolve_model_path(model_name, model_source)
    hf_config = transformers.AutoConfig.from_pretrained(path)
    config = ModelConfig.from_hf(hf_config)
    per_page = (
        2 * config.num_layers * config.num_kv_heads * config.head_dim * page_size * dtype.itemsize
    )
    free_bytes, _ = torch.cuda.mem_get_info()
    free_pages = max(int(free_bytes * 0.4) // per_page, 1)
    num_pages = min(free_pages, math.ceil(needed_tokens / page_size) + 64)

    return LLM(
        str(path),
        model_source=model_source,
        device="cuda",
        dtype=dtype,
        page_size=page_size,
        max_running_req=max_running_req,
        num_pages=num_pages,
        max_seq_len=max_seq_len,
        max_extend_tokens=max_extend_tokens,
        record_prefill_logits=record_prefill_logits,
    )


def _check_golden(max_extend_tokens: int) -> None:
    payload, golden_path = _load_payload()
    cases = payload["cases"]
    sampling = payload["sampling"]
    max_seq_len = max(len(case["input_ids"]) for case in cases) + sampling["max_tokens"]
    needed = sum(len(case["input_ids"]) + sampling["max_tokens"] for case in cases)
    llm = _build_llm(
        payload,
        page_size=16,
        max_running_req=len(cases),
        needed_tokens=needed,
        max_seq_len=max_seq_len,
        record_prefill_logits=True,
        max_extend_tokens=max_extend_tokens,
    )
    golden_logits = np.load(golden_path.parent / payload.get("logits_file", "logits.npz"))

    params = [
        SamplingParams(max_tokens=sampling["max_tokens"], ignore_eos=sampling["ignore_eos"])
        for _ in cases
    ]
    results = llm.generate([case["input_ids"] for case in cases], params)
    llm.scheduler.check_integrity()

    failures: list[str] = []
    for case, result in zip(cases, results):
        if result["token_ids"] != case["output_ids"]:
            failures.append(f"{case['id']}: tokens {result['token_ids']} != {case['output_ids']}")
        if result["stop_reason"] != case["stop_reason"]:
            failures.append(f"{case['id']}: stop {result['stop_reason']} != {case['stop_reason']}")
        if case["id"] in golden_logits.files and result["prefill_logits"] is not None:
            diff = float(
                np.max(
                    np.abs(
                        result["prefill_logits"].float().cpu().numpy() - golden_logits[case["id"]]
                    )
                )
            )
            if diff > 1e-2:
                failures.append(f"{case['id']}: logits max diff {diff:.3e}")

    assert not failures, "\n".join(failures)


@pytest.mark.gpu
@pytest.mark.slow
def test_continuous_matches_golden() -> None:
    _check_golden(max_extend_tokens=8192)


@pytest.mark.gpu
@pytest.mark.slow
def test_chunked_prefill_matches_golden() -> None:
    # chunk 预算小于 prompt 长度，强制走 ChunkedReq 路径
    _check_golden(max_extend_tokens=16)


@pytest.mark.gpu
@pytest.mark.slow
def test_fifty_mixed_matches_serial() -> None:
    payload, _ = _load_payload()
    rng = random.Random(0)
    vocab_size = 1000
    prompts = [[rng.randrange(vocab_size) for _ in range(rng.randint(8, 64))] for _ in range(50)]
    params = [SamplingParams(max_tokens=rng.randint(1, 4), ignore_eos=True) for _ in range(50)]
    max_seq_len = max(len(prompt) for prompt in prompts) + 4
    needed = sum(len(prompt) + param.max_tokens for prompt, param in zip(prompts, params))
    llm = _build_llm(
        payload,
        page_size=16,
        max_running_req=50,
        needed_tokens=needed,
        max_seq_len=max_seq_len,
        record_prefill_logits=False,
    )

    concurrent = llm.generate(prompts, params)
    serial = [llm.generate([prompt], [param])[0] for prompt, param in zip(prompts, params)]
    llm.scheduler.check_integrity()

    for index, (concurrent_result, serial_result) in enumerate(zip(concurrent, serial)):
        assert (
            concurrent_result["token_ids"] == serial_result["token_ids"]
        ), f"request {index}: {concurrent_result['token_ids']} != {serial_result['token_ids']}"
        assert concurrent_result["stop_reason"] == serial_result["stop_reason"]
