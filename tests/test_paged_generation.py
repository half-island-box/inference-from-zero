"""阶段 2 GPU 验收：paged 引擎与阶段 0 黄金基线逐 token / logits 对比。"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from minisgl.core import SamplingParams  # noqa: E402
from minisgl.models import (  # noqa: E402
    ModelConfig,
    create_model,
    load_model_weights,
    load_weight,
    resolve_model_path,
)
from minisgl.scheduler import BatchRequest, PagedEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "out" / "golden" / "Qwen__Qwen3-0.6B" / "golden.json"


def _load_payload() -> tuple[dict, Path]:
    golden_path = Path(os.environ.get("GOLDEN_PATH", str(DEFAULT_GOLDEN)))
    if not golden_path.exists():
        pytest.skip(f"golden baseline not found at {golden_path}; run tools/reference_hf.py first")
    return json.loads(golden_path.read_text(encoding="utf-8")), golden_path


def _build_engine(payload: dict, golden_path: Path, *, page_size: int, limit: int = 0):
    model_name = os.environ.get("GOLDEN_MODEL", payload["meta"]["model"])
    model_source = os.environ.get("MODEL_SOURCE", "modelscope")
    dtype = getattr(torch, payload["meta"].get("dtype", "bfloat16"))
    cases = payload["cases"][:limit] if limit else payload["cases"]
    sampling = payload["sampling"]

    model_path = resolve_model_path(model_name, model_source)
    config = ModelConfig.from_hf(transformers.AutoConfig.from_pretrained(model_path))
    model = create_model(config, device="meta", dtype=dtype, rope_device="cuda")
    load_model_weights(
        model,
        load_weight(
            model_path,
            device="cuda",
            dtype=dtype,
            model_source=model_source,
            tie_word_embeddings=config.tie_word_embeddings,
        ),
        dtype=dtype,
        assign=True,
    )

    max_seq_len = max(len(case["input_ids"]) for case in cases) + sampling["max_tokens"]
    per_page = (
        2 * config.num_layers * config.num_kv_heads * config.head_dim * page_size * dtype.itemsize
    )
    free_bytes, _ = torch.cuda.mem_get_info()
    max_pages = max(int(free_bytes * 0.4) // per_page, 1)
    needed = math.ceil(
        sum(len(case["input_ids"]) + sampling["max_tokens"] for case in cases) / page_size
    )
    num_pages = min(max(needed * 2, 256), max_pages)

    engine = PagedEngine(
        model,
        config,
        num_pages=num_pages,
        page_size=page_size,
        max_running_req=len(cases),
        max_seq_len=max_seq_len,
        device="cuda",
        dtype=dtype,
    )
    return engine, cases, sampling, golden_path


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("page_size", [16, 1])
def test_paged_matches_golden(page_size: int) -> None:
    payload, golden_path = _load_payload()
    engine, cases, sampling, golden_path = _build_engine(payload, golden_path, page_size=page_size)
    golden_logits = np.load(golden_path.parent / payload.get("logits_file", "logits.npz"))

    requests = [
        BatchRequest(
            case["input_ids"],
            SamplingParams(max_tokens=sampling["max_tokens"], ignore_eos=sampling["ignore_eos"]),
        )
        for case in cases
    ]
    run = engine.generate(requests, eos_token_id=sampling["eos_token_id"])
    engine.check_integrity()

    failures: list[str] = []
    for case, result in zip(cases, run.results):
        if result.output_ids != case["output_ids"]:
            failures.append(f"{case['id']}: tokens {result.output_ids} != {case['output_ids']}")
        if result.stop_reason != case["stop_reason"]:
            failures.append(f"{case['id']}: stop {result.stop_reason} != {case['stop_reason']}")
        if case["id"] in golden_logits.files and result.prefill_logits is not None:
            diff = float(
                np.max(
                    np.abs(result.prefill_logits.float().cpu().numpy() - golden_logits[case["id"]])
                )
            )
            if diff > 1e-2:
                failures.append(f"{case['id']}: logits max diff {diff:.3e}")

    assert not failures, "\n".join(failures)
