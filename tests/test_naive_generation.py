"""阶段 1 GPU 验收：naive 引擎与阶段 0 黄金基线逐 token / logits 对比。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from minisgl.core import SamplingParams  # noqa: E402
from minisgl.engine import naive_generate  # noqa: E402
from minisgl.models import (  # noqa: E402
    ModelConfig,
    create_model,
    load_weight,
    resolve_model_path,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "out" / "golden" / "Qwen__Qwen3-0.6B" / "golden.json"


@pytest.mark.gpu
@pytest.mark.slow
def test_naive_matches_golden() -> None:
    golden_path = Path(os.environ.get("GOLDEN_PATH", str(DEFAULT_GOLDEN)))
    if not golden_path.exists():
        pytest.skip(f"golden baseline not found at {golden_path}; run tools/reference_hf.py first")

    payload = json.loads(golden_path.read_text(encoding="utf-8"))
    model_name = os.environ.get("GOLDEN_MODEL", payload["meta"]["model"])
    model_source = os.environ.get("MODEL_SOURCE", "modelscope")
    dtype = getattr(torch, payload["meta"].get("dtype", "bfloat16"))
    limit = int(os.environ.get("NAIVE_LIMIT", "0"))

    model_path = resolve_model_path(model_name, model_source)
    config = ModelConfig.from_hf(transformers.AutoConfig.from_pretrained(model_path))
    model = create_model(config, device="cuda", dtype=dtype)
    model.load_state_dict(
        dict(
            load_weight(
                model_path,
                device="cuda",
                dtype=dtype,
                model_source=model_source,
                tie_word_embeddings=config.tie_word_embeddings,
            )
        )
    )

    golden_logits = np.load(golden_path.parent / payload.get("logits_file", "logits.npz"))
    cases = payload["cases"][:limit] if limit else payload["cases"]
    sampling = payload["sampling"]
    failures: list[str] = []

    for case in cases:
        params = SamplingParams(
            max_tokens=sampling["max_tokens"], ignore_eos=sampling["ignore_eos"]
        )
        result = naive_generate(
            model,
            case["input_ids"],
            params,
            eos_token_id=sampling["eos_token_id"],
            device="cuda",
        )
        if result.output_ids != case["output_ids"]:
            failures.append(f"{case['id']}: output tokens differ")
        if result.stop_reason != case["stop_reason"]:
            failures.append(f"{case['id']}: stop reason differs")
        if case["id"] in golden_logits.files:
            diff = float(
                np.max(
                    np.abs(result.prefill_logits.float().cpu().numpy() - golden_logits[case["id"]])
                )
            )
            if diff > 1e-2:
                failures.append(f"{case['id']}: logits max diff {diff:.3e}")

    assert not failures, "\n".join(failures)
