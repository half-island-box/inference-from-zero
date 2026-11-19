"""用合成基线对 tools/compare.py 做纯 CPU 测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.compare import compare


def _write_run(
    directory: Path,
    cases: list[dict],
    logits: dict[str, np.ndarray] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict = {"cases": cases}
    if logits is not None:
        payload["logits_file"] = "logits.npz"
        np.savez(directory / "logits.npz", **logits)
    path = directory / "golden.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _case(case_id: str, output_ids: list[int], stop_reason: str = "length") -> dict:
    return {
        "id": case_id,
        "input_ids": [1, 2, 3],
        "output_ids": output_ids,
        "stop_reason": stop_reason,
    }


def test_identical_runs_pass(tmp_path: Path) -> None:
    cases = [_case("p00", [10, 11, 12])]
    logits = {"p00": np.array([0.1, 0.2, 0.3], dtype=np.float32)}
    baseline = _write_run(tmp_path / "base", cases, logits)
    actual = _write_run(tmp_path / "act", cases, logits)

    result = compare(baseline, actual, tolerance=1e-2)
    assert result.ok
    assert result.logits_checked == 1
    assert result.max_logit_diff == 0.0


def test_token_mismatch_fails(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path / "base", [_case("p00", [10, 11])])
    actual = _write_run(tmp_path / "act", [_case("p00", [10, 99])])

    result = compare(baseline, actual, compare_logits=False)
    assert not result.ok
    assert result.token_mismatches == ["p00"]


def test_missing_case_fails(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path / "base", [_case("p00", [1]), _case("p01", [2])])
    actual = _write_run(tmp_path / "act", [_case("p00", [1])])

    result = compare(baseline, actual, compare_logits=False)
    assert not result.ok
    assert result.missing == ["p01"]


def test_stop_reason_mismatch_fails(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path / "base", [_case("p00", [1], "eos")])
    actual = _write_run(tmp_path / "act", [_case("p00", [1], "length")])

    result = compare(baseline, actual, compare_logits=False)
    assert not result.ok
    assert result.stop_mismatches == ["p00"]


def test_logits_tolerance(tmp_path: Path) -> None:
    cases = [_case("p00", [10])]
    baseline_logits = {"p00": np.zeros(4, dtype=np.float32)}
    actual_logits = {"p00": np.full(4, 0.05, dtype=np.float32)}
    baseline = _write_run(tmp_path / "base", cases, baseline_logits)
    actual = _write_run(tmp_path / "act", cases, actual_logits)

    assert compare(baseline, actual, tolerance=1e-1).ok
    strict = compare(baseline, actual, tolerance=1e-2)
    assert not strict.ok
    assert strict.logit_failures


def test_logits_shape_mismatch_fails(tmp_path: Path) -> None:
    cases = [_case("p00", [10])]
    baseline = _write_run(tmp_path / "base", cases, {"p00": np.zeros(4, dtype=np.float32)})
    actual = _write_run(tmp_path / "act", cases, {"p00": np.zeros(5, dtype=np.float32)})

    result = compare(baseline, actual)
    assert not result.ok
    assert result.logit_shape_mismatches == ["p00"]
