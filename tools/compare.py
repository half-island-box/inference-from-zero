#!/usr/bin/env python3
"""将一次运行结果与黄金基线对比（路线图阶段 0）。

greedy token 序列必须完全一致；logits 按容差比较：bf16 用 1e-2，fp32 调试时用 1e-3。

用法::


    python tools/compare.py --baseline out/golden/Qwen__Qwen3-0.6B/golden.json \
                            --actual out/run/golden.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 封装一次运行结果和基线的对比
@dataclass
class Comparison:
    num_cases: int = 0
    missing: list[str] = field(default_factory=list)
    input_mismatches: list[str] = field(default_factory=list)
    token_mismatches: list[str] = field(default_factory=list)
    stop_mismatches: list[str] = field(default_factory=list)
    logit_shape_mismatches: list[str] = field(default_factory=list)
    logit_failures: list[tuple[str, float]] = field(default_factory=list)
    max_logit_diff: float = 0.0
    logits_checked: int = 0

    @property
    def ok(self) -> bool:
        return not (
            self.missing
            or self.input_mismatches
            or self.token_mismatches
            or self.stop_mismatches
            or self.logit_shape_mismatches
            or self.logit_failures
        )


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_logits(json_path: Path, payload: dict) -> dict[str, np.ndarray]:
    name = payload.get("logits_file")
    if not name:
        return {}
    path = json_path.parent / name
    if not path.exists():
        return {}
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def compare(
    baseline_path: Path,
    actual_path: Path,
    tolerance: float = 1e-2,
    compare_logits: bool = True,
) -> Comparison:
    baseline = _load_json(baseline_path)
    actual = _load_json(actual_path)

    result = Comparison()
    baseline_cases = {case["id"]: case for case in baseline.get("cases", [])}
    actual_cases = {case["id"]: case for case in actual.get("cases", [])}
    result.num_cases = len(baseline_cases)

    for case_id, expected in baseline_cases.items():
        got = actual_cases.get(case_id)
        if got is None:
            result.missing.append(case_id)
            continue
        if expected.get("input_ids") != got.get("input_ids"):
            result.input_mismatches.append(case_id)
        if expected.get("output_ids") != got.get("output_ids"):
            result.token_mismatches.append(case_id)
        if "stop_reason" in expected and expected["stop_reason"] != got.get("stop_reason"):
            result.stop_mismatches.append(case_id)

    if compare_logits:
        expected_logits = _load_logits(baseline_path, baseline)
        actual_logits = _load_logits(actual_path, actual)
        for case_id in baseline_cases:
            if case_id not in expected_logits or case_id not in actual_logits:
                continue
            a = expected_logits[case_id]
            b = actual_logits[case_id]
            if a.shape != b.shape:
                result.logit_shape_mismatches.append(case_id)
                continue
            diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
            result.max_logit_diff = max(result.max_logit_diff, diff)
            result.logits_checked += 1
            if diff > tolerance:
                result.logit_failures.append((case_id, diff))

    return result


def format_report(
    result: Comparison,
    tolerance: float,
    compare_logits: bool,
    baseline_path: Path,
    actual_path: Path,
) -> str:
    lines = [
        f"baseline: {baseline_path}",
        f"actual:   {actual_path}",
        f"cases:    {result.num_cases} baseline cases",
    ]
    if result.missing:
        lines.append(f"  missing cases: {', '.join(result.missing)}")
    if result.input_mismatches:
        lines.append(f"  input_ids differ: {', '.join(result.input_mismatches)}")
    if result.token_mismatches:
        lines.append(f"  output tokens differ: {', '.join(result.token_mismatches)}")
    if result.stop_mismatches:
        lines.append(f"  stop reasons differ: {', '.join(result.stop_mismatches)}")

    if compare_logits:
        lines.append(
            f"logits:   {result.logits_checked} checked, "
            f"max_abs_diff={result.max_logit_diff:.3e} (tol {tolerance:.1e})"
        )
        if result.logits_checked == 0:
            lines.append("  WARNING: no logits were compared (missing logits.npz?)")
        if result.logit_shape_mismatches:
            lines.append(f"  shape mismatch: {', '.join(result.logit_shape_mismatches)}")
        if result.logit_failures:
            detail = ", ".join(f"{cid}={diff:.3e}" for cid, diff in result.logit_failures)
            lines.append(f"  over tolerance: {detail}")

    lines.append("RESULT: PASS" if result.ok else "RESULT: FAIL")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, required=True, help="golden.json to compare against")
    parser.add_argument("--actual", type=Path, required=True, help="golden.json produced by the run")
    parser.add_argument("--tolerance", type=float, default=1e-2, help="logits max_abs_diff tolerance")
    parser.add_argument("--skip-logits", action="store_true", help="only compare tokens")
    args = parser.parse_args(argv)

    result = compare(
        args.baseline,
        args.actual,
        tolerance=args.tolerance,
        compare_logits=not args.skip_logits,
    )
    print(format_report(result, args.tolerance, not args.skip_logits, args.baseline, args.actual))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
