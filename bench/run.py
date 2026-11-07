"""场景化 bench：看"稳态 decode 遇到新请求"时的尾延迟。

用法::

    python -m bench.run --scenario mixed_long_short --trace out/mixed.jsonl

``mixed_long_short`` 的构造：

* 先塞 ``--num-short`` 条短请求（默认 128 条，prompt 128 tok），让它们一起进入
  稳态 decode；
* 稳态之后，每 ``--long-every-steps`` 个 engine step 注入一条长 prompt
  （默认 20 步、3072 tok），共 ``--num-long`` 条（默认 5 条）；
* 关注两件事：
  1. 短请求的 TBT（time between tokens）在长 prompt 的 prefill 步里会被拉长多少：
     输出 p50 / p99，并和"没有 prefill 打断"的 baseline 对比；
  2. 长 prompt 的 TTFT（从 add_request 到第一个 token 落地）。

这个调度器是 "prefill 优先" 的：只要 waiting 里有能调度的请求，本步就只做
prefill，running 里的 decode 全部让路。所以长 prompt 的 prefill 步会直接变成
短请求 TBT 的一个尖峰。bench 不改调度行为，只在引擎外面包一层 ``preempt``
计数，并把每一步、每个 token 的落地时间记下来。

设计取舍：

* 手动驱动 ``LLMEngine.step()``，而不是 ``generate()``，这样能拿到每一步的
  开始/结束时间和本步实际调度的 seq；
* 每个 step 前后都 ``torch.cuda.synchronize()``，避免把异步 kernel 的时间
  算到下一步；
* 短请求的 TBT 按 token 落地时间算：同一 step 内落地的 token 共享该 step 的
  结束时间。跨过 prefill 步的 TBT 单独归到 ``interrupted`` 桶，不和安静 decode
  混在一起；
* 如果 KV cache 不够导致 ``Scheduler.preempt()``，被抢占的短请求会有重新
  prefill 的空档。这种 TBT 单独排除，并打印警告（否则 p99 会混入"重算"）。

``--trace`` 写 JSONL，事件类型：``meta`` / ``request`` / ``preempt`` / ``step`` /
``steady_start`` / ``finish`` / ``end`` / ``summary``。每行都有相对于本次运行
开始的 ``t_ms``；``step`` 事件的 ``t_ms`` 是该步结束、token 落地的时刻。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from math import ceil
from random import randint, seed

DEFAULT_MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")


# --------------------------------------------------------------------------- #
# 纯 Python 小工具（不依赖 torch / CUDA，方便单独检查）
# --------------------------------------------------------------------------- #
def make_tokens(n: int, rng_seed: int) -> list[int]:
    """固定种子的随机 token id，不走 tokenizer，长度精确可控。"""
    seed(rng_seed)
    return [randint(0, 10000) for _ in range(n)]


def percentile(values: list[float], q: float) -> float | None:
    """线性插值分位数（和 numpy.percentile 的默认口径一致）。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def stats(values: list[float]) -> dict:
    """n / mean / p50 / p90 / p99 / max，单位保持调用方传入的单位。"""
    if not values:
        return {"n": 0, "mean_ms": None, "p50_ms": None, "p90_ms": None,
                "p99_ms": None, "max_ms": None}
    return {
        "n": len(values),
        "mean_ms": round(sum(values) / len(values), 3),
        "p50_ms": round(percentile(values, 50), 3),
        "p90_ms": round(percentile(values, 90), 3),
        "p99_ms": round(percentile(values, 99), 3),
        "max_ms": round(max(values), 3),
    }


class TraceWriter:
    """JSONL trace；``--trace`` 为空时是 no-op。"""

    def __init__(self, path: str | None, t0: float):
        self.path = path
        self.t0 = t0
        self._f = None
        self._n = 0
        if path:
            path = os.path.abspath(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._f = open(path, "w", encoding="utf-8")

    def emit(self, event: str, at_t: float | None = None, **fields) -> None:
        if self._f is None:
            return
        t = time.perf_counter() if at_t is None else at_t
        record = {"type": event, "t_ms": round((t - self.t0) * 1e3, 3)}
        record.update(fields)
        self._f.write(json.dumps(record, ensure_ascii=False,
                                 separators=(",", ":")) + "\n")
        self._n += 1
        # 崩溃时最多丢最后 512 行，避免每条都 flush 拖慢 step
        if self._n % 512 == 0:
            self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.flush()
            self._f.close()
            self._f = None


# --------------------------------------------------------------------------- #
# 观测数据结构
# --------------------------------------------------------------------------- #
@dataclass
class RequestStat:
    seq_id: int
    role: str                 # "short" | "long"
    prompt_tokens: int
    max_tokens: int
    submit_step: int          # 初始短请求是 -1；长请求是注入前的 engine step
    submit_t: float           # perf_counter
    submit_phase: str         # "warmup" | "steady"

    # 每个 completion token 的落地时间 / step / 该 step 属于哪个阶段
    token_t: list[float] = field(default_factory=list)
    token_step: list[int] = field(default_factory=list)
    token_phase: list[str] = field(default_factory=list)

    completion_tokens: int = 0
    finish_t: float | None = None
    finish_reason: str | None = None
    preempt_steps: list[int] = field(default_factory=list)

    @property
    def ttft_ms(self) -> float | None:
        if not self.token_t:
            return None
        return (self.token_t[0] - self.submit_t) * 1e3


@dataclass
class StepStat:
    index: int                # 0-based engine step
    steady_step: int | None   # 稳态后的第几步；warmup 为 None
    phase: str                # "warmup" | "steady"
    kind: str                 # "prefill" | "decode" | "empty"
    start_t: float
    end_t: float
    duration: float
    batch_tokens: int
    scheduled: list[int]
    arrivals: list[int]
    emitted: list[list[int]]  # [[seq_id, completion_index], ...]
    finished: list[int]
    running: int
    waiting: int
    free_blocks: int


# --------------------------------------------------------------------------- #
# 指标汇总
# --------------------------------------------------------------------------- #
def summarize(reqs: dict[int, RequestStat], steps: list[StepStat],
              stop_reason: str, arrivals_done: int,
              steady_start_step: int | None) -> dict:
    """从 request / step 记录里算短请求 TBT 和长请求 TTFT。"""
    step_by_idx = {s.index: s for s in steps}
    first_long_step = min((r.submit_step for r in reqs.values()
                           if r.role == "long"), default=None)

    buckets: dict[str, list[float]] = {
        "baseline": [],     # 第一条长请求之前、纯 decode 的 TBT
        "quiet": [],        # 所有不跨 prefill 步的 TBT
        "interrupted": [],  # 至少跨过一个 prefill/empty 步的 TBT
        "all": [],          # 稳态内所有未被抢占/重算污染的 TBT
    }
    preempted_samples = 0
    reprefill_samples = 0

    def interruption_reasons(step0: int, step1: int) -> set[str]:
        """(step0, step1] 里打断 decode 的原因。

        * arrival:   跨过了长请求的 prefill 步；
        * reprefill: 跨过了短请求自己的重新 prefill（KV 不足导致抢占），
                     这不是"新请求到达"造成的；
        * other:     空步之类。
        """
        reasons: set[str] = set()
        for i in range(step0 + 1, step1 + 1):
            step = step_by_idx.get(i)
            if step is None or step.kind == "decode":
                continue
            roles = {reqs[sid].role for sid in step.scheduled if sid in reqs}
            if "long" in roles:
                reasons.add("arrival")
            elif "short" in roles:
                reasons.add("reprefill")
            else:
                reasons.add("other")
        return reasons

    for req in reqs.values():
        if req.role != "short":
            continue
        for k in range(1, len(req.token_t)):
            if req.token_phase[k - 1] != "steady" or req.token_phase[k] != "steady":
                continue
            step0, step1 = req.token_step[k - 1], req.token_step[k]
            tbt_ms = (req.token_t[k] - req.token_t[k - 1]) * 1e3

            # 抢占会让请求回 waiting 重新 prefill，这一段不是"新请求到达"造成的
            if any(step0 < ps <= step1 for ps in req.preempt_steps):
                preempted_samples += 1
                continue

            reasons = interruption_reasons(step0, step1)
            if "reprefill" in reasons:
                reprefill_samples += 1
                continue

            buckets["all"].append(tbt_ms)
            if reasons:
                buckets["interrupted"].append(tbt_ms)
            else:
                buckets["quiet"].append(tbt_ms)
                if first_long_step is None or step1 < first_long_step:
                    buckets["baseline"].append(tbt_ms)

    short_tbt = {name: stats(values) for name, values in buckets.items()}
    all_stats = short_tbt["all"]
    base_stats = short_tbt["baseline"]
    short_tbt["p99_minus_p50_ms"] = (
        None if all_stats["p99_ms"] is None or all_stats["p50_ms"] is None
        else round(all_stats["p99_ms"] - all_stats["p50_ms"], 3)
    )
    short_tbt["p99_over_p50"] = (
        None if not all_stats["p99_ms"] or not all_stats["p50_ms"]
        else round(all_stats["p99_ms"] / all_stats["p50_ms"], 2)
    )
    short_tbt["baseline_p99_over_p50"] = (
        None if not base_stats["p99_ms"] or not base_stats["p50_ms"]
        else round(base_stats["p99_ms"] / base_stats["p50_ms"], 2)
    )
    short_tbt["preempted_samples_excluded"] = preempted_samples
    short_tbt["reprefill_samples_excluded"] = reprefill_samples

    long_rows = []
    ttfts = []
    for req in sorted(reqs.values(), key=lambda r: r.seq_id):
        if req.role != "long":
            continue
        ttft = req.ttft_ms
        first_step = req.token_step[0] if req.token_step else None
        prefill_steps = (
            sum(1 for i in range(req.submit_step, first_step + 1)
                if i in step_by_idx and step_by_idx[i].kind == "prefill")
            if first_step is not None else 0
        )
        if ttft is not None:
            ttfts.append(ttft)
        long_rows.append({
            "seq_id": req.seq_id,
            "submit_step": req.submit_step,
            "first_token_step": first_step,
            "prefill_steps": prefill_steps,
            "ttft_ms": None if ttft is None else round(ttft, 3),
            "completion_tokens": req.completion_tokens,
            "finish_reason": req.finish_reason,
        })

    steady_steps = [s for s in steps if s.phase == "steady"]
    kinds = Counter(s.kind for s in steady_steps)
    step_summary = {
        "total": len(steps),
        "steady": len(steady_steps),
        "steady_decode": kinds["decode"],
        "steady_prefill": kinds["prefill"],
        "steady_empty": kinds["empty"],
        "decode_ms": stats([s.duration * 1e3 for s in steady_steps
                            if s.kind == "decode"]),
        "prefill_ms": stats([s.duration * 1e3 for s in steady_steps
                             if s.kind == "prefill"]),
    }

    finished = [r for r in reqs.values() if r.finish_reason is not None]
    return {
        "stop_reason": stop_reason,
        "steady_start_step": steady_start_step,
        "steady_steps": len(steady_steps),
        "arrivals_done": arrivals_done,
        "requests": {
            "total": len(reqs),
            "short": sum(1 for r in reqs.values() if r.role == "short"),
            "long": sum(1 for r in reqs.values() if r.role == "long"),
            "finished": len(finished),
            "unfinished": [r.seq_id for r in reqs.values()
                           if r.finish_reason is None],
            "preempted": sorted({r.seq_id for r in reqs.values() if r.preempt_steps}),
        },
        "steps": step_summary,
        "short_tbt": short_tbt,
        "long_ttft": {
            "samples_ms": [round(v, 3) for v in ttfts],
            **stats(ttfts),
        },
        "long_requests": long_rows,
    }


def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"


def print_summary(summary: dict, args) -> None:
    st = summary["steps"]
    tbt = summary["short_tbt"]
    print("\n" + "=" * 78)
    print("mixed_long_short 结果")
    print("=" * 78)
    print(f"  稳态起始 step: {summary['steady_start_step']} | "
          f"稳态步数: {summary['steady_steps']} | "
          f"长请求已注入: {summary['arrivals_done']}/{args.num_long} | "
          f"结束原因: {summary['stop_reason']}")
    print(f"  稳态 step: decode={st['steady_decode']} prefill={st['steady_prefill']} "
          f"empty={st['steady_empty']}")
    print(f"  decode step 耗时: p50={_fmt(st['decode_ms']['p50_ms'], 'ms')} "
          f"p99={_fmt(st['decode_ms']['p99_ms'], 'ms')} "
          f"| prefill step 耗时: p50={_fmt(st['prefill_ms']['p50_ms'], 'ms')} "
          f"p99={_fmt(st['prefill_ms']['p99_ms'], 'ms')}")

    print("\n  短请求 TBT（token 落地间隔，ms）")
    for name, label in (("baseline", "baseline（首条长请求前，纯 decode）"),
                        ("quiet", "quiet（所有未跨 prefill 的间隔）"),
                        ("interrupted", "interrupted（跨过 prefill 步）"),
                        ("all", "all（稳态内全部，排除被抢占的样本）")):
        s = tbt[name]
        print(f"    {label}: n={s['n']} "
              f"p50={_fmt(s['p50_ms'])} p99={_fmt(s['p99_ms'])} "
              f"max={_fmt(s['max_ms'])}")
    print(f"    -> all: p99 - p50 = {_fmt(tbt['p99_minus_p50_ms'], 'ms')}，"
          f"p99 / p50 = {_fmt(tbt['p99_over_p50'], 'x')}")
    print(f"    -> baseline: p99 / p50 = {_fmt(tbt['baseline_p99_over_p50'], 'x')}")
    if tbt["preempted_samples_excluded"] or tbt["reprefill_samples_excluded"]:
        print(f"    ! 有 {tbt['preempted_samples_excluded']} 个 TBT 样本因请求被抢占、"
              f"{tbt['reprefill_samples_excluded']} 个因短请求 re-prefill 被排除；"
              f"说明 KV cache 不够（这类空档不是长请求到达造成的）")

    print("\n  长请求 TTFT（add_request -> 第一个 token，ms）")
    for row in summary["long_requests"]:
        print(f"    seq{row['seq_id']}: submit_step={row['submit_step']} "
              f"first_token_step={row['first_token_step']} "
              f"prefill_steps={row['prefill_steps']} "
              f"TTFT={_fmt(row['ttft_ms'])}ms "
              f"finish={row['finish_reason']}")
    lt = summary["long_ttft"]
    print(f"    -> n={lt['n']} p50={_fmt(lt['p50_ms'])}ms "
          f"p99={_fmt(lt['p99_ms'])}ms max={_fmt(lt['max_ms'])}ms")

    req = summary["requests"]
    print(f"\n  请求: {req['finished']}/{req['total']} 完成"
          f"（short={req['short']} long={req['long']}），"
          f"未完成={len(req['unfinished'])}，"
          f"被抢占={len(req['preempted'])}")
    if req["preempted"]:
        print(f"    ! 被抢占的 seq: {req['preempted'][:10]}"
              f"{' ...' if len(req['preempted']) > 10 else ''}")
    print("=" * 78, flush=True)


# --------------------------------------------------------------------------- #
# 引擎初始化辅助
# --------------------------------------------------------------------------- #
def _install_kv_block_limit(kv_blocks: int) -> None:
    """把 allocate_kv_cache 换成固定块数（和 bench.edge 里的做法一致）。

    默认显存预算下 Qwen3-0.6B 的 KV cache 很大，想压出抢占时用 --kv-blocks
    显式给一个很小的值即可。只影响当前进程。
    """
    import torch
    from inference_from_zero.engine.model_runner import ModelRunner

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        config.num_kvcache_blocks = kv_blocks

        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim",
                           hf_config.hidden_size // hf_config.num_attention_heads)

        self.kv_cache = torch.empty(
            2, hf_config.num_hidden_layers, kv_blocks,
            self.block_size, num_kv_heads, head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    ModelRunner.allocate_kv_cache = allocate_kv_cache


# --------------------------------------------------------------------------- #
# 场景：mixed_long_short
# --------------------------------------------------------------------------- #
def run_mixed_long_short(args) -> int:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"import torch 失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("需要 CUDA：这个 bench 会真的跑模型（当前 torch 看不到 GPU）",
              file=sys.stderr)
        return 2
    if not os.path.isdir(args.model):
        print(f"找不到模型目录: {args.model}\n"
              f"用 --model /path/to/Qwen3-0.6B 指定，或改 DEFAULT_MODEL",
              file=sys.stderr)
        return 2

    if args.kv_blocks > 0:
        _install_kv_block_limit(args.kv_blocks)

    from inference_from_zero import LLM, SamplingParams

    t0 = time.perf_counter()
    trace = TraceWriter(args.trace, t0)
    llm = None

    try:
        init_t = time.perf_counter()
        llm = LLM(
            args.model,
            enforce_eager=True,  # 手写 attention 有动态 gather，录不了 CUDA Graph
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
        )
        init_s = time.perf_counter() - init_t

        sched = llm.scheduler
        bm = sched.block_manager
        block_size = bm.block_size
        num_blocks = len(bm.blocks)
        capacity = num_blocks * block_size

        trace.emit("meta", at_t=time.perf_counter(),
                   scenario="mixed_long_short",
                   model=args.model,
                   args=vars(args),
                   init_s=round(init_s, 3),
                   max_model_len=args.max_model_len,
                   max_num_batched_tokens=args.max_num_batched_tokens,
                   max_num_seqs=args.max_num_seqs,
                   block_size=block_size,
                   num_kv_blocks=num_blocks,
                   kv_capacity_tokens=capacity,
                   torch=torch.__version__)

        # 估算峰值块需求，提前提示"会不会触发抢占"
        short_blocks = ceil((args.short_prompt_len + args.short_max_tokens) / block_size)
        long_blocks = ceil((args.long_prompt_len + args.long_max_tokens) / block_size)
        need_blocks = args.num_short * short_blocks + args.num_long * long_blocks
        print(f"  引擎初始化 {init_s:.1f}s | max_model_len={args.max_model_len} "
              f"max_num_batched_tokens={args.max_num_batched_tokens} "
              f"max_num_seqs={args.max_num_seqs}")
        print(f"  KV cache: {num_blocks} blocks x {block_size} = {capacity} tokens"
              f"{'（bench 强制指定）' if args.kv_blocks > 0 else ''}")
        print(f"  峰值块需求估计: {need_blocks} blocks "
              f"(short={args.num_short}x{short_blocks}, long={args.num_long}x{long_blocks})"
              f"{' -> 可能触发 preemption' if need_blocks > num_blocks else ''}")
        if need_blocks > num_blocks:
            print("  ! KV cache 不足以同时容纳全部请求，短请求可能被抢占并重新 prefill；"
                  "TBT 里会看到重算空档（bench 会把这类样本单独排除）")

        reqs: dict[int, RequestStat] = {}
        seqs: dict[int, object] = {}
        steps: list[StepStat] = []

        state = {"step": 0, "phase": "warmup"}

        # 统计抢占：短请求被抢占后重新 prefill，不是"新请求到达"的影响
        orig_preempt = sched.preempt

        def counted_preempt(seq):
            req = reqs.get(seq.seq_id)
            if req is not None:
                req.preempt_steps.append(state["step"])
            trace.emit("preempt", step=state["step"], phase=state["phase"],
                       seq_id=seq.seq_id,
                       role=None if req is None else req.role)
            return orig_preempt(seq)

        sched.preempt = counted_preempt

        def enqueue(role: str, prompt_len: int, max_tokens: int, rng_seed: int,
                    submit_step: int, phase: str) -> RequestStat:
            prompt = make_tokens(prompt_len, rng_seed)
            sampling = SamplingParams(temperature=0.6, max_tokens=max_tokens,
                                      ignores_eos=True)
            submit_t = time.perf_counter()
            llm.add_request(prompt, sampling)
            seq = sched.waiting[-1]
            req = RequestStat(
                seq_id=seq.seq_id,
                role=role,
                prompt_tokens=prompt_len,
                max_tokens=max_tokens,
                submit_step=submit_step,
                submit_t=submit_t,
                submit_phase=phase,
            )
            reqs[seq.seq_id] = req
            seqs[seq.seq_id] = seq
            trace.emit("request", at_t=submit_t,
                       seq_id=seq.seq_id, role=role,
                       prompt_tokens=prompt_len, max_tokens=max_tokens,
                       submit_step=submit_step, phase=phase,
                       status=seq.status.name)
            return req

        # ---- 先放短请求，跑到全部进入 decode ---------------------------------
        for i in range(args.num_short):
            enqueue("short", args.short_prompt_len, args.short_max_tokens,
                    args.seed + i, submit_step=-1, phase="warmup")
        short_ids = set(reqs)

        def warmup_done() -> bool:
            if sched.waiting:
                return False
            if len(sched.running) != args.num_short:
                return False
            return all(len(reqs[sid].token_t) >= 1 for sid in short_ids)

        arrival_steps = [args.first_arrival_step + i * args.long_every_steps
                         for i in range(args.num_long)]
        next_arrival = 0
        steady = False
        steady_step = 0
        steady_start_step: int | None = None
        stop_reason = "finished"
        engine_step = 0

        print(f"\n  short x{args.num_short}: prompt={args.short_prompt_len} tok, "
              f"max_tokens={args.short_max_tokens}")
        print(f"  long  x{args.num_long}: prompt={args.long_prompt_len} tok, "
              f"max_tokens={args.long_max_tokens}, "
              f"稳态 step 索引 {arrival_steps}（0-based）注入")
        print("  开始跑 decode...\n", flush=True)

        while not llm.is_finished():
            if engine_step >= args.max_steps:
                stop_reason = "max_steps"
                break

            phase = "steady" if steady else "warmup"
            state["step"] = engine_step
            state["phase"] = phase
            arrivals: list[int] = []
            steady_step_now = steady_step if steady else None

            # 在 step 开始前注入长请求：它的 prefill 就发生在本步
            if (steady and next_arrival < len(arrival_steps)
                    and steady_step == arrival_steps[next_arrival]):
                req = enqueue(
                    "long", args.long_prompt_len, args.long_max_tokens,
                    args.seed + 1000 + next_arrival,
                    submit_step=engine_step, phase="steady",
                )
                arrivals.append(req.seq_id)
                next_arrival += 1

            prev_records = len(llm.step_records)
            torch.cuda.synchronize()
            start_t = time.perf_counter()
            outputs, num_tokens = llm.step()
            torch.cuda.synchronize()
            end_t = time.perf_counter()
            duration = end_t - start_t

            if num_tokens > 0:
                kind = "prefill"
            elif num_tokens < 0:
                kind = "decode"
            else:
                kind = "empty"

            if len(llm.step_records) > prev_records:
                scheduled = list(llm.step_records[-1].seq_ids)
            else:
                scheduled = []

            # 哪些 seq 在这一步落了新 token（同一 step 的 token 共享 end_t）
            emitted: list[list[int]] = []
            for sid, seq in seqs.items():
                n = seq.num_completion_tokens
                req = reqs[sid]
                while req.completion_tokens < n:
                    req.completion_tokens += 1
                    req.token_t.append(end_t)
                    req.token_step.append(engine_step)
                    req.token_phase.append(phase)
                    emitted.append([sid, req.completion_tokens])

            finished_ids: list[int] = []
            for seq_id, token_ids, finish_reason in outputs:
                req = reqs.get(seq_id)
                finished_ids.append(seq_id)
                if req is not None:
                    req.finish_t = end_t
                    req.finish_reason = finish_reason
                    req.completion_tokens = max(req.completion_tokens, len(token_ids))
                    trace.emit("finish", at_t=end_t,
                               seq_id=seq_id, role=req.role,
                               completion_tokens=len(token_ids),
                               finish_reason=finish_reason,
                               preemptions=len(req.preempt_steps))

            step = StepStat(
                index=engine_step,
                steady_step=steady_step_now,
                phase=phase,
                kind=kind,
                start_t=start_t,
                end_t=end_t,
                duration=duration,
                batch_tokens=abs(num_tokens),
                scheduled=scheduled,
                arrivals=arrivals,
                emitted=emitted,
                finished=finished_ids,
                running=len(sched.running),
                waiting=len(sched.waiting),
                free_blocks=len(bm.free_block_ids),
            )
            steps.append(step)
            trace.emit("step", at_t=end_t,
                       step=engine_step,
                       steady_step=steady_step_now,
                       phase=phase,
                       kind=kind,
                       start_ms=round((start_t - t0) * 1e3, 3),
                       duration_ms=round(duration * 1e3, 3),
                       batch_tokens=abs(num_tokens),
                       scheduled=scheduled,
                       arrivals=arrivals,
                       emitted=emitted,
                       finished=finished_ids,
                       running=step.running,
                       waiting=step.waiting,
                       free_blocks=step.free_blocks)

            if not steady:
                if warmup_done():
                    steady = True
                    steady_step = 0
                    steady_start_step = engine_step + 1
                    trace.emit("steady_start", at_t=end_t,
                               next_step=steady_start_step,
                               running=len(sched.running),
                               free_blocks=len(bm.free_block_ids))
                    print(f"  [step {engine_step}] warmup 结束，"
                          f"{args.num_short} 条短请求全部进入 decode；"
                          f"从 step {steady_start_step} 开始计稳态", flush=True)
            else:
                steady_step += 1

            engine_step += 1

        if not steady:
            stop_reason = "warmup_not_reached"
            trace.emit("end", step=engine_step, stop_reason=stop_reason)
            print("\n  FAIL: 没有跑到稳态 decode（短请求可能在 warmup 就结束了）",
                  file=sys.stderr)
            return 1

        trace.emit("end", step=engine_step, stop_reason=stop_reason,
                   arrivals_done=next_arrival)

        summary = summarize(reqs, steps, stop_reason, next_arrival,
                            steady_start_step)
        summary_event = {"scenario": "mixed_long_short", **summary}
        trace.emit("summary", **summary_event)
        print_summary(summary, args)

        # 基本正确性：所有请求都要有结果（被抢占也应该最终跑完）
        if summary["requests"]["unfinished"]:
            print("  FAIL: 有请求没拿到结果", file=sys.stderr)
            return 1
        if next_arrival < args.num_long:
            print(f"  FAIL: 只注入了 {next_arrival}/{args.num_long} 条长请求"
                  f"（短请求可能提前跑完，或 --max-steps 太小）", file=sys.stderr)
            return 1
        missing_ttft = [row["seq_id"] for row in summary["long_requests"]
                        if row["ttft_ms"] is None]
        if missing_ttft:
            print(f"  FAIL: 长请求没有拿到第一个 token: {missing_ttft}"
                  f"（检查 KV cache 是否装得下 {args.long_prompt_len} tok 的 prompt）",
                  file=sys.stderr)
            return 1
        if stop_reason != "finished":
            print(f"  FAIL: 结束原因不是 finished（{stop_reason}）", file=sys.stderr)
            return 1
        return 0

    except Exception as exc:  # noqa: BLE001 - bench 自身失败也要落 trace
        print(f"\n未捕获异常: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        trace.emit("error", error=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        trace.close()


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m bench.run",
        description="场景化 bench：混合长短请求下的 TBT / TTFT",
    )
    p.add_argument("--scenario", default="mixed_long_short",
                   choices=["mixed_long_short"],
                   help="跑哪个场景（目前只有 mixed_long_short）")
    p.add_argument("--trace", default=None,
                   help="JSONL trace 输出路径，例如 out/mixed.jsonl")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"本地模型目录（默认 {DEFAULT_MODEL}）")
    p.add_argument("--max-model-len", type=int, default=4096,
                   help="单序列最大长度")
    p.add_argument("--max-num-batched-tokens", type=int, default=16384,
                   help="一个 batch 的最大 token 数")
    p.add_argument("--max-num-seqs", type=int, default=512,
                   help="一个 batch 的最大序列数")
    p.add_argument("--kv-blocks", type=int, default=0,
                   help="强制指定 KV 块数（0 = 用引擎按显存算的默认值）")

    p.add_argument("--num-short", type=int, default=128, help="短请求条数")
    p.add_argument("--short-prompt-len", type=int, default=128,
                   help="短请求 prompt 长度")
    p.add_argument("--short-max-tokens", type=int, default=256,
                   help="短请求最多生成多少 token（要覆盖到所有长请求注入之后）")

    p.add_argument("--long-prompt-len", type=int, default=3072,
                   help="长请求 prompt 长度")
    p.add_argument("--long-max-tokens", type=int, default=64,
                   help="长请求最多生成多少 token")
    p.add_argument("--long-every-steps", type=int, default=20,
                   help="稳态后每隔多少个 engine step 注入一条长请求")
    p.add_argument("--num-long", type=int, default=5, help="注入几条长请求")
    p.add_argument("--first-arrival-step", type=int, default=0,
                   help="第一条长请求在稳态第几步注入（0-based 索引）；"
                        "0 表示等于 --long-every-steps")

    p.add_argument("--max-steps", type=int, default=10000,
                   help="步数上限，防死循环")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    args = p.parse_args(argv)

    if args.first_arrival_step == 0:
        args.first_arrival_step = args.long_every_steps

    if args.num_short < 1 or args.num_long < 1:
        p.error("--num-short / --num-long 必须 >= 1")
    if args.short_prompt_len < 1 or args.long_prompt_len < 1:
        p.error("--short-prompt-len / --long-prompt-len 必须 >= 1")
    if args.short_max_tokens < 2:
        p.error("--short-max-tokens 必须 >= 2（warmup 会先消耗第一个 token）")
    if args.long_max_tokens < 1:
        p.error("--long-max-tokens 必须 >= 1")
    if args.long_every_steps < 1 or args.first_arrival_step < 1:
        p.error("--long-every-steps / --first-arrival-step 必须 >= 1")
    if args.num_short + args.num_long > args.max_num_seqs:
        p.error(f"num_short + num_long ({args.num_short + args.num_long}) "
                f"超过 --max-num-seqs ({args.max_num_seqs})")
    if args.short_prompt_len + args.short_max_tokens > args.max_model_len:
        p.error(f"short prompt + max_tokens ({args.short_prompt_len}+"
                f"{args.short_max_tokens}) 超过 --max-model-len {args.max_model_len}")
    if args.long_prompt_len + args.long_max_tokens > args.max_model_len:
        p.error(f"long prompt + max_tokens ({args.long_prompt_len}+"
                f"{args.long_max_tokens}) 超过 --max-model-len {args.max_model_len}")
    if args.max_steps < 1:
        p.error("--max-steps 必须 >= 1")

    last_arrival = args.first_arrival_step + (args.num_long - 1) * args.long_every_steps
    if args.short_max_tokens <= last_arrival:
        print(f"  ! --short-max-tokens={args.short_max_tokens} 不足以撑到最后一条"
              f"长请求（稳态第 {last_arrival} 步）；短请求可能提前跑完，"
              f"导致部分长请求不会被注入", file=sys.stderr)
    if args.long_prompt_len > args.max_num_batched_tokens:
        print(f"  ! long prompt ({args.long_prompt_len}) > "
              f"--max-num-batched-tokens ({args.max_num_batched_tokens})，"
              f"长请求的 prefill 会被切成多步（TBT 会出现多个尖峰）",
              file=sys.stderr)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.scenario == "mixed_long_short":
        return run_mixed_long_short(args)
    raise AssertionError(args.scenario)


if __name__ == "__main__":
    sys.exit(main())
