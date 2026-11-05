"""边界用例 bench：把调度器 / KV cache 的极端情况摊开看。

用法::

    python -m bench.edge --case a                  # 短请求、比 KV cache 总容量大 1 个 token 的请求、短请求
    python -m bench.edge --case b                  # 三个请求跑到第 4 步，running.remove 掉一个（没有 abort 接口）
    python -m bench.edge --case c --max-tokens 1   # prompt 恰好等于 max_model_len=4096
    python -m bench.edge --case c --max-tokens 2
    python -m bench.edge --case all                # 三个 case 各起一个子进程跑

结论标记：

* ``[OK]``    行为符合预期
* ``[EDGE]``  能跑但行为可疑（引擎缺校验 / 缺清理，bench 只负责记录，不算退出失败）
* ``[FAIL]``  bench 自己的前置假设不成立或进程级异常（退出码 1）

设计取舍：三个 case 都**手动驱动** ``LLMEngine.step()`` 而不是调 ``generate()``，
因为 generate() 遇到未调度完的请求会直接 ``assert`` / 丢掉结果，bench 需要看清
每一步调度器到底做了什么。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from math import ceil
from random import randint, seed

import torch

from inference_from_zero import LLM, SamplingParams
from inference_from_zero.engine.model_runner import ModelRunner

DEFAULT_MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

CASE_TITLES = {
    "a": "短请求 → 比 KV cache 总容量大 1 token 的请求 → 短请求",
    "b": "三个请求跑到第 4 步，从 running 里硬删一个（没有 abort 接口）",
    "c": "prompt 恰好等于 max_model_len，看边界上的 block 分配",
}


# --------------------------------------------------------------------------- #
# 强制 KV 块数
#
# case a 需要一个"刚好装不下的请求"，即 prompt 长度 = 总容量 + 1 token。
# 但本仓库的 num_kvcache_blocks 是按显卡剩余显存算的（Qwen3-0.6B 每块约 29MB，
# 24G 卡上有几百块 ≈ 十几万 token），远超 max_model_len，构造不出这种请求。
# 所以这里把 allocate_kv_cache 包一层：本 benchmark 显式指定块数，
# 逻辑与 model_runner.ModelRunner.allocate_kv_cache 保持一致，只是块数固定。
# --------------------------------------------------------------------------- #
_kv_block_limit: int | None = None
_orig_allocate_kv_cache = ModelRunner.allocate_kv_cache


def _patched_allocate_kv_cache(self):
    if _kv_block_limit is None:
        return _orig_allocate_kv_cache(self)

    config = self.config
    hf_config = config.hf_config
    config.num_kvcache_blocks = _kv_block_limit

    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

    self.kv_cache = torch.empty(
        2, hf_config.num_hidden_layers, _kv_block_limit,
        self.block_size, num_kv_heads, head_dim,
    )
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1


ModelRunner.allocate_kv_cache = _patched_allocate_kv_cache


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
class Case:
    """打印 case 标题 + 逐条结论 + 汇总；FAIL 会让进程退出码变 1。"""

    def __init__(self, name: str, title: str):
        self.name = name
        self.title = title
        self.counts = {"OK": 0, "EDGE": 0, "FAIL": 0}
        print(f"\n{'=' * 78}\ncase {name}: {title}\n{'=' * 78}")

    def _emit(self, tag: str, msg: str):
        self.counts[tag] += 1
        print(f"  [{tag}] {msg}", flush=True)

    def info(self, msg: str):
        print(f"  · {msg}", flush=True)

    def ok(self, msg: str):
        self._emit("OK", msg)

    def edge(self, msg: str):
        self._emit("EDGE", msg)

    def fail(self, msg: str):
        self._emit("FAIL", msg)

    def summary(self) -> bool:
        c = self.counts
        print(f"  -> {c['OK']} OK / {c['EDGE']} EDGE / {c['FAIL']} FAIL", flush=True)
        return c["FAIL"] == 0


def exc_text(exc: BaseException) -> str:
    msg = str(exc)
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def make_tokens(n: int, rng_seed: int) -> list[int]:
    """固定种子的随机 token id，不走 tokenizer，长度精确可控。"""
    seed(rng_seed)
    return [randint(0, 10000) for _ in range(n)]


def make_engine(args, kv_blocks: int | None = None) -> LLM:
    global _kv_block_limit
    _kv_block_limit = kv_blocks

    t = time.perf_counter()
    llm = LLM(
        args.model,
        enforce_eager=True,  # 手写 attention 有动态 gather，录不了 CUDA Graph
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    torch.cuda.synchronize()
    init_s = time.perf_counter() - t

    bm = llm.scheduler.block_manager
    cfg = llm.model_runner.config
    print(f"  · 引擎初始化 {init_s:.1f}s | max_model_len={cfg.max_model_len} "
          f"max_num_batched_tokens={cfg.max_num_batched_tokens} max_num_seqs={cfg.max_num_seqs}", flush=True)
    print(f"  · KV cache: {len(bm.blocks)} blocks x {bm.block_size} = "
          f"{len(bm.blocks) * bm.block_size} tokens"
          f"{'' if kv_blocks is None else '（bench 强制指定）'}", flush=True)
    return llm


def one_step(llm: LLM):
    """跑一步调度 + 前向 + 后处理，返回 (outputs, num_tokens, 墙钟秒)。"""
    t = time.perf_counter()
    outputs, num_tokens = llm.step()
    return outputs, num_tokens, time.perf_counter() - t


def describe_step(idx: int, num_tokens: int, dt: float) -> str:
    kind = "prefill" if num_tokens > 0 else "decode"
    return f"step {idx}: {kind} work={abs(num_tokens)} tok {dt * 1e3:.1f}ms"


def drain(llm: LLM, rep: Case, finished: dict, start_step: int = 0,
          max_steps: int = 4096) -> int:
    """一直 step 到 is_finished() 或者步数用完，把完成结果收进 finished。"""
    step = start_step
    while not llm.is_finished():
        if step - start_step >= max_steps:
            rep.edge(f"{max_steps} 步还没跑完，疑似队列卡死")
            break
        outputs, num_tokens, dt = one_step(llm)
        step += 1
        rep.info(describe_step(step, num_tokens, dt))
        for seq_id, token_ids in outputs:
            finished[seq_id] = token_ids
            rep.info(f"  seq{seq_id} 完成: completion={len(token_ids)} tok")
    return step


# --------------------------------------------------------------------------- #
# case a
# --------------------------------------------------------------------------- #
def case_a(args, rep: Case) -> None:
    llm = make_engine(args, kv_blocks=args.kv_blocks)
    bm = llm.scheduler.block_manager
    capacity = len(bm.blocks) * bm.block_size

    oversize_len = capacity + 1  # 比 KV cache 总容量大 1 个 token
    oversize_blocks = ceil(oversize_len / bm.block_size)
    rep.info(f"超容量请求: {oversize_len} tokens = 容量({capacity}) + 1，"
             f"需要 {oversize_blocks} blocks > 可用 {len(bm.blocks)}")
    if oversize_len > args.max_model_len:
        rep.fail(f"{oversize_len} > max_model_len({args.max_model_len})，构造不出这个请求；"
                 f"请调小 --kv-blocks（当前 {args.kv_blocks}）")
        return
    if args.short_len + args.short_max_tokens > capacity:
        rep.fail(f"短请求({args.short_len}+{args.short_max_tokens}) 都装不下容量 {capacity}")
        return

    prompt_lens = [args.short_len, oversize_len, args.short_len]
    prompts = [make_tokens(n, args.seed + i) for i, n in enumerate(prompt_lens)]
    sps = [SamplingParams(temperature=0.6, max_tokens=args.short_max_tokens, ignores_eos=True)
           for _ in prompt_lens]
    for i, (prompt, sp) in enumerate(zip(prompts, sps)):
        try:
            llm.add_request(prompt, sp)
        except Exception as exc:  # noqa: BLE001 - 将来加了长度校验就走这里
            if i == 1:
                rep.ok(f"add_request 阶段就拒绝了超容量请求（理想行为）: {exc_text(exc)}")
            else:
                rep.fail(f"add_request 拒绝了第 {i} 个请求: {exc_text(exc)}")
            return

    seqs = list(llm.scheduler.waiting)
    rep.info("入队顺序: " + " -> ".join(f"seq{s.seq_id}({n} tok)" for s, n in zip(seqs, prompt_lens)))
    free_before = len(bm.free_block_ids)

    finished: dict[int, list[int]] = {}
    crash: Exception | None = None
    steps = 0
    while not llm.is_finished() and steps < args.max_steps:
        try:
            outputs, num_tokens, dt = one_step(llm)
        except Exception as exc:  # noqa: BLE001 - 这里就是要捕获调度器的 assert
            crash = exc
            break
        steps += 1
        rep.info(describe_step(steps, num_tokens, dt))
        for seq_id, token_ids in outputs:
            finished[seq_id] = token_ids
            rep.info(f"  seq{seq_id} 完成: completion={len(token_ids)} tok")

    rep.info(f"实际跑了 {steps} 步；free blocks {free_before} -> {len(bm.free_block_ids)}")
    for tag, seq, plen in zip(("A(短)", "B(超容量)", "C(短)"), seqs, prompt_lens):
        where = ("waiting" if seq in llm.scheduler.waiting
                 else "running" if seq in llm.scheduler.running
                 else "不在队列里")
        rep.info(f"{tag} seq{seq.seq_id}: prompt={plen} tok completion={seq.num_completion_tokens} tok "
                 f"status={seq.status.name} {where}")

    # 第一条短请求应该能正常跑完
    if seqs[0].is_finished and len(finished.get(seqs[0].seq_id, [])) == args.short_max_tokens:
        rep.ok(f"第一条短请求正常跑完（{args.short_max_tokens} tok）")
    else:
        rep.fail("第一条短请求没有正常跑完")

    # 超容量请求：can_allocate 永远返回 -1
    if seqs[1].num_completion_tokens == 0 and seqs[1] in llm.scheduler.waiting:
        rep.edge(f"超容量请求 seq{seqs[1].seq_id} 永远拿不到 block，"
                 f"scheduler 在 waiting 队头直接 break（没有拒绝 / 报错路径）")

    # 队头阻塞：后面的短请求本来装得下，但根本没被看到
    if seqs[2].status.name == "WAITING" and seqs[2].num_completion_tokens == 0:
        rep.edge(f"第二条短请求 seq{seqs[2].seq_id} 被队头阻塞：prefill 循环遇到装不下的序列就 break，"
                 f"不再往后看")

    if crash is not None:
        rep.edge(f"step() 抛 {exc_text(crash)} —— 没有「prompt 超出 KV 容量」的提前拒绝，"
                 f"generate() 会直接崩掉（而不是 ValueError）")
    elif llm.is_finished() and not seqs[1].is_finished:
        rep.edge("队列被判定为空，但超容量请求仍未完成（请求被静默丢弃）")
    elif steps >= args.max_steps:
        rep.edge(f"{args.max_steps} 步上限到了还没结束（卡死）")


# --------------------------------------------------------------------------- #
# case b
# --------------------------------------------------------------------------- #
def case_b(args, rep: Case) -> None:
    llm = make_engine(args)
    bm = llm.scheduler.block_manager

    sp = SamplingParams(temperature=0.6, max_tokens=args.b_max_tokens, ignores_eos=True)
    prompts = [make_tokens(args.b_prompt_len, args.seed * 10 + i) for i in range(3)]
    for prompt in prompts:
        llm.add_request(prompt, sp)

    seqs = list(llm.scheduler.waiting)
    rep.info("三个请求: " + ", ".join(f"seq{s.seq_id}(prompt={args.b_prompt_len} tok)" for s in seqs))

    finished: dict[int, list[int]] = {}
    try:
        for i in range(args.steps_before_remove):
            outputs, num_tokens, dt = one_step(llm)
            rep.info(describe_step(i + 1, num_tokens, dt))
            for seq_id, token_ids in outputs:
                finished[seq_id] = token_ids
                rep.info(f"  seq{seq_id} 完成: completion={len(token_ids)} tok")
    except Exception as exc:  # noqa: BLE001
        rep.fail(f"前 {args.steps_before_remove} 步就异常了: {exc_text(exc)}")
        traceback.print_exc()
        return

    if len(llm.scheduler.running) != 3:
        rep.fail(f"第 {args.steps_before_remove} 步后 running 里只剩 {len(llm.scheduler.running)} 条，"
                 f"没法做删除实验")
        return

    removed = seqs[1]
    free_before = len(bm.free_block_ids)
    blocks_before = list(removed.block_table)
    llm.scheduler.running.remove(removed)  # 引擎没有 abort 接口，只能这样模拟
    rep.info(f"running.remove(seq{removed.seq_id})，未 deallocate、未改 status（引擎没有 abort 接口）")

    try:
        steps = drain(llm, rep, finished, start_step=args.steps_before_remove)
    except Exception as exc:  # noqa: BLE001
        rep.fail(f"删除后继续跑异常: {exc_text(exc)}")
        traceback.print_exc()
        return

    rep.info(f"共 {steps} 步；free blocks {free_before} -> {len(bm.free_block_ids)} / 总量 {len(bm.blocks)}")
    rep.info(f"被删的 seq{removed.seq_id}: tokens={removed.num_tokens} "
             f"completion={removed.num_completion_tokens} status={removed.status.name} "
             f"block_table={removed.block_table}")

    survivors_ok = all(len(finished.get(s.seq_id, [])) == args.b_max_tokens
                       for s in (seqs[0], seqs[2]))
    if survivors_ok:
        rep.ok(f"另外两条请求正常跑完（各 {args.b_max_tokens} tok）")
    else:
        rep.fail("另外两条请求没有正常跑完")

    # 结果少了但没有任何报错
    if len(finished) == 2 and removed.seq_id not in finished:
        rep.edge(f"3 条请求只拿到 2 条结果，且无异常、无 abort 信号："
                 f"generate() 会静默返回少一条的列表")

    # 状态机不自洽
    if not removed.is_finished and removed not in llm.scheduler.running:
        rep.edge(f"seq{removed.seq_id} status={removed.status.name} 但已不在 running 里；"
                 f"is_finished() 只看队列，会认为全部结束")

    # block 泄漏
    leaked = [b for b in blocks_before if b in bm.used_block_ids]
    if leaked:
        rep.edge(f"seq{removed.seq_id} 的 {len(leaked)} 个 block 没归还"
                 f"（ref_count={[bm.blocks[b].ref_count for b in leaked]}，"
                 f"free blocks 结束时 {len(bm.free_block_ids)} / {len(bm.blocks)}，"
                 f"正常应该是 {len(bm.blocks) - len(leaked)}）")
    else:
        rep.ok("被删序列的 block 已回收")


# --------------------------------------------------------------------------- #
# case c
# --------------------------------------------------------------------------- #
def case_c(args, rep: Case) -> None:
    llm = make_engine(args)
    bm = llm.scheduler.block_manager
    mm = llm.model_runner.config.max_model_len

    prompt_len = mm  # prompt 恰好等于 max_model_len
    if prompt_len % bm.block_size:
        rep.info(f"注意: prompt={prompt_len} 不是 block_size({bm.block_size}) 的整数倍")

    max_tokens = args.max_tokens
    sp = SamplingParams(temperature=0.6, max_tokens=max_tokens, ignores_eos=True)
    prompt = make_tokens(prompt_len, args.seed)
    try:
        llm.add_request(prompt, sp)
    except Exception as exc:  # noqa: BLE001 - 将来加了长度校验就走这里
        rep.edge(f"add_request 拒绝了 prompt == max_model_len 的请求: {exc_text(exc)}")
        return
    seq = llm.scheduler.waiting[-1]

    need_blocks = ceil((prompt_len + max_tokens) / bm.block_size)
    rep.info(f"prompt={prompt_len} tok == max_model_len({mm})，max_tokens={max_tokens}，"
             f"最终 {prompt_len + max_tokens} tok")
    rep.info(f"按最终长度需要 {need_blocks} blocks / 可用 {len(bm.blocks)} blocks；"
             f"prompt 本身占满 {ceil(prompt_len / bm.block_size)} blocks")
    if need_blocks > len(bm.blocks):
        rep.fail("KV cache 装不下这个 case（需要更多 blocks）")
        return

    finished: dict[int, list[int]] = {}
    crash: Exception | None = None
    step = 0
    while not llm.is_finished() and step < args.max_steps:
        try:
            outputs, num_tokens, dt = one_step(llm)
        except Exception as exc:  # noqa: BLE001
            crash = exc
            break
        step += 1
        detail = (f"{describe_step(step, num_tokens, dt)} | seq_len={seq.num_tokens} "
                  f"completion={seq.num_completion_tokens}/{max_tokens} "
                  f"blocks={'-' if seq.is_finished else len(seq.block_table)} "
                  f"status={seq.status.name}")
        rep.info(detail)
        for seq_id, token_ids in outputs:
            finished[seq_id] = token_ids
            rep.info(f"  seq{seq_id} 完成: completion={len(token_ids)} tok")

    if crash is not None:
        rep.fail(f"prefill/decode 异常: {exc_text(crash)}")
        traceback.print_exc()
        return
    if step == 0:
        rep.fail("一步都没跑")
        return

    got = finished.get(seq.seq_id, [])
    if len(got) == max_tokens:
        rep.ok(f"生成了 {max_tokens} tok（prefill 一步 + decode {max_tokens - 1} 步）")
    else:
        rep.fail(f"期望 {max_tokens} tok，实际 {len(got)} tok")

    final_len = seq.num_tokens
    rep.info(f"最终序列长度 {final_len} tok（prompt {prompt_len} + completion {final_len - prompt_len}）")
    if final_len > mm:
        rep.edge(f"最终长度超过 max_model_len 共 {final_len - mm} tok：引擎不校验单序列长度上限，"
                 f"只在 warmup / CUDA Graph 里把 max_model_len 当预算用")
    else:
        rep.ok("没有超过 max_model_len")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m bench.edge",
        description="调度器 / KV cache 的边界用例 bench",
    )
    p.add_argument("--case", default="all", choices=["a", "b", "c", "all"], help="跑哪个 case")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"本地模型目录（默认 {DEFAULT_MODEL}）")
    p.add_argument("--max-model-len", type=int, default=4096,
                   help="单序列最大长度；case c 的 prompt 长度就取这个值")
    p.add_argument("--max-num-batched-tokens", type=int, default=16384,
                   help="一个 batch 的最大 token 数")
    p.add_argument("--max-tokens", type=int, default=1, help="case c 生成多少 token")
    p.add_argument("--kv-blocks", type=int, default=8,
                   help="case a 强制指定的 KV 块数（0 = 用引擎默认的显存预算）；"
                        "要保证 kv_blocks*block_size+1 <= max_model_len")
    p.add_argument("--short-len", type=int, default=128, help="case a 两条短请求的长度")
    p.add_argument("--short-max-tokens", type=int, default=2, help="case a 短请求生成多少 token")
    p.add_argument("--b-prompt-len", type=int, default=300, help="case b 每条 prompt 的长度")
    p.add_argument("--b-max-tokens", type=int, default=16, help="case b 每条请求生成多少 token")
    p.add_argument("--steps-before-remove", type=int, default=4,
                   help="case b 跑到第几步再删 running 里的序列")
    p.add_argument("--max-steps", type=int, default=4096, help="单个 case 的步数上限，防死循环")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    args = p.parse_args(argv)

    if args.kv_blocks <= 0:
        args.kv_blocks = None
    if args.max_tokens < 1:
        p.error("--max-tokens 必须 >= 1")
    if args.case == "b" and args.steps_before_remove < 1:
        p.error("--steps-before-remove 必须 >= 1")
    return args


def run_all(argv) -> int:
    """每个 case 起独立子进程：一个进程里连续建两个引擎会重复 init NCCL。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    forwarded = []
    i = 0
    while i < len(argv):
        if argv[i] == "--case":
            i += 2
            continue
        if argv[i].startswith("--case="):
            i += 1
            continue
        forwarded.append(argv[i])
        i += 1

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    codes = {}
    for case in ("a", "b", "c"):
        cmd = [sys.executable, "-m", "bench.edge", "--case", case, *forwarded]
        print(f"\n>>> {' '.join(cmd)}", flush=True)
        codes[case] = subprocess.run(cmd, cwd=repo_root).returncode

    print("\n" + "=" * 78)
    print("case 汇总: " + "  ".join(f"{c}={'PASS' if code == 0 else f'FAIL({code})'}"
                                   for c, code in codes.items()))
    print("=" * 78)
    return 0 if all(code == 0 for code in codes.values()) else 1


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.case == "all":
        return run_all(argv if argv is not None else sys.argv[1:])

    if not torch.cuda.is_available():
        print("需要 CUDA：这个 bench 会真的跑模型（当前 torch 看不到 GPU）", file=sys.stderr)
        return 2
    if not os.path.isdir(args.model):
        print(f"找不到模型目录: {args.model}\n"
              f"用 --model /path/to/Qwen3-0.6B 指定，或改 DEFAULT_MODEL", file=sys.stderr)
        return 2

    rep = Case(args.case, CASE_TITLES[args.case])
    handlers = {"a": case_a, "b": case_b, "c": case_c}
    try:
        handlers[args.case](args, rep)
    except Exception as exc:  # noqa: BLE001 - bench 自身失败也要有结论，不能吞
        rep.fail(f"未捕获异常: {exc_text(exc)}")
        traceback.print_exc()
    return 0 if rep.summary() else 1


if __name__ == "__main__":
    sys.exit(main())
