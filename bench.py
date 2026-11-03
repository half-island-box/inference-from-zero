import argparse
import csv
import os
import time
from datetime import datetime
from random import randint, seed

import torch

from inference_from_zero import LLM, SamplingParams
from inference_from_zero.utils import profiler as prof
# from vllm import LLM, SamplingParams

RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.csv")

FIELDNAMES = [
    "timestamp",
    "gpu_name",
    "gpu_count",
    "gpu_memory_gb",
    "model",
    "torch_version",
    "num_seqs",
    "max_input_len",
    "max_output_len",
    "total_tokens",
    "elapsed_s",
    "throughput_tok_s",
]


def get_gpu_info() -> dict:
    """识别当前运行所用的 GPU 型号与显存；无 CUDA 时返回占位信息。"""
    if not torch.cuda.is_available():
        return {"gpu_name": "cpu", "gpu_count": 0, "gpu_memory_gb": 0.0}

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_count": torch.cuda.device_count(),
        "gpu_memory_gb": round(props.total_memory / 1024**3, 1),
    }


def append_result(row: dict, path: str = RESULTS_CSV) -> None:
    """把一次 benchmark 结果追加到 CSV；文件不存在时先写表头。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        # 旧文件可能缺列，多出来的字段直接忽略
        writer.writerow({k: row[k] for k in FIELDNAMES if k in row})


def parse_args():
    p = argparse.ArgumentParser(description="inference-from-zero benchmark")
    # 工作量：默认从 16x256/128 放大到 64x512/256
    p.add_argument("--num-seqs", type=int, default=64, help="并发序列数")
    p.add_argument("--max-input-len", type=int, default=512, help="输入长度上限")
    p.add_argument("--max-output-len", type=int, default=256, help="输出长度上限")
    p.add_argument("--min-input-len", type=int, default=100, help="输入长度下限")
    p.add_argument("--min-output-len", type=int, default=100, help="输出长度下限")
    p.add_argument("--max-model-len", type=int, default=0,
                   help="单序列最大长度；0 表示自动取 input+output 上限并向上取整到 256")
    p.add_argument("--repeat", type=int, default=1, help="重复跑几轮，每轮追加一行结果")
    p.add_argument("--seed", type=int, default=0, help="随机种子（第 i 轮用 seed+i）")
    p.add_argument("--results", default=None,
                   help=f"结果 CSV 路径（默认 {os.path.basename(RESULTS_CSV)}）")
    p.add_argument("--profile", action="store_true",
                   help="打开 NVTX 打点 + cudaProfilerApi，配合 scripts/profile.sh 用 nsys 录 trace")
    p.add_argument("--profile-deep", action="store_true",
                   help="连每层 attention 也打点（trace 会变大，先确认瓶颈大概在哪再用）")
    p.add_argument("--profile-all", action="store_true",
                   help="默认只录第 1 轮（capture-range 只能结束一次），开启后录全部轮次")
    p.add_argument("--profile-out", default="", help="把阶段耗时汇总另存为文本文件")
    args = p.parse_args()

    if args.profile_deep:
        args.profile = True

    # profile 会拖慢运行，默认不把（偏慢的）结果写进基准 CSV，避免污染纵向对比
    args.results_explicit = args.results is not None
    args.results = args.results or RESULTS_CSV
    args.write_csv = args.results_explicit or not args.profile

    assert 1 <= args.min_input_len <= args.max_input_len, "输入长度区间非法"
    assert 1 <= args.min_output_len <= args.max_output_len, "输出长度区间非法"
    if args.max_model_len <= 0:
        need = args.max_input_len + args.max_output_len
        args.max_model_len = -(-need // 256) * 256  # 向上取整到 256 的整数倍
    assert args.max_model_len >= args.max_input_len + args.max_output_len, \
        f"max_model_len({args.max_model_len}) 必须 >= max_input_len + max_output_len"
    return args


def build_workload(args, round_seed: int):
    """按 seed 生成固定的 prompt / sampling_params，保证每轮可复现。"""
    seed(round_seed)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(args.min_input_len, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignores_eos=True,
                       max_tokens=randint(args.min_output_len, args.max_output_len))
        for _ in range(args.num_seqs)
    ]
    return prompt_token_ids, sampling_params


def main():
    args = parse_args()
    if args.profile:
        prof.enable(deep=args.profile_deep)

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # enforce_eager=True：手写 attention 有动态 gather/形状，图捕获不了
    llm = LLM(path, enforce_eager=True, max_model_len=args.max_model_len)

    # 计时前的 warmup：触发 kernel 编译与显存分配，排除冷启动开销
    with prof.nvtx("bench/warmup"):
        llm.generate(["Benchmark: "], SamplingParams())

    total_out = total_in = total_elapsed = 0
    for i in range(args.repeat):
        prompt_token_ids, sampling_params = build_workload(args, args.seed + i)
        # uncomment the following line for vllm
        # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

        torch.cuda.synchronize()
        # 只在录制的那一轮夹住 cudaProfilerApi：nsys 用 --capture-range=cudaProfilerApi
        # 就只录这一段，warmup 和后面的统计不会进 trace
        capture = args.profile and (i == 0 or args.profile_all)
        if capture:
            prof.start_capture()

        t = time.time()
        with prof.nvtx(f"bench/round:{i}"):
            llm.generate(prompt_token_ids, sampling_params, use_tqdm=args.repeat == 1)
            torch.cuda.synchronize()
        elapsed = time.time() - t

        if capture:
            prof.stop_capture()

        out_tokens = sum(sp.max_tokens for sp in sampling_params)   # 实际生成量（ignores_eos）
        in_tokens = sum(len(p) for p in prompt_token_ids)           # prefill 量
        throughput = out_tokens / elapsed
        print(f"[round {i}] in={in_tokens}tok out={out_tokens}tok "
              f"time={elapsed:.2f}s throughput={throughput:.1f}tok/s "
              f"total(in+out)={out_tokens + in_tokens} -> {(out_tokens + in_tokens) / elapsed:.1f}tok/s")

        if args.write_csv:
            append_result({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                **get_gpu_info(),
                "model": os.path.basename(os.path.normpath(path)),
                "torch_version": torch.__version__,
                "num_seqs": args.num_seqs,
                "max_input_len": args.max_input_len,
                "max_output_len": args.max_output_len,
                "total_tokens": out_tokens,
                "elapsed_s": round(elapsed, 3),
                "throughput_tok_s": round(throughput, 2),
            }, args.results)

        total_out += out_tokens
        total_in += in_tokens
        total_elapsed += elapsed

    if args.repeat > 1:
        print(f"[avg {args.repeat} rounds] out={total_out / args.repeat:.1f}tok "
              f"time={total_elapsed / args.repeat:.2f}s "
              f"throughput={total_out / total_elapsed:.1f}tok/s "
              f"(in+out)={(total_out + total_in) / total_elapsed:.1f}tok/s")
    if args.write_csv:
        print(f"Result appended to {args.results}")
    else:
        print("已跳过写 CSV（--profile 下吞吐偏低，避免污染基准）；"
              "需要写请显式传 --results <path>")
    if args.profile:
        text = prof.report("bench phase timing (CPU wall time)")
        if args.profile_out and text:
            with open(args.profile_out, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            print(f"Phase timing written to {args.profile_out}")
        print("\n下一步（在装好 nsys 的机器上）：")
        print("  scripts/profile.sh --num-seqs 16 --max-input-len 256 --max-output-len 128")
        print("  或手动：nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none \\")
        print("      --capture-range=cudaProfilerApi --capture-range-end=stop \\")
        print("      -o profiles/iftz python bench.py --profile ...")


if __name__ == "__main__":
    main()
