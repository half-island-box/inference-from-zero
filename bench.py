import argparse
import csv
import os
import time
from datetime import datetime
from random import randint, seed

import torch

from inference_from_zero import LLM, SamplingParams
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
    p.add_argument("--results", default=RESULTS_CSV, help="结果 CSV 路径")
    args = p.parse_args()

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

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # enforce_eager=True：手写 attention 有动态 gather/形状，图捕获不了
    llm = LLM(path, enforce_eager=True, max_model_len=args.max_model_len)

    # 计时前的 warmup：触发 kernel 编译与显存分配，排除冷启动开销
    llm.generate(["Benchmark: "], SamplingParams())

    total_out = total_in = total_elapsed = 0
    for i in range(args.repeat):
        prompt_token_ids, sampling_params = build_workload(args, args.seed + i)
        # uncomment the following line for vllm
        # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

        torch.cuda.synchronize()
        t = time.time()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=args.repeat == 1)
        torch.cuda.synchronize()
        elapsed = time.time() - t

        out_tokens = sum(sp.max_tokens for sp in sampling_params)   # 实际生成量（ignores_eos）
        in_tokens = sum(len(p) for p in prompt_token_ids)           # prefill 量
        throughput = out_tokens / elapsed
        print(f"[round {i}] in={in_tokens}tok out={out_tokens}tok "
              f"time={elapsed:.2f}s throughput={throughput:.1f}tok/s "
              f"total(in+out)={out_tokens + in_tokens} -> {(out_tokens + in_tokens) / elapsed:.1f}tok/s")

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
    print(f"Result appended to {args.results}")


if __name__ == "__main__":
    main()
