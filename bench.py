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
        writer.writerow(row)


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignores_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **get_gpu_info(),
        "model": os.path.basename(os.path.normpath(path)),
        "torch_version": torch.__version__,
        "num_seqs": num_seqs,
        "max_input_len": max_input_len,
        "max_output_len": max_ouput_len,
        "total_tokens": total_tokens,
        "elapsed_s": round(t, 3),
        "throughput_tok_s": round(throughput, 2),
    }
    append_result(row)
    print(f"Result appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
