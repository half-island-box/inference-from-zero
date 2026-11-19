"""固定的 prompt 与基准负载（路线图 §0.3）。

本模块刻意不依赖 torch/transformers，可在任何机器上导入；tokenize 由调用方完成。

所有性能数字都必须绑定固定的模型、dtype 和负载。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 黄金基线 prompt
#
# 只使用原始文本：阶段 6 之前不套 chat template。这些 prompt 以
# ``add_special_tokens=False`` tokenize，并逐 token 对比。
# ---------------------------------------------------------------------------
PROMPTS: list[str] = [
    "The quick brown fox jumps over the lazy dog. Explain why this sentence is "
    "useful for testing typing and tokenization.",
    "Write a Python function that reverses a singly linked list iteratively, and "
    "state its time and space complexity.",
    "Explain the difference between paged attention and a contiguous KV cache in "
    "two or three sentences.",
    "请用三句话解释什么是 KV Cache，以及它为什么能加速自回归推理。",
    "A train travels 120 km at 60 km/h and then 90 km at 45 km/h. What is the "
    "average speed for the whole journey?",
    "Describe the water cycle for a curious ten-year-old in under 100 words.",
    "What are the trade-offs between tensor parallelism and pipeline parallelism "
    "when serving a large language model?",
    "写一个快速排序的递归实现，并说明最坏情况下的时间复杂度。",
    "Summarize the plot of Romeo and Juliet in exactly five bullet points.",
    "Given the list [5, 3, 8, 1, 9, 2], sort it using insertion sort and show "
    "the state after every insertion.",
    "Explain why the softmax function is used at the output of a language model.",
    "Translate the following sentence into French and explain any idiom: "
    "\"It is raining cats and dogs.\"",
    "What is the significance of the Turing test in the philosophy of artificial "
    "intelligence?",
    "Describe how a radix tree can be used to share KV cache between requests "
    "that begin with the same system prompt.",
    "Compute the derivative of f(x) = x^3 * ln(x) and simplify the result.",
    "List three differences between TCP and UDP, with one practical example for "
    "each.",
    "Explain the concept of continuous batching in LLM serving and why it "
    "improves GPU utilization.",
    "用中文写一段 50 字左右的介绍，说明什么是张量并行。",
    "What happens, step by step, when you type a URL into a browser and press "
    "Enter?",
    "Write a short haiku about autumn leaves, then explain the imagery you used.",
]


# ---------------------------------------------------------------------------
# 基准负载
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Load:
    """一组固定的基准负载。

    ``input_lens`` / ``output_lens`` 是逐请求的输入/输出长度，二者长度必须一致。
    ``shared_prefix`` 是所有请求共享的前缀 token 数（0 表示各请求完全独立随机）。
    """

    name: str
    input_lens: tuple[int, ...]
    output_lens: tuple[int, ...]
    shared_prefix: int = 0 # 所有请求共享的前缀token数量
    description: str = ""

    @property
    def num_requests(self) -> int:
        return len(self.input_lens)


LOADS: dict[str, Load] = {
    "short_32": Load(
        name="short_32",
        input_lens=(32,),
        output_lens=(32,),
        description="短 prompt ×1：验证 decode 与小 batch 行为",
    ),
    "medium_512x8": Load(
        name="medium_512x8",
        input_lens=(512,) * 8,
        output_lens=(64,) * 8,
        description="中等 prompt ×8：验证常规 prefill 与并发",
    ),
    "long_8k": Load(
        name="long_8k",
        input_lens=(8192,),
        output_lens=(32,),
        description="长 prompt ×1：验证分块预填充与 ITL",
    ),
    "long_32k": Load(
        name="long_32k",
        input_lens=(32768,),
        output_lens=(16,),
        description="超长 prompt ×1：验证分块预填充与显存压力",
    ),
    "shared_prefix_4k": Load(
        name="shared_prefix_4k",
        input_lens=(4352, 4352),
        output_lens=(32, 32),
        shared_prefix=4096,
        description="两个请求共享 4k 前缀：验证 radix cache 命中",
    ),
    "burst_32": Load(
        name="burst_32",
        input_lens=(128,) * 32,
        output_lens=(32,) * 32,
        description="32 请求突发：验证调度与稳定性",
    ),
    "mixed_lengths": Load(
        name="mixed_lengths",
        input_lens=(32, 128, 512, 1024, 2048, 64, 256, 4096),
        output_lens=(64, 32, 128, 16, 64, 32, 96, 32),
        description="混合长度与 max_tokens：验证生命周期与公平性",
    ),
}


def make_input_ids(
    load: Load,
    seed: int = 0,
    vocab_size: int = 151936,
    token_range: tuple[int, int] = (1000, 30000),
) -> list[list[int]]:
    """为 ``load`` 生成确定性的合成 token ids。

    在 ``token_range``（避开 special token）内用固定种子随机采样，因此相同的
    ``(load, seed)`` 总是生成同一份负载。每个请求的前 ``shared_prefix`` 个
    token 完全相同。
    """
    lo, hi = token_range # 词表范围
    if lo < 0 or hi > vocab_size or lo >= hi:
        raise ValueError(f"invalid token_range {token_range} for vocab_size {vocab_size}")

    rng = random.Random(seed)
    prefix = [rng.randrange(lo, hi) for _ in range(load.shared_prefix)]
    if load.shared_prefix > min(load.input_lens):
        raise ValueError(f"{load.name}: shared_prefix exceeds the shortest input")

    out: list[list[int]] = []
    for input_len in load.input_lens:
        ids = prefix[: min(load.shared_prefix, input_len)]
        ids.extend(rng.randrange(lo, hi) for _ in range(input_len - len(ids)))
        out.append(ids)
    return out

# 拿到某一组负载数据
def get_load(name: str) -> Load:
    try:
        return LOADS[name]
    except KeyError:
        known = ", ".join(sorted(LOADS))
        raise SystemExit(f"unknown load {name!r}; available: {known}") from None
