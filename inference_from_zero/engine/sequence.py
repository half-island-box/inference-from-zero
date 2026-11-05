from copy import copy
from enum import IntEnum
from itertools import count

from ..sampling_params import SamplingParams


class SequenceStatus(IntEnum):
    """请求状态机。

    终态必须全部排在 RUNNING 之后：is_finished 用序比较（>= FINISHED_STOPPED），
    中间不能插非终态。本任务只加类型，任务 B/C 再把终态接进调度器。

    照 DeepInfer 的 SequenceStatus（csrc/engine/sequence.hpp）缩成 7 个，
    去掉这一阶段用不到的 SWAPPED / FINISHED_SO_FAILED / FINISHED_VE_FAILED。
    """

    WAITING = 0
    RUNNING = 1
    FINISHED_STOPPED = 2          # 采样到 eos
    FINISHED_LENGTH_CAPPED = 3    # 生成到 max_tokens
    FINISHED_ABORTED = 4          # 客户端取消
    FINISHED_IGNORED = 5          # 入队即拒绝（prompt 太长 / 永远放不下）
    FINISHED_PREEMPTED = 6        # 抢占后不再重算（资源不足）

    @property
    def is_finished(self) -> bool:
        return self >= SequenceStatus.FINISHED_STOPPED

    @property
    def finish_reason(self) -> str | None:
        return FINISH_REASON.get(self)


# 终态 -> 回给调用方的结束原因（字符串照 DeepInfer）；非终态不在表里，返回 None
FINISH_REASON = {
    SequenceStatus.FINISHED_STOPPED: "stop",
    SequenceStatus.FINISHED_LENGTH_CAPPED: "length",
    SequenceStatus.FINISHED_ABORTED: "abort",
    SequenceStatus.FINISHED_IGNORED: "prompt_length_exceeded",
    SequenceStatus.FINISHED_PREEMPTED: "insufficient_system_resource",
}


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING

        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids) # 总token数量
        self.num_cached_tokens = 0 # 在KV Cache中的token数量
        self.num_scheduled_tokens = 0 # 本轮调度分配处理的token数
        # 2.7 ZOS 的架构种子：已经确定要生成、但还没落地的 token，也要按上界占块。
        # 现在恒为 0，num_blocks 的行为和以前一致。
        self.total_inflight_tokens = 0

        self.is_prefill = True
        self.block_table = []

        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignores_eos = sampling_params.ignores_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        # 用序比较：所有 FINISHED_* 都算结束
        return self.status >= SequenceStatus.FINISHED_STOPPED

    @property
    def finish_reason(self) -> str | None:
        return self.status.finish_reason

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    # 向上取整分配block数量；inflight token 也按上界占块
    @property
    def num_blocks(self):
        return (self.num_tokens + self.total_inflight_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 返回某一块block中的token_ids
    def block(self, i):
        assert 0 <= i <self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
