from collections import deque

from ..config import Config
from .sequence import Sequence, SequenceStatus
from .block_manager import AllocStatus, BlockManager

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batch_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.eos = config.eos

        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size) # 物理KV Cache块管理器

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # 本步已判终态、但还没归还 block 的序列：schedule() 只负责出队，
        # 真正的 deallocate 要等本步 forward 跑完（见 LLMEngine.step）
        self.finished: list[Sequence] = []

    def is_finished(self):
        return not self.waiting and not self.running and not self.finished

    def add(self, seq: Sequence):
        # 长度检查提前到入队（DeepInfer 的 prompt_limit）：给 decode 至少留 1 个位置。
        # 只打状态、照样入队，出队和释放交给任务 B 的统一出口。
        if seq.num_prompt_tokens > self.max_model_len - 1:
            seq.status = SequenceStatus.FINISHED_IGNORED
        self.waiting.append(seq)

    def abort(self, seq_ids: int | list[int]):
        """取消 waiting / running 里的请求：只打 FINISHED_ABORTED。

        不动队列、不释放 block：出队由 schedule() 开头统一做，
        释放由 step() 在 run 返回之后做（GPU 可能还在读这些 block）。
        """
        if isinstance(seq_ids, int):
            seq_ids = [seq_ids]
        ids = set(seq_ids)
        for seq in (*self.waiting, *self.running):
            if seq.seq_id in ids and not seq.is_finished:
                seq.status = SequenceStatus.FINISHED_ABORTED

    def schedule(self) -> tuple[list[Sequence], bool]:
        schedule_seqs = []
        num_batch_tokens = 0

        # 统一出队：把 waiting / running 里已判终态的序列移进 finished，两个队列只留活的。
        # 这里只出队，不释放 block（对应 DeepInfer 的 free_finished_seqs）。
        for seq in list(self.waiting):
            if seq.is_finished:
                self.waiting.remove(seq)
                self.finished.append(seq)
        for seq in list(self.running):
            if seq.is_finished:
                self.running.remove(seq)
                self.finished.append(seq)

        # 组织prefill调度
        while self.waiting and len(schedule_seqs) < self.max_num_seqs:
            remaining = self.max_num_batch_tokens - num_batch_tokens
            if remaining == 0:
                break

            seq = self.waiting[0]
            if not seq.block_table: # 新seq，先检查前缀缓存命中数
                status, num_cached_blocks = self.block_manager.can_allocate(seq)
                if status == AllocStatus.NEVER:
                    # 永远放不下：打终态出队，继续看下一个请求（不要卡住队头）
                    seq.status = SequenceStatus.FINISHED_IGNORED
                    self.waiting.popleft()
                    self.finished.append(seq)
                    continue
                if status == AllocStatus.LATER:
                    # 暂时放不下：等 running 里的请求结束、block 回来后再试
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            if remaining < num_tokens and schedule_seqs:
                break

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batch_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

            schedule_seqs.append(seq)

        if schedule_seqs:
            return schedule_seqs, True

        # 终态清完之后可能一条能跑的都没有（全被取消、或者全都装不下）：
        # 返回空步，不要把 decode 分支的 assert 当出口
        if not self.running:
            return [], False

        # decode
        while self.running and len(schedule_seqs) < self.max_num_seqs:
            seq = self.running.popleft()

            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break

            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            schedule_seqs.append(seq)

        assert schedule_seqs
        self.running.extendleft(reversed(schedule_seqs))
        return schedule_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            # 已经终态的序列（例如本步之外被 abort）不要再 append_token
            if seq.is_finished:
                continue
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)

            # 这里只判定结局，不出队、不释放（对应 DeepInfer 的 _check_stop）：
            # 出队由 schedule() 开头统一做，释放由 step() 在 run 返回之后做。
            if not seq.ignores_eos and token_id == self.eos:
                seq.status = SequenceStatus.FINISHED_STOPPED
            elif seq.num_tokens >= self.max_model_len or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED

        