from collections import deque

from ..config import Config
from .sequence import Sequence, SequenceStatus
from .block_manager import BlockManager

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batch_tokens = config.max_num_batched_tokens
        self.eos = config.eos

        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size) # 物理KV Cache块管理器

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        schedule_seqs = []
        num_batch_tokens = 0

        # 组织prefill调度
        while self.waiting and len(schedule_seqs) < self.max_num_seqs:
            remaining = self.max_num_batch_tokens - num_batch_tokens
            if remaining == 0:
                break

            seq = self.waiting[0]
            if not seq.block_table: # 新seq，先检查前缀缓存命中数
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1: break
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
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)

            if (not seq.ignores_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                

        