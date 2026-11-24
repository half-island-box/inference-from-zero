"""连续批处理调度器：每个 iteration 动态决定 prefill / decode 的 batch。

主循环：prefill 有 pending 就优先组一个 prefill batch，否则跑所有 running 请求的
decode batch；完成 / 中止的请求立即释放槽位与页。
"""

from __future__ import annotations

import time

import torch

from minisgl.core import Batch, Req, SamplingParams
from minisgl.engine import Engine, ForwardOutput
from minisgl.env import ENV

from .batch import prepare_batch
from .cache import CacheManager
from .decode import DecodeManager
from .prefill import PrefillManager
from .table import TableManager
from .utils import RequestState


class Scheduler:
    def __init__(
        self,
        engine: Engine,
        *,
        max_extend_tokens: int = 8192,
        cache_type: str = "naive",
        eos_token_id: int | None = None,
        record_prefill_logits: bool = False,
    ) -> None:
        self.engine = engine
        self.eos_token_id = eos_token_id
        self.record_prefill_logits = record_prefill_logits

        self.table_manager = TableManager(engine.max_running_req, engine.page_table)
        self.cache_manager = CacheManager(
            engine.num_pages, engine.page_size, engine.page_table, type=cache_type
        )
        self.decode_manager = DecodeManager(engine.page_size)
        self.prefill_manager = PrefillManager(
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            decode_manager=self.decode_manager,
            max_seq_len=engine.max_seq_len,
            max_extend_tokens=max_extend_tokens,
        )
        self.token_pool = self.table_manager.token_pool

        self.states: dict[int, RequestState] = {}
        self._next_uid = 0

    # ------------------------------------------------------------------ 请求
    def add_request(
        self,
        input_ids: list[int] | torch.Tensor,
        sampling_params: SamplingParams | None = None,
        *,
        uid: int | None = None,
    ) -> int:
        if isinstance(input_ids, torch.Tensor):
            if input_ids.is_cuda:
                raise ValueError("input_ids must be a CPU tensor")
            ids = input_ids.to(torch.long).flatten()
        else:
            ids = torch.tensor(input_ids, dtype=torch.long)
        if ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if len(ids) > self.engine.max_seq_len:
            raise ValueError(
                f"input length {len(ids)} exceeds max_seq_len {self.engine.max_seq_len}"
            )

        if uid is None:
            uid = self._next_uid
        if uid in self.states:
            raise ValueError(f"duplicate request uid: {uid}")
        self._next_uid = max(self._next_uid, uid + 1)

        params = sampling_params or SamplingParams()
        self.states[uid] = RequestState(uid=uid, input_ids=ids.tolist(), sampling_params=params)
        self.prefill_manager.add_req(uid, ids, params)
        return uid

    def abort_request(self, uid: int) -> bool:
        state = self.states.get(uid)
        if state is None or state.finished:
            return False

        req = self.prefill_manager.abort_req(uid)
        if req is None:
            req = self.decode_manager.abort_req(uid)
        if req is not None:
            self._free_req(req)
        state.stop_reason = "aborted"
        state.finished = True
        state.finished_time = time.perf_counter()
        return True

    def clear_results(self) -> None:
        if not self.is_idle:
            raise RuntimeError("cannot clear results while requests are still running")
        self.states.clear()

    # ------------------------------------------------------------------ 主循环
    def step(self) -> bool:
        batch = self.prefill_manager.schedule_next_batch()
        if batch is None:
            batch = self.decode_manager.schedule_next_batch()
        if batch is None:
            if ENV.CHECK_INTEGRITY and self.is_idle:
                self.cache_manager.check_integrity()
            return False

        # 写回位置在 forward 前确定：采样出的 token 供下一次 decode 读取
        write_positions = [req.device_len for req in batch.reqs]
        self.cache_manager.allocate_paged(batch.reqs)
        prepare_batch(batch, self.engine.page_table, self.token_pool)
        self.engine.attn_backend.prepare_metadata(batch)
        sample_args = self.engine.sampler.prepare(batch)
        output = self.engine.forward_batch(batch, sample_args)
        self._process_output(batch, output, write_positions)

        if batch.is_prefill:
            for req in batch.reqs:
                if not self.states[req.uid].finished:
                    self.decode_manager.add_req(req)
        return True

    def run_until_complete(self) -> None:
        while self.step():
            pass
        if not self.is_idle:
            raise RuntimeError(
                "scheduler is stuck: pending requests cannot be admitted "
                "(insufficient KV pages or request slots)"
            )
        if ENV.CHECK_INTEGRITY:
            self.cache_manager.check_integrity()

    @property
    def is_idle(self) -> bool:
        return not self.prefill_manager.runnable and not self.decode_manager.runnable

    # ------------------------------------------------------------------ 结果
    def _process_output(
        self,
        batch: Batch,
        output: ForwardOutput,
        write_positions: list[int],
    ) -> None:
        if output.copy_done_event is not None:
            output.copy_done_event.synchronize()
        now = time.perf_counter()
        tokens = output.next_tokens_cpu.tolist()

        if batch.is_prefill and self.record_prefill_logits:
            logits = output.logits.float().cpu()
            for index, req in enumerate(batch.reqs):
                self.states[req.uid].prefill_logits = logits[index]

        for index, req in enumerate(batch.reqs):
            state = self.states[req.uid]
            token = int(tokens[index])
            if req.output_len == 0:
                self._finish(req, state, "length", now)
                continue

            is_eos = (
                self.eos_token_id is not None
                and not req.sampling_params.ignore_eos
                and token == self.eos_token_id
            )
            if is_eos:
                # EOS 不输出
                self._finish(req, state, "eos", now)
            elif not req.can_decode:
                state.output_ids.append(token)
                state.token_times.append(now)
                self._finish(req, state, "length", now)
            else:
                state.output_ids.append(token)
                state.token_times.append(now)
                self.token_pool[req.table_idx, write_positions[index]] = token
                req.append_host(torch.tensor([token], dtype=torch.long))

        if batch.is_prefill:
            for req in batch.reqs:
                if not self.states[req.uid].finished:
                    self.cache_manager.cache_req(req, finished=False)

    def _finish(self, req: Req, state: RequestState, reason: str, now: float) -> None:
        state.stop_reason = reason
        state.finished = True
        state.finished_time = now
        self.decode_manager.remove_req(req)
        self._free_req(req)

    def _free_req(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def check_integrity(self) -> None:
        self.cache_manager.check_integrity()
