"""阶段 2 的最小调度循环：先 prefill 全部 pending，再 decode 全部 running。

这是一个**静态 batch** 调度器：请求一次性全部进入，prefill 一轮后进入 decode，
直到所有请求完成。真正按 iteration 动态增删请求的连续批处理留到阶段 3。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from minisgl.attention import BaseAttnBackend, create_attention_backend
from minisgl.core import (
    Batch,
    Context,
    Req,
    SamplingParams,
    set_global_ctx,
)
from minisgl.engine.sample import greedy_sample
from minisgl.kvcache import create_kvcache_pool
from minisgl.models import BaseLLMModel, ModelConfig

from .batch import prepare_batch
from .cache import CacheManager
from .table import TableManager


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass
class BatchRequest:
    input_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    uid: int | None = None


@dataclass
class BatchGenerationResult:
    output_ids: list[int]
    stop_reason: str
    prefill_logits: torch.Tensor | None = None


@dataclass
class BatchRunResult:
    results: list[BatchGenerationResult]
    prefill_ms: float = 0.0
    decode_ms: list[float] = field(default_factory=list)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class PagedEngine:
    """带 paged KV cache 的静态 batch 引擎。"""

    def __init__(
        self,
        model: BaseLLMModel,
        config: ModelConfig,
        *,
        num_pages: int,
        page_size: int = 1,
        max_running_req: int = 16,
        max_seq_len: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        attn_backend: BaseAttnBackend | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model
        self.page_size = page_size
        self.max_running_req = max_running_req
        self.max_seq_len = min(
            max_seq_len or config.rotary_config.max_position, num_pages * page_size
        )
        aligned_max_seq_len = _align_up(self.max_seq_len, 32)

        self.ctx = Context(page_size=page_size)
        self.kv_cache = create_kvcache_pool(
            config,
            num_pages=num_pages,
            page_size=page_size,
            dtype=dtype,
            device=self.device,
        )
        # +1 行 dummy 请求；dummy 页从 num_pages * page_size 开始
        self.page_table = torch.zeros(
            (max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        self.page_table[max_running_req].fill_(num_pages * page_size)

        self.ctx.page_table = self.page_table
        self.ctx.kv_cache = self.kv_cache
        self.ctx.attn_backend = attn_backend or create_attention_backend("torch")
        set_global_ctx(self.ctx, force=True)

        self.table_manager = TableManager(max_running_req, self.page_table)
        self.cache_manager = CacheManager(num_pages, page_size, self.page_table, type="naive")
        self.num_pages = num_pages
        self.token_pool = self.table_manager.token_pool

    # ------------------------------------------------------------------ 主循环
    def generate(
        self,
        requests: list[BatchRequest],
        *,
        eos_token_id: int | None = None,
    ) -> BatchRunResult:
        if len(requests) > self.max_running_req:
            raise ValueError(
                f"batch size {len(requests)} exceeds max_running_req {self.max_running_req}"
            )
        reqs: list[Req] = []
        outputs: dict[int, list[int]] = {}
        stop_reasons: dict[int, str] = {}
        prefill_logits: dict[int, torch.Tensor] = {}
        finished: set[int] = set()

        try:
            for index, request in enumerate(requests):
                req = self._create_req(request, index)
                reqs.append(req)
                outputs[req.uid] = []
                stop_reasons[req.uid] = "length"

            run = BatchRunResult(results=[])
            if not reqs:
                return run

            _sync(self.device)
            start = time.perf_counter()
            logits = self._forward(Batch(reqs=reqs, phase="prefill"))
            _sync(self.device)
            run.prefill_ms = (time.perf_counter() - start) * 1e3
            for i, req in enumerate(reqs):
                prefill_logits[req.uid] = logits[i].detach().float().cpu()
            self._finish_step(
                reqs, logits, eos_token_id, outputs, stop_reasons, finished, prefill=True
            )

            running = [req for req in reqs if req.uid not in finished]
            while running:
                running.sort(key=lambda req: req.uid)
                _sync(self.device)
                start = time.perf_counter()
                logits = self._forward(Batch(reqs=running, phase="decode"))
                _sync(self.device)
                run.decode_ms.append((time.perf_counter() - start) * 1e3)
                self._finish_step(
                    running, logits, eos_token_id, outputs, stop_reasons, finished, prefill=False
                )
                running = [req for req in running if req.uid not in finished]

            run.results = [
                BatchGenerationResult(
                    output_ids=outputs[req.uid],
                    stop_reason=stop_reasons[req.uid],
                    prefill_logits=prefill_logits.get(req.uid),
                )
                for req in reqs
            ]
            return run
        except Exception:
            # 出错时尽量归还资源，避免后续测试误判为泄漏
            for req in reqs:
                if req.uid not in finished:
                    self._free_req(req)
                    finished.add(req.uid)
            raise

    # ------------------------------------------------------------------ 单步
    def _forward(self, batch: Batch) -> torch.Tensor:
        self.cache_manager.allocate_paged(batch.reqs)
        prepare_batch(batch, self.page_table, self.token_pool)
        self.ctx.attn_backend.prepare_metadata(batch)
        with self.ctx.forward_batch(batch):
            logits = self.model.forward()
        metadata = batch.attn_metadata
        assert metadata is not None
        return logits[metadata.get_last_indices(batch.size)]

    def _finish_step(
        self,
        reqs: list[Req],
        logits: torch.Tensor,
        eos_token_id: int | None,
        outputs: dict[int, list[int]],
        stop_reasons: dict[int, str],
        finished: set[int],
        *,
        prefill: bool,
    ) -> None:
        # 写回位置取 forward 前的 device_len：采样出的 token 将在下一次 forward 读取
        write_positions = [req.device_len for req in reqs]
        # 与参考实现一致：forward 完成后推进状态机，再处理采样结果
        for req in reqs:
            req.complete_one()

        for i, req in enumerate(reqs):
            if req.output_len == 0:
                # max_tokens=0：仍可做 prefill，但不产生任何输出 token
                stop_reasons[req.uid] = "length"
                self._free_req(req)
                finished.add(req.uid)
                continue
            token = int(greedy_sample(logits[i]).item())
            is_eos = (
                eos_token_id is not None
                and not req.sampling_params.ignore_eos
                and token == eos_token_id
            )
            if is_eos:
                stop_reasons[req.uid] = "eos"
                self._free_req(req)
                finished.add(req.uid)
            elif not req.can_decode:
                outputs[req.uid].append(token)
                stop_reasons[req.uid] = "length"
                self._free_req(req)
                finished.add(req.uid)
            else:
                outputs[req.uid].append(token)
                self.token_pool[req.table_idx, write_positions[i]] = token
                req.append_host(torch.tensor([token], dtype=torch.long))

        if prefill:
            for req in reqs:
                if req.uid not in finished:
                    self.cache_manager.cache_req(req, finished=False)

    # ------------------------------------------------------------------ 资源
    def _create_req(self, request: BatchRequest, index: int) -> Req:
        if not request.input_ids:
            raise ValueError("input_ids must not be empty")
        params = request.sampling_params
        if not params.is_greedy:
            raise NotImplementedError("stage 2 only supports greedy sampling")
        input_len = len(request.input_ids)
        if input_len > self.max_seq_len:
            raise ValueError(f"input length {input_len} exceeds max_seq_len {self.max_seq_len}")
        output_len = min(params.max_tokens, self.max_seq_len - input_len)
        if output_len < 0:
            raise ValueError("input length exceeds max_seq_len")

        uid = index if request.uid is None else request.uid
        table_idx = self.table_manager.allocate()
        input_ids = torch.tensor(request.input_ids, dtype=torch.long)
        handle = self.cache_manager.match_prefix(input_ids).cuda_handle
        self.cache_manager.lock(handle)
        self.token_pool[table_idx, :input_len] = input_ids.to(self.device)
        if handle.cached_len > 0:
            self.page_table[table_idx, : handle.cached_len] = handle.get_matched_indices()
        return Req(
            input_ids=input_ids,
            table_idx=table_idx,
            cached_len=handle.cached_len,
            output_len=output_len,
            uid=uid,
            sampling_params=params,
            cache_handle=handle,
        )

    def _free_req(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def check_integrity(self) -> None:
        self.cache_manager.check_integrity()
