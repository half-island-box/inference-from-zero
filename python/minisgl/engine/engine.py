"""推理引擎：模型、KV 池、页表、attention backend 与采样器。

Engine 只负责一次 batch forward；请求生命周期、页分配和调度属于
``scheduler.Scheduler``。阶段 2 的静态驱动 ``PagedEngine`` 复用这里的引擎部分。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from minisgl.attention import BaseAttnBackend, create_attention_backend
from minisgl.core import Batch, Context, set_global_ctx
from minisgl.kvcache import create_kvcache_pool
from minisgl.models import BaseLLMModel, ModelConfig

from .sample import BatchSamplingArgs, Sampler


@dataclass
class ForwardOutput:
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    logits: torch.Tensor  # 每个请求最后一个 query 的 logits
    copy_done_event: torch.cuda.Event | None = None


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class Engine:
    def __init__(
        self,
        model: BaseLLMModel,
        model_config: ModelConfig,
        *,
        num_pages: int,
        page_size: int = 1,
        max_running_req: int = 16,
        max_seq_len: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        attn_backend: BaseAttnBackend | str | None = None,
        sampler: Sampler | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = model
        self.model_config = model_config
        self.page_size = page_size
        self.max_running_req = max_running_req
        self.num_pages = num_pages
        self.max_seq_len = min(
            max_seq_len or model_config.rotary_config.max_position,
            num_pages * page_size,
        )
        aligned_max_seq_len = _align_up(self.max_seq_len, 32)

        self.ctx = Context(page_size=page_size)
        self.kv_cache = create_kvcache_pool(
            model_config,
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
        if attn_backend is None:
            self.ctx.attn_backend = create_attention_backend("torch")
        elif isinstance(attn_backend, str):
            self.ctx.attn_backend = create_attention_backend(attn_backend)
        else:
            self.ctx.attn_backend = attn_backend
        set_global_ctx(self.ctx, force=True)
        self.attn_backend = self.ctx.attn_backend
        self.sampler = sampler or Sampler(self.device, model_config.vocab_size)

    def forward_logits(self, batch: Batch) -> torch.Tensor:
        """执行一次模型 forward，返回扁平 token 的 logits [T, vocab]。"""
        with self.ctx.forward_batch(batch):
            return self.model.forward()

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs | None = None) -> ForwardOutput:
        logits = self.forward_logits(batch)
        metadata = batch.attn_metadata
        assert metadata is not None, "attention metadata is not prepared"
        last_logits = logits[metadata.get_last_indices(batch.size)]
        # 与参考实现一致：forward 完成后推进请求状态机
        for req in batch.reqs:
            req.complete_one()
        next_tokens_gpu = self.sampler.sample(last_logits, args)

        # 异步拷回 pinned host buffer；调度器在处理结果前同步 event
        copy_done_event: torch.cuda.Event | None = None
        if self.device.type == "cuda":
            next_tokens_cpu = torch.empty_like(next_tokens_gpu, device="cpu", pin_memory=True)
            next_tokens_cpu.copy_(next_tokens_gpu, non_blocking=True)
            copy_done_event = torch.cuda.Event()
            copy_done_event.record()
        else:
            next_tokens_cpu = next_tokens_gpu.cpu()
        return ForwardOutput(
            next_tokens_gpu=next_tokens_gpu,
            next_tokens_cpu=next_tokens_cpu,
            logits=last_logits,
            copy_done_event=copy_done_event,
        )

    def per_page_bytes(self) -> int:
        config = self.model_config
        return (
            2
            * config.num_layers
            * config.num_kv_heads
            * config.head_dim
            * self.page_size
            * self.dtype.itemsize
        )

    @classmethod
    def num_pages_for(
        cls,
        model_config: ModelConfig,
        *,
        free_bytes: int,
        page_size: int,
        dtype: torch.dtype,
        memory_ratio: float = 0.9,
    ) -> int:
        per_page = (
            2
            * model_config.num_layers
            * model_config.num_kv_heads
            * model_config.head_dim
            * page_size
            * dtype.itemsize
        )
        return max(int(free_bytes * memory_ratio) // per_page, 1)
