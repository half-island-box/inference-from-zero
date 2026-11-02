from dataclasses import dataclass
import torch

# 存放layers的共享元数据
@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    # 变长序列的累积长度
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None

    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # decode 时每条序列的上下文长度（CPU 上的 python list）。
    # attention 每层都要用，放在 CPU 上可以避免每层都做一次 GPU->CPU 同步。
    context_lens_list: list[int] | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q =None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, context_lens_list=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, context_lens_list)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
