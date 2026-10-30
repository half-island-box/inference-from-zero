import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from inference_from_zero.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx) # token的块内物理位置
    if slot == -1: return

    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D) 
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    # stride表tensor某个维度相邻元素在物理存储层面的实际元素间隔
    # 对于连续张量的stride，维度从右往左递增
    # key.shape   torch.Size([2, 4, 8]) -> toech.Stride(32, 8, 1)
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    # 用启动num_token个实例，每个实例处理一个token的KV 缓存
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)



class Attention(nn.Module):

    def __init__(self, num_heads:int, head_dim:int, num_kv_heads:int):
        super().__init__()

        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.num_heads=num_heads
        self.head_dim=head_dim
        self.num_kv_heads=num_kv_heads

        self.group=num_heads // num_kv_heads
        self.scale=head_dim ** -0.5

        # [num_slots, num_kv_heads, head_dim]
        self.k_cache = self.v_cache = torch.tensor([])


    # KV Cache + flash_attn
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None: # chunked perfill 有前置的KV 缓存
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else: # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o


    # attn自实现
    # def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor) -> torch.Tensor:
    #     # q: [B,S,H, D_h]
    #     # kv: [B, S, H_kv, D_h]

    #     B, S, H, D_h = query.shape

    #     # transpose: [B, H, S, D] 因为要在 head 维度作矩阵运算
    #     # reshape: [B, H_kv, G, S, D] 方便和后续同一 KV 组内的 query 复用 K/V
    #     query=query.transpose(1,2).reshape(B, self.num_kv_heads, self.group, S, D_h)


    #     # transpose: [B, H_kv, S, D]
    #     # reshape: [B, H_kv, 1, S, D]
    #     key=key.transpose(1,2).reshape(B, self.num_kv_heads, 1, S, D_h)
    #     value=value.transpose(1,2).reshape(B, self.num_kv_heads, 1, S, D_h)


    #     # 需要掩盖上三角：设为true
    #     mask= ~torch.ones(S, S, dtype=torch.bool, device=query.device).tril()

    #     scores=query @ key.transpose(-2, -1) *self.scale

    #     scores=scores.masked_fill(mask, float("-inf"))

    #     out=F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype) @ value
    #     return out.reshape(B, H, S, D_h).transpose(1,2)

if __name__ == "__main__":

    torch.manual_seed(42)

    

    B, S, H, Hkv, D=1, 1024, 16, 8, 128

    # device="cuda"

    

    q=torch.randn(B, S, H, D)

    k=torch.randn(B, S, Hkv, D)

    

    attn=Attention(H, D, Hkv)

    

    out=attn(q,k,k)

    

    reference=F.scaled_dot_product_attention(

    q.transpose(1,2),k.transpose(1,2), k.transpose(1,2),

    is_causal=True, scale=D ** -0.5, enable_gqa=True,

    ).transpose(1,2)

    

    print(torch.allclose(out, reference, atol=1e-4, rtol=1e-3))
