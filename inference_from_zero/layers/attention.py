import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl
from inference_from_zero.utils.context import get_context
from inference_from_zero.utils import profiler as prof


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


    # KV Cache + 手写 attention
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 新 token 的 KV 先落进分页 cache，后面的注意力直接从 cache 里读
        if k_cache.numel() and v_cache.numel():
            with prof.nvtx("attn/store_kvcache", deep=True):
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            with prof.nvtx("attn/prefill", deep=True):
                return self._prefill(q, k, v, context)
        with prof.nvtx("attn/decode", deep=True):
            return self._decode(q, context) # decode


    def _prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context):
        # q: [N, H, D]，N 是本轮调度到的所有序列新 token 首尾相接
        # k/v: [N, Hkv, D]
        cu_q = context.cu_seqlens_q.tolist()
        cu_k = context.cu_seqlens_k.tolist()

        outputs = []
        for i in range(len(cu_q) - 1):
            qs, qe = cu_q[i], cu_q[i + 1]
            ks, ke = cu_k[i], cu_k[i + 1]

            qi = q[qs:qe].unsqueeze(0)              # [1, Sq, H, D]
            prefix = (ke - ks) - (qe - qs)          # 本轮之前已经进 cache 的 token 数

            if prefix > 0:
                # chunked prefill：前缀 KV 在分页 cache 里，取出来拼到本轮 K/V 前面
                pk, pv = self._gather_kv(context.block_tables[i], prefix)
                ki = torch.cat((pk, k[ks:ke]), dim=0).unsqueeze(0)
                vi = torch.cat((pv, v[ks:ke]), dim=0).unsqueeze(0)
            else:
                ki = k[ks:ke].unsqueeze(0)
                vi = v[ks:ke].unsqueeze(0)

            outputs.append(self._attn(qi, ki, vi, offset=prefix))

        return torch.cat(outputs, dim=1)[0]         # [N, H, D]


    # decode 时一次处理多少条序列，避免 gather 出来的 K/V 撑爆显存
    decode_chunk_size = 64

    def _decode(self, q: torch.Tensor, context):
        # q: [B, H, D]；本轮新 token 的 KV 已经写进 cache，读整条序列即可
        # 注意：长度要用 CPU 上的 python list，逐条 int(GPU tensor) 会触发上万次同步
        lens = context.context_lens_list
        if lens is None:
            lens = context.context_lens.tolist()

        block_size = self.k_cache.size(1)
        num_kv_heads, head_dim = self.k_cache.size(2), self.k_cache.size(3)

        outputs = []
        for s in range(0, q.size(0), self.decode_chunk_size):
            e = min(s + self.decode_chunk_size, q.size(0))
            chunk_lens = lens[s:e]
            num_blocks = (max(chunk_lens) + block_size - 1) // block_size

            # [chunk, num_blocks] -> 展平后一次性把整批 K/V 取出来
            # [chunk, num_blocks*block_size, Hkv, D]
            # 短序列的块表右侧是 -1 填充，这些位置会被下面的 valid 掩码屏蔽，
            # 这里钳到 0 只是为了能合法 index_select
            block_ids = context.block_tables[s:e, :num_blocks].long().reshape(-1).clamp_(min=0)
            ki = self.k_cache.index_select(0, block_ids).reshape(e - s, num_blocks * block_size, num_kv_heads, head_dim)
            vi = self.v_cache.index_select(0, block_ids).reshape(e - s, num_blocks * block_size, num_kv_heads, head_dim)

            # 块是按 num_blocks 对齐取的，超出各自长度的部分是无效填充，用掩码屏蔽
            valid = (torch.arange(ki.size(1), device=q.device).unsqueeze(0)
                     < torch.tensor(chunk_lens, device=q.device, dtype=torch.long).unsqueeze(1))

            outputs.append(self._attn(q[s:e].unsqueeze(1), ki, vi, valid_mask=valid))

        return torch.cat(outputs, dim=0).squeeze(1)  # [B, H, D]


    def _gather_kv(self, block_table: torch.Tensor, length: int):
        """按物理块表把前 length 个 token 的 K/V 从分页 cache 里取出来 -> [length, Hkv, D]"""
        block_size = self.k_cache.size(1)
        num_blocks = (length + block_size - 1) // block_size

        block_ids = block_table[:num_blocks].long()
        block_ids = block_ids[block_ids >= 0]       # block_table 右侧可能有 -1 填充

        num_kv_heads, head_dim = self.k_cache.size(2), self.k_cache.size(3)
        k = self.k_cache.index_select(0, block_ids).reshape(-1, num_kv_heads, head_dim)[:length]
        v = self.v_cache.index_select(0, block_ids).reshape(-1, num_kv_heads, head_dim)[:length]
        return k, v


    def _attn(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              offset: int = 0, valid_mask: torch.Tensor | None = None):
        # query: [B, Sq, H, D]    key/value: [B, Sk, Hkv, D]
        # valid_mask: [B, Sk]，True 表示该 key 有效（decode 时用来屏蔽块内 padding）
        B, Sq, H, D_h = query.shape
        Sk = key.size(1)

        # [B, Sq, H, D] -> [B, Hkv, group, Sq, D]，同一 KV 组内的 query 共享一份 K/V
        query = query.transpose(1, 2).reshape(B, self.num_kv_heads, self.group, Sq, D_h)
        key = key.transpose(1, 2).unsqueeze(2)          # [B, Hkv, 1, Sk, D]
        value = value.transpose(1, 2).unsqueeze(2)

        scores = query @ key.transpose(-2, -1) * self.scale   # [B, Hkv, group, Sq, Sk]

        if valid_mask is not None:
            # decode：q 只有一个 token，不需要因果掩码，只要屏蔽块对齐带出来的无效 padding
            # [B, Sk] -> [B, 1, 1, 1, Sk]
            scores = scores.masked_fill(~valid_mask[:, None, None, None, :], float("-inf"))
        else:
            # prefill：第 i 个 query 只能看到 0..offset+i 的 key（offset 是它前面被缓存的 prefix 长度）
            rows = torch.arange(Sq, device=query.device).view(-1, 1) + offset
            cols = torch.arange(Sk, device=query.device).view(1, -1)
            scores = scores.masked_fill(cols > rows, float("-inf"))

        out = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype) @ value
        return out.reshape(B, H, Sq, D_h).transpose(1, 2)


if __name__ == "__main__":

    torch.manual_seed(42)

    B, S, H, Hkv, D=1, 1024, 16, 8, 128

    # device="cuda"

    q=torch.randn(B, S, H, D)

    k=torch.randn(B, S, Hkv, D)

    attn=Attention(H, D, Hkv)

    out=attn._attn(q,k,k)

    reference=F.scaled_dot_product_attention(
    q.transpose(1,2),k.transpose(1,2), k.transpose(1,2),
    is_causal=True, scale=D ** -0.5, enable_gqa=True,
    ).transpose(1,2)

    print(torch.allclose(out, reference, atol=1e-4, rtol=1e-3))
