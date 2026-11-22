"""参考 attention 后端：按请求 gather KV + SDPA。

阶段 2 先保证正确性：每个请求按页表收集 ``[0, device_len)`` 的 K/V（含本轮刚
写入的 token），再逐请求调用 SDPA。后续阶段再替换为 varlen / FlashAttention。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata


@dataclass
class TorchAttnMetadata(BaseAttnMetadata):
    q_seq_lens: list[int]
    kv_seq_lens: list[int]
    cached_lens: list[int]
    page_table: torch.Tensor  # [bs, max_kv_len]，原始 token 槽位
    last_indices: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class TorchAttnBackend(BaseAttnBackend):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, TorchAttnMetadata), "metadata is not prepared"
        ctx = get_global_ctx()
        # 先写 KV 再读，decode 的自注意力必须包含当前 token
        ctx.kv_cache.store_kv(k, v, batch.out_loc, layer_id)

        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        group = num_qo_heads // num_kv_heads
        q_offset = 0
        outputs: list[torch.Tensor] = []

        k_flat = ctx.kv_cache.k_cache(layer_id).view(-1, num_kv_heads, q.shape[-1])
        v_flat = ctx.kv_cache.v_cache(layer_id).view(-1, num_kv_heads, q.shape[-1])

        for q_len, kv_len, cached_len, slots in zip(
            metadata.q_seq_lens,
            metadata.kv_seq_lens,
            metadata.cached_lens,
            metadata.page_table,
        ):
            q_req = q[q_offset : q_offset + q_len]
            q_offset += q_len

            indices = slots[:kv_len].to(torch.long)
            k_req = k_flat.index_select(0, indices)
            v_req = v_flat.index_select(0, indices)
            if group > 1:
                # GQA：Q 头按整除分组映射到 KV 头
                k_req = k_req.repeat_interleave(group, dim=1)
                v_req = v_req.repeat_interleave(group, dim=1)

            # [T, H, D] -> [1, H, T, D]
            q_t = q_req.transpose(0, 1).unsqueeze(0)
            k_t = k_req.transpose(0, 1).unsqueeze(0)
            v_t = v_req.transpose(0, 1).unsqueeze(0)

            if cached_len == 0:
                out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
            elif q_len == 1:
                # decode：单个 query 可以看到全部历史
                out = F.scaled_dot_product_attention(q_t, k_t, v_t)
            else:
                # extend prefill：第 i 个 Q 只能看到位置 <= cached_len + i 的 K
                mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device).tril(
                    diagonal=cached_len
                )
                out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask)

            outputs.append(out.squeeze(0).transpose(0, 1))

        return torch.cat(outputs, dim=0)

    def prepare_metadata(self, batch: Batch) -> None:
        ctx = get_global_ctx()
        reqs = batch.padded_reqs
        q_seq_lens = [req.extend_len for req in reqs]
        kv_seq_lens = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_kv_len = max(kv_seq_lens)

        page_table = torch.stack([ctx.page_table[req.table_idx, :max_kv_len] for req in reqs])
        offsets = torch.tensor(
            [0] + q_seq_lens, dtype=torch.long, device=ctx.page_table.device
        ).cumsum(dim=0)
        last_indices = offsets[1:] - 1

        batch.attn_metadata = TorchAttnMetadata(
            q_seq_lens=q_seq_lens,
            kv_seq_lens=kv_seq_lens,
            cached_lens=cached_lens,
            page_table=page_table,
            last_indices=last_indices,
        )
