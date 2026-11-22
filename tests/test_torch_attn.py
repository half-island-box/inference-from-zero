"""参考 attention 后端的 extend prefill 语义。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minisgl.attention import TorchAttnBackend  # noqa: E402
from minisgl.core import (  # noqa: E402
    Batch,
    Context,
    Req,
    SamplingParams,
    set_global_ctx,
)
from minisgl.kvcache import create_kvcache_pool  # noqa: E402
from minisgl.kvcache.naive_cache import NaiveCacheHandle  # noqa: E402
from minisgl.models import ModelConfig  # noqa: E402


def _config() -> ModelConfig:
    # 后端只关心 head 数，用最小配置构造 KV 池
    from minisgl.models.config import RotaryConfig

    return ModelConfig(
        num_layers=1,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=8,
        hidden_size=32,
        vocab_size=64,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(8, 8, 128, 10000.0, None),
        hidden_act="silu",
        tie_word_embeddings=False,
        attention_bias=False,
        model_type="llama",
        architectures=("LlamaForCausalLM",),
    )


@pytest.mark.parametrize("page_size", [1, 4])
def test_extend_prefill_matches_full_sdpa(page_size: int) -> None:
    torch.manual_seed(0)
    config = _config()
    ctx = Context(page_size=page_size)
    ctx.kv_cache = create_kvcache_pool(
        config, num_pages=8, page_size=page_size, dtype=torch.float32, device=torch.device("cpu")
    )
    ctx.page_table = torch.zeros((2, 64), dtype=torch.int32)
    ctx.attn_backend = TorchAttnBackend()
    set_global_ctx(ctx, force=True)

    cached_len, new_len = 3, page_size + 1
    total_len = cached_len + new_len
    req = Req(
        input_ids=torch.arange(1, total_len + 1),
        table_idx=0,
        cached_len=cached_len,
        output_len=1,
        uid=0,
        sampling_params=SamplingParams(max_tokens=1),
        cache_handle=NaiveCacheHandle(),
    )
    batch = Batch(reqs=[req], phase="prefill")
    # 手工把 [0, total_len) 的逻辑位置映射到从 0 开始的物理 token 槽位
    ctx.page_table[0, :total_len] = torch.arange(total_len, dtype=torch.int32)
    batch.padded_reqs = [req]
    batch.out_loc = ctx.page_table[req.table_idx, cached_len:total_len].to(torch.long)

    hq, hkv, dim = 4, 2, 8
    k_prefix = torch.randn(cached_len, hkv, dim)
    v_prefix = torch.randn(cached_len, hkv, dim)
    prefix_slots = ctx.page_table[0, :cached_len].to(torch.long)
    ctx.kv_cache.store_kv(k_prefix, v_prefix, prefix_slots, layer_id=0)

    q = torch.randn(new_len, hq, dim)
    k_new = torch.randn(new_len, hkv, dim)
    v_new = torch.randn(new_len, hkv, dim)

    backend = TorchAttnBackend()
    backend.prepare_metadata(batch)
    actual = backend.forward(q, k_new, v_new, layer_id=0, batch=batch)

    # 参考实现：完整 K/V + 偏移 causal mask
    k_full = torch.cat([k_prefix, k_new])
    v_full = torch.cat([v_prefix, v_new])
    k_full = k_full.repeat_interleave(hq // hkv, dim=1)
    v_full = v_full.repeat_interleave(hq // hkv, dim=1)
    mask = torch.ones(new_len, total_len, dtype=torch.bool).tril(diagonal=cached_len)
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k_full.transpose(0, 1).unsqueeze(0),
            v_full.transpose(0, 1).unsqueeze(0),
            attn_mask=mask,
        )
        .squeeze(0)
        .transpose(0, 1)
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_requests_sharing_physical_page_are_isolated() -> None:
    torch.manual_seed(0)
    config = _config()
    ctx = Context(page_size=4)
    ctx.kv_cache = create_kvcache_pool(
        config, num_pages=8, page_size=4, dtype=torch.float32, device=torch.device("cpu")
    )
    ctx.page_table = torch.zeros((3, 64), dtype=torch.int32)
    ctx.attn_backend = TorchAttnBackend()
    set_global_ctx(ctx, force=True)

    # 两个请求故意共享同一个物理页（page 0）的不同槽位
    ctx.page_table[0, :2] = torch.tensor([0, 1], dtype=torch.int32)
    ctx.page_table[1, :2] = torch.tensor([2, 3], dtype=torch.int32)
    reqs = [
        Req(
            input_ids=torch.tensor([1, 2]),
            table_idx=table_idx,
            cached_len=0,
            output_len=1,
            uid=table_idx,
            sampling_params=SamplingParams(max_tokens=1),
            cache_handle=NaiveCacheHandle(),
        )
        for table_idx in (0, 1)
    ]
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    batch.out_loc = torch.cat([ctx.page_table[req.table_idx, :2] for req in reqs]).to(torch.long)

    hq, hkv, dim = 4, 2, 8
    q = torch.randn(4, hq, dim)
    k_new = torch.randn(4, hkv, dim)
    v_new = torch.randn(4, hkv, dim)
    backend = TorchAttnBackend()
    backend.prepare_metadata(batch)
    actual = backend.forward(q, k_new, v_new, layer_id=0, batch=batch)

    for i in range(2):
        qi = q[i * 2 : i * 2 + 2].transpose(0, 1).unsqueeze(0)
        ki = k_new[i * 2 : i * 2 + 2].repeat_interleave(hq // hkv, dim=1)
        vi = v_new[i * 2 : i * 2 + 2].repeat_interleave(hq // hkv, dim=1)
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                qi,
                ki.transpose(0, 1).unsqueeze(0),
                vi.transpose(0, 1).unsqueeze(0),
                is_causal=True,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
        torch.testing.assert_close(actual[i * 2 : i * 2 + 2], expected, atol=1e-6, rtol=1e-5)
