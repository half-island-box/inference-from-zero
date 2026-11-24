"""阶段 5 前缀缓存：radix 树单测与 cache on/off 端到端一致性（CPU）。"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from minisgl.core import Req, SamplingParams  # noqa: E402
from minisgl.engine import naive_generate  # noqa: E402
from minisgl.kvcache.radix_cache import RadixPrefixCache  # noqa: E402
from minisgl.llm import LLM  # noqa: E402
from minisgl.models import ModelConfig, convert_hf_state_dict, create_model  # noqa: E402
from minisgl.scheduler.cache import CacheManager  # noqa: E402


def _tiny_pair():
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    hf_model = transformers.AutoModelForCausalLM.from_config(
        config, attn_implementation="sdpa"
    ).eval()
    model_config = ModelConfig.from_hf(hf_model.config)
    model = create_model(model_config, architecture="LlamaForCausalLM")
    model.load_state_dict(
        convert_hf_state_dict(
            hf_model.state_dict(), tie_word_embeddings=model_config.tie_word_embeddings
        )
    )
    return model, model_config


def _make_llm(model, config, **overrides) -> LLM:
    values = {
        "device": "cpu",
        "dtype": torch.float32,
        "page_size": 4,
        "max_running_req": 8,
        "num_pages": 256,
        "max_seq_len": 256,
        "max_extend_tokens": 64,
        "cache_type": "radix",
    }
    values.update(overrides)
    return LLM(model=model, model_config=config, **values)


def _slots(length: int, page_size: int, base: int = 0) -> torch.Tensor:
    values: list[int] = []
    for page in range((length + page_size - 1) // page_size):
        start = base + page * page_size
        values.extend(range(start, start + page_size))
    return torch.tensor(values[:length], dtype=torch.int32)


def _assert_matches_naive(model, prompts, params, results) -> None:
    for prompt, param, result in zip(prompts, params, results):
        expected = naive_generate(model, prompt, param, eos_token_id=None)
        assert result["token_ids"] == expected.output_ids
        assert result["stop_reason"] == expected.stop_reason


@pytest.mark.parametrize("page_size", [1, 4])
def test_insert_match_split_and_evict(page_size: int) -> None:
    cache = RadixPrefixCache(device=torch.device("cpu"), page_size=page_size)
    ids = torch.arange(1, 13, dtype=torch.long)
    slots = _slots(12, page_size)

    assert cache.match_prefix(ids).cuda_handle.cached_len == 0
    inserted = cache.insert_prefix(ids, slots)
    assert inserted.cached_len == 0
    assert inserted.handle.cached_len == 12
    assert cache.size_info.total_size == 12

    matched = cache.match_prefix(ids).cuda_handle
    assert matched.cached_len == 12
    assert matched.get_matched_indices().tolist() == slots.tolist()

    # 前 8 个 token 相同：节点在 8 处分裂（向下对齐到页）
    other = ids.clone()
    other[8] = 999
    partial = cache.match_prefix(other).cuda_handle
    assert partial.cached_len == 8
    assert cache.size_info.total_size == 12

    cache.lock_handle(partial)
    assert cache.size_info.protected_size == 8
    assert cache.size_info.evictable_size == 4
    cache.lock_handle(partial, unlock=True)
    assert cache.size_info.protected_size == 0
    assert cache.size_info.total_size == 12

    # 插入新分支后，尾部可以被淘汰
    inserted_other = cache.insert_prefix(other, _slots(12, page_size, base=500))
    assert inserted_other.cached_len == 8
    assert cache.size_info.total_size == 12 + (12 - 8)
    # 淘汰以整节点为单位，可能比请求的 size 更大
    evicted = cache.evict(page_size)
    assert len(evicted) >= page_size and len(evicted) % page_size == 0
    assert cache.size_info.total_size == 16 - len(evicted)
    cache.check_integrity()


@pytest.mark.parametrize("page_size", [1, 4, 8])
def test_radix_matches_naive(page_size: int) -> None:
    model, config = _tiny_pair()
    rng = random.Random(1)
    shared = [rng.randrange(1, 64) for _ in range(12)]
    prompts = [shared + [rng.randrange(1, 64) for _ in range(rng.randint(1, 10))]]
    prompts += [[rng.randrange(1, 64) for _ in range(rng.randint(1, 24))] for _ in range(14)]
    params = [SamplingParams(max_tokens=rng.randint(1, 4)) for _ in prompts]

    llm = _make_llm(model, config, page_size=page_size)
    results = llm.generate(prompts, params)
    _assert_matches_naive(model, prompts, params, results)
    llm.scheduler.check_integrity()
    assert llm.scheduler.cache_manager.prefix_cache.size_info.total_size > 0


def test_shared_prefix_is_reused() -> None:
    model, config = _tiny_pair()
    rng = random.Random(2)
    shared = [rng.randrange(1, 64) for _ in range(17)]
    first_prompt = shared + [rng.randrange(1, 64) for _ in range(3)]
    second_prompt = shared + [rng.randrange(1, 64) for _ in range(5)]
    params = SamplingParams(max_tokens=3)

    llm = _make_llm(model, config, page_size=4)
    first, second = llm.generate([first_prompt, second_prompt], [params, params])
    _assert_matches_naive(model, [first_prompt, second_prompt], [params, params], [first, second])

    # 第二次请求的输入（去掉最后一个 token）至少命中一整页共享前缀
    matched = llm.scheduler.cache_manager.match_prefix(torch.tensor(second_prompt))
    expected = (len(shared) + 4) // 4 * 4
    assert matched.cuda_handle.cached_len >= expected
    assert matched.cuda_handle.cached_len % 4 == 0
    llm.scheduler.check_integrity()


def test_cache_eviction_under_tight_pages() -> None:
    model, config = _tiny_pair()
    rng = random.Random(3)
    prompts = [[rng.randrange(1, 64) for _ in range(20)] for _ in range(8)]
    params = [SamplingParams(max_tokens=2) for _ in prompts]

    # 64 个 token 的容量装不下 8×22 个 token，必然触发淘汰
    llm = _make_llm(model, config, page_size=4, num_pages=16, max_running_req=4)
    results = llm.generate(prompts, params)
    _assert_matches_naive(model, prompts, params, results)
    llm.scheduler.check_integrity()


def test_cache_req_rebinds_shared_prefix_page_table() -> None:
    """同批 prefill 的两条同前缀请求：后插入者要改用共享槽位，再释放自己的重复页。"""
    page_size = 1
    page_table = torch.zeros((4, 32), dtype=torch.int32)
    manager = CacheManager(8, page_size, page_table, type="radix")
    ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    reqs = []
    for uid in (0, 1):
        handle = manager.match_prefix(ids).cuda_handle
        manager.lock(handle)
        req = Req(
            input_ids=ids.clone(),
            table_idx=uid,
            cached_len=0,
            output_len=2,
            uid=uid,
            sampling_params=SamplingParams(max_tokens=2),
            cache_handle=handle,
        )
        manager.allocate_paged([req])
        req.cached_len = req.device_len
        reqs.append(req)

    first, second = reqs
    first_slots = page_table[0, :4].clone()
    second_own_slots = page_table[1, :4].clone()

    manager.cache_req(first, finished=False)
    manager.cache_req(second, finished=False)

    # 第二条请求的页表已改指向缓存中的共享槽位
    assert torch.equal(page_table[1, :4], first_slots)
    # 它自己重复分配的页被归还，可以在显存压力下安全复用
    assert all(int(slot) in manager.free_slots.tolist() for slot in second_own_slots)
    assert len(manager.free_slots) == 4
    manager.check_integrity()


@pytest.mark.parametrize("page_size", [1, 4, 8])
def test_two_hundred_requests_integrity(page_size: int) -> None:
    model, config = _tiny_pair()
    rng = random.Random(5)
    prompts = [[rng.randrange(1, 64) for _ in range(rng.randint(1, 24))] for _ in range(200)]
    params = [SamplingParams(max_tokens=rng.randint(1, 3)) for _ in prompts]

    # 容量远小于总 token 数，强制反复淘汰
    llm = _make_llm(model, config, page_size=page_size, num_pages=128, max_running_req=16)
    results = llm.generate(prompts, params)
    _assert_matches_naive(model, prompts, params, results)
    llm.scheduler.check_integrity()


@pytest.mark.parametrize("page_size", [1, 8])
def test_chunked_prefill_with_radix(page_size: int) -> None:
    model, config = _tiny_pair()
    rng = random.Random(4)
    shared = [rng.randrange(1, 64) for _ in range(10)]
    prompts = [shared + [rng.randrange(1, 64) for _ in range(length)] for length in (7, 13, 21)]
    params = [SamplingParams(max_tokens=3) for _ in prompts]

    llm = _make_llm(model, config, page_size=page_size, max_extend_tokens=3)
    results = llm.generate(prompts, params)
    _assert_matches_naive(model, prompts, params, results)
    llm.scheduler.check_integrity()


def test_max_extend_tokens_with_radix_and_zero_output() -> None:
    model, config = _tiny_pair()
    prompt = [5] * 9
    params = SamplingParams(max_tokens=0)

    llm = _make_llm(model, config, max_extend_tokens=2, max_running_req=2)
    result = llm.generate([prompt], [params])[0]
    _assert_matches_naive(model, [prompt], [params], [result])
    llm.scheduler.check_integrity()
