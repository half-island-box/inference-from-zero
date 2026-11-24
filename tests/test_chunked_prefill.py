"""阶段 4 分块预填充（CPU）：任意 chunk 预算 / 页大小下的语义一致性。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from minisgl.core import SamplingParams  # noqa: E402
from minisgl.engine import naive_generate  # noqa: E402
from minisgl.llm import LLM  # noqa: E402
from minisgl.models import ModelConfig, convert_hf_state_dict, create_model  # noqa: E402


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
        "num_pages": 1024,
        "max_seq_len": 256,
        "max_extend_tokens": 64,
    }
    values.update(overrides)
    return LLM(model=model, model_config=config, **values)


def _assert_matches_naive(model, prompts, params, results) -> None:
    for prompt, param, result in zip(prompts, params, results):
        expected = naive_generate(model, prompt, param, eos_token_id=None)
        assert result["token_ids"] == expected.output_ids
        assert result["stop_reason"] == expected.stop_reason


@pytest.mark.parametrize("budget", [1, 3, 8, 64])
@pytest.mark.parametrize("page_size", [1, 4, 8])
def test_chunked_matches_unchunked(budget: int, page_size: int) -> None:
    model, config = _tiny_pair()
    prompts = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [11] * 20, [3], [5] * 33]
    params = [SamplingParams(max_tokens=3) for _ in prompts]

    llm = _make_llm(model, config, page_size=page_size, max_extend_tokens=budget)
    results = llm.generate(prompts, params)
    _assert_matches_naive(model, prompts, params, results)
    llm.scheduler.check_integrity()


@pytest.mark.parametrize("page_size", [4, 8])
def test_chunk_boundary_across_pages(page_size: int) -> None:
    model, config = _tiny_pair()
    prompt = [7] * 23  # chunk=3 时每个 chunk 都会跨越页边界
    params = SamplingParams(max_tokens=2)

    llm = _make_llm(model, config, page_size=page_size, max_extend_tokens=3, max_running_req=2)
    results = llm.generate([prompt], [params])
    _assert_matches_naive(model, [prompt], [params], results)
    llm.scheduler.check_integrity()


def test_chunked_prefill_logits_match_full_prefill() -> None:
    model, config = _tiny_pair()
    prompt = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    params = SamplingParams(max_tokens=2)

    # Engine 的 Context 是全局单例，先建先用的 LLM 跑完再建第二个
    full = _make_llm(model, config, max_extend_tokens=4096, record_prefill_logits=True)
    full_result = full.generate([prompt], [params])[0]
    chunked = _make_llm(model, config, max_extend_tokens=3, record_prefill_logits=True)
    chunked_result = chunked.generate([prompt], [params])[0]

    assert chunked_result["token_ids"] == full_result["token_ids"]
    assert chunked_result["prefill_logits"] is not None
    # 分块只改变计算顺序，最后一个 chunk 的 last logits 应与整段 prefill 一致
    assert torch.allclose(
        chunked_result["prefill_logits"], full_result["prefill_logits"], atol=1e-5
    )
    chunked.scheduler.check_integrity()
    full.scheduler.check_integrity()


def test_abort_mid_chunk_frees_resources() -> None:
    model, config = _tiny_pair()
    llm = _make_llm(model, config, max_extend_tokens=4)
    uid = llm.scheduler.add_request([1] * 20, SamplingParams(max_tokens=5))

    assert llm.scheduler.step()
    pending = llm.scheduler.prefill_manager.pending_list[0]
    assert pending.uid == uid and pending.chunked_req is not None

    assert llm.scheduler.abort_request(uid)
    llm.scheduler.run_until_complete()
    llm.scheduler.check_integrity()
    assert llm.scheduler.table_manager.available_size == llm.engine.max_running_req
    assert len(llm.scheduler.cache_manager.free_slots) == llm.engine.num_pages


def test_interleave_runs_decode_between_chunks() -> None:
    model, config = _tiny_pair()
    llm = _make_llm(
        model, config, max_extend_tokens=4, sched_policy="interleave", max_running_req=4
    )
    dec_prompt, dec_param = [1, 2], SamplingParams(max_tokens=8)
    long_prompt, long_param = [5] * 24, SamplingParams(max_tokens=4)

    dec_uid = llm.scheduler.add_request(dec_prompt, dec_param)
    assert llm.scheduler.step()
    assert llm.scheduler.decode_manager.runnable

    long_uid = llm.scheduler.add_request(long_prompt, long_param)
    phases: list[str | None] = []
    for _ in range(6):
        if not llm.scheduler.step():
            break
        phases.append(llm.scheduler.last_phase)
    assert "prefill" in phases and "decode" in phases

    llm.scheduler.run_until_complete()
    states = [llm.scheduler.states[uid] for uid in (dec_uid, long_uid)]
    results = [
        {"token_ids": state.output_ids, "stop_reason": state.stop_reason} for state in states
    ]
    _assert_matches_naive(model, [dec_prompt, long_prompt], [dec_param, long_param], results)
    llm.scheduler.check_integrity()


def test_budget_one_makes_progress_without_livelock() -> None:
    model, config = _tiny_pair()
    llm = _make_llm(model, config, max_extend_tokens=1, max_running_req=2)
    prompt, param = [2] * 10, SamplingParams(max_tokens=3)
    uid = llm.scheduler.add_request(prompt, param)

    steps = 0
    while llm.scheduler.step():
        steps += 1
    assert steps >= len(prompt)

    expected = naive_generate(model, prompt, param, eos_token_id=None)
    assert llm.scheduler.states[uid].output_ids == expected.output_ids
    llm.scheduler.check_integrity()


def test_chunked_zero_max_tokens() -> None:
    model, config = _tiny_pair()
    llm = _make_llm(model, config, max_extend_tokens=2, max_running_req=2)
    prompt, param = [4] * 6, SamplingParams(max_tokens=0)

    result = llm.generate([prompt], [param])[0]
    _assert_matches_naive(model, [prompt], [param], [result])
    llm.scheduler.check_integrity()
