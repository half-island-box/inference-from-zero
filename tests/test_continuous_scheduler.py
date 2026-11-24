"""阶段 3 连续批处理与每请求采样的 CPU 测试。"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from minisgl.core import SamplingParams  # noqa: E402
from minisgl.engine import naive_generate  # noqa: E402
from minisgl.llm import LLM  # noqa: E402
from minisgl.models import ModelConfig, convert_hf_state_dict, create_model  # noqa: E402


def _tiny_pair(kind: str = "qwen3"):
    torch.manual_seed(0)
    if kind == "llama":
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
    else:
        config = transformers.Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=256,
            tie_word_embeddings=True,
        )
    hf_model = transformers.AutoModelForCausalLM.from_config(
        config, attn_implementation="sdpa"
    ).eval()
    model_config = ModelConfig.from_hf(hf_model.config)
    model = create_model(
        model_config,
        architecture="LlamaForCausalLM" if kind == "llama" else "Qwen3ForCausalLM",
    )
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
        "num_pages": 512,
        "max_seq_len": 256,
        "max_extend_tokens": 64,
    }
    values.update(overrides)
    return LLM(model=model, model_config=config, **values)


def _mixed_requests(count: int, seed: int = 0):
    rng = random.Random(seed)
    prompts, params = [], []
    for _ in range(count):
        length = rng.randint(1, 40)
        prompts.append([rng.randint(0, 63) for _ in range(length)])
        params.append(SamplingParams(max_tokens=rng.randint(1, 4)))
    return prompts, params


def test_fifty_mixed_requests_match_serial() -> None:
    model, config = _tiny_pair()
    prompts, params = _mixed_requests(50)
    llm = _make_llm(model, config)

    results = llm.generate(prompts, params)
    for prompt, param, result in zip(prompts, params, results):
        expected = naive_generate(model, prompt, param, eos_token_id=None)
        assert result["token_ids"] == expected.output_ids
        assert result["stop_reason"] == expected.stop_reason
    llm.scheduler.check_integrity()


def test_dynamic_arrival_and_abort() -> None:
    model, config = _tiny_pair()
    prompts, params = _mixed_requests(8)
    llm = _make_llm(model, config, max_running_req=4)

    long_uid = llm.scheduler.add_request([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams(max_tokens=30))
    for _ in range(2):
        llm.scheduler.step()
    assert llm.scheduler.abort_request(long_uid)
    llm.scheduler.check_integrity()
    assert llm.scheduler.states[long_uid].stop_reason == "aborted"

    uids = [
        llm.scheduler.add_request(prompt, param) for prompt, param in zip(prompts[:4], params[:4])
    ]
    for _ in range(3):
        llm.scheduler.step()
    uids += [
        llm.scheduler.add_request(prompt, param) for prompt, param in zip(prompts[4:], params[4:])
    ]
    llm.scheduler.run_until_complete()

    for uid, prompt, param in zip(uids, prompts, params):
        state = llm.scheduler.states[uid]
        expected = naive_generate(model, prompt, param, eos_token_id=None)
        assert state.output_ids == expected.output_ids
    llm.scheduler.check_integrity()


def test_max_running_req_queues_instead_of_failing() -> None:
    model, config = _tiny_pair()
    prompts, params = _mixed_requests(10)
    llm = _make_llm(model, config, max_running_req=2)

    results = llm.generate(prompts, params)
    for prompt, param, result in zip(prompts, params, results):
        assert (
            result["token_ids"]
            == naive_generate(model, prompt, param, eos_token_id=None).output_ids
        )
    llm.scheduler.check_integrity()


def test_capacity_admission_under_tight_pages() -> None:
    model, config = _tiny_pair()
    prompts = [[1] * 20] * 10
    params = [SamplingParams(max_tokens=5) for _ in prompts]
    llm = _make_llm(model, config, page_size=4, num_pages=40, max_running_req=8)

    results = llm.generate(prompts, params)
    for prompt, param, result in zip(prompts, params, results):
        assert (
            result["token_ids"]
            == naive_generate(model, prompt, param, eos_token_id=None).output_ids
        )
    llm.scheduler.check_integrity()


@pytest.mark.parametrize("page_size", [1, 8])
def test_page_sizes_match(page_size: int) -> None:
    model, config = _tiny_pair("llama")
    prompts, params = _mixed_requests(10, seed=1)
    llm = _make_llm(model, config, page_size=page_size)
    results = llm.generate(prompts, params)
    for prompt, param, result in zip(prompts, params, results):
        assert (
            result["token_ids"]
            == naive_generate(model, prompt, param, eos_token_id=None).output_ids
        )
    llm.scheduler.check_integrity()


def test_eos_and_ignore_eos() -> None:
    model, config = _tiny_pair("llama")
    prompt = [1, 2, 3]
    first = naive_generate(model, prompt, SamplingParams(max_tokens=1), eos_token_id=None)
    eos = first.output_ids[0]

    llm = _make_llm(model, config, eos_token_id=eos)
    stopped = llm.generate([prompt], [SamplingParams(max_tokens=5)])[0]
    assert stopped["token_ids"] == []
    assert stopped["stop_reason"] == "eos"

    forced = llm.generate([prompt], [SamplingParams(max_tokens=5, ignore_eos=True)])[0]
    assert len(forced["token_ids"]) == 5
    assert forced["stop_reason"] == "length"


def test_non_greedy_sampling_and_mixed_batch() -> None:
    model, config = _tiny_pair()
    prompt = [1, 2, 3, 4]
    greedy = naive_generate(model, prompt, SamplingParams(max_tokens=5), eos_token_id=None)

    llm = _make_llm(model, config)
    top_k = llm.generate([prompt], [SamplingParams(max_tokens=5, temperature=1.0, top_k=1)])[0]
    assert top_k["token_ids"] == greedy.output_ids

    params = [
        SamplingParams(max_tokens=4),
        SamplingParams(max_tokens=4, temperature=1.0, top_p=0.9),
        SamplingParams(max_tokens=4, temperature=0.8, top_k=10, top_p=0.95),
    ]
    results = llm.generate([prompt, [5, 6, 7], [8, 9]], params)
    assert all(len(result["token_ids"]) == 4 for result in results)
    assert (
        results[0]["token_ids"]
        == naive_generate(model, prompt, params[0], eos_token_id=None).output_ids
    )
    llm.scheduler.check_integrity()


def test_zero_max_tokens_and_llm_reuse() -> None:
    model, config = _tiny_pair()
    llm = _make_llm(model, config)

    zero = llm.generate([[1, 2]], [SamplingParams(max_tokens=0)])[0]
    assert zero["token_ids"] == []
    assert zero["stop_reason"] == "length"

    first = llm.generate([[1, 2, 3]], [SamplingParams(max_tokens=3)])[0]["token_ids"]
    second = llm.generate([[1, 2, 3]], [SamplingParams(max_tokens=3)])[0]["token_ids"]
    assert first == second
    llm.scheduler.check_integrity()


def test_thousand_requests_integrity() -> None:
    model, config = _tiny_pair()
    llm = _make_llm(model, config, page_size=1, max_running_req=16, num_pages=4096)

    results = llm.generate([[1]] * 1000, [SamplingParams(max_tokens=1)] * 1000)
    assert len(results) == 1000
    assert all(len(result["token_ids"]) == 1 for result in results)
    llm.scheduler.check_integrity()
