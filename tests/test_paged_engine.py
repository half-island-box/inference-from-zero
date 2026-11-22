"""阶段 2 批量生成语义（CPU 小模型）：与阶段 1 逐 token 一致。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from minisgl.core import SamplingParams  # noqa: E402
from minisgl.engine import naive_generate  # noqa: E402
from minisgl.models import ModelConfig, convert_hf_state_dict, create_model  # noqa: E402
from minisgl.scheduler import BatchRequest, PagedEngine  # noqa: E402


def _tiny_pair(kind: str):
    torch.manual_seed(0)
    if kind == "llama":
        config = transformers.LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
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
            max_position_embeddings=128,
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


def _engine(model, config, *, page_size: int = 1, num_pages: int = 128) -> PagedEngine:
    return PagedEngine(
        model,
        config,
        num_pages=num_pages,
        page_size=page_size,
        max_running_req=8,
        max_seq_len=128,
    )


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_single_request_matches_naive(kind: str) -> None:
    model, config = _tiny_pair(kind)
    prompt = [1, 2, 3, 4]
    params = SamplingParams(max_tokens=6)
    expected = naive_generate(model, prompt, params, eos_token_id=None)

    result = _engine(model, config).generate([BatchRequest(prompt, params)], eos_token_id=None)
    assert result.results[0].output_ids == expected.output_ids
    assert result.results[0].stop_reason == expected.stop_reason


def test_batch_matches_individual_runs() -> None:
    model, config = _tiny_pair("qwen3")
    prompts = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12], [13]]
    params = [SamplingParams(max_tokens=n) for n in (6, 3, 5, 1)]
    expected = [
        naive_generate(model, prompt, param, eos_token_id=None).output_ids
        for prompt, param in zip(prompts, params)
    ]

    engine = _engine(model, config, page_size=4)
    run = engine.generate(
        [BatchRequest(prompt, param) for prompt, param in zip(prompts, params)],
        eos_token_id=None,
    )
    assert [result.output_ids for result in run.results] == expected
    engine.check_integrity()


@pytest.mark.parametrize("page_size", [1, 4, 16])
def test_page_sizes_match_each_other(page_size: int) -> None:
    model, config = _tiny_pair("qwen3")
    prompts = [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14]]
    params = [SamplingParams(max_tokens=4) for _ in prompts]
    requests = [BatchRequest(prompt, param) for prompt, param in zip(prompts, params)]

    engine = _engine(model, config, page_size=page_size)
    outputs = [r.output_ids for r in engine.generate(requests, eos_token_id=None).results]
    engine.check_integrity()

    reference = _engine(model, config, page_size=1)
    expected = [r.output_ids for r in reference.generate(requests, eos_token_id=None).results]
    assert outputs == expected


def test_eos_semantics_match_naive() -> None:
    model, config = _tiny_pair("llama")
    prompt = [1, 2, 3]
    first = naive_generate(model, prompt, SamplingParams(max_tokens=1), eos_token_id=None)
    eos = first.output_ids[0]

    expected = naive_generate(model, prompt, SamplingParams(max_tokens=5), eos_token_id=eos)
    engine = _engine(model, config)
    result = engine.generate(
        [BatchRequest(prompt, SamplingParams(max_tokens=5))], eos_token_id=eos
    ).results[0]
    assert result.output_ids == expected.output_ids == []
    assert result.stop_reason == "eos"

    forced = engine.generate(
        [BatchRequest(prompt, SamplingParams(max_tokens=5, ignore_eos=True))], eos_token_id=eos
    ).results[0]
    assert len(forced.output_ids) == 5
    assert forced.stop_reason == "length"
    engine.check_integrity()


def test_zero_max_tokens_and_non_greedy() -> None:
    model, config = _tiny_pair("llama")
    engine = _engine(model, config)

    result = engine.generate(
        [BatchRequest([1, 2], SamplingParams(max_tokens=0))], eos_token_id=None
    ).results[0]
    assert result.output_ids == []
    assert result.stop_reason == "length"
    engine.check_integrity()

    with pytest.raises(NotImplementedError):
        engine.generate([BatchRequest([1, 2], SamplingParams(temperature=1.0))])


def test_engine_can_be_reused_and_frees_all_pages() -> None:
    model, config = _tiny_pair("qwen3")
    engine = _engine(model, config, page_size=4, num_pages=64)
    requests = [
        BatchRequest([1, 2, 3, 4], SamplingParams(max_tokens=4)),
        BatchRequest([5, 6], SamplingParams(max_tokens=2)),
    ]

    first = [r.output_ids for r in engine.generate(requests, eos_token_id=None).results]
    engine.check_integrity()
    second = [r.output_ids for r in engine.generate(requests, eos_token_id=None).results]
    engine.check_integrity()

    assert first == second
