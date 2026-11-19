"""naive 生成循环的 CPU 语义测试（小模型，无需 GPU/权重下载）。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from minisgl.core import SamplingParams  # noqa: E402
from minisgl.engine import naive_generate  # noqa: E402
from minisgl.models import ModelConfig, convert_hf_state_dict, create_model  # noqa: E402

from tools.hf_runtime import HFRuntime, greedy_generate  # noqa: E402


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
    return hf_model, model


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_matches_hf_greedy(kind: str) -> None:
    hf_model, model = _tiny_pair(kind)
    runtime = HFRuntime(
        model=hf_model,
        tokenizer=None,
        model_name="tiny",
        model_path="tiny",
        device="cpu",
        dtype="float32",
        eos_token_id=None,
    )
    input_ids = [1, 2, 3, 4]
    expected, reason = greedy_generate(runtime, input_ids, max_tokens=6)
    result = naive_generate(model, input_ids, SamplingParams(max_tokens=6), eos_token_id=None)

    assert result.output_ids == expected
    assert result.stop_reason == reason == "length"
    assert result.prefill_logits is not None
    assert result.prefill_logits.shape == (64,)


def test_eos_is_not_emitted() -> None:
    _, model = _tiny_pair("llama")
    ids = [1, 2, 3]
    first = naive_generate(model, ids, SamplingParams(max_tokens=1), eos_token_id=None)
    eos = first.output_ids[0]

    stop = naive_generate(model, ids, SamplingParams(max_tokens=5), eos_token_id=eos)
    assert stop.output_ids == []
    assert stop.stop_reason == "eos"

    forced = naive_generate(
        model,
        ids,
        SamplingParams(max_tokens=5, ignore_eos=True),
        eos_token_id=eos,
    )
    assert len(forced.output_ids) == 5
    assert forced.stop_reason == "length"


def test_zero_max_tokens() -> None:
    _, model = _tiny_pair("llama")
    result = naive_generate(model, [1, 2], SamplingParams(max_tokens=0))
    assert result.output_ids == []
    assert result.stop_reason == "length"


def test_non_greedy_is_rejected() -> None:
    _, model = _tiny_pair("llama")
    with pytest.raises(NotImplementedError):
        naive_generate(model, [1, 2], SamplingParams(temperature=1.0))
