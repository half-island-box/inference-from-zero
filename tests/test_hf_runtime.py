"""用小随机模型对 HF greedy 解码器做 CPU 冒烟测试。

无需 GPU 或下载权重，可在开发机上直接发现 cache 使用和停止语义的问题。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tools.hf_runtime import (  # noqa: E402
    GreedyDecoder,
    HFRuntime,
    forward_last_logits,
    greedy_generate,
)

VOCAB_SIZE = 64


def _tiny_runtime() -> HFRuntime:
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = transformers.LlamaForCausalLM(config).eval()
    return HFRuntime(
        model=model,
        tokenizer=None,
        model_name="tiny",
        model_path="tiny",
        device="cpu",
        dtype="float32",
        eos_token_id=None,
    )


def test_greedy_is_deterministic_and_length_bounded() -> None:
    runtime = _tiny_runtime()
    first, reason = greedy_generate(runtime, [1, 2, 3, 4], max_tokens=5)
    second, reason_again = greedy_generate(runtime, [1, 2, 3, 4], max_tokens=5)

    assert first == second
    assert reason == reason_again == "length"
    assert len(first) == 5


def test_eos_is_not_emitted_and_stops_generation() -> None:
    runtime = _tiny_runtime()
    decoder = GreedyDecoder(runtime, [1, 2, 3], max_tokens=5)
    decoder.prefill()
    runtime.eos_token_id = decoder.peek()

    output, reason = decoder.generate()
    assert output == []
    assert reason == "eos"


def test_zero_max_tokens() -> None:
    runtime = _tiny_runtime()
    output, reason = greedy_generate(runtime, [1, 2, 3], max_tokens=0)
    assert output == []
    assert reason == "length"


def test_forward_last_logits_shape() -> None:
    runtime = _tiny_runtime()
    logits = forward_last_logits(runtime, [1, 2, 3])
    assert logits.shape == (VOCAB_SIZE,)
    assert logits.dtype.name == "float32"
