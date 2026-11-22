"""阶段 2 第 0 步：nn.Module 迁移的严格性与注册行为。"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from minisgl.models import (  # noqa: E402
    ModelConfig,
    convert_hf_state_dict,
    create_model,
    load_model_weights,
    load_weight,
)


def _tiny_qwen3() -> transformers.PreTrainedModel:
    torch.manual_seed(0)
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
    return transformers.AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").eval()


def _load_tiny_qwen3():
    hf_model = _tiny_qwen3()
    config = ModelConfig.from_hf(hf_model.config)
    state = convert_hf_state_dict(
        hf_model.state_dict(), tie_word_embeddings=config.tie_word_embeddings
    )
    model = create_model(config)
    model.load_state_dict(state)
    return hf_model, model, state


def test_state_dict_keys_are_unique_and_tied_embedding_once() -> None:
    _, model, _ = _load_tiny_qwen3()
    keys = list(model.state_dict().keys())
    assert len(keys) == len(set(keys))
    assert sum(key.endswith("embed_tokens.weight") for key in keys) == 1
    assert not any(key.startswith("lm_head.") for key in keys)


def test_qk_norm_is_registered_under_attention_layer() -> None:
    _, model, _ = _load_tiny_qwen3()
    keys = model.state_dict()
    assert "model.layers.0.self_attn.attn.q_norm.weight" in keys
    assert "model.layers.0.self_attn.attn.k_norm.weight" in keys
    assert "model.layers.0.self_attn.q_norm.weight" not in keys


def test_rope_buffers_are_non_persistent() -> None:
    _, model, _ = _load_tiny_qwen3()
    assert not any(".rotary" in key or "cos" in key for key in model.state_dict())
    rope = model.model.layers[0].self_attn.attn._rotary  # type: ignore[attr-defined]
    assert rope._cos.device.type == "cpu"


def test_strict_load_reports_missing_unexpected_shape_and_dtype() -> None:
    _, model, state = _load_tiny_qwen3()

    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(missing)

    unexpected = dict(state)
    unexpected["model.bogus.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected keys"):
        model.load_state_dict(unexpected)

    wrong_shape = dict(state)
    key = next(iter(wrong_shape))
    wrong_shape[key] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Shape/dtype mismatch"):
        model.load_state_dict(wrong_shape)

    wrong_dtype = {key: value.to(torch.float64) for key, value in state.items()}
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        load_model_weights(model, wrong_dtype, dtype=torch.float32)


def test_streaming_loader_rewrites_qk_norm_keys(tmp_path: Path) -> None:
    hf_model = _tiny_qwen3()
    state = {
        key: value for key, value in hf_model.state_dict().items() if not key.startswith("lm_head.")
    }
    safetensors_torch.save_file(state, tmp_path / "model.safetensors")

    streamed = dict(
        load_weight(
            str(tmp_path),
            device="cpu",
            dtype=torch.float32,
            tie_word_embeddings=True,
        )
    )
    assert "model.layers.0.self_attn.attn.q_norm.weight" in streamed
    assert "model.layers.0.self_attn.attn.k_norm.weight" in streamed
    assert "lm_head.weight" not in streamed


def test_tied_embedding_checkpoint_lm_head_is_filtered() -> None:
    _, model, state = _load_tiny_qwen3()
    with_lm_head = dict(state)
    with_lm_head["lm_head.weight"] = model.model.embed_tokens.weight.detach().clone()
    load_model_weights(model, with_lm_head, dtype=torch.float32)


def test_meta_construction_and_assign_load() -> None:
    hf_model = _tiny_qwen3()
    config = ModelConfig.from_hf(hf_model.config)
    state = convert_hf_state_dict(
        hf_model.state_dict(), tie_word_embeddings=config.tie_word_embeddings
    )
    model = create_model(config, device="meta", dtype=torch.float32, rope_device="cpu")
    # non-persistent buffer 不能留在 meta
    assert model.model.layers[0].self_attn.attn._rotary._cos.device.type == "cpu"  # type: ignore[attr-defined]

    load_model_weights(model, state, dtype=torch.float32, assign=True)
    assert model.model.embed_tokens.weight.device.type == "cpu"

    input_ids = torch.tensor([[1, 5, 7, 9, 11]])
    with torch.inference_mode():
        expected = hf_model(input_ids).logits
        actual = model.forward(input_ids)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
