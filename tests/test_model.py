"""阶段 1 层/模型的纯 CPU 测试：与 HF 数值对齐、权重映射与严格加载。"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from minisgl.layers import RMSNorm, get_rope  # noqa: E402
from minisgl.models import (  # noqa: E402
    ModelConfig,
    convert_hf_state_dict,
    create_model,
    load_weight,
)


def _tiny_llama() -> transformers.PreTrainedModel:
    torch.manual_seed(0)
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
    return transformers.AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").eval()


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


def _build_ours(hf_model: transformers.PreTrainedModel, architecture: str):
    config = ModelConfig.from_hf(hf_model.config)
    state = convert_hf_state_dict(
        hf_model.state_dict(), tie_word_embeddings=config.tie_word_embeddings
    )
    model = create_model(config, architecture=architecture)
    model.load_state_dict(state)
    return model


@pytest.mark.parametrize(
    "factory, architecture",
    [(_tiny_llama, "LlamaForCausalLM"), (_tiny_qwen3, "Qwen3ForCausalLM")],
)
def test_forward_matches_hf(factory, architecture: str) -> None:
    hf_model = factory()
    ours = _build_ours(hf_model, architecture)
    input_ids = torch.tensor([[1, 5, 7, 9, 11]])

    with torch.inference_mode():
        expected = hf_model(input_ids).logits
        actual = ours.forward(input_ids)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_logits_to_keep_matches_last_positions() -> None:
    hf_model = _tiny_llama()
    ours = _build_ours(hf_model, "LlamaForCausalLM")
    input_ids = torch.tensor([[2, 4, 6, 8]])

    with torch.inference_mode():
        full = ours.forward(input_ids)
        last = ours.forward(input_ids, logits_to_keep=1)

    torch.testing.assert_close(last, full[:, -1:, :])


def test_rope_matches_hf() -> None:
    hf_model = _tiny_qwen3()
    rotary_cfg = ModelConfig.from_hf(hf_model.config).rotary_config
    rope = get_rope(
        rotary_cfg.head_dim,
        rotary_cfg.rotary_dim,
        rotary_cfg.max_position,
        rotary_cfg.base,
        None,
        str(torch.get_default_device()),
    )
    positions = torch.tensor([0, 3, 7])
    q = torch.randn(1, 3, 4, rotary_cfg.head_dim)
    k = torch.randn(1, 3, 2, rotary_cfg.head_dim)

    with torch.inference_mode():
        q_out, k_out = rope.forward(positions, q, k)
        cos, sin = hf_model.model.rotary_emb(q, positions.unsqueeze(0))
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        q_hf, k_hf = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin)

    torch.testing.assert_close(q_out, q_hf.transpose(1, 2), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(k_out, k_hf.transpose(1, 2), atol=1e-6, rtol=1e-5)


def test_rmsnorm_matches_hf_formula() -> None:
    torch.manual_seed(0)
    norm = RMSNorm(16, 1e-6)
    norm.weight.copy_(torch.randn(16))
    x = torch.randn(3, 5, 16, dtype=torch.float64)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight
    torch.testing.assert_close(
        norm.forward(x.to(torch.float32)).to(torch.float64), expected, atol=1e-5, rtol=1e-4
    )


def test_convert_keys_and_fusion() -> None:
    hf_model = _tiny_qwen3()
    state = hf_model.state_dict()
    converted = convert_hf_state_dict(state, tie_word_embeddings=True)

    assert "model.layers.0.self_attn.qkv_proj.weight" in converted
    assert "model.layers.0.mlp.gate_up_proj.weight" in converted
    assert "lm_head.weight" not in converted
    expected_qkv = torch.cat(
        [
            state["model.layers.0.self_attn.q_proj.weight"],
            state["model.layers.0.self_attn.k_proj.weight"],
            state["model.layers.0.self_attn.v_proj.weight"],
        ]
    )
    torch.testing.assert_close(converted["model.layers.0.self_attn.qkv_proj.weight"], expected_qkv)


def test_incomplete_merge_group_fails() -> None:
    hf_model = _tiny_llama()
    state = {
        key: value
        for key, value in hf_model.state_dict().items()
        if ".k_proj" not in key and ".v_proj" not in key
    }
    with pytest.raises(RuntimeError, match="incomplete merge groups"):
        convert_hf_state_dict(state)


def test_streaming_weight_loader_matches_conversion(tmp_path: Path) -> None:
    hf_model = _tiny_llama()
    state = hf_model.state_dict()
    keys = sorted(state)
    # 交错切分，确保 q/k/v 分散在不同文件里
    safetensors_torch.save_file({k: state[k] for k in keys[::2]}, tmp_path / "a.safetensors")
    safetensors_torch.save_file({k: state[k] for k in keys[1::2]}, tmp_path / "b.safetensors")

    streamed = dict(load_weight(str(tmp_path), device="cpu", dtype=torch.float32))
    converted = convert_hf_state_dict(state, tie_word_embeddings=False)
    assert set(streamed) == set(converted)
    for key in converted:
        torch.testing.assert_close(streamed[key], converted[key])


def test_strict_load_rejects_unknown_key() -> None:
    hf_model = _tiny_llama()
    config = ModelConfig.from_hf(hf_model.config)
    model = create_model(config, architecture="LlamaForCausalLM")
    state = convert_hf_state_dict(hf_model.state_dict())
    state["model.bogus.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected keys"):
        model.load_state_dict(state)
