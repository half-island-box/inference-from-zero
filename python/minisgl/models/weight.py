"""safetensors 权重加载与 key 映射。

HF checkpoint 里 q/k/v 与 gate/up 是分开的，运行时模型把它们融合成
``qkv_proj`` / ``gate_up_proj``。加载过程按文件流式进行，融合缓冲只保留同一
组的几个张量。
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterator, Mapping

import safetensors
import torch

# 分开的投影 -> (融合后的 key, 当前张量在融合结果中的槽位, 槽位顺序)
_MERGE_GROUPS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    ".q_proj": (".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", "v", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": (".gate_up_proj", "up", ("gate", "up")),
}


def resolve_model_path(model: str, model_source: str = "modelscope") -> str:
    if os.path.isdir(model):
        return model
    if model_source == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as exc:  # pragma: no cover - depends on install
            raise SystemExit("modelscope is required: uv pip install -e '.[modelscope]'") from exc
        return str(snapshot_download(model))
    if model_source == "huggingface":
        from huggingface_hub import snapshot_download

        return str(snapshot_download(model, allow_patterns=["*.safetensors", "*.json", "*.txt"]))
    raise ValueError(f"unknown model source: {model_source!r}")


def _merge_target(key: str) -> tuple[str, str, tuple[str, ...]] | None:
    for suffix, (merged_suffix, slot, slots) in _MERGE_GROUPS.items():
        if suffix in key:
            return key.replace(suffix, merged_suffix), slot, slots
    return None


def _skip(key: str) -> bool:
    return ".rotary_emb." in key or key.endswith(".inv_freq")


def convert_hf_state_dict(
    state_dict: Mapping[str, torch.Tensor], *, tie_word_embeddings: bool = False
) -> dict[str, torch.Tensor]:
    """把 HF checkpoint 的 key 转成运行时 key，并融合 QKV / gate-up。"""
    result: dict[str, torch.Tensor] = {}
    buffers: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        if _skip(key):
            continue
        if tie_word_embeddings and key.startswith("lm_head."):
            continue
        target = _merge_target(key)
        if target is None:
            result[key] = tensor
            continue

        merged_key, slot, slots = target
        buffers.setdefault(merged_key, {})[slot] = tensor
        if len(buffers[merged_key]) == len(slots):
            result[merged_key] = torch.cat([buffers[merged_key][s] for s in slots], dim=0)
            del buffers[merged_key]

    if buffers:
        raise RuntimeError(f"incomplete merge groups: {sorted(buffers)}")
    return result


def load_weight(
    model: str,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    model_source: str = "modelscope",
    tie_word_embeddings: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    """流式读取目录下所有 safetensors，产出 ``(运行时 key, 设备上的张量)``。"""
    folder = resolve_model_path(model, model_source)
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors found under {folder}")

    buffers: dict[str, dict[str, torch.Tensor]] = {}
    for file in files:
        with safetensors.safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if _skip(key) or (tie_word_embeddings and key.startswith("lm_head.")):
                    continue
                target = _merge_target(key)
                if target is None:
                    yield key, handle.get_tensor(key).to(device=device, dtype=dtype)
                    continue

                merged_key, slot, slots = target
                buffers.setdefault(merged_key, {})[slot] = handle.get_tensor(key)
                if len(buffers[merged_key]) == len(slots):
                    merged = torch.cat([buffers[merged_key][s] for s in slots], dim=0)
                    del buffers[merged_key]
                    yield merged_key, merged.to(device=device, dtype=dtype)

    if buffers:
        raise RuntimeError(f"incomplete merge groups: {sorted(buffers)}")
