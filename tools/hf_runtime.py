"""黄金基线与 benchmark 共用的 HuggingFace 运行时辅助。

torch/transformers 采用惰性导入，因此纯 CPU 工具（workloads、compare）在没有
GPU 环境时依然可以导入。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

SUPPORTED_DTYPES = ("bfloat16", "float16", "float32")


def _import_torch() -> Any:
    import torch

    return torch


def _import_transformers() -> Any:
    import transformers

    return transformers


@dataclass
class HFRuntime:
    model: Any
    tokenizer: Any
    model_name: str
    model_path: str
    device: str
    dtype: str
    eos_token_id: int | None
    meta: dict[str, Any] = field(default_factory=dict)


_DEFAULT_MODEL_DIR = "~/data"


def model_cache_dir() -> str:
    """模型下载根目录：默认 ``~/data``，可用 ``MINISGL_MODEL_DIR`` 覆盖。"""
    return os.path.expanduser(os.getenv("MINISGL_MODEL_DIR", _DEFAULT_MODEL_DIR))


def resolve_model_path(model_name: str, model_source: str) -> str:
    if os.path.isdir(model_name):
        return model_name
    cache_dir = model_cache_dir()
    if model_source == "huggingface":
        from huggingface_hub import snapshot_download

        return str(snapshot_download(model_name, cache_dir=cache_dir))
    if model_source == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as exc:  # pragma: no cover - depends on install
            raise SystemExit("--model-source modelscope requires `pip install modelscope`") from exc
        return str(snapshot_download(model_name, cache_dir=cache_dir))
    raise ValueError(f"unknown model source: {model_source!r}")


def load_hf_runtime(
    model_name: str,
    dtype: str = "bfloat16",
    device: str = "cuda",
    model_source: str = "modelscope",
    attn_implementation: str | None = None,
) -> HFRuntime:
    """以推理模式加载 HF 因果语言模型。

    若请求 CUDA 但不可用，则抛出带说明的 SystemExit：阶段 0 及其之后的所有
    验证都必须在 Linux + NVIDIA 主机上运行。
    """
    torch = _import_torch()
    transformers = _import_transformers()

    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be one of {SUPPORTED_DTYPES}, got {dtype!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Stage 0 requires Linux + NVIDIA GPU; "
            "run this on the GPU host or inside the reference Docker image."
        )

    path = resolve_model_path(model_name, model_source)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, trust_remote_code=False)

    kwargs: dict[str, Any] = {
        "torch_dtype": getattr(torch, dtype),
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = transformers.AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = getattr(model.generation_config, "eos_token_id", None)

    meta: dict[str, Any] = {
        "model": model_name,
        "model_path": path,
        "dtype": dtype,
        "device": device,
        "attn_implementation": attn_implementation,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None
        ),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "eos_token_id": eos_token_id,
    }
    return HFRuntime(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        model_path=path,
        device=device,
        dtype=dtype,
        eos_token_id=eos_token_id,
        meta=meta,
    )


def _forward(
    model: Any,
    input_ids: Any,
    past_key_values: Any = None,
    use_cache: bool = True,
) -> Any:
    """前向计算，只返回最后一个位置的 logits。

    ``logits_to_keep=1`` 避免物化 ``[seq, vocab]`` 的完整 logits，对 32k
    token 的负载尤为重要。旧版 transformers 不支持该参数，因此保留回退路径。
    """
    torch = _import_torch()
    kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": use_cache}
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    with torch.inference_mode():
        try:
            return model(**kwargs, logits_to_keep=1)
        except TypeError:
            return model(**kwargs)


class GreedyDecoder:
    """基于 HF 模型和 KV cache 的增量 greedy 解码。

    采样语义（后续所有阶段复用）：
      * 最多生成 ``max_tokens`` 个 token；
      * 若生成的 token 等于 ``eos_token_id``，该 token *不*输出（阶段 3 要求
        EOS 不能发给客户端），并且停止生成，除非设置了 ``ignore_eos``；
      * ``ignore_eos=True`` 时忽略 EOS，一直生成到 ``max_tokens``。
    """

    def __init__(
        self,
        runtime: HFRuntime,
        input_ids: list[int],
        max_tokens: int,
        ignore_eos: bool = False,
    ) -> None:
        if not input_ids:
            raise ValueError("input_ids must not be empty")
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        self.runtime = runtime
        self.input_ids = list(input_ids)
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.past_key_values: Any = None
        self.logits: Any = None
        self.generated: list[int] = []
        self.stop_reason: str | None = None

    def prefill(self) -> None:
        torch = _import_torch()
        ids = torch.tensor([self.input_ids], dtype=torch.long, device=self.runtime.device)
        out = _forward(self.runtime.model, ids, use_cache=True)
        self.past_key_values = out.past_key_values
        self.logits = out.logits[:, -1, :].float()  # 返回最后一个token的logits

    def peek(self) -> int:
        if self.logits is None:
            raise RuntimeError("call prefill() before peek()")
        return int(self.logits.argmax(dim=-1).item())

    def advance(self, token: int) -> None:
        torch = _import_torch()
        ids = torch.tensor([[token]], dtype=torch.long, device=self.runtime.device)
        out = _forward(
            self.runtime.model, ids, past_key_values=self.past_key_values, use_cache=True
        )
        self.past_key_values = out.past_key_values
        self.logits = out.logits[:, -1, :].float()

    def next_token(self) -> tuple[int | None, str | None]:
        """返回 ``(token, None)`` 或 ``(None, stop_reason)``。"""
        if self.stop_reason is not None:
            return None, self.stop_reason
        if len(self.generated) >= self.max_tokens:
            self.stop_reason = "length"
            return None, self.stop_reason

        token = self.peek()
        if (
            not self.ignore_eos
            and self.runtime.eos_token_id is not None
            and token == self.runtime.eos_token_id
        ):
            self.stop_reason = "eos"
            return None, self.stop_reason

        self.generated.append(token)
        if len(self.generated) >= self.max_tokens:
            self.stop_reason = "length"
            return token, None

        self.advance(token)
        return token, None

    def generate(self) -> tuple[list[int], str]:
        while True:
            token, reason = self.next_token()
            if token is None:
                assert reason is not None
                return self.generated, reason


def greedy_generate(
    runtime: HFRuntime,
    input_ids: list[int],
    max_tokens: int,
    ignore_eos: bool = False,
) -> tuple[list[int], str]:
    decoder = GreedyDecoder(runtime, input_ids, max_tokens, ignore_eos=ignore_eos)
    decoder.prefill()
    return decoder.generate()


def forward_last_logits(runtime: HFRuntime, input_ids: list[int]) -> Any:
    """无 cache 完整前向的最后一个位置 logits（float32）。"""
    import numpy as np

    torch = _import_torch()
    ids = torch.tensor([list(input_ids)], dtype=torch.long, device=runtime.device)
    out = _forward(runtime.model, ids, use_cache=False)
    logits = out.logits[0, -1].float().cpu().numpy().astype(np.float32)
    assert isinstance(logits, np.ndarray)
    return logits
