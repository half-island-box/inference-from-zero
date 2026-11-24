"""单进程离线接口：LLM.generate() 内部驱动 Scheduler。"""

from __future__ import annotations

from typing import Any, cast

import torch

from minisgl.core import SamplingParams
from minisgl.engine import Engine
from minisgl.models import (
    BaseLLMModel,
    ModelConfig,
    create_model,
    load_model_weights,
    load_weight,
    resolve_model_path,
)
from minisgl.scheduler.scheduler import Scheduler


class LLM:
    """离线批量生成。

    既可以从 ``model_path`` 加载模型，也可以直接传入已构造好的 ``model`` +
    ``model_config``（CPU 测试用）。
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        model: BaseLLMModel | None = None,
        model_config: ModelConfig | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_source: str = "modelscope",
        page_size: int = 1,
        max_running_req: int = 16,
        num_pages: int | None = None,
        max_seq_len: int | None = None,
        max_extend_tokens: int = 8192,
        memory_ratio: float = 0.9,
        cache_type: str = "naive",
        attention_backend: str = "torch",
        eos_token_id: int | None = None,
        tokenizer: Any = None,
        record_prefill_logits: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = tokenizer

        if model is None:
            if model_path is None:
                raise ValueError("either model or model_path must be provided")
            model, model_config, path, tokenizer = self._load_pretrained(
                model_path, model_source, self.device, dtype
            )
            self.tokenizer = tokenizer
        else:
            path = model_path or ""
            if model_config is None:
                raise ValueError("model_config is required when model is provided")

        assert model_config is not None
        if num_pages is None:
            if self.device.type == "cuda":
                free_bytes, _ = torch.cuda.mem_get_info()
                num_pages = Engine.num_pages_for(
                    model_config,
                    free_bytes=free_bytes,
                    page_size=page_size,
                    dtype=dtype,
                    memory_ratio=memory_ratio,
                )
            else:
                num_pages = 1024
        if eos_token_id is None and self.tokenizer is not None:
            eos_token_id = self.tokenizer.eos_token_id

        self.engine = Engine(
            model,
            model_config,
            num_pages=num_pages,
            page_size=page_size,
            max_running_req=max_running_req,
            max_seq_len=max_seq_len,
            device=self.device,
            dtype=dtype,
            attn_backend=attention_backend,
        )
        self.scheduler = Scheduler(
            self.engine,
            max_extend_tokens=max_extend_tokens,
            cache_type=cache_type,
            eos_token_id=eos_token_id,
            record_prefill_logits=record_prefill_logits,
        )
        self.model_path = path

    @property
    def eos_token_id(self) -> int | None:
        return self.scheduler.eos_token_id

    def generate(
        self,
        prompts: list[str] | list[list[int]] | list[torch.Tensor],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(sampling_params, SamplingParams) or sampling_params is None:
            params = sampling_params or SamplingParams()
            params_list = [params] * len(prompts)
        else:
            params_list = list(sampling_params)
            if len(params_list) != len(prompts):
                raise ValueError("sampling_params length must match prompts")

        if not self.scheduler.is_idle:
            self.scheduler.run_until_complete()
        self.scheduler.clear_results()

        uids: list[int] = []
        for prompt, params in zip(prompts, params_list):
            uids.append(self.scheduler.add_request(self._to_token_ids(prompt), params))
        self.scheduler.run_until_complete()

        results: list[dict[str, Any]] = []
        for uid in uids:
            state = self.scheduler.states[uid]
            results.append(
                {
                    "token_ids": list(state.output_ids),
                    "stop_reason": state.stop_reason,
                    "text": (
                        self.tokenizer.decode(state.output_ids)
                        if self.tokenizer is not None
                        else None
                    ),
                    "prefill_logits": state.prefill_logits,
                    "ttft_ms": state.ttft_ms,
                    "itl_ms": state.itl_ms,
                }
            )
        return results

    def _to_token_ids(self, prompt: Any) -> list[int] | torch.Tensor:
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("string prompt requires a tokenizer")
            return cast("list[int]", self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        return cast("list[int] | torch.Tensor", prompt)

    @staticmethod
    def _load_pretrained(
        model_path: str,
        model_source: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[BaseLLMModel, ModelConfig, str, Any]:
        import transformers

        path = resolve_model_path(model_path, model_source)
        hf_config = transformers.AutoConfig.from_pretrained(path)
        config = ModelConfig.from_hf(hf_config)
        model = create_model(config, device="meta", dtype=dtype, rope_device=device)
        load_model_weights(
            model,
            dict(
                load_weight(
                    path,
                    device=device,
                    dtype=dtype,
                    model_source=model_source,
                    tie_word_embeddings=config.tie_word_embeddings,
                )
            ),
            dtype=dtype,
            assign=True,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        return model, config, path, tokenizer
