"""特性开关：通过 ``MINISGL_*`` 环境变量覆盖默认值。"""

from __future__ import annotations

import os


def _env_str(name: str, default: str) -> str:
    return os.getenv(f"MINISGL_{name}", default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(f"MINISGL_{name}", str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(f"MINISGL_{name}")
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


class Env:
    """集中读取环境变量；阶段 3 只保留调度/缓存相关开关。"""

    def __init__(self) -> None:
        self.CACHE_TYPE = _env_str("CACHE_TYPE", "naive")
        self.ATTENTION_BACKEND = _env_str("ATTENTION_BACKEND", "torch")
        self.MAX_EXTEND_TOKENS = _env_int("MAX_EXTEND_TOKENS", 8192)
        self.CHECK_INTEGRITY = _env_bool("CHECK_INTEGRITY", True)

    def __str__(self) -> str:
        return (
            f"Env(cache_type={self.CACHE_TYPE}, attention_backend={self.ATTENTION_BACKEND}, "
            f"max_extend_tokens={self.MAX_EXTEND_TOKENS}, check_integrity={self.CHECK_INTEGRITY})"
        )


ENV = Env()
