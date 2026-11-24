"""模型下载目录解析（CPU）：minisgl 与 tools 两侧保持一致。"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")

from minisgl.models.weight import model_cache_dir  # noqa: E402

from tools.hf_runtime import model_cache_dir as tool_model_cache_dir  # noqa: E402


def test_default_is_home_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINISGL_MODEL_DIR", raising=False)
    expected = os.path.expanduser("~/data")
    assert model_cache_dir() == expected
    assert tool_model_cache_dir() == expected


def test_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setenv("MINISGL_MODEL_DIR", str(tmp_path))
    assert model_cache_dir() == str(tmp_path)
    assert tool_model_cache_dir() == str(tmp_path)
