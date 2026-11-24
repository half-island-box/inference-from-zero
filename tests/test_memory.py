"""显存统计必须显式绑定设备：--device cuda:N 时不能读 0 号卡的统计。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tools.memory import peak_memory_gb, reset_peak_memory  # noqa: E402


def test_peak_passes_device(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []

    def fake(device=None):  # noqa: ANN001
        calls.append(device)
        return 2_500_000_000

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", fake)
    assert peak_memory_gb("cuda:1") == 2.5
    assert calls == [torch.device("cuda:1")]


def test_peak_none_on_cpu() -> None:
    assert peak_memory_gb("cpu") is None


def test_reset_passes_device(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device=None: calls.append(device)
    )
    reset_peak_memory(torch.device("cuda:2"))
    assert calls == [torch.device("cuda:2")]


def test_reset_noop_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device=None: calls.append(device)
    )
    reset_peak_memory("cpu")
    assert calls == []


def test_llm_free_memory_uses_device(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("transformers")
    from minisgl.llm.llm import _free_memory_bytes

    calls: list = []

    def fake(device=None):  # noqa: ANN001
        calls.append(device)
        return 123, 456

    monkeypatch.setattr(torch.cuda, "mem_get_info", fake)
    assert _free_memory_bytes(torch.device("cuda:4")) == 123
    assert calls == [torch.device("cuda:4")]
