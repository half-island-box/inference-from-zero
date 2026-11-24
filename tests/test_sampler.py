"""每请求采样参数的单测。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from minisgl.core import SamplingParams  # noqa: E402
from minisgl.engine import BatchSamplingArgs, Sampler, greedy_sample  # noqa: E402


def _batch(params: list[SamplingParams]) -> SimpleNamespace:
    return SimpleNamespace(reqs=[SimpleNamespace(sampling_params=param) for param in params])


def test_greedy_fast_path_builds_no_tensors() -> None:
    sampler = Sampler(torch.device("cpu"), vocab_size=8)
    args = sampler.prepare(
        _batch([SamplingParams(), SamplingParams(top_k=1), SamplingParams(temperature=0.0)])
    )
    assert args.temperatures is None
    logits = torch.tensor([[0.1, 0.5, 0.2], [0.0, -1.0, 2.0]])
    sampled = sampler.sample(logits, args)
    assert torch.equal(sampled, greedy_sample(logits).to(torch.int32))


@pytest.mark.parametrize("top_p", [1e-6, 0.3, 0.7, 1.0])
def test_top_p_restricts_candidates(top_p: float) -> None:
    torch.manual_seed(0)
    sampler = Sampler(torch.device("cpu"), vocab_size=16)
    params = [SamplingParams(temperature=1.0, top_p=top_p)]
    args = sampler.prepare(_batch(params))
    logits = torch.arange(16, dtype=torch.float32).unsqueeze(0)

    # 找出 top_p 允许的 token 集合
    sorted_logits, sorted_indices = torch.sort(logits[0], descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    allowed = set(sorted_indices[(cumulative - probs <= max(top_p, 1e-6))].tolist())

    for _ in range(50):
        token = int(sampler.sample(logits, args)[0].item())
        assert token in allowed


def test_top_k_restricts_candidates() -> None:
    torch.manual_seed(0)
    sampler = Sampler(torch.device("cpu"), vocab_size=16)
    params = [SamplingParams(temperature=1.0, top_k=3)]
    args = sampler.prepare(_batch(params))
    logits = torch.arange(16, dtype=torch.float32).unsqueeze(0)
    allowed = set(torch.topk(logits, 3, dim=-1).indices[0].tolist())

    for _ in range(50):
        assert int(sampler.sample(logits, args)[0].item()) in allowed


def test_mixed_batch_uses_per_request_params() -> None:
    torch.manual_seed(0)
    sampler = Sampler(torch.device("cpu"), vocab_size=8)
    params = [
        SamplingParams(),  # greedy
        SamplingParams(temperature=1.0, top_p=0.5),
    ]
    args = sampler.prepare(_batch(params))
    assert args.temperatures is not None

    logits = torch.tensor(
        [
            [0.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    for _ in range(10):
        tokens = sampler.sample(logits, args).tolist()
        assert tokens == [1, 2]


def test_temperature_scales_logits() -> None:
    torch.manual_seed(0)
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    low = sampler.prepare(_batch([SamplingParams(temperature=0.1)]))
    high = sampler.prepare(_batch([SamplingParams(temperature=10.0)]))
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    torch.manual_seed(0)
    low_samples = [int(sampler.sample(logits, low)[0]) for _ in range(100)]
    torch.manual_seed(0)
    high_samples = [int(sampler.sample(logits, high)[0]) for _ in range(100)]
    # 低温度更集中在 argmax
    assert low_samples.count(3) >= high_samples.count(3)


def test_prepare_returns_none_top_k_and_top_p_when_unneeded() -> None:
    sampler = Sampler(torch.device("cpu"), vocab_size=8)
    args = sampler.prepare(_batch([SamplingParams(temperature=1.0)]))
    assert args.temperatures is not None
    assert args.top_k is None
    assert args.top_p is None


def test_explicit_args_are_accepted() -> None:
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    args = BatchSamplingArgs(temperatures=torch.tensor([1.0]))
    assert sampler.sample(torch.zeros(1, 4), args).shape == (1,)
