"""固定 prompt 与负载定义的纯 Python 检查。"""

from __future__ import annotations

from tools.workloads import LOADS, PROMPTS, make_input_ids


def test_prompts_are_fixed_and_unique() -> None:
    assert len(PROMPTS) >= 20
    assert all(prompt.strip() for prompt in PROMPTS)
    assert len(set(PROMPTS)) == len(PROMPTS)


def test_loads_are_well_formed() -> None:
    assert {"short_32", "medium_512x8", "long_8k", "long_32k", "shared_prefix_4k"} <= set(LOADS)
    for load in LOADS.values():
        assert len(load.input_lens) == len(load.output_lens), load.name
        assert all(length > 0 for length in load.input_lens), load.name
        assert all(length > 0 for length in load.output_lens), load.name
        assert load.shared_prefix <= min(load.input_lens), load.name


def test_make_input_ids_lengths_and_shared_prefix() -> None:
    for load in LOADS.values():
        inputs = make_input_ids(load, seed=7)
        assert [len(ids) for ids in inputs] == list(load.input_lens), load.name

        if load.shared_prefix:
            prefixes = [tuple(ids[: load.shared_prefix]) for ids in inputs]
            assert len(set(prefixes)) == 1, load.name
            suffixes = [tuple(ids[load.shared_prefix :]) for ids in inputs]
            assert len(set(suffixes)) == len(suffixes), load.name

        for ids in inputs:
            assert all(1000 <= token < 30000 for token in ids), load.name


def test_make_input_ids_is_deterministic() -> None:
    load = LOADS["mixed_lengths"]
    assert make_input_ids(load, seed=11) == make_input_ids(load, seed=11)
    assert make_input_ids(load, seed=11) != make_input_ids(load, seed=12)
