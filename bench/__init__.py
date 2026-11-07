"""边界用例与场景延迟 bench（对调度器 / KV cache 管理的压力测试）。

入口：

* `python -m bench.edge --case a|b|c|all`：边界正确性；
* `python -m bench.run --scenario mixed_long_short --trace out/mixed.jsonl`：
  稳态 decode 被长 prompt 到达打断时的 TBT / TTFT。

说明见 bench/README.md。

注意：仓库根目录还有一个 `bench.py`（吞吐基准）。同名时 Python 会优先把 `bench/`
当成包，所以这里补了 `bench/__main__.py`，`python -m bench` 仍然可用。
"""
