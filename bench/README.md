# 边界用例 bench

`bench/edge.py` 用真实引擎跑几个极端调度场景，把「引擎此时到底怎么表现」摊开记录，
而不是只看吞吐。每个 case 都**手动驱动** `LLMEngine.step()`（不走 `generate()`），
这样才能看清每一步调度器做了什么、有没有卡死、有没有泄漏 block。

## 命令

```bash
python -m bench.edge --case a                  # 短请求、比 KV cache 总容量大 1 个 token 的请求、短请求
python -m bench.edge --case b                  # 三个请求跑到第 4 步，abort_request 掉一个
python -m bench.edge --case c --max-tokens 1   # prompt 恰好等于 max_model_len=4096
python -m bench.edge --case c --max-tokens 2
python -m bench.edge --case all                # 三个 case 各起一个子进程（避免重复 init NCCL）
```

## 三个 case 在测什么

| case | 构造 | 关注点 |
| --- | --- | --- |
| a | 128 tok 短请求 → `capacity+1` tok 超容量请求 → 128 tok 短请求，KV cache 固定 8 块（2048 tok） | `NEVER` 拒绝（`FINISHED_IGNORED` / `prompt_length_exceeded`）；后面的短请求不再被队头阻塞；block 全部归还 |
| b | 3 条 300 tok 请求跑到第 4 步，然后 `abort_request` 掉一条 | abort 后下一步不再被调度；结果里有 `finish_reason="abort"`；block 全部归还 |
| c | probe 1：prompt = `max_model_len` = 4096；probe 2：prompt = 4095，生成 1 / 2 tok | 超长 prompt 入队即拒绝；`max_model_len - 1` 的 prompt 生成第 1 个 token 就撞上限被截断（`length`） |

## 输出怎么读

```
[OK]    符合预期
[EDGE]  能跑但行为可疑（引擎缺校验 / 缺清理，bench 只记录，不影响退出码）
[FAIL]  bench 自己的前置假设不成立或进程级异常，退出码 1
```

`--case all` 的退出码是所有子 case 的汇总。

## 常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--model` | `~/huggingface/Qwen3-0.6B/` | 本地模型目录 |
| `--max-model-len` | `4096` | case c 用 `max_model_len` 和 `max_model_len - 1` 两个 prompt 探边界 |
| `--max-tokens` | `1` | case c 生成多少 token |
| `--kv-blocks` | `8` | case a 强制指定的 KV 块数；`0` 表示用引擎默认显存预算（此时通常构造不出超容量请求） |
| `--short-len` / `--short-max-tokens` | `128` / `2` | case a 两条短请求 |
| `--b-prompt-len` / `--b-max-tokens` | `300` / `16` | case b 的三条请求 |
| `--steps-before-remove` | `4` | case b 第几步后 abort 一条 running 请求 |
| `--max-steps` | `4096` | 单个 case 的步数上限，防止死循环 |
| `--seed` | `0` | prompt token 的随机种子 |

> `case a` 的容量边界靠 `--kv-blocks` 固定块数才构造得出来。默认显存预算下
> Qwen3-0.6B 的 KV cache 有几百块（每块约 29MB），总容量远大于 `max_model_len`，
> 所以 bench 会临时接住 `ModelRunner.allocate_kv_cache`（只在 bench 进程内生效）。

## 和根目录 `bench.py` 的关系

根目录的 `bench.py` 是吞吐基准。因为同名，`bench/` 包会遮住 `bench.py` 的模块名，
所以：

- `python bench.py` 仍然照常跑吞吐基准；
- `python -m bench` 会被转发到 `bench.edge`（见 `bench/__main__.py`），
  不再等价于 `python -m bench.py`；
- 依赖 CUDA，纯 CPU 环境下直接退出码 2。

## 现在应该看到的结果（任务 A/B/C 之后）

```bash
python -m bench.edge --case a    # 4 OK / 0 EDGE / 0 FAIL
python -m bench.edge --case b    # 7 OK / 0 EDGE / 0 FAIL
python -m bench.edge --case c --max-tokens 1   # 5 OK / 0 EDGE / 0 FAIL
```

三个 case 都不再出现 `AssertionError`、空转、静默丢请求或 block 泄漏；
`--case` 输出里每一步会标成 `prefill` / `decode` / `empty`（终态出队后那一步是 `empty`）。
