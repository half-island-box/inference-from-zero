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

---

# 场景 bench：`mixed_long_short`

`bench/edge.py` 看的是边界正确性；`bench/run.py` 看的是**稳态延迟被新请求打断**：

```bash
python -m bench.run --scenario mixed_long_short --trace out/mixed.jsonl
```

构造：

- 先放 `--num-short`（默认 128）条短请求（默认 prompt 128 tok），手动 step 到它们
  全部进入 decode；
- 稳态之后，每 `--long-every-steps`（默认 20）个 engine step 注入一条
  `--long-prompt-len`（默认 3072）tok 的长 prompt，共 `--num-long`（默认 5）条；
  注入点是稳态 step 的 0-based 索引 20、40、60…，也就是先留 20 个纯 decode
  step 作为 baseline；
- 每条请求 `--short-max-tokens` / `--long-max-tokens` 默认生成 256 / 64 tok。

这个引擎是 prefill 优先调度：长 prompt 的 prefill 步会让 running 里的短请求
**整步让路**，所以它天然是短请求 TBT 的一个尖峰；长 prompt 的 TTFT 也基本等于
这一步 prefill 的时间。

## 输出指标

短请求 TBT（相邻 completion token 的落地时间差，ms）分三个桶：

| 桶 | 含义 |
| --- | --- |
| `baseline` | 第一条长请求到达之前、纯 decode 的 TBT，作为安静基线 |
| `quiet` | 稳态内所有没有跨过 prefill/empty 步的 TBT |
| `interrupted` | 跨过至少一个 prefill 步的 TBT（就是被长 prompt 打断的那些） |
| `all` | 稳态内全部 TBT（排除因 KV 不足被抢占、随后重新 prefill 的样本） |

重点看 `all` 的 `p50` / `p99` 和 `p99/p50`，再对比 `baseline` 的 `p99/p50`：
如果 p99 接近 `prefill step + decode step`，说明尖峰来自长 prompt 的 prefill。
如果 KV cache 不够，短请求会被抢占并在之后的 `prefill` 步重新计算；这类样本
（被抢占请求自己的 TBT，以及其他短请求跨过 re-prefill 步的 TBT）会被排除并打印
警告，避免把“重算”的时间算成新请求到达的影响。

长请求输出每条 TTFT（`add_request` → 第一个 token）以及 p50 / p99，并给出
`prefill_steps`（prefill 被切成几步；默认 prompt 能放进一个 batch，所以是 1）。

## trace（JSONL）

`--trace out/mixed.jsonl` 每行一个事件，所有时间都是相对本次运行开始的 `t_ms`：

| `type` | 关键字段 |
| --- | --- |
| `meta` | 参数、模型、`max_model_len`、KV block/容量 |
| `request` | `seq_id`、`role`、`prompt_tokens`、`max_tokens`、`submit_step` |
| `preempt` | `seq_id`、`step`、`phase`（KV 不足时才会出现） |
| `step` | `step`、`phase`、`kind`、`duration_ms`、`scheduled`、`emitted`、`arrivals`、`finished`、队列/空闲块 |
| `steady_start` | 进入稳态的下一步 |
| `finish` | `seq_id`、`completion_tokens`、`finish_reason` |
| `end` / `summary` | 结束原因和本次所有指标（和终端打印一致） |

`step.t_ms` 是该步结束、token 落地的时刻；`emitted` 是 `[[seq_id, completion_index], ...]`。
所以不依赖 bench 的统计代码，也能从 trace 重新算 TBT / TTFT。

## 常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--num-short` | `128` | 稳态 decode 的短请求条数 |
| `--short-prompt-len` | `128` | 短请求 prompt 长度 |
| `--short-max-tokens` | `256` | 要覆盖到最后一条长请求注入之后 |
| `--long-prompt-len` | `3072` | 长 prompt 长度 |
| `--long-max-tokens` | `64` | 长请求生成多少 token（只影响其自身 decode） |
| `--long-every-steps` | `20` | 每隔多少个 engine step 注入一条 |
| `--num-long` | `5` | 注入几条 |
| `--first-arrival-step` | `0`（= 间隔） | 第一条长请求在第几个稳态 step 注入（0-based） |
| `--kv-blocks` | `0` | 强制 KV 块数；调小可以观察抢占对 TBT 的污染 |
| `--max-num-batched-tokens` | `16384` | 小于长 prompt 时 prefill 会被切成多步 |

注意 `--short-max-tokens` 必须大于最后一条长请求的注入步，否则短请求会提前跑完，
bench 会以 FAIL 退出（长请求没注入全）。无 CUDA / 模型目录不存在时退出码 2；
bench 自身假设被打破（没到稳态、长请求拿不到 TTFT）时退出码 1。

---

# 场景 bench：`kv_pressure`

这个场景把 KV cache 压满，观察**资源不足时调度器到底在做什么**：

```bash
python -m bench.run --scenario kv_pressure \
  --trace out/kv.jsonl --gpu-memory-utilization 0.5
```

默认构造：

- `--num-requests 512` 条请求同时到达；
- 每条 prompt `--prompt-len 128` tok；
- 每条输出在 `--min-output-tokens 768` 到
  `--max-output-tokens 1024` tok 之间（固定随机种子，`ignore_eos=True`）；
- `--gpu-memory-utilization 0.5` 只是更快压满 KV；如果仍然放得下，终端会提示
  预计不会触发抢占，可继续调低该值或直接用 `--kv-blocks` 固定更小容量。

## 重点指标

| 指标 | 说明 |
| --- | --- |
| `KV 满步数(free_blocks=0)` | 末态空闲块为 0 的 step 数，并按 decode / prefill / empty 分解 |
| `抢占 events / requests` | 总 preemption 次数、有多少请求被踢过 |
| `每次释放 blocks` | 被踢请求当时持有的 KV block 数，可以验证队尾 victim 释放多少空间 |
| `白算 token` | `scheduled prefill tokens - 首次 prefill tokens`，即 re-prefill 重复计算量 |
| `白算 / 产出 token` | 白算 token 相对最终 completion token 的倍数 / 百分比 |
| `TTFT` | 全部、被踢、未被踢三组对比 |
| `TBT` | 全部、未被踢、跨过自身抢占的 token 间隔三组对比 |
| `被踢 -> 下一个 token` | 抢占发生到该请求下一个 token 落地的空档 |

预期判断方法：如果抢占发生在请求第一个 token 之后，原定义的 TTFT 不应受后续
re-prefill 影响；真正变差的是跨过抢占的 TBT / “被踢 -> 下一个 token”。如果某个
请求在被踢后又反复重算，`白算 / 产出 token` 会明显升高。

## trace 新增事件

除 `meta` / `request` / `preempt` / `step` / `finish` / `summary` 外，
`kv_pressure` 还会写：

| `type` | 关键字段 |
| --- | --- |
| `prefill_event` | `seq_id`、`tokens`、`start_cached_tokens`、`repeat`；用于统计白算 |
| `step` | 额外带 `used_blocks` / `kv_usage` |

`preempt.held_blocks` 是抢占前该序列实际占用的 block 数。
