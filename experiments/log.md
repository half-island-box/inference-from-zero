# 实验日志

每次实验追加一条记录。所有性能数字必须绑定：日期、commit、GPU 型号、模型、dtype、固定负载、配置开关。

`python tools/bench.py ... --append-log --log-stage N`（N 为 0–12 任一阶段编号）会自动生成本模板中的大部分字段并插入对应阶段小节；阶段 1 的 `scripts/generate_naive.py` 也支持同样的 `--append-log`。「结论 / 下一步」用 `--log-note` 填写，也可事后手改。

## 记录模板

```markdown
### YYYY-MM-DD — <实验名> （阶段 N）

- commit:
- GPU / 驱动 / torch / transformers:
- 模型 / dtype:
- 配置:
- 负载:
- 结果: TTFT p50/p95 = … ms；ITL p50/p95/p99 = … ms；吞吐 = … tok/s；峰值显存 = … GB
- 结论 / 下一步:
```

## 阶段 0：环境与黄金基线

### 待填 — 黄金基线生成

- commit:
- GPU / 驱动 / torch / transformers:
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bf16
- 配置: `--max-tokens 32`，20 条固定 prompt
- 可复现性: 连续两次 `--check-determinism` 结果
- 结果: prefill 耗时 / 单步 decode 耗时 / 峰值显存
- 结论:

### 阶段 0 基线（batch=1，串行）

| 负载 | 请求数 | 输入长度 | 输出长度 | TTFT p50 (ms) | ITL p50 (ms) | ITL p99 (ms) | tok/s | 峰值显存 (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_32 | 1 | 32 | 32 | | | | | |
| medium_512x8 | 8 | 512 | 64 | | | | | |
| long_8k | 1 | 8192 | 32 | | | | | |
| long_32k | 1 | 32768 | 16 | | | | | |
| shared_prefix_4k | 2 | 4352 | 32 | | | | | |
| burst_32 | 32 | 128 | 32 | | | | | |
| mixed_lengths | 8 | 32–4096 | 16–128 | | | | | |

### 2026-11-20 — bench: short_32, medium_512x8, long_8k, long_32k, shared_prefix_4k, burst_32, mixed_lengths （阶段 0）

- commit: 7ed6b5d
- GPU / 驱动 / torch / transformers: NVIDIA A10 / driver 550.54.15 / torch 2.3.1+cu121 / transformers 4.55.4
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bfloat16（modelscope）
- 配置: backend=hf, seed=0, warmup=1
- 负载: short_32, medium_512x8, long_8k, long_32k, shared_prefix_4k, burst_32, mixed_lengths

| 负载 | 请求数 | 输入长度 | 输出长度 | TTFT p50 (ms) | ITL p50 (ms) | ITL p95 (ms) | ITL p99 (ms) | tok/s | 峰值显存 (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_32 | 1 | 32 | 32 | 52.94 | 23.50 | 25.31 | 25.70 | 41.93 | 1.21 |
| medium_512x8 | 8 | 512 | 64 | 49.36 | 23.71 | 25.20 | 29.70 | 41.63 | 1.29 |
| long_8k | 1 | 8192 | 32 | 427.72 | 23.69 | 24.15 | 24.45 | 28.04 | 2.40 |
| long_32k | 1 | 32768 | 16 | 3588.87 | 75.51 | 76.20 | 76.48 | 3.44 | 5.98 |
| shared_prefix_4k | 2 | 4352 | 32 | 199.48 | 23.57 | 25.22 | 27.88 | 34.94 | 1.84 |
| burst_32 | 32 | 128 | 32 | 49.15 | 23.77 | 26.67 | 31.64 | 41.21 | 1.22 |
| mixed_lengths | 8 | 32–4096 | 16–128 | 50.13 | 24.23 | 27.22 | 31.40 | 39.92 | 1.80 |

- 结论 / 下一步: 阶段 0 黄金基线

### 2026-11-24 — bench: short_32, medium_512x8, long_8k, long_32k, shared_prefix_4k, burst_32, mixed_lengths （阶段 0）

- commit: 6ee3af2 (dirty)
- GPU / 驱动 / torch / transformers: cuda:5 / driver 570.153.02 / torch 2.11.0+cu128 / transformers 4.57.3
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bfloat16（modelscope）
- 配置: backend=hf, seed=0, warmup=1
- 负载: short_32, medium_512x8, long_8k, long_32k, shared_prefix_4k, burst_32, mixed_lengths

| 负载 | 请求数 | 输入长度 | 输出长度 | TTFT p50 (ms) | ITL p50 (ms) | ITL p95 (ms) | ITL p99 (ms) | tok/s | 峰值显存 (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_32 | 1 | 32 | 32 | 68.62 | 34.96 | 37.17 | 38.25 | 28.89 | 0.00 |
| medium_512x8 | 8 | 512 | 64 | 77.25 | 33.59 | 37.06 | 42.70 | 29.14 | 0.00 |
| long_8k | 1 | 8192 | 32 | 594.61 | 33.17 | 37.10 | 37.19 | 19.78 | 0.00 |
| long_32k | 1 | 32768 | 16 | 5213.70 | 39.03 | 49.41 | 49.98 | 2.76 | 0.00 |
| shared_prefix_4k | 2 | 4352 | 32 | 286.00 | 34.44 | 36.69 | 37.74 | 24.21 | 0.00 |
| burst_32 | 32 | 128 | 32 | 70.48 | 33.66 | 37.28 | 42.59 | 29.06 | 0.00 |
| mixed_lengths | 8 | 32–4096 | 16–128 | 79.34 | 33.85 | 37.79 | 45.67 | 28.30 | 0.00 |

- 结论 / 下一步: 阶段 0 黄金基线

## 阶段 1：最小生成器

### 2026-11-19 — 代码与 CPU 测试完成（待 GPU 验收）

- commit: 阶段 1 提交（见 `git log`）
- GPU / 驱动 / torch / transformers: 待填
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bf16（CPU 单测使用 tiny Llama/Qwen3，fp32）
- 配置: 纯 PyTorch，`F.scaled_dot_product_attention` + causal mask，无 KV Cache；q/k/v 与 gate/up 仅在加载时融合
- 已完成:
  - `core.py` / `layers/` / `models/` / `engine/`
  - `scripts/generate_naive.py`：输出可与 `tools/compare.py` 直接对比的 JSON/NPZ
- CPU 结果: `pytest tests/test_model.py tests/test_naive_generate.py -q` 全部通过；tiny Llama/Qwen3 前向与 HF fp32 对齐；greedy 输出逐 token 一致
- 待办（GPU）:
  1. `python scripts/generate_naive.py --model Qwen/Qwen3-0.6B --model-source modelscope`
  2. `python tools/compare.py ...`（20 条 prompt 逐 token + logits 容差）
  3. `pytest tests/test_naive_generation.py -m "gpu and slow"`
  4. 跑 `--bench-lengths` 填下表
- 结论: 待 GPU 数字确认后进入阶段 2



### 2026-11-20 — naive 生成（无 KV Cache） （阶段 1）

- commit: 7ed6b5d (dirty)
- GPU / 驱动 / torch / transformers: NVIDIA A10 / driver 550.54.15 / torch 2.3.1+cu121 / transformers 4.55.4
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bfloat16（modelscope）
- 配置: max_tokens=32, ignore_eos=False, seed=0, 20 条固定 prompt（raw token ids）
- 结果: prefill 均值 = 27.51 ms；单步 decode 均值 = 21.66 ms；峰值显存 = 1.25 GB

- 结论 / 下一步: 阶段 1 生成验收

### 2026-11-20 — naive decode 长度扩展 （阶段 1）

- commit: 7ed6b5d (dirty)
- GPU / 驱动 / torch / transformers: NVIDIA A10 / driver 550.54.15 / torch 2.3.1+cu121 / transformers 4.55.4
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bfloat16（modelscope）
- 配置: 无 KV Cache，每步重算全序列；bench_repeats=3
- 负载: prompt 长度 128, 256, 512, 1024, 2048

| prompt 长度 | prefill 耗时 (ms) | 单步 decode（长度 L+1）(ms) |
| --- | --- | --- |
| 128 | 139.41 | 21.64 |
| 256 | 21.96 | 21.11 |
| 512 | 22.17 | 21.57 |
| 1024 | 35.34 | 37.08 |
| 2048 | 74.88 | 76.47 |

- 结论 / 下一步: 待填



## 阶段 2：KV Cache 与批量前向

### 2026-11-24 — paged decode 随 batch size 扩展 （阶段 2）

- commit: 6ee3af2 (dirty)
- GPU / 驱动 / torch / transformers: NVIDIA GeForce RTX 3090 / driver 570.153.02 / torch 2.11.0+cu128 / transformers 4.57.3
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bfloat16（modelscope）
- KV cache: page_size=16, max_running_req=8
- warmup: 1 次/配置（不计时）
- 配置: 固定 prompt 长度 256，max_tokens=32；greedy，ignore_eos=True

| batch size | prefill (ms) | decode 均值 (ms) | 峰值显存 (GB) |
| --- | --- | --- | --- |
| 1 | 52.02 | 43.77 | 0.00 |
| 2 | 53.26 | 47.62 | 0.00 |
| 4 | 60.86 | 58.53 | 0.00 |
| 8 | 94.77 | 78.85 | 0.00 |

- 结论 / 下一步: 阶段 2 batch 扩展

### 2026-11-24 — paged decode 随序列长度扩展 （阶段 2）

- commit: 6ee3af2 (dirty)
- GPU / 驱动 / torch / transformers: NVIDIA GeForce RTX 3090 / driver 570.153.02 / torch 2.11.0+cu128 / transformers 4.57.3
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bfloat16（modelscope）
- KV cache: page_size=16, max_running_req=8
- warmup: 1 次/配置（不计时）
- 配置: batch size=1，max_tokens=32；greedy，ignore_eos=True

| prompt 长度 | prefill (ms) | decode 均值 (ms) | 峰值显存 (GB) |
| --- | --- | --- | --- |
| 128 | 42.17 | 45.19 | 0.00 |
| 256 | 43.20 | 41.27 | 0.00 |
| 512 | 69.03 | 40.97 | 0.00 |
| 1024 | 77.34 | 41.24 | 0.00 |

- 结论 / 下一步: 阶段 2 序列长度扩展

### 2026-11-24 — paged KV Cache 批量生成 （阶段 2）

- commit: 6ee3af2 (dirty)
- GPU / 驱动 / torch / transformers: NVIDIA GeForce RTX 3090 / driver 570.153.02 / torch 2.11.0+cu128 / transformers 4.57.3
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bfloat16（modelscope）
- KV cache: page_size=16, max_running_req=8
- warmup: 1 次/配置（不计时）
- 配置: max_tokens=32, ignore_eos=False, batch≤8, 20 条固定 prompt（3 个 batch）
- 结果: prefill 均值 = 69.07 ms；单步 decode 均值 = 68.37 ms；峰值显存 = 0.00 GB

- 结论 / 下一步: 阶段 2 批量生成验收

## 阶段 3：连续批处理调度器

### 2026-11-24 — 连续批处理并发压测 （阶段 3）

- commit: 6ee3af2 (dirty)
- GPU / 驱动 / torch / transformers: NVIDIA GeForce RTX 3090 / driver 570.153.02 / torch 2.11.0+cu128 / transformers 4.57.3
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bfloat16（modelscope）
- KV cache / 调度: page_size=16, num_pages=1526, max_running_req=50, max_extend_tokens=8192, cache=naive
- warmup: 1 次（不计时）
- 配置: 200 条请求，输入长度 1/4..256，输出 32；greedy，ignore_eos=True

| 模式 | 请求数 | wall (s) | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p99 (ms) | tok/s | 峰值显存 (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 并发 | 200 | 38.41 | 15071.76 | 29185.40 | 291.05 | 378.64 | 166.64 | 0.00 |
| 串行 | 200 | 269.98 | 41.54 | 49.04 | 40.21 | 54.08 | 23.71 | 0.00 |
- 相对串行加速: 7.03×

- 结论 / 下一步: 阶段 3 连续批处理并发 vs 串行

### 2026-11-24 — 连续批处理并发压测 （阶段 3）

- commit: 6ee3af2 (dirty)
- GPU / 驱动 / torch / transformers: NVIDIA GeForce RTX 3090 / driver 570.153.02 / torch 2.11.0+cu128 / transformers 4.57.3
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bfloat16（modelscope）
- KV cache / 调度: page_size=16, num_pages=1526, max_running_req=50, max_extend_tokens=8192, cache=naive
- warmup: 1 次（不计时）
- 配置: 1000 条请求，输入长度 1/4..128，输出 16；greedy，ignore_eos=True

| 模式 | 请求数 | wall (s) | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p99 (ms) | tok/s | 峰值显存 (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 并发 | 1000 | 96.91 | 46539.50 | 87936.83 | 282.36 | 384.82 | 165.09 | 0.00 |

- 结论 / 下一步: 阶段 3 1000 请求压测

### 2026-11-24 — 连续批处理并发压测 （阶段 3）

- commit: 6ee3af2 (dirty)
- GPU / 驱动 / torch / transformers: NVIDIA GeForce RTX 3090 / driver 570.153.02 / torch 2.11.0+cu128 / transformers 4.57.3
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bfloat16（modelscope）
- KV cache / 调度: page_size=16, num_pages=1526, max_running_req=50, max_extend_tokens=8192, cache=naive
- warmup: 1 次（不计时）
- 配置: 1000 条请求，输入长度 1/4..128，输出 16；greedy，ignore_eos=True

| 模式 | 请求数 | wall (s) | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p99 (ms) | tok/s | 峰值显存 (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 并发 | 1000 | 91.39 | 43787.99 | 82715.00 | 281.72 | 332.76 | 175.08 | 0.00 |

- 结论 / 下一步: 阶段 3 1000 请求压测

### 2026-11-24 — 连续批处理并发压测 （阶段 3）

- commit: 6ee3af2 (dirty)
- GPU / 驱动 / torch / transformers: NVIDIA GeForce RTX 3090 / driver 570.153.02 / torch 2.11.0+cu128 / transformers 4.57.3
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bfloat16（modelscope）
- KV cache / 调度: page_size=16, num_pages=1526, max_running_req=50, max_extend_tokens=8192, cache=naive
- warmup: 1 次（不计时）
- 配置: 1000 条请求，输入长度 1/4..128，输出 16；greedy，ignore_eos=True

| 模式 | 请求数 | wall (s) | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p99 (ms) | tok/s | 峰值显存 (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 并发 | 1000 | 99.95 | 47702.35 | 90576.74 | 308.72 | 351.57 | 160.09 | 0.00 |

- 结论 / 下一步: 阶段 3 1000 请求压测

## 阶段 4：分块预填充

### 2026-11-24 — 连续批处理并发压测 （阶段 4）

- commit: a12d9a6 (dirty)
- GPU / 驱动 / torch / transformers: NVIDIA GeForce RTX 3090 (cuda:4) / driver 570.153.02 / torch 2.11.0+cu128 / transformers 4.57.3
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bfloat16（modelscope）
- KV cache / 调度: page_size=16, num_pages=1526, max_running_req=50, max_extend_tokens=512, sched_policy=interleave, cache=naive
- warmup: 1 次（不计时）
- 配置: 16 条请求，输入长度 1/4..4096，输出 32；greedy，ignore_eos=True

| 模式 | 请求数 | wall (s) | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p99 (ms) | tok/s | 峰值显存 (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 并发 | 16 | 10.51 | 4397.99 | 8245.84 | 111.83 | 147.14 | 48.70 | 4.21 |

- 结论 / 下一步: 阶段 4 chunk=512 长 prompt + 并发 decode

### 2026-11-24 — 连续批处理并发压测 （阶段 4）

- commit: a12d9a6 (dirty)
- GPU / 驱动 / torch / transformers: NVIDIA GeForce RTX 3090 (cuda:4) / driver 570.153.02 / torch 2.11.0+cu128 / transformers 4.57.3
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bfloat16（modelscope）
- KV cache / 调度: page_size=16, num_pages=1526, max_running_req=50, max_extend_tokens=2048, sched_policy=interleave, cache=naive
- warmup: 1 次（不计时）
- 配置: 16 条请求，输入长度 1/4..4096，输出 32；greedy，ignore_eos=True

| 模式 | 请求数 | wall (s) | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p99 (ms) | tok/s | 峰值显存 (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 并发 | 16 | 8.10 | 2086.32 | 5826.04 | 82.10 | 252.49 | 63.18 | 4.68 |

- 结论 / 下一步: 阶段 4 chunk=2048 长 prompt + 并发 decode

### 2026-11-24 — 连续批处理并发压测 （阶段 4）

- commit: a12d9a6 (dirty)
- GPU / 驱动 / torch / transformers: NVIDIA GeForce RTX 3090 (cuda:4) / driver 570.153.02 / torch 2.11.0+cu128 / transformers 4.57.3
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bfloat16（modelscope）
- KV cache / 调度: page_size=16, num_pages=1526, max_running_req=50, max_extend_tokens=8192, sched_policy=interleave, cache=naive
- warmup: 1 次（不计时）
- 配置: 16 条请求，输入长度 1/4..4096，输出 32；greedy，ignore_eos=True

| 模式 | 请求数 | wall (s) | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p99 (ms) | tok/s | 峰值显存 (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 并发 | 16 | 7.48 | 1218.54 | 4732.11 | 87.76 | 667.87 | 68.49 | 6.56 |

- 结论 / 下一步: 阶段 4 chunk=8192 长 prompt + 并发 decode

### 2026-11-24 — 连续批处理并发压测 （阶段 4）

- commit: fd96108 (dirty)
- GPU / 驱动 / torch / transformers: NVIDIA GeForce RTX 3090 (cuda:4) / driver 570.153.02 / torch 2.11.0+cu128 / transformers 4.57.3
- 模型 / dtype: `Qwen/Qwen3-0.6B` / bfloat16（modelscope）
- KV cache / 调度: page_size=16, num_pages=7376, max_running_req=9, max_extend_tokens=2048, sched_policy=interleave, cache=naive
- warmup: 1 次（不计时）
- 配置: 9 条请求，输入长度 首条 32768 + 64，输出 8；greedy，ignore_eos=True

| 模式 | 请求数 | wall (s) | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p99 (ms) | tok/s | 峰值显存 (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 并发 | 9 | 8.48 | 7744.26 | 7744.33 | 97.67 | 348.75 | 8.49 | 15.41 |

- 结论 / 下一步: 阶段 4 32k prompt + 8 并发 decode

## 阶段 5：前缀缓存

（待开始）

## 阶段 6：服务化

（待开始）

## 阶段 7：高性能注意力后端

（待开始）

## 阶段 8：CUDA Graph

（待开始）

## 阶段 9：重叠调度

（待开始）

## 阶段 10：张量并行

（待开始）

## 阶段 11：MoE 与自定义 Kernel

（待开始）

## 阶段 12：剖析与验收

（待开始）
