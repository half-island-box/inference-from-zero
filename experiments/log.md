# 实验日志

每次实验追加一条记录。所有性能数字必须绑定：日期、commit、GPU 型号、模型、dtype、固定负载、配置开关。

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

| prompt 长度 | prefill 耗时 (ms) | 单步 decode（长度 L+1）(ms) |
| --- | --- | --- |
| 128 | | |
| 256 | | |
| 512 | | |
| 1024 | | |
| 2048 | | |

## 阶段 2：KV Cache 与批量前向

（待开始）

## 阶段 3：连续批处理调度器

（待开始）

## 阶段 4：分块预填充

（待开始）

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
