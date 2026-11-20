# inference-from-zero

从零复现 [mini-sglang](https://github.com/sgl-project/mini-sglang) 推理引擎的学习工程。

规格来源是 [reproduction-roadmap.zh-CN.md](reproduction-roadmap.zh-CN.md)：13 个阶段（0–12），每个阶段都要求系统可运行、有验收标准、有性能数字、有回退开关。参考实现位于本机 `/Users/bj/Coding/github/mini-sglang`，只用于卡住后对照，不直接搬运。

## 当前进度

| 阶段 | 状态 | 产物 |
| --- | --- | --- |
| 0 环境与黄金基线 | 脚手架完成，待在 GPU 上生成基线 | `tools/` + `tests/` + `experiments/log.md` |
| 1 最小生成器 | 代码与 CPU 测试完成，待在 GPU 上验收 | `python/minisgl/` + `scripts/generate_naive.py` |

## 环境要求

- Linux + NVIDIA GPU（`sgl_kernel` / `flashinfer` / `tvm-ffi` 都依赖 CUDA），macOS 只能写代码和跑 CPU 测试；
- Python 3.10–3.12，推荐 `uv`；
- 首个模型：`Qwen/Qwen3-0.6B`（bf16）。

在 GPU 主机上：

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

模型默认从 ModelScope 下载（`uv pip install -e ".[dev]"` 已包含 `modelscope`）；如需改用 HuggingFace，加 `--model-source huggingface`。

## 阶段 0：黄金基线

### 1. 生成基线

```bash
python tools/reference_hf.py --model Qwen/Qwen3-0.6B
```

默认写到 `out/golden/Qwen__Qwen3-0.6B/`：

- `golden.json`：20 条固定 prompt 的 raw `input_ids`、greedy `output_ids`、停止原因、模型/驱动版本；
- `logits.npz`：每条 prompt 最后一个位置的 logits（float32）。

验证可复现性（同一条命令跑两次必须逐 token 一致）：

```bash
python tools/reference_hf.py --model Qwen/Qwen3-0.6B --check-determinism
```

### 2. 与基线对比

```bash
python tools/compare.py \
    --baseline out/golden/Qwen__Qwen3-0.6B/golden.json \
    --actual   out/run/golden.json
```

- token 级：greedy 输出必须**逐 token 完全一致**；
- logits 级：bf16 默认容差 `1e-2`，fp32 调试时用 `--tolerance 1e-3`；
- 退出码 0 = PASS，1 = FAIL。

### 3. 基线性能数字

```bash
python tools/bench.py --model Qwen/Qwen3-0.6B --load short_32
python tools/bench.py --model Qwen/Qwen3-0.6B --load all
python tools/bench.py --list
```

阶段 0 的 bench 是**串行 batch=1 基线**，输出 TTFT / ITL p50·p95·p99 / output tok/s / 峰值显存，结果写入 `out/bench/<load>_hf.json`。后续阶段实现引擎后，同一组负载直接对比。

固定负载定义在 `tools/workloads.py`：短 prompt、512×8 并发、8k/32k 长上下文、4k 共享前缀、32 请求突发、混合长度。

### 4. 记录

每次实验把日期、commit、配置、数字写入 [experiments/log.md](experiments/log.md)。性能数字必须绑定固定的模型、dtype、负载和 GPU。

## 阶段 1：最小生成器

纯 PyTorch 实现模型前向，**没有 KV Cache**：prefill 后每生成一个 token 都重算整段序列，因此 decode 会随长度明显变慢（阶段 2 的动机）。

```bash
# CPU 单测：层/权重映射/前向对齐 HF，以及生成语义
pytest tests/test_model.py tests/test_naive_generate.py -q

# GPU：生成阶段 1 输出并与黄金基线逐 token / logits 对比
python scripts/generate_naive.py --model Qwen/Qwen3-0.6B --model-source modelscope
python tools/compare.py --baseline out/golden/Qwen__Qwen3-0.6B/golden.json \
                        --actual   out/naive/Qwen__Qwen3-0.6B/golden.json

# GPU：端到端验收
pytest tests/test_naive_generation.py -m "gpu and slow"

# 序列长度增长时单步 decode 的耗时
python scripts/generate_naive.py --model Qwen/Qwen3-0.6B \
    --bench-lengths 128,256,512,1024,2048
```

`scripts/generate_naive.py` 输出的 `golden.json` 与 `logits.npz` 和阶段 0 基线格式一致，可直接用 `tools/compare.py` 回归。

## 采样语义（后续所有阶段复用）

- 每条请求最多生成 `max_tokens` 个 token；
- 生成的 token 等于 `eos_token_id` 时**不输出该 token**并停止（`stop_reason="eos"`）；
- `ignore_eos=True` 时忽略 EOS，生成满 `max_tokens`（benchmark 使用此模式保证长度固定）；
- prompt 统一使用 raw token ids，`add_special_tokens=False`，阶段 6 之前不套 chat template。

## 目录结构

```
python/minisgl/          # 自己的引擎实现
  core.py                # SamplingParams
  layers/                # BaseOP、RMSNorm、Linear、RoPE、Attention、Embedding
  models/                # 配置、Llama/Qwen3、safetensors 权重加载
  engine/                # greedy 采样与 naive 生成循环
scripts/
  generate_naive.py      # 阶段 1 生成/对比/序列长度 benchmark
tools/
  reference_hf.py        # 黄金基线：greedy 生成 + logits
  compare.py             # 与基线对比（token 精确 / logits 容差）
  bench.py               # TTFT / ITL / 吞吐 / 峰值显存
  workloads.py           # 20 条固定 prompt + 7 组固定负载
  hf_runtime.py          # 共用的 HF 加载与 greedy 解码
tests/
  test_reference.py      # GPU 端到端回归（gpu + slow）
  test_hf_runtime.py     # CPU 小模型解码语义测试
  test_compare.py        # 对比工具单测
  test_workloads.py      # 负载定义单测
  test_model.py          # 层/模型前向对齐 HF、权重映射（CPU）
  test_naive_generate.py # naive 生成语义（CPU）
  test_naive_generation.py # 阶段 1 GPU 验收
experiments/log.md       # 实验记录
```

## 测试

```bash
pytest                       # CPU 测试可直接跑；GPU 测试在无卡机器上自动跳过
pytest -m "gpu and slow"     # 在 GPU 主机上跑端到端回归
pytest tests/test_compare.py::test_token_mismatch_fails
```

CPU 测试不下载权重，使用随机初始化的小 Llama 验证缓存与停止语义。

## 与参考实现的关系

- 参考仓库：`/Users/bj/Coding/github/mini-sglang`（upstream `sgl-project/mini-sglang`）。
- 阅读顺序：`core` → `layers` → `models` → `kvcache` → `attention` → `scheduler` → `engine` → `message/tokenizer/server` → `distributed` → `kernel`。
- 每个阶段先自己写，卡住后只对照接口和关键函数；每个优化都要能单独关闭并量化。
