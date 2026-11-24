# inference-from-zero

从零复现 [mini-sglang](https://github.com/sgl-project/mini-sglang) 推理引擎的学习工程。

规格来源是 [reproduction-roadmap.zh-CN.md](reproduction-roadmap.zh-CN.md)：13 个阶段（0–12），每个阶段都要求系统可运行、有验收标准、有性能数字、有回退开关。参考实现位于本机 `/Users/bj/Coding/github/mini-sglang`，只用于卡住后对照，不直接搬运。

## 当前进度

| 阶段 | 状态 | 产物 |
| --- | --- | --- |
| 0 环境与黄金基线 | 脚手架完成，待在 GPU 上生成基线 | `tools/` + `tests/` + `experiments/log.md` |
| 1 最小生成器 | 代码与 CPU 测试完成，待在 GPU 上验收 | `python/minisgl/` + `scripts/generate_naive.py` |
| 2 KV Cache / 批量前向 | 代码与 CPU 测试完成，待在 GPU 上验收 | `kvcache/` + `attention/` + `scheduler/` + `scripts/generate_paged.py` |
| 3 连续批处理 | 代码与 CPU 测试完成，待在 GPU 上验收 | `scheduler/` + `engine/engine.py` + `llm/` + `scripts/generate_continuous.py` |
| 4 分块预填充 | 代码与 CPU 测试完成，待在 GPU 上验收 | `scheduler/prefill.py` + `--max-prefill-length` + `--sched-policy` |

## 环境要求

- Linux + NVIDIA GPU（`sgl_kernel` / `flashinfer` / `tvm-ffi` 都依赖 CUDA），macOS 只能写代码和跑 CPU 测试；
- Python 3.10–3.12，推荐 `uv`；
- 首个模型：`Qwen/Qwen3-0.6B`（bf16）。

在 GPU 主机上：

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e ".[dev]" --torch-backend=cu128
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`--torch-backend=cu128` 让 uv 从 PyTorch 的 cu128 索引安装 torch。服务器驱动是 CUDA 12.8，
直接装 PyPI 默认 wheel（新版 CUDA 编译）会报 “The NVIDIA driver on your system is too old”。
想固化到 shell 里、之后不再带 flag：

```bash
echo 'export UV_TORCH_BACKEND=cu128' >> ~/.bashrc && source ~/.bashrc
```

已装坏的环境可以就地重装：

```bash
uv pip install --reinstall torch --torch-backend=cu128
```

模型默认从 ModelScope 下载（`uv pip install -e ".[dev]"` 已包含 `modelscope`）；如需改用 HuggingFace，加 `--model-source huggingface`。两者都默认下载到 `~/data`，可用环境变量 `MINISGL_MODEL_DIR` 覆盖。

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
# 任意阶段（0–12）收尾时，把本轮结果追加到 experiments/log.md 对应阶段小节
python tools/bench.py --model Qwen/Qwen3-0.6B --load all --append-log --log-stage 0
python tools/bench.py --model Qwen/Qwen3-0.6B --load all --append-log --log-stage 2
```

阶段 0 的 bench 是**串行 batch=1 基线**，输出 TTFT / ITL p50·p95·p99 / output tok/s / 峰值显存，结果写入 `out/bench/<load>_hf.json`。后续阶段实现引擎后，同一组负载直接对比。

固定负载定义在 `tools/workloads.py`：短 prompt、512×8 并发、8k/32k 长上下文、4k 共享前缀、32 请求突发、混合长度。

### 4. 记录

每次实验把日期、commit、配置、数字写入 [experiments/log.md](experiments/log.md)。性能数字必须绑定固定的模型、dtype、负载和 GPU。

`--append-log --log-stage N`（N 为 0–12 中任一阶段编号）会自动收集日期、`git` commit（工作区脏时标注 `dirty`）、GPU 型号/驱动、torch / transformers 版本与该轮所有负载的指标，插入日志中 `## 阶段 N` 小节的末尾；该小节不存在时在文末新建。`--log-note` 可填写「结论 / 下一步」，不填则留 `待填`。

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

# 序列长度增长时单步 decode 的耗时（并写入 log.md 阶段 1 表格）
python scripts/generate_naive.py --model Qwen/Qwen3-0.6B \
    --bench-lengths 128,256,512,1024,2048 --append-log --log-stage 1
```

`scripts/generate_naive.py` 输出的 `golden.json` 与 `logits.npz` 和阶段 0 基线格式一致，可直接用 `tools/compare.py` 回归。

`--append-log` 同样适用于生成模式（不传 `--bench-lengths`），此时记录 20 条 prompt 的 prefill/decode 均值和峰值显存；`--log-stage` 默认 1，`--log-note` 填写「结论 / 下一步」。

## 阶段 2：KV Cache、批量前向与 Paged Attention

阶段 2 先把 `layers/` / `models/` 从 `BaseOP` 迁移到 `torch.nn.Module`，再实现：

- `core.Req` / `Batch` / `Context` 状态机与全局 batch 上下文；
- `kvcache/`：`(2, layers, pages + 1, page_size, kv_heads, head_dim)` 的 KV 池与占位前缀缓存；
- `scheduler/`：请求槽位、页分配器（page_size 1/4/16）、batch 组装与静态调度循环；
- `attention/`：可插拔 backend 接口与参考实现（gather KV + SDPA，支持 extend prefill 的 causal 偏移）。

生成脚本与阶段 0/1 输出格式一致，可直接对比：

```bash
# CPU：迁移回归、页分配、attention 语义、批量生成一致性
pytest tests/test_module_migration.py tests/test_kvcache_alloc.py \
       tests/test_paged_engine.py tests/test_torch_attn.py -q

# GPU：paged 批量生成 + 黄金基线对比
python scripts/generate_paged.py --model Qwen/Qwen3-0.6B --model-source modelscope \
    --page-size 16 --warmup 1
python tools/compare.py --baseline out/golden/Qwen__Qwen3-0.6B/golden.json \
                        --actual   out/paged/Qwen__Qwen3-0.6B/golden.json
pytest tests/test_paged_generation.py -m "gpu and slow"

# 不同 batch size / 序列长度的 decode 延迟与显存，写入 log.md 阶段 2
python scripts/generate_paged.py --model Qwen/Qwen3-0.6B \
    --bench-batch-sizes 1,2,4,8 --warmup 1 --append-log --log-stage 2
python scripts/generate_paged.py --model Qwen/Qwen3-0.6B \
    --bench-seq-lens 128,256,512,1024 --warmup 1 --append-log --log-stage 2
```

`--page-size` 切换 1/4/16 时输出必须逐 token 一致；`--num-pages 0`（默认）会按负载与空闲显存自动估算页数。

`--warmup N`（默认 1）会在每个配置计时前用同形状请求预热 N 次，避免首个 batch 的 CUDA kernel 懒加载 / allocator 初始化开销（约 90–120 ms）进入 prefill 数字；`--warmup 0` 可复现未预热结果。

## 阶段 3：连续批处理与每请求采样

阶段 3 在阶段 2 的 paged KV 之上加入真正的连续批处理：

- `Scheduler` 每个 iteration 重新决定本轮跑哪些请求：prefill 有 pending 就优先组 prefill batch，否则跑所有 running 请求的 decode batch；
- `PrefillManager` 做 FCFS 准入，同时预留输入与未来输出，扣除 in-flight decode 的 `inflight_tokens`；
- `DecodeManager` 维护 running 集合，decode batch 按 `uid` 排序；
- `Engine` 负责模型、KV 池、页表与 `forward_batch`；`Sampler` 支持每请求 temperature / top_k / top_p，全 greedy 时走 argmax 快路径；
- `LLM` 提供离线 `generate()`，支持请求动态加入、中止与 1000 请求压测。

```bash
# CPU：采样器 + 连续批处理（50 混合请求、动态加入、中止、容量准入）
pytest tests/test_sampler.py tests/test_continuous_scheduler.py -q

# GPU：连续批处理生成 + 黄金基线对比
python scripts/generate_continuous.py --model Qwen/Qwen3-0.6B \
    --model-source modelscope --page-size 16 --warmup 1
python tools/compare.py --baseline out/golden/Qwen__Qwen3-0.6B/golden.json \
                        --actual   out/continuous/Qwen__Qwen3-0.6B/golden.json
pytest tests/test_continuous_generation.py -m "gpu and slow"

# 50 并发混合长度 + 串行基线，写入 log.md 阶段 3
python scripts/generate_continuous.py --model Qwen/Qwen3-0.6B \
    --page-size 16 --max-running-req 50 --bench-num-requests 200 \
    --bench-input-len 256 --bench-output-len 32 --bench-mixed \
    --bench-serial --warmup 1 --append-log --log-stage 3
```

`--bench-serial` 会额外逐条运行同一组请求，日志中自动给出相对串行加速比；`--bench-num-requests 1000` 用于连续压测与完整性检查。

## 阶段 4：分块预填充与调度策略

阶段 4 把长 prompt 拆成多个 chunk 分轮计算：

- `PrefillAdder` 按 `chunk = min(token_budget, remain_len)` 切块；没算完的请求生成 `ChunkedReq`，下一轮复用同一个 `table_idx` / `cached_len` / cache handle 继续；
- `--max-prefill-length`（别名 `--max-extend-tokens`）控制每次 iteration 的 prefill token 预算；
- `--sched-policy prefill_first|decode_first|interleave`：`interleave` 在 prefill 与 decode 同时可运行时交替调度，让 decode 插在两个 chunk 之间；
- chunked 请求不采样、不写回 token_pool、不进 DecodeManager，也不提前插入前缀缓存；中止时正常释放已分配的页与槽位。

```bash
# CPU：分块语义（预算 / 页边界 / 中止 / 交错 / 活锁）
pytest tests/test_chunked_prefill.py -q

# GPU：分块预填充生成 + 黄金基线对比
python scripts/generate_continuous.py --model Qwen/Qwen3-0.6B \
    --model-source modelscope --page-size 16 --max-prefill-length 2048 --warmup 1
python tools/compare.py --baseline out/golden/Qwen__Qwen3-0.6B/golden.json \
                        --actual   out/continuous/Qwen__Qwen3-0.6B/golden.json

# chunk size 消融，写入 log.md 阶段 4
for chunk in 512 2048 8192; do
  python scripts/generate_continuous.py --model Qwen/Qwen3-0.6B \
      --page-size 16 --max-running-req 50 --max-prefill-length $chunk \
      --sched-policy interleave --bench-num-requests 16 \
      --bench-input-len 4096 --bench-output-len 32 --bench-mixed \
      --warmup 1 --append-log --log-stage 4
done
```

`--bench-long-input-len 32768 --bench-num-requests 9 --bench-input-len 64` 用来构造「1 条 32k prompt + 8 条短请求并发」，验证分块后不 OOM 且 decode 能插入 chunk 之间。

## 采样语义（后续所有阶段复用）

- 每条请求最多生成 `max_tokens` 个 token；
- 生成的 token 等于 `eos_token_id` 时**不输出该 token**并停止（`stop_reason="eos"`）；
- `ignore_eos=True` 时忽略 EOS，生成满 `max_tokens`（benchmark 使用此模式保证长度固定）；
- prompt 统一使用 raw token ids，`add_special_tokens=False`，阶段 6 之前不套 chat template。

## 目录结构

```
python/minisgl/          # 自己的引擎实现
  core.py                # SamplingParams、Req / Batch / Context
  layers/                # nn.Module：RMSNorm、Linear、RoPE、Attention、Embedding
  models/                # 配置、Llama/Qwen3、safetensors 权重加载
  kvcache/               # KV 池（MHA）、前缀缓存接口、naive 前缀缓存
  attention/             # attention backend 接口与 torch 参考实现
  scheduler/             # 连续批处理：prefill/decode 管理、页分配、主循环
  engine/                # Engine（forward_batch）、采样器、naive 生成循环
  llm/                   # 离线 LLM.generate() 接口
  env.py                 # MINISGL_* 环境变量开关
scripts/
  generate_naive.py      # 阶段 1 生成/对比/序列长度 benchmark
  generate_paged.py      # 阶段 2 paged 批量生成/对比/batch·长度 benchmark
  generate_continuous.py # 阶段 3 连续批处理生成/并发·串行压测
tools/
  reference_hf.py        # 黄金基线：greedy 生成 + logits
  compare.py             # 与基线对比（token 精确 / logits 容差）
  bench.py               # TTFT / ITL / 吞吐 / 峰值显存
  workloads.py           # 20 条固定 prompt + 7 组固定负载
  hf_runtime.py          # 共用的 HF 加载与 greedy 解码
  memory.py              # CUDA 显存统计（显式绑定 --device，避免读到 0 号卡）
tests/
  test_reference.py      # GPU 端到端回归（gpu + slow）
  test_hf_runtime.py     # CPU 小模型解码语义测试
  test_compare.py        # 对比工具单测
  test_workloads.py      # 负载定义单测
  test_model.py          # 层/模型前向对齐 HF、权重映射（CPU）
  test_naive_generate.py # naive 生成语义（CPU）
  test_naive_generation.py # 阶段 1 GPU 验收
  test_module_migration.py # nn.Module 迁移与严格加载（CPU）
  test_kvcache_alloc.py  # 页分配、跨页与完整性（CPU）
  test_torch_attn.py     # extend prefill 的 causal 偏移（CPU）
  test_paged_engine.py   # 批量生成与阶段 1 一致（CPU）
  test_paged_generation.py # 阶段 2 GPU 验收
  test_generate_paged_log.py # 阶段 2 日志格式（CPU）
  test_sampler.py        # 每请求采样参数（CPU）
  test_continuous_scheduler.py # 连续批处理语义（CPU）
  test_continuous_generation.py # 阶段 3 GPU 验收
  test_chunked_prefill.py # 分块预填充语义（CPU）
  test_model_dir.py      # 模型下载目录解析（CPU）
  test_memory.py         # 显存统计绑定设备（CPU）
experiments/log.md       # 实验记录
```

## 测试

```bash
pytest                       # CPU 测试可直接跑；GPU 测试在无卡机器上自动跳过
pytest -m "gpu and slow"     # 在 GPU 主机上跑端到端回归
pytest tests/test_compare.py::test_token_mismatch_fails
```

CPU 测试不下载权重，使用随机初始化的小 Llama / Qwen3 验证缓存、页分配与停止语义。

## 与参考实现的关系

- 参考仓库：`/Users/bj/Coding/github/mini-sglang`（upstream `sgl-project/mini-sglang`）。
- 阅读顺序：`core` → `layers` → `models` → `kvcache` → `attention` → `scheduler` → `engine` → `message/tokenizer/server` → `distributed` → `kernel`。
- 每个阶段先自己写，卡住后只对照接口和关键函数；每个优化都要能单独关闭并量化。
