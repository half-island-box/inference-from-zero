# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 开发规范
1. 每个阶段的代码完成后，将该阶段对应的架构图补充在reproduction-roadmap.zh-CN.md对应的位置里，并补充对应的测试命令
2. 不要写太多繁杂注释，注释统一用中文，只在关键作用的代码进行注释
3. 用户没有明确声明，不要自作主张进行改动和开发
4. 涉及到hugging face的地方，统一替换成等效的modelscope代码

## 项目定位

本仓库是**从零复现 mini-sglang 推理引擎**的工程，按 [reproduction-roadmap.zh-CN.md](reproduction-roadmap.zh-CN.md) 分 13 个阶段（阶段 0–12）推进。路线图是唯一的规格来源：每个阶段的目标、要实现的模块、验收标准、常见坑都在里面，动手前先读对应阶段。

- **参考实现（只读）**：`/Users/bj/Coding/github/mini-sglang`（upstream `sgl-project/mini-sglang`，当前 `9a91cfa`）。其中 `docs/features.zh-CN.md`、`docs/structures.zh-CN.md`、`docs/reproduction-roadmap.zh-CN.md` 是本地未提交的中文文档。
- **当前状态**：阶段 0 脚手架已完成（`tools/`、`tests/`、`experiments/log.md`）；黄金基线需在 GPU 主机上运行 `tools/reference_hf.py` 生成；阶段 1 尚未开始。
- 代码应写在**本仓库**；参考仓库只用于卡住后对照接口和关键函数，不要整段搬运。先自己写，再 diff。

## 运行环境

- 运行时**必须** Linux + NVIDIA GPU：`sgl_kernel`、`flashinfer`、`tvm-ffi` JIT 都依赖 CUDA，macOS/Windows 原生不行。参考仓库根目录有 `Dockerfile`。
- 本工作区是 macOS：可以写代码、跑纯 CPU 逻辑和 lint，但任何 `import torch` + CUDA kernel 的验证都要在 GPU 机器/容器上执行。
- Python 3.10–3.12，包管理用 `uv`。
- 首个目标模型：`Qwen/Qwen3-0.6B`（bf16，GQA + QK Norm）。HF 下载受限时加 `--model-source modelscope`。
- 阶段 6 之前 prompt 一律用 **raw token ids**，不套 chat template；chat template 留到服务化阶段单独验证。

## 目录结构（路线图 §0.3）

```
python/minisgl/          # 自己的实现，模块划分与参考实现一致
tools/reference_hf.py    # 黄金基线：HF greedy 生成 + logits
tools/compare.py         # 与黄金输出对比（token 精确 / logits 容差）
tools/bench.py           # TTFT / ITL / 吞吐 / 峰值显存
tools/workloads.py       # 20 条固定 prompt + 固定测试负载
tools/hf_runtime.py      # 共用的 HF 加载与 greedy 解码（含 EOS 语义）
tests/test_reference.py  # 端到端正确性（gpu + slow）
tests/test_hf_runtime.py # 小模型 CPU 解码语义测试
tests/test_compare.py    # 对比工具单测
tests/test_workloads.py  # 负载定义单测
experiments/log.md       # 每次实验记录
out/                     # 基线/基准输出（gitignored，可在 GPU 主机重新生成）
```

## 常用命令

```bash
uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"

pytest                                  # 全部测试（无 GPU 时 gpu 用例自动跳过）
pytest tests/test_compare.py -q         # CPU 单测，本机可直接跑
pytest -m "gpu and slow"                # GPU 端到端回归

python tools/reference_hf.py --model Qwen/Qwen3-0.6B    # 生成黄金基线
python tools/reference_hf.py --check-determinism        # 同配置重跑并比对
python tools/compare.py --baseline out/golden/Qwen__Qwen3-0.6B/golden.json \
                        --actual   out/run/golden.json
python tools/bench.py --model Qwen/Qwen3-0.6B --load all
python tools/bench.py --list

black . && ruff check . --fix && mypy python/minisgl   # 与参考仓库一致的 lint 栈
```

新增或改动 `tools/` 的入口后同步更新本节与 [README.md](README.md)。

参考仓库本身的运行方式是 `python -m minisgl --model ...`（见其 `CLAUDE.md`），**不适用于**本仓库。

## 阶段工作流（每个阶段都执行）

1. 只读本阶段对应的 1–3 个模块 + 路线图对应小节，先画数据流再写代码。
2. 先定接口（`BaseOP`、`BaseAttnBackend`/`BaseAttnMetadata`、`BasePrefixCache`/`BaseCacheHandle`、`BaseKVCachePool`），再填最朴素的 PyTorch 实现。
3. 与上一阶段 / HF 黄金基线对齐，达到路线图 §2 中该阶段的**全部验收标准**才进入下一阶段。
4. 固化一组性能数字并写入 `experiments/log.md`（日期、commit、配置、TTFT、ITL p50/p99、吞吐、峰值显存）。
5. 每个优化都必须有开关（CLI flag 或环境变量），保证能一键回到上一阶段。
6. 分支 `stage-NN-topic`，验收后打 tag `v0.N-topic`。

核心路径（时间有限时）：阶段 0 → 1 → 2 → 3 → 4 → 5 → 6 → 9 → 12。

七条原则见路线图 §0.2，最关键的三条：**先正确后性能**、**一次只引入一个优化**、**任何时刻系统保持可运行**。

## 跨模块不变量（改调度/KV/attention 前先看）

这些是路线图里分散在多节、但会同时影响 scheduler / kvcache / attention 后端的设计约定：

- **`Req` 状态机**：`cached_len`（已写入 KV 的长度）、`device_len`（当前参与计算的序列长度）、`extend_len = device_len - cached_len`；`complete_one()` 做 `cached_len = device_len; device_len += 1`。`positions` 是绝对位置：prefill 为 `cached_len..device_len-1`，decode 为单个 `cached_len`。
- **KV 池布局**：`(2, num_layers, num_pages + 1, page_size, local_kv_heads, head_dim)`；`+1` 是给 CUDA Graph 填充请求用的 dummy page，从阶段 2 起就预留。
- **页表语义**：scheduler 侧 `Context.page_table` 始终按 **page_size=1 的原始 token 槽位**记录（形状 `(max_running_req + 1, aligned_max_seq_len)`，int32，列数 32 对齐，最后一行 dummy）；由 attention 后端在边界处换算成页号。这样调度逻辑与后端解耦。
- **`out_loc`**：本轮 K/V 要写入的物理槽位；prefill 是 `page_table[table_idx, cached_len:device_len]`，decode 是单个新槽位。**先 store KV，再跑 attention**（decode 的自注意力需要包含当前 token）。
- **GQA 映射**：Q 头到 KV 头用 `q_head // (num_qo_heads // num_kv_heads)`；KV 头数少于 Q 头数时是整除分组，不是取模。
- **前缀缓存生命周期**（路线图阶段 5 是最易错处）：匹配到的 handle 在 prefill 期间 `lock`、结束后 `unlock`；prefill 完整结束后才插入 radix 树；`[old_handle.cached_len, new_handle.cached_len)` 段要释放；请求结束释放尾部。匹配用 `input_ids[:-1]`，插入/匹配长度向下对齐到 `page_size`。
- **确定性**：decode batch 一律按 `uid` 排序，TP 各 rank 的调度顺序必须一致。
- **重叠调度**：`ForwardInput`（batch、sampling args、输入/写回索引）必须冻结；pinned host buffer 不能复用；GPU 仍在执行 batch 时不能释放其页，用 `lazy_free_region` 延后归还；完成判断/发结果/释放页都在"结果处理"阶段做。

## 正确性标准

- **token 级**：greedy 输出必须与 HF 逐 token 完全一致（20+ 条固定 prompt）。
- **logits 级**：bf16 下 `max_abs_diff < 1e-2`，fp32 调试收紧到 `1e-3`。
- 每个新特性都要做 on/off 对比：cache on/off、page_size 1/4/16、graph on/off、overlap on/off、attn 后端切换，输出都应在容差内一致。
- 空闲时跑完整性检查：`free_pages + used_pages == num_pages`；page_size>1 时空闲槽位必须页对齐。
- torch attention 参考后端永远保留，作为新特性/新页大小的对照。

## 参考实现使用方式

- 阅读/实现顺序：`core` → `layers` → `models` → `kvcache` → `attention` → `scheduler` → `engine` → `message/tokenizer/server` → `distributed` → `kernel`。每阶段只看对应文件。
- 上游工程约定（`Context` 全局单例、ZMQ ipc 地址由 PID 派生、`from __future__ import annotations`、100 列、严格 mypy）可参考其 `CLAUDE.md`；本仓库以自身 `pyproject.toml` 为准。
- 上游的 `docs/features.zh-CN.md`、`docs/structures.zh-CN.md` 是理解进程架构与特性的补充材料。
