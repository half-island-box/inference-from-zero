# 用 Nsight Systems 定位性能瓶颈

目标：把"229 tok/s、70 ms/step"这一个数字，拆成**每个阶段 / 每个 kernel** 的时间，
分清到底卡在 GPU 计算、访存、kernel launch 开销，还是 CPU 侧编排与同步。

---

## 1. 已经加好的东西

| 位置 | 内容 |
| --- | --- |
| `inference_from_zero/utils/profiler.py` | NVTX 打点 + 分阶段 CPU 计时，**默认关闭时零开销**（只判断一个 bool） |
| `bench.py --profile` | 打开打点，并用 `cudaProfilerApi` 夹住计时区域；结束打印阶段耗时表 |
| `bench.py --profile-deep` | 额外打开每层 attention 的 range（trace 会大，确认瓶颈大概位置后再用） |
| `bench.py --profile-all` | 默认只录第 1 轮（`--capture-range-end` 只能结束一次），需要多轮时开这个 |
| `bench.py --profile-out FILE` | 把阶段耗时表另存一份文本 |
| `scripts/profile.sh` | 一条命令跑 `nsys profile` + 打印 kernel/NVTX/CUDA API 汇总 |
| `.gitignore` | 忽略 `profiles/` |

打点名（Nsight 里按这些名字过滤 / 聚合）：

```
bench/warmup                 模型加载后的 warmup generate
bench/round:N                第 N 轮基准
  generate/add_request       tokenize + 建 Sequence + 入队（一次性）
  generate/loop              整个 decode 循环（包住所有 step）
    step/schedule            Scheduler.schedule()：挑批、分配/复用 KV block
    step/model_run:prefill   整个 prefill step
    step/model_run:decode    整个 decode step        ← 主要观察对象
    step/postprocess         append token / EOS 判断 / hash_blocks / deallocate
      run/prepare:decode     CPU 造张量 + pinned H2D（input_ids/positions/slot_mapping/block_tables）
      run/temperature_h2d    每步的 temperatures H2D
      model/forward:decode   28 层 transformer 前向
      model/logits:decode    lm_head（vocab 151936，decode 时是纯带宽开销）
      run/sampler_d2h        采样 + `.tolist()` ← 每步唯一的强制同步点
      run/reset_context      清 Context
      attn/store_kvcache     Triton KV 写回（--profile-deep）
      attn/decode            分页 attention（gather + eager softmax，--profile-deep）
  generate/decode_text       tokenizer.decode
```

---

## 2. 怎么跑

```bash
# 0) 先确认 nsys 可用（>= 2022.4，--capture-range-end 需要）
nsys --version

# 1) 推荐：走脚本，自动补 --profile、自动打印汇总
scripts/profile.sh --num-seqs 16 --max-input-len 256 --max-output-len 128

# 2) 用 bench 默认工作量（64 并发）
scripts/profile.sh

# 3) 需要逐层 attention 打点时
scripts/profile.sh --profile-deep --num-seqs 16 --max-input-len 256 --max-output-len 128

# 4) 想同时采 GPU 指标（SM 占用、显存带宽，需要权限）
EXTRA="--gpu-metrics-devices=0" scripts/profile.sh
```

等价的手动命令（脚本内部就是这个）：

```bash
nsys profile \
  -t cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true \
  -o profiles/iftz python bench.py --profile \
  --num-seqs 16 --max-input-len 256 --max-output-len 128
```

**为什么用 `--capture-range=cudaProfilerApi`**：`bench.py` 里每个计时轮次前后调用了
`torch.cuda.profiler.start()/stop()`，nsys 只录这一段。模型加载、warmup、
`torch.compile` 首次编译都不会污染 trace，trace 体积也小一个量级。

---

## 3. 看什么（按优先级）

### 3.1 先看"GPU 是否在空转"

在 `nsys-ui` 里对齐两条轨道：**CUDA HW（GPU 行）** 和 **NVTX（CPU 行，`bench/round:0` 下）**。

- GPU 行有大段空白、而 CPU 行对应的 NVTX range 还亮着 → **忙在 CPU**：
  依次看 `step/schedule`、`run/prepare:*`、`run/sampler_d2h` 谁占大头。
  → 结论指向：pinned 缓冲复用、少同步、CUDA Graph、采样不上 CPU。
- GPU 行很满 → **忙在 GPU**：看 kernel 明细（下一步）。

### 3.2 看 kernel 明细（`cuda_gpu_kern_sum`）

按总时间排序，重点盯这几个"嫌疑 kernel"：

| kernel / 操作 | 现在为什么会慢 |
| --- | --- |
| `index_select` / `copy` / `CatArrayBatchedCopy` | `_decode` 每层把整段 K/V 从分页 cache 抠成连续张量，读+写各一遍 |
| `masked_fill` / `softmax` / `elementwise` 一堆小 kernel | `_attn` 的 eager 分数矩阵 + `-inf` 掩码 + fp32 softmax |
| `vectorized_elementwise`（RMSNorm 相关） | `RMSNorm_mix` 每层 4 次 fp32 上/下转，多趟读写 |
| `gemm` / `cutlass` | 权重 GEMM；decode 时是纯带宽，看它是否被上面那些 kernel 淹没 |
| kernel 数量本身 | 28 层 × 每层 7~9 个 launch，一步 200+ launch；`--sample=none` 下能从 trace 数出来 |

判据：若"单步 kernel 总时间"远小于 step 墙钟时间 → launch/同步开销为主；
若 gather + softmax 类 kernel 占比 > 30% → 分页 attention kernel 是首要任务。

### 3.3 看 CUDA API 与拷贝（`cuda_api_sum`、`cuda_gpu_mem_time_sum`）

关注 `cudaMemcpyAsync`（H2D 的大小与次数）、`cudaStreamSynchronize`、`cudaDeviceSynchronize`。
decode 每步只该有**一次**同步（`sampler_d2h` 的 `.tolist()`）；
若 `cudaStreamSynchronize` 出现几十次/步，说明有地方在逐层取 GPU 标量（`_decode` 里的
`torch.tensor(chunk_lens, device=...)` 这类），这是可以直接消掉的。

### 3.4 阶段耗时表（不装 nsys 也能看）

`--profile` 跑完会打印：

```
===== bench phase timing (CPU wall time) =====
phase                     calls     cpu_s   ms/call   share
step/model_run:decode      1750    62.100     35.49    84.1%
step/schedule              1750     2.100      1.20     2.8%
run/sampler_d2h:decode     1750     3.200      1.83     4.3%
...
```

这是 **CPU 墙钟**（含 GPU 排队与同步等待），配合 NVTX 时间线就能区分"CPU 慢"还是"GPU 慢"。

---

## 4. 注意

- **打点会拖慢运行**：`--profile` 下的 `throughput_tok_s` 比正常跑低，**不要拿来当性能数字**。
  因此 `--profile` 默认**不写 CSV**（避免污染 `bench_results.csv` 的纵向对比）；
  确实要写就显式传 `--results /tmp/prof.csv`。
- trace 体积：默认工作量 64 并发 × ~180 步，普通 tag 大约几十~几百 MB；
  `--profile-deep` 会乘上 28 层，先确认瓶颈再用，必要时 `--num-seqs 16 --repeat 1`。
- TP > 1 时会 spawn 子进程，脚本已带 `--trace-fork-before-exec=true`；trace 里会看到多个进程，
  记得在 Nsight 里把 rank 0 的进程单独过滤出来看。
- `--sample=none`、`--cpuctxsw=none` 是刻意关掉的：CPU 采样和上下文切换追踪对
  这个"GPU kernel 密集"的场景没帮助，只会放大扰动。
- 需要**单个 kernel 的微观指标**（occupancy、访存效率、L2 命中）时，nsys 不够用，
  另开 `ncu`：
  ```bash
  ncu --set full --kernel-name regex:softmax -c 20 python bench.py --num-seqs 16 ...
  ```
  `ncu` 会强制串行化 kernel、极慢，只对已经定位到的小 kernel 用。

---

## 5. 拿到 trace 之后 -> 对应哪个优化

| trace 里的现象 | 优先做的事 |
| --- | --- |
| GPU 行大量空白 + CPU 侧 `step/model_run` 长 | CUDA Graph（先让 attention 形状固定）、pinned 复用、减少 per-layer H2D |
| `index_select`/copy 类 kernel 占比高 | 写分页 attention kernel，直接吃 block_tables |
| softmax/masked_fill 类 kernel 多且碎 | 分页 attention kernel 里融合 mask+softmax；prefill 换 varlen flash/SDPA |
| 每层一堆小 elementwise（norm/rope） | 融合 RMSNorm(+残差)、融合 RoPE |
| `gemm` 时间占比高但 GPU 利用率低 | 合并 QKV / gate+up 的 GEMM，减少 launch 与权重读放大 |
| `cudaMemcpyAsync` H2D 次数多 | `prepare_*` 改成预分配 pinned staging buffer |
| 同步次数远超 1 次/步 | 去掉 `.tolist()`（GPU 累积）、hoist `_decode` 里的 per-layer 张量构造 |

（各项具体改法与预期收益见 `project_structure_and_optimizations.md`。）

---

## 6. 在其它代码里加打点

```python
from inference_from_zero.utils import profiler as prof

# 一段代码（同时计 CPU 时间 + NVTX range）
with prof.phase("my/stage"):
    do_something()

# 高频位置，只在 --profile-deep 时才推 range（不影响正常跑分）
with prof.nvtx("my/hot_loop", deep=True):
    for _ in range(28):
        layer()

# 装饰整个函数
@prof.nvtx_fn("my/fn")
def fn():
    ...
```

不加任何参数时全部是 no-op，不会影响正常跑的性能。也可以用环境变量控制：

```bash
IFTZ_NVTX=1 IFTZ_NVTX_DEEP=1 python bench.py    # 等价于 --profile-deep
```

