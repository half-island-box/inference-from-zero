#!/usr/bin/env bash
# 用 Nsight Systems 录一次 bench 的 trace，并直接打印 kernel / NVTX / CUDA API 汇总。
#
# 用法：
#   scripts/profile.sh                                  # 默认工作量
#   scripts/profile.sh --num-seqs 16 --max-input-len 256 --max-output-len 128
#   scripts/profile.sh --profile-deep                   # 连每层 attention 都打点（trace 大）
#   PYTHON=.venv/bin/python scripts/profile.sh ...      # 指定解释器
#
# 可调环境变量：
#   OUT_DIR=profiles        输出目录
#   NAME=<自动时间戳>       trace 文件名（不含后缀）
#   TRACE=cuda,nvtx,osrt    nsys --trace 内容
#   EXTRA="--gpu-metrics-devices=0"   追加给 nsys 的参数（GPU 采样需要 root/权限）
#   STATS=0                 跳过跑完后的 nsys stats 汇总
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
OUT_DIR=${OUT_DIR:-profiles}
NAME=${NAME:-iftz_$(date +%Y%m%d_%H%M%S)}
TRACE=${TRACE:-cuda,nvtx,osrt}
EXTRA=${EXTRA:-}
STATS=${STATS:-1}

if ! command -v nsys >/dev/null 2>&1; then
  cat >&2 <<'EOF'
找不到 nsys（Nsight Systems）。
  Ubuntu:  apt install nsight-systems-2024.5.1
  NGC 镜像: 已预装，通常在 /opt/nvidia/nsight-systems/<ver>/bin/nsys
装好后确认：nsys --version   # 需要 >= 2022.4（--capture-range-end 的支持版本）
EOF
  exit 1
fi

mkdir -p "$OUT_DIR"

# 用户没显式带 --profile 就自动补上，避免录出空 trace
ARGS=("$@")
has_profile=0
for a in "${ARGS[@]:-}"; do
  [[ "$a" == "--profile" || "$a" == --profile=* || "$a" == "--profile-deep" || "$a" == "--profile-all" ]] && has_profile=1
done
if [[ $has_profile -eq 0 ]]; then
  ARGS+=("--profile")
fi

echo "=== nsys trace: $OUT_DIR/$NAME.nsys-rep ==="
echo "=== cmd: $PYTHON bench.py ${ARGS[*]} ==="

# --capture-range=cudaProfilerApi：只录 bench.py 里 torch.cuda.profiler.start()..stop() 夹住的
#                                那一段，warmup / 模型加载 / tokenize 都不进 trace。
# --sample=none / --cpuctxsw=none：减少 nsys 自身开销，避免扰动被测性能。
# --trace-fork-before-exec=true：TP>1 时会 spawn 子进程，让 nsys 跟进去。
set -x
nsys profile \
  --trace="$TRACE" \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace-fork-before-exec=true \
  --force-overwrite=true \
  ${EXTRA} \
  -o "$OUT_DIR/$NAME" \
  "$PYTHON" bench.py "${ARGS[@]}"
set +x

REP="$OUT_DIR/$NAME.nsys-rep"
echo
echo "trace 已生成: $REP"
echo "可视化: nsys-ui $REP   （或拷回本机用 Nsight Systems 打开）"

if [[ "$STATS" != "0" ]]; then
  echo
  echo "===== kernel 时间 Top ====="
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true "$REP" 2>/dev/null | head -25 || true
  echo
  echo "===== NVTX range 汇总（阶段耗时 / 每次调用） ====="
  nsys stats --report nvtx_sum --force-export=true "$REP" 2>/dev/null | head -40 || true
  echo
  echo "===== CUDA API 汇总（含 H2D/D2H 拷贝、同步） ====="
  nsys stats --report cuda_api_sum --force-export=true "$REP" 2>/dev/null | head -25 || true
  echo
  echo "更多报告：nsys stats --report cuda_gpu_trace,nvtx_pushpop_trace,cuda_gpu_mem_time_sum $REP"
fi
