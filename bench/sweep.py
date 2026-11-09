"""kv_pressure 参数扫描：一次性把「KV 容量 / prefill 水位 / 尝试间隔」的曲线跑出来。

每次运行都必须是**独立子进程**：KV block 数在 LLM 初始化时就算死，调度器参数在
构造时读进 ``self``，同一个进程里改不动。所以这个脚本只做三件事：

1. 按 OAT（one-at-a-time）生成配置：以 baseline 为原点，每条轴单独扫一遍，
   其余轴钉在 baseline 上 —— 全交叉是乘法爆炸，单轴曲线已经够读出斜率。
2. 依次跑 ``python -m bench.run --scenario kv_pressure``，每个配置一份 JSONL trace。
3. 只读 trace 里最后那条 ``type="summary"`` 记录（+ ``type="meta"`` 里的引擎参数），
   汇成 CSV + markdown 表。

用法::

    # 1) 先看要跑什么（不真跑）
    python -m bench.sweep --dry-run

    # 2) 真跑（0.6B，一条 512 请求的配置约 20 min，量力而行）
    python -m bench.sweep --outdir out/sweep

    # 3) 已有 trace 时只重新汇总（改列 / 加图不用重跑）
    python -m bench.sweep --from-jsonl 'out/sweep/*.jsonl' --outdir out/sweep

轴（``--axes name=v1,v2,...`` 覆盖，默认见 ``AXES``）::

    kv        -> --kv-blocks                    250,400,600,900
    watermark -> --prefill-watermark            0.85,0.9,0.95,0.98
    interval  -> --prefill-attempt-interval     10,30,70,140

``kv=0`` 表示不指定块数（用引擎按显存算），用来对照 ``--gpu-memory-utilization``。
其余想固定的参数走 ``--extra``，例如::

    python -m bench.sweep --extra "--gpu-memory-utilization 0.5 --max-steps 20000"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter

# --------------------------------------------------------------------------- #
# 轴定义：名字 -> (CLI flag, baseline 值, 默认扫描点)
# --------------------------------------------------------------------------- #
AXES = {
    "kv": ("--kv-blocks", 250, [250, 400, 600, 900]),
    "watermark": ("--prefill-watermark", 0.9, [0.85, 0.9, 0.95, 0.98]),
    "interval": ("--prefill-attempt-interval", 70, [10, 30, 70, 140]),
}
ORDER = ["kv", "watermark", "interval"]


def _args_of(meta: dict) -> dict:
    return meta.get("args") or {}


# --------------------------------------------------------------------------- #
# CSV 列：(列名, 取值函数(summary, meta))
# --------------------------------------------------------------------------- #
COLUMNS = [
    ("kv_blocks", lambda s, m: m.get("num_kv_blocks")),
    ("kv_capacity_tok", lambda s, m: m.get("kv_capacity_tokens")),
    ("watermark", lambda s, m: _args_of(m).get("prefill_watermark")),
    ("interval", lambda s, m: _args_of(m).get("prefill_attempt_interval")),
    ("steps", lambda s, m: s["engine_steps"]),
    ("prefill_steps", lambda s, m: s["steps"]["kinds"].get("prefill", 0)),
    ("decode_steps", lambda s, m: s["steps"]["kinds"].get("decode", 0)),
    ("empty_steps", lambda s, m: s["steps"]["kinds"].get("empty", 0)),
    ("prefill_share_pct", lambda s, m: round(
        100 * s["steps"]["kinds"].get("prefill", 0) / max(1, s["engine_steps"]), 2)),
    ("step_sum_s", lambda s, m: round(s["steps"]["duration_sum_ms"] / 1e3, 1)),
    ("preempt_events", lambda s, m: s["preemption"]["events"]),
    ("preempt_reqs", lambda s, m: s["preemption"]["requests"]),
    ("freed_blocks_p50", lambda s, m:
        s["preemption"]["freed_blocks_per_event"]["p50_ms"]),
    ("wasted_tok", lambda s, m: s["prefill_work"]["wasted_tokens"]),
    ("wasted_x_first", lambda s, m: s["prefill_work"]["wasted_over_first_pass"]),
    ("wasted_pct_out", lambda s, m: s["prefill_work"]["wasted_percent_of_output"]),
    ("ttft_p50_ms", lambda s, m: s["ttft_ms"]["all"]["p50_ms"]),
    ("ttft_p99_ms", lambda s, m: s["ttft_ms"]["all"]["p99_ms"]),
    ("admission_p50_ms", lambda s, m:
        s.get("ttft_breakdown_ms", {}).get("admission", {}).get("p50_ms")),
    ("service_p50_ms", lambda s, m:
        s.get("ttft_breakdown_ms", {}).get("service", {}).get("p50_ms")),
    ("tbt_p50_ms", lambda s, m: s["tbt_ms"]["not_preempted"]["p50_ms"]),
    ("tbt_p99_ms", lambda s, m: s["tbt_ms"]["not_preempted"]["p99_ms"]),
    ("tbt_cross_p50_ms", lambda s, m: s["tbt_ms"]["crossing_preempt"]["p50_ms"]),
    ("resume_p50_ms", lambda s, m: s["resume_after_preempt_ms"]["p50_ms"]),
    ("out_tok", lambda s, m: s["prefill_work"]["output_tokens"]),
    ("out_tok_per_s", lambda s, m: (
        None if not s["steps"].get("duration_sum_ms") else round(
            s["prefill_work"]["output_tokens"]
            / (s["steps"]["duration_sum_ms"] / 1e3), 1))),
    ("stop", lambda s, m: s["stop_reason"]),
]

# markdown 表：(列名, 表头, 小数位)
MD_COLUMNS = [
    ("kv_blocks", "kv", 0),
    ("prefill_steps", "prefill步", 0),
    ("prefill_share_pct", "prefill%", 2),
    ("preempt_events", "抢占", 0),
    ("wasted_pct_out", "白算%", 1),
    ("ttft_p50_ms", "TTFT p50", 0),
    ("admission_p50_ms", "├ 准入", 0),
    ("service_p50_ms", "└ 首prefill", 0),
    ("tbt_p50_ms", "TBT p50", 0),
    ("tbt_p99_ms", "TBT p99", 0),
    ("tbt_cross_p50_ms", "被踢恢复", 0),
    ("out_tok_per_s", "tok/s", 1),
]

AXIS_LABEL = {
    "kv": "扫描 KV 容量（--kv-blocks）",
    "watermark": "扫描 prefill 水位（--prefill-watermark）",
    "interval": "扫描 prefill 尝试间隔（--prefill-attempt-interval）",
}
AXIS_LABEL_EN = {
    "kv": "KV blocks (--kv-blocks)",
    "watermark": "prefill watermark (--prefill-watermark)",
    "interval": "prefill attempt interval (--prefill-attempt-interval)",
}

COL_OF = {"kv": "kv_blocks", "watermark": "watermark", "interval": "interval"}

# 每张表的第一列：变化中的那个参数（kv 已经在 MD_COLUMNS 里，无需重复）
AXIS_LEAD = {
    "kv": ("kv_blocks", "kv", 0),
    "watermark": ("watermark", "watermark", 2),
    "interval": ("interval", "interval", 0),
}

# 控制台表格：(列名, 表头, 小数位)
CONSOLE_COLUMNS = [
    ("prefill_share_pct", "prefill%", 2),
    ("preempt_events", "抢占", 0),
    ("wasted_pct_out", "白算%", 1),
    ("ttft_p50_ms", "TTFTp50", 0),
    ("admission_p50_ms", "准入p50", 0),
    ("tbt_p50_ms", "TBTp50", 0),
    ("resume_p50_ms", "恢复p50", 0),
    ("out_tok_per_s", "tok/s", 1),
]


# --------------------------------------------------------------------------- #
# 配置生成
# --------------------------------------------------------------------------- #
def parse_axes(specs: list[str]) -> dict[str, list | None]:
    axes: dict[str, list | None] = {name: None for name in ORDER}
    for spec in specs:
        name, _, values = spec.partition("=")
        name = name.strip()
        if name not in AXES:
            raise SystemExit(f"未知轴: {name}（可用: {', '.join(AXES)}）")
        parsed = []
        for raw in values.split(","):
            raw = raw.strip()
            if not raw:
                continue
            parsed.append(float(raw) if name != "kv" and name != "interval"
                          else int(float(raw)))
        if not parsed:
            raise SystemExit(f"轴 {name} 没有扫描点")
        axes[name] = parsed
    return axes


def make_configs(axes: dict[str, list | None]) -> list[dict]:
    """OAT：baseline + 每条轴单独变化（与 baseline 重合的点跳过、整体去重）。"""
    baseline = {name: AXES[name][1] for name in ORDER}
    configs = [dict(baseline, axis="baseline")]
    for name in ORDER:
        for value in axes.get(name) or AXES[name][2]:
            if value == baseline[name]:
                continue
            configs.append(dict(baseline, **{name: value, "axis": name}))
    seen, unique = set(), []
    for cfg in configs:
        key = (cfg["kv"], cfg["watermark"], cfg["interval"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(cfg)
    return unique


def tag_of(cfg: dict) -> str:
    return f"kv{cfg['kv']}_wm{cfg['watermark']:g}_it{cfg['interval']}"


def build_cmd(cfg: dict, trace: str, extra: list[str]) -> list[str]:
    return ([sys.executable, "-m", "bench.run",
             "--scenario", "kv_pressure", "--trace", trace,
             AXES["kv"][0], str(cfg["kv"]),
             AXES["watermark"][0], f"{cfg['watermark']:g}",
             AXES["interval"][0], str(cfg["interval"])] + extra)


# --------------------------------------------------------------------------- #
# trace 解析
# --------------------------------------------------------------------------- #
def load_trace(path: str) -> tuple[dict, dict]:
    summary = meta = None
    errors = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = rec.get("type")
            if kind == "summary":
                summary = rec
            elif kind == "meta":
                meta = rec
            elif kind == "error":
                errors.append(rec.get("error"))
    if summary is None:
        raise SystemExit(f"{path}: 没有 summary 记录（error={errors[-1:] or '无'}）")
    return summary, (meta or {})


def summarize_file(path: str) -> dict:
    summary, meta = load_trace(path)
    row = {"file": os.path.basename(path)}
    for name, take in COLUMNS:
        try:
            row[name] = take(summary, meta)
        except (KeyError, TypeError, AttributeError):
            row[name] = None
    return row


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def write_csv(rows: list[dict], path: str) -> None:
    names = ["file"] + [name for name, _ in COLUMNS]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(names) + "\n")
        for row in rows:
            fh.write(",".join("" if row.get(n) is None else str(row[n])
                              for n in names) + "\n")


def _fmt(value, digits: int | None = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{0 if digits is None else digits}f}"
    return str(value)


def _mode(values: list) -> object:
    values = [v for v in values if v is not None]
    return Counter(values).most_common(1)[0][0] if values else None


def split_axes(rows: list[dict]) -> tuple[dict, dict[str, list[dict]]]:
    """按轴分组：其他两轴等于众数（= baseline）的那些行 = 该轴的曲线。

    baseline 取众数而不是命令行参数，是因为 ``--from-jsonl`` 只看得到 trace；
    OAT 扫描里众数天然就是 baseline。
    """
    base = {name: _mode([row.get(COL_OF[name]) for row in rows]) for name in ORDER}

    def hits(row: dict, skip: str | None = None) -> bool:
        return all(name == skip or base[name] is None
                   or row.get(COL_OF[name]) == base[name] for name in ORDER)

    groups: dict[str, list[dict]] = {"baseline": [r for r in rows if hits(r)]}
    for name in ORDER:
        # 轴曲线要含 baseline 点（否则左端缺一个点，斜率看不出来）；
        # hits(r, skip=name) 本身就会把 baseline 那行包含进来。
        groups[name] = [r for r in rows if hits(r, skip=name)]
    return base, groups


def write_markdown(rows: list[dict], path: str) -> None:
    base, groups = split_axes(rows)

    lines = ["# kv_pressure 参数曲线", ""]
    lines.append(f"共 {len(rows)} 个配置，baseline = "
                 f"kv={base['kv']} / watermark={base['watermark']} / "
                 f"interval={base['interval']}。")
    lines.append("")
    lines.append("列含义：`prefill%` = prefill 步占比（= 预填充机会），"
                 "`白算%` = 重算 token / 产出 token，`├ 准入` = submit → 首次被排进 "
                 "prefill 批（排队等 KV / 等机会），`└ 首prefill` = 那次 prefill 到首 token "
                 "落地，`被踢恢复` = 被抢占到下一个 token 落地的中位空档。"
                 "每张轴表里 `★` 标的是 baseline 那一行。")
    lines.append("")

    def table(group: list[dict], axis: str = "kv") -> None:
        lead = AXIS_LEAD[axis]
        cols = MD_COLUMNS if lead[0] == MD_COLUMNS[0][0] else [lead] + MD_COLUMNS
        lines.append("| " + " | ".join(label for _, label, _ in cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for row in sorted(group, key=lambda r: tuple(
                (r.get(COL_OF[n]) is None, r.get(COL_OF[n]) or 0) for n in ORDER)):
            is_base = all(base[n] is None or row.get(COL_OF[n]) == base[n]
                          for n in ORDER)
            cells = [_fmt(row.get(name), digits) for name, _, digits in cols]
            if is_base:
                cells[0] = f"★ {cells[0]}"
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## baseline")
    lines.append("")
    table(groups["baseline"], "kv")
    for name in ORDER:
        group = groups[name]
        if len({r.get(COL_OF[name]) for r in group}) < 2:
            continue
        lines.append(f"## {AXIS_LABEL[name]}")
        lines.append("")
        table(group, name)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# 曲线图（可选：没装 matplotlib 就跳过）
# --------------------------------------------------------------------------- #
# 每张图 2x2 面板，对应的正是「供给 / 代价 / 准入 / 稳态」四个问题。
# (中文标题, 英文标题, y 轴标签, 左轴系列, 右轴系列)
PANELS = [
    ("prefill 机会与抢占次数", "prefill opportunity vs preemptions",
     "prefill step share (%)",
     [("prefill_share_pct", "prefill step share (%)", "o")],
     [("preempt_events", "preemption events", "s")]),
    ("白算 token 占比", "wasted prefill share",
     "wasted / output (%)",
     [("wasted_pct_out", "wasted prefill tokens / output tokens (%)", "o")],
     []),
    ("TTFT 拆段（log）", "TTFT breakdown (log)",
     "ms",
     [("ttft_p50_ms", "TTFT p50", "o"),
      ("admission_p50_ms", "admission p50 (queueing)", "s"),
      ("service_p50_ms", "first-prefill service p50", "^")],
     []),
    ("decode 间隔（log）", "decode gap (log)",
     "ms",
     [("tbt_p50_ms", "TBT p50 (not preempted)", "o"),
      ("tbt_p99_ms", "TBT p99 (not preempted)", "s"),
      ("tbt_cross_p50_ms", "TBT p50 crossing preempt", "^"),
      ("resume_p50_ms", "resume after preempt p50", "D")],
     []),
]

# 有 CJK 字体就用中文面板标题，没有就用英文（避免 matplotlib 画出方块 + 刷警告）
CJK_FONTS = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Microsoft YaHei",
    "PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei",
    "Droid Sans Fallback",
]


def setup_cjk_font(plt) -> bool:
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _plot_series(ax, group, xkey, series, logy):
    for key, label, marker in series:
        pts = [(row.get(xkey), row.get(key)) for row in group]
        pts = [(x, y) for x, y in pts if x is not None and y is not None]
        if not pts:
            continue
        pts.sort(key=lambda p: p[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker=marker, linewidth=1.6, markersize=5, label=label)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(fontsize=7, loc="best")


def plot_axis(group: list[dict], axis: str, path: str, plt,
              title_suffix: str = "", cjk: bool = False) -> bool:
    xkey = COL_OF[axis]
    rows = [r for r in group if r.get(xkey) is not None]
    if len({r[xkey] for r in rows}) < 2:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, panel in zip(axes.flat, PANELS):
        title_zh, title_en, ylabel, left, right = panel
        ax.set_title(title_zh if cjk else title_en, fontsize=9)
        ax.set_xlabel(AXIS_LABEL_EN[axis], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        _plot_series(ax, rows, xkey, left, logy="log" in title_en)
        if right:
            ax2 = ax.twinx()
            _plot_series(ax2, rows, xkey, right, logy=False)
            ax2.set_ylabel(right[0][1], fontsize=8)
            ax2.tick_params(labelsize=7)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")
    fig.suptitle(f"kv_pressure: {AXIS_LABEL_EN[axis]}{title_suffix}", fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def write_plots(rows: list[dict], outdir: str, prefix: str = "kv_sweep",
                title_suffix: str = "") -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - 没装就不画，不影响 CSV / md
        print(f"  跳过画图：matplotlib 不可用（{type(exc).__name__}: {exc}）",
              file=sys.stderr)
        return []
    cjk = setup_cjk_font(plt)
    if not cjk:
        print("  （没找到 CJK 字体，图内标题用英文）", file=sys.stderr)
    _, groups = split_axes(rows)
    written = []
    for axis in ORDER:
        path = os.path.join(outdir, f"{prefix}_{axis}.png")
        if plot_axis(groups[axis], axis, path, plt, title_suffix, cjk):
            written.append(path)
    return written


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m bench.sweep",
        description="kv_pressure 参数扫描 + 曲线汇总")
    p.add_argument("--axes", action="append", default=[], metavar="NAME=v1,v2,...",
                   help="覆盖某条轴的扫描点，例如 --axes kv=250,400,600")
    p.add_argument("--outdir", default="out/sweep",
                   help="trace / CSV / markdown 输出目录")
    p.add_argument("--extra", default="",
                   help="所有运行共用的额外 bench.run 参数（字符串）")
    p.add_argument("--trace-prefix", default="kv", help="trace 文件名前缀")
    p.add_argument("--timeout-s", type=int, default=7200, help="单次运行超时（秒）")
    p.add_argument("--est-min", type=float, default=22.0,
                   help="单次运行预估分钟数（只用于打印总耗时预估）")
    p.add_argument("--dry-run", action="store_true", help="只打印要跑的命令")
    p.add_argument("--from-jsonl", default=None,
                   help="不跑，直接汇总已有 trace（支持通配符），例如 'out/sweep/*.jsonl'")
    p.add_argument("--plot", action="store_true",
                   help="额外画每个轴的 PNG（需 matplotlib；没有就只输出 csv/md）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    rows: list[dict] = []

    if args.from_jsonl:
        paths = sorted(glob.glob(args.from_jsonl))
        if not paths:
            print(f"没有匹配的 trace: {args.from_jsonl}", file=sys.stderr)
            return 2
        print(f"汇总 {len(paths)} 份 trace")
        for path in paths:
            try:
                row = summarize_file(path)
            except SystemExit as exc:
                print(f"  跳过 {path}: {exc}", file=sys.stderr)
                continue
            rows.append(row)
            print(f"  {row['file']}: steps={row['steps']} "
                  f"prefill={row['prefill_steps']} 抢占={row['preempt_events']} "
                  f"白算={row['wasted_pct_out']}% TTFT p50={row['ttft_p50_ms']}ms")
    else:
        cfgs = make_configs(parse_axes(args.axes))
        extra = args.extra.split()
        total_min = len(cfgs) * args.est_min
        print(f"共 {len(cfgs)} 个配置（每个独立进程，按 {args.est_min:g} min/次 "
              f"预估 {total_min:.0f} min ≈ {total_min / 60:.1f} h）")
        for cfg in cfgs:
            trace = os.path.join(args.outdir, f"{args.trace_prefix}_{tag_of(cfg)}.jsonl")
            print(f"  [{cfg['axis']:>9}] {tag_of(cfg)} -> {trace}")
        if args.dry_run:
            print("\n--dry-run：实际命令如下")
            for cfg in cfgs:
                trace = os.path.join(
                    args.outdir, f"{args.trace_prefix}_{tag_of(cfg)}.jsonl")
                print("  " + " ".join(build_cmd(cfg, trace, extra)))
            return 0

        for i, cfg in enumerate(cfgs, 1):
            trace = os.path.join(
                args.outdir, f"{args.trace_prefix}_{tag_of(cfg)}.jsonl")
            print(f"\n[{i}/{len(cfgs)}] {tag_of(cfg)} ({cfg['axis']})", flush=True)
            t0 = time.time()
            try:
                rc = subprocess.run(build_cmd(cfg, trace, extra),
                                    timeout=args.timeout_s).returncode
            except subprocess.TimeoutExpired:
                rc = -9
                print(f"  超时（>{args.timeout_s}s）", file=sys.stderr)
            print(f"  rc={rc} 用时 {(time.time() - t0) / 60:.1f} min", flush=True)
            if not os.path.exists(trace):
                print("  没有生成 trace，跳过", file=sys.stderr)
                continue
            try:
                rows.append(summarize_file(trace))
            except SystemExit as exc:
                print(f"  解析失败: {exc}", file=sys.stderr)

    if not rows:
        print("没有任何可用结果", file=sys.stderr)
        return 1

    csv_path = os.path.join(args.outdir, "kv_sweep.csv")
    md_path = os.path.join(args.outdir, "kv_sweep.md")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    header = ["config"] + [label for _, label, _ in CONSOLE_COLUMNS]
    print("\n" + "=" * 104)
    print(" ".join(f"{h:>9}" for h in header))
    for row in rows:
        label = (f"kv{row['kv_blocks']}/wm{row['watermark']}/it{row['interval']}"
                 if row.get("kv_blocks") is not None else row["file"][:-6])
        print(" ".join([f"{label:>9}"] +
                       [f"{_fmt(row.get(k), d):>9}" for k, _, d in CONSOLE_COLUMNS]))
    print("=" * 104)
    print(f"CSV: {csv_path}\nmarkdown: {md_path}")
    if args.plot:
        suffix = ""
        for path in write_plots(rows, args.outdir, title_suffix=suffix):
            print(f"png: {path}")
    else:
        print("（加 --plot 可以顺便出 PNG 曲线图）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
