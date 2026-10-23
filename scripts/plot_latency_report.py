"""Static figures (PNG) for the real-time report, from analyze_latency.py's summary.

    python scripts/plot_latency_report.py runs/hymem_latency_randbats/latency/latency_summary.json \
        --out reports/figures

fig1  per-decision wall time, stacked by context space (+ 10 s / 150 s reference lines)
fig2  where the battle's decision time went: space x phase (decode emphasised)
fig3  mean decision latency under single changes (counterfactual estimates)
fig4  call timeline of one typical decision (the serial call chain)

Palette: the dataviz reference categorical slots 1-4 in fixed order (validated:
adjacent CVD dE >= 9.1, normal-vision >= 22.9); aqua/yellow are below 3:1 on the
surface, so values are labelled and the report carries the tables.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"planner": "#2a78d6", "executor": "#eb6834", "iso": "#1baf7a", "fold": "#eda100"}
LABELS = {"planner": "Planner（规划层）", "executor": "Executor（执行层/工具循环）", "iso": "隔离推理",
          "fold": "记忆折叠（摘要+3路抽取）", "rest": "工具执行 + 其他"}
GRAY = "#c3c2b7"

plt.rcParams.update({
    "font.sans-serif": ["Hiragino Sans GB", "PingFang HK", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def group(attrib: dict) -> dict:
    g = {"planner": 0.0, "executor": 0.0, "iso": 0.0, "fold": 0.0, "rest": 0.0}
    for k, v in attrib.items():
        sp = k.split(":")[0]
        if sp in ("fold_summary", "memory"):
            g["fold"] += v
        elif sp in g:
            g[sp] += v
        else:
            g["rest"] += v
    return g


def ref_line(ax, y, text, *, horizontal=True):
    if horizontal:   # label in the right margin, never on top of the marks
        ax.axhline(y, color=MUTED, linewidth=0.9, zorder=3)
        ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8.5, color=INK2,
                    annotation_clip=False)
    else:
        ax.axvline(y, color=MUTED, linewidth=0.9, zorder=1)
        ax.annotate(text, xy=(y, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -4),
                    textcoords="offset points", ha="left", va="top", fontsize=8.5, color=INK2)


def fig1(rep: dict, out: Path) -> None:
    dec = rep["decisions"]
    x = list(range(len(dec)))
    groups = [group(d["attrib"]) for d in dec]
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=180)
    bottom = [0.0] * len(dec)
    top = max(sum(g.values()) for g in groups) or 1.0
    for key in ("planner", "executor", "iso", "fold", "rest"):
        vals = [g[key] for g in groups]
        if not any(v > 0.01 * top for v in vals):   # invisible at this scale (see fig2 / tables)
            continue
        ax.bar(x, vals, bottom=bottom, width=0.68, color=SERIES.get(key, GRAY), label=LABELS[key],
               edgecolor=SURFACE, linewidth=0.8, zorder=2)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ref_line(ax, 150, "150 s 天梯单回合上限\n（超过即判负）")
    ref_line(ax, 10, "10 s 天梯可持续节奏\n（每回合 +10 s）")
    for i, d in enumerate(dec):
        if d["folds"]:
            ax.annotate("折叠", xy=(i, bottom[i]), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK2)
    ax.set_ylim(0, max(bottom) * 1.08)
    ax.set_xticks(x, [f"{d['idx']}\nT{d['turn']}" for d in dec], fontsize=7)
    ax.set_xlim(-0.7, len(dec) - 0.3)
    ax.set_ylabel("每次决策墙钟时间（秒）")
    ax.set_xlabel("决策序号 / 回合")
    ax.grid(axis="x", visible=False)
    dl = rep["decision_latency"]
    ax.set_title(f"HyMem 每次决策耗时（单局 {len(dec)} 次决策；均值 {dl['mean']:.0f} s，中位 {dl['median']:.0f} s，"
                 f"最大 {dl['max']:.0f} s）", loc="left", fontsize=11.5, color=INK)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=5, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(right=0.87)
    fig.savefig(out / "fig1_decision_latency.png", facecolor=SURFACE)
    plt.close(fig)


def fig2(rep: dict, out: Path) -> None:
    name = {"planner": "Planner", "executor": "Executor", "iso": "隔离推理", "fold_summary": "折叠·摘要",
            "memory": "折叠·记忆抽取", "tool": "工具执行", "other": "Python/其他"}
    phase = {"prefill": "预填充（读输入）", "decode": "解码（生成输出）"}
    items = []
    for k, v in rep["attribution_s"].items():
        sp, _, ph = k.partition(":")
        label = f"{name.get(sp, sp)} · {phase.get(ph, '')}".rstrip(" ·") if ph else name.get(sp, sp)
        items.append((label, v, ph == "decode"))
    items.sort(key=lambda t: t[1])
    total = sum(v for _, v, _ in items) or 1.0
    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=180)
    y = range(len(items))
    ax.barh(list(y), [v for _, v, _ in items], height=0.6,
            color=[SERIES["planner"] if dec else MUTED for _, _, dec in items], zorder=2)
    for i, (_, v, _) in enumerate(items):
        amount = f"{v / 60:.1f} 分钟" if v >= 60 else f"{v:.1f} 秒"
        ax.annotate(f"{amount}  ({v / total:.1%})", xy=(v, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(list(y), [lab for lab, _, _ in items], fontsize=9)
    ax.set_xlabel("整局累计墙钟时间（秒，重叠调用按并发数均分）")
    ax.set_xlim(0, max(v for _, v, _ in items) * 1.28)
    ax.grid(axis="y", visible=False)
    dec_share = sum(v for _, v, d in items if d) / total
    ax.set_title(f"时间去向：解码（生成 token）占 {dec_share:.0%}；工具执行与 Python 开销可忽略",
                 loc="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig2_time_breakdown.png", facecolor=SURFACE)
    plt.close(fig)


def fig3(rep: dict, out: Path) -> None:
    label = {"measured": "实测（当前 HyMem）",
             "async_fold": "记忆折叠移出关键路径（异步）",
             "rule_results": "Executor 不再生成 <results>（规则合成）",
             "executor_one_round": "Executor 每个任务只用 1 轮工具调用",
             "cap_output_32": "每次调用最多生成 32 token",
             "async_fold+rule_results+cap32": "以上三项（异步折叠+规则合成+限长）组合",
             "planner_only_passive_tools": "仅 Planner：工具结果被动注入（无 Executor）",
             "single_call_floor": "下限：每决策 1 次调用、只输出动作"}
    m = rep["counterfactual_mean_s"]
    keys = [k for k in label if k in m]
    keys.sort(key=lambda k: m[k])
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)
    y = range(len(keys))
    ax.barh(list(y), [m[k] for k in keys], height=0.6,
            color=[GRAY if k == "measured" else SERIES["planner"] for k in keys], zorder=2)
    for i, k in enumerate(keys):
        ax.annotate(f"{m[k]:.0f} s", xy=(m[k], i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_yticks(list(y), [label[k] for k in keys], fontsize=9)
    ref_line(ax, 10, "10 s", horizontal=False)
    ax.set_xlabel("平均每次决策耗时（秒；由实测分解推算的估计值）")
    ax.set_xlim(0, max(m.values()) * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_title("单项改动后的平均决策耗时（反事实估计，本机同一服务栈）", loc="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig3_counterfactuals.png", facecolor=SURFACE)
    plt.close(fig)


def fig4(rep: dict, out: Path, decision: int | None = None, name: str = "fig4_decision_timeline.png",
         want_fold: bool = False) -> None:
    dec = [d for d in rep["decisions"] if d["kind"] not in ("recall", "single_option") and d["n_calls"] >= 3
           and bool(d["folds"]) == want_fold]
    if not dec:
        return
    if decision is None:   # the decision closest to its group's median latency
        med = statistics.median(d["latency_s"] for d in dec)
        decision = min(dec, key=lambda d: abs(d["latency_s"] - med))["decision"]
    pick = next(d for d in rep["decisions"] if d["decision"] == decision)
    calls = [c for c in rep["calls"] if c["decision"] == decision]
    if not calls:
        return
    t0 = min(c["t_start"] for c in calls)
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(calls) + 1.6), dpi=180)
    names = []
    for i, c in enumerate(calls):
        sp = c["space"]
        col = SERIES["fold"] if sp in ("fold_summary", "memory") else SERIES.get(sp, GRAY)
        ax.barh(i, c["prefill_end"] - c["t_start"], left=c["t_start"] - t0, height=0.62, color=col, alpha=0.35,
                edgecolor=SURFACE, linewidth=0.8, zorder=2)
        ax.barh(i, c["t_end"] - c["prefill_end"], left=c["prefill_end"] - t0, height=0.62, color=col,
                edgecolor=SURFACE, linewidth=0.8, zorder=2)
        ax.annotate(f"读 {c['uncached']} / 生成 {c['completion_tokens']} tok", xy=(c["t_end"] - t0, i),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)
        kind = {"planner": "Planner", "executor": "Executor", "iso": "隔离推理", "fold_summary": "折叠·摘要",
                "memory": "折叠·记忆抽取"}.get(sp, sp)
        if sp == "executor":
            kind += "（写 <results>）" if c.get("exec_kind") == "results" else "（工具调用）"
        if c.get("finish_reason") == "length":
            kind += "［截断］"
        names.append(kind)
    ax.set_yticks(range(len(calls)), names, fontsize=8.5)
    ax.invert_yaxis()
    span = max(c["t_end"] for c in calls) - t0
    ax.set_xlim(0, span * 1.22)
    ax.set_xlabel("决策内时间（秒）；浅色 = 预填充（含排队），深色 = 解码；［截断］= 生成被 max_tokens 截断")
    ax.grid(axis="y", visible=False)
    what = "含记忆折叠的典型决策" if want_fold else "典型决策（委派、无折叠）"
    ax.set_title(f"{what}的调用链：#{pick['idx']}，回合 {pick['turn']}，{pick['latency_s']:.0f} s，"
                 f"{len(calls)} 次 LLM 调用，串行 {pick['rounds']} 轮", loc="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / name, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--timeline-decision", type=int, default=None)
    args = ap.parse_args()
    rep = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig1(rep, out)
    fig2(rep, out)
    fig3(rep, out)
    fig4(rep, out, args.timeline_decision, "fig4_decision_timeline.png", want_fold=False)
    fig4(rep, out, None, "fig5_fold_decision_timeline.png", want_fold=True)
    print(f"figures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
