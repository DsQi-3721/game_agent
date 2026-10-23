"""Collect every Jev validation output into one markdown results file (Chinese).

Reads whatever exists (missing parts are reported as not run):
  runs/jev_eval/probe.json                               API feasibility probe
  runs/jev_eval/decisions/summary.json                   offline decisions on recorded positions
  runs/jev_eval/memory/summary_<model>.json (+ _mock)    memory routing (Jev vs keyword rules)
  runs/jev_fast_vs_baselines/{report.json,traj_*.jsonl}  online System-1 battles
  runs/hymem_dual_latency_randbats/traj_hymem_dual.jsonl dual-process real-time battle
    python scripts/summarize_jev_validation.py --out reports/jev_validation_results.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HYMEM_BASELINE = {"mean": 139.5, "p50": 141.7, "p90": 258.7, "max": 293.5, "le10": 0.0}   # hymem_realtime_report.md


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def jl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def f(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def probe_section(p) -> list[str]:
    if not p:
        return ["（未运行：`python scripts/jev_probe.py`）"]
    out = [f"可用模型：{', '.join(map(str, p.get('models_listed') or []))}", "",
           "| 模型 | 实际服务版本 | 单问题延迟 mean / p50 / max (s) | 扇出 1→4→16→64 问题延迟 mean (s) | 对战请求延迟 mean / max (s) | 对战请求输入 token | 成本/决策 ($) |",
           "|---|---|---|---|---|---|---|"]
    for m, r in p.items():
        if not isinstance(r, dict) or m in ("base_url",) or "single_noul" not in r and "error" not in r:
            continue
        if "error" in r:
            out.append(f"| {m} | 错误：{r['error'][:80]} | | | | | |")
            continue
        s, b = r["single_noul"], r["battle_request"]
        fan = " → ".join(f(r["fan_out"][k]["mean"], 2) for k in sorted(r["fan_out"], key=int))
        out.append(f"| {m} | {r.get('served_by')} | {f(s['mean'])} / {f(s['p50'])} / {f(s['max'])} | {fan} | "
                   f"{f(b['mean'])} / {f(b['max'])} | {b['input_tokens']} | {b['cost_usd']} |")
    return out


def decisions_section(d) -> list[str]:
    if not d:
        return ["（未运行：`python scripts/eval_jev_decisions.py`）"]
    out = [f"样本：{d['snapshots']} 个真实对战局面（启发式 vs 最大威力双方视角 + 启发式 vs 随机）。", "",
           "| 指标 | " + " | ".join(d["models"]) + " |", "|---|" + "---|" * len(d["models"])]
    rows = [
        ("请求延迟 mean (s)", lambda s: s["latency_s"]["mean"]), ("请求延迟 p90 (s)", lambda s: s["latency_s"]["p90"]),
        ("请求延迟 max (s)", lambda s: s["latency_s"]["max"]), ("输入 token / 决策", lambda s: round(s["input_tokens_mean"])),
        ("成本 / 决策 ($)", lambda s: f"{s['cost_usd_per_decision']:.6f}"), ("错误数", lambda s: s["errors"]),
        ("选项顺序一致率（正序=倒序）", lambda s: s["order_consistency"]),
        ("正序选第一个选项的比例", lambda s: s["fwd_picks_first_option"]),
        ("倒序选第一个选项的比例", lambda s: s["rev_picks_first_option"]),
        ("非强制回合的换人比例", lambda s: s["switch_rate_unforced"]),
        ("　SimpleHeuristics 的换人比例", lambda s: s["heuristics_switch_rate_unforced"]),
        ("选择太晶招式的比例", lambda s: s["tera_rate"]),
        ("与 SimpleHeuristics 一致", lambda s: s["agree_heuristics"]), ("与最大伤害一致", lambda s: s["agree_max_damage"]),
        ("随机一致基线 (1/n)", lambda s: s["chance_agreement"]),
        ("客观失误率（免疫招式 / 漏掉先手击倒）", lambda s: s["blunder_rate"]),
        ("　其中：选了对方免疫的招式", lambda s: s["blunder_immune"]),
        ("　其中：漏掉先手确定击倒", lambda s: s["blunder_missed_ko"]),
        ("Tier 0（规则可判）局面占比", lambda s: s["tier0_positions"]),
        ("　在这些局面上 Jev 也选了确定击倒", lambda s: s["agree_tier0_on_obvious"]),
        ("SimpleHeuristics 同口径失误率", lambda s: s["simple_heuristics_blunder_rate"]),
        ("最大伤害策略同口径失误率", lambda s: s["max_damage_blunder_rate"]),
        ("门控放行率（走快路径）", lambda s: s["gate"]["coverage"]),
        ("放行局面失误率", lambda s: s["gate"]["blunder_rate_accepted"]),
        ("升级到慢路径局面的失误率", lambda s: s["gate"]["blunder_rate_escalated"]),
        ("放行局面与启发式一致", lambda s: s["gate"]["agree_heuristics_accepted"]),
        ("升级局面与启发式一致", lambda s: s["gate"]["agree_heuristics_escalated"]),
        ("P(hard) 均值", lambda s: s["p_hard_mean"]),
    ]
    for name, fn in rows:
        vals = []
        for s in d["models"].values():
            try:
                vals.append(f(fn(s)))
            except (KeyError, TypeError):
                vals.append("—")
        out.append(f"| {name} | " + " | ".join(vals) + " |")
    for m, s in d["models"].items():
        out += ["", f"**{m} 风险-覆盖曲线**（要求顺序一致，置信度阈值 t）：", "",
                "| t | 覆盖率 | 失误率 | 与启发式一致 |", "|---|---|---|---|"]
        out += [f"| {x['min_confidence']} | {f(x['coverage'])} | {f(x['blunder_rate'])} | {f(x['agree_heuristics'])} |"
                for x in s["risk_coverage"]]
        out.append(f"\n升级原因计数：{s['gate']['reasons']}")
    return out


def memory_section(jev, rules) -> list[str]:
    if not jev and not rules:
        return ["（未运行：`python scripts/eval_jev_memory_router.py`）"]
    cols = [(n, s) for n, s in (("Jev 路由", jev), ("关键词规则（无 API）", rules)) if s]
    out = ["| 指标 | " + " | ".join(n for n, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    rows = [
        ("每次路由请求延迟 mean (s)", lambda s: " / ".join(f"{x:.2f}" for x in s["requests"]["latency_s_mean_by_run"])
         + "（两次运行）" if "latency_s_mean_by_run" in s["requests"] else (
             s["requests"]["latency_s_mean"] if s["requests"]["input_tokens_mean"] else "—")),
        ("每次路由请求延迟 max (s)", lambda s: " / ".join(f"{x:.2f}" for x in s["requests"]["latency_s_max_by_run"])
         if "latency_s_max_by_run" in s["requests"] else (
             s["requests"]["latency_s_max"] if s["requests"]["input_tokens_mean"] else "—")),
        ("每请求问题数", lambda s: s["requests"]["questions_mean"]),
        ("每请求输入 token", lambda s: s["requests"]["input_tokens_mean"] or "—"),
        ("每请求成本 ($)", lambda s: f"{s['requests']['cost_usd_per_request']:.4f}" if s["requests"]["input_tokens_mean"] else "—"),
        ("阈值 min_valid / min_useful", lambda s: f"{s.get('min_valid')} / {s.get('min_useful')}"),
        ("p_valid AUROC（真事实 vs 矛盾/张冠李戴）", lambda s: s["controls"]["p_valid_auroc_true_vs_false"]),
        ("真事实被接受", lambda s: s["controls"]["true_fact_accepted"]),
        ("与当前局面矛盾的事实被拒", lambda s: s["controls"]["contradicted_rejected"]),
        ("别的对局的事实被拒", lambda s: s["controls"]["wrong_battle_rejected"]),
        ("过期数值被丢弃", lambda s: s["controls"]["stale_numbers_dropped"]),
        ("空泛建议被丢弃", lambda s: s["controls"]["generic_dropped"]),
        ("工具技巧只进执行器", lambda s: s["controls"]["tool_tip_to_executor_only"]),
        ("人工标注：类别准确率", lambda s: s["gold"]["kind_accuracy"]),
        ("人工标注：类别 macro-F1", lambda s: s["gold"]["kind_macro_f1"]),
        ("人工标注：保留/丢弃准确率", lambda s: s["gold"]["keep_accuracy"]),
        ("人工标注：保留精确率", lambda s: s["gold"]["keep_precision"]),
        ("人工标注：保留召回率", lambda s: s["gold"]["keep_recall"]),
        ("硬编码（全保留）精确率", lambda s: s["gold"]["hardcoded_keep_precision"]),
        ("真实记忆条目被丢弃比例", lambda s: s["effect"]["real_dropped"]),
        ("规划器记忆字符数：硬编码 → 路由", lambda s: f"{s['effect']['planner_chars_rule_mean']} → {s['effect']['planner_chars_routed_mean']}"),
        ("执行器记忆字符数：硬编码 → 路由", lambda s: f"{s['effect']['executor_chars_rule_mean']} → {s['effect']['executor_chars_routed_mean']}"),
    ]
    for name, fn in rows:
        vals = []
        for _, s in cols:
            try:
                vals.append(f(fn(s)))
            except (KeyError, TypeError):
                vals.append("—")
        out.append(f"| {name} | " + " | ".join(vals) + " |")
    if jev:
        out += ["", f"Jev 丢弃原因：{jev['effect']['dropped_by_reason']}",
                f"Jev 主要混淆（gold→pred）：{jev['gold']['confusion']}"]
    return out


def online_section(run: Path) -> list[str]:
    rep = load(run / "report.json")
    if not rep:
        return [f"（未运行：`python -m arena.cli run configs/experiments/jev_fast_vs_baselines.yaml`）"]
    out = ["| 智能体 | 对手 | 胜 / 负 (n) |", "|---|---|---|"]
    for a, row in rep["matrix"].items():
        for b, r in row.items():
            if a.startswith("jev") or (a == "heuristic" and b in ("random", "maxpow")):
                out.append(f"| {a} | {b} | {r['wins']} / {r['losses']} ({r['n']}) |")
    out += ["", "| 智能体 | 决策数 | 每决策延迟 mean / p50 / p90 / max (s) | ≤1 s 比例 | 门控放行比例 | 交给启发式的比例 | 顺序一致率 | 回退率 |",
            "|---|---|---|---|---|---|---|---|"]
    for a in rep["agents"]:
        if not a.startswith("jev"):
            continue
        rows = [r["extra"] for r in jl(run / f"traj_{a}.jsonl") if r.get("extra")]
        lat = [r["latency_s"] for r in rows if r.get("note") not in ("single_option", "recall")]
        acc = [r for r in rows if "s1_accept" in r]
        out.append(f"| {a} | {len(rows)} | {f(statistics.mean(lat)) if lat else '—'} / {f(pct(lat, .5))} / "
                   f"{f(pct(lat, .9))} / {f(max(lat) if lat else None)} | "
                   f"{f(sum(x <= 1.0 for x in lat) / len(lat)) if lat else '—'} | "
                   f"{f(sum(r['s1_accept'] for r in acc) / len(acc)) if acc else '—'} | "
                   f"{f(sum(r.get('note') == 'escalated' for r in rows) / len(rows)) if rows else '—'} | "
                   f"{f(sum(r['s1_consistent'] for r in acc) / len(acc)) if acc else '—'} | "
                   f"{rep['agents'][a].get('fallback_rate', 0)} |")
    return out


def _dual_stats(run: Path):
    rows = [r["extra"] for r in jl(run / "traj_hymem_dual.jsonl") if r.get("extra")]
    if not rows:
        return None
    lat = [r["latency_s"] for r in rows]
    routes: dict = {}
    for r in rows:
        k = r.get("route", r.get("note") or "other")
        routes[k] = routes.get(k, 0) + 1
    rep = load(run / "report.json") or {}
    agent = (rep.get("agents", {}).get("hymem_dual") or {})
    return {"rows": rows, "lat": lat, "routes": routes, "won": agent.get("wins"), "extra": agent.get("extra", {}),
            "fast": [r["latency_s"] for r in rows if r.get("route") in ("fast", "tier0")],
            "slow_done": [r["latency_s"] for r in rows if r.get("route") == "slow"],
            "slow_cap": sum(1 for r in rows if str(r.get("route", "")).startswith("slow_")),
            "mem": [r for r in rows if "mem_items" in r]}


def dual_section(run: Path) -> list[str]:
    cur = _dual_stats(run)
    if cur is None:
        return ["（未运行：`python -m arena.cli run configs/experiments/hymem_dual_latency_randbats.yaml`，需要 mlx 服务）"]
    old = _dual_stats(Path(str(run) + "_v1_foldbug"))
    cols = [("纯 HyMem（hymem_realtime_report）", None)] + ([("双系统 v1（折叠阻塞）", old)] if old else []) + \
        [("双系统（后台折叠）", cur)]
    head = "| 指标 | " + " | ".join(n for n, _ in cols) + " |"
    out = [head, "|---|" + "---|" * len(cols)]

    def row(name, base, fn):
        vals = [base] + [fn(x) for _, x in cols[1:]]
        out.append(f"| {name} | " + " | ".join(str(v) for v in vals) + " |")
    row("决策数", 49, lambda x: len(x["rows"]))
    row("每决策延迟 mean (s)", HYMEM_BASELINE["mean"], lambda x: f(statistics.mean(x["lat"]), 1))
    row("p50 (s)", HYMEM_BASELINE["p50"], lambda x: f(pct(x["lat"], .5), 1))
    row("p90 (s)", HYMEM_BASELINE["p90"], lambda x: f(pct(x["lat"], .9), 1))
    row("max (s)", HYMEM_BASELINE["max"], lambda x: f(max(x["lat"]), 1))
    row("≤10 s 比例", "0%", lambda x: f"{100 * sum(v <= 10 for v in x['lat']) / len(x['lat']):.0f}%")
    row("≤150 s（天梯单回合上限）比例", "—", lambda x: f"{100 * sum(v <= 150 for v in x['lat']) / len(x['lat']):.0f}%")
    row("快路径（Jev + Tier 0）回合 / 平均延迟", "—",
        lambda x: f"{len(x['fast'])} / {f(statistics.mean(x['fast']), 2) if x['fast'] else '—'} s")
    row("慢路径在预算内完成 / 平均延迟", "—",
        lambda x: f"{len(x['slow_done'])} / {f(statistics.mean(x['slow_done']), 1) if x['slow_done'] else '—'} s")
    row("慢路径超时、改打 Jev 答案", "—", lambda x: x["slow_cap"])
    row("完成的折叠（后台）", "17（同步）", lambda x: int(x["extra"].get("folds", 0)))
    row("记忆路由次数", "—", lambda x: int(x["extra"].get("mem_routes", 0)))
    row("本局胜负", "胜", lambda x: "胜" if x["won"] else "负")
    out.append("")
    out.append(f"路由分布：{cur['routes']}")
    mem = cur["mem"]
    if mem:
        out.append(f"记忆路由（慢回合）：平均条目 {f(statistics.mean(r['mem_items'] for r in mem), 1)}，"
                   f"保留 {f(statistics.mean(r['mem_kept'] for r in mem), 1)}；规划器记忆字符 "
                   f"{f(statistics.mean(r['mem_chars_planner_rule'] for r in mem), 0)}（硬编码）→ "
                   f"{f(statistics.mean(r['mem_chars_planner'] for r in mem), 0)}（路由）")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/jev_validation_results.md")
    ap.add_argument("--memory-model", default="jev-1.13.0")
    args = ap.parse_args()
    lines = ["# Jev（System One）验证结果", "",
             "由 `scripts/summarize_jev_validation.py` 自动生成；设计与解读见 `reports/jev_system1_design.md`。", "",
             "## 1. API 可行性（延迟 / 扇出 / 成本）", ""] + probe_section(load("runs/jev_eval/probe.json")) + \
            ["", "## 2. 离线决策质量（真实局面）", ""] + decisions_section(load("runs/jev_eval/decisions/summary.json")) + \
            ["", "## 3. 记忆路由（真实 HyMem 记忆 + 对照条目 + 人工标注）", ""] + \
            memory_section(load(f"runs/jev_eval/memory/summary_{args.memory_model}.json"),
                           load("runs/jev_eval/memory/summary_mock.json")) + \
            ["", "## 4. 在线对战：纯 System 1（jev_fast）vs 基线", ""] + online_section(Path("runs/jev_fast_vs_baselines")) + \
            ["", "## 5. 双系统实时性（HyMem + Jev，单局）", ""] + dual_section(Path("runs/hymem_dual_latency_randbats"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
