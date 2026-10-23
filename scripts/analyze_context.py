"""Context-management analysis over one or more run directories.

    python scripts/analyze_context.py runs/hymem_smoke_randbats runs/longctx_smoke [--out runs/_analysis]
    python scripts/analyze_context.py runs/... --export-dilution        # traces for KFD/KFSR scripts

Reads report.json + calls_<agent>.jsonl and produces:
  1. a comparison table (win rate, decisions, main-context tokens per decision, total
     prompt tokens per decision, LLM/tool calls, folds, fallbacks, key-fact survival);
  2. context_growth.html -- main-context size vs battle progress per agent, plus HyMem's
     total (all spaces) as a dashed line: the paper's Figure 4 analogue;
  3. key-fact survival KFSR_t = facts present in the decision context / facts revealed so
     far, and a rough density KFD_t (fact characters / context characters): Figure 1;
  4. --export-dilution: per-battle traces in the HyMem analysis schema
     (stats.planner_context_per_turn = [{turn, context_length, messages}]) + key_facts.jsonl.

"Main context" = the persistent decision context: HyMem's planner, or the single
conversation of a flat / summary agent.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

MAIN_SPACES = ("planner", "flat")
_STOP = {"opp", "your", "has", "holds", "ability", "is", "side", "into", "terastallized", "fainted",
         "turn", "the", "a"}


def to_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def fact_signature(text: str) -> list[str]:
    """'opp Garganacl has Salt Cure' -> ['garganacl', 'saltcure']; 'your Hitmontop fainted (turn 4)'
    -> ['hitmontop', 'fainted']."""
    text = re.sub(r"\(turn \d+\)", "", text).strip()
    m = re.match(r"(?:opp|your)\s+side has\s+(.+)$", text)
    if m:
        return [to_id(m.group(1))]
    m = re.match(r"(?:opp|your)\s+(.+?)\s+fainted", text)
    if m:
        return [to_id(m.group(1)), "fainted"]
    m = re.match(r"(?:opp|your)\s+(.+?)\s+(has ability|holds|has|is|terastallized into)\s+(.+)$", text)
    if m:
        return [to_id(m.group(1)), to_id(m.group(3))]
    return [w for w in (to_id(x) for x in text.split()) if w and w not in _STOP]


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def analyze_agent(calls: list[dict]) -> dict:
    """Per-agent aggregates + per-battle decision series."""
    battles: dict = defaultdict(lambda: {"decisions": defaultdict(lambda: {"main": 0, "total": 0, "calls": 0,
                                                                          "messages": None, "turn": 0}),
                                         "folds": 0, "won": None, "turns": 0, "key_facts": []})
    for r in calls:
        tag, space = r.get("battle_tag", ""), r.get("space", "")
        b = battles[tag]
        if space == "event:fold":
            b["folds"] += 1
            continue
        if space == "event:battle_end":
            ex = r.get("extra") or {}
            b["won"], b["turns"] = ex.get("won"), ex.get("turns", 0)
            b["key_facts"] = ex.get("key_facts") or []
            continue
        if space.startswith("event:"):
            continue
        d = b["decisions"][r.get("decision", 0)]
        toks = r.get("prompt_tokens") or int((r.get("prompt_chars") or 0) / 3.5)
        d["total"] += toks
        d["calls"] += 1
        if space in MAIN_SPACES:
            d["main"] = max(d["main"], toks)          # the largest decision context this decision
            if r.get("messages") is not None:
                d["messages"] = r["messages"]
    return battles


def survival(battle: dict) -> tuple[list[float], list[float]]:
    """KFSR_t and KFD_t for each decision that has a logged main context.

    A fact counts as introduced once revealed: by its battle turn when the log has
    turns (decision index mapped proportionally onto turns), otherwise (older logs)
    from the first decision whose context contains it."""
    raw = [(f.get("turn") if isinstance(f, dict) else None, f.get("text") if isinstance(f, dict) else f)
           for f in battle["key_facts"]]
    facts = [(t, txt, fact_signature(txt)) for t, txt in raw if txt]
    decisions = [(k, d) for k, d in sorted(battle["decisions"].items()) if d["messages"]]
    contexts = [to_id("\n".join(str(m.get("content") or "") for m in d["messages"])) for _, d in decisions]
    first_seen = {}
    for i, norm in enumerate(contexts):
        for _, txt, sig in facts:
            if txt not in first_seen and all(tok and tok in norm for tok in sig):
                first_seen[txt] = i
    kfsr, kfd = [], []
    n = max(1, len(decisions))
    for i, ((_, d), norm) in enumerate(zip(decisions, contexts)):
        horizon = (i + 1) / n * max(battle["turns"] or 1, 1)
        known = [f for f in facts
                 if (f[0] is not None and f[0] <= horizon) or (f[0] is None and first_seen.get(f[1], n) <= i)]
        if not known:
            continue
        present = [f for f in known if all(tok and tok in norm for tok in f[2])]
        kfsr.append(len(present) / len(known))
        ctx_chars = sum(len(str(m.get("content") or "")) for m in d["messages"])
        kfd.append(sum(len(f[1]) for f in present) / max(1, ctx_chars))
    return kfsr, kfd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="runs/_analysis")
    ap.add_argument("--export-dilution", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, series = [], []
    for run in map(Path, args.runs):
        report = json.loads((run / "report.json").read_text()) if (run / "report.json").exists() else {}
        for calls_path in sorted(run.glob("calls_*.jsonl")):
            agent = calls_path.stem[len("calls_"):]
            battles = analyze_agent(load_jsonl(calls_path))
            rep = (report.get("agents") or {}).get(agent, {})
            mains, totals, ncalls, kfsr_all, kfd_all, folds = [], [], [], [], [], 0
            for tag, b in battles.items():
                dec = sorted(b["decisions"].items())
                if not dec:
                    continue
                folds += b["folds"]
                n = len(dec)
                for i, (_, d) in enumerate(dec):
                    mains.append(d["main"])
                    totals.append(d["total"])
                    ncalls.append(d["calls"])
                    series.append({"agent": f"{run.name}/{agent}", "battle": tag, "progress": (i + 1) / n * 100,
                                   "main": d["main"], "total": d["total"]})
                k, f = survival(b)
                kfsr_all += k
                kfd_all += f
                if args.export_dilution:
                    tdir = out / "dilution" / run.name / agent
                    tdir.mkdir(parents=True, exist_ok=True)
                    per_turn = [{"turn": i + 1, "context_length": d["main"], "messages": d["messages"] or []}
                                for i, (_, d) in enumerate(dec)]
                    (tdir / f"{tag}.json").write_text(json.dumps(
                        {"task_id": tag, "won": b["won"], "stats": {"planner_context_per_turn": per_turn}}))
                    with open(out / "dilution" / "key_facts.jsonl", "a", encoding="utf-8") as kf:
                        kf.write(json.dumps({"task_id": tag, "agent": agent, "run": run.name,
                                             "key_facts": b["key_facts"]}) + "\n")
            n_b = len([b for b in battles.values() if b["decisions"]])
            avg = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
            rows.append({
                "run": run.name, "agent": agent, "battles": n_b,
                "win_rate": rep.get("win_rate"), "decisions": len(mains),
                "main_ctx_avg": round(avg(mains)), "main_ctx_max": max(mains) if mains else 0,
                "total_prompt_per_decision": round(avg(totals)), "llm_calls_per_decision": round(avg(ncalls), 2),
                "folds_per_battle": round(folds / n_b, 2) if n_b else 0,
                "fallback_rate": rep.get("fallback_rate"),
                "tool_calls": rep.get("tool_calls"),
                "kfsr_mean": round(avg(kfsr_all), 3) if kfsr_all else None,
                "kfsr_last_quarter": round(avg(kfsr_all[-max(1, len(kfsr_all) // 4):]), 3) if kfsr_all else None,
                "kfd_mean": round(avg(kfd_all), 4) if kfd_all else None,
            })

    cols = list(rows[0].keys()) if rows else []
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |" for r in rows]
    table = "\n".join(md)
    print(table)
    (out / "summary.md").write_text(table + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for agent in sorted({s["agent"] for s in series}):
            pts = sorted((s for s in series if s["agent"] == agent), key=lambda s: s["progress"])
            fig.add_trace(go.Scatter(x=[p["progress"] for p in pts], y=[p["main"] for p in pts],
                                     mode="lines+markers", name=f"{agent} (decision context)"))
            if any(p["total"] > p["main"] * 1.05 for p in pts):
                fig.add_trace(go.Scatter(x=[p["progress"] for p in pts], y=[p["total"] for p in pts],
                                         mode="lines", line={"dash": "dash"}, name=f"{agent} (all spaces)"))
        fig.update_layout(title="Context size per decision vs battle progress",
                          xaxis_title="battle progress (%)", yaxis_title="prompt tokens", yaxis_type="log")
        fig.write_html(str(out / "context_growth.html"))
        print(f"\nplot -> {out / 'context_growth.html'}")
    except ImportError:
        print("\n(plotly not installed: skipped the plot; pip install plotly)")
    print(f"table -> {out / 'summary.md'}")
    if args.export_dilution:
        print(f"dilution traces -> {out / 'dilution'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
