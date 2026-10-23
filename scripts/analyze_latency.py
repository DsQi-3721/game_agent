"""Real-time (latency) analysis of one HyMem run.

    python scripts/analyze_latency.py runs/hymem_latency_randbats \
        --server-log runs/_latency/mlx_server.log --bench runs/_latency/bench_llm_latency.json \
        [--mem runs/_latency/mem_samples.log] [--agent hymem] [--out runs/hymem_latency_randbats/latency]

Inputs: traj_<agent>.jsonl (per decision: wall start/end, latency), calls_<agent>.jsonl
(per LLM call: space, start, latency, prompt/cached/completion tokens; tool and fold
events), the mlx server log (its "Prompt processing progress: P/T" lines; T = the
call's uncached prompt tokens, so the P == T line marks the end of that call's
prefill), the serving benchmark (prefill/decode speed) and optional memory samples.

Outputs (in --out): latency_summary.json, latency_tables.md, latency.html (Plotly).

Wall-clock attribution: inside a decision, every instant is shared equally by the
calls active at that instant, and each call is split into prefill (start -> prefill
end, including any queueing behind other prompts) and decode (prefill end -> end).
Tool execution is its own segment; the remainder of the decision is "other"
(Python / glue). Sums over a decision therefore equal its wall time exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

LADDER = {"name": "ladder", "starting": 150, "grace": 60, "add_per_turn": 10, "max_per_turn": 150,
          "max_first_turn": 150}
CHALLENGE = {"name": "challenge", "starting": 300, "grace": 60, "add_per_turn": 10, "max_per_turn": 300,
             "max_first_turn": 300}
TICK = 5   # Showdown timer tick (server/room-battle.ts TICK_TIME)

_PROGRESS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) - INFO - Prompt processing progress: (\d+)/(\d+)")


def load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mean(xs) -> float:
    xs = list(xs)
    return statistics.mean(xs) if xs else 0.0


# --- inputs -------------------------------------------------------------------------------
def parse_server_log(path: Path | None) -> dict[int, list[float]]:
    """T (uncached prompt tokens) -> sorted epochs of 'P == T' progress lines (prefill ends)."""
    ends: dict[int, list[float]] = defaultdict(list)
    if path is None or not path.exists():
        return ends
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _PROGRESS.match(line)
            if not m or m.group(3) != m.group(4):
                continue
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + int(m.group(2)) / 1000
            ends[int(m.group(4))].append(ts)
    for v in ends.values():
        v.sort()
    return ends


def load_mem(path: Path | None) -> list[dict]:
    out = []
    if path is None or not path.exists():
        return out
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 7:
            continue
        try:
            out.append({"ts": float(p[0]), "free_pages": int(p[1]), "compressor_pages": int(p[2]),
                        "swapins": int(p[3]), "swapouts": int(p[4]), "swap_used": p[5], "mlx_rss_kb": int(p[6])})
        except ValueError:
            continue
    return out


class Model:
    """Serving cost model from the benchmark: latency = a + b * uncached + out / decode_tps."""

    def __init__(self, bench: dict | None):
        s = (bench or {}).get("summary") or {}
        self.a = float(s.get("prefill_fixed_s") or 0.17)
        self.b = float(s.get("prefill_s_per_1k_tokens") or 2.72) / 1000.0
        self.decode_tps = float(s.get("decode_tps_mean") or 10.3)

    def prefill(self, uncached: int) -> float:
        return self.a + self.b * max(0, uncached)

    def decode(self, out_tokens: int) -> float:
        return max(0, out_tokens) / self.decode_tps

    def call(self, uncached: int, out_tokens: int) -> float:
        return self.prefill(uncached) + self.decode(out_tokens)


# --- per-call enrichment ---------------------------------------------------------------------
def enrich_calls(calls: list[dict], ends: dict[int, list[float]], model: Model) -> None:
    used: set = set()
    for c in sorted(calls, key=lambda c: c["t_start"]):
        unc = max(0, int(c.get("prompt_tokens") or 0) - int(c.get("cached_tokens") or 0))
        c["uncached"] = unc
        c["t_end"] = c["t_start"] + float(c.get("latency_s") or 0.0)
        cand = [e for e in ends.get(unc, []) if c["t_start"] - 0.5 <= e <= c["t_end"] + 0.5 and (unc, e) not in used]
        if cand:
            e = cand[0]
            used.add((unc, e))
            c["prefill_end"] = min(max(e, c["t_start"]), c["t_end"])
            c["prefill_src"] = "server_log"
        else:
            c["prefill_end"] = min(c["t_start"] + model.prefill(unc), c["t_end"])
            c["prefill_src"] = "model"
        c["prefill_s"] = c["prefill_end"] - c["t_start"]
        c["decode_s"] = c["t_end"] - c["prefill_end"]
        out = int(c.get("completion_tokens") or 0)
        c["decode_tps"] = (out - 1) / c["decode_s"] if c["decode_s"] > 0.2 and out > 1 else None
        c["pred_s"] = model.call(unc, out)
        if c["space"] == "executor":
            text = str(c.get("output") or "")
            c["exec_kind"] = ("results" if "<results>" in text.lower() or
                              (c.get("finish_reason") == "stop" and text.strip()) else "tool_turn")


# --- per-decision decomposition ----------------------------------------------------------------
def sweep(segments: list[tuple[float, float, str]], lo: float, hi: float) -> tuple[dict, float]:
    """Equal-share wall-clock attribution of [lo, hi] among overlapping labelled segments.
    -> ({label: seconds}, covered seconds)."""
    pts = sorted({lo, hi, *[max(lo, min(hi, s)) for s, _, _ in segments], *[max(lo, min(hi, e)) for _, e, _ in segments]})
    out: dict = defaultdict(float)
    covered = 0.0
    for a, b in zip(pts, pts[1:]):
        if b <= a:
            continue
        active = [lab for s, e, lab in segments if s < b and e > a]
        if not active:
            continue
        dt = b - a
        covered += dt
        for lab in active:
            out[lab] += dt / len(active)
    return dict(out), covered


def rounds(intervals: list[tuple[float, float]]) -> int:
    """Number of sequential groups of overlapping intervals (the call chain depth)."""
    n, end = 0, None
    for s, e in sorted(intervals):
        if end is None or s >= end - 1e-3:
            n += 1
            end = e
        else:
            end = max(end, e)
    return n


def decompose(decisions: list[dict], calls: list[dict], tools: list[dict], folds: list[dict]) -> list[dict]:
    by_dec_calls = defaultdict(list)
    for c in calls:
        by_dec_calls[(c.get("battle_tag"), c.get("decision"))].append(c)
    by_dec_tools = defaultdict(list)
    for t in tools:
        by_dec_tools[(t.get("battle_tag"), t.get("decision"))].append(t)
    by_dec_folds = defaultdict(list)
    for f in folds:
        by_dec_folds[(f.get("battle_tag"), (f.get("extra") or {}).get("decision", f.get("decision")))].append(f)
    rows = []
    prev_turn = None
    for i, d in enumerate(decisions):
        x = d.get("extra") or {}
        key = (d.get("battle_tag"), x.get("decision"))
        recall = x.get("note") == "recall"
        cs = [] if recall else by_dec_calls.get(key, [])
        lat = float(x.get("latency_s") or 0.0)
        t0 = x.get("t_start") or (min((c["t_start"] for c in cs), default=0.0))
        t1 = x.get("t_end") or (t0 + lat)
        segs = []
        for j, c in enumerate(cs):
            segs.append((c["t_start"], c["prefill_end"], ("call", j, "prefill")))
            segs.append((c["prefill_end"], c["t_end"], ("call", j, "decode")))
        for t in ([] if recall else by_dec_tools.get(key, [])):
            ex = t.get("extra") or {}
            if ex.get("t_start"):
                segs.append((ex["t_start"], ex["t_start"] + float(ex.get("latency_s") or 0), ("tool", -1, "tool")))
        raw, covered = sweep(segs, t0, t1)
        attrib: dict = defaultdict(float)
        for c in cs:
            c["attr_prefill"] = c["attr_decode"] = 0.0
        for (kind, j, ph), v in raw.items():
            if kind == "call":
                attrib[f"{cs[j]['space']}:{ph}"] += v
                cs[j][f"attr_{ph}"] += v
            else:
                attrib["tool"] += v
        attrib = dict(attrib)
        attrib["other"] = max(0.0, (t1 - t0) - covered)
        turn = d.get("turn")
        kind = ("recall" if recall else "single_option" if x.get("note") == "single_option"
                else "same_turn" if prev_turn == turn else "new_turn")
        prev_turn = turn
        fl = [] if recall else by_dec_folds.get(key, [])
        rows.append({
            "idx": i + 1, "turn": turn, "decision": x.get("decision"), "kind": kind, "latency_s": lat,
            "t_start": t0, "t_end": t1, "n_calls": len(cs),
            "n_by_space": dict(_count(c["space"] for c in cs)),
            "rounds": rounds([(c["t_start"], c["t_end"]) for c in cs]),
            "attrib": attrib,
            "prompt_tokens": sum(int(c.get("prompt_tokens") or 0) for c in cs),
            "cached_tokens": sum(int(c.get("cached_tokens") or 0) for c in cs),
            "uncached_tokens": sum(c["uncached"] for c in cs),
            "completion_tokens": sum(int(c.get("completion_tokens") or 0) for c in cs),
            "folds": [(f.get("extra") or {}).get("reason") for f in fl],
            "fold_s": sum(float((f.get("extra") or {}).get("seconds") or 0) for f in fl),
            "first_planner": next((c for c in sorted(cs, key=lambda c: c["t_start"]) if c["space"] == "planner"), None),
            "calls": cs,
            "order": d.get("parsed_order"), "fallback": d.get("fallback"),
        })
    return rows


def executor_sessions(calls: list[dict]) -> list[dict]:
    """Group executor calls into sessions by (decision, task line of their first user message)."""
    groups: dict = defaultdict(list)
    for c in calls:
        if c["space"] != "executor":
            continue
        msgs = c.get("messages") or []
        task = ""
        for m in msgs:
            if m.get("role") == "user":
                task = str(m.get("content") or "").split("\n", 1)[0]
                break
        groups[(c.get("battle_tag"), c.get("decision"), task)].append(c)
    out = []
    for (tag, dec, task), cs in groups.items():
        cs.sort(key=lambda c: c["t_start"])
        tool_turns = [c for c in cs if c.get("exec_kind") == "tool_turn"]
        out.append({"decision": dec, "task": task[6:] if task.startswith("Task: ") else task, "calls": cs,
                    "tool_turns": len(tool_turns), "synth_calls": len(cs) - len(tool_turns),
                    "wall_s": cs[-1]["t_end"] - cs[0]["t_start"],
                    "output_tokens": sum(int(c.get("completion_tokens") or 0) for c in cs)})
    return out


_DIRECTIVE = re.compile(r"<(task|answer|isoReason|branch|to_memory|fold_thought)", re.IGNORECASE)
_ANALYSIS = re.compile(r"<analysis>(.*?)(?:</analysis>|$)", re.IGNORECASE | re.DOTALL)


def _overlap_groups(cs: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    end = None
    for c in sorted(cs, key=lambda c: c["t_start"]):
        if groups and end is not None and c["t_start"] < end - 1e-3:
            groups[-1].append(c)
            end = max(end, c["t_end"])
        else:
            groups.append([c])
            end = c["t_end"]
    return groups


def method_work(rows: list[dict]) -> dict:
    """Hardware-agnostic critical-path work per decision, by HyMem mechanism.

    Units do not depend on the serving stack: sequential LLM rounds, generated tokens on the
    critical path (parallel calls count their longest output once) and uncached input tokens.
    Latency on any stack ~= rounds * per-round overhead + out / decode_tps + in / prefill_tps.
    """
    comp: dict = defaultdict(lambda: {"rounds": 0.0, "out": 0.0, "in": 0.0})
    analysis_tok = 0.0
    n_tasks = n_folds = 0
    real = [r for r in rows if r["kind"] not in ("recall", "single_option")]
    for r in real:
        cs = sorted(r["calls"], key=lambda c: c["t_start"])
        planner_seen = False
        for c in cs:
            sp, out, unc = c["space"], int(c.get("completion_tokens") or 0), c["uncached"]
            if sp == "memory":
                continue
            if sp == "planner":
                text = str(c.get("output") or "")
                if not _DIRECTIVE.search(text):
                    key = "planner_wasted"
                elif not planner_seen:
                    key = "planner_first"
                else:
                    key = "planner_after_return"
                planner_seen = planner_seen or key != "planner_wasted"
                ana = " ".join(_ANALYSIS.findall(text))
                if text.strip():
                    analysis_tok += out * len(ana) / max(1, len(text))
                n_tasks += len(re.findall(r"<task>", text, re.IGNORECASE))
            elif sp == "executor":
                key = f"executor_{c.get('exec_kind') or 'tool_turn'}"
            elif sp == "fold_summary":
                key = "fold_summary"
                n_folds += 1
            else:
                key = sp
            comp[key]["rounds"] += 1
            comp[key]["out"] += out
            comp[key]["in"] += unc
        for g in _overlap_groups([c for c in cs if c["space"] == "memory"]):
            comp["fold_extract"]["rounds"] += 1
            comp["fold_extract"]["out"] += max(int(c.get("completion_tokens") or 0) for c in g)
            comp["fold_extract"]["in"] += sum(c["uncached"] for c in g)
    n = max(1, len(real))
    per = {k: {m: v / n for m, v in d.items()} for k, d in comp.items()}
    mech = {
        "baseline_planner_step": ["planner_first"],
        "delegation_round_trips": ["executor_tool_turn", "executor_results", "planner_after_return", "iso"],
        "synchronous_fold": ["fold_summary", "fold_extract"],
        "truncated_steps": ["planner_wasted"],
    }
    by_mech = {m: {u: sum(per.get(k, {}).get(u, 0.0) for k in ks) for u in ("rounds", "out", "in")}
               for m, ks in mech.items()}
    tot = {u: sum(v[u] for v in by_mech.values()) for u in ("rounds", "out", "in")}
    deleg_keys = mech["delegation_round_trips"]
    fold_keys = mech["synchronous_fold"]
    return {
        "per_decision_by_call_type": per, "per_decision_by_mechanism": by_mech, "per_decision_total": tot,
        "planner_analysis_tokens_per_decision": analysis_tok / n,
        "tasks_per_decision": n_tasks / n, "folds_per_decision": n_folds / n,
        "per_task": {u: sum(comp[k][u] for k in deleg_keys) / max(1, n_tasks) for u in ("rounds", "out", "in")},
        "per_fold": {u: sum(comp[k][u] for k in fold_keys) / max(1, n_folds) for u in ("rounds", "out", "in")},
    }


def _count(it):
    out = defaultdict(int)
    for v in it:
        out[v] += 1
    return out


# --- Showdown timer simulation -----------------------------------------------------------------
def simulate_timer(rows: list[dict], rules: dict) -> dict:
    """Replay the per-decision latencies under Showdown's battle timer (room-battle.ts):
    bank = starting + grace; +add_per_turn at each new turn (capped at starting); +min(add, 5)
    for a mid-turn request; per request at most min(bank, max_per_turn); time drains in 5 s
    ticks. -> first decision that would have forfeited, and the bank trace."""
    bank = rules["starting"] + rules["grace"]
    last_turn = None
    trace = []
    for r in rows:
        first = last_turn is None
        if not first:
            if r["turn"] is not None and last_turn is not None and r["turn"] > last_turn:
                bank = min(bank + rules["add_per_turn"], rules["starting"])
            else:
                bank += min(rules["add_per_turn"], TICK)
        limit = min(bank, rules["max_first_turn"] if first else rules["max_per_turn"])
        used = math.floor(r["latency_s"] / TICK) * TICK
        trace.append({"idx": r["idx"], "turn": r["turn"], "bank_before": bank, "limit": limit, "used": used})
        if used >= limit:
            return {"rules": rules["name"], "timeout_at_decision": r["idx"], "timeout_at_turn": r["turn"],
                    "limit_s": limit, "latency_s": round(r["latency_s"], 1), "trace": trace}
        bank -= used
        last_turn = r["turn"] if r["turn"] is not None else last_turn
    return {"rules": rules["name"], "timeout_at_decision": None, "timeout_at_turn": None, "trace": trace,
            "bank_end": bank}


# --- counterfactual estimates --------------------------------------------------------------------
def counterfactuals(rows: list[dict], model: Model) -> dict:
    """Per-decision latency under single changes, from the measured decomposition.
    Estimates: they assume the rest of the decision would have unfolded the same way."""
    out: dict = {}
    base = [r["latency_s"] for r in rows if r["kind"] not in ("recall", "single_option")]
    out["measured"] = base

    def adjusted(fn):
        vals = []
        for r in rows:
            if r["kind"] in ("recall", "single_option"):
                continue
            vals.append(max(0.0, fn(r)))
        return vals

    a = lambda r, pre: sum(v for k, v in r["attrib"].items() if k.startswith(pre))  # noqa: E731
    # 1) folds off the critical path (asynchronous consolidation)
    out["async_fold"] = adjusted(lambda r: r["latency_s"] - a(r, "fold_summary") - a(r, "memory"))
    # 2) executor writes no <results> block (rule-based synthesis from distilled tool outputs)
    def is_synth(c):
        return c["space"] == "executor" and c.get("exec_kind") == "results"

    def no_synth(r):
        return r["latency_s"] - sum(c["attr_prefill"] + c["attr_decode"] for c in r["calls"] if is_synth(c))
    out["rule_results"] = adjusted(no_synth)
    # 3) terse generations: every call emits at most N tokens (its decode share scaled down)
    def terse(cap):
        def f(r):
            saved = 0.0
            for c in r["calls"]:
                o = int(c.get("completion_tokens") or 0)
                if o > cap:
                    saved += c["attr_decode"] * (o - cap) / o
            return r["latency_s"] - saved
        return f
    out["cap_output_32"] = adjusted(terse(32))
    # 4) combination of 1-3
    def combo(r):
        lat = r["latency_s"] - a(r, "fold_summary") - a(r, "memory")
        for c in r["calls"]:
            if c["space"] in ("fold_summary", "memory"):
                continue
            o = int(c.get("completion_tokens") or 0)
            if is_synth(c):
                lat -= c["attr_prefill"] + c["attr_decode"]
            elif o > 32:
                lat -= c["attr_decode"] * (o - 32) / o
        return lat
    out["async_fold+rule_results+cap32"] = adjusted(combo)
    # 4b) executor uses a single tool round per task (all tool calls in one turn), then synthesises
    def one_round(r):
        lat = r["latency_s"]
        seen: dict = defaultdict(int)
        for c in sorted(r["calls"], key=lambda c: c["t_start"]):
            if c["space"] == "executor" and c.get("exec_kind") == "tool_turn":
                task = next((str(m.get("content") or "").split("\n", 1)[0] for m in (c.get("messages") or [])
                             if m.get("role") == "user"), "")
                seen[task] += 1
                if seen[task] > 1:
                    lat -= c["attr_prefill"] + c["attr_decode"]
        return lat
    out["executor_one_round"] = adjusted(one_round)
    # 4c) planner only: tool facts injected passively (ms of Python + a few hundred prompt tokens),
    #     no executor / isolated reasoning calls; folds kept
    def planner_only(r):
        keep = sum(c["attr_prefill"] + c["attr_decode"] for c in r["calls"]
                   if c["space"] in ("planner", "fold_summary", "memory"))
        return keep + r["attrib"].get("other", 0.0) + model.prefill(400) - model.a
    out["planner_only_passive_tools"] = adjusted(planner_only)
    # 5) floor: ONE call per decision with the planner's first prompt, answer-only output (~12 tokens)
    def floor(r):
        fp = r["first_planner"]
        if fp is None:
            return r["latency_s"]
        return fp["prefill_s"] + model.decode(12)
    out["single_call_floor"] = adjusted(floor)
    return out


# --- report ------------------------------------------------------------------------------------
def table(headers: list[str], rows: list[list]) -> str:
    def f(v):
        if isinstance(v, float):
            return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"
        return str(v)
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(f(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--agent", default="hymem")
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--mem", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out or run / "latency")
    out.mkdir(parents=True, exist_ok=True)

    decisions = load_jsonl(run / f"traj_{args.agent}.jsonl")
    records = load_jsonl(run / f"calls_{args.agent}.jsonl")
    bench = json.loads(Path(args.bench).read_text()) if args.bench and Path(args.bench).exists() else None
    model = Model(bench)
    calls = [r for r in records if not str(r.get("space", "")).startswith("event:")]
    for c in calls:
        c["t_start"] = c.get("t_start") or (c["ts"] - float(c.get("latency_s") or 0))
    tools = [r for r in records if r.get("space") == "event:tool"]
    folds = [r for r in records if r.get("space") == "event:fold"]
    ends = parse_server_log(Path(args.server_log) if args.server_log else None)
    enrich_calls(calls, ends, model)
    rows = decompose(decisions, calls, tools, folds)
    real = [r for r in rows if r["kind"] not in ("recall", "single_option")]
    lats = [r["latency_s"] for r in real]
    report: dict = {"run": str(run), "agent": args.agent, "n_decisions": len(rows), "n_llm_decisions": len(real)}

    # 1) decision latency distribution
    battle_wall = (max(r["t_end"] for r in rows) - min(r["t_start"] for r in rows)) if rows else 0
    turns = max((r["turn"] or 0) for r in rows) if rows else 0
    report["decision_latency"] = {
        "mean": mean(lats), "median": pct(lats, 0.5), "p90": pct(lats, 0.9), "max": max(lats) if lats else 0,
        "min": min(lats) if lats else 0, "total": sum(r["latency_s"] for r in rows),
        "share_le_10s": sum(x <= 10 for x in lats) / max(1, len(lats)),
        "share_le_30s": sum(x <= 30 for x in lats) / max(1, len(lats)),
        "share_le_60s": sum(x <= 60 for x in lats) / max(1, len(lats)),
        "share_le_150s": sum(x <= 150 for x in lats) / max(1, len(lats)),
        "battle_wall_s": battle_wall, "turns": turns,
        "per_turn_s": battle_wall / max(1, turns),
        "waiting_for_server_s": battle_wall - sum(r["latency_s"] for r in rows),
    }
    # 2) wall-clock attribution
    tot = defaultdict(float)
    for r in rows:
        for k, v in r["attrib"].items():
            tot[k] += v
    total_wall = sum(tot.values()) or 1.0
    report["attribution_s"] = dict(sorted(tot.items(), key=lambda kv: -kv[1]))
    by_space = defaultdict(float)
    by_phase = defaultdict(float)
    for k, v in tot.items():
        sp, _, ph = k.partition(":")
        by_space[sp] += v
        by_phase[ph or sp] += v
    report["by_space_s"] = dict(sorted(by_space.items(), key=lambda kv: -kv[1]))
    report["by_phase_s"] = dict(sorted(by_phase.items(), key=lambda kv: -kv[1]))
    # 3) per-call statistics by space
    per_space = {}
    for sp in sorted({c["space"] for c in calls}):
        cs = [c for c in calls if c["space"] == sp]
        ptok = sum(int(c.get("prompt_tokens") or 0) for c in cs)
        ctok = sum(int(c.get("cached_tokens") or 0) for c in cs)
        tps = [c["decode_tps"] for c in cs if c["decode_tps"]]
        per_space[sp] = {
            "calls": len(cs), "calls_per_llm_decision": len(cs) / max(1, len(real)),
            "prompt_tokens_mean": ptok / len(cs), "cached_share": ctok / max(1, ptok),
            "uncached_mean": mean(c["uncached"] for c in cs),
            "completion_tokens_mean": mean(int(c.get("completion_tokens") or 0) for c in cs),
            "latency_mean": mean(c["latency_s"] for c in cs),
            "prefill_s_mean": mean(c["prefill_s"] for c in cs), "decode_s_mean": mean(c["decode_s"] for c in cs),
            "decode_tps_median": pct(tps, 0.5) if tps else None,
            "prefill_from_server_log": sum(c["prefill_src"] == "server_log" for c in cs) / len(cs),
        }
    report["per_space"] = per_space
    # 4) structure
    report["structure"] = {
        "llm_calls_per_decision": mean(r["n_calls"] for r in real),
        "sequential_rounds_per_decision": mean(r["rounds"] for r in real),
        "max_rounds": max((r["rounds"] for r in real), default=0),
        "prompt_tokens_per_decision": mean(r["prompt_tokens"] for r in real),
        "uncached_tokens_per_decision": mean(r["uncached_tokens"] for r in real),
        "completion_tokens_per_decision": mean(r["completion_tokens"] for r in real),
        "decisions_with_fold": sum(bool(r["folds"]) for r in real),
        "fold_reasons": dict(_count(f for r in real for f in r["folds"])),
        "fold_s_mean": mean(r["fold_s"] for r in real if r["folds"]),
        "decision_kinds": dict(_count(r["kind"] for r in rows)),
        "tool_calls": len(tools),
        "tool_time_total_s": sum(float((t.get("extra") or {}).get("latency_s") or 0) for t in tools),
        "tool_time_max_s": max((float((t.get("extra") or {}).get("latency_s") or 0) for t in tools), default=0.0),
    }
    sess = executor_sessions(calls)
    if sess:
        report["executor_sessions"] = {
            "n": len(sess), "per_llm_decision": len(sess) / max(1, len(real)),
            "tool_turns_mean": mean(x["tool_turns"] for x in sess),
            "tool_turns_dist": dict(_count(x["tool_turns"] for x in sess)),
            "wall_s_mean": mean(x["wall_s"] for x in sess), "wall_s_max": max(x["wall_s"] for x in sess),
            "output_tokens_mean": mean(x["output_tokens"] for x in sess),
        }
    fr = defaultdict(lambda: defaultdict(int))
    for c in calls:
        fr[c["space"]][c.get("finish_reason") or "?"] += 1
    report["finish_reasons"] = {k: dict(v) for k, v in fr.items()}
    ex_tool = [c for c in calls if c.get("exec_kind") == "tool_turn"]
    ex_res = [c for c in calls if c.get("exec_kind") == "results"]

    def cached_share(cs):
        return sum(int(c.get("cached_tokens") or 0) for c in cs) / max(1, sum(int(c.get("prompt_tokens") or 0) for c in cs))
    report["executor_calls"] = {
        "tool_turns": len(ex_tool), "results_calls": len(ex_res),
        "tool_turn_output_tokens_mean": mean(int(c.get("completion_tokens") or 0) for c in ex_tool),
        "synthesis_output_tokens_mean": mean(int(c.get("completion_tokens") or 0) for c in ex_res),
        "tool_turn_cached_share": cached_share(ex_tool), "synthesis_cached_share": cached_share(ex_res),
        "tool_turn_latency_mean": mean(c["latency_s"] for c in ex_tool),
        "synthesis_latency_mean": mean(c["latency_s"] for c in ex_res),
        "tool_turns_truncated": sum(c.get("finish_reason") == "length" for c in ex_tool),
        "results_truncated": sum(c.get("finish_reason") == "length" for c in ex_res),
        "tool_calls_per_tool_turn": len(tools) / max(1, len(ex_tool)),
    }
    # 5) model check (sequential calls only: no overlap with another call)
    def overlaps(c):
        return any(o is not c and o["t_start"] < c["t_end"] - 0.05 and o["t_end"] > c["t_start"] + 0.05 for o in calls)
    seq = [c for c in calls if not overlaps(c)]
    if seq:
        errs = [c["latency_s"] - c["pred_s"] for c in seq]
        report["model_check"] = {"sequential_calls": len(seq), "mean_abs_err_s": mean(abs(e) for e in errs),
                                 "mean_err_s": mean(errs),
                                 "measured_mean_s": mean(c["latency_s"] for c in seq),
                                 "decode_tps_measured_median": pct([c["decode_tps"] for c in seq if c["decode_tps"]], 0.5),
                                 "bench_decode_tps": model.decode_tps, "bench_prefill_tps": 1 / model.b}
    par = [c for c in calls if overlaps(c)]
    if par:
        report["parallel_calls"] = {
            "n": len(par),
            "mean_queue_prefill_excess_s": mean(c["prefill_s"] - model.prefill(c["uncached"]) for c in par),
            "spaces": dict(_count(c["space"] for c in par)),
        }
    report["method_work"] = method_work(rows)
    # 6) timer
    report["timer"] = {"ladder": {k: v for k, v in simulate_timer(rows, LADDER).items() if k != "trace"},
                       "challenge": {k: v for k, v in simulate_timer(rows, CHALLENGE).items() if k != "trace"}}
    # 7) counterfactuals
    cf = counterfactuals(rows, model)
    report["counterfactual_mean_s"] = {k: mean(v) for k, v in cf.items()}
    report["counterfactual_median_s"] = {k: pct(v, 0.5) for k, v in cf.items()}
    # speed-up needed for targets with the current call structure
    m_lat = report["decision_latency"]["mean"]
    report["speedup_needed"] = {f"mean<={t}s": round(m_lat / t, 1) for t in (10, 30, 60)}
    # 8) memory pressure during the run
    mem = load_mem(Path(args.mem) if args.mem else None)
    if mem and rows:
        lo, hi = rows[0]["t_start"], rows[-1]["t_end"]
        inside = [m for m in mem if lo <= m["ts"] <= hi]
        if len(inside) >= 2:
            report["memory"] = {
                "samples": len(inside),
                "swapins_delta_pages": inside[-1]["swapins"] - inside[0]["swapins"],
                "swapouts_delta_pages": inside[-1]["swapouts"] - inside[0]["swapouts"],
                "free_pages_min": min(m["free_pages"] for m in inside),
                "mlx_rss_gb_min": min(m["mlx_rss_kb"] for m in inside) / 1024 / 1024,
                "mlx_rss_gb_max": max(m["mlx_rss_kb"] for m in inside) / 1024 / 1024,
                "swap_used_end": inside[-1]["swap_used"],
            }
    # per decision (compact) for the report
    report["decisions"] = [{
        "idx": r["idx"], "decision": r["decision"], "turn": r["turn"], "kind": r["kind"],
        "latency_s": round(r["latency_s"], 1),
        "n_calls": r["n_calls"], "rounds": r["rounds"], "folds": r["folds"],
        "completion_tokens": r["completion_tokens"], "uncached_tokens": r["uncached_tokens"],
        "attrib": {k: round(v, 1) for k, v in sorted(r["attrib"].items(), key=lambda kv: -kv[1])},
        "order": r["order"]} for r in rows]
    report["calls"] = [{
        "decision": c.get("decision"), "space": c["space"], "t_start": round(c["t_start"], 3),
        "prefill_end": round(c["prefill_end"], 3), "t_end": round(c["t_end"], 3),
        "prompt_tokens": c.get("prompt_tokens"), "cached_tokens": c.get("cached_tokens", 0),
        "uncached": c["uncached"], "completion_tokens": c.get("completion_tokens"),
        "finish_reason": c.get("finish_reason"), "prefill_src": c["prefill_src"],
        "exec_kind": c.get("exec_kind"),
        "attr_prefill": round(c.get("attr_prefill", 0.0), 3), "attr_decode": round(c.get("attr_decode", 0.0), 3),
    } for c in sorted(calls, key=lambda c: c["t_start"])]
    report["counterfactual_values"] = {k: [round(v, 2) for v in vals] for k, vals in cf.items()}
    (out / "latency_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    # --- markdown tables -------------------------------------------------------------------------
    dl = report["decision_latency"]
    md = [f"# Latency tables: {run.name} ({args.agent})", ""]
    md.append(table(["decisions (LLM)", "mean s", "median s", "p90 s", "max s", "<=10s", "<=30s", "<=60s", "<=150s",
                     "battle wall s", "turns", "s/turn"],
                    [[f"{len(rows)} ({len(real)})", dl["mean"], dl["median"], dl["p90"], dl["max"],
                      f"{dl['share_le_10s']:.0%}", f"{dl['share_le_30s']:.0%}", f"{dl['share_le_60s']:.0%}",
                      f"{dl['share_le_150s']:.0%}", dl["battle_wall_s"], dl["turns"], dl["per_turn_s"]]]))
    md += ["", "## Wall-clock attribution (all decisions)", ""]
    md.append(table(["component", "seconds", "share"],
                    [[k, v, f"{v / total_wall:.1%}"] for k, v in report["attribution_s"].items()]))
    md += ["", "## By context space", ""]
    md.append(table(["space", "seconds", "share"], [[k, v, f"{v / total_wall:.1%}"] for k, v in report["by_space_s"].items()]))
    md += ["", "## By phase", ""]
    md.append(table(["phase", "seconds", "share"], [[k, v, f"{v / total_wall:.1%}"] for k, v in report["by_phase_s"].items()]))
    md += ["", "## Per call, by space", ""]
    md.append(table(["space", "calls", "calls/decision", "prompt tok", "cached", "uncached tok", "output tok",
                     "latency s", "prefill s", "decode s", "decode tok/s"],
                    [[sp, v["calls"], v["calls_per_llm_decision"], v["prompt_tokens_mean"], f"{v['cached_share']:.0%}",
                      v["uncached_mean"], v["completion_tokens_mean"], v["latency_mean"], v["prefill_s_mean"],
                      v["decode_s_mean"], v["decode_tps_median"] or "-"] for sp, v in per_space.items()]))
    st = report["structure"]
    md += ["", "## Structure per LLM decision", ""]
    md.append(table(["LLM calls", "sequential rounds", "max rounds", "prompt tok", "uncached tok", "output tok",
                     "decisions with fold", "fold s (mean)", "tool calls", "tool time total s"],
                    [[st["llm_calls_per_decision"], st["sequential_rounds_per_decision"], st["max_rounds"],
                      st["prompt_tokens_per_decision"], st["uncached_tokens_per_decision"],
                      st["completion_tokens_per_decision"], st["decisions_with_fold"], st["fold_s_mean"],
                      st["tool_calls"], st["tool_time_total_s"]]]))
    if "executor_sessions" in report:
        es, ec = report["executor_sessions"], report["executor_calls"]
        md += ["", "## Executor sessions", ""]
        md.append(table(["sessions", "per decision", "tool turns (mean)", "tool-turn dist", "wall s (mean)",
                         "wall s (max)", "output tok/session", "tok per tool turn", "tok per synthesis",
                         "cached (tool turns)", "cached (synthesis)", "tools per turn"],
                        [[es["n"], es["per_llm_decision"], es["tool_turns_mean"], es["tool_turns_dist"],
                          es["wall_s_mean"], es["wall_s_max"], es["output_tokens_mean"],
                          ec["tool_turn_output_tokens_mean"], ec["synthesis_output_tokens_mean"],
                          f"{ec['tool_turn_cached_share']:.0%}", f"{ec['synthesis_cached_share']:.0%}",
                          ec["tool_calls_per_tool_turn"]]]))
    md += ["", "## Finish reasons by space", ""]
    md.append(table(["space", "finish reasons"], [[k, v] for k, v in report["finish_reasons"].items()]))
    mw = report["method_work"]
    md += ["", "## Hardware-agnostic critical-path work per decision (by mechanism)", ""]
    md.append(table(["mechanism", "sequential rounds", "generated tok (critical path)", "uncached input tok"],
                    [[k, v["rounds"], v["out"], v["in"]] for k, v in mw["per_decision_by_mechanism"].items()]
                    + [["total", mw["per_decision_total"]["rounds"], mw["per_decision_total"]["out"],
                        mw["per_decision_total"]["in"]]]))
    md += ["", f"Per delegated task: {mw['per_task']['rounds']:.2f} rounds, {mw['per_task']['out']:.0f} generated tok, "
               f"{mw['per_task']['in']:.0f} input tok; tasks/decision {mw['tasks_per_decision']:.2f}. "
               f"Per fold: {mw['per_fold']['rounds']:.2f} rounds, {mw['per_fold']['out']:.0f} generated tok, "
               f"{mw['per_fold']['in']:.0f} input tok; folds/decision {mw['folds_per_decision']:.2f}. "
               f"Planner <analysis>: {mw['planner_analysis_tokens_per_decision']:.0f} tok/decision."]
    md += ["", "## Showdown timer replay", ""]
    md.append(table(["rules", "timeout at decision", "turn", "limit s", "decision latency s"],
                    [[k, v.get("timeout_at_decision") or "none", v.get("timeout_at_turn") or "-",
                      v.get("limit_s", "-"), v.get("latency_s", "-")] for k, v in report["timer"].items()]))
    md += ["", "## Counterfactual mean / median decision latency (estimates)", ""]
    md.append(table(["scenario", "mean s", "median s"],
                    [[k, report["counterfactual_mean_s"][k], report["counterfactual_median_s"][k]] for k in cf]))
    if "model_check" in report:
        mc = report["model_check"]
        md += ["", f"Cost model check on {mc['sequential_calls']} non-overlapping calls: mean |error| "
                   f"{mc['mean_abs_err_s']:.2f} s (measured mean {mc['measured_mean_s']:.1f} s); measured decode "
                   f"{mc['decode_tps_measured_median']:.2f} tok/s vs benchmark {mc['bench_decode_tps']:.2f}."]
    md += ["", "## Per decision", ""]
    md.append(table(["#", "turn", "kind", "latency s", "calls", "rounds", "out tok", "uncached tok", "fold", "top components"],
                    [[d["idx"], d["turn"], d["kind"], d["latency_s"], d["n_calls"], d["rounds"], d["completion_tokens"],
                      d["uncached_tokens"], ",".join(d["folds"]) or "-",
                      "; ".join(f"{k} {v}" for k, v in list(d["attrib"].items())[:3])] for d in report["decisions"]]))
    (out / "latency_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # --- plots -------------------------------------------------------------------------------------
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        comps = [k for k, _ in sorted(tot.items(), key=lambda kv: -kv[1])]
        fig = make_subplots(rows=3, cols=1, subplot_titles=(
            "Decision latency by component (wall-clock, equal-share for overlapping calls)",
            "LLM call timeline (prefill = start to server prefill end, decode = rest)",
            "Decision latency CDF"), vertical_spacing=0.09, row_heights=[0.4, 0.35, 0.25])
        x = [f"#{r['idx']} T{r['turn']}" for r in rows]
        for k in comps:
            fig.add_trace(go.Bar(name=k, x=x, y=[r["attrib"].get(k, 0) for r in rows]), row=1, col=1)
        for yv, name in ((10, "10 s (sustainable ladder pace)"), (150, "150 s (ladder per-turn cap)")):
            fig.add_hline(y=yv, line_dash="dot", annotation_text=name, row=1, col=1)
        t_base = min((c["t_start"] for c in calls), default=0)
        spaces = sorted({c["space"] for c in calls})
        for ph, color in (("prefill", "#4C78A8"), ("decode", "#F58518")):
            fig.add_trace(go.Bar(
                name=f"timeline {ph}", orientation="h", showlegend=True,
                y=[c["space"] for c in calls],
                x=[(c["prefill_s"] if ph == "prefill" else c["decode_s"]) for c in calls],
                base=[((c["t_start"] if ph == "prefill" else c["prefill_end"]) - t_base) for c in calls],
                marker_color=color, hovertext=[f"d{c.get('decision')} {c['space']} in={c.get('prompt_tokens')} "
                                               f"cached={c.get('cached_tokens', 0)} out={c.get('completion_tokens')}"
                                               for c in calls]), row=2, col=1)
        s = sorted(lats)
        fig.add_trace(go.Scatter(x=s, y=[(i + 1) / len(s) for i in range(len(s))], mode="lines+markers",
                                 name="decision latency CDF"), row=3, col=1)
        fig.update_layout(barmode="stack", height=1500, title=f"HyMem real-time analysis: {run.name}")
        fig.update_yaxes(title_text="seconds", row=1, col=1)
        fig.update_yaxes(categoryorder="array", categoryarray=spaces, row=2, col=1)
        fig.update_xaxes(title_text="seconds since first call", row=2, col=1)
        fig.update_xaxes(title_text="decision latency (s)", row=3, col=1)
        fig.write_html(str(out / "latency.html"))
    except ImportError:
        pass
    print("\n".join(md[:40]))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
