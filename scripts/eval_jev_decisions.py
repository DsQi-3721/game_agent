"""Offline evaluation of System-1 (Jev) battle decisions on recorded positions.

For every snapshot (pickled ``Battle``) it sends the same single request the agents send
(``best`` + ``best_rev`` + ``hard``) and measures

  feasibility   latency (mean / p50 / p90 / max), input tokens and cost per decision
  reliability   order consistency (does reversing the options change the answer?),
                first-position bias
  quality       objective blunders computed by code -- picking a move the target is immune
                to while a damaging move exists; missing a KO that lands before the opponent
                moves -- plus agreement with poke-env's SimpleHeuristics and max-damage
  routing       coverage / blunder rate of the confidence gate, and a risk-coverage sweep

    python scripts/eval_jev_decisions.py --snapshots data/jev_eval/snapshots --models jev-1.13.0,jev-1.14.0
    python scripts/eval_jev_decisions.py --mock max_damage     # pipeline check without an API key
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import random
import statistics
import sys
import time
from pathlib import Path

from arena.agents.system1.decider import FastDecider, RouteConfig, obvious_choice
from arena.agents.system1.facts import build_option_facts, reference_choices
from arena.llm import typesafe as ts
from arena.state.views import legal_options

PRICE_PER_TOKEN = 0.042e-6


def first_strike_ko(f) -> bool:
    order = f.fields.get("order", "")
    return f.ko_now and ("move first" in order or "priority" in order) and "guaranteed" in f.fields.get("damage", "")


def position_meta(battle, facts) -> dict:
    refs = reference_choices(battle, facts)
    damaging = [f for f in facts if f.exp_damage is not None and f.exp_damage > 0 and not f.no_effect]
    return {
        "n_options": len(facts),
        "keys": [f.key for f in facts],
        "immune": [f.key for f in facts if f.no_effect],
        "has_damaging": bool(damaging),
        "first_strike_ko": [f.key for f in facts if first_strike_ko(f)],
        "ko_now": [f.key for f in facts if f.ko_now],
        "force_switch": bool(getattr(battle, "force_switch", False)),
        "tier0": (lambda o: o.key if o else None)(obvious_choice(battle, facts)),
        "refs": refs,
    }


def blunders(key: str, meta: dict) -> dict:
    immune = key in meta["immune"] and meta["has_damaging"]
    missed_ko = bool(meta["first_strike_ko"]) and key not in meta["ko_now"]
    return {"immune": immune, "missed_ko": missed_ko, "any": immune or missed_ko}


def mock_responder(policy: str, seed: int = 0):
    rng = random.Random(seed)

    def reply(state, questions):
        out = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                if policy == "max_damage":
                    def dmg(k):
                        d = q["criteria"][k].get("damage", "") if isinstance(q["criteria"][k], dict) else ""
                        try:
                            return float(d.split("estimated ")[1].split("-")[0])
                        except (IndexError, ValueError):
                            return -1.0
                    pick = max(keys, key=dmg)
                else:
                    pick = rng.choice(keys)
                out[qid] = ts.uniform_choice(q["criteria"], pick, rng.uniform(0.3, 0.95))
            else:
                out[qid] = {"type": "noul", "noul": rng.uniform(0, 1)}
        return out
    return reply


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def rate(rows, pred):
    return round(sum(1 for r in rows if pred(r)) / len(rows), 4) if rows else None


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    lat = [r["latency_s"] for r in ok]
    acc = [r for r in ok if r["accept"]]
    rej = [r for r in ok if not r["accept"]]
    s = {
        "n": len(rows), "errors": len(rows) - len(ok),
        "latency_s": {"mean": round(statistics.mean(lat), 4) if lat else None, "p50": pct(lat, 0.5),
                      "p90": pct(lat, 0.9), "max": max(lat) if lat else None},
        "input_tokens_mean": round(statistics.mean(r["input_tokens"] for r in ok), 1) if ok else None,
        "cost_usd_per_decision": (round(statistics.mean(r["input_tokens"] for r in ok) * PRICE_PER_TOKEN, 6)
                                  if ok else None),
        "order_consistency": rate(ok, lambda r: r["consistent"]),
        "fwd_picks_first_option": rate(ok, lambda r: r["fwd_top"] == (r.get("branch_keys") or r["meta"]["keys"])[0]),
        "rev_picks_first_option": rate(ok, lambda r: r["rev_top"] == (r.get("branch_keys") or r["meta"]["keys"])[-1]),
        "switch_rate_unforced": rate([r for r in ok if not r["meta"]["force_switch"]],
                                     lambda r: r["choice"].startswith("switch_")),
        "heuristics_switch_rate_unforced": rate([r for r in ok if not r["meta"]["force_switch"]],
                                                lambda r: (r["meta"]["refs"].get("simple_heuristics") or "").startswith("switch_")),
        "tera_rate": rate(ok, lambda r: r["choice"].endswith("_tera")),
        "agree_heuristics": rate([r for r in ok if r["meta"]["refs"].get("simple_heuristics")],
                                 lambda r: r["choice"] == r["meta"]["refs"]["simple_heuristics"]),
        "agree_max_damage": rate([r for r in ok if r["meta"]["refs"].get("max_damage")],
                                 lambda r: r["choice"] == r["meta"]["refs"]["max_damage"]),
        "chance_agreement": round(statistics.mean(1 / r["meta"]["n_options"] for r in ok), 4) if ok else None,
        "blunder_rate": rate(ok, lambda r: r["blunder"]["any"]),
        "blunder_immune": rate(ok, lambda r: r["blunder"]["immune"]),
        "blunder_missed_ko": rate(ok, lambda r: r["blunder"]["missed_ko"]),
        "positions_with_first_strike_ko": rate(ok, lambda r: bool(r["meta"]["first_strike_ko"])),
        "tier0_positions": rate(ok, lambda r: bool(r["meta"].get("tier0"))),
        "agree_tier0_on_obvious": rate([r for r in ok if r["meta"].get("tier0")],
                                       lambda r: r["choice"] in r["meta"]["first_strike_ko"]),
        "gate": {"coverage": rate(ok, lambda r: r["accept"]),
                 "blunder_rate_accepted": rate(acc, lambda r: r["blunder"]["any"]),
                 "blunder_rate_escalated": rate(rej, lambda r: r["blunder"]["any"]),
                 "agree_heuristics_accepted": rate([r for r in acc if r["meta"]["refs"].get("simple_heuristics")],
                                                   lambda r: r["choice"] == r["meta"]["refs"]["simple_heuristics"]),
                 "agree_heuristics_escalated": rate([r for r in rej if r["meta"]["refs"].get("simple_heuristics")],
                                                    lambda r: r["choice"] == r["meta"]["refs"]["simple_heuristics"]),
                 "reasons": {}},
        "p_hard_mean": round(statistics.mean(r["p_hard"] for r in ok if r["p_hard"] is not None), 4)
        if any(r["p_hard"] is not None for r in ok) else None,
    }
    for r in rej:
        s["gate"]["reasons"][r["reason"]] = s["gate"]["reasons"].get(r["reason"], 0) + 1
    # reference policies on the same positions
    for ref in ("simple_heuristics", "max_damage"):
        with_ref = [r for r in ok if r["meta"]["refs"].get(ref)]
        s[f"{ref}_blunder_rate"] = rate(with_ref, lambda r, ref=ref: blunders(r["meta"]["refs"][ref], r["meta"])["any"])
    # risk-coverage: accept when consistent and confidence >= t
    sweep = []
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        a = [r for r in ok if r["consistent"] and r["confidence"] >= t]
        sweep.append({"min_confidence": t, "coverage": rate(ok, lambda r, t=t: r["consistent"] and r["confidence"] >= t),
                      "blunder_rate": rate(a, lambda r: r["blunder"]["any"]),
                      "agree_heuristics": rate([r for r in a if r["meta"]["refs"].get("simple_heuristics")],
                                               lambda r: r["choice"] == r["meta"]["refs"]["simple_heuristics"])})
    s["risk_coverage"] = sweep
    return s


async def run_model(model: str, snaps: list[Path], client, concurrency: int, cfg_kw: dict) -> list[dict]:
    """One request per snapshot, exactly as the agents send it."""
    from arena.s1.context import s1_meta
    decider = FastDecider(client, RouteConfig(model=model, **cfg_kw))
    sem = asyncio.Semaphore(concurrency)

    async def one(path: Path) -> dict:
        battle = pickle.loads(path.read_bytes())
        options = legal_options(battle)
        facts = build_option_facts(battle, options)
        meta = position_meta(battle, facts)
        async with sem:
            t0 = time.perf_counter()
            try:
                with s1_meta(agent="offline_snapshots", purpose="action", battle_tag=battle.battle_tag,
                             decision_id=path.stem, turn=battle.turn, tier="teacher"):
                    fd = await decider.decide(battle, options)
            except Exception as e:  # noqa: BLE001
                return {"snapshot": path.name, "model": model, "error": f"{type(e).__name__}: {e}"[:300],
                        "meta": meta}
            wall = time.perf_counter() - t0
        fwd_top = max(fd.p_fwd, key=fd.p_fwd.get)
        rev_top = max(fd.p_rev, key=fd.p_rev.get)
        return {"snapshot": path.name, "model": model, "served_by": fd.model, "turn": battle.turn,
                "mode": decider.cfg.mode, "branch": fd.branch, "p_stay": fd.p_stay,
                "branch_confidence": round(fd.branch_confidence, 4), "choice_confidence": round(fd.choice_confidence, 4),
                "branch_keys": fd.branch_keys,
                "choice": fd.key, "confidence": round(fd.confidence, 4), "margin": round(fd.margin, 4),
                "consistent": fd.consistent, "fwd_top": fwd_top, "rev_top": rev_top,
                "p_hard": fd.p_hard, "accept": fd.accept, "reason": fd.reason_code,
                "latency_s": round(fd.latency_s, 4), "wall_s": round(wall, 4), "input_tokens": fd.input_tokens,
                "top": fd.ranked(4), "meta": meta, "blunder": blunders(fd.key, meta)}

    return await asyncio.gather(*(one(p) for p in snaps))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default="data/jev_eval/snapshots")
    ap.add_argument("--models", default="jev-1.13.0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--out", default="runs/jev_eval/decisions")
    ap.add_argument("--mock", default="", choices=["", "max_damage", "random"],
                    help="offline pipeline check with a mock Jev")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="split", choices=["split", "flat"], help="question layout (decider.py)")
    ap.add_argument("--record", default="", help="also write every request to this experience log (s1exp format)")
    args = ap.parse_args()

    snaps = sorted(Path(args.snapshots).glob("*.pkl"))
    if args.limit:
        snaps = random.Random(args.seed).sample(snaps, min(args.limit, len(snaps)))
    if not snaps:
        sys.exit(f"no snapshots in {args.snapshots} (run scripts/collect_decision_snapshots.py first)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = ts.MockTypeSafeClient(mock_responder(args.mock, args.seed)) if args.mock else ts.TypeSafeClient()
    if not args.mock and not client.configured:
        sys.exit("TYPESAFE_API_KEY is not set: add it to the project .env (or use --mock)")
    if args.record:
        from arena.s1.recorder import ExperienceRecorder, RecordingClient
        client = RecordingClient(client, ExperienceRecorder(args.record),
                                 backend="mock" if args.mock else "typesafe", role="teacher")

    report = {"snapshots": len(snaps), "mock": args.mock or None, "models": {}}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        rows = await run_model(model, snaps, client, args.concurrency, {"mode": args.mode})
        tag = f"mock_{args.mock}" if args.mock else model
        with open(out / f"rows_{tag}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["models"][tag] = summarise(rows)
        s = report["models"][tag]
        print(f"[{tag}] n={s['n']} err={s['errors']} latency mean={s['latency_s']['mean']} p90={s['latency_s']['p90']} "
              f"consistency={s['order_consistency']} blunders={s['blunder_rate']} "
              f"(heur {s['simple_heuristics_blunder_rate']}, maxdmg {s['max_damage_blunder_rate']}) "
              f"agree_heur={s['agree_heuristics']} coverage={s['gate']['coverage']} "
              f"blunder@accepted={s['gate']['blunder_rate_accepted']}")
    (out / ("summary_mock.json" if args.mock else "summary.json")).write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    if hasattr(client, "aclose"):
        await client.aclose()
    print(f"-> {out}")


if __name__ == "__main__":
    asyncio.run(main())
