"""Evaluate the Jev memory router on memory items from a real HyMem battle.

One request per fold (exactly what the agent sends): every memory item HyMem's contexts held
right after that fold, plus control items with known answers (see build_memory_bench.py).

  feasibility  latency and input tokens per routing request
  controls     contradicted / wrong-battle facts rejected, true facts accepted (p_valid AUROC),
               stale numbers / generic advice dropped, tool tips sent to the executor only
  gold         kind accuracy and keep/drop agreement on hand-labelled items
               (data/jev_eval/memory_gold.json, labelled at the item's first appearance)
  effect       share of HyMem's memory items dropped, planner / executor memory size
               (hard-coded view vs routed view)

    python scripts/eval_jev_memory_router.py --bench data/jev_eval/memory_bench.jsonl
    python scripts/eval_jev_memory_router.py --mock          # pipeline check without an API key
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from arena.agents.system1.memory_router import (
    KINDS,
    MemoryItem,
    MemoryRouter,
    RoutedItem,
    RoutedMemory,
    RouterConfig,
)
from arena.llm import typesafe as ts

PRICE_PER_TOKEN = 0.042e-6


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", text.lower()).strip()


def auroc(pos: list[float], neg: list[float]):
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def mock_reply(state, questions):
    """Keyword heuristics standing in for Jev (pipeline check only)."""
    out = {}
    for qid, q in questions.items():
        note = q["instructions"]["note"].lower() if isinstance(q["instructions"], dict) else ""
        if q["type"] == "choice":
            kind = ("tool_tip" if any(t in note for t in ("damage_calc", "switch_eval", "pokedex", "type_matchup"))
                    else "calc_result" if re.search(r"\d+-\d+%", note)
                    else "current_numbers" if "% hp" in note
                    else "plan" if note.startswith(("use ", "switch ", "ko ", "chip "))
                    else "opponent_set" if note.startswith("opp ") else "generic" if "always" in note
                    else "battle_event")
            out[qid] = ts.uniform_choice(q["criteria"], kind, 0.7)
        else:
            out[qid] = {"type": "noul", "noul": 0.2 if "fainted" in note and qid.startswith("valid_") else 0.7}
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/jev_eval/memory_bench.jsonl")
    ap.add_argument("--gold", default="data/jev_eval/memory_gold.json")
    ap.add_argument("--model", default="jev-1.13.0")
    ap.add_argument("--out", default="runs/jev_eval/memory")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--rescore", default="", help="recompute summary_<tag> from saved rows (no API calls)")
    ap.add_argument("--min-valid", type=float, default=None)
    ap.add_argument("--min-useful", type=float, default=None)
    ap.add_argument("--suffix", action="store_true", help="with --rescore: write to a threshold-suffixed file")
    args = ap.parse_args()
    if args.rescore:
        rescore(args)
        return

    folds = [json.loads(x) for x in open(args.bench, encoding="utf-8")]
    gold = {g["norm"]: g for g in json.load(open(args.gold, encoding="utf-8"))} if Path(args.gold).exists() else {}
    client = ts.MockTypeSafeClient(mock_reply) if args.mock else ts.TypeSafeClient()
    if not args.mock and not client.configured:
        sys.exit("TYPESAFE_API_KEY is not set: add it to the project .env (or use --mock)")
    # every fold is a complete, independent request (worst case: no cached kinds)
    router = MemoryRouter(client, RouterConfig(model=args.model, cache_kinds=False))
    sem = asyncio.Semaphore(args.concurrency)

    async def one(fo: dict):
        items = [MemoryItem(i["id"], i["text"], i["source"], i["turn"], i["trusted"], i["kind_hint"])
                 for i in fo["items"]]
        ctrl = [MemoryItem(f"c{k}", c["text"], f"control.{c['control']}") for k, c in enumerate(fo["controls"])]
        async with sem:
            routed: RoutedMemory = await router.route(fo["state"], items + ctrl)
        return fo, routed, len(router.questions(items + ctrl))

    results = await asyncio.gather(*(one(fo) for fo in folds))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "mock" if args.mock else args.model
    rows, seen_gold, requests = [], set(), []
    for fo, routed, n_q in results:
        requests.append({"fold": fo["fold"], "latency_s": routed.latency_s, "input_tokens": routed.input_tokens,
                         "questions": n_q, "rule_planner_chars": fo["rule_planner_chars"],
                         "rule_executor_chars": fo["rule_executor_chars"]})
        ctrl_meta = {f"control.{c['control']}": c for c in fo["controls"]}
        for r in routed.items:
            row = {"fold": fo["fold"], "turn": fo["turn"], "id": r.item.id, "text": r.item.text,
                   "source": r.item.source, "trusted": r.item.trusted, "kind_hint": r.item.kind_hint,
                   "item_turn": r.item.turn, "kind": r.kind, "p_kind": round(r.p_kind, 4),
                   "p_valid": r.p_valid, "p_useful": r.p_useful}
            if r.item.source.startswith("control."):
                row["control"] = ctrl_meta.get(r.item.source, {})
            g = gold.get(norm(r.item.text))
            if g is not None and g["norm"] not in seen_gold and not r.item.trusted:
                seen_gold.add(g["norm"])
                row["gold_kind"], row["gold_keep"] = g["gold_kind"], g["gold_keep"]
            rows.append(row)
    with open(out / f"rows_{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / f"requests_{tag}.json").write_text(json.dumps(requests, indent=1))
    summary = score(rows, requests, RouterConfig(model=args.model), tag)
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "sizes"}, indent=1, ensure_ascii=False)[:4000])
    if hasattr(client, "aclose"):
        await client.aclose()


def score(rows: list[dict], requests: list[dict], cfg: RouterConfig, tag: str) -> dict:
    """Place every judged item with ``cfg``'s thresholds and compute all metrics (no API calls)."""
    router = MemoryRouter(None, cfg)
    by_fold: dict = {}
    for r in rows:
        item = MemoryItem(r["id"], r["text"], r["source"], r.get("item_turn"), r["trusted"], r.get("kind_hint", ""))
        dest, dropped = router.place(item, r["kind"], r["p_valid"], r["p_useful"])
        r["dest"], r["dropped"] = list(dest), dropped
        by_fold.setdefault(r["fold"], []).append(RoutedItem(item, r["kind"], r["p_kind"], r["p_valid"], r["p_useful"],
                                                            dest, dropped))
    sizes = []
    for q in requests:
        routed = RoutedMemory([x for x in by_fold.get(q["fold"], []) if not x.item.source.startswith("control.")],
                              cfg=cfg)
        sizes.append({"fold": q["fold"], "rule_planner_chars": q["rule_planner_chars"],
                      "routed_planner_chars": len(routed.render("planner")),
                      "rule_executor_chars": q["rule_executor_chars"],
                      "routed_executor_chars": len(routed.render("executor"))})

    def ctrl(name):
        return [r for r in rows if r["source"] == f"control.{name}"]

    def frac(xs, pred):
        return round(sum(1 for x in xs if pred(x)) / len(xs), 4) if xs else None

    true_v = [r["p_valid"] for r in ctrl("true_fact") if r["p_valid"] is not None]
    false_v = [r["p_valid"] for r in ctrl("contradicted") + ctrl("wrong_battle") if r["p_valid"] is not None]
    real = [r for r in rows if not r["source"].startswith("control.")]
    gold_rows = [r for r in rows if "gold_kind" in r]
    per_kind = {}
    for k in KINDS:
        tp = sum(1 for r in gold_rows if r["kind"] == k and r["gold_kind"] == k)
        fp = sum(1 for r in gold_rows if r["kind"] == k and r["gold_kind"] != k)
        fn = sum(1 for r in gold_rows if r["kind"] != k and r["gold_kind"] == k)
        if tp + fn:
            p = tp / (tp + fp) if tp + fp else 0.0
            rcl = tp / (tp + fn)
            per_kind[k] = {"n": tp + fn, "precision": round(p, 3), "recall": round(rcl, 3),
                           "f1": round(2 * p * rcl / (p + rcl), 3) if p + rcl else 0.0}
    kept_pred = [not r["dropped"] for r in gold_rows]
    kept_gold = [bool(r["gold_keep"]) for r in gold_rows]
    tp = sum(a and b for a, b in zip(kept_pred, kept_gold))
    prec = tp / sum(kept_pred) if sum(kept_pred) else None
    rec = tp / sum(kept_gold) if sum(kept_gold) else None
    lat = [q["latency_s"] for q in requests]
    toks = [q["input_tokens"] for q in requests]
    return {
        "model": tag, "folds": len(requests), "min_valid": cfg.min_valid, "min_useful": cfg.min_useful,
        "requests": {"latency_s_mean": round(statistics.mean(lat), 4), "latency_s_max": round(max(lat), 4),
                     "questions_mean": round(statistics.mean(q["questions"] for q in requests), 1),
                     "input_tokens_mean": round(statistics.mean(toks)),
                     "cost_usd_per_request": round(statistics.mean(toks) * PRICE_PER_TOKEN, 6)},
        "controls": {
            "p_valid_auroc_true_vs_false": auroc(true_v, false_v),
            "true_fact_accepted": frac(ctrl("true_fact"), lambda r: not r["dropped"]),
            "contradicted_rejected": frac(ctrl("contradicted"), lambda r: bool(r["dropped"])),
            "wrong_battle_rejected": frac(ctrl("wrong_battle"), lambda r: bool(r["dropped"])),
            "stale_numbers_dropped": frac(ctrl("stale_numbers"), lambda r: bool(r["dropped"])),
            "generic_dropped": frac(ctrl("generic"), lambda r: bool(r["dropped"])),
            "tool_tip_to_executor_only": frac(ctrl("tool_tip"), lambda r: r["dest"] == ["executor"]),
        },
        "gold": {"n": len(gold_rows),
                 "kind_accuracy": frac(gold_rows, lambda r: r["kind"] == r["gold_kind"]),
                 "kind_macro_f1": round(statistics.mean(v["f1"] for v in per_kind.values()), 3) if per_kind else None,
                 "per_kind": per_kind,
                 "keep_accuracy": frac(gold_rows, lambda r: (not r["dropped"]) == bool(r["gold_keep"])),
                 "keep_precision": round(prec, 3) if prec is not None else None,
                 "keep_recall": round(rec, 3) if rec is not None else None,
                 "hardcoded_keep_precision": round(sum(kept_gold) / len(kept_gold), 3) if kept_gold else None,
                 "confusion": Counter(f"{r['gold_kind']}->{r['kind']}" for r in gold_rows
                                      if r["kind"] != r["gold_kind"]).most_common(12)},
        "effect": {
            "real_items": len(real),
            "real_dropped": frac(real, lambda r: bool(r["dropped"])),
            "dropped_by_reason": dict(Counter(r["dropped"] for r in real if r["dropped"])),
            "kind_by_source": {src: dict(Counter(r["kind"] for r in real if r["source"].split(".")[0] == src))
                               for src in sorted({r["source"].split(".")[0] for r in real})},
            "planner_chars_rule_mean": round(statistics.mean(s["rule_planner_chars"] for s in sizes)),
            "planner_chars_routed_mean": round(statistics.mean(s["routed_planner_chars"] for s in sizes)),
            "executor_chars_rule_mean": round(statistics.mean(s["rule_executor_chars"] for s in sizes)),
            "executor_chars_routed_mean": round(statistics.mean(s["routed_executor_chars"] for s in sizes)),
        },
        "sizes": sizes,
    }


def rescore(args) -> None:
    """Recompute a saved run's summary with other thresholds (no API calls)."""
    out = Path(args.out)
    rows = [json.loads(x) for x in open(out / f"rows_{args.rescore}.jsonl", encoding="utf-8")]
    requests = json.loads((out / f"requests_{args.rescore}.json").read_text())
    cfg = RouterConfig(model=args.model)
    if args.min_valid is not None:
        cfg.min_valid = args.min_valid
    if args.min_useful is not None:
        cfg.min_useful = args.min_useful
    summary = score(rows, requests, cfg, args.rescore)
    name = f"summary_{args.rescore}" + (f"_v{cfg.min_valid}_u{cfg.min_useful}" if args.suffix else "")
    (out / f"{name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("min_valid", "min_useful", "controls", "effect")}, indent=1)[:2500])
    print({k: summary["gold"][k] for k in ("kind_accuracy", "keep_accuracy", "keep_precision", "keep_recall")})


if __name__ == "__main__":
    asyncio.run(main())
