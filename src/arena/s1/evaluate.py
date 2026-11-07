"""Score any System-One backend against held-out teacher labels.

Questions are sent grouped by their original request (same state), exactly as the agents send
them, so latency is realistic and the local scorer's prefix cache applies. Metrics, overall and
per purpose (action / memory / controller ...) and per question group (stay, best_move, kind ...):

  agree   top-1 agreement with the target's argmax
  kl      KL(target || prediction), nats
  brier   multi-class Brier score
  ece     expected calibration error of the prediction's top probability vs agreement (10 bins).
          Note: a student that reproduces a SOFT teacher exactly is "under-confident" by this measure
          (teacher top-1 of 0.6, yet its argmax is matched ~90% of the time), so ECE rises as KL falls;
  conf_gap  mean |student top-1 probability - teacher top-1 probability|: does the student carry the
          teacher's confidence (what the gate reads)?
  latency seconds per request

``decision`` replays every action request (stay / best_move / best_switch, both option orders)
through the live System-1 combination and gate (``FastDecider.combine``), once with the target
answers and once with the predictions: decision agreement, how often the gate would let the
model act alone (``accept``), how often it is right when it does (``accepted_agree``) and why it
escalates -- an offline preview of the cascade.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import Optional

from arena.llm.typesafe import Answer
from arena.s1.prompts import target_distribution
from arena.s1.recorder import answer_dict


def _qgroup(qid: str) -> str:
    base = qid[:-4] if qid.endswith("_rev") else qid
    return base.split("_m")[0] if base.startswith(("kind_", "valid_", "useful_")) else base


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> Optional[float]:
    if not pairs:
        return None
    tot, err = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        g = [(c, a) for c, a in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if g:
            err += len(g) / tot * abs(statistics.mean(c for c, _ in g) - statistics.mean(float(a) for _, a in g))
    return round(err, 4)


def _metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {"n": len(rows), "agree": round(statistics.mean(r["agree"] for r in rows), 4),
            "kl": round(statistics.mean(r["kl"] for r in rows), 4),
            "brier": round(statistics.mean(r["brier"] for r in rows), 4),
            "ece": ece([(r["conf"], r["agree"]) for r in rows]),
            "conf_gap": round(statistics.mean(abs(r["conf"] - r["target_conf"]) for r in rows), 4)}


def _as_answer(question: dict, keys: list[str], dist: list[float]):
    if question.get("type") == "noul":
        return SimpleNamespace(probabilities=None, noul=dist[keys.index("true")] if "true" in keys else None)
    return SimpleNamespace(probabilities=dict(zip(keys, dist)), noul=None)


def gate_config(**overrides):
    """The cascade's student gate (S1Config student_* defaults), P(hard) off as recommended."""
    from arena.agents.system1.decider import RouteConfig
    return RouteConfig(**{"min_confidence": 0.5, "min_margin": 0.15, "max_hard": 1.0, "min_branch_confidence": 0.3,
                          **overrides})


def replay_decision(group: list[dict], dists: dict, cfg) -> Optional[dict]:
    """One action request's answers (qid -> distribution over that question's keys) -> the live
    combine + gate result, or None when the request has no decision questions."""
    from arena.agents.system1.decider import FastDecider

    by_q = {ex["qid"]: ex for ex in group if ex["qid"] in dists}

    def keys_of(base: str) -> list[str]:
        if base in by_q:
            return list(by_q[base]["keys"])
        return list(reversed(by_q[base + "_rev"]["keys"])) if base + "_rev" in by_q else []

    flat = "best" in by_q or "best_rev" in by_q
    mkeys, skeys, fkeys = keys_of("best_move"), keys_of("best_switch"), keys_of("best")
    if not (fkeys if flat else (mkeys or skeys)):
        return None
    answers = {q: _as_answer(ex["question"], ex["keys"], dists[q]) for q, ex in by_q.items()}
    decider = FastDecider(None, cfg.__class__(**{**cfg.__dict__, "mode": "flat" if flat else "split"}))
    return decider.combine(answers, keys=fkeys, move_keys=mkeys, switch_keys=skeys)


def _decision_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    acc = [r for r in rows if r["accept"]]
    return {"n": len(rows), "agree": round(statistics.mean(r["agree"] for r in rows), 4),
            "accept": round(len(acc) / len(rows), 4),
            "accepted_agree": round(statistics.mean(r["agree"] for r in acc), 4) if acc else None,
            "target_accept": round(statistics.mean(r["target_accept"] for r in rows), 4),
            "consistent": round(statistics.mean(r["consistent"] for r in rows), 4),
            "branch_agree": round(statistics.mean(r["branch_agree"] for r in rows), 4),
            "reasons": dict(Counter(r["reason"] for r in rows).most_common()),
            **({"s2_n": len(s2), "s2_agree": round(statistics.mean(s2), 4)} if (s2 := [r["s2_agree"] for r in rows
                                                                                       if r["s2_agree"] is not None]) else {}),
            **({"played_n": len(pl), "played_agree": round(statistics.mean(pl), 4),
                "target_played_agree": round(statistics.mean(r["target_played_agree"] for r in rows
                                                             if r["target_played_agree"] is not None), 4)}
               if (pl := [r["played_agree"] for r in rows if r["played_agree"] is not None]) else {})}


async def evaluate(client, examples: list[dict], *, limit_requests: Optional[int] = None,
                   model: Optional[str] = None, gate=None, keep_predictions: bool = False) -> dict:
    """``keep_predictions``: also return every request's predicted distributions (``predictions``),
    e.g. to sweep gate thresholds offline (``scripts/s1_gate_sweep.py``)."""
    by_req: dict = defaultdict(list)
    for ex in examples:
        by_req[(ex["group"], ex["request"])].append(ex)
    reqs = list(by_req.values())[:limit_requests] if limit_requests else list(by_req.values())
    rows, latencies, errors, first_error = [], [], 0, ""
    decisions: list[dict] = []
    kept: list[dict] = []
    gate = gate or gate_config()
    for group in reqs:
        questions = {ex["qid"]: ex["question"] for ex in group}
        t0 = time.perf_counter()
        try:
            res = await client.system_one(group[0]["state"], questions, model=model)
        except Exception as e:  # noqa: BLE001
            errors += 1
            first_error = first_error or f"{type(e).__name__}: {str(e)[:300]}"
            continue
        latencies.append(res.latency_s or time.perf_counter() - t0)
        preds: dict = {}
        for ex in group:
            a: Optional[Answer] = res.answers.get(ex["qid"])
            if a is None:
                errors += 1
                continue
            pred = target_distribution(ex["question"], answer_dict(a), ex["keys"])
            preds[ex["qid"]] = pred
            tgt = ex["target"]
            kl = sum(t * (math.log(max(t, 1e-9)) - math.log(max(p, 1e-9))) for t, p in zip(tgt, pred) if t > 0)
            brier = sum((p - t) ** 2 for p, t in zip(pred, tgt))
            top = max(range(len(pred)), key=lambda j: pred[j])
            rows.append({"purpose": ex["purpose"], "qgroup": _qgroup(ex["qid"]),
                         "agree": top == max(range(len(tgt)), key=lambda j: tgt[j]), "kl": kl, "brier": brier,
                         "conf": pred[top], "target_conf": max(tgt)})
        if keep_predictions:
            kept.append({"group": group[0]["group"], "request": group[0]["request"], "preds": preds})
        if group[0].get("purpose", "action") == "action":
            want = replay_decision(group, {ex["qid"]: ex["target"] for ex in group if ex["qid"] in preds}, gate)
            got = replay_decision(group, preds, gate)
            if want is not None and got is not None:
                s2, played = group[0].get("s2_key"), group[0].get("played_key")
                decisions.append({"agree": got["key"] == want["key"], "accept": got["accept"],
                                  "target_accept": want["accept"], "consistent": got["consistent"],
                                  "branch_agree": got["branch"] == want["branch"], "reason": got["reason_code"],
                                  "s2_agree": (got["key"] == s2) if s2 else None,
                                  "played_agree": (got["key"] == played) if played else None,
                                  "target_played_agree": (want["key"] == played) if played else None})
    out = {"requests": len(latencies), "errors": errors, "first_error": first_error, "overall": _metrics(rows),
           "latency_s": {"mean": round(statistics.mean(latencies), 4), "p50": round(statistics.median(latencies), 4),
                         "p90": round(sorted(latencies)[int(0.9 * (len(latencies) - 1))], 4),
                         "max": round(max(latencies), 4)} if latencies else {},
           "decision": _decision_metrics(decisions),
           "by_purpose": {p: _metrics([r for r in rows if r["purpose"] == p]) for p in sorted({r["purpose"] for r in rows})},
           "by_question": {q: _metrics([r for r in rows if r["qgroup"] == q]) for q in sorted({r["qgroup"] for r in rows})}}
    if keep_predictions:
        out["predictions"] = kept
    return out
