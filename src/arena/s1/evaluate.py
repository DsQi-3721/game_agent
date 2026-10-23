"""Score any System-One backend against held-out teacher labels.

Questions are sent grouped by their original request (same state), exactly as the agents send
them, so latency is realistic and the local scorer's prefix cache applies. Metrics, overall and
per purpose (action / memory / controller ...) and per question group (stay, best_move, kind ...):

  agree   top-1 agreement with the target's argmax
  kl      KL(target || prediction), nats
  brier   multi-class Brier score
  ece     expected calibration error of the prediction's top probability vs agreement (10 bins)
  latency seconds per request
"""

from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict
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
            "ece": ece([(r["conf"], r["agree"]) for r in rows])}


async def evaluate(client, examples: list[dict], *, limit_requests: Optional[int] = None,
                   model: Optional[str] = None) -> dict:
    by_req: dict = defaultdict(list)
    for ex in examples:
        by_req[(ex["group"], ex["request"])].append(ex)
    reqs = list(by_req.values())[:limit_requests] if limit_requests else list(by_req.values())
    rows, latencies, errors, first_error = [], [], 0, ""
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
        for ex in group:
            a: Optional[Answer] = res.answers.get(ex["qid"])
            if a is None:
                errors += 1
                continue
            pred = target_distribution(ex["question"], answer_dict(a), ex["keys"])
            tgt = ex["target"]
            kl = sum(t * (math.log(max(t, 1e-9)) - math.log(max(p, 1e-9))) for t, p in zip(tgt, pred) if t > 0)
            brier = sum((p - t) ** 2 for p, t in zip(pred, tgt))
            top = max(range(len(pred)), key=lambda j: pred[j])
            rows.append({"purpose": ex["purpose"], "qgroup": _qgroup(ex["qid"]),
                         "agree": top == max(range(len(tgt)), key=lambda j: tgt[j]), "kl": kl, "brier": brier,
                         "conf": pred[top]})
    out = {"requests": len(latencies), "errors": errors, "first_error": first_error, "overall": _metrics(rows),
           "latency_s": {"mean": round(statistics.mean(latencies), 4), "p50": round(statistics.median(latencies), 4),
                         "max": round(max(latencies), 4)} if latencies else {},
           "by_purpose": {p: _metrics([r for r in rows if r["purpose"] == p]) for p in sorted({r["purpose"] for r in rows})},
           "by_question": {q: _metrics([r for r in rows if r["qgroup"] == q]) for q in sorted({r["qgroup"] for r in rows})}}
    return out
