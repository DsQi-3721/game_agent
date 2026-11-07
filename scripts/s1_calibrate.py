"""Fit the scorer's calibration temperature on a held-out split, from saved predictions (no GPU).

    python scripts/s1_calibrate.py --data /data/wangxinqi/s1v6/kakuna/valid.jsonl \
        --preds runs/s1_eval/v6/s1v6-2b-ep2-valid.preds.jsonl \
        --check /data/wangxinqi/s1v6/kakuna/test_T7.jsonl=runs/s1_eval/v6/s1v6-2b-ep2-T7.preds.jsonl

The scorer answers softmax(letter logits / T) (``s1_serve.py --temperature``). Saved predictions were made at
T = 1, so the answer at another T is p^(1/T), renormalised. T is chosen on the fit split by the cross-entropy
to the teacher's soft targets; ``--check`` reports question-level ECE and cross-entropy at T = 1 and at the
fitted T on other splits (held out from the fit).
"""

from __future__ import annotations

import argparse
import json
import math

from arena.s1.dataset import read_jsonl
from arena.s1.evaluate import ece


def pairs(data: str, preds: str) -> list[tuple[list[float], list[float]]]:
    ex = {(e["group"], e["request"], e["qid"]): e["target"] for e in read_jsonl(data)}
    out = []
    for r in read_jsonl(preds):
        for qid, p in r["preds"].items():
            t = ex.get((r["group"], r["request"], qid))
            if t is not None and len(t) == len(p):
                out.append((p, t))
    return out


def at(p: list[float], temp: float) -> list[float]:
    w = [max(x, 1e-12) ** (1 / temp) for x in p]
    z = sum(w)
    return [x / z for x in w]


def metrics(rows, temp: float) -> dict:
    ce, cal = 0.0, []
    for p, t in rows:
        q = at(p, temp)
        ce -= sum(ti * math.log(max(qi, 1e-12)) for ti, qi in zip(t, q))
        top = max(range(len(q)), key=q.__getitem__)
        cal.append((q[top], top == max(range(len(t)), key=t.__getitem__)))
    return {"n": len(rows), "cross_entropy": round(ce / max(1, len(rows)), 4), "ece": ece(cal)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--check", action="append", default=[], help="DATA=PREDS of another split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = pairs(args.data, args.preds)
    grid = [round(0.5 + 0.05 * i, 2) for i in range(71)]          # 0.5 .. 4.0
    best = min(grid, key=lambda t: metrics(rows, t)["cross_entropy"])
    res = {"fit": {"T=1": metrics(rows, 1.0), f"T={best}": metrics(rows, best)}, "temperature": best, "check": {}}
    for spec in args.check:
        d, p = spec.split("=", 1)
        r = pairs(d, p)
        res["check"][d] = {"T=1": metrics(r, 1.0), f"T={best}": metrics(r, best)}
    print(json.dumps(res, indent=1))
    if args.out:
        open(args.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
