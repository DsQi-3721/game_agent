"""Experience -> training examples for a local System-One student.

One example per answered question of a teacher request (``backend`` starting with one of
``teachers``, e.g. ``typesafe``):

  target   the teacher's full distribution over the rendered options (soft labels, so the
           student also learns the teacher's uncertainty -- the routing signal)
  + S2     when System 2 completed the decision, its choice is mixed in for the action
           questions (stay / best_move / best_switch / flat best), weight ``s2_weight``:
           the slow system corrects the fast one
  weight   battle outcome: won ``won_weight``, lost ``lost_weight``, unknown ``unknown_weight``
  split    by battle (all questions of one battle stay on one side; no leakage)
  self     ``self_labels`` (no teacher, HANDOVER 4.4): a request answered by the student itself becomes an
           example only where something corrects it -- System 2's choice (``s2_key``, mixed in with
           ``s2_weight``) or, with ``won_self_weight`` > 0, the move played in a battle that was won
           (self-imitation). The student's own distribution is the soft base, so the target stays
           calibrated; questions the correction says nothing about (best_switch when System 2 attacked)
           are skipped. Decisions with a teacher answer keep the teacher as their label.
  debias   ``debias_orders``: a Choice asked in both option orders (``q`` / ``q_rev``) gets the mean of
           the two teacher distributions as the target of both -- the decider averages the orders
           anyway, and Jev's own top choice flips with the order on ~20% of these questions, which a
           student copying each order would inherit as gate rejections ("order_inconsistent")
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from arena.s1.prompts import options, target_distribution


@dataclass
class BuildConfig:
    teachers: tuple = ("typesafe",)
    purposes: Optional[tuple] = None
    s2_weight: float = 0.5
    won_weight: float = 1.0
    lost_weight: float = 0.5
    unknown_weight: float = 0.8
    valid_frac: float = 0.15
    max_per_request: Optional[int] = None
    self_labels: bool = False
    won_self_weight: float = 0.0
    debias_orders: bool = False
    seed: int = 0


def split_of(key: str, valid_frac: float) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "valid" if h < valid_frac else "train"


def s2_onehot(qid: str, keys: list[str], s2_key: str) -> Optional[list[float]]:
    """System 2's played option as a target for the action questions it determines."""
    base = qid[:-4] if qid.endswith("_rev") else qid
    if base == "stay":
        is_move = s2_key.startswith("move_")
        return [1.0 if (k == "true") == is_move else 0.0 for k in keys]
    if base in ("best", "best_move", "best_switch") and s2_key in keys:
        return [1.0 if k == s2_key else 0.0 for k in keys]
    return None


def debias_pairs(examples: list[dict]) -> None:
    """In place: ``q`` and ``q_rev`` of one request both get the key-wise mean of their targets."""
    by_qid = {ex["qid"]: ex for ex in examples}
    for qid, fwd in by_qid.items():
        rev = by_qid.get(qid + "_rev")
        if rev is None or set(fwd["keys"]) != set(rev["keys"]):
            continue
        mean = {k: (a + b) / 2 for k, a in zip(fwd["keys"], fwd["target"])
                for k2, b in zip(rev["keys"], rev["target"]) if k2 == k}
        for ex in (fwd, rev):
            ex["target"] = [round(mean[k], 6) for k in ex["keys"]]
            ex["source"] += "+debiased"


def build(records: list[dict], cfg: Optional[BuildConfig] = None) -> list[dict]:
    cfg = cfg or BuildConfig()
    rng = random.Random(cfg.seed)
    decisions = {(r.get("agent"), r.get("decision_id")): r for r in records
                 if r.get("type") == "decision" and r.get("decision_id")}
    outcomes = {r.get("battle_tag"): r for r in records if r.get("type") == "outcome" and r.get("battle_tag")}

    def is_teacher(r: dict) -> bool:
        return r.get("type") == "s1" and any(t and str(r.get("backend", "")).startswith(t) for t in cfg.teachers)

    taught = {(r.get("agent"), r.get("decision_id")) for r in records if is_teacher(r)}
    out: list[dict] = []
    for i, r in enumerate(records):
        if r.get("type") != "s1":
            continue
        teacher = is_teacher(r)
        if not teacher and not (cfg.self_labels and r.get("purpose") == "action"
                                and (r.get("agent"), r.get("decision_id")) not in taught):
            continue
        if cfg.purposes and r.get("purpose") not in cfg.purposes:
            continue
        group = r.get("battle_tag") or r.get("snapshot") or f"offline-{i}"
        oc = outcomes.get(r.get("battle_tag"))
        weight = (cfg.unknown_weight if oc is None else cfg.won_weight if oc.get("won") else cfg.lost_weight)
        dec = decisions.get((r.get("agent"), r.get("decision_id")))
        s2_key = dec.get("s2_key") if dec else None
        if not teacher:
            if s2_key and cfg.s2_weight > 0:
                fix, mix, self_source = s2_key, cfg.s2_weight, "self+s2"
            elif cfg.won_self_weight > 0 and oc is not None and oc.get("won") and (dec or {}).get("final_key"):
                fix, mix, self_source = dec["final_key"], cfg.won_self_weight, "self+won"
            else:
                continue
        items = list((r.get("questions") or {}).items())
        if cfg.max_per_request and len(items) > cfg.max_per_request:
            items = rng.sample(items, cfg.max_per_request)
        first = len(out)
        for qid, q in items:
            a = (r.get("answers") or {}).get(qid)
            if not a:
                continue
            try:
                keys = [k for k, _ in options(q)]
            except ValueError:
                continue
            target = target_distribution(q, a, keys)
            if not teacher:
                hot = s2_onehot(qid, keys, fix)
                if hot is None:
                    continue
                target = [(1 - mix) * t + mix * h for t, h in zip(target, hot)]
                source = self_source
            else:
                source = "teacher"
                if s2_key and cfg.s2_weight > 0 and r.get("purpose") == "action":
                    hot = s2_onehot(qid, keys, s2_key)
                    if hot is not None:
                        target = [(1 - cfg.s2_weight) * t + cfg.s2_weight * h for t, h in zip(target, hot)]
                        source = "teacher+s2"
            out.append({"id": f"{group}:{r.get('decision_id') or i}:{qid}", "request": i, "group": group,
                        "purpose": r.get("purpose", ""), "qid": qid, "state": r.get("state"), "question": q,
                        "keys": keys, "target": [round(t, 6) for t in target], "weight": weight, "source": source,
                        "teacher": r.get("model", ""), "split": split_of(group, cfg.valid_frac),
                        "s2_key": s2_key, "final_key": (dec or {}).get("final_key")})
        if cfg.debias_orders:
            debias_pairs(out[first:])
    return out


def write_jsonl(examples: list[dict], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def read_jsonl(path: Path | str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def describe(examples: list[dict]) -> dict:
    return {"n": len(examples), "by_split": dict(Counter(e["split"] for e in examples)),
            "by_purpose": dict(Counter(e["purpose"] for e in examples)),
            "by_source": dict(Counter(e["source"] for e in examples)),
            "groups": len({e["group"] for e in examples})}
