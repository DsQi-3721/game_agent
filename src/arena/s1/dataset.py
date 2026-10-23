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


def build(records: list[dict], cfg: Optional[BuildConfig] = None) -> list[dict]:
    cfg = cfg or BuildConfig()
    rng = random.Random(cfg.seed)
    decisions = {(r.get("agent"), r.get("decision_id")): r for r in records
                 if r.get("type") == "decision" and r.get("decision_id")}
    outcomes = {r.get("battle_tag"): r for r in records if r.get("type") == "outcome" and r.get("battle_tag")}
    out: list[dict] = []
    for i, r in enumerate(records):
        if r.get("type") != "s1" or not any(str(r.get("backend", "")).startswith(t) for t in cfg.teachers):
            continue
        if cfg.purposes and r.get("purpose") not in cfg.purposes:
            continue
        group = r.get("battle_tag") or r.get("snapshot") or f"offline-{i}"
        oc = outcomes.get(r.get("battle_tag"))
        weight = (cfg.unknown_weight if oc is None else cfg.won_weight if oc.get("won") else cfg.lost_weight)
        dec = decisions.get((r.get("agent"), r.get("decision_id")))
        s2_key = dec.get("s2_key") if dec else None
        items = list((r.get("questions") or {}).items())
        if cfg.max_per_request and len(items) > cfg.max_per_request:
            items = rng.sample(items, cfg.max_per_request)
        for qid, q in items:
            a = (r.get("answers") or {}).get(qid)
            if not a:
                continue
            try:
                keys = [k for k, _ in options(q)]
            except ValueError:
                continue
            target = target_distribution(q, a, keys)
            source = "teacher"
            if s2_key and cfg.s2_weight > 0 and r.get("purpose") == "action":
                hot = s2_onehot(qid, keys, s2_key)
                if hot is not None:
                    target = [(1 - cfg.s2_weight) * t + cfg.s2_weight * h for t, h in zip(target, hot)]
                    source = "teacher+s2"
            out.append({"id": f"{group}:{r.get('decision_id') or i}:{qid}", "request": i, "group": group,
                        "purpose": r.get("purpose", ""), "qid": qid, "state": r.get("state"), "question": q,
                        "keys": keys, "target": [round(t, 6) for t in target], "weight": weight, "source": source,
                        "teacher": r.get("model", ""), "split": split_of(group, cfg.valid_frac)})
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
