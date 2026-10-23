"""The single rendering of a typed question for a local System-One model.

Server, trainer and dataset all call ``render`` so a student is scored on exactly the text it
was trained on. Every question type becomes a lettered multiple choice:

  choice  the criteria in order            -> P over option keys
  noul    "true" / "false" (+ criteria)    -> P(true) is the Noul answer
  score   the levels in order ("0", "1",.) -> P over levels, score = argmax

The state comes first in the user message, so all questions of one request share the longest
possible token prefix (the scorer prefills it once).
"""

from __future__ import annotations

import json
import string
from typing import Any

LABELS = string.ascii_uppercase + string.ascii_lowercase       # 52 single-token letters
SYSTEM = ("You are a fast decision model. Read the state, then answer the question by choosing exactly one "
          "of the options. Reply with the option's letter only.")


def _text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def options(question: dict) -> list[tuple[str, Any]]:
    """(key, description) pairs in the order they are shown."""
    t = question.get("type")
    crit = question.get("criteria")
    if t == "choice":
        return list((crit or {}).items())
    if t == "noul":
        crit = crit or {}
        return [("true", crit.get("true") or "yes"), ("false", crit.get("false") or "no")]
    if t == "score":
        return [(str(i), level) for i, level in enumerate(crit or [])]
    raise ValueError(f"unsupported question type {t!r}")


def render(state: Any, question: dict) -> tuple[list[dict], list[str]]:
    """-> (chat messages, option keys in label order)."""
    opts = options(question)
    if len(opts) > len(LABELS):
        raise ValueError(f"at most {len(LABELS)} options can be scored, got {len(opts)}")
    lines = []
    for label, (key, desc) in zip(LABELS, opts):
        d = _text(desc)
        lines.append(f"{label}) {key}" + (f": {d}" if d else ""))
    user = (f"STATE:\n{_text(state)}\n\nQUESTION ({question.get('type')}):\n{_text(question.get('instructions'))}"
            f"\n\nOPTIONS:\n" + "\n".join(lines) + "\n\nAnswer with one letter.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], [k for k, _ in opts]


def to_answer(question: dict, keys: list[str], probs: list[float]) -> dict:
    """Probabilities over the rendered options -> a TypeSafe-shaped answer."""
    n = len(keys)
    pmap = {k: float(p) for k, p in zip(keys, probs)}
    t = question.get("type")
    if t == "noul":
        return {"type": "noul", "noul": pmap.get("true", 0.0)}
    top = max(pmap, key=pmap.get)
    conf = (pmap[top] - 1 / n) / (1 - 1 / n) if n > 1 else 1.0
    if t == "score":
        return {"type": "score", "score": int(top), "probabilities": pmap, "confidence": round(max(0.0, conf), 4)}
    return {"type": "choice", "choice": top, "probabilities": pmap, "confidence": round(max(0.0, conf), 4)}


def target_distribution(question: dict, answer: dict, keys: list[str]) -> list[float]:
    """A recorded answer -> probabilities aligned with ``keys`` (the training target)."""
    if question.get("type") == "noul":
        p = float(answer.get("noul") or 0.0)
        return [p if k == "true" else 1 - p for k in keys]
    probs = answer.get("probabilities") or {}
    vals = [max(0.0, float(probs.get(k, 0.0))) for k in keys]
    z = sum(vals)
    return [v / z for v in vals] if z > 0 else [1 / len(keys)] * len(keys)
