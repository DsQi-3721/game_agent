"""phi(o | d): relevance-conditioned distillation of one raw tool observation.

Three stages, as in the reference ``distillation.py``:
  1. preservation filter -- pull candidate critical facts (percentages, numbers,
     Pokemon/move names) out of the FULL raw text before anything is dropped;
  2. chunked relevance filter -- keep the lines that mention the task's entities
     (rule mode) instead of hard truncation;
  3. task-relevance gate -- re-append preserved numbers that the filter dropped.
Short observations (most battle tools print a few lines) pass through untouched.
``llm`` mode replaces stage 2 with an auxiliary-model compression call (the
reference ``compress_tool_output``) and falls back to the rules on any failure.
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable, Optional

from poke_env.data import to_id_str

from arena.agents.hymem.prompts import DISTILL_PROMPT

_NUM = re.compile(r"\b\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?%|\b\d+HKO\b|\bOHKO\b", re.IGNORECASE)


def preserve(raw: str, limit: int = 12) -> list[str]:
    """Stage 1: damage/percentage/KO facts with a few words of context."""
    out = []
    for line in raw.splitlines():
        if _NUM.search(line):
            out.append(line.strip())
        if len(out) >= limit:
            break
    return out


def _task_terms(task: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'\-]+", task or "")
    terms = {to_id_str(w) for w in words if len(w) >= 4}
    return {t for t in terms if t} - {"what", "with", "does", "their", "your", "from", "move", "moves",
                                      "damage", "much", "check", "which", "would", "against", "opponent"}


def relevance_filter(raw: str, task: str, max_chars: int) -> tuple[str, int]:
    """Stage 2 (rules): keep header + task-relevant lines in original order."""
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    terms = _task_terms(task)
    scored = []
    for i, ln in enumerate(lines):
        norm = to_id_str(ln)
        score = sum(1 for t in terms if t in norm) * 2 + (1 if _NUM.search(ln) else 0) + (3 if i == 0 else 0)
        scored.append((score, i, ln))
    keep, used = set(), 0
    for score, i, ln in sorted(scored, key=lambda x: (-x[0], x[1])):
        if used + len(ln) + 1 > max_chars and keep:
            continue
        keep.add(i)
        used += len(ln) + 1
    kept = [ln for i, ln in enumerate(lines) if i in keep]
    return "\n".join(kept), len(lines) - len(kept)


async def distill(raw: str, task: str, *, mode: str = "rule", max_chars: int = 700,
                  llm_call: Optional[Callable[[list[dict]], Awaitable[str]]] = None) -> tuple[str, dict]:
    raw = raw or ""
    stats = {"raw_chars": len(raw), "dropped_lines": 0, "mode": "passthrough"}
    if len(raw) <= max_chars:
        return raw, stats
    candidates = preserve(raw)
    text = ""
    if mode == "llm" and llm_call is not None:
        try:
            text = (await llm_call([{"role": "user", "content": DISTILL_PROMPT.format(
                max_chars=max_chars, task=task, raw=raw[:12000])}])).strip()
            stats["mode"] = "llm"
        except Exception:  # noqa: BLE001
            text = ""
        if len(text) > int(max_chars * 1.3):
            text = ""
    if not text:
        text, dropped = relevance_filter(raw, task, max_chars)
        stats.update(mode="rule", dropped_lines=dropped)
        if dropped:
            text += f"\n(+{dropped} less relevant lines omitted)"
    missing = [c for c in candidates if c not in text][:4]
    if missing:                                                  # stage 3
        text += "\nKey numbers: " + " | ".join(missing)
    stats["out_chars"] = len(text)
    return text, stats
