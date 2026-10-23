"""Subtask complexity router (paper Fig. 2b "quality assessment"; reference
``complexity_router.py``): decides whether a planner ``<task>`` is a single
look-up for the executor or a multi-step question for an isolated session.

Mode ``rule``: the reference's fast path with battle vocabulary; ambiguous tasks
default to the executor. Mode ``llm``: ambiguous tasks go to a small JSON
classification call on the auxiliary model, falling back to the rule estimate on
failure or low confidence. Mode ``off``: every task goes to the executor.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from arena.agents.hymem.prompts import ROUTER_PROMPT

_SIMPLE = ("damage", "calc", "how much", "speed", "faster", "slower", "look up", "lookup", "what does",
           "effective", "effectiveness", "type ", "set of", "sets", "likely moves", "hazard", "list",
           "find", "battle log", "log ", "item", "ability", "stats", "hp ", "ko ", "ohko", "2hko")
# Battle vocabulary. Unlike web research, "X vs Y" / "compare" are usually simple
# calculations here (speed of A vs B), so they are not complexity hints.
_COMPLEX = ("predict", "should i", "should we", "evaluate", "plan", "best play", "best move",
            "best action", "risk", "endgame", "win condition", "which is better", "whether", "if i ",
            "set up", "setup", "sack", "tera now", "terastallize now", "verify", " and then ",
            "strategy", "outplay", "mind game", "double switch")


@dataclass(frozen=True)
class Route:
    target: str        # "executor" | "iso"
    confidence: float
    source: str        # fast_path | llm | fallback | off
    reason: str = ""


def fast_path(task: str, hint_iso: bool = False) -> Optional[Route]:
    t = f" {(task or '').strip().lower()} "
    if len(t) > 260:
        return None
    if any(k in t for k in _COMPLEX):
        return Route("iso", 0.85, "fast_path", "complex reasoning indicator")
    if any(k in t for k in _SIMPLE) and not hint_iso:
        return Route("executor", 0.9, "fast_path", "single look-up / calculation")
    return None


def _parse(text: str) -> Optional[Route]:
    m = re.search(r"\{[^{}]*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    label = str(d.get("complexity", "")).lower()
    label = {"simple": "search", "lookup": "search", "complex": "branch", "reasoning": "branch"}.get(label, label)
    if label not in ("search", "branch"):
        return None
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    return Route("iso" if label == "branch" else "executor", conf, "llm", str(d.get("reason", ""))[:200])


class Router:
    def __init__(self, mode: str, confidence_threshold: float = 0.6):
        self.mode = mode
        self.threshold = confidence_threshold
        self.history: list[Route] = []

    async def route(self, task: str, hint_iso: bool = False,
                    llm_call: Optional[Callable[[list[dict]], Awaitable[str]]] = None) -> Route:
        """``llm_call(messages) -> text`` is used only in ``llm`` mode for ambiguous tasks."""
        if self.mode == "off":
            r = Route("executor", 1.0, "off")
        else:
            r = fast_path(task, hint_iso)
            if r is None and self.mode == "llm" and llm_call is not None:
                try:
                    text = await llm_call([{"role": "user", "content": ROUTER_PROMPT.format(task=task[:1500])}])
                    r = _parse(text)
                except Exception:  # noqa: BLE001
                    r = None
                if r is not None and r.confidence < self.threshold:
                    r = None
            if r is None:
                r = Route("iso" if hint_iso else "executor", 0.5, "fallback", "ambiguous")
        self.history.append(r)
        return r
