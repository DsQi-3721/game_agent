"""Per-battle state of the HyMem agent.

One ``BattleSession`` per battle tag holds everything the planner persists: the
battle brief (q), the typed history H^p, the memory store, the fold trigger and
the log cursor, plus telemetry. Sessions live in a small LRU (one agent instance
plays all battles of a pairing) and are dropped once the battle has ended and no
decision for it is still in flight.
"""

from __future__ import annotations

import contextvars
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional

from arena.agents.hymem.config import HyMemConfig
from arena.agents.hymem.memory import HyMemStore
from arena.agents.hymem.messages import Entry
from arena.agents.hymem.triggers import FoldTrigger
from arena.state.battle_log import LogDigester


@dataclass
class DecisionStats:
    planner_calls: int = 0
    planner_prompt_tokens: list = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    exec_calls: int = 0
    iso_calls: int = 0
    tool_calls: int = 0
    routed_to_iso: int = 0
    parse_failures: int = 0
    folds: list = field(default_factory=list)
    directives: list = field(default_factory=list)
    by_space: dict = field(default_factory=dict)   # space -> prompt tokens
    llm_time_by_space: dict = field(default_factory=dict)   # space -> summed call latency (s)
    tool_time_s: float = 0.0       # executing tools (Python)
    distill_time_s: float = 0.0    # phi on tool observations
    t_start: float = 0.0           # wall clock when act() started this decision
    extra: dict = field(default_factory=dict)   # subclass telemetry merged into the trajectory


# Background work (folds, deep-thinking jobs) books its LLM calls and counters to its own DecisionStats instead of
# whatever decision happens to be current (``sess.stats`` reads this scope first).
_STATS_SCOPE: contextvars.ContextVar[Optional[DecisionStats]] = contextvars.ContextVar("hymem_stats_scope",
                                                                                        default=None)


@contextmanager
def stats_scope(stats: DecisionStats):
    token = _STATS_SCOPE.set(stats)
    try:
        yield stats
    finally:
        _STATS_SCOPE.reset(token)


@dataclass
class BattleSession:
    tag: str
    brief: str
    trigger: FoldTrigger
    digester: LogDigester = field(default_factory=LogDigester)
    store: HyMemStore = field(default_factory=HyMemStore)
    entries: list = field(default_factory=list)    # H^p: list[Entry]
    cursor: int = 0
    decision_idx: int = 0
    planner_turns_total: int = 0
    iso_used_window: int = 0
    last_planner_tokens: int = 0
    last_key: Optional[tuple] = None     # (rqid, replay length) of the previous decision
    last_order: Optional[str] = None
    closed: bool = False
    in_flight: int = 0
    decision_stats: DecisionStats = field(default_factory=DecisionStats)

    @property
    def stats(self) -> DecisionStats:
        """The current decision's stats, or the running background job's (``stats_scope``)."""
        return _STATS_SCOPE.get() or self.decision_stats

    @stats.setter
    def stats(self, value: DecisionStats) -> None:
        self.decision_stats = value

    def add(self, entry: Entry) -> None:
        self.entries.append(entry)

    def history_text(self, entries: Optional[list] = None) -> str:
        return "\n".join(e.render() for e in (self.entries if entries is None else entries))

    def keep_recent(self, n: int, cut: Optional[int] = None, decision: Optional[int] = None) -> None:
        """After a fold: keep this decision's entries plus the last ``n`` before them. A fold that ran in the
        background summarised only ``entries[:cut]`` (as of decision ``decision``): everything added after the cut
        is kept untouched, it was never summarised."""
        cut = len(self.entries) if cut is None else cut
        d = self.decision_idx if decision is None else decision
        old, new = self.entries[:cut], self.entries[cut:]
        cur = [e for e in old if e.decision == d]
        prev = [e for e in old if e.decision != d][-n:] if n > 0 else []
        self.entries = prev + cur + new


class SessionStore:
    def __init__(self, cfg: HyMemConfig, max_sessions: int = 8):
        self.cfg = cfg
        self.max_sessions = max_sessions
        self._s: "OrderedDict[str, BattleSession]" = OrderedDict()

    def get(self, tag: str) -> Optional[BattleSession]:
        return self._s.get(tag)

    def get_or_create(self, tag: str, brief_fn: Callable[[], str]) -> BattleSession:
        sess = self._s.get(tag)
        if sess is None:
            sess = BattleSession(tag=tag, brief=brief_fn(), trigger=FoldTrigger(self.cfg))
            self._s[tag] = sess
            self._evict()
        self._s.move_to_end(tag)
        return sess

    def _evict(self) -> None:
        while len(self._s) > self.max_sessions:
            victim = next((t for t, s in self._s.items() if s.in_flight == 0), None)
            if victim is None:
                return
            self._s.pop(victim)

    def close(self, tag: str) -> Optional[BattleSession]:
        sess = self._s.get(tag)
        if sess is None:
            return None
        sess.closed = True
        if sess.in_flight == 0:
            self._s.pop(tag, None)
        return sess

    def release(self, sess: BattleSession) -> None:
        sess.in_flight = max(0, sess.in_flight - 1)
        if sess.closed and sess.in_flight == 0:
            self._s.pop(sess.tag, None)

    def __len__(self) -> int:
        return len(self._s)
