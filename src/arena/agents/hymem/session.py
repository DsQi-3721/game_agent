"""Per-battle state of the HyMem agent.

One ``BattleSession`` per battle tag holds everything the planner persists: the
battle brief (q), the typed history H^p, the memory store, the fold trigger and
the log cursor, plus telemetry. Sessions live in a small LRU (one agent instance
plays all battles of a pairing) and are dropped once the battle has ended and no
decision for it is still in flight.
"""

from __future__ import annotations

from collections import OrderedDict
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
    stats: DecisionStats = field(default_factory=DecisionStats)

    def add(self, entry: Entry) -> None:
        self.entries.append(entry)

    def history_text(self, entries: Optional[list] = None) -> str:
        return "\n".join(e.render() for e in (self.entries if entries is None else entries))

    def keep_recent(self, n: int) -> None:
        """After a fold: keep this decision's entries plus the last ``n`` before them."""
        cur = [e for e in self.entries if e.decision == self.decision_idx]
        prev = [e for e in self.entries if e.decision != self.decision_idx][-n:] if n > 0 else []
        self.entries = prev + cur


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
