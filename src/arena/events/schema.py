"""Structured progress events emitted by a running evaluation.

Transport is append-only JSONL (``runs/<name>/events.jsonl``); the TUI tails it.
Every event carries a monotonic ``seq`` so a reader can resume without
double-counting after a reconnect.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    RUN_STARTED = "run_started"        # data: agents[], format, n_pairings, n_battles, total_battles
    PAIRING_STARTED = "pairing_started"  # data: a, b, index
    BATTLE_STARTED = "battle_started"  # data: battle_tag, a, b  (enables open-in-browser)
    TURN = "turn"                      # data: TurnData (the per-decision stream)
    BATTLE_FINISHED = "battle_finished"  # data: winner, turns
    MATCH_FINISHED = "match_finished"  # data: a, b, a_wins, b_wins, ties, avg_turns, totals{}
    METRIC = "metric"                  # data: elo{}, win_rate{}, cost_usd{}, completed/total
    RUN_FINISHED = "run_finished"      # data: report_path, charts_path, aborted
    ERROR = "error"                    # data: message
    EXTERNAL_STARTED = "external_started"  # data: name, username (external bot launching)
    EXTERNAL_READY = "external_ready"      # data: name (logged in, accepting challenges)
    EXTERNAL_DIED = "external_died"        # data: name, message (process exited mid-match)


@dataclass
class Event:
    type: EventType
    ts: float
    seq: int
    run: str
    data: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        d["type"] = self.type.value
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> Optional["Event"]:
        """Tolerant parse: returns None on blank/partial/corrupt lines."""
        line = line.strip()
        if not line:
            return None
        try:
            d = json.loads(line)
            return cls(type=EventType(d["type"]), ts=d["ts"], seq=d["seq"],
                       run=d["run"], data=d.get("data", {}))
        except Exception:  # noqa: BLE001  (a half-written tail line is expected, not an error)
            return None
