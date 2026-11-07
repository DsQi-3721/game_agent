"""Per-turn decision trajectory logging (one JSONL per agent per experiment).

Captures the raw material for post-hoc reasoning analysis: rendered state, full
prompt, raw LLM output, parsed action, fallback flag. One logger instance per
agent name; appended line-buffered so a crash keeps what ran.
"""

from __future__ import annotations

import json
from pathlib import Path

from arena.observation.capture import CURRENT_REQUEST


class TrajectoryLogger:
    def __init__(self, path: Path | str, agent_name: str, on_turn=None):
        self.path = Path(path)
        self.agent_name = agent_name
        self.on_turn = on_turn   # optional tee: called with the record dict after writing
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def log_turn(self, *, battle_tag, turn, state_text, prompt, raw_llm_output,
                 tool_calls=None, parsed_order=None, fallback_used=False, note="",
                 extra=None) -> None:
        """``extra`` holds agent-specific per-decision telemetry (e.g. context sizes
        per space, folds); written under the ``extra`` key only when given."""
        rec = {
            "agent": self.agent_name,
            "battle_tag": battle_tag,
            "turn": turn,
            "state": state_text,
            "prompt": prompt,
            "output": raw_llm_output,
            "n_tool_calls": len(tool_calls or []),
            "parsed_order": parsed_order,
            "fallback": fallback_used,
            "note": note,
        }
        cur = CURRENT_REQUEST.get()
        if cur is not None and cur[0] == battle_tag:
            rec["rqid"] = cur[1]          # joins this record with the stream / server log of that request
        if extra:
            rec["extra"] = extra
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()
        if self.on_turn is not None:
            try:
                self.on_turn(rec)
            except Exception:  # noqa: BLE001  (a tee failure must never break a turn)
                pass

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:  # noqa: BLE001
            pass
