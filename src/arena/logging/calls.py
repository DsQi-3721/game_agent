"""Per-LLM-call JSONL log (one file per agent: ``calls_<agent>.jsonl``).

Agents with several context spaces (planner / executor / isolated reasoning /
memory fold ...) make many LLM calls per decision. This records each one: which
space made it, the exact messages it saw, what came back, tokens and latency.
That is the raw material for context-dilution analysis and, later, SFT/RL data.

Levels: ``full`` (messages + output), ``meta`` (sizes/tokens only), ``off``.
The file is opened per record in append mode, so several agent instances (one per
pairing) can share it without any close/lifecycle handling. Named ``calls_*`` on
purpose: the TUI treats ``traj_*`` files as decision trajectories.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

LEVELS = ("full", "meta", "off")


class CallLogger:
    def __init__(self, path: Path | str, agent_name: str, level: str = "full"):
        if level not in LEVELS:
            raise ValueError(f"call log level must be one of {LEVELS}, got {level!r}")
        self.path = Path(path)
        self.agent_name = agent_name
        self.level = level
        if level != "off":
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def beside(cls, traj_logger, level: str = "full") -> Optional["CallLogger"]:
        """A call log next to an agent's trajectory file (None if no trajectory logger)."""
        if traj_logger is None or level == "off":
            return None
        path = Path(traj_logger.path).with_name(f"calls_{traj_logger.agent_name}.jsonl")
        return cls(path, traj_logger.agent_name, level)

    def log(self, *, battle_tag: str, decision: int, space: str, messages: list[dict],
            output: str, reasoning: str = "", prompt_tokens: int = 0,
            completion_tokens: int = 0, latency_s: float = 0.0, extra: Optional[dict] = None,
            t_start: Optional[float] = None, cached_tokens: int = 0, finish_reason: str = "") -> None:
        """``t_start`` is the wall-clock (``time.time()``) start of the call, for timelines;
        ``ts`` is when the record was written (= end of the call)."""
        if self.level == "off":
            return
        rec = {
            "ts": round(time.time(), 3),
            "agent": self.agent_name,
            "battle_tag": battle_tag,
            "decision": decision,
            "space": space,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_s": round(latency_s, 3),
            "n_messages": len(messages),
            "prompt_chars": sum(len(str(m.get("content") or "")) for m in messages),
            "output_chars": len(output or ""),
        }
        if t_start is not None:
            rec["t_start"] = round(t_start, 3)
        if cached_tokens:
            rec["cached_tokens"] = cached_tokens
        if finish_reason:
            rec["finish_reason"] = finish_reason
        if self.level == "full":
            rec["messages"] = [{"role": m.get("role"), "content": m.get("content")} for m in messages]
            rec["output"] = output
            if reasoning:
                rec["reasoning"] = reasoning
        if extra:
            rec["extra"] = extra
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except OSError:  # logging must never break a decision
            pass
