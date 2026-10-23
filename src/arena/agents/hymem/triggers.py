"""Fold trigger (paper Eq. 17; reference ``hymem_trigger.FoldTrigger``).

Fires on the planner's explicit request or when any deterministic predicate holds:
context ratio, executor calls, consecutive executor failures, tool calls, planner
turns (all counted since the last fold) or consecutive very-low-confidence answers.
Constraints from the reference: ``max_folds`` and ``min_turns_between_folds``.
Unlike the reference, the tool counter resets at each fold (the paper's
"since the last fold"; the reference's never-cleared history kept it triggered).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from arena.agents.hymem.config import HyMemConfig


@dataclass
class FoldTrigger:
    cfg: HyMemConfig
    executor_calls: int = 0
    consecutive_failures: int = 0
    planner_turns: int = 0
    tool_calls: int = 0
    confidences: list = field(default_factory=list)
    folds: int = 0
    last_fold_at: Optional[int] = None    # global planner-turn index of the last fold
    last_reason: str = ""

    def record_planner_turn(self) -> None:
        self.planner_turns += 1

    def record_executor(self, success: bool, n_tools: int = 0) -> None:
        self.executor_calls += 1
        self.tool_calls += n_tools
        self.consecutive_failures = 0 if success else self.consecutive_failures + 1

    def record_tools(self, n: int) -> None:
        self.tool_calls += n

    def record_confidence(self, conf: Optional[float]) -> None:
        if conf is None:
            return
        self.confidences = (self.confidences + [conf])[-10:]

    def reset(self, global_turn: int) -> None:
        self.executor_calls = self.consecutive_failures = self.planner_turns = self.tool_calls = 0
        self.confidences = []
        self.folds += 1
        self.last_fold_at = global_turn

    def can_fold(self, global_turn: int) -> bool:
        if self.cfg.max_folds is not None and self.folds >= self.cfg.max_folds:
            return False
        if self.last_fold_at is not None and global_turn - self.last_fold_at < self.cfg.min_turns_between_folds:
            return False
        return True

    def should_fold(self, *, context_tokens: int, window: int, global_turn: int,
                    active: bool = False) -> Optional[str]:
        """Reason string if a fold is due now (reference priority order), else None."""
        if not self.can_fold(global_turn):
            return None
        c = self.cfg
        if active:
            reason = "active_fold"
        elif self.executor_calls >= c.n_exec:
            reason = "executor_calls_threshold"
        elif window and context_tokens / window > c.ctx_ratio:
            reason = "context_overflow"
        elif self.consecutive_failures >= c.n_fail:
            reason = "consecutive_failures"
        elif self.tool_calls >= c.n_tool:
            reason = "tool_calls_threshold"
        elif self.planner_turns >= c.n_turn:
            reason = "planner_turns_threshold"
        elif (len(self.confidences) >= c.n_low_conf
              and all(x < c.low_conf for x in self.confidences[-c.n_low_conf:])):
            reason = "consecutive_low_confidence"
        else:
            return None
        self.last_reason = reason
        return reason
