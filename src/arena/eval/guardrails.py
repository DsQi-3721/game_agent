"""Cost/usage guardrails: hard caps + a dry-run estimator.

A shared budget across the whole experiment. Agents report token usage after each
LLM call; the tournament checks the guardrail between battles and aborts cleanly
when a cap is hit, so a runaway config can't silently burn money.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Guardrails:
    max_cost_usd: Optional[float] = None
    max_tokens: Optional[int] = None
    max_battles: Optional[int] = None
    dry_run: bool = False

    spent_usd: float = 0.0
    spent_tokens: int = 0
    battles_done: int = 0
    _notes: list[str] = field(default_factory=list)

    def record(self, cost_usd: float, tokens: int) -> None:
        self.spent_usd += cost_usd
        self.spent_tokens += tokens

    def battle_done(self) -> None:
        self.battles_done += 1

    def check(self) -> None:
        if self.max_cost_usd is not None and self.spent_usd >= self.max_cost_usd:
            raise BudgetExceeded(f"cost cap ${self.max_cost_usd} reached (${self.spent_usd:.4f})")
        if self.max_tokens is not None and self.spent_tokens >= self.max_tokens:
            raise BudgetExceeded(f"token cap {self.max_tokens} reached ({self.spent_tokens})")
        if self.max_battles is not None and self.battles_done >= self.max_battles:
            raise BudgetExceeded(f"battle cap {self.max_battles} reached")

    def remaining_battles(self) -> Optional[int]:
        return None if self.max_battles is None else max(0, self.max_battles - self.battles_done)

    @staticmethod
    def estimate(n_matchups: int, n_battles: int, avg_turns: float,
                 calls_per_turn: float, tokens_per_call: int,
                 price_in: float, price_out: float, out_frac: float = 0.4) -> dict:
        """Rough pre-run cost estimate (dry-run)."""
        total_calls = n_matchups * n_battles * avg_turns * calls_per_turn
        total_tokens = total_calls * tokens_per_call
        in_tok = total_tokens * (1 - out_frac)
        out_tok = total_tokens * out_frac
        cost = (in_tok * price_in + out_tok * price_out) / 1e6
        return {
            "matchups": n_matchups,
            "battles_total": n_matchups * n_battles,
            "est_llm_calls": round(total_calls),
            "est_tokens": round(total_tokens),
            "est_cost_usd": round(cost, 2),
        }
