"""Evaluation metrics: Wilson CI for win rates, Elo from match results."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def elo_from_results(
    results: Iterable["MatchResultLike"],
    k: float = 32.0,
    base: float = 1000.0,
) -> dict[str, float]:
    """Compute Elo ratings by replaying per-battle outcomes.

    Each result contributes ``a_wins`` wins for A and ``b_wins`` for B; we replay
    them as individual games (ties as 0.5/0.5) so order-independence is approximate
    but stable for a fixed result set.
    """
    ratings: dict[str, float] = defaultdict(lambda: base)
    games: list[tuple[str, str, float]] = []
    for r in results:
        for _ in range(r.a_wins):
            games.append((r.a, r.b, 1.0))
        for _ in range(r.b_wins):
            games.append((r.a, r.b, 0.0))
        for _ in range(r.ties):
            games.append((r.a, r.b, 0.5))

    for a, b, score in games:
        ra, rb = ratings[a], ratings[b]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        ratings[a] = ra + k * (score - ea)
        ratings[b] = rb + k * ((1.0 - score) - (1.0 - ea))
    return dict(ratings)


class MatchResultLike:  # structural doc only; real type is eval.runner.MatchResult
    a: str
    b: str
    a_wins: int
    b_wins: int
    ties: int
