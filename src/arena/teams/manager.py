"""Team assignment policy for evaluation matches.

Three modes (decision-quality vs team-luck control):
  - MIRROR: both sides get the SAME team (isolates decision quality). Role
    alternation (who is P1) is the runner's job, driven by match_idx.
  - POOL_RANDOM: each side independently draws a team from the tier pool.
  - AGENT_BUILT: agents construct their own legal team (v2; needs validator).

Random formats (e.g. gen9randombattle) need no team -> (None, None).
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Optional

from arena.teams.pool import TeamPool


class AssignmentMode(str, Enum):
    MIRROR = "mirror"
    POOL_RANDOM = "pool_random"
    AGENT_BUILT = "agent_built"


def is_random_format(fmt: str) -> bool:
    return "random" in fmt.lower()


class TeamManager:
    def __init__(self, pool: Optional[TeamPool] = None, rng: Optional[random.Random] = None):
        self.pool = pool or TeamPool()
        self.rng = rng or random.Random()

    def assign(self, fmt: str, mode: AssignmentMode, match_idx: int = 0) -> tuple[Optional[str], Optional[str]]:
        if is_random_format(fmt):
            return None, None

        if mode == AssignmentMode.AGENT_BUILT:
            raise NotImplementedError("agent-built teams land in v2 (needs Showdown validator)")

        teams = self.pool.get(fmt)
        if not teams:
            raise ValueError(
                f"no teams for {fmt!r} in pool; run scripts/ingest_teams.py {fmt} "
                f"or drop pastes under data/teams/{fmt}/custom/"
            )

        if mode == AssignmentMode.MIRROR:
            # Same team both sides; stable across a swap-pair so the only thing that
            # changes between the two matches of a pair is which agent is P1.
            team = teams[(match_idx // 2) % len(teams)]
            return team, team

        if mode == AssignmentMode.POOL_RANDOM:
            return self.rng.choice(teams), self.rng.choice(teams)

        raise ValueError(f"unknown mode {mode}")
