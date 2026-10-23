"""Single match-up runner: play N battles between two agents, collect results.

Tournament matrices, Elo and full trajectory/cost metrics arrive in M6; this is
the minimal building block that produces a ``MatchResult`` for one pairing.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from poke_env.data import to_id_str
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from poke_env.teambuilder.teambuilder import Teambuilder

from arena.agents.base import Agent, AgentContext
from arena.agents.adapter import AgentPlayer
from arena.config.schema import ExternalConfig
from arena.teams.manager import AssignmentMode, TeamManager, is_random_format

AgentBuilder = Callable[[AgentContext], Agent]

_account_counter = itertools.count()


def unique_account(label: str) -> AccountConfiguration:
    """A unique Showdown username per player (<=18 chars, alnum-ish).

    Required because many players connect concurrently and stale connections
    linger on the server; reusing a name triggers ``|nametaken|``.
    """
    token = f"{os.getpid() % 100000}{next(_account_counter)}"
    safe = "".join(c for c in label if c.isalnum())[:8] or "agent"
    return AccountConfiguration(f"{safe}{token}"[:18], None)


@dataclass
class AgentSpec:
    name: str
    build: AgentBuilder   # build(ctx) -> Agent; ctx carries llm/renderer/augments/logger


@dataclass
class ExternalAgentSpec:
    """An out-of-process opponent. Not built in our process; instead a separate
    bot connects to the Showdown server as ``username`` and we challenge it."""

    name: str
    username: str
    cfg: ExternalConfig


Spec = Union[AgentSpec, ExternalAgentSpec]


def _orient_external(fmt: str, mode: str, a_name: str, b_name: str, external_is_a: bool,
                     our_wins: int, our_losses: int, n_finished: int,
                     turns: list[int]) -> "MatchResult":
    """Build a MatchResult from OUR (internal) side's counters, assigning wins to
    the correct a/b slot depending on which side was external. Pure -> unit-tested."""
    ties = max(0, n_finished - our_wins - our_losses)
    if external_is_a:                      # a = external opponent, b = our agent
        a_wins, b_wins = our_losses, our_wins
    else:                                  # a = our agent, b = external opponent
        a_wins, b_wins = our_wins, our_losses
    return MatchResult(
        fmt=fmt, mode=mode, a=a_name, b=b_name, n_battles=n_finished,
        a_wins=a_wins, b_wins=b_wins, ties=ties, turns=turns,
        avg_turns=sum(turns) / len(turns) if turns else 0.0,
    )


@dataclass
class MatchResult:
    fmt: str
    mode: str
    a: str
    b: str
    n_battles: int
    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0
    avg_turns: float = 0.0
    turns: list[int] = field(default_factory=list)

    @property
    def a_winrate(self) -> float:
        return self.a_wins / self.n_battles if self.n_battles else 0.0


class _PoolTeambuilder(Teambuilder):
    """Yields a (optionally random) team from a pool on each battle."""

    def __init__(self, pastes: list[str], rng: random.Random, fixed: bool = False):
        self._packed = [self.join_team(self.parse_showdown_team(p)) for p in pastes]
        self._rng = rng
        self._fixed = fixed

    def yield_team(self) -> str:
        return self._packed[0] if self._fixed else self._rng.choice(self._packed)


class MatchRunner:
    def __init__(self, team_manager: Optional[TeamManager] = None, rng: Optional[random.Random] = None):
        self.tm = team_manager or TeamManager()
        self.rng = rng or random.Random()

    def _teambuilders(self, fmt: str, mode: AssignmentMode):
        if is_random_format(fmt):
            return None, None
        if mode == AssignmentMode.MIRROR:
            team = self.rng.choice(self.tm.pool.get(fmt))
            return ConstantTeambuilder(team), ConstantTeambuilder(team)
        if mode == AssignmentMode.POOL_RANDOM:
            pool = self.tm.pool.get(fmt)
            return _PoolTeambuilder(pool, self.rng), _PoolTeambuilder(pool, self.rng)
        raise NotImplementedError(f"mode {mode} not supported in runner yet")

    async def run_match(
        self,
        spec_a: Spec,
        spec_b: Spec,
        fmt: str,
        mode: AssignmentMode = AssignmentMode.MIRROR,
        n_battles: int = 10,
        ctx_a: Optional[AgentContext] = None,
        ctx_b: Optional[AgentContext] = None,
        save_replays: Optional[str] = None,
        announce: bool = False,
        turn_delay: float = 0.0,
        run_dir: Optional[str] = None,
        ext_logger=None,
    ) -> MatchResult:
        if isinstance(spec_a, ExternalAgentSpec) or isinstance(spec_b, ExternalAgentSpec):
            return await self._run_external_match(
                spec_a, spec_b, fmt, mode, n_battles, ctx_a, ctx_b,
                save_replays=save_replays, announce=announce, turn_delay=turn_delay,
                run_dir=run_dir or "runs", ext_logger=ext_logger)
        tb_a, tb_b = self._teambuilders(fmt, mode)
        agent_a = spec_a.build(ctx_a or AgentContext())
        agent_b = spec_b.build(ctx_b or AgentContext())
        # Showdown allows only ONE challenge between a given pair of usernames at a
        # time, so a single pairing's battles serialize. Real concurrency comes from
        # running many *different* pairings in parallel (tournament layer).
        pa = AgentPlayer(agent_a, battle_format=fmt, team=tb_a,
                         account_configuration=unique_account(spec_a.name),
                         save_replays=save_replays or False,
                         announce=announce, turn_delay=turn_delay,
                         max_concurrent_battles=1)
        pb = AgentPlayer(agent_b, battle_format=fmt, team=tb_b,
                         account_configuration=unique_account(spec_b.name),
                         turn_delay=turn_delay,
                         max_concurrent_battles=1)
        await pa.battle_against(pb, n_battles=n_battles)

        turns = [b.turn for b in pa.battles.values() if b.finished]
        res = MatchResult(
            fmt=fmt, mode=mode.value, a=spec_a.name, b=spec_b.name, n_battles=n_battles,
            a_wins=pa.n_won_battles, b_wins=pb.n_won_battles,
            ties=pa.n_finished_battles - pa.n_won_battles - pb.n_won_battles,
            turns=turns,
            avg_turns=sum(turns) / len(turns) if turns else 0.0,
        )
        res.agent_a = agent_a  # attached for metric inspection (not serialized)
        res.agent_b = agent_b
        return res

    async def _run_external_match(
        self, spec_a: Spec, spec_b: Spec, fmt: str, mode: AssignmentMode,
        n_battles: int, ctx_a, ctx_b, *, save_replays, announce, turn_delay,
        run_dir: str, ext_logger,
    ) -> MatchResult:
        from arena.external.launcher import ExternalLauncher

        external_is_a = isinstance(spec_a, ExternalAgentSpec)
        ext_spec: ExternalAgentSpec = spec_a if external_is_a else spec_b  # type: ignore[assignment]
        int_spec: AgentSpec = spec_b if external_is_a else spec_a          # type: ignore[assignment]
        int_ctx = ctx_b if external_is_a else ctx_a

        tb, _ = self._teambuilders(fmt, mode)
        agent = int_spec.build(int_ctx or AgentContext())
        our = AgentPlayer(
            agent, battle_format=fmt, team=tb,
            account_configuration=unique_account(int_spec.name),
            save_replays=save_replays or False, announce=announce,
            turn_delay=turn_delay, max_concurrent_battles=1,
            opp_logger=ext_logger, opp_name=ext_spec.name)
        our_username = our.username

        # Fresh login name per launch so a pairing never inherits orphaned battle
        # rooms from a prior pairing's connection under the same fixed name (which
        # otherwise stalls the next match). Both sides must agree on this exact name.
        eff_username = f"{ext_spec.username[:10]}{os.getpid() % 1000}{next(_account_counter)}"[:18]

        launcher = ExternalLauncher(ext_spec.cfg, ext_spec.name)
        try:
            await launcher.start(fmt, our_username, Path(run_dir), username=eff_username)
            await launcher.wait_ready()
            await self._challenge_with_watchdog(
                our, to_id_str(eff_username), n_battles, launcher,
                stall_timeout=ext_spec.cfg.stall_timeout_s)
        finally:
            await launcher.stop()

        turns = [b.turn for b in our.battles.values() if b.finished]
        res = _orient_external(
            fmt, mode.value, spec_a.name, spec_b.name, external_is_a,
            our.n_won_battles, our.n_lost_battles, our.n_finished_battles, turns)
        # attach the in-process agent to its slot for cost/behavioral accounting
        if external_is_a:
            res.agent_b = agent
        else:
            res.agent_a = agent
        return res

    @staticmethod
    async def _challenge_with_watchdog(our: AgentPlayer, opp_id: str, n_battles: int,
                                       launcher, stall_timeout: float = 420.0) -> None:
        """Send N challenges to the external username; abort if it (a) dies, or
        (b) stalls -- alive but no battle finishes within ``stall_timeout`` (a hung
        external would otherwise block ``send_challenges`` on its queue forever)."""
        import time

        send = asyncio.ensure_future(our.send_challenges(opp_id, n_battles))

        async def watchdog() -> None:
            last_done = our.n_finished_battles
            last_t = time.monotonic()
            while not send.done():
                if not launcher.is_alive():
                    send.cancel()
                    return
                done = our.n_finished_battles
                if done != last_done:
                    last_done, last_t = done, time.monotonic()
                elif time.monotonic() - last_t > stall_timeout:
                    send.cancel()  # alive but hung -> stop waiting
                    return
                await asyncio.sleep(1.0)

        wd = asyncio.ensure_future(watchdog())
        budget = 120 + n_battles * 600  # absolute ceiling on top of the stall detector
        try:
            await asyncio.wait_for(send, timeout=budget)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass  # partial results: completed battles already counted on `our`
        finally:
            wd.cancel()
