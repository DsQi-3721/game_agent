"""Adapter bridging a framework-agnostic ``Agent`` onto a poke-env ``Player``.

Written once, reused by every agent. The Player owns the network identity
(account, server, concurrency); the Agent owns the decision logic. ``choose_move``
returns a coroutine — poke-env awaits awaitable return values.

Watch mode: ``announce`` prints each battle's spectate URL when it starts (open it
in a logged-in browser at the same server to watch live); ``turn_delay`` slows each
turn so fast (non-LLM) battles are watchable in real time.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.player import Player

from arena.agents.base import Agent


_OPP_ACTION_VERBS = ("move", "switch", "drag", "-terastallize", "-mega", "-dynamax")


def opponent_actions(split_messages, opponent_role: str) -> list[str]:
    """Pure: pull the opponent's per-turn actions out of a batch of battle-log
    lines (e.g. ``|move|p2a: Pikachu|Thunderbolt`` -> ``"move Thunderbolt"``)."""
    out: list[str] = []
    if not opponent_role:
        return out
    for event in split_messages:
        if len(event) < 3 or event[1] not in _OPP_ACTION_VERBS:
            continue
        if not event[2].startswith(opponent_role):
            continue
        action = event[3] if len(event) > 3 else event[1]
        verb = "switch" if event[1] in ("switch", "drag") else event[1]
        out.append(f"{verb} {action}")
    return out


class AgentPlayer(Player):
    def __init__(self, agent: Agent, *, announce: bool = False,
                 turn_delay: float = 0.0, watch_url: str = "http://localhost:8000",
                 opp_logger=None, opp_name: str = "opponent",
                 **player_kwargs):
        self.agent = agent
        self._announce = announce
        self._turn_delay = turn_delay
        self._watch_url = watch_url
        self._announced: set[str] = set()
        # grey-box capture of an EXTERNAL opponent's per-turn move (from the public
        # battle log); None for ordinary in-process opponents.
        self._opp_logger = opp_logger
        self._opp_name = opp_name
        super().__init__(**player_kwargs)

    def choose_move(self, battle: AbstractBattle):
        return self._decide(battle)

    async def _decide(self, battle: AbstractBattle):
        if self._announce and battle.battle_tag not in self._announced:
            self._announced.add(battle.battle_tag)
            print(f"[watch] {self._watch_url}/{battle.battle_tag}", flush=True)
        order = await self.agent.act(battle)
        if self._turn_delay:
            await asyncio.sleep(self._turn_delay)
        return order

    async def _handle_battle_message(self, split_messages):
        await super()._handle_battle_message(split_messages)
        if self._opp_logger is None:
            return
        # find the battle these messages belong to (first line carries >tag)
        try:
            tag = split_messages[0][0].replace(">", "").strip() if split_messages and split_messages[0] else None
            battle = self.battles.get(tag) if tag else None
        except Exception:  # noqa: BLE001
            battle = None
        if battle is None or battle.opponent_role is None:
            return
        for parsed in opponent_actions(split_messages, battle.opponent_role):
            try:
                self._opp_logger.log_turn(
                    battle_tag=battle.battle_tag, turn=battle.turn, state_text="",
                    prompt="", raw_llm_output="", tool_calls=[],
                    parsed_order=parsed, fallback_used=False, note="external",
                )
            except Exception:  # noqa: BLE001  (logging must never break a battle)
                pass

    def _battle_finished_callback(self, battle: AbstractBattle):
        super()._battle_finished_callback(battle)
        try:
            self.agent.on_battle_end(battle)
        except Exception:  # noqa: BLE001  (an agent hook must never break the client)
            pass

    def teampreview(self, battle: AbstractBattle):
        return self._teampreview(battle)

    async def _teampreview(self, battle: AbstractBattle) -> str:
        order = await self.agent.choose_team(battle)
        # None -> agent has no preference; use poke-env's random ordering.
        return order if order is not None else self.random_teampreview(battle)
