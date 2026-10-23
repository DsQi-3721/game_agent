"""Non-LLM baseline agents — thin wrappers over poke-env's built-in heuristics.

These are the control anchors every evaluation is measured against. They delegate
to poke-env's static helpers, so no ``Player`` instance is required and they unit
test offline against a recorded ``Battle``.
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from arena.agents.base import Agent, AgentContext
from arena.registry.agents import register_agent


@register_agent("random", label="Random")
class RandomAgent(Agent):
    name = "random"

    async def act(self, battle: AbstractBattle) -> BattleOrder:
        return Player.choose_random_move(battle)


@register_agent("max_base_power", label="Max Base Power")
class MaxBasePowerAgent(Agent):
    name = "max_base_power"

    async def act(self, battle: AbstractBattle) -> BattleOrder:
        return MaxBasePowerPlayer.choose_singles_move(battle)


@register_agent("simple_heuristics", label="Simple Heuristics")
class SimpleHeuristicsAgent(Agent):
    name = "simple_heuristics"

    async def act(self, battle: AbstractBattle) -> BattleOrder:
        order, _score = SimpleHeuristicsPlayer.choose_singles_move(battle)
        return order


#: name -> factory, used by AgentRegistry (M3) and offline tests
BASELINES = {
    RandomAgent.name: RandomAgent,
    MaxBasePowerAgent.name: MaxBasePowerAgent,
    SimpleHeuristicsAgent.name: SimpleHeuristicsAgent,
}


def make_baseline(name: str, ctx: Optional[AgentContext] = None) -> Agent:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; have {sorted(BASELINES)}")
    return BASELINES[name](ctx)
