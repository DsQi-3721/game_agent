"""Framework-agnostic Agent abstraction.

An Agent only decides; it knows nothing about the network or the async PS client.
`observation` is simply the poke-env ``Battle`` object. A thin ``AgentPlayer``
adapter (see ``adapter.py``) bridges any Agent onto poke-env's ``Player`` so the
same decision code can run live, while remaining unit-testable offline against a
recorded ``Battle``.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

if TYPE_CHECKING:  # forward refs for components added in later milestones
    from arena.llm.client import LLMClient
    from arena.logging.trajectory import TrajectoryLogger
    from arena.state.renderer import StateRenderer


@dataclass
class AgentContext:
    """Shared collaborators handed to an Agent.

    All optional: baselines need none of it; LLM agents (M3+) get ``renderer`` /
    ``llm`` / ``logger`` / ``augmentations`` populated. ``rng`` gives each agent a
    reproducible random source seeded from the experiment config.
    """

    rng: random.Random = field(default_factory=random.Random)
    renderer: Optional["StateRenderer"] = None
    llm: Optional["LLMClient"] = None
    logger: Optional["TrajectoryLogger"] = None
    full_info: bool = False
    augmentations: Optional[object] = None  # AugmentationRegistry, added in M4
    config: dict = field(default_factory=dict)


class Agent(ABC):
    """Decide one action given the current battle. Network-unaware."""

    #: human-readable id used in tournament reports / trajectory logs
    name: str = "agent"

    def __init__(self, ctx: Optional[AgentContext] = None):
        self.ctx: AgentContext = ctx or AgentContext()

    @abstractmethod
    async def act(self, battle: AbstractBattle) -> BattleOrder:
        """Return a legal order for this turn."""

    async def choose_team(self, battle: AbstractBattle) -> Optional[str]:
        """Team-preview order, or ``None`` to let the adapter pick a default.

        Returning ``None`` means "no preference"; ``AgentPlayer`` then uses
        poke-env's random team-preview. LLM/team-building agents (M3+) override
        this. Random battles never call it.
        """
        return None

    def on_battle_end(self, battle: AbstractBattle) -> None:
        """Called once when a battle finishes (win or tie). Default: no-op.

        Synchronous because poke-env's finish callback is; stateful agents use it to
        release per-battle state. The battle may end while an ``act()`` for it is
        still awaiting the LLM, so implementations must not tear down state that a
        running decision still uses.
        """
