"""Unified augmentation abstraction.

Each augmentation is implemented ONCE; config decides its mode per item:
  - OFF: not used
  - PASSIVE: computed and injected into the prompt automatically
  - TOOL: exposed as a callable the agent may invoke (ReAct, M5)

This lets us study "information given" vs "information the agent must seek".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon


class AugmentMode(str, Enum):
    OFF = "off"
    PASSIVE = "passive"
    TOOL = "tool"


@dataclass
class AugmentationResult:
    name: str
    text: str
    data: dict = field(default_factory=dict)


class Tool(ABC):
    name: str

    @property
    @abstractmethod
    def schema(self) -> dict:
        """OpenAI-style function schema."""

    @abstractmethod
    async def call(self, battle: AbstractBattle, arguments: dict) -> str:
        ...


class Augmentation(ABC):
    name: str

    @abstractmethod
    def compute(self, battle: AbstractBattle) -> AugmentationResult:
        """Passive view: deterministic text derived from the battle."""

    def as_tool(self) -> Optional[Tool]:
        """Override to expose this augmentation as an agent-callable tool."""
        return None


def function_schema(name: str, description: str, properties: Optional[dict] = None,
                    required: Optional[list] = None) -> dict:
    """OpenAI-style function schema (shared by all tools)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {},
                           "required": list(required or [])},
        },
    }


def identifier_of(battle: AbstractBattle, mon: Pokemon) -> Optional[str]:
    """Reverse-lookup the Showdown identifier (e.g. 'p1: Passimian') for a mon."""
    for pool in (battle.team, battle.opponent_team):
        for ident, p in pool.items():
            if p is mon:
                return ident
    return None


class AugmentationRegistry:
    def __init__(self, augmentations: list[Augmentation], modes: dict[str, AugmentMode]):
        self._augs = {a.name: a for a in augmentations}
        self._modes = {k: AugmentMode(v) for k, v in modes.items()}

    def mode(self, name: str) -> AugmentMode:
        return self._modes.get(name, AugmentMode.OFF)

    def passive(self, battle: AbstractBattle) -> list[AugmentationResult]:
        out: list[AugmentationResult] = []
        for name, aug in self._augs.items():
            if self.mode(name) != AugmentMode.PASSIVE:
                continue
            try:
                res = aug.compute(battle)
            except Exception as e:  # noqa: BLE001  (augmentations must never crash a turn)
                res = AugmentationResult(name, f"(unavailable: {type(e).__name__})")
            if res and res.text:
                out.append(res)
        return out

    def tools(self) -> list[Tool]:
        tools = []
        for name, aug in self._augs.items():
            if self.mode(name) == AugmentMode.TOOL:
                t = aug.as_tool()
                if t is not None:
                    tools.append(t)
        return tools

    @classmethod
    def from_config(cls, modes: dict[str, str]) -> "AugmentationRegistry":
        from arena.registry.augments import entries

        augs = [aug_cls() for aug_cls in entries().values()]
        return cls(augs, modes)
