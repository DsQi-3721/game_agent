"""Type-matchup augmentation: move effectiveness + defensive typing summary.

Tool ``type_matchup``: effectiveness of any attacking type or move on any Pokemon.
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon_type import PokemonType

from arena.augment.base import Augmentation, AugmentationResult, Tool, function_schema
from arena.augment.calc_util import mon_names, resolve_mon, resolve_move
from arena.registry.augments import register_augment

_TAGS = {0: "immune", 0.25: "double-resist", 0.5: "resisted", 1: "neutral", 2: "super-effective",
         4: "double-super"}


def _types(mon) -> str:
    return "/".join(t.name.title() for t in mon.types if t is not None)


class TypeMatchupTool(Tool):
    name = "type_matchup"

    @property
    def schema(self) -> dict:
        return function_schema(
            self.name,
            "Type effectiveness multiplier of an attacking type (e.g. 'Ground') or a move (e.g. "
            "'Earthquake') against a Pokemon (yours or the opponent's).",
            {"attack": {"type": "string", "description": "attacking type or move name"},
             "defender": {"type": "string", "description": "defending Pokemon (default: opponent's active)"}},
            required=["attack"],
        )

    async def call(self, battle: AbstractBattle, arguments: dict) -> str:
        dfn = resolve_mon(battle, arguments.get("defender"), default="theirs")
        if dfn is None:
            return f"type_matchup: unknown defender. Valid names -- {mon_names(battle)}"
        attack = str(arguments.get("attack") or "").strip()
        try:
            ptype = PokemonType.from_name(attack)
            label = ptype.name.title()
        except Exception:  # noqa: BLE001  (not a type name -> try a move)
            mv = resolve_move(battle, None, attack)
            if mv is None:
                return f"type_matchup: {attack!r} is neither a type nor a move."
            ptype, label = mv.type, f"{mv.id} ({mv.type.name.title()})"
        m = dfn.damage_multiplier(ptype)
        return f"{label} vs {dfn.species} [{_types(dfn)}]: x{m} {_TAGS.get(m, '')}".rstrip()


@register_augment("type_matchup")
class TypeMatchupAugmentation(Augmentation):
    name = "type_matchup"

    def compute(self, battle: AbstractBattle) -> AugmentationResult:
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if me is None or opp is None:
            return AugmentationResult(self.name, "")

        lines = [f"Typing: you {_types(me)} vs opponent {_types(opp)}."]

        if battle.available_moves:
            lines.append("Your move effectiveness on the opponent:")
            for mv in battle.available_moves:
                m = opp.damage_multiplier(mv)
                lines.append(f"  - {mv.id} ({mv.type.name}): x{m} {_TAGS.get(m, f'x{m}')}")

        # defensive: how do the opponent's revealed move types fare on us
        opp_move_types = {mv.type for mv in opp.moves.values() if mv.type is not None}
        if opp_move_types:
            dvals = {t.name: me.damage_multiplier(t) for t in opp_move_types}
            threats = [f"{name} x{v}" for name, v in dvals.items() if v >= 2]
            if threats:
                lines.append("Opponent's revealed types that threaten you: " + ", ".join(threats))

        return AugmentationResult(self.name, "\n".join(lines))

    def as_tool(self) -> Optional[Tool]:
        return TypeMatchupTool()
