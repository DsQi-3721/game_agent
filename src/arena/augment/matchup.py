"""Composite matchup: one call answers "how does this 1v1 look?".

Damage both ways (our moves on them, their known/likely moves on us), speed
order and immunities. Fewer, richer tools mean fewer executor turns for a small
model; the fine-grained tools stay available for targeted questions.
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle

from arena.augment.base import Augmentation, AugmentationResult, Tool, function_schema
from arena.augment.calc_util import known_or_likely_moves, mon_names, resolve_mon
from arena.augment.damage import damage_lines, tera_for
from arena.augment.speed import speed_text
from arena.registry.augments import register_augment
from arena.state.observe import hp_pct


def matchup_text(battle: AbstractBattle, mine, theirs) -> str:
    lines = [f"== your {mine.species} ({hp_pct(mine)}%) vs opp {theirs.species} ({hp_pct(theirs)}%) =="]
    moves = list(battle.available_moves) if mine is battle.active_pokemon and battle.available_moves else None
    ours = damage_lines(battle, mine, theirs, moves)
    lines.append("Your moves on it: " + ("; ".join(
        f"{d.move} {d.pct_lo:.0f}-{d.pct_hi:.0f}%" + (f" ({d.ko_lo}HKO)" if d.ko_lo and d.ko_lo == d.ko_hi else "")
        for d in ours) or "no damaging moves"))
    if battle.can_tera and mine is battle.active_pokemon:
        tera = tera_for(battle, mine)
        if tera:
            with_tera = damage_lines(battle, mine, theirs, moves, attacker_tera=tera)[:2]
            if with_tera:
                lines.append(f"If you terastallize ({tera}): " + "; ".join(
                    f"{d.move} {d.pct_lo:.0f}-{d.pct_hi:.0f}%" for d in with_tera))
    immune = [m.id for m in (moves or known_or_likely_moves(battle, mine))
              if m.base_power > 0 and theirs.damage_multiplier(m) == 0]
    if immune:
        lines.append("It is immune to: " + ", ".join(immune))
    theirs_hits = damage_lines(battle, theirs, mine)
    lines.append("Its known/likely moves on you: " + ("; ".join(
        f"{d.move} {d.pct_lo:.0f}-{d.pct_hi:.0f}%" for d in theirs_hits) or "nothing damaging known"))
    lines.append(speed_text(battle, mine, theirs))
    return "\n".join(lines)


class MatchupTool(Tool):
    name = "matchup"

    @property
    def schema(self) -> dict:
        return function_schema(
            self.name,
            "Full 1v1 summary between one of your Pokemon and one of the opponent's: damage both "
            "ways (% HP, hits to KO), immunities, and who moves first. Best first tool to call.",
            {"mine": {"type": "string", "description": "your Pokemon (default: your active)"},
             "theirs": {"type": "string", "description": "opponent's Pokemon (default: their active)"}},
        )

    async def call(self, battle: AbstractBattle, arguments: dict) -> str:
        mine = resolve_mon(battle, arguments.get("mine"), default="mine", side="mine")
        theirs = resolve_mon(battle, arguments.get("theirs"), default="theirs", side="theirs")
        if mine is None or theirs is None:
            return (f"matchup: unknown or unrevealed Pokemon (an opponent's Pokemon must have been sent out; "
                    f"use opponent_sets for unrevealed ones). Valid names -- {mon_names(battle)}")
        return matchup_text(battle, mine, theirs)


@register_augment("matchup")
class MatchupAugmentation(Augmentation):
    name = "matchup"

    def compute(self, battle: AbstractBattle) -> AugmentationResult:
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        if me is None or opp is None:
            return AugmentationResult(self.name, "")
        return AugmentationResult(self.name, matchup_text(battle, me, opp))

    def as_tool(self) -> Optional[Tool]:
        return MatchupTool()
