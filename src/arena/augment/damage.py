"""Damage-calc augmentation: estimated damage, in both directions.

Wraps poke-env's gen9 ``calculate_damage``. The opponent's hidden stats (and its
most likely item/ability) come from the opponent model (``arena.knowledge``) and
are filled only for the duration of each calculation, so results are estimates
expressed as % of the defender's max HP plus hits-to-KO from its current HP.

Passive: our moves on the opponent's active, and its revealed/likely moves on ours.
Tool ``damage_calc``: any attacker / defender / move.
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle

from arena.augment.base import Augmentation, AugmentationResult, Tool, function_schema
from arena.augment.calc_util import (
    clean_move_name,
    damage_estimate,
    known_or_likely_moves,
    mon_names,
    resolve_mon,
    resolve_move,
)
from arena.knowledge.estimate import is_own
from arena.knowledge.opponent_model import DEFAULT_MODEL
from arena.state.observe import tera_type_of
from arena.registry.augments import register_augment


def damage_lines(battle: AbstractBattle, attacker, defender, moves=None, *,
                 attacker_tera: Optional[str] = None, defender_tera: Optional[str] = None) -> list:
    """DamageEstimate per damaging move (status / failed calcs skipped)."""
    out = []
    for mv in moves if moves is not None else known_or_likely_moves(battle, attacker):
        if mv is None or (mv.base_power <= 0 and mv.id not in ("seismictoss", "nightshade")):
            continue
        d = damage_estimate(battle, attacker, defender, mv, attacker_tera=attacker_tera,
                            defender_tera=defender_tera)
        if d is not None:
            out.append(d)
    out.sort(key=lambda d: d.pct_hi, reverse=True)
    return out


def tera_for(battle: AbstractBattle, mon) -> Optional[str]:
    """Our tera type (request) or the opponent's most likely one (set prior)."""
    if mon.is_terastallized:
        return None
    if is_own(battle, mon):
        return tera_type_of(battle, mon)
    prior = DEFAULT_MODEL.set_prior(battle, mon)
    top = prior.top("tera") if prior else None
    return top.name if top else None


class DamageCalcTool(Tool):
    name = "damage_calc"

    @property
    def schema(self) -> dict:
        return function_schema(
            self.name,
            "Estimate damage as % of the defender's max HP (and hits to KO) for one move or for "
            "all of the attacker's known/likely moves. Works both ways: your Pokemon attacking the "
            "opponent's, or the opponent's attacking yours (hidden stats/items are estimated).",
            {"attacker": {"type": "string", "description": "attacking Pokemon (default: your active)"},
             "defender": {"type": "string", "description": "defending Pokemon (default: opponent's active)"},
             "move": {"type": "string", "description": "optional move name; omit to list all moves"},
             "tera": {"type": "boolean", "description": "simulate the attacker terastallizing first"},
             "defender_tera": {"type": "string", "description": "optional: simulate the defender terastallized into this type"}},
        )

    async def call(self, battle: AbstractBattle, arguments: dict) -> str:
        atk = resolve_mon(battle, arguments.get("attacker"), default="mine")
        dfn = resolve_mon(battle, arguments.get("defender"), default="theirs")
        if atk is None or dfn is None:
            return f"damage_calc: unknown Pokemon. Valid names -- {mon_names(battle)}"
        moves = None
        tera = str(arguments.get("tera", "")).lower() in ("true", "1", "yes")
        if arguments.get("move"):
            tera = tera or clean_move_name(arguments["move"])[1]
            mv = resolve_move(battle, atk, arguments["move"])
            if mv is None:
                return f"damage_calc: unknown move {arguments['move']!r}."
            moves = [mv]
        atk_tera = tera_for(battle, atk) if tera else None
        rows = damage_lines(battle, atk, dfn, moves, attacker_tera=atk_tera,
                            defender_tera=arguments.get("defender_tera") or None)
        if not rows:
            return f"damage_calc: no damaging move found for {atk.species} on {dfn.species}."
        notes = sorted({d.note for d in rows if d.note})
        return "\n".join([d.line() for d in rows] + [f"({n})" for n in notes])


@register_augment("damage_calc")
class DamageAugmentation(Augmentation):
    name = "damage_calc"

    def compute(self, battle: AbstractBattle) -> AugmentationResult:
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if me is None or opp is None:
            return AugmentationResult(self.name, "")
        ours = damage_lines(battle, me, opp, [m for m in battle.available_moves])
        theirs = damage_lines(battle, opp, me)
        if not ours and not theirs:
            return AugmentationResult(self.name, "")
        lines = ["Estimated damage (% of max HP):"]
        lines += [f"  - {d.line()}" for d in ours]
        if theirs:
            lines.append("Opponent's known/likely moves on you:")
            lines += [f"  - {d.line()}" for d in theirs]
        rows = {d.move: [round(d.pct_lo, 1), round(d.pct_hi, 1)] for d in ours}
        return AugmentationResult(self.name, "\n".join(lines), data={"damage": rows})

    def as_tool(self) -> Optional[Tool]:
        return DamageCalcTool()
