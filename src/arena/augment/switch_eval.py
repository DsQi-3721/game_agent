"""Switch evaluation: what each possible switch-in faces right now.

Per candidate: entry-hazard damage (Stealth Rock by Rock effectiveness, Spikes
layers, Toxic Spikes, Sticky Web; Heavy-Duty Boots / Magic Guard / grounding
respected), the opponent active's worst known/likely hit on it, and speed order.
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition

from arena.augment.base import Augmentation, AugmentationResult, Tool, function_schema
from arena.augment.calc_util import mon_names, resolve_mon
from arena.augment.damage import damage_lines
from arena.augment.speed import speed_band
from arena.knowledge.estimate import is_own
from arena.registry.augments import register_augment
from arena.state.observe import hp_pct

_SPIKES = {1: 12.5, 2: 100 / 6, 3: 25.0}


def _grounded(mon) -> bool:
    return (PokemonType.FLYING not in mon.types and mon.ability != "levitate"
            and mon.item != "airballoon")


def hazard_text(battle: AbstractBattle, mon) -> tuple[float, str]:
    """(entry damage %, description) for switching ``mon`` in on our side."""
    side = battle.side_conditions or {}
    if mon.item == "heavydutyboots" or mon.ability == "magicguard":
        immune = "Heavy-Duty Boots" if mon.item == "heavydutyboots" else "Magic Guard"
        return 0.0, (f"no hazard damage ({immune})" if side else "no hazards")
    dmg, parts = 0.0, []
    if SideCondition.STEALTH_ROCK in side:
        sr = 12.5 * mon.damage_multiplier(PokemonType.ROCK)
        dmg += sr
        parts.append(f"Stealth Rock -{sr:.0f}%")
    if _grounded(mon):
        layers = side.get(SideCondition.SPIKES, 0) or 0
        if layers:
            dmg += _SPIKES.get(min(int(layers), 3), 25.0)
            parts.append(f"Spikes x{layers} -{_SPIKES.get(min(int(layers), 3), 25.0):.0f}%")
        if SideCondition.TOXIC_SPIKES in side and not ({PokemonType.POISON, PokemonType.STEEL} & set(mon.types)):
            parts.append("Toxic Spikes poisons")
        if SideCondition.STICKY_WEB in side:
            parts.append("Sticky Web -1 Spe")
    return dmg, (", ".join(parts) if parts else "no hazards")


def switch_line(battle: AbstractBattle, cand, opp) -> str:
    dmg, hz = hazard_text(battle, cand)
    bits = [f"{cand.species} {hp_pct(cand)}%", f"entry: {hz}"]
    if opp is not None:
        hits = damage_lines(battle, opp, cand)
        if hits:
            w = hits[0]
            bits.append(f"worst known/likely hit: {w.move} {w.pct_lo:.0f}-{w.pct_hi:.0f}%")
        a = speed_band(battle, cand)[1]
        b_lo, b, b_hi = speed_band(battle, opp)[:3]
        if a is not None and b is not None:
            bits.append("faster" if a > b_hi else ("slower" if a < b_lo else "speed close"))
    return " | ".join(bits)


class SwitchEvalTool(Tool):
    name = "switch_eval"

    @property
    def schema(self) -> dict:
        return function_schema(
            self.name,
            "Evaluate switching in one of your benched Pokemon (or all available switches): "
            "entry hazard damage, the opponent's worst known/likely hit on it, and speed.",
            {"candidate": {"type": "string", "description": "your benched Pokemon (omit for all switches)"}},
        )

    async def call(self, battle: AbstractBattle, arguments: dict) -> str:
        opp = battle.opponent_active_pokemon
        name = arguments.get("candidate")
        if name:
            cand = resolve_mon(battle, name, default="mine", side="mine")
            if cand is None or not is_own(battle, cand):
                return f"switch_eval: {name!r} is not one of your Pokemon. Valid names -- {mon_names(battle)}"
            cands = [cand]
        else:
            cands = list(battle.available_switches)
        if not cands:
            return "switch_eval: no switches available."
        return "\n".join(switch_line(battle, c, opp) for c in cands)


@register_augment("switch_eval")
class SwitchEvalAugmentation(Augmentation):
    name = "switch_eval"

    def compute(self, battle: AbstractBattle) -> AugmentationResult:
        if not battle.available_switches:
            return AugmentationResult(self.name, "")
        opp = battle.opponent_active_pokemon
        lines = ["Switch-in outlook:"] + [f"  - {switch_line(battle, c, opp)}" for c in battle.available_switches]
        return AugmentationResult(self.name, "\n".join(lines))

    def as_tool(self) -> Optional[Tool]:
        return SwitchEvalTool()
