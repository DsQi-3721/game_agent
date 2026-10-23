"""Speed check: who moves first, with the opponent's speed as an estimated band.

Accounts for stat stages, paralysis, Choice Scarf (known, or its probability from
the opponent's set prior), Tailwind, Trick Room, and lists priority moves.
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status

from arena.augment.base import Augmentation, AugmentationResult, Tool, function_schema
from arena.augment.calc_util import boost_mult, known_or_likely_moves, mon_names, resolve_mon
from arena.knowledge.estimate import is_own
from arena.knowledge.opponent_model import DEFAULT_MODEL
from arena.registry.augments import register_augment


def _mult(battle: AbstractBattle, mon, scarf: bool) -> float:
    m = boost_mult((mon.boosts or {}).get("spe", 0))
    if mon.status == Status.PAR and mon.ability != "quickfeet":
        m *= 0.5
    if scarf:
        m *= 1.5
    side = battle.side_conditions if is_own(battle, mon) else battle.opponent_side_conditions
    if SideCondition.TAILWIND in (side or {}):
        m *= 2
    return m


def speed_band(battle: AbstractBattle, mon) -> tuple[Optional[float], Optional[float], Optional[float], float, str]:
    """(low, likely, high, scarf_probability, note) of effective speed."""
    if is_own(battle, mon):
        base = (mon.stats or {}).get("spe")
        if base is None:
            return None, None, None, 0.0, ""
        scarf = mon.item == "choicescarf"
        v = base * _mult(battle, mon, scarf)
        return v, v, v, 1.0 if scarf else 0.0, ""
    est = DEFAULT_MODEL.estimate(battle, mon)
    if est is None:
        return None, None, None, 0.0, "unknown species"
    p_scarf = 0.0
    if mon.item == "choicescarf":
        p_scarf = 1.0
    elif mon.item in (None, "", "unknown_item"):
        prior = DEFAULT_MODEL.set_prior(battle, mon)
        p_scarf = next((c.p for c in (prior.items if prior else ()) if c.id == "choicescarf"), 0.0)
    m = _mult(battle, mon, p_scarf >= 1.0)
    low, likely, high = est.low["spe"] * m, est.stats["spe"] * m, est.high["spe"] * m
    if 0.0 < p_scarf < 1.0:
        high = est.high["spe"] * _mult(battle, mon, True)
    return low, likely, high, p_scarf, est.note


def _priority(battle, mon) -> list[str]:
    """Damaging priority moves only (Protect & co. do not change who hits first)."""
    return [f"{m.id} ({m.priority:+d})" for m in known_or_likely_moves(battle, mon, k=6)
            if m.priority > 0 and m.base_power > 0]


def speed_text(battle: AbstractBattle, mine, theirs) -> str:
    a_lo, a, a_hi, _, _ = speed_band(battle, mine)
    b_lo, b, b_hi, p_scarf, note = speed_band(battle, theirs)
    if a is None or b is None:
        return f"speed: unknown for {mine.species if a is None else theirs.species}"
    tr = Field.TRICK_ROOM in (battle.fields or {})
    band = f"{b:.0f}" if b_lo == b_hi else f"{b_lo:.0f}-{b_hi:.0f} (likely {b:.0f})"
    scarf = f", Choice Scarf {round(p_scarf * 100)}%" if 0 < p_scarf < 1 else ""
    if tr:
        verdict = "you move first (Trick Room)" if a < b_lo else ("opponent moves first (Trick Room)" if a > b_hi else "unclear")
    elif a > b_hi:
        verdict = "you move first"
    elif a < b_lo:
        verdict = "opponent moves first"
    elif a > b:
        verdict = "you are probably faster"
    elif a < b:
        verdict = "opponent is probably faster"
    else:
        verdict = "speed tie likely"
    lines = [f"Speed: your {mine.species} {a:.0f} vs opp {theirs.species} {band}{scarf} -> {verdict}"
             + (f" [{note}]" if note else "")]
    pr_opp, pr_me = _priority(battle, theirs), _priority(battle, mine)
    if pr_opp:
        lines.append(f"Opponent priority moves (known/likely): {', '.join(pr_opp)}")
    if pr_me:
        lines.append(f"Your priority moves: {', '.join(pr_me)}")
    return "\n".join(lines)


class SpeedTool(Tool):
    name = "speed_check"

    @property
    def schema(self) -> dict:
        return function_schema(
            self.name,
            "Compare effective Speed (stages, paralysis, Choice Scarf odds, Tailwind, Trick Room) of "
            "one of your Pokemon vs one of the opponent's, and list priority moves.",
            {"mine": {"type": "string", "description": "your Pokemon (default: your active)"},
             "theirs": {"type": "string", "description": "opponent's Pokemon (default: their active)"}},
        )

    async def call(self, battle: AbstractBattle, arguments: dict) -> str:
        mine = resolve_mon(battle, arguments.get("mine"), default="mine", side="mine")
        theirs = resolve_mon(battle, arguments.get("theirs"), default="theirs", side="theirs")
        if mine is None or theirs is None:
            return (f"speed_check: unknown or unrevealed Pokemon (an opponent's Pokemon must have been sent out; "
                    f"use opponent_sets for unrevealed ones). Valid names -- {mon_names(battle)}")
        return speed_text(battle, mine, theirs)


@register_augment("speed_check")
class SpeedAugmentation(Augmentation):
    name = "speed_check"

    def compute(self, battle: AbstractBattle) -> AugmentationResult:
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        if me is None or opp is None:
            return AugmentationResult(self.name, "")
        return AugmentationResult(self.name, speed_text(battle, me, opp))

    def as_tool(self) -> Optional[Tool]:
        return SpeedTool()
