"""Speed check: who moves first, with the opponent's speed as an estimated band.

Accounts for stat stages, paralysis, Choice Scarf (known, or its probability from
the opponent's set prior), Iron Ball, Tailwind, Trick Room, Protosynthesis / Quark Drive
Speed boosts, and Speed abilities under the current field (Swift Swim, Chlorophyll, Sand Rush,
Slush Rush, Surge Surfer, Quick Feet, Unburden, Slow Start; off under Neutralizing Gas): the
opponent's unknown ability widens its band to every ability it may have. Lists priority moves.
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.data import to_id_str

from arena.augment.base import Augmentation, AugmentationResult, Tool, function_schema
from arena.augment.calc_util import boost_mult, known_or_likely_moves, mon_names, resolve_mon
from arena.knowledge.estimate import is_own
from arena.knowledge.opponent_model import DEFAULT_MODEL
from arena.registry.augments import register_augment


_WEATHER_SPEED = {"swiftswim": ("RAINDANCE", "PRIMORDIALSEA"), "chlorophyll": ("SUNNYDAY", "DESOLATELAND"),
                  "sandrush": ("SANDSTORM",), "slushrush": ("SNOWSCAPE", "HAIL")}


def _ability_mult(battle: AbstractBattle, mon, ability: Optional[str]) -> float:
    """Speed modifiers from the ability (Showdown data/abilities.ts onModifySpe), given the field now."""
    if not ability or Field.NEUTRALIZING_GAS in (battle.fields or {}):
        return 1.0
    weather = {w.name for w in (battle.weather or {})}
    if ability in _WEATHER_SPEED and weather & set(_WEATHER_SPEED[ability]) and mon.item != "utilityumbrella":
        return 2.0
    if ability == "surgesurfer" and Field.ELECTRIC_TERRAIN in (battle.fields or {}):
        return 2.0
    if ability == "quickfeet" and mon.status is not None:
        return 1.5
    if ability == "unburden" and _lost_item(battle, mon):
        return 2.0
    if ability == "slowstart" and Effect.SLOW_START in (mon.effects or {}):
        return 0.5
    return 1.0


def _transformed_into_own(battle: AbstractBattle, mon):
    """The own Pokemon an opponent copied with Transform / Imposter (its Speed is then ours), if any."""
    from arena.observation.tracker import mon_key, tracker_for
    if not getattr(mon, "transformed", False):
        return None
    ident = next((k for k, v in battle.opponent_team.items() if v is mon), None)
    h = tracker_for(battle).mons.get(mon_key(ident)) if ident else None
    if not (h and h.temp_forme):
        return None
    return next((v for v in battle.team.values() if v.species == to_id_str(h.temp_forme)), None)


def _lost_item(battle: AbstractBattle, mon) -> bool:
    """Unburden is on only after an item is lost while this Pokemon is on the field (it ends on switching)."""
    from arena.observation.tracker import mon_key, tracker_for
    ident = next((k for pool in (battle.team, battle.opponent_team) for k, v in pool.items() if v is mon), None)
    if not ident or mon.item:
        return False
    tr = tracker_for(battle)
    h = tr.mons.get(mon_key(ident))
    return bool(h and h.lost_item and h.lost_seq is not None and h.lost_seq > tr.entered.get(mon_key(ident), -1))


def _mult(battle: AbstractBattle, mon, scarf: bool, ability: Optional[str] = None) -> float:
    """Every Speed multiplier but the ability's unless ``ability`` is given (own Pokemon: the real one)."""
    m = boost_mult((mon.boosts or {}).get("spe", 0))
    if mon.status == Status.PAR and (ability or mon.ability) != "quickfeet":
        m *= 0.5
    if scarf:
        m *= 1.5
    if mon.item == "ironball":
        m *= 0.5
    side = battle.side_conditions if is_own(battle, mon) else battle.opponent_side_conditions
    if SideCondition.TAILWIND in (side or {}):
        m *= 2
    effects = mon.effects or {}
    if Effect.QUARKDRIVESPE in effects or Effect.PROTOSYNTHESISSPE in effects:
        m *= 1.5
    return m * _ability_mult(battle, mon, ability)


def speed_band(battle: AbstractBattle, mon) -> tuple[Optional[float], Optional[float], Optional[float], float, str]:
    """(low, likely, high, scarf_probability, note) of effective speed."""
    if is_own(battle, mon):
        base = (mon.stats or {}).get("spe")
        if base is None:
            return None, None, None, 0.0, ""
        scarf = mon.item == "choicescarf"
        v = base * _mult(battle, mon, scarf, ability=mon.ability)
        return v, v, v, 1.0 if scarf else 0.0, ""
    copied = _transformed_into_own(battle, mon)
    if copied is not None and (copied.stats or {}).get("spe"):
        v = copied.stats["spe"] * _mult(battle, mon, mon.item == "choicescarf", ability=copied.ability)
        return v, v, v, 1.0 if mon.item == "choicescarf" else 0.0, f"transformed into your {copied.species}"
    est = DEFAULT_MODEL.estimate(battle, mon)
    if est is None:
        return None, None, None, 0.0, "unknown species"
    p_scarf = 0.0
    if mon.item == "choicescarf":
        p_scarf = 1.0
    elif mon.item == "unknown_item":           # None = known to hold nothing
        prior = DEFAULT_MODEL.set_prior(battle, mon)
        p_scarf = next((c.p for c in (prior.items if prior else ()) if c.id == "choicescarf"), 0.0)
    # the ability: revealed, or every one it may have (a Swift Swim user in rain may be twice as fast)
    abilities = [mon.ability] if mon.ability else sorted(getattr(mon, "possible_abilities", None) or [None])
    mults = [_mult(battle, mon, p_scarf >= 1.0, ability=a) for a in abilities] or [_mult(battle, mon, p_scarf >= 1.0)]
    low, likely, high = est.low["spe"] * min(mults), est.stats["spe"] * min(mults), est.high["spe"] * max(mults)
    note = est.note
    if max(mults) > min(mults):
        fast = [a for a, m in zip(abilities, mults) if m == max(mults)]
        note = (note + "; " if note else "") + f"faster if its ability is {'/'.join(fast)}"
    if 0.0 < p_scarf < 1.0:
        high = max(high, est.high["spe"] * max(_mult(battle, mon, True, ability=a) for a in abilities))
    return low, likely, high, p_scarf, note


def speed_limits(battle: AbstractBattle, mon) -> tuple[Optional[float], Optional[float]]:
    """Every Speed the opponent's Pokemon could legally have now -- any spread (random battles: fixed EVs and
    nature, any IVs), any of its possible abilities, a Choice Scarf or Iron Ball unless its item is known.
    Own Pokemon: exact. A "moves first / second" verdict is certain only outside this range."""
    if is_own(battle, mon) or _transformed_into_own(battle, mon) is not None:
        v = speed_band(battle, mon)[1]
        return v, v
    from arena.knowledge.estimate import _compute
    gen = getattr(battle, "gen", 9) or 9
    level = int(getattr(mon, "level", 100) or 100)
    random_fmt = "randombattle" in (getattr(battle, "format", "") or "")
    lo_spread = {"evs": [85] * 6 if random_fmt else [0] * 6, "ivs": [0] * 6,
                 "nature": "hardy" if random_fmt else "brave"}            # brave: -Spe
    hi_spread = {"evs": [85] * 6 if random_fmt else [252] * 6, "ivs": [31] * 6,
                 "nature": "hardy" if random_fmt else "jolly"}            # jolly: +Spe
    lo_raw, hi_raw = _compute(mon.species, level, lo_spread, gen), _compute(mon.species, level, hi_spread, gen)
    if lo_raw is None or hi_raw is None:
        return None, None
    abilities = [mon.ability] if mon.ability else sorted(getattr(mon, "possible_abilities", None) or [None])
    item_known = mon.item != "unknown_item"     # None = known to hold nothing
    scarf = [mon.item == "choicescarf"] if item_known else [False, True]
    lows, highs = [], []
    for a in abilities:
        for sc in scarf:
            m = _mult(battle, mon, sc, ability=a)
            if not item_known:
                lows.append(m * 0.5)                                       # an Iron Ball is legal too
            lows.append(m)
            highs.append(m)
    return lo_raw["spe"] * min(lows), hi_raw["spe"] * max(highs)


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
