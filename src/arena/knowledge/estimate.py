"""Stat estimates for Pokemon whose stats are hidden (the opponent's).

Randbats sets are near-deterministic (level shown in the protocol, 85 EVs / 31 IVs,
neutral nature), so estimates are essentially exact. For usage-based formats the
likely value comes from the most common spread and the low/high band spans the
common spreads (or generic 0/252 EV bounds when nothing is known).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from poke_env.data import GenData, to_id_str
from poke_env.stats import compute_raw_stats

from arena.knowledge.priors import SetPrior

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
_NEUTRAL = {"evs": [85] * 6, "ivs": [31] * 6, "nature": "hardy", "p": 1.0}


@dataclass(frozen=True)
class StatEstimate:
    stats: dict            # likely values, incl. "hp"
    low: dict
    high: dict
    exact: bool
    note: str              # e.g. "exact (your Pokemon)" / "est. L80 85EV"


def _compute(species: str, level: int, spread: dict, gen: int) -> Optional[dict]:
    data = GenData.from_gen(gen)
    sid = species if species in data.pokedex else to_id_str(species)
    if sid not in data.pokedex:
        return None
    nature = (spread.get("nature") or "hardy").lower()
    if nature not in data.natures:
        nature = "hardy"
    vals = compute_raw_stats(sid, list(spread.get("evs") or _NEUTRAL["evs"]),
                             list(spread.get("ivs") or _NEUTRAL["ivs"]), level, nature, data)
    return dict(zip(STATS, vals))


def _bound(species: str, level: int, gen: int, ev: int, mult: float) -> Optional[dict]:
    base = _compute(species, level, {"evs": [ev] * 6, "ivs": [31] * 6, "nature": "hardy"}, gen)
    if base is None:
        return None
    return {k: (v if k == "hp" else math.floor(v * mult)) for k, v in base.items()}


def is_own(battle, mon) -> bool:
    return any(p is mon for p in battle.team.values())


def estimate_stats(battle, mon, prior: Optional[SetPrior] = None) -> Optional[StatEstimate]:
    gen = getattr(battle, "gen", 9) or 9
    if is_own(battle, mon):
        st = dict(mon.stats or {})
        if mon.max_hp:
            st["hp"] = mon.max_hp
        if all(isinstance(st.get(k), (int, float)) for k in STATS):
            return StatEstimate(st, st, st, True, "exact (your Pokemon)")
    level = int(getattr(mon, "level", 100) or 100)
    species = mon.species
    spreads = list(prior.spreads) if prior and prior.spreads else []
    if spreads:
        likely = _compute(species, level, spreads[0], gen)
        common = [s for s in spreads if s.get("p", 0) >= 0.05] or spreads[:1]
        vals = [v for v in (_compute(species, level, s, gen) for s in common) if v]
        if likely is None or not vals:
            return None
        low = {k: min(v[k] for v in vals) for k in STATS}
        high = {k: max(v[k] for v in vals) for k in STATS}
        exact = len(common) == 1 and common[0].get("p", 0) >= 0.95
        ev = spreads[0].get("evs") or []
        note = (f"est. L{level} {ev[0]}EV" if exact and len(set(ev)) == 1
                else f"est. L{level} {spreads[0].get('nature', 'hardy')} {'/'.join(map(str, ev))}")
        return StatEstimate(likely, low, high, exact, note)
    likely = _compute(species, level, _NEUTRAL, gen)
    if likely is None:
        return None
    random_fmt = "randombattle" in (getattr(battle, "format", "") or "")
    if random_fmt:
        return StatEstimate(likely, likely, likely, True, f"est. L{level} 85EV")
    low = _bound(species, level, gen, 0, 0.9) or likely
    high = _bound(species, level, gen, 252, 1.1) or likely
    return StatEstimate(likely, low, high, False, f"est. L{level} (spread unknown)")
