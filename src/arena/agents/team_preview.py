"""Team preview (OU): choose the lead and team order from both teams instead of poke-env's random order.

Inputs are only what is public at team preview: our own sets and the opponent's six species.

    score(ours) = log P(lead | species)  +  0.01 * mean matchup(ours, theirs) / 100

``P(lead | species)``: how often a species leads when it is brought, from 77,176 public gen9ou battles at
1600+ (PokéChamp logs, ``scripts/build_lead_stats.py`` -> ``data/knowledge/gen9ou_lead_rates.json``),
Laplace-smoothed towards 1/6. It captures team roles (hazard, web and weather leads) that a matchup score
misses: on 1,000 held-out human battles at 1600+ (T6) it picks the human's lead 37.8% of the time (top 2:
60.2%), matchup alone 18.6%, random 16.7%. The matchup term only breaks ties:

    matchup(ours, theirs) = our best estimated damage to it (% of its HP, our known moves)
                          - its best estimated damage to us (its likely moves from the usage-based set prior)
                          + 10 if we are faster, - 10 if slower (likely speeds)

The rest of the team follows in score order (in 6v6 singles only the lead is decided here). ``team_order``
returns None when the preview is incomplete; the caller then says so instead of silently picking at random.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from statistics import mean
from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon

from poke_env.data import to_id_str

from arena.augment.calc_util import damage_estimate, known_or_likely_moves
from arena.augment.speed import speed_band
from arena.knowledge import store

LEAD_RATES_FILE = "gen9ou_lead_rates.json"
MATCHUP_WEIGHT = 0.01


def lead_prior(species: str) -> float:
    """P(lead | brought), smoothed towards 1/6; 1/6 for species without statistics."""
    r = (store.load(LEAD_RATES_FILE).get("species") or {}).get(to_id_str(species)) or {}
    return (r.get("lead", 0) + 1) / (r.get("team", 0) + 6)


def _best_damage(battle: AbstractBattle, attacker: Pokemon, defender: Pokemon) -> float:
    best = 0.0
    for mv in known_or_likely_moves(battle, attacker):
        if mv is None or (mv.base_power <= 0 and mv.id not in ("seismictoss", "nightshade")):
            continue
        try:
            d = damage_estimate(battle, attacker, defender, mv)
        except Exception:  # noqa: BLE001  (an unknown forme in the calc must not break team preview)
            d = None
        if d is not None:
            best = max(best, (d.pct_lo + d.pct_hi) / 2)
    return min(best, 100.0)


def matchup(battle: AbstractBattle, ours: Pokemon, theirs: Pokemon) -> float:
    score = _best_damage(battle, ours, theirs) - _best_damage(battle, theirs, ours)
    a, b = speed_band(battle, ours)[1], speed_band(battle, theirs)[1]
    if a is not None and b is not None and a != b:
        score += 10.0 if a > b else -10.0
    return score


@contextmanager
def _previewed(battle: AbstractBattle, theirs: list[Pokemon]):
    """The damage calc looks Pokemon up in ``opponent_team``, which is empty at team preview: list the
    previewed ones there for the duration of the (synchronous) scoring, then restore it exactly."""
    saved = battle._opponent_team
    role = battle.opponent_role or ("p2" if battle.player_role == "p1" else "p1")
    battle._opponent_team = {**saved, **{f"{role}: {p.species}": p for p in theirs if p not in saved.values()}}
    try:
        yield
    finally:
        battle._opponent_team = saved


def _their_lead_weights(battle: AbstractBattle, theirs: list[Pokemon]) -> list[float]:
    """How much each of their Pokemon counts in the matchup term: half uniform, half its chance to lead (the
    team-aware opponent model's lead rates); uniform under the plain prior."""
    n = len(theirs)
    uniform = [1.0 / n] * n
    from arena.knowledge.opponent_model import DEFAULT_MODEL
    if DEFAULT_MODEL.name != "team":
        return uniform
    try:
        from arena.opponent.profile import profile_for
        prof = profile_for(battle, with_threats=False)
    except Exception:  # noqa: BLE001
        prof = None
    if prof is None:
        return uniform
    from arena.opponent.kb import kb_for, resolve_species
    kb = kb_for(battle.format)
    rates = [max((kb["species"].get(resolve_species(kb, o.species) or "", {}) or {}).get("lead", 0.0), 1e-3)
             for o in theirs]
    z = sum(rates)
    return [0.5 / n + 0.5 * r / z for r in rates]


def team_order(battle: AbstractBattle) -> Optional[tuple[list[int], list[float]]]:
    """(1-based team positions in the chosen order, score per position) or None."""
    ours = list(battle.team.values())
    theirs = list(getattr(battle, "teampreview_opponent_team", None) or [])
    if len(ours) < 2 or not theirs:
        return None
    weights = _their_lead_weights(battle, theirs)
    with _previewed(battle, theirs):
        scores = [math.log(lead_prior(m.species))
                  + MATCHUP_WEIGHT * sum(w * matchup(battle, m, o) for o, w in zip(theirs, weights)) / 100
                  for m in ours]
    order = sorted(range(len(ours)), key=lambda i: (-scores[i], i))
    return [i + 1 for i in order], scores


def teampreview_message(battle: AbstractBattle) -> Optional[str]:
    res = team_order(battle)
    if res is None:
        return None
    return "/team " + "".join(str(i) for i in res[0])
