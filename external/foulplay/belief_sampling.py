"""Full opponent sets drawn from our opponent model's beliefs, inside Foul Play's environment (the look-ahead service
and the offline search evaluation import it).

Foul Play determinizes the hidden opponent sets by sampling from its own candidate sets (data.foulplay.cc full sets,
Showdown sets, Smogon usage) that are still consistent with what its tracker observed. ``direct_set`` draws a set
from OUR distributions instead -- item, ability, Tera type, the unrevealed moves (by inclusion probability) and a
spread that goes with the item -- and keeps it only if Foul Play's own checks accept it: known moves / item /
ability / Tera, impossible items and abilities, its Speed range, and ``set_makes_logical_sense`` (no Choice item
with setup moves, Light Clay without screens ...). ``None`` when no draw passes; the caller falls back.
"""

import random

from fp import constants
from fp.data.sets import PokemonMoveset, PokemonSet, PredictedPokemonSet


def pick(dist, exclude=(), rng=random):
    rows = [(k, v) for k, v in dist.items() if k not in exclude and v > 0]
    if not rows:
        return None
    r = rng.random() * sum(v for _k, v in rows)
    for k, v in rows:
        r -= v
        if r <= 0:
            return k
    return rows[-1][0]


def direct_set(pkmn, bel, tries=8, rng=random):
    known = [m.name for m in pkmn.moves]
    for _ in range(tries):
        if pkmn.item == constants.UNKNOWN_ITEM:
            excl = set(pkmn.impossible_items)
            if not pkmn.can_have_choice_item:
                excl |= set(constants.CHOICE_ITEMS)
            item = pick(bel.get("items") or {}, excl, rng)
            if item is None:
                return None
        else:
            item = pkmn.item
        ability = pkmn.ability or pick(bel.get("abilities") or {}, set(pkmn.impossible_abilities), rng)
        if ability is None:
            return None
        tera = pkmn.tera_type if pkmn.terastallized else pick(bel.get("tera") or {}, (), rng)
        moves = list(known)
        pool = {m: q for m, q in (bel.get("moves") or {}).items() if m not in moves}
        while len(moves) < 4 and pool:
            m = pick(pool, (), rng)
            pool.pop(m)
            moves.append(m)
        rows = (bel.get("spreads_by_item") or {}).get(item) or bel.get("spreads") or []
        chosen = None
        for _ in range(4):
            i = pick({j: r[2] for j, r in enumerate(rows)}, (), rng)
            if i is None:
                break
            ps = PokemonSet(ability=ability, item=item, nature=rows[i][0], evs=tuple(rows[i][1]), count=1,
                            tera_type=tera)
            if ps.speed_check(pkmn):
                chosen = ps
                break
        if chosen is None:
            continue
        pps = PredictedPokemonSet(pkmn_set=chosen, pkmn_moveset=PokemonMoveset(moves=tuple(moves)))
        if pps.set_makes_logical_sense():
            return pps
    return None
