"""Damage evidence: what a hit on our Pokemon says about the attacker's item and attacking stat.

Our own HP is exact, so the damage of an opponent's hit is known to the HP point. At each request the model
tabulates, for the opponent's active against ours, the damage range of its likely moves under each hypothesis
that changes it: the item class (Choice Band / Specs, Life Orb, Expert Belt, a type-boosting item, or a neutral
item) x its attacking stat (from the candidate spreads: nature and Attack / Sp. Atk EVs). When the opponent moves
first in that turn (so the position is the one tabulated) and hits once, without a critical hit, the observed
damage keeps the hypotheses whose range contains it -- a Specs Kyurem and a Modest 252 SpA one look different.
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.move import Move
from poke_env.calc import calculate_damage

from arena.knowledge.estimate import StatEstimate, _compute

TYPE_ITEMS = {"blackglasses": "dark", "charcoal": "fire", "mysticwater": "water", "miracleseed": "grass",
              "magnet": "electric", "nevermeltice": "ice", "blackbelt": "fighting", "poisonbarb": "poison",
              "softsand": "ground", "sharpbeak": "flying", "twistedspoon": "psychic", "silverpowder": "bug",
              "hardstone": "rock", "spelltag": "ghost", "dragonfang": "dragon", "metalcoat": "steel",
              "silkscarf": "normal", "fairyfeather": "fairy"}
# moves whose damage depends on more than the attacker's item and attacking stat (fixed damage, variable power,
# the item itself, weather / terrain / turn order, the target's stats): the table would be wrong for them
SKIP_MOVES = {"seismictoss", "nightshade", "superfang", "ruination", "naturesmadness", "endeavor", "finalgambit",
              "counter", "mirrorcoat", "metalburst", "bodypress", "foulplay", "psyshock", "psystrike", "secretsword",
              "beatup", "photongeyser", "terablast", "lastrespects", "ragefist", "storedpower", "powertrip",
              "acrobatics", "knockoff", "facade", "hex", "venoshock", "brine", "weatherball", "risingvoltage",
              "expandingforce", "terrainpulse", "boltbeak", "fishiousrend", "gyroball", "electroball", "heavyslam",
              "heatcrash", "lowkick", "grassknot", "payback", "avalanche", "revenge", "stompingtantrum", "temperflare",
              "ragingbull", "collisioncourse", "electrodrift", "lashout", "hardpress", "crushgrip", "eruption",
              "waterspout", "dragonenergy", "reversal", "flail", "assurance", "retaliate", "fusionbolt", "fusionflare",
              "solarbeam", "solarblade", "psyblade", "hydrosteam", "ivycudgel", "terastarstorm", "poltergeist",
              "smellingsalts", "wakeupslap", "rollout", "iceball", "furycutter", "echoedvoice", "round", "pursuit",
              "mistyexplosion", "grassyglide", "meteorbeam", "skyattack", "futuresight", "doomdesire"}
# abilities whose damage effect the table cannot follow (counts of fainted allies, activation state, field-wide
# stat drops, type-changing moves): no damage evidence while the attacker or the field may have one
SKIP_ABILITIES = {"supremeoverlord", "protosynthesis", "quarkdrive", "analytic", "sniper", "flashfire", "guts",
                  "toxicboost", "flareboost", "solarpower", "sandforce", "stakeout", "rivalry", "aerilate", "pixilate",
                  "refrigerate", "galvanize", "normalize", "liquidvoice", "protean", "libero", "moxie",
                  "beastboost", "swordofruin", "beadsofruin", "tabletsofruin", "vesselofruin", "parentalbond",
                  "skilllink", "hustle", "defeatist", "slowstart", "orichalcumpulse", "hadronengine", "embodyaspect"}


def item_class(item: str, move: Move) -> str:
    """The item as far as this move's damage is concerned ("neutral" when it does not change it)."""
    cat = move.category.name.lower()
    if item == "choiceband" and cat == "physical":
        return item
    if item == "choicespecs" and cat == "special":
        return item
    if item in ("lifeorb", "expertbelt", "wellspringmask", "hearthflamemask", "cornerstonemask"):
        return item
    if item == "muscleband" and cat == "physical" or item == "wiseglasses" and cat == "special":
        return item
    t = TYPE_ITEMS.get(item)
    if t and move.type.name.lower() == t:
        return item
    return "neutral"


def attack_stat(species: str, nature: str, evs, special: bool) -> Optional[int]:
    st = _compute(species, 100, {"evs": list(evs), "ivs": [31] * 6, "nature": nature}, 9)
    return None if st is None else st["spa" if special else "atk"]


def hit_table(battle, opp, ours, belief, moves: list[str], *, max_stats: int = 4) -> dict:
    """{move: {"special", "classes": {item: class}, "rows": [[class, attacking stat, lo, hi]]}} in HP points of
    ``ours``; hypotheses from ``belief`` (items with p >= 0.02, its candidate spreads)."""
    from arena.augment.calc_util import assumed
    from arena.state.observe import identifier_in
    atk_id, def_id = identifier_in(battle, opp), identifier_in(battle, ours)
    if atk_id is None or def_id is None:
        return {}
    ability = next((a for a, p in belief.abilities.items() if p >= 0.5), None) if not opp.ability else None
    possible = {opp.ability} if opp.ability else {a for a, p in belief.abilities.items() if p >= 0.05}
    field_ruin = any(m.ability in ("swordofruin", "beadsofruin", "tabletsofruin", "vesselofruin")
                     for m in list(battle.team.values()) + list(battle.opponent_team.values()) if m.active)
    if possible & SKIP_ABILITIES or field_ruin or (ours.ability in SKIP_ABILITIES):
        return {}
    out = {}
    base_spread = belief.spreads[0] if belief.spreads else ("serious", [84] * 6, 1.0)
    for mid in moves:
        if mid in SKIP_MOVES:
            continue
        try:
            mv = opp.moves.get(mid) or Move(mid, gen=9)
        except Exception:  # noqa: BLE001
            continue
        if mv.base_power <= 0 or mv.category.name == "STATUS" or (mv.n_hit and mv.n_hit[1] > 1):
            continue
        special = mv.category.name == "SPECIAL"
        classes = {x: item_class(x, mv) for x, p in belief.items.items() if p >= 0.02}
        stats = []
        for nature, evs, _p in belief.spreads:
            v = attack_stat(belief.species, nature, evs, special)
            if v is not None and v not in stats:
                stats.append(v)
        stats = stats[:max_stats]
        rows = []
        for cls in sorted(set(classes.values()) | {"neutral"}):
            for v in stats:
                st = _compute(belief.species, int(getattr(opp, "level", 100) or 100),
                              {"evs": list(base_spread[1]), "ivs": [31] * 6, "nature": base_spread[0]}, 9)
                if st is None:
                    continue
                st = dict(st)
                st["spa" if special else "atk"] = v
                est = StatEstimate(st, st, st, False, "")
                item = "leftovers" if cls == "neutral" else cls     # any item that leaves damage alone
                try:
                    with assumed(opp, est, item=item, ability=ability):
                        lo, hi = calculate_damage(atk_id, def_id, mv, battle)
                except Exception:  # noqa: BLE001  (poke-env's calc has unhandled corners)
                    continue
                rows.append([cls, v, float(lo), float(hi)])
        if rows:
            out[mid] = {"special": special, "classes": classes, "rows": rows}
    return out


def damage_ok(entry_species: str, item: str, nature: str, evs, obs: dict, tol_frac: float = 0.03) -> bool:
    """Is the observed damage possible for this (item, spread)? No matching row: no information (True)."""
    cls = obs["classes"].get(item, "neutral")
    v = attack_stat(entry_species, nature, evs, obs["special"])
    if v is None:
        return True
    rows = [r for r in obs["rows"] if r[0] == cls]
    if not rows:
        return True
    near = min(rows, key=lambda r: abs(r[1] - v))
    if abs(near[1] - v) > 0.03 * v:
        return True
    d = obs["damage"]
    tol = max(2.0, tol_frac * d)
    return near[2] - tol <= d <= near[3] + tol
