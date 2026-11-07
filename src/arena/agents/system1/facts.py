"""Battle position -> compact state + per-option facts for a System One model.

Jev-style models judge well but do not calculate (docs: "keep the arithmetic in code";
numbers, counting and multi-hop reasoning are weak spots; irrelevant detail costs accuracy).
So everything numeric is computed here with the arena tools (damage calc with estimated
hidden stats, speed bands, hazards, opponent set priors) and handed over as short semantic
statements: effectiveness label, damage range with a KO verdict, who moves first, the risk a
switch-in takes. Option keys are descriptive (models read option names literally).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.data import to_id_str

from arena.augment.calc_util import damage_estimate, known_or_likely_moves
from arena.augment.damage import tera_for
from arena.augment.speed import speed_band, speed_limits
from arena.knowledge.estimate import is_own
from arena.augment.switch_eval import hazard_text
from arena.knowledge import store
from arena.knowledge.opponent_model import DEFAULT_MODEL
from arena.observation.builder import observe
from arena.observation.schema import Phase
from arena.observation.trace import REVIVAL_NOTE, action_constraints, phase_of
from arena.state.observe import hp_pct, known_ability, known_item, revealed_moves, tera_type_of
from arena.state.views import Option

_EFF = {0: "NO EFFECT (target is immune)", 0.25: "double resisted (x0.25)", 0.5: "resisted (x0.5)",
        1: "neutral (x1)", 2: "super effective (x2)", 4: "double super effective (x4)"}


@dataclass
class OptionFacts:
    key: str
    option: Option
    fields: dict                    # labelled facts: action / vs_opponent / damage / order / risk ...
    exp_damage: Optional[float]     # mean estimated % of the target's max HP (moves only)
    ko_now: bool = False            # KO this turn if the estimate holds, or possible with a high roll
    no_effect: bool = False         # the target is immune
    ko_margin: Optional[float] = None   # lowest estimated damage minus the target's HP (% points)

    @property
    def text(self) -> str:
        return "; ".join(str(v) for v in self.fields.values()) + "."


def option_key(o: Option) -> str:
    if o.kind == "move":
        return f"move_{o.id}" + ("_tera" if o.tera else "")
    return f"switch_{o.id}"


def _types(mon: Pokemon) -> list[str]:
    return [t.name.title() for t in mon.types if t is not None]


def _boosts(mon: Pokemon) -> dict:
    return {k: v for k, v in (mon.boosts or {}).items() if v}


def ko_label(lo: float, hi: float, target_hp: float, unless: tuple[str, ...] = ()) -> tuple[str, bool]:
    """Verdict for an ESTIMATED damage range (% of max HP) against the target's current HP %. It holds only
    if the estimate does (assumed spread / item / ability, no critical hit); ``unless`` names what could
    still let the target survive (Focus Sash, Sturdy, Multiscale, a Substitute, a defensive Tera)."""
    if hi <= 0:
        return "deals no damage", False
    if target_hp <= 0:
        return "target already fainted", False
    tail = f" (unless {', '.join(unless)})" if unless else ""
    if lo >= target_hp:
        return f"KO this turn if the estimate holds{tail}", True
    if hi >= target_hp:
        return f"chance to KO this turn (high damage roll){tail}", True
    n_best, n_worst = math.ceil(target_hp / hi), math.ceil(target_hp / max(lo, 1e-6))
    return (f"{n_best} hits to KO" if n_best == n_worst else f"{n_best}-{n_worst} hits to KO"), False


def no_damage_label(note: str) -> str:
    """No damage that comes from an ability or item the estimate assumed (Levitate, Air Balloon, Flash Fire ...),
    not from the types: true only if the assumption is."""
    assumed = re.findall(r"assumes ([^,;]+)", note or "")
    why = f"assumes {' and '.join(assumed)}" if assumed else "its ability or item is not revealed"
    return f"no damage if the estimate holds ({why})"


SURVIVAL_ABILITIES = {"sturdy": "Sturdy", "multiscale": "Multiscale", "shadowshield": "Shadow Shield",
                      "disguise": "Disguise", "iceface": "Ice Face", "tabletsofruin": "", "terashell": "Tera Shell"}


def survival_risks(battle: AbstractBattle, target: Pokemon, can_tera: bool) -> tuple[str, ...]:
    """Why a target the estimate says is KOed might survive this hit (public information only)."""
    out = []
    full = hp_pct(target) >= 100
    item = target.item
    ability = target.ability
    abilities = [ability] if ability else sorted(getattr(target, "possible_abilities", None) or [])
    if full:
        if item == "focussash" or (item == "unknown_item" and _may_hold(battle, target, "focussash")):
            out.append("it holds a Focus Sash")
        out += [f"its ability is {SURVIVAL_ABILITIES[a]}" for a in abilities if SURVIVAL_ABILITIES.get(a)]
    elif any(a in ("disguise", "iceface") for a in abilities):
        out += [f"its {SURVIVAL_ABILITIES[a]} is intact" for a in abilities if a in ("disguise", "iceface")]
    if any(e.name == "SUBSTITUTE" for e in (target.effects or {})):
        out.append("its Substitute takes the hit")
    if can_tera:
        out.append("it Terastallizes defensively")
    return tuple(out)


def _may_hold(battle: AbstractBattle, mon: Pokemon, item: str) -> bool:
    prior = DEFAULT_MODEL.set_prior(battle, mon)
    floor = 0.02 if prior is not None and prior.source == "gen9ou+team" else 0.0   # it lists every possible item
    return prior is None or any(c.id == item and c.p >= floor for c in prior.items)


# move-data keys that change what a damaging move does beyond its damage (Showdown data/moves.ts)
NOTABLE = ("selfSwitch", "secondary", "secondaries", "self", "drain", "recoil", "mindBlownRecoil", "onAfterHit",
           "onHit", "basePowerCallback", "damageCallback", "damage", "overrideOffensiveStat",
           "overrideOffensivePokemon", "overrideDefensiveStat", "multihit", "forceSwitch", "heal", "selfdestruct",
           "onTry", "onTryHit", "ignoreDefensive", "ignoreAbility", "willCrit", "breaksProtect", "sleepUsable")
DELAYED = {"futuresight", "doomdesire"}    # damage lands two turns later, not now
FIRST_TURN_ONLY = {"fakeout", "firstimpression"}   # fail once the user has moved since it came in
HP_SET = {"endeavor", "finalgambit"}       # damage set by HP: the target down to the user's HP / the user's HP
HALF_HP = {"superfang", "ruination", "naturesmadness"}


def move_power_text(mv: Move) -> str:
    entry = mv.entry or {}
    if entry.get("damage") == "level":
        return "fixed damage equal to your level"
    if mv.id in HALF_HP:
        return "fixed damage: half the target's current HP"
    if mv.id == "endeavor":
        return "fixed damage: brings the target down to your current HP"
    if mv.id == "finalgambit":
        return "fixed damage equal to your current HP (you faint)"
    if entry.get("damageCallback"):
        return "fixed damage (see effect)"
    if entry.get("basePowerCallback"):
        return f"variable power (base {mv.base_power})" if mv.base_power else "variable power (see effect)"
    return f"power {mv.base_power}"


def _speed_relation(battle: AbstractBattle, mine: Pokemon, theirs: Pokemon) -> str:
    """Who acts first between two Pokemon using normal-priority moves. "moves first / second" is certain:
    it holds for every Speed the opponent could legally have (``speed_limits``); "likely ..." holds for its
    usual spreads and items (``speed_band``) and says what would flip it."""
    a = speed_band(battle, mine)[1]
    b_lo, b, b_hi, p_scarf, note = speed_band(battle, theirs)
    if a is None or b is None:
        return "speed order unknown"
    if is_own(battle, theirs):
        lim_lo, lim_hi = b_lo, b_hi
    else:
        lim_lo, lim_hi = speed_limits(battle, theirs)
        lim_lo = b_lo if lim_lo is None else min(lim_lo, b_lo)
        lim_hi = b_hi if lim_hi is None else max(lim_hi, b_hi)
    trick = Field.TRICK_ROOM in (battle.fields or {})
    faster_certain, slower_certain = (a < lim_lo, a > lim_hi) if trick else (a > lim_hi, a < lim_lo)
    faster_likely, slower_likely = (a < b_lo, a > b_hi) if trick else (a > b_hi, a < b_lo)
    if faster_certain:
        return "moves first"
    if slower_certain:
        return "moves second"
    if faster_likely:
        why = [f"it holds a Choice Scarf ({round(p_scarf * 100)}% likely)" if 0 < p_scarf < 1 else
               "it holds a Choice Scarf" if p_scarf == 0 and theirs.item == "unknown_item" else "",
               "it runs an unusual Speed spread or ability"]
        return f"likely moves first (unless {' or '.join(w for w in why if w)})"
    if slower_likely:
        return "likely moves second (unless it runs an unusual slower spread or item)"
    return "speed is close (could move first or second)"


_WEATHER_TYPE = {"sunnyday": "FIRE", "desolateland": "FIRE", "raindance": "WATER", "primordialsea": "WATER",
                 "sandstorm": "ROCK", "hail": "ICE", "snow": "ICE", "snowscape": "ICE"}
_TERRAIN_TYPE = {"electricterrain": "ELECTRIC", "grassyterrain": "GRASS", "mistyterrain": "FAIRY",
                 "psychicterrain": "PSYCHIC"}
_FORME_TYPE = {"ivycudgel": {"wellspring": "WATER", "hearthflame": "FIRE", "cornerstone": "ROCK"},
               "ragingbull": {"combat": "FIGHTING", "blaze": "FIRE", "aqua": "WATER"}}
_PLATE_MOVES = ("judgment", "multiattack", "technoblast")


def effective_move_type(battle: AbstractBattle, attacker: Optional[Pokemon], move: Move,
                        tera: Optional[str] = None):
    """The type a move actually hits with (the damage calc already knows; the label must agree):
    Weather Ball / Terrain Pulse by field, Tera Blast with a tera, Ivy Cudgel / Raging Bull by forme,
    Judgment / Multi-Attack / Techno Blast by the held plate / memory / drive."""
    from poke_env.battle.pokemon_type import PokemonType

    def pt(name):
        try:
            return PokemonType[name.upper()]
        except KeyError:
            return move.type
    mid = move.id
    if mid == "weatherball":
        for w in (battle.weather or {}):
            t = _WEATHER_TYPE.get(w.name.lower().replace("_", ""))
            if t:
                return pt(t)
    if mid == "terrainpulse":
        for f in (battle.fields or {}):
            t = _TERRAIN_TYPE.get(f.name.lower().replace("_", ""))
            if t:
                return pt(t)
    if mid == "terablast":
        if tera:
            return pt(tera)
        if attacker is not None and attacker.is_terastallized and attacker.tera_type is not None:
            return attacker.tera_type
    if mid in _FORME_TYPE and attacker is not None:
        for suffix, t in _FORME_TYPE[mid].items():
            if attacker.species.endswith(suffix):
                return pt(t)
    if mid in _PLATE_MOVES and attacker is not None and attacker.item:
        entry = store.dex().get("items", {}).get(attacker.item) or {}
        t = entry.get("onPlate") or entry.get("onMemory") or entry.get("onDrive")
        if t:
            return pt(t)
    return move.type


def _move_effect(move: Move, limit: int = 180) -> str:
    entry = store.dex().get("moves", {}).get(move.id) or {}
    desc = (entry.get("desc") or "").strip()
    return desc[:limit] + ("..." if len(desc) > limit else "")


def _is_damaging(mv: Move) -> bool:
    return mv.category.name != "STATUS"


def _estimate(battle, attacker, defender, mv: Move, attacker_tera=None):
    """Damage estimate (% of the defender's max HP); fixed-damage moves the calc does not model are
    computed from their rule (half the target's current HP)."""
    if mv.id in HALF_HP:
        from arena.augment.calc_util import DamageEstimate
        cur = hp_pct(defender)
        pct = cur / 2 if not is_immune(defender, mv) else 0.0
        return DamageEstimate(attacker.species, defender.species, mv.id, pct, pct, None, None, "half current HP")
    if mv.id in HP_SET:
        return _hp_set_estimate(battle, attacker, defender, mv)
    d = damage_estimate(battle, attacker, defender, mv, attacker_tera=attacker_tera)
    lo_hits, hi_hits = mv.n_hit
    if d is not None and lo_hits != hi_hits:      # poke-env's calc uses the average hit count (2-5 -> 3.5)
        from dataclasses import replace
        avg = (lo_hits + hi_hits) / 2
        n_lo = hi_hits if attacker.ability == "skilllink" else (max(lo_hits, 4) if attacker.item == "loadeddice"
                                                               else lo_hits)
        d = replace(d, pct_lo=d.pct_lo / avg * n_lo, pct_hi=d.pct_hi / avg * hi_hits,
                    note=(d.note + "; " if d.note else "") + f"{n_lo}-{hi_hits} hits")
    return d


def _abs_hp(battle, mon) -> tuple[Optional[float], Optional[float], str]:
    """(current, max) HP: exact for our Pokemon, the estimated max HP times the shown % for the opponent's."""
    if is_own(battle, mon):
        return float(mon.current_hp or 0), float(mon.max_hp or 0), ""
    est = DEFAULT_MODEL.estimate(battle, mon)
    if est is None:
        return None, None, ""
    mx = float(est.stats["hp"])
    return (mon.current_hp_fraction or 0) * mx, mx, f"{mon.species}: {est.note}"


def _hp_set_estimate(battle, attacker, defender, mv: Move):
    from arena.augment.calc_util import DamageEstimate
    (a_cur, _a_max, n1), (d_cur, d_max, n2) = _abs_hp(battle, attacker), _abs_hp(battle, defender)
    if a_cur is None or d_cur is None or not d_max:
        return None
    if is_immune(defender, mv):
        dmg = 0.0
    elif mv.id == "endeavor":
        dmg = max(0.0, d_cur - a_cur)
    else:
        dmg = min(d_cur, a_cur)
    pct = 100.0 * dmg / d_max
    return DamageEstimate(attacker.species, defender.species, mv.id, pct, pct, None, None,
                          "; ".join(x for x in (n1, n2) if x))


def acted_since_entry(battle, mon) -> bool:
    """Whether ``mon`` (ours) has used a move since it last came in: public, the player saw it."""
    from arena.observation.tracker import mon_key, tracker_for
    key = next((k for k, m in battle.team.items() if m is mon), None)
    h = tracker_for(battle).mons.get(mon_key(key)) if key else None
    return bool(h and h.moved_since_entry)


def is_immune(defender: Pokemon, mv: Move) -> bool:
    try:
        return defender.damage_multiplier(mv) == 0
    except Exception:  # noqa: BLE001
        return False


def _best_hit(battle, attacker, defender, moves=None, attacker_tera=None):
    best = None
    for mv in (moves if moves is not None else known_or_likely_moves(battle, attacker)):
        if mv is None or not _is_damaging(mv):
            continue
        d = _estimate(battle, attacker, defender, mv, attacker_tera)
        if d is not None and (best is None or d.pct_hi > best.pct_hi):
            best = d
    return best


def threat_to(battle: AbstractBattle, mon: Pokemon, opp: Pokemon) -> str:
    """The opponent active's strongest known/likely hit on ``mon``."""
    d = _best_hit(battle, opp, mon)
    if d is None:
        return f"opp {opp.species} has no known damaging move against {mon.species}"
    verdict, _ = ko_label(d.pct_lo, d.pct_hi, hp_pct(mon),
                          ("you Terastallize defensively",) if battle.can_tera and mon.active else ())
    return (f"opp {opp.species}'s strongest known/likely move {d.move} does {d.pct_lo:.0f}-{d.pct_hi:.0f}% "
            f"to {mon.species} (it has {hp_pct(mon)}%): {verdict}")


def build_option_facts(battle: AbstractBattle, options: list[Option]) -> list[OptionFacts]:
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    out: list[OptionFacts] = []
    speed_me = _speed_relation(battle, me, opp) if me is not None and opp is not None and not me.fainted else ""
    revival = phase_of(battle) is Phase.REVIVAL
    obs = observe(battle)
    active_obs = next((d for d in obs.own_team if d['active']), {})
    recorded_moves = {m['id']: m for m in active_obs.get('moves', [])}
    opp_can_tera = opp is not None and not opp.is_terastallized and not obs.tera["opponent"].get("used")
    for o in options:
        key = option_key(o)
        if o.kind == "move" and o.order.order.id == "recharge":
            out.append(OptionFacts(key, o, {"action": "recharge (forced: your Pokemon must recharge this turn and "
                                                      "cannot act; no other option exists)"}, None))
        elif o.kind == "move":
            mv: Move = o.order.order
            name = (mv.entry or {}).get("name", mv.id)
            acc = "never misses" if mv.accuracy is True else f"accuracy {round(float(mv.accuracy) * 100)}%"
            prio = f", priority {mv.priority:+d}" if mv.priority else ""
            tera = tera_for(battle, me) if o.tera and me is not None else None
            action = (f"use {name} ({mv.type.name.title()} {mv.category.name.lower()}, {move_power_text(mv)}, "
                      f"{acc}{prio})")
            if tera:
                action = f"terastallize into {tera} (your type becomes {tera}), then " + action
            fields = {"action": action}
            recorded_move = recorded_moves.get(mv.id)
            if recorded_move is not None:
                fields['pp'] = ({'remaining': recorded_move.get('pp'), 'maximum': recorded_move.get('maxpp'),
                                 'knowledge': recorded_move.get('pp_knowledge', 'unrecorded')})
            exp, ko_now, immune, margin = None, False, False, None
            if _is_damaging(mv):
                if any(k in (mv.entry or {}) for k in NOTABLE):
                    eff = _move_effect(mv)
                    if eff:
                        fields["effect"] = eff
                if opp is not None:
                    mult = opp.damage_multiplier(effective_move_type(battle, me, mv, tera))
                    immune = mult == 0
                    fields["vs_opponent"] = f"{_EFF.get(mult, f'x{mult:g}')} against opp {opp.species}"
                    first_turn_over = mv.id in FIRST_TURN_ONLY and me is not None and acted_since_entry(battle, me)
                    d = (_estimate(battle, me, opp, mv, tera) if me is not None and mv.id not in DELAYED
                         and not first_turn_over else None)
                    if mv.id in DELAYED:
                        fields["damage"] = "no damage this turn: it hits whatever is in the target's slot 2 turns later"
                    if first_turn_over:
                        fields["damage"] = "fails: it only works on the first turn after this Pokemon came in"
                    if d is not None:
                        risks = survival_risks(battle, opp, can_tera=opp_can_tera)
                        verdict, ko_now = ko_label(d.pct_lo, d.pct_hi, hp_pct(opp), risks)
                        if d.pct_hi <= 0 and not immune and not (opp.ability and opp.item != "unknown_item"):
                            verdict = no_damage_label(d.note)   # from an assumed ability / item, not the types
                        exp = (d.pct_lo + d.pct_hi) / 2
                        margin = d.pct_lo - hp_pct(opp)
                        fields["damage"] = (f"estimated {d.pct_lo:.0f}-{d.pct_hi:.0f}% of its max HP "
                                            f"(it has {hp_pct(opp)}% left): {verdict}")
            else:
                eff = _move_effect(mv)
                fields["effect"] = f"status move: {eff}" if eff else "status move"
            if speed_me:
                if mv.priority > 0:
                    fields["order"] = "priority move: acts before normal moves"
                elif mv.priority < 0:
                    fields["order"] = "negative priority: acts after normal moves"
                elif speed_me == "moves first":
                    fields["order"] = "you move first"
                elif speed_me == "moves second":
                    fields["order"] = "you move second"
                else:
                    fields["order"] = f"you {speed_me}"
            out.append(OptionFacts(key, o, fields, exp, ko_now, immune, margin))
        else:
            # A switch spends this turn: the newcomer takes the opponent's attack on entry and acts
            # only from next turn. Jev reads literally, so every verdict here says which turn it is about.
            mon: Pokemon = o.order.order
            if revival:
                out.append(OptionFacts(key, o, {"action": f"revive {mon.species} ({'/'.join(_types(mon))}): it returns "
                                                          f"to your team with half its HP and does not switch in"}, None))
                continue
            status = f", {mon.status.name}" if mon.status else ""
            fields = {"action": f"switch to {mon.species} ({hp_pct(mon)}% HP{status}, {'/'.join(_types(mon))})"}
            if not battle.force_switch:
                fields["cost"] = "switching uses this turn: you deal no damage this turn"
            _, hz = hazard_text(battle, mon)
            fields["entry_hazards"] = hz
            if opp is not None:
                if not battle.force_switch:
                    fields["risk"] = "it takes the opponent's attack this turn as it comes in: " + threat_to(battle, mon, opp)
                else:
                    fields["risk"] = threat_to(battle, mon, opp)
                fields["speed"] = f"from next turn {mon.species} {_speed_relation(battle, mon, opp)}"
                d = _best_hit(battle, mon, opp, moves=list(mon.moves.values()))
                if d is not None:
                    verdict, _ = ko_label(d.pct_lo, d.pct_hi, hp_pct(opp))
                    fields["offense"] = (f"from next turn, if it is still standing, its best move vs opp {opp.species} "
                                         f"is {d.move}: {d.pct_lo:.0f}-{d.pct_hi:.0f}% "
                                         f"({verdict.replace('this turn', 'next turn')})")
            out.append(OptionFacts(key, o, fields, None))
    return out


def _mon_state(battle: AbstractBattle, mon: Pokemon, own: bool, d_obs: Optional[dict] = None,
               exact_hp: bool = True) -> dict:
    d = {"species": mon.species, "hp": f"{hp_pct(mon)}%", "types": _types(mon)}
    if own and exact_hp and d_obs and d_obs.get("hp") and d_obs["hp"][1]:
        d["hp"] = f"{hp_pct(mon)}% ({d_obs['hp'][0]}/{d_obs['hp'][1]})"
    if mon.status:
        d["status"] = mon.status.name
    if _boosts(mon):
        d["stat_changes"] = _boosts(mon)
    if own:
        if mon.ability:
            d["ability"] = mon.ability
        if mon.item:
            d["item"] = mon.item
    else:
        if revealed_moves(mon):
            d["revealed_moves"] = revealed_moves(mon)
        if d_obs is not None:
            d["item"] = _item_text(d_obs["item"])
            if d_obs["ability"]["status"] == "known":
                d["ability"] = d_obs["ability"]["value"]
        else:
            if known_item(mon):
                d["item"] = known_item(mon)
            if known_ability(mon):
                d["ability"] = known_ability(mon)
        prior = DEFAULT_MODEL.set_prior(battle, mon)
        if prior is not None:
            d["likely_moves"] = [c.id for c in prior.likely_moves(4)]
    if mon.is_terastallized:
        d["terastallized"] = (tera_type_of(battle, mon) or "?")
    if d_obs is not None:
        effects = [_effect_text(e) for e in d_obs.get("effects") or []]
        if effects:
            d["effects"] = effects
        if d_obs.get("last_move"):
            d["last_move"] = d_obs["last_move"]["move"]
    return d


def _item_text(item: dict) -> str:
    if item["status"] == "known":
        return item["value"]
    if item["status"] == "none":
        how = f", {item['how'].replace('move: ', '')}" if item.get("how") else ""
        return f"none (lost {item['lost']}{how})" if item.get("lost") else "none"
    return "unknown"


def _effect_text(e: dict) -> str:
    extra = [f"{k} {e[k]}" for k in ("move", "count") if e.get(k) is not None]
    return e["effect"] + (f" ({', '.join(extra)})" if extra else "")


def _turns_text(left: Optional[dict]) -> str:
    if not left:
        return ""
    n = left["min"] if left["min"] == left["max"] else f"{left['min']}-{left['max']}"
    return f" ({n} turns left)"


def _bench_text(d: dict, own: bool) -> str:
    if d["fainted"]:
        return f"{d['species']} fainted" + (" (may have been revived since)" if d.get("fainted_unknown") else "")
    if own:
        hp = round(100 * d["hp"][0] / d["hp"][1]) if d["hp"][1] else 0
        bits = [f"{d['species']} {hp}%" + (f" {d['status']}" if d.get("status") else "")]
        bits.append("item " + _item_text(d['item']))
        if d.get("tera_type"):
            bits.append(f"tera {d['tera_type']}")
        if d.get("moves"):
            bits.append("moves " + "/".join(m["id"] for m in d["moves"]))
        return ", ".join(bits)
    seen = f" (last seen turn {d['hp_seen_turn']})" if d.get("hp_seen_turn") is not None else ""
    if d.get("fainted_unknown"):
        seen += " (may have fainted since)"
    bits = [f"{d['species']} {d['hp_pct']}%{seen}" + (f" {d['status']}" if d.get("status") else "")]
    if d["item"]["status"] != "unknown":
        bits.append("item " + _item_text(d["item"]))
    if d["ability"]["status"] == "known":
        bits.append(f"ability {d['ability']['value']}")
    if d.get("revealed_moves"):
        bits.append("moves " + "/".join(m["id"] for m in d["revealed_moves"]))
    if d.get("terastallized"):
        bits.append(f"terastallized {d['terastallized']['type']}")
    return ", ".join(bits)


def _tera_text(t: dict, team: list[dict]) -> str | bool:
    if not t.get("used"):
        return False
    who = next((d["species"] for d in team if d["ident"] == t.get("by")), None)
    return f"by {who or 'an earlier Pokemon'} ({t['type']})"


def build_state(battle: AbstractBattle, *, extra_facts: Optional[list[str]] = None) -> dict:
    """Compact JSON state from the battle's PlayerObservation (``arena.observation.builder``); numbers
    already turned into verdicts where they matter. State version 2 (I/O audit): full own team with fainted
    ones, opponent preview and per-Pokemon dossiers, effects and last moves, hazard layers and remaining
    turns, Tera used vs available now, why actions are missing."""
    obs = observe(battle)
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    state: dict = {"battle": f"Pokemon Showdown {battle.format} singles, turn {battle.turn}"}
    you: dict = {}
    me_obs = next((d for d in obs.own_team if d["active"]), None)
    if me is not None and not me.fainted:
        you["active"] = _mon_state(battle, me, own=True, d_obs=me_obs, exact_hp="stats" not in obs.unrecorded)
    tera_type = (me_obs or {}).get("tera_type") or (tera_type_of(battle, me) if me is not None else None)
    you["tera"] = {"type": tera_type, "available_now": obs.tera["own"].get("available_now", False),
                   "used_this_battle": _tera_text(obs.tera["own"], obs.own_team) if obs.tera["own"].get("used") is not None
                   else "unknown"}
    you["bench"] = [_bench_text(d, own=True) for d in obs.own_team if not d["active"]]
    you["pokemon_left"] = obs.pokemon_left["own"]
    state["you"] = you
    them: dict = {}
    opp_obs = next((d for d in obs.opponent_revealed_team if d["active"]), None)
    if opp is not None:
        them["active"] = _mon_state(battle, opp, own=False, d_obs=opp_obs)
    them["tera_used_this_battle"] = (_tera_text(obs.tera["opponent"], obs.opponent_revealed_team)
                                     if obs.tera["opponent"].get("used") is not None
                                     else "unknown")
    if obs.opponent_preview:
        them["team_preview"] = [to_id_str(p["details"].split(",")[0]) for p in obs.opponent_preview]
    seen = [d for d in obs.opponent_revealed_team if not d["active"]]
    if seen:
        them["seen_bench"] = [_bench_text(d, own=False) for d in seen]
    them["pokemon_left"] = obs.pokemon_left["opponent"]
    state["opponent"] = them
    if me is not None and opp is not None and not me.fainted:
        state["speed"] = f"your {me.species} {_speed_relation(battle, me, opp)}"
        state["threat"] = threat_to(battle, me, opp)
    field = {}
    if obs.weather:
        field["weather"] = obs.weather["weather"] + _turns_text(obs.weather.get("turns_left"))
    if obs.field_conditions:
        field["field"] = [f["field"] + _turns_text(f.get("turns_left")) for f in obs.field_conditions]
    for who, key in (("own", "your_side"), ("opponent", "opponent_side")):
        conds = obs.side_conditions[who]
        if conds:
            field[key] = [c["condition"] + (f" x{c['layers']}" if "layers" in c else "")
                          + _turns_text(c.get("turns_left")) for c in conds]
    incoming = [f"{cond} on {'your' if who == 'own' else 'the opponent'}'s side (from turn {v['turn']})"
                for who in ("own", "opponent") for cond, v in obs.slot_conditions[who].items()]
    if incoming:
        field["incoming"] = incoming
    if field:
        state["field"] = field
    phase = phase_of(battle)
    if phase is Phase.REVIVAL:
        state["note"] = REVIVAL_NOTE
    elif battle.force_switch:
        state["note"] = "your active Pokemon must be replaced: choose a switch"
    constraints = action_constraints(getattr(battle, "_last_request", None) or {})
    if constraints:
        you["this_turn"] = constraints
    if extra_facts:
        state["known_facts"] = list(extra_facts)
    from arena.s1.resources import policy_resources
    state['resources'] = policy_resources(battle, obs)
    return state


def reference_choices(battle: AbstractBattle, facts: list[OptionFacts]) -> dict:
    """Simple references for offline evaluation: max-damage option and poke-env's heuristic."""
    refs: dict = {}
    dmg = [f for f in facts if f.exp_damage is not None and not f.option.tera]
    if dmg:
        refs["max_damage"] = max(dmg, key=lambda f: f.exp_damage).key
    try:
        from poke_env.player.baselines import SimpleHeuristicsPlayer
        order, _ = SimpleHeuristicsPlayer.choose_singles_move(battle)
        refs["simple_heuristics"] = next((f.key for f in facts if f.option.order.message == order.message), None)
    except Exception:  # noqa: BLE001
        pass
    return refs


def species_id(name: str) -> str:
    return to_id_str(name)
