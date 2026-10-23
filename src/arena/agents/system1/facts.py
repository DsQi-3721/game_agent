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
from dataclasses import dataclass
from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.data import to_id_str

from arena.augment.calc_util import damage_estimate, known_or_likely_moves
from arena.augment.damage import tera_for
from arena.augment.speed import speed_band
from arena.augment.switch_eval import hazard_text
from arena.knowledge import store
from arena.knowledge.opponent_model import DEFAULT_MODEL
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
    ko_now: bool = False            # guaranteed or possible KO this turn
    no_effect: bool = False         # the target is immune

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


def ko_label(lo: float, hi: float, target_hp: float) -> tuple[str, bool]:
    """Verdict for a damage range (% of max HP) against the target's current HP %."""
    if hi <= 0:
        return "deals no damage", False
    if target_hp <= 0:
        return "target already fainted", False
    if lo >= target_hp:
        return "guaranteed KO this turn", True
    if hi >= target_hp:
        return "chance to KO this turn (high damage roll)", True
    n_best, n_worst = math.ceil(target_hp / hi), math.ceil(target_hp / max(lo, 1e-6))
    return (f"{n_best} hits to KO" if n_best == n_worst else f"{n_best}-{n_worst} hits to KO"), False


def _speed_relation(battle: AbstractBattle, mine: Pokemon, theirs: Pokemon) -> str:
    a = speed_band(battle, mine)[1]
    b_lo, b, b_hi, p_scarf, _ = speed_band(battle, theirs)
    if a is None or b is None:
        return "speed order unknown"
    trick = Field.TRICK_ROOM in (battle.fields or {})
    if (a > b_hi and not trick) or (a < b_lo and trick):
        rel = "moves first"
    elif (a < b_lo and not trick) or (a > b_hi and trick):
        rel = "moves second"
    else:
        rel = "speed is close (could move first or second)"
    if 0 < p_scarf < 1 and not trick and rel == "moves first":
        rel += f" (unless the opponent holds a Choice Scarf, {round(p_scarf * 100)}% likely)"
    return rel


def _move_effect(move: Move) -> str:
    entry = store.dex().get("moves", {}).get(move.id) or {}
    desc = (entry.get("desc") or "").strip()
    return desc[:140] + ("..." if len(desc) > 140 else "")


def _best_hit(battle, attacker, defender, moves=None, attacker_tera=None):
    best = None
    for mv in (moves if moves is not None else known_or_likely_moves(battle, attacker)):
        if mv is None or (mv.base_power <= 0 and mv.id not in ("seismictoss", "nightshade")):
            continue
        d = damage_estimate(battle, attacker, defender, mv, attacker_tera=attacker_tera)
        if d is not None and (best is None or d.pct_hi > best.pct_hi):
            best = d
    return best


def threat_to(battle: AbstractBattle, mon: Pokemon, opp: Pokemon) -> str:
    """The opponent active's strongest known/likely hit on ``mon``."""
    d = _best_hit(battle, opp, mon)
    if d is None:
        return f"opp {opp.species} has no known damaging move against {mon.species}"
    verdict, _ = ko_label(d.pct_lo, d.pct_hi, hp_pct(mon))
    return (f"opp {opp.species}'s strongest known/likely move {d.move} does {d.pct_lo:.0f}-{d.pct_hi:.0f}% "
            f"to {mon.species} (it has {hp_pct(mon)}%): {verdict}")


def build_option_facts(battle: AbstractBattle, options: list[Option]) -> list[OptionFacts]:
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    out: list[OptionFacts] = []
    speed_me = _speed_relation(battle, me, opp) if me is not None and opp is not None and not me.fainted else ""
    for o in options:
        key = option_key(o)
        if o.kind == "move":
            mv: Move = o.order.order
            name = (mv.entry or {}).get("name", mv.id)
            acc = "never misses" if mv.accuracy is True else f"accuracy {round(float(mv.accuracy) * 100)}%"
            prio = f", priority {mv.priority:+d}" if mv.priority else ""
            tera = tera_for(battle, me) if o.tera and me is not None else None
            action = f"use {name} ({mv.type.name.title()} {mv.category.name.lower()}, power {mv.base_power}, {acc}{prio})"
            if tera:
                action = f"terastallize into {tera} (your type becomes {tera}), then " + action
            fields = {"action": action}
            exp, ko_now, immune = None, False, False
            if opp is not None and (mv.base_power > 0 or mv.id in ("seismictoss", "nightshade")):
                mult = opp.damage_multiplier(mv)
                immune = mult == 0
                fields["vs_opponent"] = f"{_EFF.get(mult, f'x{mult:g}')} against opp {opp.species}"
                d = damage_estimate(battle, me, opp, mv, attacker_tera=tera) if me is not None else None
                if d is not None:
                    verdict, ko_now = ko_label(d.pct_lo, d.pct_hi, hp_pct(opp))
                    exp = (d.pct_lo + d.pct_hi) / 2
                    fields["damage"] = (f"estimated {d.pct_lo:.0f}-{d.pct_hi:.0f}% of its max HP "
                                        f"(it has {hp_pct(opp)}% left): {verdict}")
            else:
                eff = _move_effect(mv)
                fields["effect"] = f"status move: {eff}" if eff else "status move"
            if speed_me:
                if mv.priority > 0:
                    fields["order"] = "priority move: acts before normal moves"
                elif speed_me.startswith("moves first") and mv.priority == 0:
                    fields["order"] = "you move first"
                else:
                    fields["order"] = f"you {speed_me}"
            out.append(OptionFacts(key, o, fields, exp, ko_now, immune))
        else:
            # A switch spends this turn: the newcomer takes the opponent's attack on entry and acts
            # only from next turn. Jev reads literally, so every verdict here says which turn it is about.
            mon: Pokemon = o.order.order
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


def _mon_state(battle: AbstractBattle, mon: Pokemon, own: bool) -> dict:
    d = {"species": mon.species, "hp": f"{hp_pct(mon)}%", "types": _types(mon)}
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
        if known_item(mon):
            d["item"] = known_item(mon)
        if known_ability(mon):
            d["ability"] = known_ability(mon)
        prior = DEFAULT_MODEL.set_prior(battle, mon)
        if prior is not None:
            d["likely_moves"] = [c.name for c in prior.likely_moves(4)]
    if mon.is_terastallized:
        d["terastallized"] = (tera_type_of(battle, mon) or "?")
    return d


def build_state(battle: AbstractBattle, *, extra_facts: Optional[list[str]] = None) -> dict:
    """Compact JSON state; numbers already turned into verdicts where they matter."""
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    state: dict = {"battle": f"Pokemon Showdown {battle.format} singles, turn {battle.turn}"}
    you: dict = {}
    if me is not None and not me.fainted:
        you["active"] = _mon_state(battle, me, own=True)
        tera = tera_type_of(battle, me)
        you["can_terastallize"] = (tera if battle.can_tera and tera else "no")
    you["bench"] = [f"{p.species} {hp_pct(p)}%" + (f" {p.status.name}" if p.status else "")
                    for p in battle.team.values() if not p.active and not p.fainted]
    you["pokemon_left"] = sum(1 for p in battle.team.values() if not p.fainted)
    state["you"] = you
    them: dict = {}
    if opp is not None:
        them["active"] = _mon_state(battle, opp, own=False)
    seen = [p for p in battle.opponent_team.values() if not p.active]
    if seen:
        them["seen_bench"] = [f"{p.species} {'fainted' if p.fainted else str(hp_pct(p)) + '%'}" for p in seen]
    them["pokemon_left"] = 6 - sum(1 for p in battle.opponent_team.values() if p.fainted)
    state["opponent"] = them
    if me is not None and opp is not None and not me.fainted:
        state["speed"] = f"your {me.species} {_speed_relation(battle, me, opp)}"
        state["threat"] = threat_to(battle, me, opp)
    field = {}
    if battle.weather:
        field["weather"] = ",".join(w.name.lower() for w in battle.weather)
    if battle.fields:
        field["field"] = ",".join(f.name.lower() for f in battle.fields)
    if battle.side_conditions:
        field["your_side"] = [c.name.lower() for c in battle.side_conditions]
    if battle.opponent_side_conditions:
        field["opponent_side"] = [c.name.lower() for c in battle.opponent_side_conditions]
    if field:
        state["field"] = field
    if battle.force_switch:
        state["note"] = "your active Pokemon must be replaced: choose a switch"
    if extra_facts:
        state["known_facts"] = list(extra_facts)
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
