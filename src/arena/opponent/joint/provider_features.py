"""Mechanics-based features for each concrete action in the predicted player's view."""
from __future__ import annotations

import math
import numpy as np
from poke_env.data import GenData, to_id_str

DEX = GenData.from_gen(9)
NAMES = (
    "attack", "status", "switch", "tera", "setup", "heal", "protect", "pivot",
    "power", "effectiveness", "accuracy", "priority", "stab", "physical", "special",
    "damage_proxy", "recoil", "drain", "hazard", "hazard_removal", "inflict_status",
    "repeat_status", "setup_hp", "setup_boosts", "heal_missing_hp", "heal_hp",
    "switch_missing_hp", "switch_hp", "switch_matchup", "switch_offense", "switch_speed",
    "switch_hazard_loss", "switch_status", "switch_regenerator", "switch_resists",
    "tera_missing_hp", "tera_stab_gain", "tera_defense_gain", "tera_attack", "tera_setup",
    "attack_low_hp", "attack_target_low_hp", "priority_target_low_hp", "setup_threat",
    "heal_threat", "pivot_threat", "switch_threat", "status_target_low_hp",
    "status_already_statused", "power_accuracy", "move_pp", "hazard_present",
    "protect_low_hp", "status_hp", "switch_force", "switch_pivot",
)
INDEX = {name: i for i, name in enumerate(NAMES)}


def species_data(name):
    key = to_id_str(name or "")
    return DEX.pokedex.get(key, {})


def hp_fraction(value):
    if value is None:
        return 1.
    text = str(value)
    if "fnt" in text:
        return 0.
    if "/" in text:
        hp, maximum = text.split()[0].split("/")
        return float(hp)/float(maximum)
    return float(text.split()[0])/100


def effectiveness(move_type, types):
    # Roost can leave a pure Flying Pokémon temporarily typeless.
    typeless = {"THREE_QUESTION_MARKS", "???"}
    if not move_type or move_type.upper() in typeless:
        return 1.
    return math.prod(DEX.type_chart[t.upper()][move_type.upper()] for t in types if t.upper() not in typeless)


def types_for(mon):
    return species_data(mon.get("initial_species", mon.get("species"))).get("types", [])


def matchup(attacking_types, defending_types):
    return max((effectiveness(t, defending_types) for t in attacking_types), default=1.)


def action_features(state, criteria):
    """No labels, player IDs, generator parameters or future protocol are accepted."""
    config, public = state["hypothesized_own_config"], state["public_context"]
    team = {p["slot_id"]: p for p in config["team"]}
    own = next((p for p in team.values() if p.get("active")), next(iter(team.values())))
    other = next((p for p in public.get("opponent_seen", []) if p.get("active")), {})
    other_types = species_data(other.get("species")).get("types", [])
    if other.get("terastallized") and other["terastallized"].get("type", "").lower() != "stellar":
        other_types = [other["terastallized"]["type"]]
    effects = public.get("target_active_public_effects") or {}
    own_types = effects.get("current_types") or types_for(own)
    hp, other_hp = hp_fraction(own.get("hp")), (other.get("hp_pct", 100) or 0)/100
    threat = math.log2(max(.125, matchup(other_types, own_types)))/3
    boosts = effects.get("boosts") or {}
    own_hazards = {to_id_str(s.get("id", s.get("condition", ""))) if isinstance(s, dict) else to_id_str(s)
                   for s in public.get("side_conditions", {}).get("own", [])}
    other_hazards = {to_id_str(s.get("id", s.get("condition", ""))) if isinstance(s, dict) else to_id_str(s)
                     for s in public.get("side_conditions", {}).get("opponent", [])}
    moves = [p[3] for line in public.get("recent_public_events", [])
             if len(p := line.split("|")) > 3 and p[1] == "move" and p[2].startswith(state["target_side"])]
    last = to_id_str(moves[-1]) if moves else None
    move_state = {m["id"]: m for a in config.get("active_constraints", []) for m in a.get("move_state", [])}
    result = []
    for action in criteria.values():
        values = {}
        if action["kind"] in {"switch", "revive"}:
            incoming = team[action["switch_slot"]]
            types = types_for(incoming)
            incoming_hp = hp_fraction(incoming.get("hp"))
            incoming_threat = math.log2(max(.125, matchup(other_types, types)))/3
            rocks = .125*effectiveness("Rock", types) if "stealthrock" in own_hazards else 0.
            spikes = .125 if "spikes" in own_hazards and "Flying" not in types else 0.
            loss = 0. if incoming.get("item") == "heavydutyboots" else rocks+spikes
            values.update(switch=1., switch_missing_hp=1-hp, switch_hp=incoming_hp,
                switch_matchup=-incoming_threat, switch_offense=math.log2(max(.125, matchup(types, other_types)))/3,
                switch_speed=species_data(incoming.get("initial_species")).get("baseStats", {}).get("spe", 80)/150,
                switch_hazard_loss=min(loss, 1.), switch_status=float(len(str(incoming.get("hp", "")).split()) > 1),
                switch_regenerator=float(own.get("current_ability") == "regenerator")*(1-hp),
                switch_resists=threat-incoming_threat, switch_threat=threat,
                switch_force=float(state["phase"] == "force_switch"), switch_pivot=float(state["phase"] == "pivot_switch"))
        elif action["kind"] == "move":
            move_id = action["move_id"]
            move = DEX.moves.get(move_id, {})
            category = move.get("category", "Status")
            attack = category != "Status"
            setup = category == "Status" and any(v > 0 for v in move.get("boosts", {}).values())
            healing = bool(move.get("heal")) or move_id in {"rest", "wish", "strengthsap"}
            pivot = bool(move.get("selfSwitch"))
            protect = move_id in {"protect", "detect", "kingsshield", "silktrap", "banefulbunker"}
            power = min(move.get("basePower", 0) or (60 if attack else 0), 150)/150
            accuracy = 1. if move.get("accuracy", True) is True else move["accuracy"]/100
            eff = effectiveness(move.get("type"), other_types)
            stat = "atk" if category == "Physical" else "spa"
            base_stat = own.get("base_stats", {}).get(stat, 250)/350
            stab = float(move.get("type", "").lower() in [t.lower() for t in own_types])
            hazard = bool(move.get("sideCondition"))
            tera = bool(action.get("tera"))
            tera_type = own.get("tera_type", "Normal") or "Normal"
            tera_defense = math.log2(max(.125, matchup(other_types, [tera_type])))/3 if tera_type.lower() != "stellar" else threat
            values.update(attack=float(attack), status=float(not attack), tera=float(tera), setup=float(setup),
                heal=float(healing), protect=float(protect), pivot=float(pivot), power=power,
                effectiveness=math.log(max(.1, eff))*attack, accuracy=(accuracy-1)*attack,
                priority=move.get("priority", 0)/3, stab=stab*attack, physical=float(category == "Physical"),
                special=float(category == "Special"), damage_proxy=min(3., power*eff*base_stat*(1+.5*stab))/3,
                recoil=float(bool(move.get("recoil"))), drain=float(bool(move.get("drain"))),
                hazard=float(hazard), hazard_removal=float(move_id in {"rapidspin", "defog", "mortalspin", "tidyup", "courtchange"}),
                inflict_status=float(bool(move.get("status"))), repeat_status=float(not attack and move_id == last),
                setup_hp=setup*hp, setup_boosts=setup*sum(max(0, v) for v in boosts.values())/12,
                heal_missing_hp=healing*(1-hp), heal_hp=healing*hp, tera_missing_hp=tera*(1-hp),
                tera_stab_gain=tera*attack*(float(move.get("type", "").lower() == tera_type.lower())-stab),
                tera_defense_gain=tera*(threat-tera_defense), tera_attack=tera*attack, tera_setup=tera*setup,
                attack_low_hp=attack*(1-hp), attack_target_low_hp=attack*(1-other_hp),
                priority_target_low_hp=move.get("priority", 0)/3*(1-other_hp), setup_threat=setup*threat,
                heal_threat=healing*threat, pivot_threat=pivot*threat, status_target_low_hp=(not attack)*(1-other_hp),
                status_already_statused=float(bool(move.get("status")) and bool(other.get("status"))),
                power_accuracy=power*accuracy, move_pp=move_state.get(move_id, {}).get("pp", 16)/32,
                hazard_present=float(move.get("sideCondition") in other_hazards),
                protect_low_hp=protect*(1-hp), status_hp=(not attack)*hp)
        result.append([float(values.get(name, 0.)) for name in NAMES])
    return np.asarray(result, dtype=np.float64)


def compatible_actions(criteria, events, side, config):
    """Resolve only the target's publicly executed choice, not its hidden request."""
    tera = any(line.startswith(f"|-terastallize|{side}") for line in events)
    # An Encore applied earlier in this resolution can replace the move the
    # player selected. The executed move then identifies only the action kind.
    overridden = any(line.startswith(f"|-start|{side}") and "|Encore" in line for line in events)
    chosen = None
    for line in events:
        parts = line.split("|")
        if len(parts) < 3 or not parts[2].startswith(side):
            continue
        if parts[1] in {"faint", "drag"}:
            break  # a later replacement says nothing about the original choice
        if len(parts) < 4:
            continue
        if parts[1] == "move" and not any(s.startswith("[from]") for s in parts[4:]):
            chosen = ("move", to_id_str(parts[3])); break
        if parts[1] == "switch":
            species = to_id_str(parts[3].split(",")[0])
            chosen = ("switch", species); break
    team = {m["slot_id"]: to_id_str(m["initial_species"]) for m in config["team"]}
    if chosen is None:
        return np.asarray([bool(a.get("tera")) if tera else True for a in criteria.values()])
    kind, value = chosen
    return np.asarray([(a["kind"] == "move" and (overridden or a["move_id"] == value) and bool(a.get("tera")) == tera)
        if kind == "move" else (a["kind"] in {"switch", "revive"} and team.get(a["switch_slot"]) == value)
        for a in criteria.values()])
