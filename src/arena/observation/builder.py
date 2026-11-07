"""PlayerObservation (obs-1) from a poke-env Battle: one source of game facts for every consumer.

Sources, in order of authority: the private request (own team: exact HP, stats, PP, items, abilities, tera
type; what may be chosen now), the public protocol as poke-env parsed it (current state of both sides),
and ``tracker`` (history poke-env drops: last moves, item / ability provenance, Tera per side, when effects
and conditions started, who set the weather). Every fact that is not the player's own exact knowledge says
how it is known (``Knowledge``) and, when only a range is known, gives the range.

Rules the builder follows (the audit's "do not mix" list):
- an opponent item is ``unknown`` until revealed, ``none`` once it is known to be gone (with what was lost,
  how and when), never "holds X" after X was knocked off;
- "cannot Terastallize now" (request) and "Tera already used this battle" (events) are separate fields;
- stackable hazards carry layers, timed conditions their start turn and remaining turns (a range when the
  duration depends on an item the player cannot see), never one in place of the other;
- the own team lists all Pokemon, fainted included; bench PP is what the player last saw for that Pokemon;
- Metamon-converted battles carry no protocol history; what the converter could not recover is listed in
  the request's ``unrecorded`` and reported as such, not as "none".
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.side_condition import STACKABLE_CONDITIONS
from poke_env.data import to_id_str

from arena.observation.schema import (SCHEMA_VERSION, Fact, Knowledge, Phase, PlayerObservation, Provenance,
                                      Source, Tier)
from arena.observation.trace import phase_of, selectable_actions
from arena.observation.tracker import WEATHER_ITEMS, mon_key, tracker_for

# turns a timed condition lasts by default, and with its extending item
DURATIONS = {"raindance": (5, 8), "sunnyday": (5, 8), "sandstorm": (5, 8), "snow": (5, 8), "snowscape": (5, 8),
             "electricterrain": (5, 8), "grassyterrain": (5, 8), "mistyterrain": (5, 8), "psychicterrain": (5, 8),
             "reflect": (5, 8), "lightscreen": (5, 8), "auroraveil": (5, 8), "tailwind": (4, 4),
             "safeguard": (5, 5), "mist": (5, 5), "trickroom": (5, 5), "gravity": (5, 5), "magicroom": (5, 5),
             "wonderroom": (5, 5)}
EXTENDERS = {**WEATHER_ITEMS, "electricterrain": "terrainextender", "grassyterrain": "terrainextender",
             "mistyterrain": "terrainextender", "psychicterrain": "terrainextender", "reflect": "lightclay",
             "lightscreen": "lightclay", "auroraveil": "lightclay"}


def _status(mon) -> Optional[str]:
    st = getattr(mon, "status", None)
    return st.name.lower() if st is not None else None


def _effects(mon, history) -> list[dict]:
    out = []
    for eff, counter in (getattr(mon, "effects", None) or {}).items():
        name = to_id_str(eff.name)
        d = {"effect": name}
        tracked = history.effects.get(name) if history else None
        if tracked:
            d.update({k: v for k, v in tracked.items() if k != "turn"})
            d["since_turn"] = tracked["turn"]
        if counter and eff.name.startswith(("PERISH", "STOCKPILE")):
            d["count"] = counter
        out.append(d)
    return out


def _turns_left(cond: str, start: Optional[int], now: int, item: Optional[str], item_known: bool) -> Optional[dict]:
    """Remaining turns of a timed condition at the start of turn ``now``: exact if the setter's item is
    known, else the range over default / extended duration. None when the start is not known."""
    if cond not in DURATIONS or start is None or start < 0:
        return None
    base, ext = DURATIONS[cond]
    lo_d, hi_d = (ext, ext) if item_known and item == EXTENDERS.get(cond) else (
        (base, base) if item_known or base == ext else (base, ext))
    elapsed = max(0, now - start)
    lo, hi = max(0, lo_d - elapsed), max(0, hi_d - elapsed)
    return {"min": lo, "max": hi, "knowledge": (Knowledge.DERIVED if lo == hi else Knowledge.ESTIMATED).value}


def _item_fact(mon, history) -> dict:
    """An opponent's item: known, none (known to be gone: what, how, when) or unknown."""
    item = getattr(mon, "item", None)
    lost = history.lost_item if history else None
    if item not in (None, "", "unknown_item"):
        return {"status": "known", "value": item, "knowledge": Knowledge.REVEALED.value,
                **({"since_turn": history.item_turn, "how": history.item_how} if history and history.item else {})}
    if lost or item in (None, ""):
        return {"status": "none", "lost": lost, "how": history.lost_how if history else None,
                "turn": history.lost_turn if history else None, "knowledge": Knowledge.REVEALED.value}
    return {"status": "unknown"}


def _own_mon(battle, entry: dict, tr) -> dict:
    ident = entry["ident"]
    mon = battle.team.get(ident)
    cond = str(entry.get("condition", ""))
    hp, _, rest = cond.partition(" ")
    cur, _, mx = hp.partition("/")
    d = {"ident": ident, "details": entry.get("details"), "species": getattr(mon, "species", None),
         "active": bool(entry.get("active")), "fainted": cond.endswith("fnt"),
         "hp": [int(cur), int(mx)] if mx.isdigit() and cur.isdigit() else [0, 0],
         "status": rest if rest and rest != "fnt" else None, "stats": entry.get("stats"),
         "item": {"status": "known" if entry.get("item") else "none", "value": entry.get("item") or None,
                  "knowledge": Knowledge.OWN_PRIVATE.value},
         "ability": entry.get("ability"), "base_ability": entry.get("baseAbility"),
         "tera_type": entry.get("teraType"), "terastallized": entry.get("terastallized") or None}
    moves = []
    if entry.get('item') in ('unknownitem', 'unknown_item'):
        d['item'] = {'status': 'unknown', 'value': None, 'knowledge': 'unrecorded'}
    req = getattr(battle, "_last_request", None) or {}
    act = (req.get("active") or [{}])[0] if entry.get("active") else {}
    by_id = {m["id"]: m for m in act.get("moves") or [] if "pp" in m}
    for mid in entry.get("moves") or []:
        m = {"id": mid}
        recorded = entry.get('policy_moves')
        if recorded is not None:
            source = recorded[mid]
            m.update(pp=source.get('pp'), maxpp=source.get('maxpp'),
                     pp_knowledge=Knowledge.RECONSTRUCTED.value if source.get('pp') is not None else 'unrecorded')
        elif mid in by_id:
            m.update(pp=by_id[mid]["pp"], maxpp=by_id[mid]["maxpp"], pp_knowledge=Knowledge.OWN_PRIVATE.value)
        elif mon is not None and mid in mon.moves:
            mv = mon.moves[mid]
            m.update(pp=mv.current_pp, maxpp=mv.max_pp, pp_knowledge="last_seen")
        moves.append(m)
    d["moves"] = moves
    if 'policy_hp_fraction' in entry:
        d['hp_fraction'] = entry['policy_hp_fraction']
        d['recorded_types'] = entry['policy_types']
        d['last_observed_effect'] = entry['policy_effect']
        d['knowledge'] = 'last_seen' if entry.get('policy_last_seen') else 'reconstructed'
    h = tr.mons.get(mon_key(ident))
    if mon is not None and entry.get("active"):
        d["boosts"] = {k: v for k, v in (mon.boosts or {}).items() if v}
        d["effects"] = _effects(mon, h)
    if h and h.last_move:
        d["last_move"] = {"move": h.last_move, "turn": h.last_move_turn}
    return d


def _opp_mon(battle, key: str, mon, tr, opp_role: str) -> dict:
    h = tr.mons.get(mon_key(key))
    shown = (h.temp_forme or h.forme) if h else None
    d = {"ident": key, "species": to_id_str(shown) if shown else mon.species, "level": mon.level,
         "active": bool(mon.active), "fainted": bool(mon.fainted),
         "hp_pct": round((mon.current_hp_fraction or 0) * 100),
         # benched: what was last shown; it can change unseen (Regenerator, Natural Cure, Wish passes)
         "hp_knowledge": "current" if mon.active or mon.fainted else "last_seen",
         **({"hp_seen_turn": h.hp_turn} if h and h.hp_turn is not None and not mon.active else {}),
         "status": _status(mon) if not mon.fainted else "fnt",
         "item": _item_fact(mon, h),
         "ability": ({"status": "known", "value": mon.ability, "knowledge": Knowledge.REVEALED.value}
                     if mon.ability else {"status": "unknown",
                                          "candidates": sorted(getattr(mon, "possible_abilities", None) or [])}),
         "revealed_moves": [{"id": m.id, "pp_used": (m.max_pp - m.current_pp)
                             if m.max_pp and 'opponent_pp' not in (battle._last_request or {}).get('unrecorded', []) else None}
                            for m in mon.moves.values()],
         "terastallized": ({"type": mon.tera_type.name.title() if mon.tera_type else None}
                           if mon.is_terastallized else None)}
    if mon.active:
        d["boosts"] = {k: v for k, v in (mon.boosts or {}).items() if v}
        d["effects"] = _effects(mon, h)
    if h and h.last_move:
        d["last_move"] = {"move": h.last_move, "turn": h.last_move_turn}
    return d


def _conditions(conds: dict, side: str, tr, now: int, setter_item, layers_known: bool = True,
                start_known: bool = True) -> list[dict]:
    out = []
    for c, value in (conds or {}).items():
        cid = to_id_str(c.name)
        d = {"condition": cid}
        if c in STACKABLE_CONDITIONS:
            if layers_known:
                d["layers"] = int(value)
        elif start_known:
            d["since_turn"] = int(value)
        info = tr.sides.get((side, cid)) or {}
        left = _turns_left(cid, info.get("clock", d.get("since_turn")), now, *setter_item(info)) if start_known else None
        if left:
            d["turns_left"] = left
        out.append(d)
    return out


def observe(battle, provenance: Optional[Provenance] = None) -> PlayerObservation:
    tr = tracker_for(battle)
    req = getattr(battle, "_last_request", None) or {}
    role = battle.player_role or "p1"
    opp_role = "p2" if role == "p1" else "p1"
    now = tr.now
    tr.expire_slots(now)
    prov = provenance or Provenance(source=Source.LIVE, tier=Tier.STRICT, format=battle.format or "",
                                    battle_id=battle.battle_tag, player_side=role)
    own = [_own_mon(battle, e, tr) for e in (req.get("side") or {}).get("pokemon") or []]
    opp = [_opp_mon(battle, k, m, tr, opp_role) for k, m in battle.opponent_team.items()]
    unattributed = int(getattr(battle, "_arena_unattributed_faints", 0) or 0)
    for d in opp:                         # converted replay: the count changed, but not for which Pokemon
        if (unattributed > 0 and not d["active"] and not d["fainted"]) or (unattributed < 0 and d["fainted"]):
            d["fainted_unknown"] = True
    preview = [{"details": det, "item": has} for det, has in tr.preview.get(opp_role, [])]
    if not preview and getattr(battle, "teampreview_opponent_team", None):
        preview = [{"details": p.species, "item": None} for p in battle.teampreview_opponent_team]
    unrecorded = list(req.get("unrecorded") or [])

    def setter_item(info: dict) -> tuple[Optional[str], bool]:
        """(item the setter of a timed condition held when it was set, whether that is known): it fixes
        the duration. Own Pokemon: the request's item, unless the record shows it was lost after the set."""
        setter = (info or {}).get("setter")
        if not setter:
            return None, False
        if setter.startswith(role):
            mon = next((m for k, m in battle.team.items() if mon_key(k) == setter), None)
            h = tr.mons.get(setter)
            if h and h.lost_item and h.lost_seq is not None and h.lost_seq > info.get("seq", 0):
                return h.lost_item, True
            if h and h.lost_item and not getattr(mon, "item", None):
                return None, True
            return getattr(mon, "item", None), mon is not None
        return info.get("setter_item"), bool(info.get("setter_item_public"))

    if "condition_start_turns" in unrecorded:      # converted replay state: when conditions began is unknown
        tr.weather, tr.fields, tr.sides = {}, {}, {}
        start_known = False
    else:
        start_known = True
    weather = None
    for w, start in (battle.weather or {}).items():
        wid = to_id_str(w.name)
        info = tr.weather if tr.weather.get("id") == wid else {}
        weather = {"weather": wid, "since_turn": info.get("turn", start) if start_known else None,
                   "source": info.get("source") or None,
                   "turns_left": _turns_left(wid, info.get("clock", start), now, *setter_item(info)) if start_known else None}
    fields = []
    for f, start in (battle.fields or {}).items():
        fid = to_id_str(f.name)
        info = tr.fields.get(fid, {})
        fields.append({"field": fid, "since_turn": info.get("turn", start) if start_known else None,
                       "source": info.get("source") or None,
                       "turns_left": _turns_left(fid, info.get("clock", start), now, *setter_item(info))
                       if start_known else None})
    layers_known = "layers" not in unrecorded
    side_conditions = {"own": _conditions(battle.side_conditions, role, tr, now, setter_item, layers_known,
                                          start_known),
                       "opponent": _conditions(battle.opponent_side_conditions, opp_role, tr, now, setter_item,
                                               layers_known, start_known)}
    can_now = bool(((req.get("active") or [{}])[0]).get("canTerastallize"))
    tera = {}
    for who, side in (("own", role), ("opponent", opp_role)):
        used = tr.tera.get(side)
        tera[who] = {"used": used is not None, **({"by": used["mon"], "type": used["type"], "turn": used["turn"]}
                                                   if used else {})}
    tera["own"]["available_now"] = can_now
    if "tera_history" in unrecorded:          # converted replay state: whether Tera was spent is unknown
        tera["own"]["used"] = tera["opponent"]["used"] = None
    effects = []
    for d in own + opp:
        for e in d.get("effects") or []:
            effects.append({"mon": d["ident"], **e})
    obs = PlayerObservation(
        provenance=prov, request_id=req.get("rqid"), request_phase=phase_of(battle), actual_turn=now,
        decision_index=-1, event_sequence=len(getattr(battle, "_replay_data", None) or []),
        own_team=own, opponent_preview=preview, opponent_revealed_team=opp,
        active_positions={f"{role}a": next((d["ident"] for d in own if d["active"]), None),
                          f"{opp_role}a": next((d["ident"] for d in opp if d["active"]), None)},
        weather=weather, field_conditions=fields, side_conditions=side_conditions,
        slot_conditions={"own": tr.slots.get(role, {}), "opponent": tr.slots.get(opp_role, {})},
        observable_effects_and_counters=effects,
        public_history={"log": battle.battle_tag, "end": len(getattr(battle, "_replay_data", None) or [])},
        selectable_actions=selectable_actions(req), tera=tera, unrecorded=unrecorded,
        pokemon_left={"own": max(len(own), int(req.get("team_size") or 0)) - sum(d["fainted"] for d in own),
                      "opponent": (len(preview) or 6) - sum(d["fainted"] for d in opp) - unattributed},
        schema_version=SCHEMA_VERSION)
    return obs
