"""Metamon trajectories -> System-One training examples (our state + typed questions).

Metamon (UT-Austin-RPL/metamon) ships human replays reconstructed to a first-person view and self-play
games as ``{"states": [UniversalState], "actions": [int]}`` (one ``.json.lz4`` per battle and player).
Every state is turned back into a poke-env ``Battle``:

  our side    a Showdown request built from the state (``Battle.parse_request``, exactly what a live
              player receives): active moves with PP, the switch-ins, items, abilities, tera type.
              Metamon stores base stats only, so stats are a fixed standard spread (``STAT_SPREAD``)
  opponent    the active Pokemon with what was revealed: HP, status, boosts, known moves, item,
              ability, tera; its team preview; how many it has left
  field       weather, terrain, side conditions, tera availability, forced switch

then ``FastDecider.build_request`` renders the same state and questions the live agent sends. The
recorded action (Metamon's index: 0-3 moves and 9-12 tera moves in alphabetical move order, 4-8
switches in alphabetical species order, -1 = unknown) maps to our option key.

Fields Metamon does not keep (exact stats, the opponent's benched HP, fainted teammates, turn number)
are approximated; ``validate`` measures what that costs against battles where everything is known.

Move availability. A 0-PP move stays at 0 PP and disabled. Metamon has no ``disabled`` flags, so the other
disables are derived where the replay determines them (``derive_disabled``: choice lock and Gigaton Hammer /
Blood Moon from the move this same Pokemon used last turn; Taunt, Encore, Heal Block, Throat Chop from its
effect) and recorded under ``request["derived"]``; what cannot be known (other disables, trapping, hard
locks, exact stats) is listed in ``request["unrecorded"]`` instead of being presented as "no constraint".
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from poke_env.battle.battle import Battle
from poke_env.battle.field import Field
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.data import GenData, to_id_str
from poke_env.stats import compute_raw_stats

STAT_SPREAD = {"evs": [84] * 6, "ivs": [31] * 6, "nature": "serious"}   # random-battle style, neutral
CHOICE_ITEMS = {"choiceband", "choicescarf", "choicespecs"}
NO_REPEAT = {"gigatonhammer", "bloodmoon"}            # cannot be selected two turns in a row
# request fields a Metamon state cannot provide; consumers must treat them as unknown, not as "none"
UNRECORDED = ("disabled", "trapped", "maybeTrapped", "maybeDisabled", "maybeLocked", "lockedMove", "stats",
              "condition_start_turns", "fields_beyond_most_recent", "side_conditions_beyond_most_recent",
              "slot_conditions",        # Future Sight / Wish / Healing Wish: not in the format
              "effects_beyond_one",     # one volatile per active Pokemon
              "item_loss_details",      # an item known gone, but not which one or how
              "opponent_pp",            # opponent moves alone do not recover their usage counts
              "bench_details")          # benched Pokemon: HP / status as last shown at a decision step, step index for the turn
# Metamon writes enum names with "_" removed ("trickroom" for Field.TRICK_ROOM), which poke-env's protocol
# parser does not read ("trickroom" -> Field.UNKNOWN); map them back by name. Unknown names are reported
# under request["unmapped"], never dropped silently.
_FIELDS = {f.name.lower().replace("_", ""): f for f in Field}
_WEATHERS = {w.name.lower().replace("_", ""): w for w in Weather}
_WEATHERS["snow"] = Weather.SNOWSCAPE     # early Gen 9 logs (to 2023) name the same weather "Snow"
_STATS = ("hp", "atk", "def", "spa", "spd", "spe")
_BOOSTS = {"atk_boost": "atk", "spa_boost": "spa", "def_boost": "def", "spd_boost": "spd", "spe_boost": "spe",
           "accuracy_boost": "accuracy", "evasion_boost": "evasion"}
_LOG = logging.getLogger("convert_metamon")
_LOG.setLevel(logging.ERROR)


def name_id(name: str) -> str:
    """Metamon's ordering key (lowercase, alphanumerics only) == Showdown ids."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def read_trajectory(path: str | Path) -> dict:
    """A Metamon trajectory file (``.json.lz4``, read with the lz4 command; plain ``.json`` also accepted)."""
    if str(path).endswith(".json"):
        return json.loads(Path(path).read_text())
    raw = subprocess.run(["lz4", "-dc", str(path)], capture_output=True, check=True).stdout
    return json.loads(raw)


def _gen(fmt: str) -> int:
    m = re.match(r"gen(\d+)", fmt or "gen9")
    return int(m.group(1)) if m else 9


def _species_name(data: GenData, sid: str) -> str:
    entry = data.pokedex.get(sid) or {}
    for forme in entry.get("cosmeticFormes") or []:     # Vivillon-Garden: the dex entry is named after the base
        if name_id(forme) == sid:
            return forme
    return entry.get("name", sid)


def _apply_types(mon: Pokemon, types_text: str, tera: str, unmapped: Optional[list] = None) -> None:
    """Metamon stores the CURRENT types (after tera, Soak, Burn Up ...); poke-env re-derives them from the dex.
    An opponent's ``tera`` is only filled once it terastallized."""
    from poke_env.battle.pokemon_type import PokemonType
    types = [t for t in (types_text or "").split() if t != "notype"]
    if tera and tera != "notype" and types == [tera]:
        mon.terastallize(tera)
        return
    if not types or {t.name.lower() for t in mon.types if t is not None} == set(types):
        return
    try:
        mon._type_1 = PokemonType.from_name(types[0])
        mon._type_2 = PokemonType.from_name(types[1]) if len(types) > 1 else None
    except Exception:  # noqa: BLE001  (unknown type name: species types kept, reported)
        if unmapped is not None:
            unmapped.append(f"types:{types_text}")


def _stats(data: GenData, sid: str, level: int, unmapped: Optional[list] = None) -> dict:
    try:
        raw = compute_raw_stats(sid, STAT_SPREAD["evs"], STAT_SPREAD["ivs"], level, STAT_SPREAD["nature"], data)
    except Exception:  # noqa: BLE001  (unknown forme: neutral stats, reported)
        raw = [300, 200, 200, 200, 200, 200]
        if unmapped is not None:
            unmapped.append(f"stats:{sid}")
    return dict(zip(_STATS, raw))


def _status(code: str) -> str:
    return "" if code in ("nostatus", "", None) else (" fnt" if code == "fnt" else f" {code}")


def own_terastallized(data: GenData, mon: dict, can_tera: bool) -> Optional[str]:
    """Metamon keeps the CURRENT types: an own Pokemon showing exactly its tera type, with Tera spent, is
    terastallized -- unless that is also its normal typing (then the state cannot tell; not claimed)."""
    tera = (mon.get("tera_type") or "notype").lower()
    types = [t for t in (mon.get("types") or "").split() if t != "notype"]
    if can_tera or tera == "notype" or types != [tera]:
        return None
    base = [t.lower() for t in ((data.pokedex.get(name_id(mon["name"])) or {}).get("types") or [])]
    return None if base == [tera] else tera.title()


def _req_mon(data: GenData, mon: dict, active: bool, can_tera: bool = True, unmapped: Optional[list] = None) -> dict:
    sid, lvl = name_id(mon["name"]), int(mon.get("lvl") or 100)
    st = _stats(data, sid, lvl, unmapped)
    hp = max(0, round(float(mon.get("hp_pct") or 0) * st["hp"]))
    cond = f"{hp}/{st['hp']}{_status(mon.get('status'))}" if mon.get("status") != "fnt" else "0 fnt"
    tera = mon.get("tera_type") or ""
    base = _species_name(data, name_id(mon.get("base_species") or (data.pokedex.get(sid) or {}).get("baseSpecies") or sid))
    return {"ident": f"p1: {base}", "details": f"{_species_name(data, sid)}, L{lvl}",
            "condition": cond, "active": active, "stats": {k: v for k, v in st.items() if k != "hp"},
            "moves": [name_id(m["name"]) for m in mon.get("moves") or []],
            "policy_moves": {name_id(m['name']): {'pp': m.get('current_pp'), 'maxpp': m.get('max_pp')}
                             for m in mon.get('moves') or []},
            "policy_hp_fraction": mon.get('hp_pct'),
            "policy_types": [t for t in (mon.get('types') or '').split() if t != 'notype'],
            "policy_effect": mon.get('effect'),
            "baseAbility": name_id(mon.get("ability") or ""), "ability": name_id(mon.get("ability") or ""),
            "item": "" if mon.get("item") in ("nothing", "noitem") else name_id(mon.get("item") or ""),
            "teraType": tera.title() if tera and tera != "notype" else "",
            "terastallized": own_terastallized(data, mon, can_tera) or ""}


def history_extras(states: list[dict], t: int, actions: Optional[list] = None) -> dict:
    """What the trajectory before step ``t`` adds to the state at ``t`` (Metamon keeps only the actives and
    the Pokemon one may switch to now):

    - own Pokemon seen fainted earlier (a forced-switch step shows the fainted active), and own Pokemon seen
      earlier that are simply not listed now (no switch offered, e.g. trapped) -- kept apart: the second
      kind is still in the battle;
    - every opponent Pokemon seen earlier as it was last shown (step index = when), and which of them
      fainted (the opponent's count went down when its active was replaced); when the count fell by more
      than that explains (two faints between decisions, or a Pokemon that fainted without ever being shown
      at a decision), the rest are ``opp_unattributed_faints`` -- the live player saw which one, Metamon
      does not keep it. A negative value means the count went back up (Revival Blessing) for a Pokemon not
      shown since: one of those marked fainted is back, but not which;
    - which side has Terastallized and with which Pokemon;
    - the move each active used last turn, when that is certain."""
    cur = states[t]
    data = GenData.from_gen(_gen(cur.get("format") or "gen9ou"))

    def key(m: dict) -> str:
        return name_id(m.get("base_species") or m["name"])

    own_now = {key(cur["player_active_pokemon"])} | {key(p) for p in cur.get("available_switches") or []}
    own_seen: dict = {}
    own_fainted: dict = {}
    own_tera = None
    opp_seen: dict = {}
    opp_fainted: set = set()
    opp_tera = None
    for i, s in enumerate(states[: t + 1]):
        for m in [s["player_active_pokemon"], *(s.get("available_switches") or [])]:
            own_seen[key(m)] = m
            if m.get("status") == "fnt" or float(m.get("hp_pct") or 0) <= 0:
                own_fainted[key(m)] = m
        me = s["player_active_pokemon"]
        if own_tera is None and i < t and actions and i < len(actions) and actions[i] is not None and actions[i] >= 9:
            own_tera = {"mon": me["name"], "type": me.get("tera_type") or "unknown", "step": i}   # chose a tera move
        if own_tera is None and own_terastallized(data, me, bool(s.get("can_tera"))):
            own_tera = {"mon": me["name"], "type": me["tera_type"], "step": i}
        o = s["opponent_active_pokemon"]
        opp_seen[key(o)] = {**o, "_seen_step": i}
        shown_down = key(o) not in opp_fainted and (o.get("status") == "fnt" or float(o.get("hp_pct") or 0) <= 0)
        if shown_down:
            opp_fainted.add(key(o))             # shown fainted (e.g. at a forced switch after a double KO)
        elif o.get("status") != "fnt" and float(o.get("hp_pct") or 0) > 0:
            opp_fainted.discard(key(o))         # back in battle (Revival Blessing)
        if opp_tera is None and (o.get("tera_type") or "notype") != "notype":
            opp_tera = {"mon": o["name"], "type": o["tera_type"], "step": i}
        if i:
            before = states[i - 1]
            drop = int(before.get("opponents_remaining") or 6) - int(s.get("opponents_remaining") or 6) - shown_down
            if drop > 0 and key(before["opponent_active_pokemon"]) != key(o):
                opp_fainted.add(key(before["opponent_active_pokemon"]))   # replaced while the count fell
    cur_opp = key(cur["opponent_active_pokemon"])
    lost = 6 - int(cur.get("opponents_remaining") or 6)
    out = {"opp_unattributed_faints": lost - len(opp_fainted),
           "own_fainted": [m for k, m in own_fainted.items() if k not in own_now],
           "own_unlisted": [m for k, m in own_seen.items() if k not in own_now and k not in own_fainted],
           "opp_bench": [{**m, **({"status": "fnt", "hp_pct": 0.0} if k in opp_fainted else {})}
                         for k, m in opp_seen.items() if k != cur_opp],
           "own_tera": own_tera, "opp_tera": opp_tera,
           "last_move": last_move_of_active(cur, states[t - 1], actions[t - 1]) if t and actions else None}
    if t:                                   # the opponent's previous move belongs to the same active Pokemon
        before = states[t - 1]["opponent_active_pokemon"]
        prev = (cur.get("opponent_prev_move") or {}).get("name")
        if key(before) == cur_opp and prev and prev != "nomove":
            out["opp_last_move"] = name_id(prev)
    return out


def is_selfplay(file_name: str) -> bool:
    """Metamon self-play trajectories (full information) are named ``metamon-<format>-...``; human replays
    (own side reconstructed) ``<n>-<format>-...``."""
    return Path(file_name).name.startswith("metamon-")


def last_move_of_active(state: dict, prev_state: Optional[dict], prev_action: Optional[int]) -> Optional[str]:
    """The move the current active used last turn, if the trajectory shows it stayed in and moved:
    same active in the previous decision state, a move chosen there, and that move reported as the
    player's previous move now. None when any of that is uncertain."""
    if prev_state is None or prev_action is None or prev_state.get("forced_switch"):
        return None
    me, before = state.get("player_active_pokemon") or {}, prev_state.get("player_active_pokemon") or {}
    if name_id(me.get("name", "")) != name_id(before.get("name", "")):
        return None
    chosen = action_key(prev_state, prev_action)
    used = name_id((state.get("player_prev_move") or {}).get("name", ""))
    if not chosen or not chosen.startswith("move_") or chosen.removesuffix("_tera")[len("move_"):] != used:
        return None
    return used


ITEM_SWAPS = {"trick", "switcheroo"}                 # the user's item changes hands: no lock from it


def derive_disabled(moves: list[dict], *, item: str, last_move: Optional[str], effect: str,
                    data: GenData, item_exact: bool = True) -> dict[str, str]:
    """{move id: reason} for the disables a replay state determines (others stay unrecorded).

    A choice lock needs the held item to be the real one: ``item_exact=False`` (Metamon's human replays,
    where the player's own set is reconstructed / predicted) derives no lock -- measured on the labelled
    data, 14.8% of such "locks" contradict the move the player then used, vs 0.7% with self-play's exact
    items."""
    ids = [name_id(m["name"]) for m in moves]
    out: dict[str, str] = {}
    for m, mid in zip(moves, ids):
        if m.get("current_pp") is not None and int(m["current_pp"]) <= 0:
            out[mid] = "no PP left"
    if last_move in ids:
        if item in CHOICE_ITEMS and item_exact and last_move not in ITEM_SWAPS:
            for mid in ids:
                if mid != last_move:
                    out.setdefault(mid, f"locked into {last_move} by {item}")
        if last_move in NO_REPEAT:
            out.setdefault(last_move, "cannot be used twice in a row")
        if effect == "encore":
            for mid in ids:
                if mid != last_move:
                    out.setdefault(mid, f"Encore: must repeat {last_move}")
    for mid in ids:
        entry = data.moves.get(mid) or {}
        if effect == "taunt" and entry.get("category") == "Status":
            out.setdefault(mid, "Taunt")
        if effect == "healblock" and (entry.get("flags") or {}).get("heal"):
            out.setdefault(mid, "Heal Block")
        if effect == "throatchop" and (entry.get("flags") or {}).get("sound"):
            out.setdefault(mid, "Throat Chop")
    return out


def build_request(state: dict, data: GenData, *, last_move: Optional[str] = None, own_exact: bool = False,
                  own_fainted: Optional[list[dict]] = None) -> dict:
    """``last_move``: the move the active used last turn (``last_move_of_active``), for derived disables.
    ``own_exact``: the state's own sets are the real ones (Metamon self-play), not reconstructed (human
    replays); only then is a choice lock derived from the held item. ``own_fainted``: own Pokemon seen
    earlier in the trajectory and gone now (``history_extras``); None = unknown (marked unrecorded)."""
    me = state["player_active_pokemon"]
    can_tera = bool(state.get("can_tera"))
    unmapped: list = []
    side = [_req_mon(data, me, True, can_tera, unmapped)] + [_req_mon(data, p, False, can_tera, unmapped)
                                                             for p in state.get("available_switches") or []]
    for p in own_fainted or []:            # recovered from the trajectory: Pokemon seen earlier, gone now
        side.append(_req_mon(data, {**p, "status": "fnt", "hp_pct": 0.0}, False, can_tera, unmapped))
    req: dict = {"side": {"name": "p1user", "id": "p1", "pokemon": side}, "rqid": 1, "team_size": 6,
                 "unrecorded": list(UNRECORDED) + ["layers"] + ([] if own_fainted is not None else ["own_fainted"]),
                 "unmapped": unmapped}
    if state.get("forced_switch"):
        req["forceSwitch"] = [True]
    else:
        mv = me.get("moves") or []
        disabled = derive_disabled(mv, item=name_id(me.get("item") or ""), last_move=last_move,
                                   effect=name_id(me.get("effect") or ""), data=data, item_exact=own_exact)
        moves = []
        for m in mv:
            mid, pp = name_id(m["name"]), m.get("current_pp")
            entry = {"move": m["name"], "id": mid, "pp": int(pp) if pp is not None else int(m.get("max_pp") or 1),
                     "maxpp": int(m.get("max_pp") or 1), "target": "normal", "disabled": mid in disabled}
            moves.append(entry)
        active: dict = {"moves": moves}
        if moves and all(m["disabled"] for m in moves):     # Showdown: nothing usable -> Struggle, no Tera
            active = {"moves": [{"move": "Struggle", "id": "struggle", "target": "randomNormal", "disabled": False}]}
        else:
            tera = me.get("tera_type") or ""
            if state.get("can_tera") and tera and tera != "notype":
                active["canTerastallize"] = tera.title()
        req["active"] = [active]
        if disabled:
            req["derived"] = {"disabled": disabled}
        if any(m.get("current_pp") is None for m in mv):
            req["unrecorded"].append("pp")
    return req


_SIDE = {name_id(c.name): c for c in SideCondition}


def _conditions(text: str, unmapped: Optional[list] = None) -> list:
    """Metamon writes ids ("stealthrock spikes"); poke-env's parser expects Showdown names. Unknown ids are
    reported in ``unmapped``."""
    out = []
    for tok in (text or "").split():
        if tok == "noconditions":           # Metamon's placeholder for an empty side
            continue
        if tok in _SIDE:
            out.append(_SIDE[tok])
        elif unmapped is not None:
            unmapped.append(f"side_condition:{tok}")
    return out


def _set_status(mon: Pokemon, code: str, unmapped: list) -> None:
    if code == "fnt":
        mon._status = Status.FNT
    elif code not in ("nostatus", "", None):
        try:
            mon._status = Status[code.upper()]
        except KeyError:
            unmapped.append(f"status:{code}")


def to_battle(state: dict, *, tag: str = "metamon", turn: int = 1, last_move: Optional[str] = None,
              own_exact: bool = False, extras: Optional[dict] = None) -> Battle:
    """A poke-env Battle (our view, player p1) that carries the state's information. ``extras``
    (``history_extras``) adds what earlier steps of the trajectory show; without it those fields are marked
    unrecorded."""
    if extras is not None and last_move is None:
        last_move = extras.get("last_move")
    fmt = state.get("format") or "gen9ou"
    gen = _gen(fmt)
    data = GenData.from_gen(gen)
    b = Battle(tag, "p1user", _LOG, gen=gen)
    b._format = fmt
    b._player_role = "p1"
    b._turn = turn
    b._arena_source_state = state
    b.parse_request(build_request(state, data, last_move=last_move, own_exact=own_exact,
                                  own_fainted=(extras or {}).get("own_fainted") if extras is not None else None))
    unmapped = b._last_request.setdefault("unmapped", [])   # Metamon values with no meaning here, reported
    for entry in b._last_request["side"]["pokemon"]:      # poke-env reads neither teraType nor terastallized
        mon = b.team.get(entry["ident"])
        if mon is not None and entry.get("terastallized"):
            mon.terastallize(entry["terastallized"])
    me = b.active_pokemon
    if me is not None:
        for k, stat in _BOOSTS.items():
            me._boosts[stat] = int(state["player_active_pokemon"].get(k) or 0)
    opp_d = state["opponent_active_pokemon"]
    osid = name_id(opp_d["name"])
    opp: Pokemon = b.get_pokemon(f"p2a: {_species_name(data, osid)}", details=f"{_species_name(data, osid)}, "
                                 f"L{int(opp_d.get('lvl') or 100)}")
    opp._active = True
    opp._current_hp = round(float(opp_d.get("hp_pct") or 0) * 100)
    opp._max_hp = 100
    _set_status(opp, opp_d.get("status") or "nostatus", unmapped)
    for k, stat in _BOOSTS.items():
        opp._boosts[stat] = int(opp_d.get(k) or 0)
    for mv in opp_d.get("moves") or []:
        opp._add_move(name_id(mv["name"]))
    if opp_d.get("item") in ("nothing", "noitem"):
        opp._item = None                        # known to hold nothing (what was lost is not recorded)
    elif opp_d.get("item") not in (None, "", "unknownitem"):
        opp._item = name_id(opp_d["item"])
    if opp_d.get("ability") not in (None, "", "unknownability"):
        opp._ability = name_id(opp_d["ability"])
    _apply_types(opp, opp_d.get("types"), opp_d.get("tera_type") or "notype", unmapped)
    if me is not None and not me.is_terastallized:
        _apply_types(me, state["player_active_pokemon"].get("types"), "notype", unmapped)
    _add_history(b, state, data, extras, last_move)
    w = name_id(state.get("weather") or "noweather")
    if w != "noweather":
        if w in _WEATHERS:
            b._weather = {_WEATHERS[w]: 0}
        else:
            unmapped.append(f"weather:{w}")
    f = name_id(state.get("battle_field") or "nofield")
    if f != "nofield":
        if f in _FIELDS:
            b._fields = {_FIELDS[f]: 0}
        else:
            unmapped.append(f"field:{f}")
    b._side_conditions = {c: 1 for c in _conditions(state.get("player_conditions"), unmapped)}
    b._opponent_side_conditions = {c: 1 for c in _conditions(state.get("opponent_conditions"), unmapped)}
    return b


def _effect(mon: Pokemon, name: str, data: GenData, *, start_of_turn: bool = False,
            unmapped: Optional[list] = None) -> bool:
    """Start Metamon's effect id on ``mon`` as poke-env names it ("saltcure" -> SALT_CURE). Not started (and
    reported in ``unmapped``): an id poke-env has no volatile for (e.g. a side condition Metamon put in the
    slot), and a single-turn effect (Protect, Roost, Custap Berry ...) at a start-of-turn decision -- Metamon's
    slot keeps the most recent effect, which cannot still be active when a new turn begins."""
    from poke_env.battle.effect import Effect
    if not name or name == "noeffect":
        return False
    by_id = {name_id(k): k for k in Effect.__members__}
    if name not in by_id:
        if unmapped is not None:
            unmapped.append(f"effect:{name}")
        return False
    if start_of_turn and Effect[by_id[name]].ends_on_turn:
        if unmapped is not None:
            unmapped.append(f"stale_effect:{name}")
        return False
    mon.start_effect(by_id[name])
    return True


def _add_history(b: Battle, state: dict, data: GenData, extras: Optional[dict], last_move: Optional[str]) -> None:
    """Preview, effects, last moves, opponents seen earlier and Tera history, as the PlayerObservation reads
    them (poke-env objects plus a pre-filled tracker, since a converted battle has no protocol lines)."""
    from poke_env.battle.pokemon import Pokemon as _Pokemon
    from arena.observation.tracker import MonHistory, Tracker

    prev = state.get("opponent_teampreview") or []
    if prev:
        b._teampreview_opponent_team = {_Pokemon(gen=data.gen, species=_species_name(data, name_id(x))) for x in prev}
    tr = Tracker(player_role="p1")
    me, opp = b.active_pokemon, b.opponent_active_pokemon
    unmapped = b._last_request.setdefault("unmapped", [])
    start_of_turn = not state.get("forced_switch")
    if me is not None:
        _effect(me, name_id(state["player_active_pokemon"].get("effect") or ""), data, start_of_turn=start_of_turn,
                unmapped=unmapped)
        if last_move:                       # it moved last turn, so it has acted since it came in
            ident = next(e["ident"] for e in b._last_request["side"]["pokemon"] if e.get("active"))
            tr.mons[ident] = MonHistory(last_move=last_move, moved_since_entry=True)
    if opp is not None:
        _effect(opp, name_id(state["opponent_active_pokemon"].get("effect") or ""), data, start_of_turn=start_of_turn,
                unmapped=unmapped)
    ex = extras or {}
    if opp is not None and ex.get("opp_last_move"):
        tr.mons.setdefault(f"p2: {_species_name(data, name_id(state['opponent_active_pokemon']['name']))}",
                           MonHistory()).last_move = ex["opp_last_move"]
    for o in ex.get("opp_bench") or []:
        sid = name_id(o["name"])
        mon = b.get_pokemon(f"p2: {_species_name(data, sid)}", details=f"{_species_name(data, sid)}, L{int(o.get('lvl') or 100)}")
        mon._active = False
        mon._current_hp, mon._max_hp = round(float(o.get("hp_pct") or 0) * 100), 100
        _set_status(mon, o.get("status") or "nostatus", b._last_request.setdefault("unmapped", []))
        if mon._status == Status.FNT:
            mon._current_hp = 0
        for mv in o.get("moves") or []:
            mon._add_move(name_id(mv["name"]))
        if o.get("item") in ("nothing", "noitem"):
            mon._item = None
        elif o.get("item") not in (None, "", "unknownitem"):
            mon._item = name_id(o["item"])
        if o.get("ability") not in (None, "", "unknownability"):
            mon._ability = name_id(o["ability"])
        if (o.get("tera_type") or "notype") != "notype":
            mon.terastallize(o["tera_type"])
        tr.mons.setdefault(f"p2: {_species_name(data, sid)}", MonHistory()).hp_turn = o.get("_seen_step")
    for m in ex.get("own_unlisted") or []:    # in the battle but not offered as a switch now: listed, not an option
        sid = name_id(m["name"])
        base = _species_name(data, name_id(m.get("base_species") or (data.pokedex.get(sid) or {}).get("baseSpecies") or sid))
        if f"p1: {base}" not in b.team:
            mon = _Pokemon(gen=data.gen, species=_species_name(data, sid))
            mon._current_hp, mon._max_hp = round(float(m.get("hp_pct") or 0) * 100), 100
            mon._item = None if m.get("item") in ("nothing", "noitem") else name_id(m.get("item") or "")
            mon._ability = name_id(m.get('ability') or '') or None
            mon._level = int(m.get('lvl') or 100)
            for move in m.get('moves') or []:
                mon._add_move(name_id(move['name']))
            b._team[f"p1: {base}"] = mon
            entry = _req_mon(data, m, False, bool(state.get('can_tera')), unmapped)
            entry['policy_last_seen'] = True
            b._last_request['side']['pokemon'].append(entry)
    if not state.get("can_tera"):
        key, mine = next(((k, m) for k, m in b.team.items() if m.is_terastallized), (None, None))
        hist = ex.get("own_tera")
        if key is None and hist:
            key = next((k for k, m in b.team.items() if name_id(m.species) == name_id(hist["mon"])), None)
        tr.tera["p1"] = {"mon": key or "p1: (earlier Pokemon)",
                         "type": (mine.tera_type.name.title() if mine and mine.tera_type else
                                  hist["type"].title() if hist else "unknown"), "turn": None}
    if ex.get("opp_tera"):
        t = ex["opp_tera"]
        tr.tera["p2"] = {"mon": f"p2: {_species_name(data, name_id(t['mon']))}", "type": t["type"].title(), "turn": None}
    b._arena_tracker = tr
    if ex.get("opp_unattributed_faints"):        # which opponent Pokemon fainted (or came back) is not kept
        b._arena_unattributed_faints = int(ex["opp_unattributed_faints"])
        b._last_request.setdefault("unrecorded", []).append("opp_faint_identity")
    if extras is None:
        b._last_request.setdefault("unrecorded", []).extend(["tera_history", "opp_bench", "last_moves"])
    else:                                       # history from the trajectory: filled only where certain
        # Tera: inferred from the tera action and the types the format keeps (a Tera into the Pokemon's own
        # typing leaves no trace), so it is a reconstruction, not a record
        b._last_request.setdefault("unrecorded", []).append("tera_history_inferred")
        if not last_move:
            b._last_request.setdefault("unrecorded", []).append("last_move")
        if not ex.get("opp_last_move"):
            b._last_request.setdefault("unrecorded", []).append("opp_last_move")


def _most_recent(rep: dict) -> str:
    return name_id(max(rep, key=rep.get).name) if rep else ""


def _universal_mon(m, own: bool) -> dict:
    types = [t.name.lower() for t in m.types if t is not None]
    base = (GenData.from_gen(9).pokedex.get(m.species) or {}).get("baseSpecies") or m.species
    d = {"name": m.species, "base_species": name_id(base),   # Metamon keeps both (Mimikyu-Busted: mimikyu)
         "hp_pct": m.current_hp_fraction or 0.0, "types": " ".join(types + ["notype"] * (2 - len(types))),
         "item": (m.item or "nothing") if own else ("unknownitem" if m.item == "unknown_item" else (m.item or "nothing")),
         "ability": (m.ability or "") if own else (m.ability or "unknownability"), "lvl": m.level,
         "status": m.status.name.lower() if m.status else ("fnt" if m.fainted else "nostatus"), "effect": "noeffect",
         "moves": [{"name": mv.id, "current_pp": mv.current_pp, "max_pp": mv.max_pp} for mv in m.moves.values()],
         "tera_type": (m.tera_type.name.lower() if getattr(m, "tera_type", None) else "notype")}
    for k, stat in (("atk_boost", "atk"), ("spa_boost", "spa"), ("def_boost", "def"), ("spd_boost", "spd"),
                    ("spe_boost", "spe"), ("accuracy_boost", "accuracy"), ("evasion_boost", "evasion")):
        d[k] = (m.boosts or {}).get(stat, 0)
    return d


def from_battle(b) -> dict:
    """A poke-env Battle (player view) -> the UniversalState fields Metamon keeps (inverse of ``to_battle``;
    used to validate the round trip, and to label our own battles with Metamon policies)."""
    me, opp = b.active_pokemon, b.opponent_active_pokemon
    req = getattr(b, "_last_request", None) or {}
    tera = next((p.get("teraType") for p in (req.get("side") or {}).get("pokemon", []) if p.get("active")), None)
    tera_of = {e["ident"]: e.get("teraType") for e in (req.get("side") or {}).get("pokemon", [])}
    pm = _universal_mon(me, True)
    if tera:
        pm["tera_type"] = name_id(tera)
    if req.get("active"):     # the request's own move list (PP, disabled)
        pm["moves"] = [{"name": mv["id"], "current_pp": mv.get("pp", 1), "max_pp": mv.get("maxpp", 1)}
                       for mv in req["active"][0].get("moves", [])]
    from arena.observation.tracker import tracker_for
    tr = tracker_for(b)
    def first_effect(mon) -> str:            # Metamon keeps one effect per active Pokemon
        names = [name_id(e.name) for e in (getattr(mon, "effects", None) or {})]
        return names[0] if names else "noeffect"
    def prev_move(side: str) -> dict:
        key = tr.active.get(side, (None,))[0]
        h = tr.mons.get(key) if key else None
        return {"name": h.last_move if h and h.last_move else "nomove"}
    pm["effect"] = first_effect(me)
    opp_u = _universal_mon(opp, False)
    opp_u["effect"] = first_effect(opp)
    role = b.player_role or "p1"
    return {"format": b.format, "player_active_pokemon": pm, "opponent_active_pokemon": opp_u,
            "player_prev_move": prev_move(role), "opponent_prev_move": prev_move("p2" if role == "p1" else "p1"),
            "opponent_teampreview": [name_id(p.species) for p in getattr(b, "teampreview_opponent_team", None) or []],
            "available_switches": [{**_universal_mon(p, True), **({"tera_type": name_id(tera_of[k])} if tera_of.get(k) else {})}
                                   for k, p in b.team.items() if p in b.available_switches],
            "opponents_remaining": 6 - sum(1 for p in b.opponent_team.values() if p.fainted),
            # Metamon keeps one side condition per side and one field: the entry with the largest value
            # (UniversalState.universal_conditions / universal_field), whatever the others are
            "player_conditions": _most_recent(b.side_conditions) or "noconditions",
            "opponent_conditions": _most_recent(b.opponent_side_conditions) or "noconditions",
            "weather": name_id(next(iter(b.weather)).name) if b.weather else "noweather",
            "battle_field": _most_recent(b.fields) or "nofield",
            "forced_switch": bool(b.force_switch) or not b.available_moves, "can_tera": bool(b.can_tera)}



def action_key(state: dict, action: int) -> Optional[str]:
    """Metamon action index -> our option key (``move_x``, ``move_x_tera``, ``switch_y``); None if unknown."""
    if action is None or action < 0:
        return None
    if state.get("forced_switch") and (action <= 3 or action >= 9):
        return None                         # a move during a forced switch: the environment replaced it
    if action >= 9 and not state.get("can_tera"):
        action -= 9                         # tera no longer available: Showdown plays the plain move
    moves = sorted((name_id(m["name"]) for m in state["player_active_pokemon"].get("moves") or []))
    switches = sorted(name_id(p["name"]) for p in state.get("available_switches") or [])
    if action <= 3:
        return f"move_{moves[action]}" if action < len(moves) else None
    if 4 <= action <= 8:
        return f"switch_{switches[action - 4]}" if action - 4 < len(switches) else None
    if 9 <= action <= 12:
        return f"move_{moves[action - 9]}_tera" if action - 9 < len(moves) else None
    return None


def convert_state(state: dict, *, tag: str = "metamon", turn: int = 1, last_move: Optional[str] = None,
                  own_exact: bool = False, extras: Optional[dict] = None, resource_ledger=None):
    """-> (state dict, questions, option keys, battle) exactly as the live System-1 agent would send them.
    ``last_move``: see ``last_move_of_active`` (pass it whenever the trajectory is at hand); ``own_exact``:
    see ``build_request`` (``is_selfplay(file name)``)."""
    from arena.agents.system1.decider import FastDecider, RouteConfig
    from arena.state.views import legal_options

    battle = to_battle(state, tag=tag, turn=turn, last_move=last_move, own_exact=own_exact, extras=extras)
    if resource_ledger is not None:
        battle._arena_policy_resources = resource_ledger
    options = legal_options(battle)
    st, questions, facts = FastDecider(None, RouteConfig()).build_request(battle, options)
    st.setdefault("opponent", {})["pokemon_left"] = int(state.get("opponents_remaining") or 6)
    return st, questions, [f.key for f in facts], battle


def iter_examples(paths: Iterable[str | Path], *, skip_unknown: bool = True, errors: Optional[dict] = None):
    """Yield one record per decision: {"battle", "step", "state", "questions", "keys", "action_key", "won"}.
    Records left out are counted in ``errors`` (reason -> count) when given: unreadable files, conversion
    errors (by exception type), unknown recorded actions, decisions with a single option."""
    def skip(reason: str) -> None:
        if errors is not None:
            errors[reason] = errors.get(reason, 0) + 1

    for p in paths:
        name = Path(p).name
        won = None if not re.search(r"_(WIN|LOSS)\.json", name) else name.endswith("_WIN.json.lz4")
        try:
            traj = read_trajectory(p)
        except Exception as e:  # noqa: BLE001
            skip(f"unreadable file: {type(e).__name__}")
            continue
        for i, (s, a) in enumerate(zip(traj["states"], traj["actions"])):
            key = action_key(s, a)
            if key is None and skip_unknown:
                skip("recorded action unknown")
                continue
            try:
                st, qs, keys, _b = convert_state(s, tag=name, turn=i + 1, own_exact=is_selfplay(name),
                                                 extras=history_extras(traj["states"], i, traj["actions"]))
            except Exception as e:  # noqa: BLE001
                skip(f"conversion error: {type(e).__name__}")
                continue
            if len(keys) < 2:
                skip("single option")
                continue
            yield {"battle": name, "step": i, "state": st, "questions": qs, "keys": keys, "action_key": key,
                   "action_legal": key in keys if key else None, "won": won, "format": s.get("format")}


def policy_targets(state: dict, probs: list[float], questions: dict, *, with_mass: bool = False) -> dict:
    """A Metamon policy's distribution over its 13 actions -> soft targets for our questions.

      stay          P(any move, tera or not) vs P(any switch), renormalised over the legal ones
      best_move     P(move k) + P(move k with tera), over the moves offered by the question
      best_switch   P(switch j), over the switch-ins offered
      (``_rev`` questions get the same distribution in their own option order; ``hard`` has no target)
    Returns {qid: [p per option in the question's order]} for the questions it can label. With
    ``with_mass=True``: {qid: (target, mass)} where mass is the probability of the branch the question is
    conditioned on (P(attack) for best_move, P(switch) for best_switch, 1 for stay) -- a conditional target
    built from a branch the policy all but rules out is mostly the policy's probability floor, not a preference.
    """
    moves = sorted(name_id(m["name"]) for m in state["player_active_pokemon"].get("moves") or [])
    switches = sorted(name_id(p["name"]) for p in state.get("available_switches") or [])
    p_key: dict = {}
    for i, mv in enumerate(moves[:4]):
        p_key[f"move_{mv}"] = float(probs[i])
        p_key[f"move_{mv}_tera"] = float(probs[9 + i]) if len(probs) > 9 + i else 0.0
    for j, sp in enumerate(switches[:5]):
        p_key[f"switch_{sp}"] = float(probs[4 + j])
    offered = [k for q in questions.values() if q["type"] == "choice" for k in q["criteria"]]
    total = sum(p_key.get(k, 0.0) for k in set(offered)) or 1.0
    out: dict = {}
    for qid, q in questions.items():
        base = qid[:-4] if qid.endswith("_rev") else qid
        if base in ("best_move", "best_switch", "best"):
            keys = list(q["criteria"])
            vals = [p_key.get(k, 0.0) for k in keys]
            z = sum(vals)
            if z > 0:
                out[qid] = ([v / z for v in vals], 1.0 if base == "best" else z / total)
        elif base == "stay":
            mv = {k for k in offered if k.startswith("move_")}
            sw = {k for k in offered if k.startswith("switch_")}
            pm, ps = sum(p_key.get(k, 0.0) for k in mv), sum(p_key.get(k, 0.0) for k in sw)
            if pm + ps > 0:
                out[qid] = ([pm / (pm + ps), ps / (pm + ps)], 1.0)      # options are (true = attack, false = switch)
    return out if with_mass else {q: v[0] for q, v in out.items()}
