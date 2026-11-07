"""Causal decision history and full-action supervision shared by collection and deployment."""

from __future__ import annotations

from collections import deque
from copy import deepcopy

from arena.llm.typesafe import choice

POLICY_INSTRUCTIONS = (
    "You control `you`. Choose the legal action that gives you the best chance to win the whole battle. "
    "Use the recent decisions and revealed information. Consider future matchups, preserving essential "
    "Pokemon, recovery, setup, switching, and the once-per-battle Terastallization resource. "
    "Damage estimates are conditional estimates, not guaranteed outcomes."
)


def full_action_question(questions: dict) -> dict:
    """Keep legal option keys, including ordinary and Tera moves, in a single Choice."""
    if "best" in questions:
        criteria = deepcopy(questions["best"]["criteria"])
    else:
        criteria = {}
        for qid in ("best_move", "best_switch"):
            criteria.update(deepcopy((questions.get(qid) or {}).get("criteria") or {}))
    return choice(POLICY_INSTRUCTIONS, criteria)


def full_action_target(universal: dict, probabilities: list, keys: list[str], *,
                       index_keys: list | None = None) -> tuple[list[float], float]:
    from arena.s1.convert_metamon import action_key

    mass = dict.fromkeys(keys, 0.0)
    for index, probability in enumerate(probabilities):
        if index_keys is None and index >= 9 and not universal.get("can_tera"):
            continue
        key = index_keys[index] if index_keys is not None else action_key(universal, index)
        if key in mass:
            mass[key] += float(probability)
    total = sum(mass.values())
    if total <= 0:
        raise ValueError("teacher assigns no probability to the recorded legal actions")
    return [mass[key] / total for key in keys], max(0.0, 1.0 - total)


def omit_unrecorded_estimates(state: dict, question: dict) -> tuple[dict, dict]:
    """Historical Metamon records do not contain the actual EVs or battle stats."""
    state, question = deepcopy(state), deepcopy(question)
    for key in ("speed", "threat"):
        state.pop(key, None)
    state["own_stats"] = "not recorded; numeric damage and speed estimates are omitted"
    for fields in question["criteria"].values():
        for key in ("damage", "order", "risk", "speed", "offense"):
            fields.pop(key, None)
    return state, question


def compact_state(state: dict) -> dict:
    result = {}
    for side in ("you", "opponent"):
        whole = state.get(side) or {}
        active = whole.get("active") or {}
        result[side] = {key: active.get(key) for key in ("species", "hp", "status", "last_move")}
        result[side]["left"] = whole.get("pokemon_left")
    return result


class DecisionHistory:
    """The last completed decisions, their submitted actions and observed successor states."""

    def __init__(self, size: int = 8):
        self.size = size
        self.rows = deque(maxlen=size + 1)

    def observe(self, decision, state: dict) -> list[str]:
        snap = compact_state(state)
        if self.rows and self.rows[-1]["decision"] == decision:
            self.rows[-1]["state"] = snap
        else:
            self.rows.append({"decision": decision, "state": snap, "action": None})
        return self.events()

    def events(self) -> list[str]:
        rows = list(self.rows)
        events = []
        for before, after in zip(rows, rows[1:]):
            old, new = before["state"], after["state"]
            a, b = old["you"], old["opponent"]
            c, d = new["you"], new["opponent"]
            opponent_move = d.get("last_move") if b["species"] == d["species"] else "switched"
            events.append(
                f"{a['species']} {a['hp']} vs {b['species']} {b['hp']}; "
                f"played {before['action'] or 'unrecorded'}; "
                f"next {c['species']} {c['hp']} / {d['species']} {d['hp']}; "
                f"opp last {opponent_move or 'unknown'}; left {c['left']}/{d['left']}"
            )
        return events

    def acted(self, key: str | None, *, decision=None) -> None:
        if self.rows:
            if decision is not None and self.rows[-1]['decision'] != decision:
                raise ValueError('submitted action belongs to a different decision')
            self.rows[-1]["action"] = key


def history_for(battle, size: int = 8) -> DecisionHistory:
    history = getattr(battle, "_arena_policy_history", None)
    if history is None:
        history = battle._arena_policy_history = DecisionHistory(size)
    return history


def add_history(battle, state: dict, size: int) -> dict:
    history = history_for(battle, size)
    recent = history.events() if battle.finished else history.observe(decision_key(battle), state)
    return {**state, 'recent_decisions': recent}


def decision_key(battle):
    request = getattr(battle, "_last_request", None) or {}
    return request.get('rqid', battle.turn)


def begin_policy_decision(battle, size: int) -> None:
    """Create the current row before routing, including forced, heuristic and retry paths."""
    if not size or battle.finished:
        return
    history = history_for(battle, size)
    if not history.rows or history.rows[-1]['decision'] != decision_key(battle):
        from arena.agents.system1.facts import build_state
        add_history(battle, build_state(battle), size)


def record_policy_action(battle, key: str | None, size: int) -> None:
    if not size or battle.finished:
        return
    begin_policy_decision(battle, size)
    history_for(battle, size).acted(key, decision=decision_key(battle))


def record_policy_submission(battle, message: str) -> None:
    """Record the sent choice, including client substitutions after a rejected order."""
    history = getattr(battle, '_arena_policy_history', None)
    if history is None or not message.startswith('/choose '):
        return
    from arena.observation.trace import choice_action_id, ident_name
    from poke_env.data import to_id_str
    request = battle._last_request
    action = choice_action_id(message.split('|', 1)[0], request)
    if action and action.startswith('move:'):
        key = 'move_' + action[5:].replace('+tera', '_tera')
    elif action and action.startswith(('switch:', 'revive:')):
        ident = action.split(':', 1)[1]
        mon = next(m for m in request['side']['pokemon'] if ident_name(m['ident']) == ident)
        key = 'switch_' + to_id_str(mon['details'].split(',', 1)[0])
    else:
        key = action or 'unrecorded'
    record_policy_action(battle, key, history.size)


def reject_policy_action(battle) -> None:
    """A refused attempt has no successor; its replacement starts at the same decision state."""
    history = getattr(battle, '_arena_policy_history', None)
    if history and history.rows and not history.rows[-1].get('terminal'):
        history.rows.pop()


def finish_policy_history(battle, size: int) -> None:
    """Attach the terminal successor from the resolved battle, not its last pre-action request."""
    if not size or not getattr(battle, '_arena_policy_history', None):
        return
    from arena.observation.tracker import mon_key, tracker_for
    tracker = tracker_for(battle)
    state = {}
    for side, role, active, team in (('you', battle.player_role, battle.active_pokemon, battle.team),
                                     ('opponent', battle.opponent_role, battle.opponent_active_pokemon, battle.opponent_team)):
        total = battle._team_size.get(role, len(team) if side == 'you' else 6)
        state[side] = {'active': {}, 'pokemon_left': total - sum(m.fainted for m in team.values())}
        if active is not None:
            ident = next(k for k, mon in team.items() if mon is active)
            observed = tracker.mons.get(mon_key(ident))
            state[side]['active'] = {'species': active.species,
                'hp': f'{round(100 * (active.current_hp_fraction or 0))}%',
                'status': active.status.name if active.status else None,
                'last_move': observed.last_move if observed else None}
    history = history_for(battle, size)
    history.observe(('terminal', decision_key(battle)), state)
    history.rows[-1]['terminal'] = True
    history.rows[-1]['won'] = battle.won


def capture_universal(battle) -> dict:
    """Kakuna observes every living own reserve; the action mask separately records trapping."""
    from arena.s1.convert_metamon import _universal_mon, from_battle, name_id

    state = from_battle(battle)
    tera = {entry["ident"]: entry.get("teraType")
            for entry in battle._last_request["side"]["pokemon"]}
    state["available_switches"] = []
    for ident, mon in battle.team.items():
        if mon.active or mon.fainted:
            continue
        record = _universal_mon(mon, True)
        if tera.get(ident):
            record["tera_type"] = name_id(tera[ident])
        state["available_switches"].append(record)
    return state
