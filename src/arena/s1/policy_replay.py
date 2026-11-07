"""Rebuild stored policy inputs from the same player messages used at runtime."""
from __future__ import annotations

from copy import deepcopy
import logging

from poke_env.battle import Battle
from poke_env.data import GenData

from arena.agents.system1.decider import FastDecider, RouteConfig
from arena.s1.policy_data import (DecisionHistory, full_action_question, history_for,
                                 omit_unrecorded_estimates)
from arena.state.views import legal_options


def rebuild_native(rows, history_events=8):
    """Yield refreshed rows in request order; labels and accepted actions stay attached to their keys."""
    battle, cursor = None, 0
    decider = FastDecider(None, RouteConfig(mode='flat', debias=False, ask_hard=False,
                                          history_events=history_events))
    for row in rows:
        if battle is None:
            role = row['request']['side']['id']
            username = next(event[3] for event in row['public_prefix']
                            if len(event) > 3 and event[1:3] == ['player', role])
            fmt = row['universal']['format']
            battle = Battle(row['battle_tag'], username, logging.getLogger(__name__),
                            gen=GenData.from_format(fmt).gen)
            battle._format = fmt
        if row['battle_tag'] != battle.battle_tag:
            raise ValueError('rebuild_native expects one battle in request order')
        prefix = row['public_prefix']
        for event in prefix[cursor:]:
            battle.parse_message(list(event))
        cursor = len(prefix)
        battle.parse_request(deepcopy(row['request']))
        state, questions, facts = decider.build_request(battle, legal_options(battle))
        keys = [f.key for f in facts]
        if keys != row['keys']:
            raise ValueError(f"legal keys changed: {row['battle_tag']}:{row['request']['rqid']}")
        refreshed = {**row, 'state': state, 'question': questions['best']}
        history_for(battle, history_events).acted(row['played_key'])
        yield refreshed


def rebuild_historical(rows, history_events=8):
    """Reconstruct every preceding decision before refreshing sampled historical positions."""
    from arena.s1.convert_metamon import (action_key, convert_state, history_extras,
                                          is_selfplay, read_trajectory)
    from arena.s1.resources import ResourceLedger

    rows = list(rows)
    path = rows[0]['provenance']['record']
    trajectory = read_trajectory(path)
    states, actions = trajectory['states'], trajectory['actions']
    selected = {r['provenance']['step']: r for r in rows}
    history, resources = DecisionHistory(history_events), ResourceLedger()
    for step in range(max(selected) + 1):
        state, questions, _, _ = convert_state(states[step], tag=path, turn=step + 1,
            own_exact=is_selfplay(path), extras=history_extras(states, step, actions), resource_ledger=resources)
        state['recent_decisions'] = history.observe(step, state)
        history.acted(action_key(states[step], actions[step]))
        if step not in selected:
            continue
        row = selected[step]
        state, question = omit_unrecorded_estimates(state, full_action_question(questions))
        if list(question['criteria']) != row['keys']:
            raise ValueError(f"legal keys changed: {path}:{step}")
        yield {**row, 'state': state, 'question': question,
               'provenance': {**row['provenance'], 'history_decisions': len(state['recent_decisions'])}}
