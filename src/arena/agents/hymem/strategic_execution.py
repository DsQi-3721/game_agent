"""Compile the active strategic phase into current actions and descendant search routes."""
from copy import deepcopy
from poke_env.battle.move import Move

from arena.agents.hymem.strategic_state import SETUP, RECOVERY, HAZARDS, REMOVAL, PIVOT


def route_moves(route, state):
    moves = set(state['own'][route['actor']]['moves'])
    intent = route['intent']
    group = {'setup': SETUP, 'recover': RECOVERY, 'set_hazards': HAZARDS,
             'remove_hazards': REMOVAL, 'pivot': PIVOT}.get(intent)
    if group is not None:
        return sorted(moves & group)
    return sorted(mid for mid in moves if Move(mid, gen=9).category.name != 'STATUS'
                  or mid in {'toxic', 'willowisp', 'leechseed'})


def compile_execution(plan, state, legal_keys=()):
    if plan is None or plan.status != 'active' or plan.stage is None:
        return None
    stage = plan.stage
    routes = []
    for index, route in enumerate(stage['branches'] + [stage]):
        routes.append({'name': f"{stage['id']}:{index}:{route['actor']}:{route['intent']}",
                       'actor': route['actor'], 'intent': route['intent'], 'moves': route_moves(route, state),
                       'when': deepcopy(route.get('when', [])),
                       'switch_out': route['intent'] == 'pivot' and any(c['metric'] == 'active'
                           and c['pokemon'] == route['actor'] and c['value'] is False for c in stage['complete_when'])})
    execution = plan.execution(state)
    preferred = []
    if execution:
        actor = execution['actor']; moves = route_moves(execution, state)
        for key in legal_keys:
            if not state['own'][actor]['active']:
                match = key == 'switch_' + actor
            else:
                match = key.startswith('move_') and key[5:].removesuffix('_tera') in moves
            if match:
                preferred.append(key)
    goal = deepcopy(stage['complete_when'])
    return {'plan_id': plan.id, 'source_decision': plan.source_decision, 'decision': state['decision'],
            'stage': stage['id'], 'goal': goal, 'routes': routes, 'candidate_actions': preferred,
            'student_facts': [f"Strategic objective: {plan.definition['win_condition']}",
                              f"Current phase: {stage['objective']}. Preserve: {', '.join(plan.definition['members'])}."]}
