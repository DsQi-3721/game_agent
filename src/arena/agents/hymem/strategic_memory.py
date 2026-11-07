"""Conditional strategic experiences grounded in executed actions and observed outcomes."""
from copy import deepcopy
import json
from pathlib import Path
import threading


class StrategicExperienceStore:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines() if line] if self.path.exists() else []
        self.lock = threading.Lock()

    def append(self, rows):
        if not rows:
            return
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a') as file:
                for row in rows:
                    file.write(json.dumps(row, ensure_ascii=False) + '\n')
                    self.rows.append(deepcopy(row))

    def retrieve(self, state, limit=4):
        selected = []
        with self.lock:
            rows = list(self.rows)
        for row in reversed(rows):
            condition = row['conditions']; actor = state['own'].get(condition['actor'])
            target = condition.get('target')
            if (row['battle'] == state['battle'] or not actor or not actor['alive']
                    or not set(condition['required_moves']).issubset(actor['moves'])):
                continue
            if target and (target not in state['opponent'] or not state['opponent'][target]['alive']):
                continue
            selected.append(deepcopy(row))
            if len(selected) == limit:
                break
        return selected


def experience(plan, stage, actions, after, result, events, number):
    if not actions:
        return None
    before = actions[0]['before']; actor = stage['actor']; target = stage.get('target', '')
    from arena.agents.hymem.strategic_execution import route_moves
    return {'id': f"{plan.id}:experience:{number}", 'battle': before['battle'],
            'conditions': {'actor': actor, 'target': target, 'own_hp_at_start': before['own'][actor]['hp'],
                           'required_moves': route_moves(stage, before),
                           'opponent_known_moves': before['opponent'].get(target, {}).get('moves', [])},
            'method': {'intent': stage['intent'], 'goal': deepcopy(stage['complete_when']),
                       'actions': [a['action'] for a in actions]},
            'result': {'status': result, 'start_decision': actions[0]['decision'],
                       'end_decision': after['decision'], 'own_remaining': after['pokemon_left']['own'],
                       'opponent_remaining': after['pokemon_left']['opponent'],
                       'goal_observations': [{**c, 'observed': _observed(c, after)} for c in stage['complete_when']]},
            'exceptions': [{'event': e['id'], 'protocol': e['protocol']} for e in events
                           if e['turn'] >= actions[0]['turn'] and e['kind'] in ('-immune', '-fail', 'faint', '-heal')][-8:]}


def _observed(condition, state):
    from arena.agents.hymem.strategic_plan import condition_value
    return condition_value(condition, state)
