"""Two-stage searches of identical root states with explicit action comparisons."""
from itertools import combinations
import time

from poke_engine import MctsResult, State, monte_carlo_tree_search


def to_key(name):
    if name.startswith('switch '):
        return 'switch_' + ''.join(c for c in name[7:].lower() if c.isalnum())
    return 'move_' + name[:-5] + '_tera' if name.endswith('-tera') else 'move_' + name


def search_sample(state, milliseconds, actions=(), initial=None):
    started, cpu = time.perf_counter(), time.process_time()
    native = State.from_string(state)
    focused = []
    if actions and initial:
        values = {k: v['total_score'] / v['visits'] for k, v in initial['ours'].items() if v['visits']}
        focused = list(dict.fromkeys([k for k in actions if k in values] + [max(values, key=values.get)]))
    if len(focused) >= 2:
        from poke_engine import mcts_compare
        names = [initial['names'][k] for k in focused]
        result = MctsResult._from_rust(mcts_compare(native, milliseconds, names))
    else:
        result = monte_carlo_tree_search(native, milliseconds, threads=1)
    def side(options):
        return {to_key(o.move_choice): {'visits': o.visits, 'total_score': o.total_score} for o in options}
    return {'ours': side(result.side_one), 'theirs': side(result.side_two),
            'names': {to_key(o.move_choice): o.move_choice for o in result.side_one},
            'visits': result.total_visits, 'focused_actions': focused if len(focused) >= 2 else [],
            'wall_s': time.perf_counter() - started, 'cpu_s': time.process_time() - cpu}


def merge(first, second):
    result = {k: first[k] + second[k] for k in ('visits', 'wall_s', 'cpu_s')}
    result['names'] = {**first['names'], **second['names']}
    result['focused_actions'] = second['focused_actions']
    for side in ('ours', 'theirs'):
        result[side] = {}
        for key in first[side].keys() | second[side].keys():
            result[side][key] = {field: first[side].get(key, {}).get(field, 0)
                                + second[side].get(key, {}).get(field, 0)
                                for field in ('visits', 'total_score')}
    return result


def summarize(states, results, milliseconds, task=None, stage='complete', preparation_cpu_s=0):
    caller_cpu = time.thread_time()
    total_weight = sum(weight for state, weight in states)
    policies, opponents, evidence = {}, {}, []
    for index, ((state, weight), result) in enumerate(zip(states, results)):
        weight /= total_weight
        policy = {}
        for key, value in result['ours'].items():
            share = value['visits'] / max(1, result['visits'])
            score = value['total_score'] / value['visits'] if value['visits'] else None
            policy[key] = {'share': share, 'score': score, 'visits': value['visits']}
            accumulated = policies.setdefault(key, [0., 0., 0., 0])
            accumulated[0] += weight * share
            accumulated[3] += value['visits']
            if score is not None:
                accumulated[1] += weight * score
                accumulated[2] += weight
        for key, value in result['theirs'].items():
            opponents[key] = opponents.get(key, 0.) + weight * value['visits'] / max(1, result['visits'])
        native = State.from_string(state)
        active = native.side_two.pokemon[int(native.side_two.active_index)]
        evidence.append({'sample': index, 'weight': weight,
                         'assumed_active_set': {'species': active.id, 'item': active.item,
                                                'moves': [m.id for m in active.moves if m.id != 'none']},
                         'focused_actions': result['focused_actions'], 'policy': policy})
    policy = {key: {'share': value[0], 'score': value[1] / value[2] if value[2] else None,
                    'visits': value[3]} for key, value in policies.items()}
    comparison = None
    if task and stage == 'complete' and any(r['focused_actions'] for r in results):
        pairs = []
        for a, b in combinations(task['actions'], 2):
            if a not in policy or b not in policy or policy[a]['score'] is None or policy[b]['score'] is None:
                continue
            differences = [e['policy'][a]['score'] - e['policy'][b]['score'] for e in evidence
                           if a in e['policy'] and b in e['policy']
                           and e['policy'][a]['score'] is not None and e['policy'][b]['score'] is not None]
            pairs.append({'actions': [a, b], 'mean_score_difference': policy[a]['score'] - policy[b]['score'],
                          'samples_favoring_first': sum(d > 0 for d in differences),
                          'samples_favoring_second': sum(d < 0 for d in differences)})
        comparison = {'task_id': task['id'], 'requested_actions': task['actions'], 'pairs': pairs,
                      'values': {k: policy[k] for k in task['actions'] if k in policy},
                      'root_basis': 'same current state and configuration weights in both stages'}
    return {'policy': policy, 'opp_policy': opponents, 'evidence': evidence, 'task': task,
            'stage': stage, 'comparison': comparison, 'iterations': [r['visits'] for r in results],
            'compute': {'samples': len(states), 'search_ms': milliseconds, 'threads': 1,
                        'budget_ms': len(states) * milliseconds,
                        'worker_cpu_s': sum(r['cpu_s'] for r in results),
                        'worker_wall_s': sum(r['wall_s'] for r in results),
                        'preparation_and_aggregation_cpu_s': preparation_cpu_s + time.thread_time() - caller_cpu}}
