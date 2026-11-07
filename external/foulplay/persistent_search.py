"""Continue one native tree per sampled state, with live S1 and strategic guidance."""
import asyncio
import json
import time
from copy import deepcopy

from poke_engine import State, SearchTree
from action_comparison import summarize


def native_plan(plan, state):
    if not plan:
        return None
    teams = [state.side_one.pokemon, state.side_two.pokemon]
    def index(side, name):
        indices = {mon.id.lower(): i for i, mon in enumerate(teams[side])}
        return indices[name]
    def condition(c):
        side = 0 if c['side'] == 'own' else 1
        return {**c, 'side': side, 'pokemon': index(side, c['pokemon']) if c.get('pokemon') else -1,
                'value': float(c['value'])}
    return {'goal': [condition(c) for c in plan['goal']],
            'routes': [{**route, 'actor': index(0, route['actor']),
                        'when': [condition(c) for c in route['when']]} for route in plan['routes']]}


def run_sample(state, milliseconds, plan, guidance, publish, stale, seed):
    tree = SearchTree(state, json.dumps(native_plan(plan, State.from_string(state))), seed)
    cpu = time.thread_time(); wall = time.perf_counter()
    first = min(200, max(1, milliseconds // 2))
    used = 0.; initial_visits = 0; first_result = None
    result = None
    while used < milliseconds and not stale():
        chunk = first if not initial_visits else min(100, max(1, int(milliseconds - used)))
        prior = dict(guidance.get('student', {}))
        for action in guidance.get('comparison_actions', []):
            prior[action] = prior.get(action, 0.) + 1.
        result = json.loads(tree.advance(chunk, json.dumps(prior)))
        used += result['slice_wall_s'] * 1000
        result.update(wall_s=time.perf_counter() - wall, cpu_s=time.thread_time() - cpu,
                      focused_actions=list(guidance.get('comparison_actions', [])), initial_visits=initial_visits,
                      search_elapsed_ms=used)
        if not initial_visits:
            initial_visits = result['visits']; result['initial_visits'] = initial_visits
            first_result = deepcopy(result)
            publish(first_result)
    if result:
        result['initial_visits'] = initial_visits
        result['retained_visits'] = first_result['visits']
    return result


async def progressive_search(states, milliseconds, plan, guidance, pool, emit, stale, preparation, seed, task=None):
    loop = asyncio.get_running_loop()
    initial = asyncio.Queue()
    jobs = []
    for i, (state, _) in enumerate(states):
        def publish(result, index=i):
            loop.call_soon_threadsafe(initial.put_nowait, (index, result))
        jobs.append(loop.run_in_executor(pool, run_sample, state, milliseconds, plan, guidance,
                                         publish, stale, seed + i))
    # A worker failure must reach the HTTP caller instead of leaving the initial queue waiting.
    async def collect_initial():
        results = [None] * len(states)
        for _ in states:
            i, result = await initial.get(); results[i] = result
        return results
    reader = asyncio.create_task(collect_initial())
    workers = asyncio.gather(*jobs)
    try:
        done, _ = await asyncio.wait({reader, workers}, return_when=asyncio.FIRST_COMPLETED)
        if workers in done:
            full = workers.result()
            if not reader.done():
                await asyncio.sleep(0)
            if not reader.done():
                return
        first = await reader
        if stale():
            return
        await emit(summary(states, first, min(200, max(1, milliseconds // 2)), plan, 'initial', preparation, task))
        if task:
            guidance['comparison_actions'] = task['actions']
        full = await workers
        if not stale() and all(full):
            await emit(summary(states, full, milliseconds, plan, 'complete', preparation, task))
    finally:
        reader.cancel()


def summary(states, results, milliseconds, plan, stage, preparation, task=None):
    out = summarize(states, results, milliseconds, task, stage=stage, preparation_cpu_s=preparation)
    total = sum(weight for _, weight in states)
    for key, row in out['policy'].items():
        visits = [result['ours'].get(key, {}).get('visits', 0) for result in results]
        row['min_sample_visits'] = min(visits)
        row['coverage'] = sum(weight for (_, weight), n in zip(states, visits) if n > 0) / total
    out['tree'] = [{key: result[key] for key in ('visits', 'start_visits', 'nodes', 'root_eval',
                    'initial_visits', 'guided_iterations', 'guided_choices', 'descendant_guided_choices',
                    'prior_iterations', 'goal_hits', 'max_depth', 'route_counts', 'paths', 'search_elapsed_ms')}
                   for result in results]
    out['strategic_plan'] = plan
    out['plan_progress'] = ({plan['stage']: {
        'reach_fraction': sum(weight * r['goal_hits'] / max(1, r['guided_iterations'])
                              for (_, weight), r in zip(states, results)) / total,
        'meaning': 'fraction of guided simulations reaching the phase goal, not a win probability'}} if plan else {})
    return out
