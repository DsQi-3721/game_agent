"""Parallel Showdown games with one frozen policy per rollout round."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from concurrent.futures import Future
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import queue
import random
import threading
import time

from arena.agents.base import Agent, AgentContext
from arena.agents.system1.decider import FastDecider, RouteConfig
from arena.s1.policy_data import begin_policy_decision, record_policy_action, finish_policy_history
from arena.state.views import legal_options


class PolicyBatcher:
    """One inference thread batches requests from poke-env's event loop."""
    def __init__(self, policy, batch_size=8, wait_ms=5):
        self.policy, self.batch_size, self.wait = policy, batch_size, wait_ms/1000
        self.queue = queue.Queue()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    async def score(self, example):
        future = Future(); self.queue.put((example,future))
        return await asyncio.wrap_future(future)

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None: return
            batch = [item]; deadline = time.monotonic()+self.wait
            while len(batch) < self.batch_size:
                try: batch.append(self.queue.get(timeout=max(0,deadline-time.monotonic())))
                except queue.Empty: break
            try:
                results = self.policy.predict([e for e,_ in batch])
                for (_,f),result in zip(batch,results):
                    if not f.cancelled(): f.set_result(result)
            except Exception as error:
                for _,f in batch:
                    if not f.cancelled(): f.set_exception(error)

    def close(self):
        self.queue.put(None); self.worker.join()


class PolicyAgent(Agent):
    name = 'ppo_policy'
    team_preview = 'lead_prior'
    opponent_model = 'prior'

    def __init__(self, batcher, seed, history=8, name='ppo_policy'):
        super().__init__(AgentContext(rng=random.Random(seed)))
        self.name, self.batcher, self.history = name, batcher, history
        self.decider = FastDecider(None, RouteConfig(mode='flat', debias=False, ask_hard=False, history_events=history))
        self.rows, self.errors, self.episodes = defaultdict(list), defaultdict(list), []

    async def act(self, battle):
        started = time.perf_counter()
        begin_policy_decision(battle, self.history)
        options = legal_options(battle)
        state, qs, facts = self.decider.build_request(battle, options)
        ex = {'state': state, 'question': qs['best'], 'keys': [f.key for f in facts]}
        if not facts: raise ValueError('No legal battle action')
        scored = await self.batcher.score(ex)
        action = self.ctx.rng.choices(range(len(facts)), weights=scored['probs'])[0]
        order = facts[action].option.order
        request = deepcopy(battle._last_request)
        self.rows[battle.battle_tag].append({**ex, 'ids': scored['ids'], 'action': action,
            'logp': math.log(scored['probs'][action]), 'value': scored['value'],
            'rqid': request['rqid'], 'request': request, 'order': order.message,
            'latency_s': time.perf_counter()-started})
        record_policy_action(battle, ex['keys'][action], self.history)
        return order

    def on_battle_message(self, battle_tag, text, player):
        if '|error|' in text or '|bigerror|' in text:
            self.errors[battle_tag].append(text)

    def on_battle_end(self, battle):
        reward = 1.0 if battle.won else -1.0 if battle.lost else 0.0
        self.episodes.append({'battle_tag': battle.battle_tag, 'completed': battle.finished,
            'reward': reward, 'invalid': self.errors.pop(battle.battle_tag, []),
            'steps': self.rows.pop(battle.battle_tag, [])})
        finish_policy_history(battle, self.history)


def verify_submissions(episodes, streams, username):
    from poke_env.data import to_id_str
    from arena.observation.capture import choice_attempts, load_stream
    from arena.observation.trace import choice_action_id
    for episode in episodes:
        path = Path(streams) / f'{episode["battle_tag"]}.{to_id_str(username)}.jsonl'
        accepted = {r['rqid']:r['data'].split('|',1)[0] for r in choice_attempts(load_stream(path)) if r['error'] is None}
        for step in episode['steps']:
            actual = accepted.get(step['rqid'])
            if actual is None or choice_action_id(actual,step['request']) != choice_action_id(step['order'],step['request']):
                episode['invalid'].append(f'action substitution at rqid={step["rqid"]}')
        if not episode['steps']: episode['invalid'].append('missing decision trajectory')


async def play_match(actor, frozen, cfg, opponent, team_a, team_b, games, seed, directory):
    from poke_env.ps_client.account_configuration import AccountConfiguration
    from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
    from arena.agents.adapter import AgentPlayer
    from arena.agents.baselines import SimpleHeuristicsAgent
    from arena.eval.runner import MatchRunner, unique_account
    from arena.external.launcher import ExternalLauncher
    from arena.config.schema import ExternalConfig
    import yaml

    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    learner = PolicyAgent(actor,seed,cfg['history_events'])
    ours = AgentPlayer(learner, battle_format='gen9ou', team=ConstantTeambuilder(Path(team_a).read_text()),
                       account_configuration=unique_account('ppo'), max_concurrent_battles=1,
                       stream_dir=str(directory/'streams'))
    theirs = None; launcher = None
    try:
        if opponent in ('kakuna','foulplay'):
            template = yaml.safe_load(Path(f'configs/experiments/strong_{opponent}_ou.yaml').read_text())
            spec = next(a['external'] for a in template['agents'] if a['name']==opponent)
            spec['env'] = {**spec.get('env',{}), 'CUDA_VISIBLE_DEVICES': cfg['ppo']['teacher_device'], 'OMP_NUM_THREADS':'1'}
            if opponent == 'kakuna':
                spec['env']['METAMON_KAKUNA_EPOCH'] = str(cfg['ppo']['teacher_checkpoint'])
            ext = ExternalConfig(**spec)
            username = unique_account(opponent).username
            launcher = ExternalLauncher(ext,opponent)
            await launcher.start('gen9ou',ours.username,directory,username=username,team_file=str(Path(team_b).resolve()))
            await launcher.wait_ready()
            await MatchRunner._challenge_with_watchdog(ours,username,games,launcher,
                stall_timeout=ext.stall_timeout_s,battle_timeout=ext.battle_timeout_s)
        else:
            agent = (PolicyAgent(frozen,seed+1,cfg['history_events'],name='frozen_policy')
                     if opponent=='frozen' else SimpleHeuristicsAgent(AgentContext(rng=random.Random(seed+1))))
            theirs = AgentPlayer(agent,battle_format='gen9ou',team=ConstantTeambuilder(Path(team_b).read_text()),
                                  account_configuration=unique_account(opponent),max_concurrent_battles=1)
            await asyncio.wait_for(ours.battle_against(theirs,n_battles=games), timeout=120+games*600)
        if ours.n_finished_battles != games or len(learner.episodes) != games:
            raise RuntimeError(f'Incomplete match: {ours.n_finished_battles}/{games}; no artificial terminal rewards')
        verify_submissions(learner.episodes,directory/'streams',ours.username)
        for e in learner.episodes: e.update(opponent=opponent,team_a=str(team_a),team_b=str(team_b))
        return learner.episodes
    finally:
        if launcher: await launcher.stop()
        await ours.ps_client.stop_listening()
        if theirs: await theirs.ps_client.stop_listening()


async def collect_round(actor, frozen, cfg, round_id, directory):
    teams = sorted((Path(cfg['root'])/'teams/train').glob('*.txt'))
    rng = random.Random(cfg['seed']+round_id)
    target = cfg['ppo']['decisions_per_round']
    episodes, rejected, wave = [], [], 0
    cycle = cfg['ppo']['opponent_cycle']
    while sum(len(e['steps']) for e in episodes) < target:
        jobs = []
        for i in range(cfg['ppo']['parallel_matches']):
            index = wave*cfg['ppo']['parallel_matches']+i
            own,other = rng.sample(teams,2)
            jobs.append(play_match(actor,frozen,cfg,cycle[index % len(cycle)],own,other,8,
                cfg['seed']+round_id*10000+index,Path(directory)/f'match_{index:04d}'))
        results = await asyncio.gather(*jobs)
        for matches in results:
            for e in matches:
                (rejected if e['invalid'] else episodes).append(e)
        if not any(not e['invalid'] for matches in results for e in matches):
            raise RuntimeError('No accepted-action episodes in rollout wave')
        wave += 1
    return episodes, rejected


async def evaluate_battles(actor, frozen, cfg, directory, final=False):
    from arena.eval.metrics import wilson_interval
    setting = cfg['evaluation']
    teams = sorted(Path(setting['final_pool'] if final else Path(cfg['root'])/'teams/dev').glob('*.txt'))
    opponents = ['kakuna','foulplay'] if final else setting['opponents']
    games = setting['final_games_per_team'] if final else setting['games_per_team']
    limit = asyncio.Semaphore(cfg['ppo']['parallel_matches'])
    async def play(opponent,i,team):
        async with limit:
            return await play_match(actor,frozen,cfg,opponent,team,team,games,cfg['seed']+500000+i,
                                    Path(directory)/opponent/f'team{i:02d}')
    matches = await asyncio.gather(*(play(o,i,t) for o in opponents for i,t in enumerate(teams)))
    episodes = [e for part in matches for e in part]
    if any(e['invalid'] for e in episodes):
        raise RuntimeError('Development games contain rejected/substituted actions; inspect captured streams')
    result = {}
    for opponent in opponents:
        games = [e for e in episodes if e['opponent']==opponent]
        wins = sum(e['reward']==1 for e in games); ties = sum(e['reward']==0 for e in games)
        latency = sorted(s['latency_s'] for e in games for s in e['steps'])
        result[opponent] = {'games':len(games),'wins':wins,'ties':ties,'win_rate':wins/len(games),
            'score':(wins+.5*ties)/len(games),'win_ci95':wilson_interval(wins,len(games)),
            'latency_mean':sum(latency)/len(latency),'latency_p90':latency[int(.9*(len(latency)-1))]}
    Path(directory).mkdir(parents=True,exist_ok=True)
    (Path(directory)/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
