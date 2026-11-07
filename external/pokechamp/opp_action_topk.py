"""Native PokéChamp opponent prompts and LLM predictions on causal test prefixes."""
import argparse,asyncio,json,logging,os,pickle,random,sys,time
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from pathlib import Path
from functools import lru_cache

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE/'repo'),str(ROOT/'scripts')];os.chdir(HERE/'repo')
from opp_baseline_topk import OUT
from poke_env.environment.battle import Battle
from poke_env.player.player import Player
from poke_env.player.local_simulation import LocalSim
from poke_env.data import GenData
from pokechamp import data_cache
from pokechamp.prompts import state_translate
from bayesian.team_predictor import BayesianTeamPredictor

# Parsing a stored configuration is pure and repeated across native stat queries.
BayesianTeamPredictor._parse_config_key=lru_cache(None)(BayesianTeamPredictor._parse_config_key)
BayesianTeamPredictor._extract_moves_from_config_key=lru_cache(None)(BayesianTeamPredictor._extract_moves_from_config_key)


def action_key(action):
    def norm(s):return ''.join(c for c in s.lower() if c.isalnum())
    for kind in ('move','switch'):
        if isinstance(action.get(kind),str):
            name=norm(action[kind])
            if kind=='switch':name=norm(GenData.from_gen(9).pokedex.get(name,{}).get('baseSpecies',name))
            return kind+':'+name
    return None


class OfflineFeed:
    MESSAGES_TO_IGNORE=Player.MESSAGES_TO_IGNORE
    def __init__(self,battle):
        self.battle=battle;self.username=battle.player_username;self.switch_set=set();self.logger=logging.getLogger('replay')
    async def _get_battle(self,tag):return self.battle
    async def _handle_battle_request(self,*args,**kwargs):pass
    async def update(self,lines):
        messages=[line.split('|') for line in lines if line and line!='|']
        await Player._handle_battle_message(self,[[f'>{self.battle.battle_tag}'],['',''],['',''],*messages])


def make_sim(battle):
    return LocalSim(battle,data_cache.get_cached_move_effect(),data_cache.get_cached_pokemon_move_dict(),
                    data_cache.get_cached_ability_effect(),data_cache.get_cached_pokemon_ability_dict(),
                    data_cache.get_cached_item_effect(),data_cache.get_cached_pokemon_item_dict(),
                    GenData.from_gen(9),True,format='gen9ou',prompt_translate=state_translate)


async def game_prompts(game,side,queries):
    observer='p2' if side=='p1' else 'p1'
    username=next(line.split('|')[3] for line in game['public'] if line.startswith(f'|player|{observer}|'))
    battle=Battle(game['id'],username,logging.getLogger('replay'),9);battle.player_role=observer
    first=next(d['request'] for d in game['decisions'] if d['side']==observer)
    battle.parse_request(first|{'rqid':1});feed=OfflineFeed(battle);start=0;t=0;result=[]
    by_cutoff=defaultdict(list)
    for d in game['decisions']:by_cutoff[d['cutoff']].append(d)
    for end,decisions in sorted(by_cutoff.items()):
        await feed.update(game['private'][observer][start:end]);start=end;battle._format='gen9ou'
        ours=next((d for d in decisions if d['side']==observer),None)
        if ours:battle.parse_request(ours['request']|{'rqid':1})
        target=next((d for d in decisions if d['side']==side),None)
        if target is None:continue
        if t in queries:
            sim=make_sim(battle)
            _,state,_=sim.state_translate(sim.battle)
            system,state,_,constraint,choices=sim.get_opponent_prompt(state)
            moves=sim.get_opponent_current_moves()
            switches=[m.species for m in sim.battle.opponent_team.values() if not m.fainted and not m.active]
            candidates=([{'move':m} for m in moves] if not sim.battle.opponent_active_pokemon.fainted else [])+[{'switch':m} for m in switches]
            # Labels are attached only after native prompt/candidate construction.
            action=target['selected'];label={action['kind']:action.get('move',action.get('species'))}
            result.append(dict(game=game['id'],side=side,decision=t,phase='forced_switch' if target['request'].get('forceSwitch') else 'turn',
                               system=system,prompt=state+choices+constraint,candidates=candidates,label=label))
        t+=1
        if t>max(queries):break
    return result


def prepare_group(job):
    path,selected=job;result=[]
    import numpy as np
    random.seed(2027+int(path.stem));np.random.seed(2027+int(path.stem))
    for game in pickle.loads(path.read_bytes()):
        for side in game['cases']:
            indices=selected.get((game['id'],side))
            if indices:result.extend(asyncio.run(game_prompts(game,side,indices)))
    return result


def prepare(args):
    population=[];paths=sorted((OUT/'cases').glob('*.pkl'))
    for path in paths:
        for game in pickle.loads(path.read_bytes()):
            for side,case in game['cases'].items():
                population.extend((path.stem,game['id'],side,t) for t in range(1,len(case['cutoffs'])))
    chosen=random.Random(2027).sample(population,args.points) if args.points else population;groups=defaultdict(lambda:defaultdict(set))
    for group,game,side,t in chosen:groups[group][game,side].add(t)
    result=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(prepare_group,[(OUT/'cases'/f'{g}.pkl',dict(q)) for g,q in sorted(groups.items())]):result.extend(part)
    assert len(result)==len(chosen)
    path=OUT/'pokechamp_action_cases.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in result))
    print(json.dumps(dict(points=len(result),population=len(population),phases=dict(Counter(r['phase'] for r in result)))))


def evaluate(args):
    from openai import OpenAI
    from pokechamp import gpt_player
    # Reuse the HTTP connection pool; the official prompt and generation call
    # remain unchanged. This endpoint serves only the local frozen backbone.
    client=OpenAI(base_url='http://127.0.0.1:18791/v1',api_key='local')
    gpt_player.OpenAI=lambda **kwargs:client
    rows=[json.loads(line) for line in (OUT/'pokechamp_action_cases.jsonl').read_text().splitlines()]
    if args.points:rows=rows[:args.points]
    def predict(row):
        llm=gpt_player.GPTPlayer(api_key='local');counts=Counter();invalid=0
        for _ in range(args.samples):
            response,_,_=llm.get_LLM_action(row['system'],row['prompt'],model='pokechamp-qwen3-8b',temperature=0.3,json_format=True,max_tokens=64)
            try:action=action_key(json.loads(response))
            except (ValueError,AttributeError):action=None
            if action:counts[action]+=1
            else:invalid+=1
        order=sorted(counts,key=lambda a:(-counts[a],a));target=action_key(row['label'])
        return {k:row[k] for k in ('game','side','decision','phase')}|dict(target=target,counts=counts,invalid=invalid,hits=[int(target in order[:k]) for k in (1,3,5)],candidate_covered=target in {action_key(a) for a in row['candidates']})
    started=time.time();results=[]
    with (OUT/'pokechamp_action_predictions.jsonl').open('w') as out,ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(predict,rows):
            results.append(r);out.write(json.dumps(r)+'\n');out.flush()
            if len(results)%32==0:print(json.dumps(dict(points=len(results),seconds=round(time.time()-started,1))),flush=True)
    hits=[sum(r['hits'][i] for r in results) for i in range(3)]
    metric=dict(count=len(results),hits=hits,**{f'top{k}':h/len(results) for k,h in zip((1,3,5),hits)})
    result=dict(method='PokéChamp opponent prompt + Qwen3-8B',category='LLM',source='https://github.com/sethkarten/pokechamp',commit='0f84c46',
                backbone='/data/models/Qwen3-8B',model_variant='Official GPTPlayer interface with local Qwen3-8B, thinking disabled; not the paper Llama/GPT checkpoint.',
                player_games=len({(r['game'],r['side']) for r in results}),decision_points=len(results),
                metrics=dict(moves=None,item=None,ability=None,tera=None,action=metric),seconds=time.time()-started,
                samples=args.samples,temperature=.3,invalid_generations=sum(r['invalid'] for r in results),
                input='Official causal replay parser and opponent prompts; observer own private requests only. No opponent true configuration, legal-action list, submitted action or future observations enter inference.',
                action_scope=('All non-preview points of the shared locked cohort.' if not args.points else 'Random fixed non-preview subset of the shared locked cohort.')+' Predict submitted move identity or switch species; Tera flags are merged because the native opponent prompt does not emit them.',
                ranking='Empirical action frequencies from repeated unmodified GPTPlayer generations; ties sorted by action name. These are Monte Carlo Top-k results, not the paper original evaluation scores.',
                candidate_coverage=sum(r['candidate_covered'] for r in results)/len(results),phases=dict(Counter(r['phase'] for r in results)))
    (OUT/('pokechamp.json' if not args.points or len(results)==1024 else 'pokechamp_probe.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','evaluate']);p.add_argument('--points',type=int,default=1024);p.add_argument('--workers',type=int,default=12);p.add_argument('--samples',type=int,default=32)
    p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args();OUT=args.output;prepare(args) if args.command=='prepare' else evaluate(args)
