"""Native Metamon NaiveUsagePredictor on the shared hidden-configuration cases."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import date, datetime
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE/'repo'),str(ROOT/'scripts')]
os.environ.setdefault('METAMON_CACHE_DIR','/data/wangxinqi/metamon_cache')
import numpy as np
from opp_baseline_topk import BASE,OUT,FIELDS,Scores
from metamon.backend.replay_parser.forward import SimProtocol
from metamon.backend.replay_parser.replay_state import ParsedReplay
from metamon.backend.team_prediction import team as team_module
from metamon.backend.team_prediction.team import PokemonSet,TeamSet
from metamon.backend.team_prediction.predictor import NaiveUsagePredictor
from metamon.backend.team_prediction.usage_stats import legacy_team_builder
from metamon.backend.team_prediction.usage_stats.stat_reader import SmogonStat,PreloadedSmogonUsageStats


def norm(s): return ''.join(c for c in s.lower() if c.isalnum())


def initialize(samples,save=False):
    global MODEL,SAMPLES,SAVE,PRIOR,COMPLETIONS
    SAMPLES,SAVE=samples,save
    raw=SmogonStat('gen9ou','/data/wangxinqi/opponent_kb/smogon_stats',date='2026-10',rank=0,verbose=False)
    # Supply existing Smogon data to the official lookup/sampling code. No
    # test-game information is used to construct these frequency tables.
    stats=object.__new__(PreloadedSmogonUsageStats)
    stats.gen=9;stats.format='gen9ou';stats.rank=0
    stats._movesets=raw.movesets;stats._inclusive={};stats._lower_rank_fallbacks=[]
    legacy_team_builder.get_usage_stats=lambda *a,**kw:stats
    team_module.get_usage_stats=lambda *a,**kw:stats
    MODEL=NaiveUsagePredictor(usage_stats_rank=0)
    PRIOR=stats;COMPLETIONS={}
    random.seed(2027);np.random.seed(2027)
    names=sorted({blank(s).name for s in json.loads((BASE/'catalog.json').read_text())['bySpecies']})
    public=json.loads((OUT/'cases.json').read_text())['public_species']
    extra=sorted({blank(s.removesuffix('-*')).name for s in public}-set(names))
    # Keep the original batches/seeds unchanged when adding public form aliases.
    batches=[names[i:i+6] for i in range(0,len(names),6)]+[extra[i:i+6] for i in range(0,len(extra),6)]
    for selected in batches:
        selected+=names[:6-len(selected)]
        mons=[blank(s) for s in selected];team=TeamSet(mons[0],mons[1:],'gen9ou')
        new=set(selected)-COMPLETIONS.keys()
        for name in new:COMPLETIONS[name]=[]
        for _ in range(SAMPLES):
            for mon in proposals(team):
                if mon.name in new:COMPLETIONS[mon.name].append(mon)


def blank(name):
    return PokemonSet(name=name,gen=9,moves=PokemonSet.default_moves(name,9),
                      ability=PokemonSet.MISSING_ABILITY,item=PokemonSet.MISSING_ITEM,
                      nature=PokemonSet.MISSING_NATURE,tera_type=PokemonSet.MISSING_TERA_TYPE,
                      evs=PokemonSet.default_evs(9),ivs=PokemonSet.default_ivs(9))


def proposals(team):
    # Preserve proposal move order: native merging uses set iteration when
    # filling a limited number of unknown slots, so a second merge can differ.
    captured=[];fill=PokemonSet.fill_from_PokemonSet
    def capture(p,other):
        captured.append(deepcopy(other));return fill(p,other)
    PokemonSet.fill_from_PokemonSet=capture
    MODEL.predict(team,date=date(2026,10,1))
    PokemonSet.fill_from_PokemonSet=fill
    return captured


def snapshots(game):
    """Forward parsing only; never run the replay parser's backward backfill."""
    replay=ParsedReplay(gameid=game['id'],time_played=datetime(2026,11,3),format='gen9ou')
    protocol=SimProtocol(replay)
    queries={}
    for side,case in game['cases'].items():
        for t,end in enumerate(case['cutoffs']):queries.setdefault(end,[]).append((side,t))
    cutoff=0
    for end,requests in sorted(queries.items()):
        for line in game['public'][cutoff:end]:
            if line:protocol.interpret_message([x.strip() for x in line.split('|')[1:]])
        cutoff=end
        for side,t in requests:
            mons=protocol.curr_turn.pokemon_1 if side=='p1' else protocol.curr_turn.pokemon_2
            partial=[PokemonSet.from_ReplayPokemon(mon,9) for mon in mons]
            yield side,t,TeamSet(lead=partial[0],reserve=partial[1:],format='gen9ou')


def predictions(team,cache):
    result=[]
    for mon in team.pokemon:
        key=pickle.dumps(mon.to_dict())
        if key not in cache:
            values={'moves':Counter()}
            # The native predictor samples each Pokemon independently when all
            # six names are known, then fills only missing fields. Reuse its
            # unconditional draws and apply that exact native merge operation.
            for base in COMPLETIONS[mon.name]:
                filled=deepcopy(mon);filled.fill_from_PokemonSet(base)
                for move in filled.moves:
                    if move not in (PokemonSet.MISSING_MOVE,PokemonSet.NO_MOVE):values['moves'][norm(move)]+=1
            for name,column in [('item','items'),('ability','abilities'),('tera','tera_types')]:
                # Exact categorical marginals of weighted_random_choice; no
                # sampling error is needed for these independent single fields.
                values[name]={norm(k):v for k,v in PRIOR[mon.name][column].items() if k not in ('Other','Nothing') and v>0}
                if not values[name]:values[name]={'none':1.}
            cache[key]=values
        result.append(cache[key])
    return result


def evaluate_group(path):
    destination=OUT/'metamon_usage_groups'/f'{path.stem}.json'
    if SAVE and destination.exists():return json.loads(destination.read_text())
    random.seed(2027+int(path.stem));np.random.seed(2027+int(path.stem))
    scores=Scores();n=0;decisions=0;states=0;started=time.time()
    for game in pickle.loads(path.read_bytes()):
        cache={}
        try:
            for side,t,team in snapshots(game):
                case=game['cases'][side]
                scores.add(predictions(team,cache),case['truth'],case['hidden'][t]);decisions+=1
        except Exception as exc:
            raise RuntimeError(f'{game["id"]}: {exc}') from exc
        n+=len(game['cases']);states+=len(cache)
    result=dict(player_games=n,decision_points=decisions,sampled_states=states,metrics=scores.result(),seconds=time.time()-started)
    if SAVE:destination.write_text(json.dumps(result))
    return result


def main(args):
    global OUT
    OUT=args.output
    paths=sorted((OUT/'cases').glob('*.pkl'))
    if args.groups:paths=paths[:args.groups]
    (OUT/'metamon_usage_groups').mkdir(exist_ok=True)
    totals=dict(player_games=0,decision_points=0,sampled_states=0);scores=Scores();failures=[];started=time.time()
    initialize(args.samples,not args.groups)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(evaluate_group,p):p for p in paths}
        for i,future in enumerate(as_completed(futures)):
            try:r=future.result()
            except Exception as exc:
                failures.append(dict(group=futures[future].stem,error=str(exc)));print(json.dumps(failures[-1]),flush=True);continue
            for k in totals:totals[k]+=r[k]
            for name,m in r['metrics'].items():
                scores.count[name]+=m['count'];scores.empty[name]+=m['empty_predictions']
                scores.hits[name]=[a+b for a,b in zip(scores.hits[name],m['hits'])]
            print(json.dumps(dict(groups=i+1,seconds=round(time.time()-started,1),**totals)),flush=True)
    if failures:
        (OUT/'metamon_usage_errors.json').write_text(json.dumps(failures,indent=2));raise RuntimeError(f'{len(failures)} groups failed')
    result=dict(method='Metamon NaiveUsagePredictor',category='traditional',source='https://github.com/UT-Austin-RPL/metamon',
                samples=args.samples,**totals,metrics=scores.result()|{'action':None},seconds=time.time()-started,
                input='Causal public prefix parsed by official SimProtocol; initial revealed fields via PokemonSet.from_ReplayPokemon. No backward filling or hidden labels.',
                prior='Smogon gen9ou October 2026, rating cutoff 0; official SmogonStat parser and native predictor. Cross-tier/historical fallbacks absent from this frozen data source.',
                sampling='Exact native categorical marginals; hidden-move rankings from 64 native unmerged proposals per species followed by native fill_from_PokemonSet. All six names are known, so attribute sampling is independent of teammates.',
                action='Unsupported: team completion does not predict opponent actions.')
    if not args.groups:
        ref=json.loads((OUT/'cases.json').read_text());assert totals['decision_points']==ref['decisions'];assert scores.count==ref['counts']
    (OUT/('metamon_usage_probe.json' if args.groups else 'metamon_usage.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=24);p.add_argument('--samples',type=int,default=64);p.add_argument('--groups',type=int,default=0)
    p.add_argument('--output',type=Path,default=OUT)
    main(p.parse_args())
