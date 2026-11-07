"""Evaluate PokéChamp's native Bayesian configuration predictor, without search."""
import argparse,json,os,pickle,sys,time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE/'repo'),str(ROOT/'scripts')]
os.chdir(HERE/'repo')
from opp_baseline_topk import OUT,FIELDS,Scores
from bayesian.team_predictor import BayesianTeamPredictor

def norm(s):return ''.join(c for c in s.lower() if c.isalnum())

def initialize():
    global MODEL,NAMES
    MODEL=BayesianTeamPredictor(cache_file='/tmp/metamon_cache/gen9ou_team_predictor_full.pkl')
    MODEL.load_and_train()
    NAMES={norm(s):s for s in MODEL.species_counts}
    MODEL._parse_config_key=lru_cache(None)(MODEL._parse_config_key)
    MODEL._extract_moves_from_config_key=lru_cache(None)(MODEL._extract_moves_from_config_key)

@lru_cache(None)
def predict(species,moves):
    value=MODEL.predict_component_probabilities(NAMES.get(norm(species),species),revealed_moves=list(moves))
    result={name:{norm(k):v for k,v in value.get(key,[])} for name,key in [('moves','moves'),('item','items'),('ability','abilities')]}
    # The full-config API exposes a single Tera guess, but no ranked Tera marginal.
    config=MODEL.predict_pokemon_config(NAMES.get(norm(species),species),revealed_moves=list(moves))
    tera=config.get('tera_type');result['tera']={norm(tera):1.} if tera else {}
    return result

def evaluate_group(path):
    scores=Scores();decisions=0;n=0
    for game in pickle.loads(path.read_bytes()):
        for side,case in game['cases'].items():
            names=[line.split('|')[3].split(',')[0].removesuffix('-*') for line in game['public'] if line.startswith(f'|poke|{side}|')]
            slots={norm(name):i for i,name in enumerate(names)};actors={};moves=[set() for _ in names];start=0
            for t,end in enumerate(case['cutoffs']):
                for line in game['public'][start:end]:
                    p=line.split('|')
                    if len(p)<4 or not p[2].startswith(side):continue
                    if p[1] in ('switch','drag','replace'):
                        name=norm(p[3].split(',')[0])
                        if name in slots:actors[p[2]]=slots[name]
                    if p[1]=='move' and not any(v.startswith('[from]') for v in p[4:]):moves[actors[p[2]]].add(p[3])
                predictions=[predict(name,tuple(sorted(observed))) for name,observed in zip(names,moves)]
                scores.add(predictions,case['truth'],case['hidden'][t]);decisions+=1;start=end
            n+=1
    return dict(player_games=n,decision_points=decisions,metrics=scores.result())

def main(args):
    global OUT
    OUT=args.output
    paths=sorted((OUT/'cases').glob('*.pkl'));paths=paths[:args.groups] if args.groups else paths
    initialize();scores=Scores();games=decisions=0;started=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,r in enumerate(pool.map(evaluate_group,paths)):
            games+=r['player_games'];decisions+=r['decision_points']
            for name,m in r['metrics'].items():
                scores.count[name]+=m['count'];scores.empty[name]+=m['empty_predictions']
                scores.hits[name]=[a+b for a,b in zip(scores.hits[name],m['hits'])]
            if i%50==0:print(json.dumps(dict(groups=i+1,decision_points=decisions,seconds=time.time()-started)),flush=True)
    if not args.groups:
        ref=json.loads((OUT/'cases.json').read_text());assert decisions==ref['decisions'];assert scores.count==ref['counts']
    metrics=scores.result();metrics['tera']['top3']=metrics['tera']['top5']=None;metrics['tera']['hits'][1:]=[None,None]
    result=dict(method='PokéChamp BayesianTeamPredictor',category='traditional',source='https://github.com/sethkarten/pokechamp',
                commit='0f84c46',player_games=games,decision_points=decisions,metrics=metrics|{'action':None},seconds=time.time()-started,
                input='Public team preview and moves revealed before each decision; no hidden labels or future observations.',
                prior=dict(dataset='Native Metamon modern_replays gen9ou team cache',teams=MODEL.total_teams,species=len(MODEL.species_counts)),
                method_detail='Official component marginals with (1 + observed move matches)^2 weighting. The native function accepts teammates but does not use them in these component predictions.',
                unsupported='The native full-config interface supplies one Tera guess (Top-1 only); no Tera Top-3/5 marginal or opponent-action prediction.')
    (OUT/('pokechamp_bayes_probe.json' if args.groups else 'pokechamp_bayes.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--groups',type=int,default=0);p.add_argument('--workers',type=int,default=12)
    p.add_argument('--output',type=Path,default=OUT)
    main(p.parse_args())
