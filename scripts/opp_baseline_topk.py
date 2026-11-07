"""Shared locked cases and Top-k scoring for native opponent-model baselines."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/opponent_joint'
OUT = BASE / 'baselines'
FIELDS = {'moves': (17, 18, 19, 20), 'item': (1,), 'ability': (2,), 'tera': (4,)}


def aligned_private(game):
    private={}
    for side in ('p1','p2'):
        aligned=[];j=0
        for line in game['public']:
            while game['private'][side][j].startswith('|-hint|') and game['private'][side][j]!=line:j+=1
            observed=game['private'][side][j];j+=1
            assert line.split('|')[1:2]==observed.split('|')[1:2], (game['id'],side,line,observed)
            aligned.append(observed)
        private[side]=aligned
    return private


def correct_preview_aliases(game,side,case):
    """Match public form aliases to roster slots when removing disclosures."""
    def norm(s):return ''.join(c for c in s.lower() if c.isalnum())
    preview=[line.split('|')[3].split(',')[0] for line in game['public'] if line.startswith(f'|poke|{side}|')]
    aliases={norm(name):i for i,name in enumerate(preview) if norm(name)!=case['truth'][i]['species']}
    if not aliases:return
    import numpy as np
    actors={};seen=np.zeros((6,21),bool);start=0
    for t,end in enumerate(case['cutoffs']):
        for line in game['public'][start:end]:
            p=line.split('|')
            if len(p)<4 or not p[2].startswith(side):continue
            if p[1] in ('switch','drag','replace'):actors[p[2]]=aliases.get(norm(p[3].split(',')[0]))
            slot=actors.get(p[2])
            if slot is None:continue
            if p[1] in ('-item','-enditem','-ability','-terastallize'):
                seen[slot,{'-item':1,'-enditem':1,'-ability':2,'-terastallize':4}[p[1]]]=True
            if p[1]=='move' and not any(x.startswith('[from]') for x in p[4:]):
                for k,move in enumerate(case['truth'][slot]['moves']):
                    if move==norm(p[3]):seen[slot,17+k]=True
        case['hidden'][t]&=~seen;start=end


class Scores:
    def __init__(self):
        self.count = {k: 0 for k in FIELDS}
        self.hits = {k: [0, 0, 0] for k in FIELDS}
        self.empty = {k: 0 for k in FIELDS}

    def add(self, predictions, truth, hidden):
        for slot, mon in enumerate(truth):
            for name, positions in FIELDS.items():
                values = mon['moves'] if name == 'moves' else [mon[name]]
                targets = [v for p, v in zip(positions, values) if hidden[slot][p]]
                revealed = {v for p, v in zip(positions, values) if not hidden[slot][p]} if name == 'moves' else set()
                distribution = predictions[slot].get(name, {})
                order = sorted((v for v in distribution if v not in revealed and distribution[v] > 0), key=lambda v: (-distribution[v], v))[:5]
                self.count[name] += len(targets)
                if not order: self.empty[name] += len(targets)
                for i, k in enumerate((1, 3, 5)):
                    self.hits[name][i] += sum(v in order[:k] for v in targets)

    def result(self):
        return {name: dict(count=n, hits=self.hits[name], empty_predictions=self.empty[name],
                          **{f'top{k}': h/n if n else None for k,h in zip((1,3,5),self.hits[name])})
                for name,n in self.count.items()}


def prepare_group(job):
    import numpy as np
    import orjson
    import pyarrow.parquet as pq
    group, rows, tokens = job
    selected = {(r['battle_id'],r['side']): r for r in rows}
    packed = pq.read_table(BASE/'training/data'/rows[0]['file'], columns=['payload'])['payload'].to_pylist()
    field_ids = np.zeros(len(tokens), np.int8)
    for i,t in enumerate(tokens):
        for prefix, field in [('item:',1),('ability:',2),('type:',4)]:
            if t.startswith(prefix): field_ids[i] = field
    records = []
    with gzip.open(BASE/'data/test'/f'group-{group:05}.jsonl.gz','rb') as stream:
        for line in stream:
            game = orjson.loads(line)
            sides = [s for s in ('p1','p2') if (game['id'],s) in selected]
            if not sides: continue
            cases = {}
            for side in sides:
                row = selected[(game['id'],side)]
                data = pickle.loads(packed[row['row']])
                n = sum(d['side']==side for d in game['decisions'])
                hidden = np.asarray(data['config_unknown'][:n]).reshape(n,6,21).copy()
                hidden &= np.asarray(data['configuration']).reshape(1,6,21)>0
                preview = data['preview'][0]
                seen = np.zeros((6,21),bool)
                for t in range(n):
                    a,z = data['public_events_offsets'][t:t+2]
                    for event,numeric in zip(data['public_events'][a:z],data['public_numeric'][a:z]):
                        field = field_ids[event[1]]
                        if numeric[0] == 1 and field:
                            seen[preview == event[2],field] = True
                    hidden[t] &= ~seen
                truth = []
                for mon in game['teams'][side]:
                    def norm(s): return ''.join(c for c in s.lower() if c.isalnum())
                    truth.append(dict(species=norm(mon['species']), moves=[norm(m) for m in mon['moves']],
                                      item=norm(mon['item']) or 'none', ability=norm(mon['ability']), tera=norm(mon['teraType'])))
                cases[side] = dict(truth=truth,hidden=hidden,cutoffs=[d['cutoff'] for d in game['decisions'] if d['side']==side])
                correct_preview_aliases(game,side,cases[side])
            decisions = [{k:d[k] for k in ('side','cutoff','request')} | {'selected':d['actions'][d['target']]} for d in game['decisions']]
            private=aligned_private(game)
            records.append(dict(id=game['id'], public=game['public'], private=private, teams=game['teams'], decisions=decisions, cases=cases))
    with (OUT/'cases'/f'{group:05}.pkl').open('wb') as f: pickle.dump(records,f,protocol=5)
    counts = {name:sum(int(c['hidden'][:,:,positions].sum()) for g in records for c in g['cases'].values()) for name,positions in FIELDS.items()}
    public_species=sorted({line.split('|')[3].split(',')[0] for g in records for line in g['public'] if line.startswith('|poke|')})
    return dict(player_games=sum(len(g['cases']) for g in records),decisions=sum(len(c['cutoffs']) for g in records for c in g['cases'].values()),counts=counts,public_species=public_species)


def prepare(args):
    import pandas as pd
    sys.path.insert(0,str(ROOT/'src'))
    from arena.opponent.joint.tokens import Vocabulary
    OUT.mkdir(exist_ok=True); (OUT/'cases').mkdir(exist_ok=True)
    index = pd.read_parquet(BASE/'training/data/index.parquet')
    rows = index[(index.source=='controlled') & (index.split=='test')].sample(8192,random_state=2027).to_dict('records')
    grouped = defaultdict(list)
    for row in rows: grouped[int(row['battle_id'].split('-')[1])].append(row)
    tokens = Vocabulary(BASE/'catalog.json').tokens
    started=time.time(); total=dict(player_games=0,decisions=0,counts={k:0 for k in FIELDS});public_species=set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,r in enumerate(pool.map(prepare_group,[(g,rows,tokens) for g,rows in sorted(grouped.items())])):
            for k in ('player_games','decisions'): total[k]+=r[k]
            for k,n in r['counts'].items(): total['counts'][k]+=n
            public_species.update(r['public_species'])
            if i%40==0: print(json.dumps(dict(groups=i+1,seconds=round(time.time()-started,1),**total)),flush=True)
    reference=json.loads((BASE/'training/topk.json').read_text())
    print(json.dumps(dict(final_counts=total)),flush=True)
    assert total['decisions']==reference['decision_points']
    assert total['counts']=={k:reference['metrics'][k]['count'] for k in FIELDS}
    total.update(seconds=time.time()-started,scope=reference['configuration_scope'],move_metric=reference['move_metric'],public_species=sorted(public_species))
    (OUT/'cases.json').write_text(json.dumps(total,indent=2))
    print(json.dumps(total,indent=2))


def prepare_joint(args):
    """Use exactly the games, players and disclosure masks of the joint run."""
    import numpy as np
    from functools import lru_cache
    from types import SimpleNamespace
    sys.path.insert(0,str(ROOT/'src'))
    from arena.opponent.joint.data import games
    from arena.opponent.joint.engine import Mechanics
    from arena.opponent.joint.filter import JointOpponentModel
    from arena.opponent.joint.tokens import ident
    out=args.output;(out/'cases').mkdir(parents=True,exist_ok=True)
    engine=Mechanics()
    rules=lru_cache(None)(lambda species:engine.call('options',species=species))
    total=dict(player_games=0,decisions=0,counts={k:0 for k in FIELDS});species=set();cohort=[]
    try:
        for group in range(args.start_group,args.start_group+args.groups):
            source=list(games(BASE/'data/test'/f'group-{group:05}.jsonl.gz'))
            player=source[group%len(source)]['players'][1];records=[]
            for game in [g for g in source if player in g['players']][:args.games]:
                side='p'+str(game['players'].index(player)+1)
                preview=[l.split('|')[3].split(',')[0] for l in game['public'] if l.startswith('|poke|'+side+'|')]
                parser=SimpleNamespace(preview=preview,side=side,sampler=SimpleNamespace(rules=rules))
                cutoffs=[d['cutoff'] for d in game['decisions'] if d['side']==side]
                truth=[dict(species=ident(m['species']),moves=[ident(x) for x in m['moves']],
                    item=ident(m['item']) or 'none',ability=ident(m['ability']),tera=ident(m['teraType'])) for m in game['teams'][side]]
                hidden=np.zeros((len(cutoffs),6,21),bool)
                for t,end in enumerate(cutoffs):
                    known=JointOpponentModel._configuration_facts(parser,game['public'][:end])
                    for slot,mon in enumerate(truth):
                        for field,key in [('item','item'),('ability','ability'),('tera','teraType')]:hidden[t,slot,FIELDS[field][0]]=key not in known[slot]
                        for k,move in enumerate(mon['moves']):hidden[t,slot,17+k]=move not in known[slot].get('moves',[])
                decisions=[{k:d[k] for k in ('side','cutoff','request')}|{'selected':d['actions'][d['target']]} for d in game['decisions']]
                records.append(dict(id=game['id'],public=game['public'],private=aligned_private(game),teams=game['teams'],decisions=decisions,
                    cases={side:dict(truth=truth,hidden=hidden,cutoffs=cutoffs)}))
                total['player_games']+=1;total['decisions']+=len(cutoffs)
                for field,positions in FIELDS.items():total['counts'][field]+=int(hidden[:,:,positions].sum())
                species.update(preview);cohort.append(dict(group=group,game=game['id'],side=side,player=player,scenario=game['test_group']))
            (out/'cases'/f'{group:05}.pkl').write_bytes(pickle.dumps(records,protocol=5))
    finally:engine.close()
    total.update(public_species=sorted(species),cohort=cohort,
        scope='Identical complete player-games for all methods; masks use public facts before each decision.',
        move_metric='Micro recall@k over true unrevealed moves. Unsupported fields remain null.')
    (out/'cases.json').write_text(json.dumps(total,indent=2));print(json.dumps(total,indent=2))


def report():
    reference=json.loads((BASE/'training/topk.json').read_text())
    methods=[dict(method='Current neural model',category='supervised',player_games=reference['player_games'],
                  decision_points=reference['decision_points'],metrics=reference['metrics'],
                  configuration_input=reference['prediction_scope'],action_input=reference['action_scope'])]
    action=OUT/'current_public_action.json'
    if action.exists():
        public=json.loads(action.read_text())
        methods[0]['conditional_action_diagnostic']=methods[0]['metrics']['action']
        methods[0]['metrics']['action']=public['metrics']
        methods[0]['action_evaluation']=public
        methods[0]['action_input']=public['input']
    for name in ('foulplay','metamon_usage','pokechamp_bayes','metamon_maskgit','pokechamp'):
        path=OUT/f'{name}.json'
        if path.exists():methods.append(json.loads(path.read_text()))
    result=dict(protocol=json.loads((OUT/'cases.json').read_text()),methods=methods,
                notes=['Hidden moves use micro recall, not per-slot accuracy.',
                       'Unsupported tasks are null. Missing predictions on supported tasks count as misses.',
                       'Action comparisons use the fixed 1024-case non-preview subset and merge Tera flags. The current public-outcome head is compared with the PokéChamp opponent prompt; its old oracle-configuration policy diagnostic is stored separately.',
                       'Foul Play additionally uses the observer own private configuration for damage inference. Native priors are not retrained on the locked test games.'])
    status=OUT/'method_status.json'
    if status.exists():result['method_status']=json.loads(status.read_text())
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    for method in methods:
        print(method['method'])
        for name in (*FIELDS,'action'):
            m=method['metrics'].get(name)
            print(name, ' / '.join(f'{100*m[f"top{k}"]:.2f}' if m[f'top{k}'] is not None else '—' for k in (1,3,5)) if m else '—')


def report_joint(args):
    import numpy as np
    from opp_joint_public import aggregate
    out=args.output.parent;joint=args.joint_directory or out/'joint';protocol=json.loads((args.output/'cases.json').read_text())
    hidden_counts=[]
    for path in sorted((out/'baselines/cases').glob('*.pkl')):
        for game in pickle.loads(path.read_bytes()):
            for case in game['cases'].values():hidden_counts.extend(case['hidden'][:,:,17:21].sum(-1).ravel().tolist())
    protocol['move_recall_upper_bound']={f'top{k}':sum(min(k,n) for n in hidden_counts)/sum(hidden_counts) for k in (1,3,5)}
    summary=json.loads((joint/'metrics.json').read_text());rows=[]
    for group in sorted({r['group'] for r in protocol['cohort']}):
        path=joint/'parts'/f'{group:05}.json';part=json.loads(path.read_text())
        rows.extend(part['records'])
    assert len({r['game'] for r in rows})==protocol['player_games']
    assert sum('action' in r for r in rows)==protocol['decisions']
    summary['metrics']={name:aggregate([s for r in rows for s in r['hidden'].get(name,[])]) for name in FIELDS}
    summary['metrics']['action']=aggregate([r['action'] for r in rows if 'action' in r and r['phase']!='preview'])
    summary['by_phase']={phase:aggregate([r['action'] for r in rows if 'action' in r and r['phase']==phase]) for phase in ('preview','turn','forced')}
    summary['by_group']={}
    for name in ('known','unseen_team','unseen_set','unseen_family','abrupt','gradual','confounded'):
        groups={r['group'] for r in protocol['cohort'] if r['scenario']==name}
        selected=[r for r in rows if r['group'] in groups]
        metrics={field:aggregate([s for r in selected for s in r['hidden'].get(field,[])]) for field in FIELDS}
        metrics['action']=aggregate([r['action'] for r in selected if 'action' in r and r['phase']!='preview'])
        summary['by_group'][name]=dict(metrics=metrics,windows=len(selected),
            invalid_updates=sum(not r['valid_update'] for r in selected),
            update_seconds=sum(r['update_seconds'] or 0 for r in selected))
    summary['action_by_encounter']={str(encounter):aggregate([r['action'] for r in rows if r['encounter']==encounter and 'action' in r and r['phase']!='preview']) for encounter in (0,1)}
    assert {name:summary['metrics'][name]['count'] for name in FIELDS}==protocol['counts']
    summary['latency_scope']=f"Wall-clock time with {summary['workers']} concurrent filter workers on shared CPUs and two shared GPUs."
    summary['complete_player_games']=sum(all(r['valid_update'] for r in rows if r['game']==game) for game in {r['game'] for r in rows})
    summary['attempted_updates']=sum(r['update_attempted'] for r in rows)
    summary['failed_update_cases']=[dict(game=r['game'],window=r['window'],mismatch=r['mismatch'],cost=r['inference_cost']) for r in rows if r['update_attempted'] and not r['valid_update']]
    summary['update_failure_rate']=len(summary['failed_update_cases'])/summary['attempted_updates']
    summary['unprocessed_window_rate']=sum(not r['valid_update'] for r in rows)/len(rows)
    summary['valid_decision_rate']=sum(r['valid_prediction'] and 'action' in r for r in rows)/protocol['decisions']
    summary['valid_action_prediction_rate']=sum(r['valid_prediction'] and 'action' in r and r['phase']!='preview' for r in rows)/summary['metrics']['action']['count']
    summary['metrics_on_valid_predictions']={name:aggregate([s for r in rows if r['valid_prediction'] for s in r['hidden'].get(name,[])]) for name in FIELDS}
    summary['metrics_on_valid_predictions']['action']=aggregate([r['action'] for r in rows if r['valid_prediction'] and 'action' in r and r['phase']!='preview'])
    summary['configuration_diversity']=dict(median=float(np.median([r['configurations'] for r in rows if r['valid_update']])),
        fraction_at_most_two=sum(r['valid_update'] and r['configurations']<=2 for r in rows)/max(1,sum(r['valid_update'] for r in rows)))
    summary['initial_hidden_metrics']={name:aggregate([s for r in rows if r.get('phase')=='preview' for s in r['hidden'].get(name,[])]) for name in FIELDS}
    exclusions={name:dict(supported=set(),removed=set()) for name in FIELDS}
    for row in rows:
        if not row['valid_prediction']:continue
        for name,scores in row['hidden'].items():
            for score in scores:
                key=(row['game'],score['slot'],score['target']);sets=exclusions[name]
                if score['covered']:sets['supported'].add(key)
                elif key in sets['supported']:sets['removed'].add(key)
    summary['true_attribute_support_loss']={name:dict(ever_supported=len(sets['supported']),later_removed=len(sets['removed']),
        rate=len(sets['removed'])/len(sets['supported']) if sets['supported'] else None) for name,sets in exclusions.items()}
    (joint/'metrics.json').write_text(json.dumps(summary,indent=2))
    methods=[dict(method='Joint C/X/M/Z filter',category='supervised + SMC',metrics=summary['metrics'],
        input='Causal public history plus observer own legitimate private information; opponent labels are scoring-only.')]
    extra_methods=[name for name in ('pokechamp_qwen38_27b','qwen38_27b') if (out/'baselines'/f'{name}.json').exists()]
    for name in ['current_neural','foulplay','metamon_usage','pokechamp_bayes','pokechamp',*extra_methods]:
        result=json.loads((out/'baselines'/f'{name}.json').read_text())
        for field in FIELDS:
            if result['metrics'][field] is not None:assert result['metrics'][field]['count']==protocol['counts'][field]
        if result['metrics']['action'] is not None:assert result['metrics']['action']['count']==summary['metrics']['action']['count']
        methods.append(result)
    result=dict(protocol=protocol,methods=methods,joint_runtime=summary,
        notes=[f"{protocol['player_games']} player-games from {len({r['player'] for r in protocol['cohort']})} players; this is a matched complete-game evaluation with limited population precision.",
               'Moves use micro recall@k. Actions merge Tera, exclude preview and retain unexecuted submissions.',
               'The public neural action head is trained on visible outcomes; the joint filter uses its conditional submitted-action head. This comparison is not an isolated ablation of filtering.',
               'Encounter breakdowns describe different games; they do not by themselves establish a causal benefit from cross-game memory.',
               'Unsupported targets are null; supported but missing/invalid predictions count as misses.',
               'Missing predictions receive zero Top-k hits, NLL -log(1e-12), and action Brier penalty 2; valid-only metrics are retained separately.',
               'After an unapplied resolution, the remaining decisions in that game count as missing predictions. Latency statistics include attempted updates only.'])
    valid_actions={(r['game'],r['decision']) for r in rows if r['valid_prediction'] and 'action' in r and r['phase']!='preview'}
    joint_actions={(r['game'],r['decision']):r for r in rows if 'action' in r and r['phase']!='preview'}
    subset={'joint':summary['metrics_on_valid_predictions']['action']}
    paired={}
    action_sources=[('neural','current_public_action_predictions.jsonl'),('pokechamp','pokechamp_action_predictions.jsonl')]
    action_sources.extend((name,name+'_predictions.jsonl') for name in extra_methods)
    for name,filename in action_sources:
        predictions=[r for line in (args.output/filename).read_text().splitlines() if (r:=json.loads(line))['phase']!='preview']
        selected=[r for r in predictions if (r['game'],r['decision']) in valid_actions]
        assert len(selected)==len(valid_actions)
        subset[name]=dict(count=len(selected),**{f'top{k}':sum(r['hits'][i] for r in selected)/len(selected) for i,k in enumerate((1,3,5))})
        blocks=defaultdict(lambda:np.zeros(4))
        for prediction in predictions:
            row=joint_actions[prediction['game'],prediction['decision']]
            blocks[row['group']]+=np.r_[1,np.array(row['action']['hits'])-prediction['hits']]
        values=np.stack(list(blocks.values()));rng=np.random.default_rng(2027)
        sampled=values[rng.integers(len(values),size=(5000,len(values)))].sum(1)
        bounds=np.quantile(sampled[:,1:]/sampled[:,:1],[.025,.975],axis=0)
        paired[name]={f'top{k}':dict(difference=float(values[:,i+1].sum()/values[:,0].sum()),interval_95=bounds[:,i].tolist()) for i,k in enumerate((1,3,5))}
    result['action_on_same_valid_subset']=subset
    result['paired_action_differences']=dict(unit='player; both selected encounters resampled together',groups=len(blocks),joint_minus_baseline=paired)
    diagnostic=joint.parent/'failure_diagnosis.json'
    if diagnostic.exists():result['failure_diagnosis']=json.loads(diagnostic.read_text())
    (joint.parent/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    for method in methods:
        print(method['method'])
        for name in (*FIELDS,'action'):
            metric=method['metrics'].get(name)
            print(name,' / '.join(f'{100*metric[f"top{k}"]:.2f}' if metric[f'top{k}'] is not None else '—' for k in (1,3,5)) if metric else '—')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','prepare-joint','report','report-joint']);p.add_argument('--workers',type=int,default=12)
    p.add_argument('--output',type=Path,default=BASE/'training/comparison/baselines')
    p.add_argument('--joint-directory',type=Path)
    p.add_argument('--start-group',type=int,default=70);p.add_argument('--groups',type=int,default=14);p.add_argument('--games',type=int,default=2)
    args=p.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='prepare-joint':prepare_joint(args)
    elif args.command=='report-joint':report_joint(args)
    else:report()
