"""Evaluate the actual joint filter on causal histories, before each decision."""
import argparse,json,multiprocessing,sys,time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from arena.opponent.joint.data import games
from arena.opponent.joint.filter import JointOpponentModel
from arena.opponent.joint.tokens import ident
from opp_baseline_topk import aligned_private


def identity(action):
    if action.startswith('team:'):return 'lead:'+action.split(':')[1][0]
    if action.startswith('move:'):return ':'.join(action.split(':')[:2])
    return action


def score(values,truth,valid=True):
    order=sorted((key for key in values if values[key]>0),key=lambda key:(-values[key],key)) if valid else []
    p=values.get(truth,0.) if valid else 0.
    return dict(hits=[int(truth in order[:k]) for k in (1,3,5)],probability=p,
                nll=float(-np.log(max(p,1e-12))),covered=bool(p>0))


def evaluate(job):
    args,group=job;base=args.directory;device=f'cuda:{group%2}';torch.set_num_threads(1)
    source=list(games(base/'data'/args.split/f'group-{group:05}.jsonl.gz'))
    player=source[group%len(source)]['players'][1]
    selected=[g for g in source if player in g['players']][:args.games]
    records=[];model=None;started=time.time()
    stream=(args.output/'parts'/f'{group:05}.jsonl').open('w')
    try:
        for encounter,game in enumerate(selected):
            private=aligned_private(game);blocked=False
            side='p'+str(game['players'].index(player)+1);observer='p2' if side=='p1' else 'p1'
            initial=game['public'][:game['decisions'][0]['cutoff']]
            preview=[line.split('|')[3].split(',')[0] for line in initial if line.startswith('|poke|'+side+'|')]
            if model is None:
                model=JointOpponentModel(args.model or base/'training/adaptation/model.pt',base/'catalog.json',game['teams'][observer],preview,
                    observer=observer,particles=args.particles,configurations=args.configurations,branches=args.branches,
                    initial_events=initial,seed=1009+group,game_id=game['id'],opponent_id=player,device=device,time_budget=args.time_budget,
                    move_budget=args.move_budget)
            else:
                model.new_game(game['teams'][observer],preview,game['id'],initial_events=initial,observer=observer)
            i=0;window=0;decision=0
            while i<len(game['decisions']) and (not args.max_windows or window<args.max_windows):
                j=i+1
                while j<len(game['decisions']) and game['decisions'][j]['cutoff']==game['decisions'][i]['cutoff']:j+=1
                start=game['decisions'][i]['cutoff'];end=game['decisions'][j]['cutoff'] if j<len(game['decisions']) else len(game['public'])
                row=next((r for r in game['decisions'][i:j] if r['side']==side),None)
                before=model.query();record=dict(game=game['id'],group=group,player=player,side=side,encounter=encounter,window=window,
                    valid_prediction=before['prediction_valid'] and not blocked,hidden={})
                record['action_mass_error']=abs(sum(a['probability'] for a in before['prediction']['actions'])-1) if record['valid_prediction'] else None
                # Truth and the modelled player's request are used only below to score.
                if row:
                    record['decision']=decision;decision+=1
                    distribution=defaultdict(float)
                    for a in before['prediction']['actions']:distribution[identity(a['action'])]+=a['probability']
                    target=identity(row['actions'][row['target']]['key'])
                    metric=score(distribution,target,record['valid_prediction'])
                    metric.update(target=target,top5=sorted(distribution,key=distribution.get,reverse=True)[:5],
                        confidence=max(distribution.values()) if record['valid_prediction'] else None,
                        brier=float(sum(p*p for p in distribution.values())-2*metric['probability']+1) if record['valid_prediction'] else 2.)
                    record['action']=metric;record['phase']='preview' if row['request'].get('teamPreview') else 'forced' if row['request'].get('forceSwitch') else 'turn'
                    known_fields=model._configuration_facts(game['public'][:start])
                    for name,key in [('moves','moves'),('item','item'),('ability','ability'),('tera','teraType')]:
                        scores=[]
                        for slot,mon in enumerate(game['teams'][side]):
                            known=known_fields[slot]
                            values={ident(x['value']) or 'none':x['probability'] for x in before['opponent']['initial_configuration'][slot][key]}
                            if key=='moves':
                                public=set(known.get('moves',[]));values={k:v for k,v in values.items() if k not in public}
                                scores.extend(dict(slot=slot+1,target=ident(move),**score(values,ident(move),record['valid_prediction'])) for move in mon[key] if ident(move) not in public)
                            elif key not in known:scores.append(dict(slot=slot+1,target=ident(mon[key]) or 'none',**score(values,ident(mon[key]) or 'none',record['valid_prediction'])))
                        record['hidden'][name]=scores
                our=next((command for s,command in game['commands'][i:j] if s==observer),None)
                tic=time.time();attempted=not blocked
                after=model.update(game['public'][start:end],our,private_events=private[observer][start:end]) if attempted else before
                calls=[]
                while attempted:
                    cost=after['inference_cost'];phase=after['pending_update']['phase'] if after['pending_update'] else None
                    calls.append(dict(seconds=cost['seconds'],pending_phase=phase))
                    if model.pending is None:break
                    print(json.dumps(dict(group=group,game=game['id'],window=window,pending=phase,
                        calls=len(calls),seconds=round(time.time()-tic,2),completed_actions=cost.get('completed_actions'),
                        recovery_candidates=cost.get('completed_recovery_candidates'))),flush=True)
                    after=model.resume()
                applied=attempted and after['prediction_valid'] and after['inference_cost'].get('observation_applied',False)
                blocked=not applied
                fact_error=0.
                if applied:
                    for known,mon in zip(model.known,after['opponent']['initial_configuration']):
                        for key in ('item','ability','teraType','moves'):
                            if key not in known:continue
                            values=known[key] if key=='moves' else [known[key]]
                            for value in values:
                                mass=sum(r['probability'] for r in mon[key] if ident(r['value'])==ident(value))
                                fact_error=max(fact_error,abs(mass-1))
                record.update(update_seconds=time.time()-tic if attempted else None,update_attempted=attempted,valid_update=applied,ess=after['uncertainty']['effective_sample_size'],
                    configurations=after['joint_posterior']['distinct_configurations'],mismatch=after['model_mismatch'],inference_cost=after['inference_cost'],
                    calls=calls,confirmed_fact_error=fact_error if applied else None)
                records.append(record);i=j;window+=1
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                if attempted:print(json.dumps(dict(group=group,game=game['id'],window=window,valid=applied,seconds=round(record['update_seconds'],2))),flush=True)
                if group==args.start_group and applied and window==4:
                    (args.output/'opponent.json').write_text(json.dumps(model.dossier(),ensure_ascii=False,indent=2,allow_nan=False))
        def peak_rss(pid):
            return next(int(line.split()[1])/1024 for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('VmHWM:'))
        result=dict(records=records,seconds=time.time()-started,model_version=model.version,
            memory_mb=dict(python_peak_rss=peak_rss('self'),engine_peak_rss=peak_rss(model.engine.process.pid),cuda_peak_allocated=torch.cuda.max_memory_allocated(device)/2**20))
        (args.output/'parts'/f'{group:05}.json').write_text(json.dumps(result,allow_nan=False))
        return result
    finally:
        stream.close()
        if model is not None:
            model.close();del model
        torch.cuda.empty_cache()


def aggregate(rows):
    if not rows:return dict(count=0,top1=None,top3=None,top5=None)
    result=dict(count=len(rows),**{f'top{k}':float(np.mean([r['hits'][i] for r in rows])) for i,k in enumerate((1,3,5))},
                nll=float(np.mean([r['nll'] for r in rows])),coverage=float(np.mean([r['covered'] for r in rows])))
    if 'brier' in rows[0]:result['brier']=float(np.mean([r['brier'] for r in rows]))
    calibrated=[r for r in rows if r.get('confidence') is not None]
    if calibrated:
        confidence=np.array([r['confidence'] for r in calibrated]);correct=np.array([r['hits'][0] for r in calibrated]);bins=np.minimum((confidence*10).astype(int),9)
        result['ece_valid_predictions']=float(sum(abs(correct[bins==i].sum()-confidence[bins==i].sum()) for i in range(10))/len(calibrated))
        result['calibration_count']=len(calibrated)
    return result


def main(args):
    if args.output is None:args.output=args.directory/'training'/('joint_'+args.split)
    (args.output/'parts').mkdir(parents=True,exist_ok=True)
    started=time.time();parts=[]
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(evaluate,(args,g)) for g in range(args.start_group,args.start_group+args.groups)]
        for future in as_completed(futures):parts.append(future.result())
    rows=[r for p in parts for r in p['records']]
    actions=[r['action'] for r in rows if 'action' in r and r['phase']!='preview']
    metrics={name:aggregate([s for r in rows for s in r['hidden'].get(name,[])]) for name in ('moves','item','ability','tera')}
    metrics['action']=aggregate(actions)
    summary=dict(model_version=parts[0]['model_version'],split=args.split,games=len({r['game'] for r in rows}),windows=len(rows),
        particles=args.particles,configuration_proposals=args.configurations,mechanics_branches=args.branches,move_budget=args.move_budget,
        max_windows_per_game=args.max_windows,seconds=time.time()-started,metrics=metrics,
        workers=args.workers,peak_memory_mb_per_worker={key:max(p['memory_mb'][key] for p in parts) for key in parts[0]['memory_mb']},
        by_phase={phase:aggregate([r['action'] for r in rows if 'action' in r and r['phase']==phase]) for phase in ('preview','turn','forced')},
        update_failure_rate=float(np.mean([not r['valid_update'] for r in rows if r['update_attempted']])),
        unprocessed_window_rate=float(np.mean([not r['valid_update'] for r in rows])),
        valid_prediction_rate=float(np.mean([r['valid_prediction'] for r in rows])),
        median_update_seconds=float(np.median([r['update_seconds'] for r in rows if r['update_attempted']])),
        p95_update_seconds=float(np.quantile([r['update_seconds'] for r in rows if r['update_attempted']],.95)),
        scope='Actual joint C/X/M/Z filtering. Opponent truth is scoring-only. Hidden fields scored before decisions; moves are micro recall. Actions merge Tera and use original roster switch identity; preview is reported separately. Invalid full-history posteriors count as misses.')
    (args.output/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=ROOT/'artifacts/opponent_joint');p.add_argument('--split',choices=['calibration','test'],default='calibration')
    p.add_argument('--output',type=Path);p.add_argument('--model',type=Path)
    p.add_argument('--time-budget',type=float,default=60);p.add_argument('--start-group',type=int,default=0)
    p.add_argument('--groups',type=int,default=2);p.add_argument('--games',type=int,default=1);p.add_argument('--max-windows',type=int,default=0)
    p.add_argument('--particles',type=int,default=1024);p.add_argument('--configurations',type=int,default=128);p.add_argument('--branches',type=int,default=32)
    p.add_argument('--move-budget',type=int,default=32)
    p.add_argument('--workers',type=int,default=2);main(p.parse_args())
