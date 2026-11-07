"""Evaluate public neural predictions on the shared native-baseline cases."""
import argparse,json,pickle,sys,time
from collections import defaultdict
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from arena.opponent.joint.network import JointNetwork
from arena.opponent.joint.tokens import Vocabulary,ident
from arena.opponent.joint.training_data import EpisodeDataset,collate_episodes
from arena.opponent.joint.training_objective import TrainingTask,to_device
from opp_joint_topk import TopK,configuration_marginals


def configuration_scores(task,h,batch,episodes,cases,metrics,samples):
    for i,episode in enumerate(episodes):
        case=cases[episode['battle_id'],episode['side']]
        points=((batch['segments'][i]==1)&batch['score'][i]&(batch['target'][i]>=0)).nonzero().flatten()
        assert len(points)==len(case['cutoffs'])
        preview=torch.tensor([task.vocab.token('species',s) for s in case['public_preview']],device=h.device)
        for jj in points.split(8):
            predicted=configuration_marginals(task,h[i,jj],preview[None].expand(len(jj),-1),samples)
            truth=batch['configuration'][i,jj].reshape(-1,6,21)
            hidden=torch.as_tensor(case['hidden'],device=h.device)[batch['position'][i,jj]]
            for name,field in [('item',1),('ability',2),('tera',4)]:
                keep=hidden[:,:,field]&(truth[:,:,field]>0)
                metrics[name].add(predicted[name][keep],truth[:,:,field][keep])
            moves=truth[:,:,17:];unknown=hidden[:,:,17:]&(moves>0)
            values=predicted['moves'];values[:,:,~task.config_allowed[17]]=-torch.inf
            revealed=((~hidden[:,:,17:])&(moves>0)).nonzero()
            if len(revealed):
                row,slot,move=revealed.T;values[row,slot,moves[row,slot,move]]=-torch.inf
            order=values.argsort(dim=-1,descending=True,stable=True)[...,:5]
            metrics['moves'].add_order(order[:,:,None,:].expand(-1,-1,4,-1)[unknown],moves[unknown])


@torch.inference_mode()
def main(args):
    torch.set_num_threads(3);base=ROOT/'artifacts/opponent_joint';out=args.output;device=args.device
    action_cases=[json.loads(line) for line in (out/'pokechamp_action_cases.jsonl').read_text().splitlines()]
    queries=defaultdict(dict)
    for row in action_cases:queries[row['game'],row['side']][row['decision']]=row
    protocol=json.loads((out/'cases.json').read_text());cohort=protocol.get('cohort')
    dataset=EpisodeDataset(base/'training/data','controlled','test',limit=0 if cohort else 8192,length=100000)
    dataset.rows=[row for row in dataset.rows if (row.battle_id,row.side) in queries]
    previous={};last={};cases={}
    for row in cohort or []:
        key=(row['game'],row['side']);previous[key]=last.get(row['player']);last[row['player']]=key
    if args.configuration_samples or cohort:
        for path in sorted((out/'cases').glob('*.pkl')):
            for game in pickle.loads(path.read_bytes()):
                for side,case in game['cases'].items():
                    preview=[line.split('|')[3].split(',')[0] for line in game['public'] if line.startswith('|poke|'+side+'|')]
                    cases[game['id'],side]=case|{'public_preview':preview}
    model=JointNetwork.load(args.model or base/'training/model.pt',device);vocab=Vocabulary(base/'catalog.json');task=TrainingTask(model,vocab).to(device).eval()
    # Native PokéChamp Dex supplies only public species/form equivalences.
    dex=json.loads((ROOT/'external/pokechamp/repo/poke_env/data/static/pokedex/gen9pokedex.json').read_text())
    def switch_name(s):return ident(dex.get(s,{}).get('baseSpecies',s))
    results=[];started=time.time();config_metrics=defaultdict(TopK)
    for start in range(0,len(dataset),args.batch):
        samples=[dataset[i] for i in range(start,min(start+args.batch,len(dataset)))]
        if cohort:
            # Match the selected encounters: no earlier unselected game enters
            # the first encounter; the second may use the first public history.
            for sample in samples:
                key=(sample['battle_id'],sample['side']);query=dataset.read(dataset.locations[key]);parts=[]
                if previous[key] is not None:
                    past=dataset.read(dataset.locations[previous[key]])
                    parts.append((past,0,len(past['target']),False,0))
                parts.append((query,0,len(query['target']),True,1));sample['parts']=parts;sample['has_history']=len(parts)>1
        raw=collate_episodes(samples);batch=to_device(raw,device)
        torch.manual_seed(2027+start)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            h,_=model.encode(batch['public_events'],batch['public_numeric'],batch['public_state'],channel=1,position=batch['position'],segments=batch['segments'])
            z,_=model.strategy_paths(h,samples=4,game_start=batch['game_start'])
            for i,sample in enumerate(samples):
                for t,row in queries[sample['battle_id'],sample['side']].items():
                    j=((batch['segments'][i]==1)&(batch['position'][i]==t)).nonzero().item()
                    preview=(torch.tensor([vocab.token('species',s) for s in cases[sample['battle_id'],sample['side']]['public_preview']],device=device)[None]
                             if cohort else batch['preview'][i,j][None])
                    keys=model.outcome_keys(task.move_tokens,preview,task.kind_tokens)
                    probs=torch.stack([model.outcome_logits(h[i,j][None],path[i,j][None],keys).float().softmax(-1)[0] for path in z]).mean(0)
                    width=len(vocab.move_ids);move=probs[:2*width].reshape(width,2).sum(-1).tolist();rest=probs[2*width:].tolist()
                    values={'move:'+vocab.tokens[token].split(':',1)[1]:p for token,p in zip(vocab.move_ids,move)}
                    for token,p in zip(preview[0].tolist(),rest[:6]):
                        name='switch:'+switch_name(vocab.tokens[token].split(':',1)[1]);values[name]=values.get(name,0)+p
                    values['unknown']=rest[6]
                    kind,name=next(iter(row['label'].items()));name=switch_name(ident(name)) if kind=='switch' else ident(name);target=kind+':'+name
                    order=sorted(values,key=lambda x:(-values[x],x));hits=[int(target in order[:k]) for k in (1,3,5)]
                    results.append(dict(game=row['game'],side=row['side'],decision=t,phase=row['phase'],target=target,top5=order[:5],hits=hits))
            if args.configuration_samples:
                torch.manual_seed(1701+start)
                configuration_scores(task,h,batch,samples,cases,config_metrics,args.configuration_samples)
        print(json.dumps(dict(player_games=min(start+args.batch,len(dataset)),points=len(results),seconds=round(time.time()-started,1))),flush=True)
    hits=[sum(r['hits'][i] for r in results) for i in range(3)]
    metric=dict(count=len(results),hits=hits,**{f'top{k}':h/len(results) for k,h in zip((1,3,5),hits)})
    result=dict(method='Current model public-outcome head',metrics=metric,player_games=len(dataset),decision_points=len(results),seconds=time.time()-started,
                input='Public-history encoder and trained outcome head; no true hidden configuration, private state, submitted-action history or true legal-action list.',
                history_scope='Same selected encounters as the joint filter; complete first-encounter public support for the second encounter.' if cohort else 'Up to 32 public support steps from a previous game.',
                action_scope='Same fixed non-preview cases as PokéChamp. Move/Tera classes are marginalized to move identity; switches use public species identity. Unknown outcome remains a class and counts as a miss for submitted-action targets.',
                limitation='This evaluates the existing head trained on observable outcomes, not the conditional policy head or joint particle-filter marginal. Submission labels include actions that did not execute.')
    assert len(results)==len(action_cases)
    if args.configuration_samples:
        result.update(method='Current neural prediction heads',category='supervised',configuration_samples=args.configuration_samples,
            decision_points=protocol['decisions'],metrics={k:v.result() for k,v in config_metrics.items()}|{'action':metric},
            configuration_input='Causal public-history configuration decoder with sampled prefixes; shared unrevealed-field masks are used only for scoring. No joint filtering or true hidden decoder prefixes.')
        assert {k:v.count for k,v in config_metrics.items()}==protocol['counts']
    (out/('current_neural.json' if args.configuration_samples else 'current_public_action.json')).write_text(json.dumps(result,indent=2))
    (out/'current_public_action_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'artifacts/opponent_joint/baselines')
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch',type=int,default=16);p.add_argument('--configuration-samples',type=int,default=0)
    p.add_argument('--model',type=Path)
    main(p.parse_args())
