"""Player-conditioned prediction and adaptation in fixed multi-game sequences."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from arena.opponent.joint.network import JointNetwork
from arena.opponent.joint.training_data import EpisodeDataset,collate_episodes
from arena.opponent.joint.training_objective import to_device
from arena.opponent.joint.strategy import BEHAVIORS,behavior_opportunities,behavior_features,advance_changes,change_marginals


def prepare(base,split,device):
    torch.set_num_threads(2);dataset=EpisodeDataset(base/'training/data','controlled',split)
    players=defaultdict(list)
    for row in dataset.rows:players[row.player_id].append(row)
    groups=defaultdict(list)
    for player,rows in sorted(players.items()):
        if len(rows)<4:continue
        role,group,_=player.split('-');group=int(group)
        mode='abrupt' if role=='train' and group%6==0 else 'gradual' if role=='train' and group%6==1 else 'stable'
        groups[mode].append(player)
    rng=np.random.default_rng(5281);selected=[]
    budgets={'stable':128 if split=='train' else 64,'abrupt':384 if split=='train' else 32,'gradual':128 if split=='train' else 32}
    for mode,pool in groups.items():
        selected.extend((mode,str(p)) for p in rng.choice(pool,min(budgets[mode],len(pool)),replace=False))
    model=JointNetwork.load(base/'training/model.pt',device);jobs=[];output=[]
    for mode,player in selected:
        rows=sorted(players[player],key=lambda r:(int(r.order),r.battle_id))[:4]
        jobs.extend((mode,player,encounter,row) for encounter,row in enumerate(rows))
    with torch.inference_mode():
        for start in range(0,len(jobs),8):
            chunk=jobs[start:start+8];raw=[dataset.read(row) for _,_,_,row in chunk]
            samples=[dict(parts=[(data,0,len(data['target']),True,0)],source='controlled',battle_id=row.battle_id,
                player_id=player,side=row.side,has_history=False) for (_,player,_,row),data in zip(chunk,raw)]
            labels=[dict(mode=mode,player=player,encounter=encounter,game=row.battle_id) for mode,player,encounter,row in chunk]
            output.extend(encode_block(model,samples,labels,device))
            if start%80==0:print(json.dumps(dict(split=split,encoded_games=min(start+8,len(jobs)),total=len(jobs))),flush=True)
    directory=base/'training/adaptation';directory.mkdir(exist_ok=True)
    torch.save(dict(games=output,checkpoint=model.metadata,scope='Known configuration/state diagnostic; causal actor histories. Labels are separate from predictor inputs.'),directory/f'{split}.pt')
    print(json.dumps(dict(split=split,players=len(selected),games=len(output),modes={k:sum(m==k for m,_ in selected) for k in groups})),flush=True)


@torch.inference_mode()
def encode_block(model,samples,labels,device):
    batch=to_device(collate_episodes(samples),device)
    h,_=model.encode(batch['events'],batch['numeric'],batch['state'],batch['configuration'],
        position=batch['position'],segments=batch['segments']);output=[]
    for i,(sample,label) in enumerate(zip(samples,labels)):
        n=sample['parts'][0][2];entry=dict(**label,h=h[i,:n].cpu(),configuration=batch['configuration'][i,0].cpu())
        for key in ('features','actions','action_mask','target','teacher','teacher_valid','phase','change'):
            entry[key]=batch[key][i,:n].cpu()
        output.append(entry)
    return output


def prepare_external(base,device):
    from arena.opponent.joint.data import games
    from arena.opponent.joint.tokens import Vocabulary
    from arena.opponent.joint.training_data import controlled_records
    torch.set_num_threads(2);model=JointNetwork.load(base/'training/model.pt',device);vocab=Vocabulary(base/'catalog.json')
    output=[];samples=[];labels=[]
    for path in sorted((base/'training/adaptation/source/data').glob('*/*.gz')):
        mode='same_team' if path.parent.name=='calibration' else 'unseen_family'
        for game in games(path):
            for side,data in controlled_records(game,vocab):
                player=game['players'][int(side[1])-1]
                samples.append(dict(parts=[(data,0,len(data['target']),True,0)],source='controlled',battle_id=game['id'],
                    player_id=player,side=side,has_history=False))
                labels.append(dict(mode=mode,player=player,encounter=game['round'],game=game['id']))
                if len(samples)==8:
                    output.extend(encode_block(model,samples,labels,device));samples=[];labels=[]
    if samples:output.extend(encode_block(model,samples,labels,device))
    output.sort(key=lambda r:(r['player'],r['encounter']))
    torch.save(dict(games=output,checkpoint=model.metadata),base/'training/adaptation/external.pt')
    print(json.dumps(dict(player_games=len(output),players=len({r['player'] for r in output}))),flush=True)


def scores(probabilities,targets):
    p=np.asarray(probabilities);y=np.asarray(targets);ix=np.arange(len(y));order=np.argsort(-p,axis=-1)
    return dict(count=len(y),nll=float(-np.log(np.maximum(p[ix,y],1e-12)).mean()),
        brier=float((np.square(p).sum(-1)-2*p[ix,y]+1).mean()),
        **{f'top{k}':float((order[:,:k]==y[:,None]).any(-1).mean()) for k in (1,3,5)})


def paired(rows,first,second):
    groups=defaultdict(list)
    for row in rows:
        if first in row and second in row:groups[row['player']].append(row[second]-row[first])
    delta=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(5281)
    samples=rng.choice(delta,(1000,len(delta))).mean(-1)
    return dict(nll_reduction=float(delta.mean()),confidence_interval=np.quantile(samples,[.025,.975]).tolist(),players=len(delta))


@torch.inference_mode()
def diagnose(base,device):
    torch.set_num_threads(2);model=JointNetwork.load(base/'training/model.pt',device)
    records=torch.load(base/'training/adaptation/validation.pt',weights_only=True)['games']
    players=defaultdict(list)
    for row in records:players[row['player']].append(row)
    rows=[];states={};variances=[];gate_values=[]
    def game(record,previous=None,frozen=False):
        h=record['h'].to(device);features=record['features'].to(device);tokens=record['actions'].to(device)
        keys=model.action_features(features)+model.embedding(tokens).sum(-2)
        states=[];start=previous
        for t in range(len(h)):
            mean,lv,change,_=model.strategy_distribution(h[t],previous)
            variances.append(float(lv.exp().mean()))
            if previous is not None:
                modes=change.softmax(-1);gate=model.transition(torch.cat([h[t],previous])).sigmoid()*(modes[0]+.5*modes[2]);gate_values.append(float(gate.mean()))
            if start is None:start=mean
            previous=start if frozen else mean;states.append(previous)
        z=torch.stack(states);q=model.policy(torch.cat([h,z],-1));logits=(keys*q[:,None]).sum(-1)/(model.config.width**.5)
        p=logits.masked_fill(~record['action_mask'].to(device),-torch.inf).softmax(-1).cpu().numpy()
        return p,previous
    for player,sequence in players.items():
        previous=None
        for record in sequence[:3]:_,previous=game(record,previous)
        states[player]=previous
    for pi,(player,sequence) in enumerate(players.items()):
        target=sequence[3];p3,_=game(target,states[player]);p0,_=game(target)
        _,one=game(sequence[2]);p1,_=game(target,one);wrong=list(states)[(pi+1)%len(states)];pw,_=game(target,states[wrong]);pf,_=game(target,frozen=True)
        valid=np.flatnonzero((target['target'].numpy()>=0)&(target['phase'].numpy()==1)&(target['action_mask'].sum(-1).numpy()>1))
        y=target['target'].numpy()
        for t in valid:
            row=dict(player=player,mode=target['mode'],early=bool(t in valid[:5]),late=bool(t in valid[len(valid)//2:]))
            for name,p in [('history_3',p3),('history_1',p1),('history_0',p0),('wrong_history',pw),('frozen_z',pf)]:
                row[name]=float(-np.log(max(p[t,y[t]],1e-12)))
            rows.append(row)
    result=dict(scope='Existing network, mean Z, known C/X, full causal game histories; four games per player.',
        mean_transition_variance=float(np.mean(variances)),mean_retention=float(np.mean(gate_values)))
    for mode in ('stable','abrupt','gradual'):
        early=[r for r in rows if r['mode']==mode and r['early']];late=[r for r in rows if r['mode']==mode and r['late']]
        result[mode]=dict(early_nll={k:float(np.mean([r[k] for r in early])) for k in ('history_0','history_1','history_3','wrong_history')},
            cross_game=paired(early,'history_3','history_0'),within_game=paired(late,'history_0','frozen_z'))
    (base/'training/adaptation/diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


@torch.inference_mode()
def particle_sequence(model,sequence,particles,seed,initial=None,frozen=False,freeze_after=None,capture_transition=False):
    device=next(model.parameters()).device;generator=torch.Generator(device=device).manual_seed(seed)
    z=None if initial is None else initial.clone();weights=torch.full((particles,),1/particles,device=device)
    records=[]
    for record in sequence:
        change_history=torch.zeros((particles,5),device=device,dtype=torch.int8)
        h=record['h'].to(device);keys=model.action_features(record['features'].to(device))+model.embedding(record['actions'].to(device)).sum(-2)
        mask=record['action_mask'].to(device)
        # The action score is affine in Z; compute its small projection once.
        base=(keys*(h@model.policy.weight[:,:model.config.width].T+model.policy.bias)[:,None]).sum(-1)
        projection=keys@model.policy.weight[:,model.config.width:]
        if model.behavior_prior is not None:
            projection=torch.cat([projection,behavior_features(record['features'].to(device))*model.config.width**.5],-1)
        scale=model.config.width**.5*model.action_temperature;root=z
        for t in range(len(h)):
            if int(record['target'][t])<0 or int(record['phase'][t])>=3:continue
            detected=change_marginals(change_history,weights).cpu().numpy()
            change_probability=float(1-detected[0])
            eligible=torch.full((particles,),bool(int(record['phase'][t])==1 and mask[t].sum()>1),device=device)
            proposed,_,mode,_,_=model.sample_strategy(h[t].expand(particles,-1),z,generator=generator,eligible=eligible)
            freeze=frozen or (freeze_after is not None and t>=freeze_after)
            if freeze_after is not None and t==freeze_after:root=z
            if freeze:
                if root is None:root=proposed
                proposed=root
            probabilities=((base[t]+proposed@projection[t].T)/scale).masked_fill(~mask[t],-torch.inf).softmax(-1)
            p=(weights[:,None]*probabilities).sum(0)
            target=int(record['target'][t]);phase=int(record['phase'][t])
            if target>=0 and phase<3:
                ordered=p.argsort(descending=True);chosen=float(p[target])
                teacher=record['teacher'][t].to(device)
                teacher_kl=float((teacher*(teacher.clamp_min(1e-30).log()-p.clamp_min(1e-30).log())).sum())
                row=dict(player=record['player'],game=record['game'],encounter=record['encounter'],mode=record['mode'],
                    step=t,phase=phase,eligible=bool(phase==1 and mask[t].sum()>1),change=int(record['change'][t]),
                    nll=-float(np.log(max(chosen,1e-12))),brier=float(p.square().sum()-2*p[target]+1),teacher_kl=teacher_kl,
                    **{f'top{k}':int(bool((ordered[:k]==target).any())) for k in (1,3,5)})
                if capture_transition:
                    mass=torch.stack([weights[mode==k].sum() for k in range(3)])
                    evidence=torch.stack([(weights*probabilities[:,target])[mode==k].sum() for k in range(3)])
                    row['transition']=dict(prior_mass=mass.cpu().tolist(),
                        action_likelihood=(evidence/mass.clamp_min(1e-30)).cpu().tolist(),
                        posterior_mass=(evidence/evidence.sum()).cpu().tolist())
                if not freeze:
                    weights*=probabilities[:,target];weights/=weights.sum();z=proposed
                    change_history=advance_changes(change_history,mode,eligible)
                row['change_probability']=change_probability
                row['change_probabilities']=detected.tolist()
                f=record['features'][t,record['action_mask'][t]]
                row['behavior']={name:dict(probability=float(p[:len(f)].cpu()@f[:,BEHAVIORS[name][1]]),
                    target=int(f[target,BEHAVIORS[name][1]])) for name,eligible in behavior_opportunities(f,phase).items() if eligible}
                records.append(row)
                if not freeze and float(1/weights.square().sum())<particles*.5:
                    positions=(torch.arange(particles,device=device)+torch.rand((),device=device,generator=generator))/particles
                    indices=torch.searchsorted(weights.cumsum(0),positions).clamp_max(particles-1)
                    z=z[indices];weights.fill_(1/particles)
                    change_history=change_history[indices]
        if z is not None:
            positions=(torch.arange(particles,device=device)+torch.rand((),device=device,generator=generator))/particles
            indices=torch.searchsorted(weights.cumsum(0),positions).clamp_max(particles-1)
            z=z[indices];weights.fill_(1/particles)
    return records,z


def evaluate_particles(base,device,particles,seed):
    torch.set_num_threads(1);path=base/'training/adaptation/model.pt'
    model=JointNetwork.load(path if path.exists() else base/'training/model.pt',device)
    records=torch.load(base/'training/adaptation/validation.pt',weights_only=True)['games']
    records+=torch.load(base/'training/adaptation/external.pt',weights_only=True)['games']
    players=defaultdict(list)
    for row in records:players[row['player']].append(row)
    roots={};support_rows=[]
    for i,(player,sequence) in enumerate(players.items()):
        rows,root=particle_sequence(model,sequence[:3],particles,seed+i)
        roots[player]=root;support_rows.extend(rows)
        if i%16==0:print(json.dumps(dict(stage='support',players=i+1)),flush=True)
    output=[]
    names=list(players)
    for i,(player,sequence) in enumerate(players.items()):
        methods={}
        methods['history_3'],_=particle_sequence(model,sequence[3:],particles,seed+i,roots[player])
        methods['history_0'],_=particle_sequence(model,sequence[3:],particles,seed+i)
        _,one=particle_sequence(model,sequence[2:3],particles,seed+i)
        methods['history_1'],_=particle_sequence(model,sequence[3:],particles,seed+i,one)
        methods['wrong_history'],_=particle_sequence(model,sequence[3:],particles,seed+i,roots[names[(i+1)%len(names)]])
        methods['frozen_z'],_=particle_sequence(model,sequence[3:],particles,seed+i,frozen=True)
        eligible=[r['step'] for r in methods['history_3'] if r['eligible']]
        for j,row in enumerate(methods['history_3']):
            if not row['eligible']:continue
            entry=dict(player=player,game=row['game'],mode=row['mode'],step=row['step'],change=row['change'],
                early=row['step'] in eligible[:5],late=row['step'] in eligible[len(eligible)//2:])
            entry['methods']={name:{k:values[j][k] for k in ('nll','brier','top1','top3','top5','teacher_kl','change_probability','change_probabilities','behavior')} for name,values in methods.items()}
            output.append(entry)
        if i%16==0:print(json.dumps(dict(stage='query',players=i+1)),flush=True)
    events=[dict(player=r['player'],game=r['game'],onset=int(t-1)) for r in records if r['encounter']==3 for t in (r['change']==1).nonzero().flatten()]
    result=dict(seed=seed,particles=particles,scope='Known C/X diagnostic; submitted-action supervision isolates strategy inference.',rows=output,support=support_rows,events=events)
    summary={}
    for mode in ('stable','abrupt','gradual','same_team','unseen_family'):
        rows=[r for r in output if r['mode']==mode]
        def aggregate(values):return {name:{key:float(np.mean([r['methods'][name][key] for r in values])) for key in ('nll','brier','top1','top3','top5')} for name in ('history_0','history_1','history_3','wrong_history','frozen_z')}
        early=[r for r in rows if r['early']];late=[r for r in rows if r['late']]
        summary[mode]=dict(early=aggregate(early),late=aggregate(late))
    stable=np.array([r['change_probabilities'][1:] for r in support_rows if r['mode']=='stable' and r['eligible']])
    threshold=np.quantile(stable,.975,axis=0);delays=[]
    for event in events:
        rows=[r for r in output if r['player']==event['player'] and r['game']==event['game'] and r['step']>event['onset']][:5]
        detect=[j for j,r in enumerate(rows) if np.any(np.array(r['methods']['history_3']['change_probabilities'][1:])>threshold)]
        delays.append(detect[0]+1 if detect else None)
    summary['change']=dict(threshold=threshold.tolist(),events=len(delays),recall=float(np.mean([d is not None for d in delays])) if delays else None,
        median_delay=float(np.median([d for d in delays if d is not None])) if any(d is not None for d in delays) else None)
    result['summary']=summary
    (base/f'training/adaptation/particles-{seed}.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2),flush=True)


def change_ablation(base,device,seed):
    torch.set_num_threads(1);model=JointNetwork.load(base/'training/adaptation/model.pt',device)
    records=torch.load(base/'training/adaptation/validation.pt',weights_only=True)['games'];players=defaultdict(list)
    for record in records:players[record['player']].append(record)
    path=base/f'training/adaptation/particles-{seed}.json';result=json.loads(path.read_text())
    indexed={(r['player'],r['step']):r for r in result['rows']};changes=[]
    for i,(player,sequence) in enumerate(players.items()):
        record=sequence[3];labels=record['change']
        if record['mode'] not in ('abrupt','gradual') or not bool((labels>0).any()):continue
        onset=int((labels>0).nonzero()[0])-1
        _,root=particle_sequence(model,sequence[:3],result['particles'],seed+i)
        rows,_=particle_sequence(model,sequence[3:],result['particles'],seed+i,root,freeze_after=onset)
        errors=[]
        for row in rows:
            entry=indexed.get((player,row['step']))
            if entry is None:continue
            if row['step']<onset:errors.append(abs(row['nll']-entry['methods']['history_3']['nll']))
            entry['methods']['frozen_after_change']={k:row[k] for k in ('nll','brier','top1','top3','top5','teacher_kl')}
            entry['after_change']=row['step']>=onset
        changes.append(dict(player=player,mode=record['mode'],onset=onset,before_change_max_error=max(errors,default=0)))
    result['change_ablation']=changes;path.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(seed=seed,changes=len(changes),before_change_max_error=max(r['before_change_max_error'] for r in changes))),flush=True)


def fit_behavior_prior(base,device):
    torch.set_num_threads(1);torch.manual_seed(5281)
    checkpoint=base/'training/adaptation/model.pt'
    model=JointNetwork.load(checkpoint if checkpoint.exists() else base/'training/model.pt',device);model.behavior_prior=None
    records=torch.load(base/'training/adaptation/train.pt',weights_only=True)['games']
    width=max(r['features'].shape[1] for r in records);previous={};groups={};features=[];logits=[];teachers=[];indices=[]
    with torch.inference_mode():
        for ri,record in enumerate(records):
            h=record['h'].to(device);z=previous.get(record['player']);states=[]
            for t in range(len(h)):
                if int(record['phase'][t])<3:z=model.strategy_distribution(h[t],z)[0]
                states.append(z)
            previous[record['player']]=z
            f=record['features'].to(device);a=record['actions'].to(device);mask=record['action_mask'].to(device)
            base_logits=model.policy_logits(h,f,torch.stack(states),action_tokens=a).masked_fill(~mask,-1e9)
            valid=(record['phase']==1)&record['teacher_valid']&(record['action_mask'].sum(-1)>1)
            change=(record['change']>0).nonzero();onset=int(change[0])-1 if len(change) else len(h)
            for t in valid.nonzero().flatten().tolist():
                key=(record['player'],int(t>=onset));group=groups.setdefault(key,len(groups))
                phi=behavior_features(f[t]);pad=width-len(phi)
                features.append(torch.nn.functional.pad(phi,(0,0,0,pad)))
                logits.append(torch.nn.functional.pad(base_logits[t],(0,pad),value=-1e9))
                teachers.append(torch.nn.functional.pad(record['teacher'][t].to(device),(0,pad)))
                indices.append(group)
            if ri%400==0:print(json.dumps(dict(stage='behavior_inputs',games=ri)),flush=True)
    x=torch.stack(features).clone();offset=torch.stack(logits).clone();q=torch.stack(teachers).clone()
    index=torch.tensor(indices,device=device);counts=torch.bincount(index).float()
    coefficients=torch.zeros((len(groups),x.shape[-1]),device=device,requires_grad=True)
    optimizer=torch.optim.Adam([coefficients],lr=.05);history=[]
    for epoch in range(1,41):
        losses=[]
        for selected in torch.randperm(len(index),device=device).split(4096):
            coefficient=coefficients[index[selected]]
            logp=(offset[selected]+(x[selected]*coefficient[:,None]).sum(-1)).log_softmax(-1)
            loss=(-(q[selected]*logp).sum(-1)+.125*coefficient.square().sum(-1)/counts[index[selected]]).mean()
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch,objective=float(np.mean(losses))))
        if epoch%10==0:print(json.dumps(history[-1]),flush=True)
    with torch.no_grad():
        model.behavior_prior=dict(mean=coefficients.mean(0).cpu(),scale=coefficients.std(0).clamp_min(.2).cpu())
        before=float(-(q*offset.log_softmax(-1)).sum(-1).mean())
        after=float(-(q*(offset+(x*coefficients[index,None]).sum(-1)).log_softmax(-1)).sum(-1).mean())
    bank={name:[] for name in BEHAVIORS}
    for record in records:
        found=set()
        for t in range(len(record['h'])):
            mask=record['action_mask'][t];f=record['features'][t,mask]
            for name,eligible in behavior_opportunities(f,int(record['phase'][t])).items():
                if eligible and name not in found and len(bank[name])<16:
                    bank[name].append(dict(h=record['h'][t],features=f,actions=record['actions'][t,mask]));found.add(name)
        if all(len(probes)==16 for probes in bank.values()):break
    model.behavior_probes=bank
    model.save(base/'training/adaptation/model.pt',**model.metadata)
    report=dict(player_regimes=len(groups),decisions=len(index),dimensions=x.shape[-1],
        teacher_cross_entropy_before=before,teacher_cross_entropy_after=after,
        prior={k:v.tolist() for k,v in model.behavior_prior.items()},history=history)
    (base/'training/adaptation/behavior_training.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)


def runtime_check(base,device):
    import copy
    from arena.opponent.joint.data import games
    from arena.opponent.joint.filter import JointOpponentModel,systematic
    from arena.opponent.joint.rejuvenation import replay_path
    from opp_experiment_utils import windows
    torch.set_num_threads(1);source=list(games(base/'data/calibration/group-00000.jsonl.gz'))
    game=source[0];player=game['players'][1]
    following=next(g for g in source[1:] if player in g['players'])
    side='p'+str(following['players'].index(player)+1);observer='p2' if side=='p1' else 'p1'
    initial=game['public'][:game['decisions'][0]['cutoff']]
    model=JointOpponentModel(base/'training/adaptation/model.pt',base/'catalog.json',copy.deepcopy(game['teams']['p1']),
        [m['species'] for m in game['teams']['p2']],initial_teams=[copy.deepcopy(game['teams']['p2'])],
        initial_events=initial,particles=1024,configurations=4,device=device,opponent_id=player)
    parameters={k:v.detach().cpu().clone() for k,v in model.model.state_dict().items()}
    try:
        for i,record in enumerate(windows(game,'p1')):
            model.update(**record);assert model.mismatch is None,model.mismatch
            if i==5:break
        result=model.query();before={a['action']:a['probability'] for a in result['prediction']['actions']}
        change_before=model.change_history.clone();path_error=0.;change_replay=True
        protected=set(map(int,model.handles))
        positive=np.flatnonzero(model.weights>0)
        changed=(model.change_history[torch.as_tensor(positive,device=device)]>0).sum(-1)
        representatives={int(np.argmax(model.weights)),int(positive[int(changed.argmax())])}
        for i in representatives:
            node,_=replay_path(model,model.configurations[model.c[i]],model.paths[i],model.path_start_z[i],protected)
            change_replay&=node is not None and torch.equal(node['change_history'][0],model.change_history[i])
            if node is not None:
                path_error=max(path_error,float((node['z'][0]-model.z[i]).abs().max()))
                if node['handle'] not in protected:model.engine.call('release',handles=[node['handle']])
        profile=result['opponent']['strategy']['behavior_profile'];error=0.
        with torch.inference_mode():
            for name,probes in model.model.behavior_probes.items():
                values=[]
                for probe in probes:
                    h=probe['h'].to(device)[None,None].expand(model.n,1,-1)
                    f=probe['features'].to(device)[None,None].expand(model.n,1,-1,-1)
                    a=probe['actions'].to(device)[None,None].expand(model.n,1,-1,-1)
                    p=model.model.policy_logits(h,f,model.z[:,None],action_tokens=a)[:,0].softmax(-1)
                    values.append((p*f[:,0,:,BEHAVIORS[name][1]]).sum(-1))
                expected=float(torch.tensor(model.weights,device=device,dtype=torch.float32)@torch.stack(values).mean(0))
                error=max(error,abs(expected-profile[name]['probability']))
        state_path=base/'training/adaptation/player_state.pt';torch.save(model.snapshot(),state_path)
        rng=copy.deepcopy(model.rng);indices=rng.permutation(systematic(model.weights,model.n,rng))
        expected_z=model.z[torch.as_tensor(indices,device=device)].clone()
        next_initial=following['public'][:following['decisions'][0]['cutoff']]
        model.new_game(copy.deepcopy(following['teams'][observer]),[m['species'] for m in following['teams'][side]],
            following['id'],initial_events=next_initial,observer=observer)
        carry=bool(torch.equal(model.z,expected_z));reset=not model.journal and not model.facts and all(not f for f in model.known) and model.memory is None and model.applied_windows==0 and not bool(model.change_history.any())
        evidence=model.query()['opponent']['strategy']['evidence']
        model.restore(torch.load(state_path,weights_only=False))
        restored=model.query();restore_error=max(abs(a['probability']-before[a['action']]) for a in restored['prediction']['actions'])
        change_restored=torch.equal(change_before,model.change_history) and restored['opponent']['strategy']['recent_change']==result['opponent']['strategy']['recent_change']
        model.new_game(copy.deepcopy(following['teams'][observer]),[m['species'] for m in following['teams'][side]],
            following['id'],initial_events=next_initial,observer=observer,opponent_id='unseen-player')
        isolated=model.z is None and model.start_z is None and model.games_seen==1 and model.past_windows==0
        fixed=all(torch.equal(parameters[k],v.detach().cpu()) for k,v in model.model.state_dict().items())
        summary=dict(applied_windows=6,profile_probability_max_error=error,profiles=len(profile),
            same_player_state_carried=carry,new_game_state_reset=reset,side_changed=side!='p2',
            cross_game_evidence=evidence,snapshot_prediction_max_error=restore_error,different_player_isolated=isolated,network_weights_fixed=fixed,
            change_history_replayed=change_replay,change_snapshot_restored=change_restored,strategy_replay_max_error=path_error)
        summary['pass']=bool(len(profile)==4 and error<=1e-6 and carry and reset and restore_error<=1e-6 and isolated and fixed and change_replay and change_restored and path_error<=1e-5)
        (base/'training/adaptation/profile.json').write_text(json.dumps(dict(opponent_id=player,games_observed=1,strategy=result['opponent']['strategy']),ensure_ascii=False,indent=2))
        (base/'training/adaptation/runtime.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    finally:model.close()


def check_change_posterior(base):
    from arena.opponent.joint.filter import condition_weights,systematic
    prior=np.array([.42,.12,.06,.28,.08,.04])
    modes=torch.tensor([0,1,2,0,1,2]);eligible=torch.ones(6,dtype=torch.bool)
    history=advance_changes(torch.zeros((6,5),dtype=torch.int8),modes,eligible)
    policy=np.array([[.1,.9],[.8,.2],[.5,.5],[.3,.7],[.9,.1],[.6,.4]])
    observation=np.array([.85,.15]);likelihood=policy@observation
    posterior,_=condition_weights(prior,likelihood)
    mass=prior[:,None]*policy*observation[None];mass=mass.sum(-1);mass/=mass.sum()
    expected=np.array([mass[[0,3]].sum(),mass[[1,4]].sum(),mass[[2,5]].sum()])
    actual=change_marginals(history,posterior).numpy();error=float(np.max(abs(actual-expected)))
    uninformative,_=condition_weights(prior,policy@np.array([.4,.4]))
    no_evidence_error=float(np.max(abs(change_marginals(history,uninformative).numpy()-change_marginals(history,prior).numpy())))
    no_clock=torch.equal(history,advance_changes(history,2-modes,torch.zeros_like(eligible)))
    overlapping=advance_changes(history,torch.ones_like(modes),eligible)
    overlap=change_marginals(overlapping,posterior).numpy()
    overlap_error=float(abs(overlap[1]-1)+abs(overlap[2]-mass[[2,5]].sum()))
    expired=history
    for _ in range(5):expired=advance_changes(expired,torch.zeros_like(modes),eligible)
    expiry=not bool(expired.any())
    indices=systematic(posterior,1024,np.random.default_rng(83))
    sampled=change_marginals(history[indices],np.full(1024,1/1024)).numpy()
    sampling_error=float(np.max(abs(sampled-expected)))
    report=dict(exact_posterior_max_error=error,uninformative_observation_max_error=no_evidence_error,
        forced_decision_preserves_history=no_clock,overlapping_change_max_error=overlap_error,
        five_decision_expiry=expiry,resampled_marginal_max_error=sampling_error,
        prior=change_marginals(history,prior).tolist(),posterior=actual.tolist())
    report['pass']=bool(max(error,no_evidence_error,overlap_error)<1e-12 and no_clock and expiry and sampling_error<.002)
    (base/'training/adaptation/change_posterior.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


def diagnose_changes(base,device):
    torch.set_num_threads(1);directory=base/'training/adaptation'
    model=JointNetwork.load(directory/'model.pt',device);players=defaultdict(list)
    for row in torch.load(directory/'validation.pt',weights_only=True)['games']:players[row['player']].append(row)
    run=json.loads((directory/'particles-83.json').read_text())
    reference={(r['player'],r['game'],r['step']):r for r in json.loads((directory/'change_reference.json').read_text())}
    selected={}
    for event in run['events']:
        rows=[r for r in run['rows'] if r['player']==event['player'] and r['game']==event['game'] and r['step']>=event['onset']][:5]
        following=[r for r in run['rows'] if r['player']==event['player'] and r['game']==event['game'] and r['step']>event['onset']][:5]
        if len(rows)==5 and len(following)==5 and sum(reference[(r['player'],r['game'],r['step'])]['kl'] for r in rows)>=5:selected[event['player']]=event
    result=[]
    for i,(player,sequence) in enumerate(players.items()):
        if player not in selected:continue
        _,root=particle_sequence(model,sequence[:3],run['particles'],83+i)
        rows,_=particle_sequence(model,sequence[3:],run['particles'],83+i,root,capture_transition=True)
        onset=selected[player]['onset'];decisions=[]
        for row in [r for r in rows if r['eligible'] and r['step']>=onset][:5]:
            truth=reference[(player,row['game'],row['step'])]
            decisions.append(dict(step=row['step'],model_transition=row['transition'],teacher_kl=row['teacher_kl'],
                oracle_log_likelihood_ratio=truth['log_likelihood_ratio'],oracle_kl=truth['kl']))
        result.append(dict(player=player,onset=onset,decisions=decisions))
    (directory/'change_diagnostic.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(player_games=len(result),decisions=sum(len(r['decisions']) for r in result))),flush=True)


def summarize(base):
    directory=base/'training/adaptation';runs=[json.loads((directory/f'particles-{seed}.json').read_text()) for seed in (83,107,211)]
    rows=[dict(row,seed=run['seed']) for run in runs for row in run['rows']]
    def averages(selected,methods,metric_keys=('nll','brier','top1','top3','top5')):
        groups=defaultdict(list)
        for row in selected:groups[row['player']].append(row)
        return {method:{key:float(np.mean([np.mean([r['methods'][method][key] for r in group]) for group in groups.values()]))
            for key in metric_keys} for method in methods}
    def compare(selected,full,control,minimum=0.,metric_keys=('nll','brier','top1','top3','top5')):
        values=averages(selected,(full,control),metric_keys)
        interval=paired([dict(player=r['player'],**{m:r['methods'][m]['nll'] for m in (full,control)}) for r in selected],full,control)
        reduction=1-values[full]['nll']/values[control]['nll']
        return dict(methods=values,relative_nll_reduction=reduction,paired=interval,
            decisions=len({(r['player'],r['game'],r['step']) for r in selected}),
            passed=bool(reduction>=minimum and interval['confidence_interval'][0]>0 and values[full]['brier']<=values[control]['brier']))
    within=compare([r for r in rows if r['mode']=='stable' and r['late']],'history_0','frozen_z',.05)
    cross={}
    for mode in ('stable','same_team'):
        selected=[r for r in rows if r['mode']==mode and r['early']]
        cross[mode]=compare(selected,'history_3','history_0',.05)
        cross[mode]['wrong_player_control']=compare(selected,'history_3','wrong_history')
        cross[mode]['history_curve']=averages(selected,('history_0','history_1','history_3','wrong_history'))
    unknown=dict(cross_game=compare([r for r in rows if r['mode']=='unseen_family' and r['early']],'history_3','frozen_z'),
        within_game=compare([r for r in rows if r['mode']=='unseen_family' and r['late']],'history_0','frozen_z'))
    after={mode:compare([r for r in rows if r['mode']==mode and r.get('after_change')],
        'history_3','frozen_after_change') for mode in ('abrupt','gradual')}
    changed_groups=defaultdict(list)
    for row in rows:
        if row.get('after_change'):changed_groups[(row['mode'],row['seed'],row['player'])].append(row)
    for mode in after:
        curve=[]
        for start,end in ((0,5),(5,10),(10,20)):
            selected=[r for (m,*_),group in changed_groups.items() if m==mode for r in group[start:end]]
            if selected:
                curve.append(dict(decisions_after_change=[start+1,end],players=len({r['player'] for r in selected}),
                    methods=averages(selected,('history_3','frozen_after_change'),('nll','brier','teacher_kl'))))
        after[mode]['recovery_curve']=curve
    threshold=np.quantile([r['change_probabilities'][1:] for r in runs[0]['support'] if r['mode']=='stable' and r['eligible']],.975,axis=0)
    def alarm(row):return bool(np.any(np.array(row['methods']['history_3']['change_probabilities'][1:])>threshold))
    false_alarms={mode:float(np.mean([alarm(r) for r in rows if r['mode']==mode]))
        for mode in ('stable','same_team','unseen_family')}
    game_alarms=defaultdict(list)
    for row in rows:
        if row['mode'] in false_alarms:game_alarms[(row['mode'],row['seed'],row['player'],row['game'])].append(alarm(row))
    game_false_alarms={mode:float(np.mean([any(v) for (m,*_),v in game_alarms.items() if m==mode])) for mode in false_alarms}
    teacher_rows=json.loads((directory/'change_reference.json').read_text())
    reference={(r['player'],r['game'],r['step']):r['kl'] for r in teacher_rows}
    gradual_groups=defaultdict(list)
    for row in teacher_rows:
        if row['mode']=='gradual' and row['gradual'] and row['ordinary']:gradual_groups[(row['player'],row['game'])].append(row['kl'])
    unchanged=[dict(player=player,game=game) for (player,game),kl in gradual_groups.items() if max(kl)<1e-10]
    detections=[]
    for run in runs:
        for event in run['events']:
            following=[r for r in run['rows'] if r['player']==event['player'] and r['game']==event['game'] and r['step']>event['onset']][:5]
            actions=[r for r in run['rows'] if r['player']==event['player'] and r['game']==event['game'] and r['step']>=event['onset']][:5]
            alarms=[i+1 for i,r in enumerate(following) if alarm(r)]
            kl=sum(reference.get((r['player'],r['game'],r['step']),0.) for r in actions)
            detections.append(dict(**event,seed=run['seed'],delay=alarms[0] if alarms else None,
                observed_decisions=len(following),teacher_kl=kl,identifiable=len(following)==5 and len(actions)==5 and kl>=5))
    def detection_summary(values):
        delay=[v['delay'] for v in values if v['delay'] is not None]
        return dict(events=len({(v['player'],v['game'],v['onset']) for v in values}),
            recall=len(delay)/len(values),median_delay=float(np.median(delay)) if delay else None)
    change=dict(threshold=threshold.tolist(),threshold_source='Stable support games, seed 83; 2.5% false-alarm allocation to each change type, fixed for every query game and seed.',
        false_positive_rate=false_alarms,game_false_positive_rate=game_false_alarms,all_events=detection_summary(detections),
        identifiable_events=detection_summary([d for d in detections if d['identifiable']]),detections=detections,
        identifiable_definition='Five ordinary nontrivial decisions whose teacher KL against the old policy sums to at least five nats.')
    change['label_quality']=dict(teacher_reconstruction_max_error=max(r['teacher_error'] for r in teacher_rows),
        gradual_player_games=len(gradual_groups),gradual_labels_without_policy_change=unchanged)
    diagnostic=json.loads((directory/'change_diagnostic.json').read_text())
    likelihood_ratios=[v['model_transition']['action_likelihood'][1]/v['model_transition']['action_likelihood'][0]
        for case in diagnostic for v in case['decisions']]
    change['candidate_likelihood_diagnostic']=dict(seed=83,particles=1024,player_games=len(diagnostic),decisions=len(likelihood_ratios),
        reset_less_likely_than_keep=sum(v<1 for v in likelihood_ratios),mean_log_likelihood_ratio=float(np.log(likelihood_ratios).mean()),
        scope='Single-decision likelihoods conditional on sampled transition modes; this is not a cumulative Bayes factor.')
    behavior={};binary=[]
    for name in BEHAVIORS:
        selected=[]
        for row in rows:
            if row['mode'] not in ('stable','same_team','unseen_family'):continue
            if name not in row['methods']['history_3']['behavior']:continue
            entry={k:row[k] for k in ('player','game','step')};entry['methods']={}
            for method in ('history_3','history_0'):
                value=row['methods'][method]['behavior'][name];p=np.clip(value['probability'],1e-12,1-1e-12);y=value['target']
                entry['methods'][method]=dict(nll=float(-np.log(p if y else 1-p)),brier=float((p-y)**2))
            selected.append(entry)
        behavior[name]=compare(selected,'history_3','history_0',metric_keys=('nll','brier'));binary.extend(selected)
    behavior['combined']=compare(binary,'history_3','history_0',metric_keys=('nll','brier'))
    runtime=json.loads((directory/'runtime.json').read_text());finite=json.loads((base/'training/recovery/finite.json').read_text())['summary']
    posterior_check=json.loads((directory/'change_posterior.json').read_text())
    identifiable=change['identifiable_events']
    checks=dict(behavior_profile=runtime['profiles']==4 and runtime['profile_probability_max_error']<=1e-6 and behavior['combined']['passed'],
        within_game=within['passed'],cross_game=all(v['passed'] and v['wrong_player_control']['passed'] for v in cross.values()),
        strategy_change=max(false_alarms.values())<=.05 and identifiable['recall']>=.8 and (identifiable['median_delay'] or 99)<=3 and all(v['passed'] for v in after.values()),
        unseen_behavior=all(v['passed'] for v in unknown.values()),state_continuity=runtime['pass'] and finite['pass'] and posterior_check['pass'])
    behavior_training=json.loads((directory/'behavior_training.json').read_text())
    reference=json.loads((base/'training/adaptation.json').read_text())['reference']
    report=dict(task='joint_model_step_5_player_understanding_and_adaptation',particles=1024,seeds=[83,107,211],
        checkpoint=str(directory/'model.pt'),
        method=dict(encoder='Frozen 8-layer, width-256 causal Transformer',
            strategy='32 neural coordinates and 12 continuous behavior coefficients; keep/reset/drift particle transitions',
            behavior_prior='Empirical Gaussian from regularized teacher-policy fits on training players',
            change_inference='Posterior marginal of the transition modes within the last five meaningful ordinary decisions, using the same joint particle weights as C/X/M/Z; abrupt and gradual marginals may overlap',
            online='Observation likelihood updates particle weights and strategy states; neural weights remain fixed'),
        data=dict(training_players=640,training_player_games=2560,validation_players=128,validation_player_games=512,
            additional_players=80,additional_player_games=320,query_decisions=len(runs[0]['rows']),
            behavior_fit_decisions=behavior_training['decisions'],behavior_fit_regimes=behavior_training['player_regimes']),
        training=dict(behavior_prior={k:behavior_training[k] for k in ('dimensions','teacher_cross_entropy_before','teacher_cross_entropy_after')}),
        scope='Strategy-effect measurements hold C/X known and feed observed submitted actions; the separate native joint-filter check marginalizes unknown actions. Complete hidden-configuration benchmarks belong to step 6.',
        within_game=within,cross_game=cross,unseen_behavior=unknown,after_change=after,change_detection=change,
        behavior_profile=behavior,runtime=runtime,finite_posterior=finite,change_posterior=posterior_check,criteria=checks,passed=all(checks.values()),reference=reference,
        limitations=['The fixed scenarios are development measurements, not locked-test generalization.',
            'Same-team and unseen-family sequences use separately generated players.',
            'Behavior intervals are model-posterior quantiles on fixed reference situations.',
            'Only five independent abrupt-change events meet the specified teacher-KL evidence criterion.'])
    (base/'training/adaptation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('within_game','cross_game','unseen_behavior','after_change','criteria','passed')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['prepare','external','diagnose','evaluate','behavior-prior','runtime','change-check','change-ablation','change-diagnostic','summarize'])
    parser.add_argument('--directory',type=Path,default=ROOT/'artifacts/opponent_joint')
    parser.add_argument('--split',choices=['train','validation'],default='validation')
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--particles',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=83)
    args=parser.parse_args()
    if args.operation=='prepare':prepare(args.directory,args.split,args.device)
    elif args.operation=='external':prepare_external(args.directory,args.device)
    elif args.operation=='diagnose':diagnose(args.directory,args.device)
    elif args.operation=='behavior-prior':fit_behavior_prior(args.directory,args.device)
    elif args.operation=='runtime':runtime_check(args.directory,args.device)
    elif args.operation=='change-check':check_change_posterior(args.directory)
    elif args.operation=='change-ablation':change_ablation(args.directory,args.device,args.seed)
    elif args.operation=='change-diagnostic':diagnose_changes(args.directory,args.device)
    elif args.operation=='summarize':summarize(args.directory)
    else:evaluate_particles(args.directory,args.device,args.particles,args.seed)
