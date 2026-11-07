"""History importance sampling for missing configuration and state support."""
import math
import time
from collections import defaultdict

import numpy as np
import torch

from .data import collate
from .intent import actor_row
from .tokens import ident
from .training_data import PublicState
from .strategy import change_eligible,advance_changes


@torch.inference_mode()
def score_paths(model,teams,trajectories):
    """Correct uniform-action, observation-conditioned path proposals.

    Each native step samples (a, X') with density L(o,a,X') / sum_a L(o,a).
    Multiplying by pi(a)*sum_a L(o,a) recovers the actor's joint path density.
    Z is sampled from its causal transition, so its prior/proposal ratio is one.
    """
    groups=defaultdict(list);nodes={}
    for i,path in enumerate(trajectories):
        if path is None or not path['complete']:continue
        public=PublicState(model.vocab,model.side,[model.vocab.token('species',s) for s in model.preview])
        history=list(model.initial_events);public.update(history);previous=np.zeros(4,np.int64)
        cutoff=0;position=model.start_position;rows=[];selected=[];decision_steps=[];preview_count=0
        for t,(record,step,view) in enumerate(zip(model.journal,path['steps'],path['rows'])):
            row,actions=actor_row(model.vocab,view,teams[i],history[cutoff:],model.side,previous,position,public)
            j=next(k for k,a in enumerate(actions) if a['key']==step['action'] or
                   (a['kind']=='team' and a['order'][0]==int(step['action'][5])))
            step['noise']=None
            if actions[j]['kind']!='waiting':
                rows.append(row);selected.append(j);decision_steps.append(t)
                preview_count+=actions[j]['kind']=='team';previous=row['actions'][j]
                cutoff=len(history);position+=1
            history.extend(record['events']);public.update(record['events'])
        nodes[i]=dict(rows=rows,selected=selected,decision_steps=decision_steps,previous=previous,
            cutoff=cutoff,position=position,log_weight=path['log_mechanics_mass']-preview_count*math.lgamma(6))
        groups[len(rows)].append(i)
    for length,indices in groups.items():
        batch=collate([nodes[i]['rows'] for i in indices],model.device)
        h,memory=model.model.encode(batch['actor_events'],batch['actor_numeric'],batch['actor_state'],
            batch['configuration'],position=batch['position'])
        roots=None
        if model.start_z is not None:
            roots=model.start_z[torch.as_tensor(model.rng.integers(len(model.start_z),size=len(indices)),device=model.device)]
        z=roots;states=[];change_history=torch.zeros((len(indices),5),device=model.device,dtype=torch.int8)
        for t in range(length):
            eligible=change_eligible(batch['actor_state'][:,t],batch['action_mask'][:,t])
            z,noise,mode,_,_=model.model.sample_strategy(h[:,t],z,generator=model.torch_rng,eligible=eligible);states.append(z)
            change_history=advance_changes(change_history,mode,eligible)
            for k,i in enumerate(indices):
                step=trajectories[i]['steps'][nodes[i]['decision_steps'][t]]
                step['noise']=noise[k].cpu().tolist()
        logits=model.model.policy_logits(h,batch['features'],torch.stack(states,1),action_tokens=batch['actions']).float()
        logp=logits.masked_fill(~batch['action_mask'],-torch.inf).log_softmax(-1)
        logp=model.update_action_log_probs(logp,batch['action_mask'])
        labels=torch.tensor([nodes[i]['selected'] for i in indices],device=model.device)
        scores=logp.gather(-1,labels[...,None]).squeeze(-1).sum(-1).cpu().numpy()
        for k,i in enumerate(indices):
            nodes[i].update(log_weight=nodes[i]['log_weight']+float(scores[k]),
                log_density=sum(s['random_log_probability'] for s in trajectories[i]['steps'])+float(scores[k])-sum(s['action'].startswith('team:') for s in trajectories[i]['steps'])*math.lgamma(6),memory=[m[k] for m in memory],
                z=z[k],change_history=change_history[k],start_z=None if roots is None else roots[k].cpu().tolist())
    return nodes


def recover(model,deadline=float('inf'),work=None):
    """Resume full-history proposals; weight a batch only after every path finishes."""
    from .filter import normalize,systematic
    started=time.monotonic();protected=set(map(int,model.handles))
    if work is None:
        mass=np.bincount(model.c,weights=model.weights,minlength=len(model.configurations))
        anchors=[model.configurations[i] for i in np.argsort(-mass) if mass[i]>0]
        stop=next((i+1 for i,r in enumerate(model.journal) if r['event_id']==model.mismatch['at']),len(model.journal))
        focus=model.initial_events+[e for record in model.journal[:stop] for e in record['events']]
        active=next((line.split('|')[3].split(',')[0] for line in reversed(focus)
                     if line.startswith(('|switch|'+model.side,'|drag|'+model.side))),None)
        slot=next((i for i,s in enumerate(model.preview) if ident(s)==ident(active)),None)
        work=dict(anchors=anchors,slot=slot,batch=0,diagnostics=[],teams=None,handles=[],paths=[],
            runs=0,seconds=0.,reused_histories=0,proposal=None)
        model.coverage['recovery_attempts']=model.coverage.get('recovery_attempts',0)+1
        if any(len(f.get('moves',[]))>4 for f in model.known):
            model.mismatch['diagnosis']='revealed initial moves exceed the legal four slots'
            model.cost.update(recovery_batches=0,recovery_seconds=0.,observation_applied=False)
            return False,None
    accepted=0;pending=False
    while work['batch']<2:
        if time.monotonic()>=deadline:pending=True;break
        if work['teams'] is None:
            teams,info,proposal=model.sampler.repair(model.preview,work['anchors'],model.context,
                model.known,work['slot'],count=model.configuration_budget,deadline=deadline,work=work['proposal'])
            work['proposal']=proposal
            if proposal is not None:pending=True;break
            work['diagnostics'].append(info)
            if not teams:break
            starts=model.engine.call('batch',jobs=[dict(method='create',args=dict(teams={model.observer:model.own_team,model.side:t})) for t in teams])
            work.update(teams=teams,logratio=None,handles=[r['handle'] for r in starts],paths=[None]*len(teams),
                seeds=model.rng.integers(1,2**30,size=len(teams)).tolist())
            if deadline!=float('inf'):pending=True;break
        teams=work['teams'];paths=work['paths'];info=work['diagnostics'][-1]
        if work['logratio'] is None:
            work['logratio']=model.sampler.repair_weights(teams,info,model.context,model.prior_context,model.known)
        missing=[i for i,p in enumerate(paths) if p is None or p.get('timed_out')]
        work['reused_histories']+=len(paths)-len(missing)
        if missing and time.monotonic()<deadline:
            if deadline!=float('inf'):missing=missing[:max(1,math.ceil(4*(deadline-time.monotonic())))]
            updates=model.engine.call('transition_groups',jobs=[dict(handle=work['handles'][i],seed=work['seeds'][i],
                continuation=paths[i].get('continuation') if paths[i] else None) for i in missing],
                side=model.side,observer=model.observer,records=model.journal,history=model.initial_events,
                branches=model.branches,budget_ms=None if deadline==float('inf') else max(0,(deadline-time.monotonic())*1000))
            for i,path in zip(missing,updates):
                if path is not None:paths[i]=path
        if any(p is None or p.get('timed_out') for p in paths) or time.monotonic()>=deadline:
            pending=True;break
        nodes=score_paths(model,teams,paths);logw=np.full(len(teams),-np.inf)
        for i,node in nodes.items():logw[i]=work['logratio'][i]+node['log_weight']
        weights=normalize(logw)
        info.update(complete_histories=len(nodes),proposed_configurations=len(teams),
            mechanics_exact=all(p.get('likelihood_is_exact',False) for p in paths))
        work['runs']+=sum(p['attempts'] for p in paths)
        if weights is None:
            unused=set(work['handles'])|{p['handle'] for p in paths if p['complete']}
            if unused-protected:model.engine.call('release',handles=list(unused-protected))
            work.update(teams=None,handles=[],paths=[],batch=work['batch']+1)
            continue
        indices=systematic(weights,model.n,model.rng)
        model.configurations=teams;model.c=indices
        model.handles=np.array([paths[i]['handle'] for i in indices]);model.weights=np.full(model.n,1/model.n)
        model.paths=[paths[i]['steps'] for i in indices];model.path_start_z=[nodes[i]['start_z'] for i in indices]
        model.path_log_density=np.array([nodes[i]['log_density'] for i in indices])
        model.previous=np.stack([nodes[i]['previous'] for i in indices])
        model.memory=[torch.stack([nodes[i]['memory'][j] for i in indices]) for j in range(len(next(iter(nodes.values()))['memory']))]
        model.z=torch.stack([nodes[i]['z'] for i in indices])
        model.change_history=torch.stack([nodes[i]['change_history'] for i in indices])
        # Inference memory follows the same decision boundaries as actor memory.
        chosen=paths[indices[0]];model.events=list(model.initial_events);model.inference_memory=None
        model.position=model.start_position;model.cutoff=0
        for record,step,view in zip(model.journal,chosen['steps'],chosen['rows']):
            if step['noise'] is not None:
                phase=0 if view['request'].get('teamPreview') else 2 if view['request'].get('forceSwitch') else 1
                model._encode_inference(model.events[model.cutoff:],model._public_view().state(phase))
                model.inference_memory=model.context_memory;model.cutoff=len(model.events);model.position+=1
            model.events.extend(record['events'])
        model._encode_inference(model.events[model.cutoff:]);model.cached=None;model.mismatch=None
        model.applied_windows=len(model.journal);accepted=len(nodes)
        model.coverage['recovery_weighting']=info['weighting'];model.cost['recovery_mechanics_exact']=info['mechanics_exact']
        break
    work['seconds']+=time.monotonic()-started
    model.coverage['last_recovery']=work['diagnostics']
    model.cost.update(recovery_seconds=work['seconds'],recovery_batches=len(work['diagnostics']),
        recovery_mechanics_runs=work['runs']+(sum(p['attempts'] for p in work['paths'] if p) if pending else 0),
        completed_recovery_candidates=sum(p is not None and not p.get('timed_out',False) for p in work['paths']),
        total_recovery_candidates=len(work['paths']),reused_recovery_histories=work['reused_histories'],
        complete_recovery_histories=accepted,observation_supported=bool(accepted),observation_applied=bool(accepted),
        observation_probability_estimate=None,log_observation_probability_estimate=None,
        recovery_target='p(C) times the causal actor and mechanics likelihood of the complete history')
    if pending:return None,work
    unused=(protected|set(work['handles'])|{p['handle'] for p in work['paths'] if p and p['complete']})-set(map(int,model.handles))
    if unused:model.engine.call('release',handles=list(unused))
    if not accepted:
        model.mismatch['diagnosis']='no compatible joint history found in two completed proposal batches'
        model.cost['last_applied_window']=model.applied_windows
    return bool(accepted),None
