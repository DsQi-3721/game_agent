"""Block Metropolis updates of C with reconstructed X, M and Z histories."""
import math
import time
import numpy as np
import torch
from .data import collate
from .training_data import PublicState
from .strategy import change_eligible,advance_changes
from .configuration import CONFIGURATION_BLOCKS


def log_acceptance(old_target,new_target,forward,reverse):
    return np.minimum(0.,new_target-old_target+reverse-forward)


def configuration_groups(model,weights):
    unique={};mapping=[]
    for team in model.configurations:
        key=tuple(model.vocab.configuration(team))
        mapping.append(unique.setdefault(key,len(unique)))
    groups=np.asarray(mapping)[model.c]
    return groups,np.bincount(groups,weights=weights,minlength=len(unique))


def configuration_support(model):
    _,mass=configuration_groups(model,model.weights)
    p=mass[mass>0];p/=p.sum()
    return dict(distinct=len(p),effective_sample_size=float(1/(p@p)),entropy=float(-(p*np.log(p)).sum()))


def resample_configurations(model,weights):
    """Preserve C mass exactly; sample X/M/Z conditionally within each C.

    One representative per supported C is a computation allocation, not a
    probability floor. Its particle weight remains that C's mass / its count.
    """
    from .filter import systematic
    groups,mass=configuration_groups(model,weights);active=np.flatnonzero(mass)
    counts=np.ones(len(active),dtype=int)
    if model.n>len(active):
        extra=systematic(mass[active],model.n-len(active),model.rng)
        counts+=np.bincount(extra,minlength=len(active))
    parents=[]
    for c,count in zip(active,counts):
        members=np.flatnonzero((groups==c)&(weights>0))
        parents.extend(members[systematic(weights[members]/mass[c],count,model.rng)])
    parents=np.array(parents);updated=np.repeat(mass[active]/counts,counts)
    after=np.bincount(groups[parents],weights=updated,minlength=len(mass))
    model.cost['configuration_resampling']=dict(configurations_before=len(active),
        configurations_after=int(np.count_nonzero(after)),
        max_probability_error=float(np.abs(after-mass).max()),discarded_probability=float(mass[after==0].sum()))
    return parents,updated


def refresh(model,resampled,deadline=float("inf"),work=None):
    if work is None:
        before=configuration_support(model)
        threshold=min(16.,max(2.,model.configuration_budget/4))
        low=before['effective_sample_size']<threshold
        triggered=bool(model.move_budget and (resampled or low))
        model.cost['configuration_support']=dict(before=before,threshold=threshold,triggered=triggered,
            reason='low_configuration_ess' if low else 'particle_resampling' if resampled else None)
        if not triggered:
            model.cost['configuration_support']['after']=before
            return None
        public=model._public_view();species=public.actors.get(public.active.get(model.side))
        focus=next((i for i,s in enumerate(model.preview) if model.vocab.token('species',s)==species),None) if low else None
    else:focus=None
    _,work=rejuvenate(model,model.move_budget,focus_slot=focus,deadline=deadline,work=work)
    model.cost['configuration_support']['after']=configuration_support(model)
    return work


@torch.inference_mode()
def replay_path(model,team,path,start_z,protected):
    from .filter import actor_row
    engine=model.engine;handle=engine.call('create',teams={model.observer:model.own_team,model.side:team})['handle']
    events=list(model.initial_events);cutoff=0;position=model.start_position
    public=PublicState(model.vocab,model.side,[model.vocab.token('species',s) for s in model.preview]);public.update(events)
    previous=np.zeros(4,np.int64);density=0.;runs=0;rows=[];selections=[];noises=[];steps=[]
    for record,step in zip(model.journal,path):
        view=engine.call('view',handle=handle,side=model.side,events=events)
        row,actions=actor_row(model.vocab,view,team,events[cutoff:],model.side,previous,position,public)
        selected=next((i for i,a in enumerate(actions) if a['key']==step['action'] or
                       (a['kind']=='team' and step['action'].startswith('team:') and
                        a['order'][0]==int(step['action'][5]))),None)
        if selected is None:
            if handle not in protected:engine.call('release',handles=[handle])
            return None,runs
        action=actions[selected];consumed=step['noise'] is not None
        if action['kind']=='team':density-=math.log(math.factorial(len(team)-1))
        if consumed:
            rows.append(row);selections.append(selected);noises.append(step['noise'])
        commands=[]
        if record['our_command'] is not None:commands.append([model.observer,record['our_command']])
        command=step['command'] if action['kind']=='team' else action.get('command')
        if command:commands.append([model.side,command])
        result=engine.call('transition',handle=handle,commands=commands,tape=step['random_tape'],
            events=record['private_events'] if record['private_events'] is not None else record['events'],
            private=record['private_events'] is not None,observer=model.observer)
        runs+=1
        if not result['matches']:
            if handle not in protected:engine.call('release',handles=[handle])
            return None,runs
        next_handle=result['handle']
        if handle!=next_handle and handle not in protected:engine.call('release',handles=[handle])
        handle=next_handle;density+=result['log_probability']
        steps.append({**step,'command':command})
        if consumed:
            previous=row['actions'][selected];cutoff=len(events);position+=1
        events.extend(record['events']);public.update(record['events'])
    # Causal attention permits one batched pass over the retained history.
    # Rule-inconsistent proposals are rejected before any neural recomputation.
    batch=collate([rows],model.device)
    h,memory=model.model.encode(batch['actor_events'],batch['actor_numeric'],batch['actor_state'],batch['configuration'],position=batch['position'])
    z=None if start_z is None else torch.tensor(start_z,device=model.device)[None];states=[]
    change_history=torch.zeros((1,5),device=model.device,dtype=torch.int8)
    for i,noise in enumerate(noises):
        eligible=change_eligible(batch['actor_state'][:,i],batch['action_mask'][:,i])
        z,_,mode,_,_=model.model.sample_strategy(h[:,i],z,noise=torch.tensor(noise,device=model.device)[None],eligible=eligible);states.append(z[0])
        change_history=advance_changes(change_history,mode,eligible)
    logits=model.model.policy_logits(h,batch['features'],torch.stack(states)[None],action_tokens=batch['actions']).masked_fill(~batch['action_mask'],-torch.inf)
    target=torch.tensor(selections,device=model.device)[None,:,None]
    logp=model.update_action_log_probs(logits.log_softmax(-1),batch['action_mask'])
    density+=float(logp.gather(-1,target).sum())
    return dict(handle=handle,log_density=density,memory=memory,previous=previous,z=z,change_history=change_history,path=steps),runs


@torch.inference_mode()
def rejuvenate(model,count=32,focus_slot=None,deadline=float("inf"),work=None):
    """Bounded block MH work; complete proposals commit atomically.

    The retained path density is accumulated by filtering. Only a proposed C
    requires historical replay; the old target is reused without approximation.
    """
    started=time.monotonic()
    if work is None:
        indices=model.rng.choice(np.flatnonzero(model.weights>0),min(count,np.count_nonzero(model.weights)),replace=False)
        blocks=[]
        for _ in indices:
            slot=int(model.rng.integers(len(model.preview)))
            if focus_slot is not None and model.rng.random()<.5:slot=focus_slot
            blocks.append((slot,CONFIGURATION_BLOCKS[int(model.rng.choice(6,p=[.1]*5+[.5]))]))
        work=dict(indices=indices,blocks=blocks,teams=[model.configurations[model.c[i]] for i in indices],cursor=0,
            proposed=None,priors=None,accepted=0,unchanged=0,illegal=0,history_rejections=0,mechanics_runs=0,seconds=0.,
            configurations_before=configuration_support(model)['distinct'])
    accepted_before=work['accepted']
    indices=work['indices'];teams=work['teams'];old_handles=set(map(int,model.handles));generated=set()
    proposed_now=False
    if work['proposed'] is None and time.monotonic()<deadline:
        proposal_start=time.monotonic()
        proposed,forward,reverse=model.sampler.propose_blocks(teams,work['blocks'],model.context,model.known)
        work.update(proposed=proposed,forward=forward,reverse=reverse)
        model.cost['configuration_proposal_seconds']=time.monotonic()-proposal_start
        proposed_now=True
    if work['proposed'] is not None and work['priors'] is None and time.monotonic()<deadline and (not proposed_now or deadline==float('inf')):
        prior_start=time.monotonic()
        work.update(priors=model.sampler.log_probability(teams+work['proposed'],model.prior_context),
            legality=model.engine.call('legal',teams=work['proposed']))
        model.cost['configuration_prior_seconds']=time.monotonic()-prior_start
    keys={tuple(model.vocab.configuration(t)):j for j,t in enumerate(model.configurations)}
    while work['priors'] is not None and work['cursor']<len(indices) and time.monotonic()<deadline:
        k=work['cursor'];work['cursor']+=1;i=indices[k];proposed=work['proposed'][k]
        key=tuple(model.vocab.configuration(proposed))
        if key==tuple(model.vocab.configuration(teams[k])):work['unchanged']+=1;continue
        if work['legality'][k]['errors']:work['illegal']+=1;continue
        protected=set(map(int,model.handles))
        new,runs=replay_path(model,proposed,model.paths[i],model.path_start_z[i],protected)
        work['mechanics_runs']+=runs
        if new is None:work['history_rejections']+=1;continue
        generated.add(new['handle'])
        ratio=log_acceptance(work['priors'][k]+model.path_log_density[i],
            work['priors'][len(indices)+k]+new['log_density'],work['forward'][k],work['reverse'][k])
        if math.log(model.rng.random())>=ratio:continue
        if key not in keys:keys[key]=len(model.configurations);model.configurations.append(proposed)
        model.c[i]=keys[key];model.handles[i]=new['handle'];model.previous[i]=new['previous'];model.paths[i]=new['path']
        model.path_log_density[i]=new['log_density']
        for stored,rebuilt in zip(model.memory,new['memory']):stored[i]=rebuilt[0]
        model.z[i]=new['z'][0];model.change_history[i]=new['change_history'][0]
        work['accepted']+=1
    unused=(old_handles|generated)-set(map(int,model.handles))
    if unused:model.engine.call('release',handles=list(unused))
    complete=work['cursor']==len(indices)
    if complete:
        unique={};configurations=[];remapped=[]
        for c in model.c:
            team=model.configurations[c];key=tuple(model.vocab.configuration(team))
            if key not in unique:unique[key]=len(configurations);configurations.append(team)
            remapped.append(unique[key])
        model.c=np.array(remapped);model.configurations=configurations
    if complete or work['accepted']!=accepted_before:model.cached=None
    work['seconds']+=time.monotonic()-started
    result={k:work[k] for k in ('accepted','unchanged','illegal','history_rejections','mechanics_runs','seconds','configurations_before')}
    result.update(proposals=len(indices),completed_proposals=work['cursor'],complete=complete,
        metropolis_rejections=work['cursor']-sum(work[k] for k in ('accepted','unchanged','illegal','history_rejections')),
        configurations_after=configuration_support(model)['distinct'],old_path_replays=0)
    model.cost['configuration_move']=result
    return result,None if complete else work
