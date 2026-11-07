"""Conditional action priors and submitted-action posteriors for one C/X/M/Z hypothesis."""
import time
import numpy as np
import torch
from .data import action_tokens, collate
from .training_data import PublicState


def actor_row(vocab,view,team,events,side,previous,position,public,*,encoded_events=None,configuration=None):
    e,num=vocab.events(events,side,public.actors.copy()) if encoded_events is None else encoded_events
    if not len(e):e=np.zeros((1,4),np.int64);num=np.zeros((1,8),np.float32)
    request=view['request'];phase=0 if request.get('teamPreview') else 2 if request.get('forceSwitch') else 1
    state=public.state(phase)
    public_state=state.copy();mons=request.get('side',{}).get('pokemon',[]);active=next((m for m in mons if m['active']),None)
    def hp(m):
        parts=m['condition'].split()[0].split('/');return float(parts[0])/float(parts[1]) if len(parts)==2 else 0.
    if active:
        state[3]=hp(active)
        for k,m in enumerate((request.get('active') or [{}])[0].get('moves',[])[:4]):state[4+k]=m.get('pp',0)/64
        for k,m in enumerate(mons):state[8+k]=hp(m)
        for k,s in enumerate(('atk','def','spa','spd','spe')):state[20+k]=active.get('stats',{}).get(s,0)/700
    actions=view['actions'];features=np.asarray(view['features'],np.float32).copy()
    repeat=public.last.get(public.active.get(side))
    for k,action in enumerate(actions):features[k,27]=float(action.get('move')==repeat) if action['kind']=='move' else 0.
    if request.get('teamPreview'):
        indices=[next(i for i,a in enumerate(actions) if a['order'][0]==lead) for lead in range(1,7)];actions=[actions[i] for i in indices];features=features[indices]
    if not actions:actions=[{'key':'waiting','kind':'waiting'}];features=np.zeros((1,32),np.float32)
    slot=next((view['slots'][i] for i,m in enumerate(mons) if m['active']),1)
    tokens=np.array([action_tokens(vocab,a,team,slot) if a['kind']!='waiting' else [0]*4 for a in actions])
    row={'events':e,'numeric':num,'actor_events':np.concatenate([e,[[4,*previous[:3]]]]),
         'actor_numeric':np.concatenate([num,[[0.,0,0,0,0,0,0,1.]]]),'state':public_state,'actor_state':state,
         'configuration':np.array(vocab.configuration(team)) if configuration is None else configuration,
         'features':features,'actions':tokens,'position':position}
    return row,actions


@torch.inference_mode()
def action_prior(model, vocab, view, team, history, side, *, memory=None,
                 strategy=None, previous=(0, 0, 0, 0), position=0, event_start=0):
    """Score actions using only this player's hypothesized information state."""
    device=next(model.parameters()).device
    public=PublicState(vocab,side,[vocab.token('species',m['species']) for m in team])
    public.update(history)
    row,actions=actor_row(vocab,view,team,history[event_start:],side,previous,position,public)
    batch=collate([[row]],device)
    h,memory=model.encode(batch['actor_events'],batch['actor_numeric'],batch['actor_state'],
                         batch['configuration'],memory=memory,position=batch['position'])
    z=model.strategy_distribution(h[:,0])[0] if strategy is None else torch.as_tensor(strategy,device=device).reshape(1,-1)
    logits=model.policy_logits(h,batch['features'],z[:,None],action_tokens=batch['actions'])
    probabilities=logits[0,0].double().softmax(-1).cpu().numpy()
    if view['request'].get('teamPreview'):
        # The trained policy scores leads; all unmodeled reserve orders share
        # their lead's probability uniformly, while remaining distinct actions.
        by_lead={a['order'][0]:p for a,p in zip(actions,probabilities)}
        actions=view['actions']
        counts={lead:sum(a['order'][0]==lead for a in actions) for lead in by_lead}
        probabilities=np.array([by_lead[a['order'][0]]/counts[a['order'][0]] for a in actions])
    return dict(actions=actions,probabilities=probabilities,memory=memory,strategy=z)


def submission_commands(side, actions, our_command):
    observer='p2' if side=='p1' else 'p1'
    commands=[]
    for action in actions:
        pair=[] if our_command is None else [[observer,our_command]]
        if action.get('command'):pair.append([side,action['command']])
        commands.append(pair)
    return commands


def submission_likelihoods(engine, handle, side, actions, events, *, our_command=None,
                           private_events=None, branches=32, keep=4, seed=83):
    """Mechanics depends on X and the submitted action, not on M or Z."""
    observer='p2' if side=='p1' else 'p1'
    commands=submission_commands(side,actions,our_command)
    return engine.call('transitions',handle=handle,commands=commands,
                        events=events if private_events is None else private_events,
                        private=private_events is not None,observer=observer,
                        branches=branches,keep=keep,seed=seed)


def infer_submission(engine, handle, side, prior, events, *, our_command=None,
                     private_events=None, branches=32, keep=4, seed=83):
    """Marginalize submitted actions, preserving their joint next-state mass."""
    started=time.perf_counter();actions=prior['actions']
    p=np.asarray(prior['probabilities'],dtype=np.float64);p=p/p.sum()
    results=submission_likelihoods(engine,handle,side,actions,events,our_command=our_command,
        private_events=private_events,branches=branches,keep=keep,seed=seed)
    likelihood=np.array([r['likelihood'] for r in results]);mass=p*likelihood
    evidence=float(mass.sum());posterior=mass/evidence if evidence>0 else np.zeros_like(mass)
    outcomes=[]
    if evidence>0:
        for action,probability,result in zip(actions,p,results):
            for outcome in result['outcomes']:
                outcomes.append(dict(action=action['key'],command=action.get('command'),
                    handle=outcome['handle'],probability=float(probability*outcome['weight']/evidence),
                    random_tape=outcome['random_tape'],random_log_probability=outcome['random_log_probability']))
    return dict(observation_probability=evidence,observation_supported=evidence>0,
        likelihood_is_exact=all(r['likelihood_is_exact'] for r in results),
        actions=[dict(action=a['key'],prior_probability=float(pi),observation_likelihood=float(li),
                      posterior_probability=float(po)) for a,pi,li,po in zip(actions,p,likelihood,posterior)],
        outcomes=outcomes,inference_cost=dict(seconds=time.perf_counter()-started,
            actions=len(actions),mechanics_runs=sum(r['attempts'] for r in results)))

