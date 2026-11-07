"""Readable opponent dossier; complete distributions remain available via query()."""
from collections import defaultdict


def marginal(rows,limit=3,field=None):
    ranked=sorted(rows,key=lambda r:-r['probability'])
    candidates=[]
    for row in ranked[:limit]:
        value=row['value']
        if value=='' and field in ('item','ability','status'):value='healthy' if field=='status' else 'none'
        candidates.append(dict(value=value,probability=row['probability']))
    return dict(candidates=candidates,other_probability=sum(r['probability'] for r in ranked[limit:]))


def dossier(model):
    belief=model.query();opponent=belief['opponent'];cost=belief['inference_cost']
    valid=belief['prediction_valid'] and cost.get('observation_applied',True)
    team=[];names={'teraType':'tera_type','tera':'terastallized_type'}
    for initial,current,known in zip(opponent['initial_configuration'],opponent['current_state'],model.known):
        hidden={}
        if valid:
            for field in ('item','ability','nature','teraType','evs','ivs'):
                if field not in known:hidden[names.get(field,field)]=marginal(initial[field],field=field)
            moves=[r for r in initial['moves'] if r['value'] not in known.get('moves',[])]
            hidden['moves']=dict(inclusion_probabilities={r['value']:r['probability'] for r in moves[:8]},
                omitted_expected_count=sum(r['probability'] for r in moves[8:]))
        team.append(dict(slot=initial['slot'],species=initial['species'],confirmed={names.get(k,k):v for k,v in known.items()},
            hidden_configuration=hidden,current_state={names.get(k,k):marginal(v,field=k) for k,v in current.items() if k!='slot'} if valid else None))
    actions=[];leads=defaultdict(float)
    if valid:
        for row in belief['prediction']['actions']:
            key=row['action'].split(':');probability=row['probability']
            if key[0]=='team':leads[int(key[1][0])]+=probability
            elif key[0]=='move':actions.append(dict(type='move',move=key[1],terastallize=key[2]=='tera=true',probability=probability))
            elif key[0]=='switch':
                slot=int(key[1]);actions.append(dict(type='switch',slot=slot,species=model.preview[slot-1],probability=probability))
            else:actions.append(dict(type=key[0],probability=probability))
    prediction=dict(actions=actions)
    if leads:
        prediction=dict(leads=[dict(slot=s,species=model.preview[s-1],probability=p) for s,p in sorted(leads.items(),key=lambda x:-x[1])],
            remaining_order='uniform over permutations of the remaining five slots')
    turn=next((int(e.split('|')[2]) for e in reversed(model.events) if e.startswith('|turn|')),0)
    knowledge=[dict(slot=m['slot'],species=m['species'],revealed={names.get(k,k):v for k,v in m['revealed'].items()}) for m in opponent['information']['observed_about_us']]
    result=dict(format=belief['format'],opponent_id=opponent['id'],game_id=model.game_id,
        turn=turn,games_observed=opponent['games_observed'],updated_at=belief['updated_at'],
        prediction_valid=bool(valid),team=team,opponent_knowledge=knowledge,
        behavior_profile=opponent['strategy']['behavior_profile'] if valid else None,
        current_action_tendencies=opponent['strategy']['current_action_tendencies'] if valid else None,
        strategy_change=opponent['strategy']['recent_change'] if valid else None,
        prediction=prediction,action_entropy=belief['uncertainty']['action_entropy'] if valid else None)
    if belief['pending_update']:result['pending_update']=belief['pending_update']['phase']
    if not valid:result['unavailable_reason']='observation update pending' if belief['pending_update'] else (belief['model_mismatch'] or {}).get('diagnosis') or cost.get('reason')
    return rounded(result)


def rounded(value):
    if isinstance(value,float):return round(value,6)
    if isinstance(value,dict):return {k:rounded(v) for k,v in value.items()}
    if isinstance(value,list):return [rounded(v) for v in value]
    return value
