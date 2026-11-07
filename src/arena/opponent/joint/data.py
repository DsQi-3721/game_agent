"""Chronological actor/inference views. Truth is attached only as loss targets."""
import gzip,orjson
import numpy as np
import torch

FAMILIES=('rule','search','learned','random','adaptive','heldout')

def games(path):
    with gzip.open(path,'rt') as handle:
        for line in handle:yield orjson.loads(line)

def teacher_distribution(row):
    # Forced continuation requests can omit PP; the old held-out provider then
    # logged null probabilities. A single legal action has exactly unit mass.
    p=np.asarray(row['distribution'],np.float32)
    if len(p)==1:return np.ones(1,np.float32),True
    valid=bool(np.isfinite(p).all() and abs(float(p.sum())-1)<1e-5)
    return (p if valid else np.zeros_like(p)),valid

def action_tokens(vocab,action,team,active_slot=1):
    if action['kind']=='move':return [vocab.token('move',action['move']),vocab.token('type',team[active_slot-1].get('teraType','Normal')) if action.get('tera') else 0,0,0]
    if action['kind']=='switch':return [vocab.token('species',team[action['slot']-1]['species']),0,0,0]
    return [vocab.token('species',team[action['order'][0]-1]['species']),0,0,0]

def collate(chunks,device='cpu'):
    """Batch the actor-view inputs used by online inference."""
    b=len(chunks);t=max(map(len,chunks));e=max(len(r['actor_events']) for c in chunks for r in c);a=max(len(r['features']) for c in chunks for r in c)
    shapes={'actor_events':(b,t,e,4),'actor_numeric':(b,t,e,8),'actor_state':(b,t,32),
        'configuration':(b,t,126),'features':(b,t,a,32),'actions':(b,t,a,4),'position':(b,t),'action_mask':(b,t,a)}
    integers={'actor_events','configuration','actions','position'}
    batch={k:np.zeros(shape,dtype=np.int64 if k in integers else bool if k=='action_mask' else np.float32) for k,shape in shapes.items()}
    for i,chunk in enumerate(chunks):
        for j,row in enumerate(chunk):
            for key in ('actor_events','actor_numeric','features','actions'):batch[key][i,j,:len(row[key])]=row[key]
            for key in ('actor_state','configuration','position'):batch[key][i,j]=row[key]
            batch['action_mask'][i,j,:len(row['features'])]=True
        batch['action_mask'][i,len(chunk):,0]=True
    return {k:torch.as_tensor(v,device=device) for k,v in batch.items()}
