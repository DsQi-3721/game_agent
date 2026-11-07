"""Compact player-game records and chronological support/query training episodes."""
from collections import OrderedDict
from pathlib import Path
import pickle
import re
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from .tokens import ident, STATS
from .human_data import windows, executed_actions, revealed_facts
from .data import action_tokens, teacher_distribution


class PublicState:
    def __init__(self, vocab, side, preview):
        self.vocab, self.side, self.preview = vocab, side, preview
        self.actors, self.active, self.hp, self.last = {}, {}, {}, {}
        self.turn = 0

    def update(self, lines):
        for line in lines:
            p = line.split('|')
            if len(p) < 3: continue
            kind, actor = p[1:3]
            side = actor[:2]
            if kind == 'turn': self.turn = int(actor)
            if side not in ('p1', 'p2'): continue
            if kind in ('switch', 'drag', 'replace', 'detailschange', '-formechange'):
                self.actors[actor] = self.vocab.token('species', p[3].split(',')[0])
                self.active[side] = actor
            if kind in ('switch', 'drag', '-damage', '-heal'):
                value = p[4] if kind in ('switch', 'drag') else p[3]
                match = re.match(r'(\d+)/(\d+)', value)
                if match: self.hp[actor] = int(match[1])/int(match[2])
            if kind == 'faint': self.hp[actor] = 0.
            if kind == 'move' and not any(x.startswith('[from]') for x in p[4:]): self.last[actor] = ident(p[3])

    def state(self, phase):
        x = np.zeros(32, np.float32)
        x[0] = min(self.turn, 200)/200
        x[1:3] = [phase == 0, phase == 2]
        x[3] = self.hp.get(self.active.get(self.side), 1.)
        foe = 'p2' if self.side == 'p1' else 'p1'
        x[4] = self.hp.get(self.active.get(foe), 1.)
        return x

    def pair(self):
        foe = 'p2' if self.side == 'p1' else 'p1'
        return [self.actors.get(self.active.get(s), 0) for s in (self.side, foe)]


def pack_steps(rows, config=None):
    result = {}
    for name in ('events', 'public_events'):
        arrays = [r[name] for r in rows]
        result[name] = np.concatenate(arrays).astype(np.int16)
        result[name+'_offsets'] = np.cumsum([0]+[len(x) for x in arrays], dtype=np.int32)
    for name in ('numeric', 'public_numeric'):
        result[name] = np.concatenate([r[name] for r in rows]).astype(np.float32)
    for name in ('state', 'public_state', 'phase', 'target', 'change', 'active', 'preview'):
        result[name] = np.asarray([r[name] for r in rows])
    result['facts'] = [(t, *fact) for t,r in enumerate(rows) for fact in r['facts']]
    result['configuration'] = config
    if config is not None:
        for name in ('features', 'actions', 'teacher', 'teacher_valid', 'config_unknown'):
            result[name] = [r[name] for r in rows]
    return result


def human_records(game, vocab):
    all_facts = revealed_facts(game)
    for side_index, side in enumerate(('p1', 'p2')):
        preview = [vocab.token('species', s) for s in game['preview'][side_index]]
        view = PublicState(vocab, side, preview)
        rows, cutoff = [], 0
        for window in windows(game):
            increment = game['public'][cutoff:window['cutoff']]
            view.update(increment)
            e,n = vocab.events(increment, side, view.actors)
            if not len(e): e,n = np.zeros((1,4),np.int16),np.zeros((1,8),np.float32)
            actions, reasons, tera = executed_actions(game, window)
            action = actions[side_index]
            y = 2*len(vocab.move_ids)+6
            if action:
                if action['kind'] == 'move':
                    token = vocab.token('move', action['value'])
                    if token in vocab.move_index: y = 2*vocab.move_index[token]+int(tera[side_index] is not None)
                else:
                    species = vocab.token('species', action['value'])
                    if species in preview: y = 2*len(vocab.move_ids)+preview.index(species)
            phase = 0 if window['phase'] == 'preview' else 1
            facts = []
            for fact in all_facts:
                if fact['side'] != side or not window['cutoff'] <= fact['event_index'] < window['end']: continue
                if fact['field'] == 'item' and fact['scope'] == 'initial': field,kind='item','item'
                elif fact['field'] == 'tera_type': field,kind='tera','type'
                else: continue
                # Query identities come from the public team preview. Newly
                # switched combatants can be resolved in the outcome, not input.
                subject = view.actors.get(fact['subject'])
                if subject is None:
                    for line in game['public'][window['cutoff']:fact['event_index']+1]:
                        p=line.split('|')
                        if len(p)>3 and p[1] in ('switch','drag','replace') and p[2]==fact['subject']:
                            subject=vocab.token('species',p[3].split(',')[0])
                if subject in preview:
                    facts.append((subject,vocab.ids['query:'+field],vocab.token(kind,fact['value'])))
            state = view.state(phase)
            rows.append(dict(events=e,numeric=n,public_events=e,public_numeric=n,state=state,public_state=state,
                             phase=phase,target=y,change=-100,active=view.pair(),preview=preview,facts=facts))
            cutoff=window['cutoff']
        increment=game['public'][cutoff:];view.update(increment)
        e,n=vocab.events(increment,side,view.actors)
        if len(e):
            row=dict(rows[-1]);row.update(events=e,numeric=n,public_events=e,public_numeric=n,state=view.state(3),public_state=view.state(3),phase=3,target=-100,change=-100,facts=[])
            rows.append(row)
        yield side, pack_steps(rows)


def controlled_records(game, vocab):
    for si,side in enumerate(('p1','p2')):
        team=game['teams'][side]
        config=np.asarray(vocab.configuration(team),np.int64)
        preview=[vocab.token('species',m['species']) for m in team]
        view=PublicState(vocab,side,preview)
        rows=[];cutoff=0;previous=[0,0,0,0];last_family=None;lag_change=0
        known=np.zeros(126,bool);known[::21]=True
        for d in game['decisions']:
            if d['side']!=side:continue
            increment=game['public'][cutoff:d['cutoff']]
            view.update(increment);e,n=vocab.events(increment,side,view.actors)
            if not len(e):e,n=np.zeros((1,4),np.int16),np.zeros((1,8),np.float32)
            for line in increment:
                p=line.split('|')
                if len(p)<4 or not p[2].startswith(side):continue
                species=view.actors.get(p[2]);slot=preview.index(species) if species in preview else -1
                if slot<0:continue
                field=None;value=None
                if p[1] in ('-item','-enditem') and not any(x.startswith('[from]') for x in p[4:]):field=1;value=vocab.token('item',p[3])
                if p[1]=='-ability' and not any(x.startswith('[from]') for x in p[4:]):field=2;value=vocab.token('ability',p[3])
                if p[1]=='-terastallize':field=4;value=vocab.token('type',p[3])
                if field is not None and config[slot*21+field]==value:known[slot*21+field]=True
                if p[1]=='move' and not any(x.startswith('[from]') for x in p[4:]):
                    value=vocab.token('move',p[3])
                    for k in range(17,21):
                        if config[slot*21+k]==value:known[slot*21+k]=True
            request=d['request'];phase=0 if request.get('teamPreview') else 2 if request.get('forceSwitch') else 1
            state=view.state(phase);public_state=state.copy()
            active=next((m for m in request['side']['pokemon'] if m['active']),None)
            if active:
                hp=active['condition'].split()[0].split('/');state[3]=float(hp[0])/float(hp[1]) if len(hp)==2 else 0
                for k,move in enumerate((request.get('active') or [{}])[0].get('moves',[])[:4]):state[4+k]=move.get('pp',0)/64
                for k,mon in enumerate(request['side']['pokemon']):
                    hp=mon['condition'].split()[0].split('/');state[8+k]=float(hp[0])/float(hp[1]) if len(hp)==2 else 0
                for k,stat in enumerate(('atk','def','spa','spd','spe')):state[20+k]=active.get('stats',{}).get(stat,0)/700
            actions=d['actions'];feat=np.asarray(d['features'],np.float32).copy();target=d['target'];teacher,teacher_valid=teacher_distribution(d)
            repeat=view.last.get(view.active.get(side))
            for k,action in enumerate(actions):feat[k,27]=float(action.get('move')==repeat) if action['kind']=='move' else 0
            if phase==0:
                lead=actions[target]['order'][0];ix=[next(k for k,a in enumerate(actions) if a['order'][0]==j) for j in range(1,7)]
                teacher=np.asarray([sum(prob for prob,a in zip(teacher,actions) if a['order'][0]==j) for j in range(1,7)],np.float32)
                actions=[actions[k] for k in ix];feat=feat[ix];target=lead-1
            slot=next((d['slots'][k] for k,m in enumerate(request['side']['pokemon']) if m['active']),1)
            action_ids=np.asarray([action_tokens(vocab,a,team,slot) for a in actions],np.int64)
            ae=np.concatenate([e,np.asarray([[4,*previous[:3]]])]);an=np.concatenate([n,np.asarray([[0,0,0,0,0,0,0,1]],np.float32)])
            family=d['truth']['family'];current_change=1 if last_family is not None and family!=last_family else 2 if d['truth']['change']=='gradual' else 0
            rows.append(dict(events=ae,numeric=an,public_events=e,public_numeric=n,state=state,public_state=public_state,
                             phase=phase,target=target,change=lag_change,active=view.pair(),preview=preview,facts=[],
                             features=feat,actions=action_ids,teacher=teacher,teacher_valid=teacher_valid,config_unknown=~known.copy()))
            cutoff=d['cutoff'];previous=action_ids[target].tolist();last_family=family;lag_change=current_change
        increment=game['public'][cutoff:];view.update(increment)
        e,n=vocab.events(increment,side,view.actors)
        if len(e):
            row=dict(rows[-1]);row.update(events=e,numeric=n,public_events=e,public_numeric=n,phase=3,target=-100,change=-100,facts=[],teacher_valid=False,config_unknown=np.zeros(126,bool))
            rows.append(row)
        yield side,pack_steps(rows,config)


class EpisodeDataset(Dataset):
    def __init__(self, directory, source, split, limit=0, training=False, length=64, support=32):
        self.directory=Path(directory);self.training=training;self.length=length;self.support=support
        index=pq.read_table(self.directory/'index.parquet').to_pandas()
        self.locations={(r.battle_id,r.side):r for r in index.itertuples()}
        selected=index[(index.source==source)&(index.split==split)]
        if limit and len(selected)>limit:selected=selected.sample(limit,random_state=2027)
        self.rows=list(selected.itertuples());self.files={};self.cache=OrderedDict()

    def __len__(self):return len(self.rows)

    def read(self,row):
        path=row.file
        if path not in self.files:self.files[path]=pq.ParquetFile(self.directory/path)
        group,offset=divmod(row.row,32);key=(path,group)
        if key not in self.cache:
            self.cache[key]=self.files[path].read_row_group(group,columns=['payload'])['payload'].to_pylist()
            if len(self.cache)>8:self.cache.popitem(last=False)
        else:self.cache.move_to_end(key)
        return pickle.loads(self.cache[key][offset])

    def __getitem__(self,index):
        row=self.rows[index];query=self.read(row)
        n=len(query['target']);start=np.random.randint(max(1,n-self.length+1)) if self.training and n>self.length else 0
        parts=[]
        if row.previous_battle_id is not None and (row.previous_battle_id,row.previous_side) in self.locations:
            past=self.read(self.locations[(row.previous_battle_id,row.previous_side)])
            parts.append((past,max(0,len(past['target'])-self.support),len(past['target']),False,0))
        if start:parts.append((query,max(0,start-32),start,False,1))
        parts.append((query,start,min(n,start+self.length),True,1))
        return {'parts':parts,'source':row.source,'battle_id':row.battle_id,'player_id':row.player_id,'side':row.side,'has_history':len(parts)>1 and parts[0][0] is not query}


def collate_episodes(samples):
    controlled=samples[0]['source']=='controlled'
    expanded=[]
    for sample in samples:
        rows=[]
        for data,start,end,score,segment in sample['parts']:
            for t in range(start,end):
                row={k:data[k][t] for k in ('state','public_state','phase','target','change','active','preview')}
                for name,num in [('events','numeric'),('public_events','public_numeric')]:
                    a,z=data[name+'_offsets'][t:t+2];row[name]=data[name][a:z];row[num]=data[num][a:z]
                row.update(score=score,segment=segment,position=t,game_start=t==start and (not rows or rows[-1]['segment']!=segment),facts=[f[1:] for f in data['facts'] if f[0]==t])
                if controlled:
                    for k in ('features','actions','teacher','teacher_valid','config_unknown'):row[k]=data[k][t]
                    row['configuration']=data['configuration']
                rows.append(row)
        expanded.append(rows)
    b=len(samples);t=max(map(len,expanded));e=max(len(r['events']) for rows in expanded for r in rows);ep=max(len(r['public_events']) for rows in expanded for r in rows)
    shapes={'events':(b,t,e,4),'numeric':(b,t,e,8),'public_events':(b,t,ep,4),'public_numeric':(b,t,ep,8),
            'state':(b,t,32),'public_state':(b,t,32),'phase':(b,t),'target':(b,t),'change':(b,t),'preview':(b,t,6),'active':(b,t,2),
            'valid':(b,t),'score':(b,t),'segments':(b,t),'position':(b,t),'game_start':(b,t)}
    integers={'events','public_events','phase','target','change','preview','active','segments','position','actions','configuration'}
    boolean={'valid','score','game_start','action_mask','teacher_valid','config_unknown'}
    if controlled:
        a=max(len(r['actions']) for rows in expanded for r in rows)
        shapes.update(actions=(b,t,a,4),features=(b,t,a,32),teacher=(b,t,a),action_mask=(b,t,a),teacher_valid=(b,t),configuration=(b,t,126),config_unknown=(b,t,126))
    batch={k:np.zeros(shape,dtype=np.int64 if k in integers else bool if k in boolean else np.float32) for k,shape in shapes.items()}
    batch['target'].fill(-100);batch['change'].fill(-100);batch['segments'].fill(-1)
    facts=[]
    for i,rows in enumerate(expanded):
        for j,row in enumerate(rows):
            for name in ('events','numeric','public_events','public_numeric'):batch[name][i,j,:len(row[name])]=row[name]
            for name in ('state','public_state','phase','target','change','preview','active','score','position','game_start'):batch[name][i,j]=row[name]
            batch['segments'][i,j]=row['segment'];batch['valid'][i,j]=True
            if controlled:
                for name in ('features','actions','teacher'):batch[name][i,j,:len(row[name])]=row[name]
                for name in ('teacher_valid','configuration','config_unknown'):batch[name][i,j]=row[name]
                batch['action_mask'][i,j,:len(row['actions'])]=True
            if row['score']:
                facts.extend((i,j,*fact) for fact in row['facts'])
        if controlled:batch['action_mask'][i,len(rows):,0]=True
    out={k:torch.from_numpy(v) for k,v in batch.items()}
    out['facts']=torch.tensor(facts,dtype=torch.long).reshape(-1,5)
    out['channel']=0 if controlled else 1;out['source']=samples[0]['source'];out['games']=[s['battle_id'] for s in samples]
    out['has_history']=torch.tensor([s['has_history'] for s in samples])
    return out
