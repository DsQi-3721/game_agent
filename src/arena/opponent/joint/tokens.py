"""Typed event/configuration tokens; observation channels stay explicit."""
import json
import re
from pathlib import Path
import numpy as np

STATS=('hp','atk','def','spa','spd','spe')
def ident(value):return re.sub('[^a-z0-9]','',str(value or '').lower())

class Vocabulary:
    def __init__(self,catalog):
        self.catalog=json.loads(Path(catalog).read_text()) if isinstance(catalog,(str,Path)) else catalog
        self.tokens=list(self.catalog['tokens'])
        extra=[*(f'stat:{s}' for s in STATS),*(f'status:{s}' for s in ('brn','par','slp','frz','psn','tox','none')),
               *(f'effect:{s}' for s in ('raindance','sunnyday','sandstorm','hail','snow','snowscape','primordialsea','desolateland','deltastream','none','trickroom','wonderroom','magicroom','electricterrain','grassyterrain','mistyterrain','psychicterrain','typechange','typeadd','substitute','confusion','leechseed','perish0','perish1','perish2','perish3','stockpile1','stockpile2','stockpile3','quarkdriveatk','quarkdrivedef','quarkdrivespa','quarkdrivespd','quarkdrivespe','protosynthesisatk','protosynthesisdef','protosynthesisspa','protosynthesisspd','protosynthesisspe')),
               *(f'query:{s}' for s in ('item','ability','tera','move')),
               *(f'action:{s}' for s in ('move','switch','unknown','tera')),
               'event:source','event:start','event:teampreview','event:replace','event:upkeep']
        self.tokens=list(dict.fromkeys(self.tokens+extra));self.ids={v:i for i,v in enumerate(self.tokens)}
        self.move_ids=[i for i,t in enumerate(self.tokens) if t.startswith('move:')]
        self.move_index={i:j for j,i in enumerate(self.move_ids)}
        self.action_kind_ids=[self.ids['action:'+s] for s in ('move','switch','unknown','tera')]
        self.species={ident(k):v for k,v in self.catalog['bySpecies'].items()}
        self.config_fields=('species','item','ability','nature','type',*('ev',)*6,*('iv',)*6,*('move',)*4)
    def token(self,kind,value):
        value=ident(value)
        if kind=='species' and f'species:{value}' not in self.ids:
            for base in ('flabebe','floette','florges','deerling','sawsbuck','vivillon'):
                if value.startswith(base):value=base;break
        return self.ids.get(f'{kind}:{value}',1)
    def configuration(self,team):
        values=[]
        for mon in team:
            values.extend([self.token('species',mon['species']),self.token('item',mon.get('item') or 'none'),
                self.token('ability',mon['ability']),self.token('nature',mon.get('nature','serious')),self.token('type',mon.get('teraType','Normal'))])
            values.extend(self.ids[f'ev:{mon.get("evs",{}).get(s,0)}'] for s in STATS)
            values.extend(self.ids[f'iv:{mon.get("ivs",{}).get(s,31)}'] for s in STATS)
            values.extend([self.token('move',m) for m in mon['moves']]+[0]*(4-len(mon['moves'])))
        return values
    def decode_configuration(self,tokens):
        result=[];width=len(self.config_fields)
        for start in range(0,len(tokens),width):
            values=[self.tokens[int(t)].split(':',1)[-1] for t in tokens[start:start+width]]
            if len(values)!=width:break
            result.append({'species':values[0],'name':values[0],'item':'' if values[1]=='none' else values[1],
                'ability':values[2],'nature':values[3],'teraType':values[4],'level':100,
                'evs':dict(zip(STATS,map(int,values[5:11]))),'ivs':dict(zip(STATS,map(int,values[11:17]))),'moves':[m for m in values[17:21] if m!='<pad>']})
        return result
    def events(self,lines,side,actors=None):
        tokens=[];numeric=[]
        actors={} if actors is None else actors
        for line in lines:
            p=line.split('|')
            if len(p)<2 or 'event:'+p[1] not in self.ids:continue
            kind=p[1];row=[self.ids['event:'+kind],0,0,0];n=[0.]*8
            actor=p[2] if len(p)>2 else '';n[0]=1. if actor.startswith(side) else -1. if actor.startswith('p') else 0.
            if kind=='turn':n[1]=min(float(actor),200)/200
            value=p[3] if len(p)>3 else ''
            if kind=='poke':row[1]=self.token('species',value.split(',')[0])
            if kind in {'switch','drag','replace','detailschange','-formechange'}:
                row[1]=self.token('species',value.split(',')[0]);actors[actor]=row[1]
            row[2]=actors.get(actor,0)
            if kind=='move':row[1]=self.token('move',value)
            if kind in {'-item','-enditem'}:row[1]=self.token('item',value or 'none')
            if kind=='-ability':row[1]=self.token('ability',value)
            if kind=='-terastallize':row[1]=self.token('type',value)
            if kind in {'-boost','-unboost','-setboost'}:row[1]=self.token('stat',value)
            if kind in {'-status','-curestatus','cant'}:row[1]=self.token('status',value)
            if kind=='-weather':row[1]=self.token('effect',actor)
            if kind in {'-fieldstart','-fieldend','-sidestart','-sideend','-start','-end','-activate','-singleturn','-singlemove','-prepare'}:
                effect=value if kind not in {'-fieldstart','-fieldend'} else actor
                prefix,_,name=effect.partition(':')
                row[1]=self.token(prefix.strip(),name) if prefix in {'move','item','ability'} else self.token('effect',effect)
                if row[1]==1:row[1]=self.token('move',effect)
            if kind=='move' and len(p)>4:row[3]=actors.get(p[4],0)
            condition=p[4] if kind in {'switch','drag'} and len(p)>4 else value
            if '/' in condition:
                match=re.match(r'(\d+)/(\d+)',condition)
                if match:n[2]=int(match[1])/int(match[2])
            if 'fnt' in condition:n[3]=1.
            if kind in {'-boost','-unboost'} and len(p)>4:n[4]=float(p[4])/6*(1 if kind=='-boost' else -1)
            if kind in {'-status','-curestatus','cant'}:
                for j,status in enumerate(('brn','par','slp','frz','psn','tox')):
                    if status in value:n[5]=(j+1)/6
            n[6]=float(kind=='-crit');n[7]=float(kind=='-miss')
            tokens.append(row);numeric.append(n)
            for source in p[4:]:
                if not source.startswith('[from] '):continue
                kind_name=source[7:];category,_,name=kind_name.partition(':')
                source_id=self.token(category.strip(),name) if category in {'item','move','ability'} else self.token('effect',kind_name)
                holder=next((s[5:] for s in p[4:] if s.startswith('[of] ')),actor)
                tokens.append([self.ids['event:source'],source_id,actors.get(holder,0),row[2]]);numeric.append([1. if holder.startswith(side) else -1.,0.,0.,0.,0.,0.,0.,0.])
        return np.array(tokens,dtype=np.int64).reshape(-1,4),np.array(numeric,dtype=np.float32).reshape(-1,8)
