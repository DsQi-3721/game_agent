"""Rule-masked autoregressive C proposals, with explicit prior/proposal scores."""
import copy,time
import numpy as np
import torch
from .tokens import STATS,ident

CONFIGURATION_BLOCKS=((1,),(2,),(4,),(17,18,19,20),(3,*range(5,17)),tuple(range(1,21)))

def importance_logweights(log_prior,log_proposal,log_mass):
    return log_mass+log_prior-log_proposal

class ConfigurationSampler:
    def __init__(self,model,vocab,engine,generator=None):
        self.model=model;self.vocab=vocab;self.engine=engine;self.options={};self.device=next(model.parameters()).device;self.generator=generator
    def rules(self,species):
        key=ident(species)
        if key not in self.options:self.options[key]=self.engine.call('options',species=species)
        return self.options[key]
    def allowed(self,preview,prefix,position,facts=()):
        slot,field=divmod(position,21);rules=self.rules(preview[slot]);v=self.vocab;part=prefix[slot*21:];known=facts[slot] if slot<len(facts) else {}
        if field==0:return [v.token('species',preview[slot])]
        kind=v.config_fields[field]
        if field==1:values=[v.token('item',x) for x in [*rules['items'],'none']]
        elif field==2:values=[v.token('ability',x) for x in rules['abilities']]
        elif field==3:values=[v.token('nature',x) for x in rules['natures']]
        elif field==4:values=[v.token('type',x) for x in rules['types']]
        elif field<11:
            used=sum(int(v.tokens[t].split(':')[1]) for t in part[5:]);values=[v.ids[f'ev:{n}'] for n in range(min(252,510-used)+1)]
        elif field<17:values=[v.ids[f'iv:{n}'] for n in range(32)]
        else:
            used=set(part[17:]);values=[v.token('move',x) for x in rules['moves'] if v.token('move',x) not in used]
            missing={v.token('move',m) for m in known.get('moves',[])}-used
            if field>17:
                if 0 in used:values=[0]
                elif not missing:values=[*values,0]
            if len(missing)>=21-field:values=list(missing)
        fixed={1:'item',2:'ability',3:'nature',4:'teraType'}.get(field)
        if fixed and fixed in known:values=[v.token(kind,known[fixed] or 'none')]
        return sorted(set(t for t in values if t!=1))
    @torch.inference_mode()
    def log_probability(self,teams,context,proposal=False,facts=(),per_token=False):
        unique={};indices=[];selected=[]
        for team in teams:
            key=tuple(self.vocab.configuration(team))
            if key not in unique:unique[key]=len(selected);selected.append(team)
            indices.append(unique[key])
        teams=selected;tokens=torch.tensor(list(unique),device=self.device)
        prefix=torch.cat([torch.full((len(teams),1),2,device=self.device),tokens[:,:-1]],1)
        logits=self.model.config_logits(context.expand(len(teams),-1),prefix,proposal=proposal).float();mask=np.zeros(tuple(logits.shape),bool)
        allowed_cache={}
        for i,(team,values) in enumerate(zip(teams,unique)):
            preview=[m['species'] for m in team]
            for j,target in enumerate(values):
                slot=j//21;key=(slot,preview[slot],values[21*slot:j])
                if key not in allowed_cache:allowed_cache[key]=self.allowed(preview,values[:j],j,facts)
                mask[i,j,allowed_cache[key]]=True
        logp=logits.masked_fill(~torch.as_tensor(mask,device=self.device),-torch.inf).log_softmax(-1)
        values=logp.gather(-1,tokens[...,None]).squeeze(-1)
        return (values if per_token else values.sum(-1)).cpu().numpy()[indices]
    @torch.inference_mode()
    def draw(self,preview,count,context,facts=(),proposal=False,deadline=float("inf"),work=None):
        projected=self.model.context_projection(context).expand(count,-1)
        if work is None:
            prefixes=[[] for _ in range(count)];logq=np.zeros(count);start=0
            token=torch.full((count,),2,device=self.device,dtype=torch.long)
            h,cache=self.model.config_prefill(projected,token[:,None]);h=h[:,0]
        else:
            prefixes=work['prefixes'];logq=work['logq'];start=work['position']
            token=torch.tensor([p[-1] if p else 2 for p in prefixes],device=self.device)
            h=torch.as_tensor(work['hidden'],device=self.device)
            cache=[tuple(torch.as_tensor(t,device=self.device) for t in pair) for pair in work['cache']]
        head=self.model.proposal if proposal else self.model.configuration
        for position in range(start,126):
            if time.monotonic()>=deadline:
                return prefixes,logq,dict(prefixes=prefixes,logq=logq,position=position,hidden=h.cpu().numpy(),
                    cache=[tuple(t.cpu().numpy() for t in pair) for pair in cache])
            if position:h=self.model.config_step(projected,token,torch.full_like(token,position),cache)
            logits=head(h).float();mask=torch.zeros_like(logits,dtype=torch.bool)
            for i,prefix in enumerate(prefixes):mask[i,self.allowed(preview,prefix,position,facts)]=True
            logp=logits.masked_fill(~mask,-torch.inf).log_softmax(-1)
            chosen=torch.multinomial(logp.exp(),1,generator=self.generator)
            logq+=logp.gather(-1,chosen).squeeze(-1).cpu().numpy();token=chosen[:,0]
            for prefix,value in zip(prefixes,token.cpu().tolist()):prefix.append(value)
        return prefixes,logq,None
    @torch.inference_mode()
    def propose_blocks(self,teams,blocks,context,facts):
        """Draw each block once; illegal complete teams are MH rejections.

        Unedited tokens stay fixed. The reverse density scores the old block
        with its own prefixes, under the same history-conditioned proposal.
        """
        prefixes=[self.vocab.configuration(t) for t in teams]
        tokens=torch.tensor(prefixes,device=self.device)
        positions=[[21*slot+field for field in fields] for slot,fields in blocks]
        forward=np.zeros(len(teams))
        projected=self.model.context_projection(context).expand(len(teams),-1)
        hidden,cache=self.model.config_prefill(projected,torch.cat([tokens.new_full((len(teams),1),2),tokens[:,:-1]],1))
        rows=[i for i,p in enumerate(positions) for _ in p];cols=[c for p in positions for c in p]
        logits=self.model.proposal(hidden[rows,cols]).float();mask=torch.zeros_like(logits,dtype=torch.bool)
        for j,(i,col) in enumerate(zip(rows,cols)):
            mask[j,self.allowed([m['species'] for m in teams[i]],prefixes[i][:col],col,facts)]=True
        old=logits.masked_fill(~mask,-torch.inf).log_softmax(-1).gather(-1,tokens[rows,cols,None]).squeeze(-1).cpu().numpy()
        reverse=np.bincount(rows,weights=old,minlength=len(teams))
        all_rows=torch.arange(len(teams),device=self.device)
        previous=None
        for step in range(max(map(len,positions))):
            rows=[i for i,p in enumerate(positions) if step<len(p)]
            cols=[positions[i][step] for i in rows]
            current=torch.tensor([p[min(step,len(p)-1)] for p in positions],device=self.device)
            if step:
                if bool(((current-previous)>1).any()):
                    intermediate=torch.maximum(previous,current-1)
                    self.model.config_step(projected,tokens[all_rows,intermediate-1],intermediate,cache)
                decoded=self.model.config_step(projected,tokens[all_rows,current-1],current,cache)
            else:decoded=hidden[all_rows,current]
            previous=current
            logits=self.model.proposal(decoded[rows]).float()
            mask=torch.zeros_like(logits,dtype=torch.bool)
            for j,(i,col) in enumerate(zip(rows,cols)):
                preview=[m['species'] for m in teams[i]]
                mask[j,self.allowed(preview,prefixes[i][:col],col,facts)]=True
            logp=logits.masked_fill(~mask,-torch.inf).log_softmax(-1)
            chosen=torch.multinomial(logp.exp(),1,generator=self.generator)
            forward[rows]+=logp.gather(-1,chosen).squeeze(-1).cpu().numpy()
            tokens[rows,cols]=chosen[:,0]
            for i,col,value in zip(rows,cols,chosen[:,0].cpu().tolist()):prefixes[i][col]=value
        proposed=[copy.deepcopy(team) for team in teams]
        names={'type':'teraType','ev':'evs','iv':'ivs','move':'moves'}
        for i,((slot,fields),values) in enumerate(zip(blocks,prefixes)):
            decoded=self.vocab.decode_configuration(values)[slot]
            for field in fields:
                name=names.get(self.vocab.config_fields[field],self.vocab.config_fields[field])
                proposed[i][slot][name]=decoded[name]
        return proposed,forward,reverse
    @torch.inference_mode()
    def repair(self,preview,anchors,context,facts,active_slot,count=128,deadline=float("inf"),work=None):
        """Enumerated local edits plus an AR proposal with full posterior support.

        Local quadrature and AR samples share one mixture denominator. Invalid
        AR draws retain their zero contribution, without conditional rejection.
        """
        if work is None:
            count=min(count,128);global_count=min(16,max(1,count//8));limit=count-global_count
            pool=[];seen=set()
            def add(team):
                key=tuple(self.vocab.configuration(team))
                if key not in seen and len(pool)<limit:seen.add(key);pool.append(team)
            def compatible(mon,known):
                return set(known.get('moves',[]))<=set(mon['moves']) and all(
                    ident(mon[k])==ident(known[k]) for k in ('item','ability','teraType') if k in known)
            def patch(team):
                out=copy.deepcopy(team)
                for slot,mon in enumerate(out):
                    known=facts[slot]
                    for k in ('item','ability','teraType'):
                        if k in known:mon[k]=known[k]
                    missing=[m for m in known.get('moves',[]) if m not in mon['moves']]
                    for move in missing:
                        index=next((j for j,m in reversed(list(enumerate(mon['moves']))) if m not in known['moves']),None)
                        if index is None:return None
                        mon['moves'][index]=move
                return out
            bases=[p for team in anchors if (p:=patch(team)) is not None]
            for team in bases[:16]:add(team)
            slots=[active_slot]+[s for s in range(6) if s!=active_slot] if active_slot is not None else list(range(6))
            for slot in slots:
                bank=[self.vocab.catalog['sets'][i] for i in self.vocab.species.get(ident(preview[slot]),[])]+[t[slot] for t in anchors]
                for mon in bank:
                    if not compatible(mon,facts[slot]):continue
                    for base in bases[:8]:
                        team=copy.deepcopy(base)
                        team[slot].update({k:copy.deepcopy(mon[k]) for k in ('item','ability','nature','teraType','evs','ivs','moves')})
                        add(team)
                if len(pool)>=limit:break
            # The local component is a finite legal distribution; its normalizer is
            # evaluated explicitly, unlike rejection-conditioned AR probabilities.
            checks=self.engine.call('legal',teams=pool) if pool else []
            pool=[t for t,r in zip(pool,checks) if not r['errors']]
            work=dict(pool=pool,global_count=global_count,draw=None)
        pool=work['pool'];global_count=work['global_count']
        prefixes,_,unfinished=self.draw(preview,global_count,context,facts,proposal=True,deadline=deadline,work=work['draw'])
        if unfinished is not None:
            work['draw']=unfinished
            return None,None,work
        draws=[self.vocab.decode_configuration(p) for p in prefixes]
        for team in draws:
            for i,mon in enumerate(team):
                mon['species']=preview[i];mon['name']=anchors[0][i].get('name',preview[i]);mon['gender']=anchors[0][i]['gender']
        legal=self.engine.call('legal',teams=draws);accepted=[i for i,r in enumerate(legal) if not r['errors']]
        teams=pool+[draws[i] for i in accepted]
        return teams,dict(local_candidates=len(pool),ar_draws=global_count,
            legal_ar_candidates=len(accepted),weighting='local quadrature + AR importance sampling with complete mixture density'),None
    @torch.inference_mode()
    def repair_weights(self,teams,info,context,prior_context,facts):
        ar_logq=self.log_probability(teams,context,proposal=True,facts=facts)
        count=info['local_candidates'];local_logq=ar_logq[:count].astype(float)
        if count:local_logq-=np.logaddexp.reduce(local_logq)
        local={tuple(self.vocab.configuration(t)):p for t,p in zip(teams[:count],local_logq)}
        alpha=.9 if count else 0.
        mixture=np.logaddexp(np.log(alpha)+np.array([local.get(tuple(self.vocab.configuration(t)),-np.inf) for t in teams]) if alpha else -np.inf,
                             np.log(1-alpha)+ar_logq)
        mass=np.r_[np.log(alpha)+local_logq if alpha else [],np.full(info['legal_ar_candidates'],np.log(1-alpha)-np.log(info['ar_draws']))]
        prior=self.log_probability(teams,prior_context)
        return importance_logweights(prior,mixture,mass)
    @torch.inference_mode()
    def sample(self,preview,count,context,prior_context=None,facts=(),proposal=False):
        accepted=[];weights=[];attempted=0
        for _ in range(8):
            n=max(4,count-len(accepted));prefixes,logq,_=self.draw(preview,n,context,facts,proposal)
            teams=[self.vocab.decode_configuration(p) for p in prefixes]
            for team in teams:
                for slot,mon in enumerate(team):mon['species']=preview[slot];mon['name']=preview[slot]
            checks=self.engine.call('legal',teams=teams);attempted+=n
            good=[i for i,r in enumerate(checks) if not r['errors']];selected=[teams[i] for i in good]
            if selected:
                logp=self.log_probability(selected,prior_context if prior_context is not None else context)
                accepted.extend(selected);weights.extend(logp-logq[good])
            if len(accepted)>=count:break
        return accepted[:count],np.array(weights[:count]),dict(attempted_teams=attempted,
            accepted_teams=min(count,len(accepted)),untrained_species=[s for s in preview if ident(s) not in self.vocab.species])
