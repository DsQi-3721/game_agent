"""Predictive objectives with source-specific supervision and no answer encoder."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TrainingTask(nn.Module):
    def __init__(self, model, vocab):
        super().__init__()
        self.model=model;self.vocab=vocab
        self.register_buffer('move_tokens',torch.tensor(vocab.move_ids))
        self.register_buffer('kind_tokens',torch.tensor(vocab.action_kind_ids))
        fields=['species','item','ability','nature','type',*['ev']*6,*['iv']*6,*['move']*4]*6
        allowed=torch.zeros(126,len(vocab.tokens),dtype=torch.bool)
        for i,field in enumerate(fields):
            allowed[i]=torch.tensor([token.startswith(field+':') for token in vocab.tokens])
        self.register_buffer('config_allowed',allowed)
        self.field_positions={field:[i for i,f in enumerate(fields) if f==field] for field in ('item','ability','nature','type','ev','iv','move')}
        self.fact_allowed={vocab.ids['query:'+q]:torch.tensor([t.startswith(kind+':') for t in vocab.tokens]) for q,kind in [('item','item'),('tera','type')]}

    def probabilities(self,batch,out):
        selected=batch['score']&batch['valid']&(batch['target']>=0)&(batch['phase']<3)
        h=out['h'][selected];z=out['z'][:,selected];phase=batch['phase'][selected]
        if batch['source']=='human':
            keys=self.model.outcome_keys(self.move_tokens,batch['preview'][selected],self.kind_tokens)
            width=len(self.move_tokens)*2+7
            legal=torch.ones((len(h),width),dtype=torch.bool,device=h.device)
            preview=phase==0;legal[preview,:len(self.move_tokens)*2]=False
            logits=[self.model.outcome_logits(h,path,keys).float().masked_fill(~legal,-1e9) for path in z]
        else:
            features=batch['features'][selected];tokens=batch['actions'][selected];legal=batch['action_mask'][selected]
            logits=[self.model.policy_logits(h,features,path,tokens).float().masked_fill(~legal,-1e9) for path in z]
        log_probability=torch.logsumexp(torch.stack([x.log_softmax(-1) for x in logits]),0)-math.log(len(logits))
        return log_probability,selected

    def configuration_loss(self,context,targets,unknown,proposal=False):
        decoder=torch.cat([torch.full((len(targets),1),2,device=targets.device,dtype=torch.long),targets[:,:-1]],1)
        logits=self.model.config_logits(context,decoder,proposal).float().masked_fill(~self.config_allowed[None],-1e9)
        loss=F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none').reshape_as(targets)
        parts=[];metrics={}
        for field,positions in self.field_positions.items():
            valid=unknown[:,positions]&(targets[:,positions]!=0)
            if valid.any():
                value=loss[:,positions][valid].mean();parts.append(value);metrics[field]=value.detach()
        return torch.stack(parts).mean() if parts else loss.sum()*0,metrics

    def attributes(self,batch,out):
        f=batch['facts'];losses=[];metrics={}
        for field,allowed in self.fact_allowed.items():
            rows=f[f[:,3]==field]
            if not len(rows):continue
            if len(rows)>256:rows=rows[torch.randperm(len(rows),device=rows.device)[:256]]
            logits=self.model.attribute_logits(out['h'][rows[:,0],rows[:,1]],rows[:,2:4]).float()
            logits=logits.masked_fill(~allowed.to(logits.device),-1e9)
            value=F.cross_entropy(logits,rows[:,4]);losses.append(value);metrics[self.vocab.tokens[field]]=value.detach()
        return torch.stack(losses).mean() if losses else out['h'].sum()*0,metrics

    def forward(self,batch,samples=4):
        out=self.model(batch,samples=samples)
        logp,selected=self.probabilities(batch,out);target=batch['target'][selected]
        losses=-logp.gather(-1,target[:,None]).squeeze(-1)
        # Extra emphasis on the first five decisions tests knowledge transferred
        # across games. Weights depend on input position, never the target class.
        weights=1.+(batch['position'][selected]<5).float()
        action_loss=(weights*losses).sum()/weights.sum()
        loss=action_loss;metrics={'nll':losses.detach().mean(),'decisions':selected.sum().detach()}
        if batch['source']=='human':
            auxiliary,detail=self.attributes(batch,out);loss=loss+.25*auxiliary
            metrics.update(attribute_nll=auxiliary.detach(),**detail)
        else:
            teacher_valid=batch['teacher_valid'][selected]
            if teacher_valid.any():
                teacher=-(batch['teacher'][selected]*logp).sum(-1)[teacher_valid].mean()
                loss=loss+.35*teacher;metrics['teacher_ce']=teacher.detach()
            changes=selected&(batch['change']>=0)
            change_loss=F.cross_entropy(out['change'][changes].float(),batch['change'][changes])
            loss=loss+.1*change_loss;metrics['change_ce']=change_loss.detach()
            # Public-only encoder: actor C/X and private submissions never enter C/q.
            public,_=self.model.encode(batch['public_events'],batch['public_numeric'],batch['public_state'],
                                        channel=1,position=batch['position'],segments=batch['segments'])
            points=[]
            for i in range(selected.shape[0]):
                valid=selected[i].nonzero().flatten()
                if len(valid):points.append(torch.stack([torch.tensor(i,device=valid.device),valid[torch.randint(len(valid),(),device=valid.device)]]))
            ix=torch.stack(points);i,j=ix.T;target_config=batch['configuration'][i,j];unknown=batch['config_unknown'][i,j]
            config_loss,detail=self.configuration_loss(public[i,j],target_config,unknown)
            loss=loss+.25*config_loss;metrics['config_nll']=config_loss.detach()
            next_step=j+1;keep=(next_step<public.shape[1])
            i,j,next_step=i[keep],j[keep],next_step[keep]
            keep=batch['valid'][i,next_step]&(batch['segments'][i,j]==batch['segments'][i,next_step])
            i,j,next_step=i[keep],j[keep],next_step[keep]
            if len(i):
                proposal_loss,_=self.configuration_loss(public[i,next_step],batch['configuration'][i,j],batch['config_unknown'][i,j],True)
                loss=loss+.15*proposal_loss;metrics['post_observation_proposal_nll']=proposal_loss.detach()
        metrics['loss']=loss.detach()
        return loss,metrics


def to_device(batch,device):
    return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
