"""Standalone neural evaluation; observable human outcomes and true intents stay separate."""
from collections import defaultdict
import math
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
from torch.utils.data import DataLoader
from .metrics import Scores,Traditional
from .training_data import EpisodeDataset,collate_episodes
from .training_objective import to_device


def evaluation_loader(directory,source,split,limit,batch=24,workers=2):
    return DataLoader(EpisodeDataset(directory,source,split,limit=limit),batch_size=batch,
                      collate_fn=collate_episodes,num_workers=workers,pin_memory=True)


def predictive(task,batch,out,samples,history):
    previous=None
    if history=='shuffle':
        # Permute completed support-game states, never another row's padding
        # or already-started query game (support lengths differ across players).
        boundary=(batch['game_start']*torch.arange(batch['game_start'].shape[1],device=out['h'].device)).max(-1).values
        rows=(boundary>0).nonzero().flatten();donors=rows.roll(1)
        previous=torch.zeros_like(out['z'][:,:,0])
        previous[:,rows]=out['z'][:,donors,boundary[donors]-1]
    z,change=task.model.strategy_paths(out['h'],samples,batch['game_start'],history,previous_games=previous)
    logp,selected=task.probabilities(batch,dict(h=out['h'],z=z,change=change))
    return logp,selected


def probability_temperature(logp,temperature):
    return (logp/temperature).softmax(-1)


class HumanFrequency:
    def __init__(self,counts,move_count):
        self.counts=counts;self.moves=2*move_count
    def probabilities(self,batch,selected):
        active=batch['active'][selected][:,0].cpu().numpy()
        preview=batch['preview'][selected].cpu().numpy();phase=batch['phase'][selected].cpu().numpy()
        rows=[]
        for a,p,team in zip(active,phase,preview):
            count=self.counts.get((int(p==0),int(a)),{})
            population=self.counts[(int(p==0),-1)]
            keys=[*range(self.moves),*[self.moves+int(s) for s in team],-1]
            prior=np.asarray([population.get(k,0)+.05 for k in keys],float)
            if p==0:prior[:self.moves]=0
            prior/=prior.sum()
            values=np.asarray([count.get(k,0) for k in keys],float)+25*prior
            if p==0:values[:self.moves]=0
            rows.append(values/values.sum())
        return np.stack(rows)


def traditional_probabilities(batch,selected,parameters):
    model=Traditional(parameters);probabilities=[];populations=[]
    for i in range(batch['target'].shape[0]):
        for j in range(batch['target'].shape[1]):
            if not batch['valid'][i,j] or batch['target'][i,j]<0 or batch['phase'][i,j]>=3:continue
            n=int(batch['action_mask'][i,j].sum())
            row={'features':batch['features'][i,j,:n].cpu().numpy(),'target':int(batch['target'][i,j])}
            mixture,population=model.predict(row,i)
            if selected[i,j]:
                width=batch['actions'].shape[2];probabilities.append(np.pad(mixture,(0,width-n)));populations.append(np.pad(population,(0,width-n)))
    return np.stack(probabilities),np.stack(populations)


@torch.inference_mode()
def evaluate(task,data,device,temperature=1.,change_temperature=1.,change_threshold=.5,ablation=False,baseline=None,attribute_batches=20):
    task.eval();scores=defaultdict(Scores);phase_scores=defaultdict(lambda:defaultdict(Scores));early=defaultdict(Scores);changes=Scores()
    field_scores=defaultdict(Scores);q_scores=defaultdict(Scores);paired=defaultdict(list);total_games=0;seen_games=set();label_counts=np.zeros(3,int);pred_counts=np.zeros(3,int)
    groups=defaultdict(Scores)
    change_labels=[];change_probabilities=[];delays=[]
    for index,raw in enumerate(data):
        batch=to_device(raw,device);total_games+=len(batch['games']);seen_games.update(batch['games'])
        torch.manual_seed(701+index)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=task.model(batch,samples=4);logp,selected=task.probabilities(batch,out)
            methods={'predictive':probability_temperature(logp,temperature).cpu().numpy()}
            if ablation:
                for name,samples,history in [('no_previous_game',4,'none'),('wrong_player_history',4,'shuffle'),('mean_z',0,'full'),('sampled_z_32',32,'full')]:
                    torch.manual_seed(701+index)
                    values,_=predictive(task,batch,out,samples,history)
                    methods[name]=probability_temperature(values,temperature).cpu().numpy()
        y=batch['target'][selected].cpu().numpy();phases=batch['phase'][selected].cpu().numpy()
        positions=batch['position'][selected].cpu().numpy();has=batch['has_history'][:,None].expand_as(selected)[selected].cpu().numpy()
        ij=selected.nonzero().cpu().numpy()
        if baseline is not None:
            if batch['source']=='human':methods['species_frequency']=baseline.probabilities(batch,selected)
            else:
                methods['traditional_bayes'],methods['population_logit']=traditional_probabilities(batch,selected,baseline)
                teacher=batch['teacher'][selected].cpu().numpy();available=batch['teacher_valid'][selected].cpu().numpy()
                scores['teacher_reference'].add_many(teacher[available],y[available])
        for name,p in methods.items():
            losses=scores[name].add_many(p,y)
            for phase in np.unique(phases):phase_scores[('preview','turn','forced_switch')[phase]][name].add_many(p[phases==phase],y[phases==phase])
            select=has&(positions<5)
            if select.any():early[name].add_many(p[select],y[select])
            if name in ('predictive','no_previous_game','wrong_player_history'):
                for i in range(len(batch['games'])):
                    ix=(ij[:,0]==i)&select
                    if ix.any():paired[(batch['games'][i],name)].extend(losses[ix].tolist())
        if batch['source']=='controlled':
            labels=batch['change'][selected].cpu().numpy();cp=(out['change'][selected].float()/change_temperature).softmax(-1).cpu().numpy();valid=labels>=0
            changes.add_many(cp[valid],labels[valid]);label_counts+=np.bincount(labels[valid],minlength=3);pred_counts+=np.bincount(cp[valid].argmax(-1),minlength=3)
            change_labels.extend(labels[valid].tolist());change_probabilities.extend((1-cp[valid,0]).tolist())
            for i in range(len(batch['games'])):
                group=('known','unseen_team','unseen_configuration','unseen_policy_family','abrupt_change','gradual_change','configuration_strategy_confounding')[int(batch['games'][i].split('-')[1])%7]
                ix=ij[:,0]==i
                if ix.any():groups[group].add_many(methods['predictive'][ix],y[ix])
                yy=labels[ix];pp=1-cp[ix,0]
                for event in np.flatnonzero(yy==1):
                    detections=np.flatnonzero(pp[event:event+6]>=change_threshold)
                    delays.append(int(detections[0]) if len(detections) else -1)
        if index<attribute_batches:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if batch['source']=='human':
                    facts=batch['facts']
                    for field,allowed in task.fact_allowed.items():
                        f=facts[facts[:,3]==field]
                        if not len(f):continue
                        logits=task.model.attribute_logits(out['h'][f[:,0],f[:,1]],f[:,2:4]).float().masked_fill(~allowed.to(device),-1e9)
                        field_scores[task.vocab.tokens[field]].add_many(logits.softmax(-1).cpu().numpy(),f[:,4].cpu().numpy())
                else:
                    public,_=task.model.encode(batch['public_events'],batch['public_numeric'],batch['public_state'],channel=1,position=batch['position'],segments=batch['segments'])
                    ix=[]
                    for i in range(selected.shape[0]):
                        js=selected[i].nonzero().flatten()
                        if len(js):ix.append((i,int(js[len(js)//2])))
                    ii,jj=map(lambda x:torch.tensor(x,device=device),zip(*ix));targets=batch['configuration'][ii,jj];mask=batch['config_unknown'][ii,jj]
                    decoder=torch.cat([torch.full((len(ii),1),2,device=device,dtype=torch.long),targets[:,:-1]],1)
                    logits=task.model.config_logits(public[ii,jj],decoder).float().masked_fill(~task.config_allowed[None],-1e9)
                    for field,positions_ in task.field_positions.items():
                        valid=mask[:,positions_]
                        if valid.any():field_scores[field].add_many(logits[:,positions_][valid].softmax(-1).cpu().numpy(),targets[:,positions_][valid].cpu().numpy())
                    nn=jj+1;keep=nn<public.shape[1];ii,jj,nn=ii[keep],jj[keep],nn[keep]
                    keep=batch['valid'][ii,nn]&(batch['segments'][ii,jj]==batch['segments'][ii,nn]);ii,jj,nn=ii[keep],jj[keep],nn[keep]
                    if len(ii):
                        targets=batch['configuration'][ii,jj];mask=batch['config_unknown'][ii,jj];decoder=torch.cat([torch.full((len(ii),1),2,device=device,dtype=torch.long),targets[:,:-1]],1)
                        prior=task.model.config_logits(public[ii,jj],decoder).float().masked_fill(~task.config_allowed[None],-1e9)
                        proposal=task.model.config_logits(public[ii,nn],decoder,True).float().masked_fill(~task.config_allowed[None],-1e9)
                        for name,values in [('prior',prior),('post_observation_proposal',proposal)]:
                            q_scores[name].add_many(values[mask].softmax(-1).cpu().numpy(),targets[mask].cpu().numpy())
    comparisons={}
    rng=np.random.default_rng(11)
    for name in ('no_previous_game','wrong_player_history'):
        delta=[np.mean(v)-np.mean(paired[(game,'predictive')]) for (game,method),v in paired.items() if method==name and (game,'predictive') in paired]
        if delta:
            # Opposing sides of one game are one bootstrap cluster.
            d=np.asarray(delta);samples=np.asarray([rng.choice(d,len(d)).mean() for _ in range(500)])
            comparisons[name]={'nll_increase_over_full':float(d.mean()),'95pct_game_bootstrap':np.quantile(samples,[.025,.975]).tolist(),'games':len(d)}
    result={'player_games':total_games,'unique_games':len(seen_games),'methods':{k:v.result() for k,v in scores.items()},
            'phase':{p:{k:v.result() for k,v in methods.items()} for p,methods in phase_scores.items()},
            'first_five_with_previous_game':{k:v.result() for k,v in early.items()},'cross_game_paired':comparisons,
            'attributes':{k:v.result() for k,v in field_scores.items()},'proposal_teacher_forced_probe':{k:v.result() for k,v in q_scores.items()},
            'change':{'scores':changes.result(),'true_class_counts':label_counts.tolist(),'predicted_class_counts':pred_counts.tolist()},
            'controlled_test_groups':{k:v.result() for k,v in groups.items()},
            'scope':'64-step query prefix per player-game, up to 32 prior-game support steps. Human targets are visible outcomes; controlled targets are submitted actions (preview scores lead only). Configuration/proposal scores use true autoregressive prefixes, not full posterior candidate coverage.'}
    if change_labels:
        yy=np.asarray(change_labels)>0;pp=np.asarray(change_probabilities);prediction=pp>=change_threshold;delay=np.asarray(delays)
        result['change'].update(prevalence=float(yy.mean()),average_precision=float(average_precision_score(yy,pp)),roc_auc=float(roc_auc_score(yy,pp)) if yy.any() and not yy.all() else None,
            threshold=change_threshold,false_positive_rate=float(prediction[~yy].mean()),recall=float(prediction[yy].mean()) if yy.any() else None,
            abrupt_events=len(delay),abrupt_detected_within_five_steps=float((delay>=0).mean()) if len(delay) else None,
            mean_delay_if_detected=float(delay[delay>=0].mean()) if (delay>=0).any() else None)
    return result
