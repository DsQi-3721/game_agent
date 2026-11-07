"""Joint C/X/M/Z particle filter, isolated from our action selection.

Updates consume one engine resolution window (possibly a forced switch), not an
arbitrary partial damage line. The original event sequence is kept in the journal.
"""
import copy,json,time,itertools
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from .engine import Mechanics
from .configuration import ConfigurationSampler
from .data import collate
from .tokens import Vocabulary,ident
from .network import JointNetwork
from .observations import reveals
from .training_data import PublicState
from .recovery import recover as recover_belief
from .intent import actor_row,submission_commands
from .strategy import change_eligible,advance_changes,change_marginals
from .rejuvenation import refresh,configuration_support,resample_configurations


def normalize(logw):
    finite=np.isfinite(logw)
    if not finite.any():return None
    p=np.zeros_like(logw);p[finite]=np.exp(logw[finite]-np.max(logw[finite]));return p/p.sum()

def systematic(p,n,rng):
    return np.searchsorted(np.cumsum(p),(rng.random()+np.arange(n))/n).clip(0,len(p)-1)


def condition_weights(weights,likelihood):
    logw=np.full(len(weights),-np.inf);valid=(weights>0)&(likelihood>0)
    logw[valid]=np.log(weights[valid])+np.log(likelihood[valid])
    posterior=normalize(logw)
    return posterior,float(np.logaddexp.reduce(logw))


class JointOpponentModel:
    def __init__(self,model_path,catalog,observer_team,preview,observer='p1',particles=1024,configurations=128,branches=32,device='cuda:1',seed=83,game_id='battle',initial_teams=None,initial_events=None,time_budget=None,opponent_id=None,move_budget=32,behavior_evidence=True):
        self.model=JointNetwork.load(model_path,device);self.vocab=Vocabulary(catalog);self.engine=Mechanics()
        self.rules_version='showdown-gen9ou-'+self.engine.call('version')[:8];self.opponent_id=opponent_id
        self.device=device;self.observer=observer;self.side='p2' if observer=='p1' else 'p1';self.own_team=copy.deepcopy(observer_team);self.preview=list(preview)
        self.n=particles;self.configuration_budget=min(configurations,particles);self.branches=branches;self.move_budget=move_budget;self.rng=np.random.default_rng(seed);self.game_id=game_id
        self.torch_rng=torch.Generator(device=device).manual_seed(seed);self.sampler=ConfigurationSampler(self.model,self.vocab,self.engine,self.torch_rng);self.journal=[];self.events=[];self.initial_events=[];self.facts=[];self.known=[{} for _ in preview]
        self.position=0;self.games_seen=1;self.revision=0;self.mismatch=None;self.cost={};self.cached=None;self.inference_memory=None;self.context_memory=None;self.prior_context=None;self.context=None;self.memory=None;self.time_budget=time_budget
        self.z=None;self.previous=np.zeros((particles,4),np.int64);self.weights=np.ones(particles)/particles;self.cutoff=0;self.version=f'joint-{self.model.config.layers}x{self.model.config.width}-step{self.model.metadata.get("steps",0)}';self.coverage={}
        if self.model.behavior_prior is not None:self.version+=f'-behavior{len(self.model.behavior_prior["mean"])}'
        self.start_memory=None;self.start_z=None;self.start_position=0
        self.behavior_evidence=behavior_evidence
        self.applied_windows=0;self.past_windows=0;self.pending=None
        self.change_history=torch.zeros((self.n,5),device=device,dtype=torch.int8)
        self.version+='-jointchange'
        self.initial_events=[f'|poke|{self.side}|{s}|' for s in preview]+[f'|poke|{self.observer}|{m["species"]}|' for m in observer_team]
        if initial_events is not None:self.initial_events=list(initial_events)
        self.events=list(self.initial_events);self._encode_inference(self.events)
        self.prior_context=self.context.clone();self._initialize(initial_teams)
    def _public_view(self):
        public=PublicState(self.vocab,self.side,[self.vocab.token('species',s) for s in self.preview])
        public.update(self.events)
        return public
    def _encode_inference(self,events,state=None):
        public=self._public_view();tokens,nums=self.vocab.events(events,self.side,public.actors.copy())
        if not len(tokens):tokens=np.zeros((1,4),np.int64);nums=np.zeros((1,8),np.float32)
        phase=0 if not self.journal else 2 if public.hp.get(public.active.get(self.side),1.)==0 else 1
        state=torch.as_tensor(public.state(phase) if state is None else state,device=self.device)[None,None]
        with torch.inference_mode():
            h,self.context_memory=self.model.encode(torch.tensor(tokens,device=self.device)[None,None],torch.tensor(nums,device=self.device)[None,None],state,memory=self.inference_memory,channel=1,position=torch.tensor([[self.position]],device=self.device))
        self.context=h[:,0]
    def _initialize(self,teams=None):
        if teams is None:
            teams,logw,diagnostics=self.sampler.sample(self.preview,self.configuration_budget,self.context,self.prior_context,self.known)
            self.coverage.update(diagnostics)
        else:logw=np.zeros(len(teams));self.coverage['supplied_configurations']=True
        if not teams:raise ValueError('No legal configuration proposal; candidate coverage failed')
        # Gender is public in team-preview details and affects Attract etc.; it
        # must not inherit a simulator seed's arbitrary default.
        genders={}
        for line in self.events:
            parts=line.split('|')
            if len(parts)>3 and parts[1] in ('poke','switch','drag'):
                details=parts[3].split(', ');gender=next((x for x in details[1:] if x in ('M','F')),None)
                if gender:genders[(parts[2][:2],ident(details[0]))]=gender
        for side,team in [(self.observer,self.own_team),*((self.side,t) for t in teams)]:
            for mon in team:
                rules=self.sampler.rules(mon['species']);mon['gender']=genders.get((side,ident(mon['species']))) or mon.get('gender') or rules.get('gender') or ('M' if self.rng.random()<rules.get('genderRatio',{}).get('M',.5) else 'F')
        cp=normalize(logw);indices=systematic(np.ones(len(teams))/len(teams),self.n,self.rng);self.configurations=teams;self.c=indices
        starts=[]
        for team in teams:starts.append(self.engine.call('create',teams={self.observer:self.own_team,self.side:team})['handle'])
        self.handles=np.asarray(starts)[indices];counts=np.bincount(indices,minlength=len(teams))
        self.weights=cp[indices]/counts[indices];self.weights/=self.weights.sum();self.cached=None
        self.coverage['distinct_initial_configurations']=len(teams)
        self.paths=[[] for _ in range(self.n)];self.path_log_density=np.zeros(self.n)
        self.path_start_z=[None]*self.n if self.z is None else self.z.float().cpu().tolist()
    def update_action_log_probs(self,logp,mask):
        if self.behavior_evidence:return logp
        return (-mask.sum(-1,keepdim=True).float().log()).expand_as(logp).masked_fill(~mask,-torch.inf)
    @torch.inference_mode()
    def _prepare(self):
        if self.cached is not None:return self.cached
        handles=sorted(set(map(int,self.handles)))
        views=dict(zip(handles,self.engine.call('views',handles=handles,side=self.side,events=self.events)))
        rows=[];actions=[];row_cache={};public=self._public_view()
        encoded=self.vocab.events(self.events[self.cutoff:],self.side,public.actors.copy())
        configurations={i:np.array(self.vocab.configuration(self.configurations[i])) for i in set(self.c)}
        for i in range(self.n):
            key=(self.handles[i],self.c[i],tuple(self.previous[i]))
            if key not in row_cache:
                row_cache[key]=actor_row(self.vocab,views[self.handles[i]],self.configurations[self.c[i]],self.events[self.cutoff:],self.side,self.previous[i],self.position,public,
                    encoded_events=encoded,configuration=configurations[self.c[i]])
            row,a=row_cache[key];rows.append([row]);actions.append(a)
        if not self.mismatch:self._encode_inference(self.events[self.cutoff:],rows[int(np.argmax(self.weights))][0]['state'])
        batch=collate(rows,self.device)
        # Causal replay and online updates use the same FP32 precision.
        with torch.autocast('cuda',enabled=False):
            h,memory=self.model.encode(batch['actor_events'],batch['actor_numeric'],batch['actor_state'],batch['configuration'],memory=self.memory,position=batch['position'])
            eligible=change_eligible(batch['actor_state'][:,0],batch['action_mask'][:,0])
            generator=self.torch_rng
            if self.pending and self.pending['phase'] in ('refresh','recovery'):
                generator=torch.Generator(device=self.device);generator.set_state(self.torch_rng.get_state())
            z,noise,mode,_,_=self.model.sample_strategy(h[:,0],self.z,generator=generator,eligible=eligible)
            logits=self.model.policy_logits(h,batch['features'],z[:,None],action_tokens=batch['actions']).float().masked_fill(~batch['action_mask'],-1e9)
            logp=logits[:,0].log_softmax(-1)
            update_logp=self.update_action_log_probs(logp,batch['action_mask'][:,0]).cpu().numpy()
            logp=logp.cpu().numpy();probabilities=np.exp(logp)
        self.cached={'actions':actions,'p':probabilities,'logp':logp,'update_p':np.exp(update_logp),'update_logp':update_logp,'z':z,'noise':noise,'memory':memory,'tokens':batch['actions'][:,0].cpu().numpy(),'features':batch['features'][:,0].cpu().numpy(),'mode':mode,'eligible':eligible,'views':views}
        return self.cached
    def predict(self,conditional=False):
        prepared=self._prepare();total=defaultdict(float)
        for i,(actions,p) in enumerate(zip(prepared['actions'],prepared['p'])):
            for a,prob in zip(actions,p):
                if a['kind']=='team':
                    tail=[s for s in range(1,7) if s!=a['order'][0]]
                    for rest in itertools.permutations(tail):total['team:'+''.join(map(str,(a['order'][0],*rest)))]+=self.weights[i]*float(prob)/120
                else:total[a['key']]+=self.weights[i]*float(prob)
        result={'actions':[{'action':k,'probability':v} for k,v in sorted(total.items(),key=lambda x:-x[1])]}
        if conditional:
            by=defaultdict(lambda:defaultdict(float));mass=defaultdict(float)
            for i,actions in enumerate(prepared['actions']):
                mass[int(self.c[i])]+=self.weights[i]
                for a,p in zip(actions,prepared['p'][i]):by[int(self.c[i])][a['key']]+=self.weights[i]*float(p)
            result['conditional']=[{'configuration':c,'weight':mass[c],'actions':{a:p/mass[c] for a,p in by[c].items()},'preview_tail':'uniform over remaining 5! orders'} for c in sorted(mass,key=mass.get,reverse=True) if mass[c]>0]
        return result
    def update(self,events,our_command=None,private_events=None,event_id=None,sequence=None,recover=True):
        seq=(self.pending['sequence'] if self.pending else len(self.journal)) if sequence is None else sequence
        record={'events':list(events),'our_command':our_command,'private_events':private_events,'event_id':event_id or f'{self.game_id}:{seq}'}
        if self.pending:
            if seq!=self.pending['sequence'] or record!=self.pending['record']:
                raise ValueError('Finish the pending observation with resume() before sending another window')
            return self.resume()
        if seq<len(self.journal):
            if record==self.journal[seq]:return self.query()
            journal=copy.deepcopy(self.journal);journal[seq]=record;self.revision+=1;self._replay(journal,seq);return self.query()
        if seq!=len(self.journal):raise ValueError('A resolution window is missing; provide it before the next window')
        if any(r['event_id']==record['event_id'] for r in self.journal):return self.query()
        started=time.monotonic()
        self.cost={'observation_applied':False,'observation_supported':None,'calls':0,'total_seconds':0.}
        self.pending=dict(phase='mechanics',record=record,sequence=seq,recover=recover)
        if self.mismatch:
            self.journal.append(record);self.events.extend(events);self._record_facts(events,seq);self.cached=None;self._encode_inference(self.events[self.cutoff:])
            self.pending.update(phase='recovery',work=None)
            if not recover:self.pending=None
        else:
            prepared=self._prepare();jobs=[];groups=[]
            for handle in dict.fromkeys(self.handles[self.weights>0]):
                members=np.flatnonzero((self.handles==handle)&(self.weights>0));i=members[0]
                actions=prepared['views'][handle]['actions'] or prepared['actions'][i]
                indices=np.array([a['order'][0]-1 for a in actions]) if actions[0]['kind']=='team' else np.arange(len(actions))
                p=prepared['update_p'][members][:,indices].astype(float);p/=p.sum(1,keepdims=True)
                jobs.append(dict(handle=int(handle),commands=submission_commands(self.side,actions,our_command),seed=int(self.rng.integers(1,2**30))))
                groups.append((handle,members,actions,indices,p))
            self.pending.update(jobs=jobs,groups=groups,results=[[None]*len(j['commands']) for j in jobs],
                retry=False,retry_attempts=0,mechanics_seconds=0.,reused_actions=0)
        return self._continue(started)

    def resume(self):
        """Continue the same observation, including output preparation in its budget."""
        if self.pending is None:return self.query()
        return self._continue(time.monotonic())

    def _continue(self,started):
        phase=self.pending['phase'] if self.pending else None
        deadline=float('inf') if self.time_budget is None else started+(.25 if phase=='refresh' else .75)*self.time_budget
        if phase=='mechanics':self._mechanics(deadline)
        if self.pending and self.pending['phase']=='posterior' and (phase=='posterior' or self.time_budget is None):
            tick=time.monotonic();self._apply_observation(self.pending);self.cost['posterior_seconds']=time.monotonic()-tick
        if self.pending and self.pending['phase']=='recovery' and (phase=='recovery' or self.time_budget is None):
            accepted,work=recover_belief(self,deadline,self.pending.get('work'))
            if work is None:self.pending=None
            else:self.pending['work']=work
        if self.pending and self.pending['phase']=='refresh' and (phase=='refresh' or self.time_budget is None):
            work=refresh(self,self.cost['resampled'],deadline,self.pending.get('work'))
            if work is None:self.pending=None
            else:self.pending['work']=work
        self.cost['calls']+=1;self.cost['update_complete']=self.pending is None
        prediction_start=time.monotonic();result=self.query()
        self.cost['prediction_seconds']=time.monotonic()-prediction_start
        self.cost['seconds']=time.monotonic()-started;self.cost['total_seconds']+=self.cost['seconds']
        return result

    def _mechanics(self,deadline):
        work=self.pending;record=work['record'];started=time.monotonic()
        while time.monotonic()<deadline:
            jobs=[];indices=[]
            for k,(job,results) in enumerate(zip(work['jobs'],work['results'])):
                missing=[i for i,r in enumerate(results) if r is None or r.get('timed_out')]
                work['reused_actions']+=len(results)-len(missing)
                if missing:
                    jobs.append(dict(job,commands=[job['commands'][i] for i in missing],
                        continuations=[results[i].get('continuation') if results[i] else None for i in missing]))
                    indices.append((k,missing))
            if jobs:
                resolved=self.engine.call('transition_groups',jobs=jobs,
                    events=record['private_events'] if record['private_events'] is not None else record['events'],
                    private=record['private_events'] is not None,observer=self.observer,
                    branches=0 if work['retry'] else self.branches,keep=self.n,
                    budget_ms=None if deadline==float('inf') else max(0,1000*(deadline-time.monotonic())))
                for (k,missing),results in zip(indices,resolved):
                    if results is not None:
                        for i,r in zip(missing,results):work['results'][k][i]=r
            results=[r for rs in work['results'] for r in rs]
            self.cost.update(completed_actions=sum(r is not None and not r.get('timed_out',False) for r in results),
                total_actions=len(results),reused_actions=work['reused_actions'])
            if any(r is None or r.get('timed_out') for r in results):break
            if work['recover'] and not work['retry'] and any(not r['likelihood_is_exact'] for r in results) and not any(r['likelihood']>0 for r in results):
                work['retry']=True;work['retry_attempts']=sum(r['attempts'] for r in results)
                work['results']=[[None]*len(j['commands']) for j in work['jobs']]
                continue
            work['mechanics_seconds']+=time.monotonic()-started
            work['phase']='posterior'
            return
        work['mechanics_seconds']+=time.monotonic()-started
        self.cost['mechanics_seconds']=work['mechanics_seconds']

    def _apply_observation(self,work):
        prepared=self.cached;record=work['record'];events=record['events'];old_handles=set(self.handles)
        cache={};likelihoods=np.zeros(self.n)
        for (handle,members,actions,indices,p),results in zip(work['groups'],work['results']):
            likelihood=np.array([r['likelihood'] for r in results]);likelihoods[members]=p@likelihood
            cache[handle]=(actions,indices,results,likelihood)
        weights,log_evidence=condition_weights(self.weights,likelihoods)
        self.journal.append(record);self.events.extend(events);self._record_facts(events,work['sequence'])
        generated={o['handle'] for rs in work['results'] for r in rs for o in r['outcomes']}
        self.cost.update(mechanics_runs=sum(r['attempts'] for rs in work['results'] for r in rs)+work['retry_attempts'],
            mechanics_seconds=work['mechanics_seconds'],distinct_states_before=len(cache),mechanics_retry=work['retry'],
            observation_supported=weights is not None,observation_applied=weights is not None,
            observation_probability_estimate=float(np.exp(log_evidence)),log_observation_probability_estimate=log_evidence if weights is not None else None,
            likelihood_is_exact=all(r['likelihood_is_exact'] for rs in work['results'] for r in rs),
            observation_estimator='all legal actions marginalized; conditional state sampling')
        self.pending=None
        if weights is None:
            self.mismatch={'reason':'no particle explains this resolution window','at':record['event_id'],'diagnosis':'unresolved: parser/rules, configuration coverage, or finite Monte Carlo budget'}
            if generated-old_handles:self.engine.call('release',handles=list(generated-old_handles))
            self.cached=None;self._encode_inference(self.events[self.cutoff:])
            if work['recover']:self.pending=dict(phase='recovery',record=record,sequence=work['sequence'],work=None)
            return
        ess=float(1/(weights@weights));resampled=ess<self.n*.5
        parents,next_weights=resample_configurations(self,weights) if resampled else (np.arange(self.n),weights)
        handles=self.handles[parents].copy();previous=self.previous[parents].copy();paths=[];path_density=self.path_log_density[parents].copy()
        noise=prepared['noise'].cpu().tolist()
        for j,i in enumerate(parents):
            if weights[i]==0:paths.append(self.paths[i]);continue
            actions,indices,results,likelihood=cache[self.handles[i]]
            mass=prepared['update_p'][i,indices].astype(float)*likelihood
            selected=int(self.rng.choice(len(actions),p=mass/mass.sum()));a=actions[selected]
            outcomes=results[selected]['outcomes'];op=np.array([o['weight'] for o in outcomes])
            o=outcomes[int(self.rng.choice(len(op),p=op/op.sum()))]
            handles[j]=o['handle'];previous[j]=prepared['tokens'][i,indices[selected]]
            path_density[j]+=float(prepared['update_logp'][i,indices[selected]])+o['random_log_probability']
            if a['kind']=='team':path_density[j]-=np.log(120.)
            paths.append(self.paths[i]+[dict(action=a['key'],command=a.get('command'),random_tape=o['random_tape'],noise=None if a['kind']=='waiting' else noise[i])])
        self.c=self.c[parents];self.handles=handles;self.paths=paths;self.path_log_density=path_density
        self.applied_windows=len(self.journal);self.path_start_z=[self.path_start_z[i] for i in parents];self.weights=next_weights
        consumed=any(w>0 and prepared['actions'][i][0]['kind']!='waiting' for w,i in zip(self.weights,parents))
        parent_tensor=torch.as_tensor(parents,device=self.device)
        self.change_history=advance_changes(self.change_history,prepared['mode'],prepared['eligible'])[parent_tensor]
        if consumed:
            self.previous=previous;self.memory=[m[parent_tensor] for m in prepared['memory']];self.z=prepared['z'][parent_tensor]
            self.inference_memory=self.context_memory;self.cutoff=len(self.events)-len(events);self.position+=1
        else:
            self.previous=self.previous[parents]
            if self.memory:self.memory=[m[parent_tensor] for m in self.memory]
            if self.z is not None:self.z=self.z[parent_tensor]
        self._encode_inference(self.events[self.cutoff:]);self.cached=None
        unused=(old_handles|generated)-set(self.handles)
        if unused:self.engine.call('release',handles=list(map(int,unused)))
        self.cost.update(effective_sample_size=ess,resampled=resampled,distinct_states_after=len(set(handles)))
        self.pending=dict(phase='refresh',record=record,sequence=work['sequence'],work=None)

    def _resample(self):
        if 1/(self.weights@self.weights)<self.n*.5:
            indices,weights=resample_configurations(self,self.weights);self.c=self.c[indices];self.handles=self.handles[indices];self.previous=self.previous[indices]
            self.paths=[self.paths[i] for i in indices];self.path_start_z=[self.path_start_z[i] for i in indices];self.path_log_density=self.path_log_density[indices]
            self.change_history=self.change_history[torch.as_tensor(indices,device=self.device)]
            self.z=self.z[torch.as_tensor(indices,device=self.device)];self.memory=[m[torch.as_tensor(indices,device=self.device)] for m in self.memory];self.weights=weights;self.cached=None
    def _record_facts(self,events,seq):
        # These are observed facts, not inferred original items after Trick.
        for j,line in enumerate(events):
            p=line.split('|')
            holder=next((x[5:] for x in p[3:] if x.startswith('[of] ')),p[2] if len(p)>2 else '')
            if len(p)>2 and (p[2].startswith(self.side) or holder.startswith(self.side)) and (p[1] in ('move','-item','-enditem','-ability','-terastallize','-damage','-heal','faint','-status','-boost','-unboost') or any(x.startswith(('[from] ability:','[from] item:')) for x in p[3:])):
                self.facts.append({'event':f'{seq}:{j}','kind':p[1],'subject':p[2] if p[2].startswith(self.side) else holder,'value':p[3:]})
                # Move/ability changes can be copied or transformed; the engine
                # enforces their historical consistency. Do not freeze them as C.
        self.known=self._configuration_facts(self.events)
    def _configuration_facts(self,events):
        known=[{} for _ in self.preview]
        if any('zoroark' in ident(s) for s in self.preview):return known
        active=None;altered_moves=False;changed_items=set();changed_abilities=False
        for line in events:
            p=line.split('|')
            if len(p)<3:continue
            kind=p[1]
            if kind=='move' and len(p)>3 and ident(p[3]) in ('skillswap','roleplay','entrainment','simplebeam','worryseed','doodle'):changed_abilities=True
            source=next((x.split('item:',1)[1] for x in p[3:] if x.startswith('[from] item:')),None)
            holder=next((x[5:] for x in p[3:] if x.startswith('[of] ')),p[2])
            if source and holder.startswith(self.side) and active is not None and active not in changed_items:known[active]['item']=ident(source)
            ability=next((x.split('ability:',1)[1] for x in p[3:] if x.startswith('[from] ability:')),None)
            if ability and holder.startswith(self.side) and active is not None and not changed_abilities and ident(ability) in self.sampler.rules(self.preview[active])['abilities']:known[active]['ability']=ident(ability)
            if not p[2].startswith(self.side):continue
            if kind in ('switch','drag','replace'):
                species=ident(p[3].split(',')[0]);active=next((i for i,s in enumerate(self.preview) if ident(s)==species),None);altered_moves=False
            if active is None:continue
            if kind=='-transform':altered_moves=True
            if kind=='move' and not altered_moves and not any(x.startswith('[from]') for x in p[4:]):
                move=ident(p[3]);known[active].setdefault('moves',[])
                if move!='struggle' and move not in known[active]['moves']:known[active]['moves'].append(move)
                if move in ('mimic','sketch','transform'):altered_moves=True
            if kind=='-terastallize':known[active]['teraType']=ident(p[3])
            if kind in ('-item','-enditem'):
                acquired=kind=='-item' and any(x.startswith('[from]') and x!='[from] ability: Frisk' for x in p[4:])
                if acquired:changed_items.add(active)
                if active not in changed_items:known[active]['item']=ident(p[3])
                if kind=='-enditem':changed_items.add(active)
            if kind=='-ability' and not changed_abilities and not any(x.startswith('[from]') for x in p[4:]) and ident(p[3]) in self.sampler.rules(self.preview[active])['abilities']:known[active]['ability']=ident(p[3])
            if kind in ('-activate','-start','-end') and not changed_abilities and len(p)>3:
                ability=ident(p[3].removeprefix('ability: '))
                if ability in self.sampler.rules(self.preview[active])['abilities']:known[active]['ability']=ability
        return known
    def _replay(self,journal,changed):
        self.cached=None
        self.journal=copy.deepcopy(journal)
        self.events=self.initial_events+[e for r in journal for e in r['events']]
        self.facts=[]
        for seq,record in enumerate(journal):self._record_facts(record['events'],seq)
        self.mismatch={'reason':'history changed; posterior requires recomputation','at':journal[changed]['event_id']}
        self.applied_windows=None;self.cost={}
        self._encode_inference(self.events[self.cutoff:])
        self.pending=dict(phase='recovery',record=journal[-1],sequence=len(journal)-1,work=None)
        self.cost.update(calls=0,total_seconds=0.,observation_applied=False)
        self.resume()
    def query(self):
        prepared=self._prepare()
        if 'report' not in prepared:prepared['report']=self._report(prepared)
        return dict(prepared['report'],coverage=self.coverage,model_mismatch=self.mismatch,
            pending_update=None if self.pending is None else {'event_id':self.pending['record']['event_id'],'phase':self.pending['phase']},
            prediction_valid=self.mismatch is None and (self.pending is None or self.pending['phase']=='refresh'),
            updated_at=datetime.now(timezone.utc).isoformat(),inference_cost=self.cost)

    def _report(self,prepared):
        configuration=[];states=[];views=prepared['views'];support=configuration_support(self)
        configuration_mass=np.bincount(self.c,weights=self.weights,minlength=len(self.configurations));state_mass=defaultdict(float)
        for handle,w in zip(self.handles,self.weights):state_mass[handle]+=w
        for slot,species in enumerate(self.preview):
            fields={k:defaultdict(float) for k in ('item','ability','nature','teraType','moves','evs','ivs')}
            current={k:defaultdict(float) for k in ('hp_fraction','item','ability','status','locked_move','pp','boosts','volatiles','tera')}
            for ci in np.flatnonzero(configuration_mass):
                w=configuration_mass[ci];mon=self.configurations[ci][slot]
                for k,values in fields.items():
                    if k=='moves':
                        for move in mon[k]:values[move]+=w
                    else:values[json.dumps(mon[k],sort_keys=True) if isinstance(mon[k],dict) else str(mon[k])]+=w
            for handle,w in state_mass.items():
                x=next(m for m in views[handle]['state'] if m['slot']==slot+1)
                for k,value in [('hp_fraction',round(x['hp']/x['maxhp'],4)),('item',x['item']),('ability',x['ability']),('status',x['status']),('locked_move',x.get('locked_move'))]:current[k][str(value)]+=w
                for k,value in [('pp',{m['id']:m['pp'] for m in x['moves']}),('boosts',x['boosts']),('volatiles',x['volatiles']),('tera',x['tera'])]:current[k][json.dumps(value,sort_keys=True)]+=w
            def ranked(values,numeric=False):return [{'value':float(k) if numeric else json.loads(k) if isinstance(k,str) and (k.startswith(('{','[','"')) or k in ('null','true','false')) else None if k=='None' else k,'probability':float(v)} for k,v in sorted(values.items(),key=lambda kv:-kv[1])]
            configuration.append({'slot':slot+1,'species':species,**{k:ranked(v) for k,v in fields.items()}});states.append({'slot':slot+1,**{k:ranked(v,k=='hp_fraction') for k,v in current.items()}})
        prediction=self.predict();p=np.array([a['probability'] for a in prediction['actions']]);change=change_marginals(self.change_history,self.weights).cpu().tolist()
        behavior={name:float(np.einsum('i,ij,ij->',self.weights,prepared['p'],prepared['features'][...,index])) for name,index in [('attack',0),('status',1),('switch',2),('tera',3),('recovery',11),('setup',12)]}
        revealed=reveals(self.events,self.observer,[m['species'] for m in self.own_team])
        for fields in revealed:
            if 'moves' in fields:fields['moves']=list(dict.fromkeys(fields['moves']))
        return {'format':'gen9ou','rules_version':self.rules_version,'model_version':self.version,'prototype_version':self.version,'snapshot_id':f'{self.game_id}:{self.revision}:{len(self.journal)}',
            'opponent':{'id':self.opponent_id,'games_observed':self.games_seen,'facts':self.facts,'initial_configuration':configuration,'current_state':states,
                'information':{'public_events_observed':len(self.events),'observed_about_us':[{'slot':i+1,'species':mon['species'],'revealed':revealed[i]} for i,mon in enumerate(self.own_team)]},
                'strategy':{'current_action_tendencies':behavior,
                    'recent_change':{'window_decisions':5,'probability':float(1-change[0]),
                        'stable_probability':change[0],'type_marginals':dict(zip(('abrupt','gradual'),change[1:]))} if self.position else None,
                    'evidence':{'games_observed':self.games_seen,'resolution_windows':self.past_windows+(self.applied_windows or 0)},
                    'behavior_profile':self.model.behavior_profile(self.z if self.z is not None else prepared['z'],self.weights)}},
            'joint_posterior':{'handle':f'{self.game_id}:{self.revision}:{len(self.journal)}','particles':self.n,'distinct_configurations':support['distinct'],'distinct_states':len(set(self.handles))},
            'prediction':prediction,'uncertainty':{'action_entropy':float(-(p*np.log(np.maximum(p,1e-300))).sum()),'effective_sample_size':float(1/(self.weights@self.weights)),'configuration_effective_sample_size':support['effective_sample_size'],'configuration_entropy':support['entropy']},
            'applied_windows':self.applied_windows}
    def dossier(self):
        from .dossier import dossier
        return dossier(self)
    def sample_joint(self,count=1):
        prepared=self._prepare();out=[]
        for i in self.rng.choice(self.n,count,p=self.weights):
            actions=prepared['actions'][i];p=prepared['p'][i,:len(actions)];j=int(self.rng.choice(len(actions),p=p/p.sum()))
            action=copy.deepcopy(actions[j])
            if action['kind']=='team':
                tail=[s for s in range(1,7) if s!=action['order'][0]];self.rng.shuffle(tail);action['order']=[action['order'][0],*tail];order=''.join(map(str,action['order']));action['key']='team:'+order;action['command']='team '+order
            out.append({'configuration':copy.deepcopy(self.configurations[self.c[i]]),'current_state':prepared['views'][self.handles[i]]['state'],'information':{'previous_submitted_action':[self.vocab.tokens[t] for t in self.previous[i] if t],'public_events':list(self.events)},'strategy':prepared['z'][i].cpu().tolist(),'action':action})
        return out
    def _unfinished_results(self):
        if self.pending is None:return []
        if self.pending['phase'] in ('mechanics','posterior'):
            results=[r for rs in self.pending['results'] for r in rs]
        elif self.pending['phase']=='recovery' and self.pending.get('work'):
            results=self.pending['work']['paths']
        else:results=[]
        return [r for r in results if r and r.get('timed_out')]

    def _state_handles(self):
        handles=set(map(int,self.handles))
        if self.pending and self.pending['phase'] in ('mechanics','posterior'):
            handles.update(o['handle'] for rs in self.pending['results'] for r in rs if r for o in r['outcomes'])
        if self.pending and self.pending['phase']=='recovery' and self.pending.get('work'):
            work=self.pending['work'];handles.update(work['handles'])
            handles.update(p['handle'] for p in work['paths'] if p and p['complete'])
        return handles

    def _map_pending_handles(self,mapping):
        if not self.pending:return
        if self.pending['phase'] in ('mechanics','posterior'):
            for job in self.pending['jobs']:job['handle']=mapping[job['handle']]
            self.pending['groups']=[(mapping[h],*rest) for h,*rest in self.pending['groups']]
            for rs in self.pending['results']:
                for r in rs:
                    if r:
                        for o in r['outcomes']:o['handle']=mapping[o['handle']]
        if self.pending['phase']=='recovery' and self.pending.get('work'):
            work=self.pending['work'];work['handles']=[mapping[h] for h in work['handles']]
            for path in work['paths']:
                if path and path['complete']:path['handle']=mapping[path['handle']]

    def snapshot(self):
        fields=('own_team','preview','observer','side','n','configuration_budget','branches','move_budget','behavior_evidence','game_id','opponent_id','rules_version','journal','events','initial_events','facts','known','position','games_seen','revision','mismatch','cost','version','coverage','configurations','c','handles','weights','previous','cutoff','start_position','time_budget','paths','path_start_z','path_log_density','applied_windows','past_windows','pending')
        state={k:copy.deepcopy(getattr(self,k)) for k in fields};state['rng']=copy.deepcopy(self.rng.bit_generator.state);state['torch_rng']=self.torch_rng.get_state()
        for k in ('z','context','prior_context','start_z','change_history'):state[k]=getattr(self,k).detach().cpu() if getattr(self,k) is not None else None
        for k in ('memory','inference_memory','context_memory','start_memory'):state[k]=[m.detach().cpu() for m in getattr(self,k)] if getattr(self,k) else None
        state['cached']=copy.deepcopy(self.cached)
        if state['cached']:
            state['cached']['z']=state['cached']['z'].cpu();state['cached']['noise']=state['cached']['noise'].cpu();state['cached']['memory']=[m.cpu() for m in state['cached']['memory']]
            for k in ('mode','eligible'):state['cached'][k]=state['cached'][k].cpu()
        state['engine_states']={int(h):self.engine.call('snapshot',handle=int(h)) for h in self._state_handles()}
        state['engine_continuations']=self.engine.call('snapshot_continuations',handles=[r['continuation'] for r in self._unfinished_results()])
        return state
    def restore(self,state):
        if state['rules_version']!=self.rules_version or state['version']!=self.version:raise ValueError('Replay this history with the matching rule engine and model checkpoint')
        data=copy.deepcopy(state);engines=data.pop('engine_states');continuations=data.pop('engine_continuations');rng_state=data.pop('rng');torch_state=data.pop('torch_rng')
        self.engine.call('release_continuations',handles=[r['continuation'] for r in self._unfinished_results()])
        self.engine.call('release',handles=list(self._state_handles()))
        mapping={h:self.engine.call('load',state=s)['handle'] for h,s in engines.items()}
        for k,v in data.items():setattr(self,k,v)
        self.handles=np.array([mapping[int(h)] for h in self.handles]);self.rng.bit_generator.state=rng_state
        self._map_pending_handles(mapping)
        pending_mapping=self.engine.call('load_continuations',states=continuations)
        for result in self._unfinished_results():result['continuation']=pending_mapping[str(result['continuation'])]
        self.torch_rng=torch.Generator(device=self.device);self.torch_rng.set_state(torch_state);self.sampler.generator=self.torch_rng
        for k in ('z','context','prior_context','start_z','change_history'):
            if getattr(self,k) is not None:setattr(self,k,getattr(self,k).to(self.device))
        for k in ('memory','inference_memory','context_memory','start_memory'):
            if getattr(self,k):setattr(self,k,[m.to(self.device) for m in getattr(self,k)])
        if self.cached:
            self.cached['z']=self.cached['z'].to(self.device);self.cached['noise']=self.cached['noise'].to(self.device);self.cached['memory']=[m.to(self.device) for m in self.cached['memory']];self.cached['views']={mapping[int(h)]:v for h,v in self.cached['views'].items()}
            for k in ('mode','eligible'):self.cached[k]=self.cached[k].to(self.device)
    def fork(self):
        other=copy.copy(self);other.engine=Mechanics();other.sampler=ConfigurationSampler(self.model,self.vocab,other.engine);other.rng=np.random.default_rng();other.restore(self.snapshot());return other
    def condition(self,events,our_command=None,private_events=None):
        other=self.fork();other.update(events,our_command,private_events);return other
    def forecast(self,our_command,samples=16):
        other=self.fork()
        try:return other._forecast(our_command,samples)
        finally:other.close()
    def _forecast(self,our_command,samples=16):
        # Current action probabilities are evaluated before applying our
        # hypothetical simultaneous choice; only subsequent observations change.
        prepared=self._prepare();out=[]
        for i in self.rng.choice(self.n,samples,p=self.weights):
            actions=prepared['actions'][i];p=prepared['p'][i,:len(actions)];a=actions[int(self.rng.choice(len(actions),p=p/p.sum()))];commands=[[self.observer,our_command]]
            if a.get('command'):commands.append([self.side,a['command']])
            result=self.engine.call('transition',handle=int(self.handles[i]),commands=commands,branches=self.branches,keep=1)
            if result['outcomes']:
                o=result['outcomes'][0];out.append({'probability':result['likelihood']/samples,'opponent_action':a['key'],'events':o['events']});self.engine.call('release',handles=[o['handle']])
        return out
    def new_game(self,observer_team,preview,game_id,initial_events=None,opponent_id=None,observer=None):
        same_player=opponent_id is None or opponent_id==self.opponent_id
        if not same_player:self.z=None;self.opponent_id=opponent_id
        self.past_windows=self.past_windows+(self.applied_windows or 0) if same_player else 0
        indices=systematic(self.weights,self.n,self.rng)
        indices=self.rng.permutation(indices)
        if self.z is not None:self.z=self.z[torch.as_tensor(indices,device=self.device)]
        # Training separates games in the causal encoder; only strategy paths
        # carry information across the support/query game boundary.
        self.memory=None;self.inference_memory=None;self.position=0
        self.change_history=torch.zeros_like(self.change_history)
        self.start_memory=None;self.start_z=self.z;self.start_position=0
        self.engine.call('release_continuations',handles=[r['continuation'] for r in self._unfinished_results()])
        old=self._state_handles();self.pending=None;self.own_team=copy.deepcopy(observer_team);self.preview=list(preview);self.game_id=game_id;self.games_seen=self.games_seen+1 if same_player else 1
        if observer is not None:self.observer=observer;self.side='p2' if observer=='p1' else 'p1'
        self.applied_windows=0;self.journal=[];self.facts=[];self.known=[{} for _ in preview];self.mismatch=None;self.cutoff=0;self.revision=0;self.previous=np.zeros((self.n,4),np.int64);self.coverage={};self.cost={}
        self.initial_events=[f'|poke|{self.side}|{s}|' for s in preview]+[f'|poke|{self.observer}|{m["species"]}|' for m in observer_team];self.events=list(self.initial_events)
        if initial_events is not None:self.initial_events=list(initial_events);self.events=list(initial_events)
        self._encode_inference(self.events);self.prior_context=self.context.clone();self._initialize()
        self.engine.call('release',handles=list(map(int,old-set(self.handles))))
    def close(self):self.engine.close()
