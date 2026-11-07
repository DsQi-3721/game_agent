"""Every-decision Top-k evaluation on locked games with complete ground truth."""
import argparse,json,pickle,subprocess,sys,tempfile,time
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from arena.opponent.joint.network import JointNetwork
from arena.opponent.joint.tokens import Vocabulary
from arena.opponent.joint.training_objective import TrainingTask,to_device
from arena.opponent.joint.training_evaluation import probability_temperature
from arena.opponent.joint.training_data import EpisodeDataset,collate_episodes


class TopK:
    def __init__(self):self.count=0;self.hits=[0,0,0]
    def add(self,values,target):
        order=values.argsort(dim=-1,descending=True,stable=True)[:,:5]
        self.add_order(order,target)
    def add_order(self,order,target):
        self.count+=len(target)
        for i,k in enumerate((1,3,5)):
            self.hits[i]+=int((order[:,:k]==target[:,None]).any(-1).sum())
    def result(self):
        return dict(count=self.count,hits=self.hits,**{f'top{k}':h/self.count for k,h in zip((1,3,5),self.hits)}) if self.count else {'count':0,'hits':self.hits}


def config_step(model,context,token,position,cache):
    """The same config_logits decoder, using cached keys/values for one token."""
    x=model.embedding(token)[:,None]+model.position.weight[position]+context[:,None]
    for i,layer in enumerate(model.layers):
        q,k,v=layer.qkv(layer.norm(x)).chunk(3,-1)
        q,k,v=[a.reshape(len(x),1,layer.heads,-1).transpose(1,2) for a in (q,k,v)]
        if position==0:
            shape=(len(x),layer.heads,126,k.shape[-1]);cache.append((k.new_empty(shape),v.new_empty(shape)))
        keys,values=cache[i];keys[:,:,position:position+1]=k;values[:,:,position:position+1]=v
        y=F.scaled_dot_product_attention(q,keys[:,:,:position+1],values[:,:,:position+1])
        x=x+layer.output(y.transpose(1,2).flatten(2));x=x+layer.ff(layer.ffnorm(x))
    return model.norm(x)[:,0]


def configuration_marginals(task,context,preview,samples):
    # Only public context and preview enter this function. Hidden labels and
    # private move-slot identities are never inputs to the decoder.
    model=task.model;b=len(context);width=len(task.vocab.tokens)
    context=model.context_projection(context).repeat_interleave(samples,0)
    preview=preview.repeat_interleave(samples,0);token=torch.full((len(context),),2,device=context.device,dtype=torch.long)
    result={name:torch.zeros(b,6,width,device=context.device) for name in ('item','ability','tera','moves')}
    cache=[];used=torch.zeros(len(context),width,device=context.device,dtype=torch.bool)
    allowed=[mask.nonzero().flatten() for mask in task.config_allowed[:21]]
    output=[model.configuration.weight[ids] for ids in allowed]
    for position in range(126):
        slot,field=divmod(position,21);decoded=config_step(model,context,token,position,cache)
        if field==0:token=preview[:,slot];used.zero_();continue
        ids=allowed[field];logits=F.linear(decoded,output[field]).float()
        if field>=17:logits=logits.masked_fill(used[:,ids],-torch.inf)
        p=logits.softmax(-1)
        if field in (1,2,4):result[{1:'item',2:'ability',4:'tera'}[field]][:,slot,ids]=p.reshape(b,samples,-1).mean(1)
        if field>=17:result['moves'][:,slot,ids]+=p.reshape(b,samples,-1).mean(1)
        token=ids[torch.multinomial(p,1)[:,0]]
        if field>=17:used.scatter_(1,token[:,None],True)
    # With no duplicate moves, summed slot probabilities estimate inclusion
    # probabilities. Known moves are removed only when scoring hidden moves.
    return result


def hidden_mask(raw,vocab):
    """Exclude explicit public disclosures/mutations, including effect sources."""
    fields=torch.zeros(len(vocab.tokens),dtype=torch.long)
    for token,name in enumerate(vocab.tokens):
        for prefix,field in (('item:',1),('ability:',2),('type:',4)):
            if name.startswith(prefix):fields[token]=field
    unknown=raw['config_unknown'].unflatten(-1,(6,21)).clone()
    events=raw['public_events'];own=(raw['public_numeric'][...,0]==1)&(raw['segments']==1)[...,None]
    holder=(events[...,2,None]==raw['preview'][:,:,None,:])&own[...,None]
    disclosed=fields[events[...,1]]
    for field in (1,2,4):
        seen=(holder&(disclosed==field)[...,None]).any(2).int().cumsum(1)>0
        unknown[...,field]&=~seen
    return unknown


def alias_masks(base,vocab):
    """Public-protocol scoring corrections shared with the native baselines."""
    directory=base/'baselines';audit=directory/'preview_alias_audit.json';result={}
    if not audit.exists():return result
    affected=json.loads(audit.read_text())['affected']
    selected={(a['game'],a['side']) for a in affected}
    for group in sorted({a['group'] for a in affected}):
        for game in pickle.loads((directory/'cases'/f'{group}.pkl').read_bytes()):
            for side,case in game['cases'].items():
                if (game['id'],side) in selected:
                    preview=tuple(vocab.token('species',m['species']) for m in case['truth'])
                    result[(game['id'],preview)]=torch.from_numpy(case['hidden'])
    return result


def apply_alias_masks(raw,unknown,corrections):
    for i,game in enumerate(raw['games']):
        query=(raw['segments'][i]==1)&raw['score'][i]&(raw['target'][i]>=0)
        indices=query.nonzero().flatten()
        key=(game,tuple(raw['preview'][i,indices[0]].tolist()))
        if key in corrections:unknown[i,indices]=corrections[key][raw['position'][i,indices]]
    return unknown


@torch.inference_mode()
def evaluate_shard(args):
    torch.set_num_threads(3);base=args.directory;out=base/'training'
    checkpoint=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    model=JointNetwork.load(out/'model.pt',args.device);task=TrainingTask(model,Vocabulary(base/'catalog.json')).to(args.device).eval()
    temperature=json.loads((out/'calibration_controlled.json').read_text())['action']['temperature']
    dataset=EpisodeDataset(out/'data','controlled','test',limit=args.games,length=100000)
    dataset.rows=dataset.rows[args.shard::args.shards]
    data=DataLoader(dataset,batch_size=args.batch,collate_fn=collate_episodes,num_workers=2,pin_memory=True)
    corrections=alias_masks(base,task.vocab)
    metrics=defaultdict(TopK);phases=defaultdict(TopK);started=time.time();records=0;decisions=0;hidden_move_sets=0;excluded=defaultdict(int)
    for index,raw in enumerate(data):
        unknown=apply_alias_masks(raw,hidden_mask(raw,task.vocab),corrections).to(args.device)
        batch=to_device(raw,args.device);records+=len(batch['games']);torch.manual_seed(701+index*args.shards+args.shard)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            encoded=model(batch,samples=4);logp,selected=task.probabilities(batch,encoded)
            probabilities=probability_temperature(logp,temperature);target=batch['target'][selected]
            metrics['action'].add(probabilities,target)
            phase=batch['phase'][selected]
            for value,name in enumerate(('preview','turn','forced_switch')):
                keep=phase==value
                if keep.any():phases[name].add(probabilities[keep],target[keep])
            public,_=model.encode(batch['public_events'],batch['public_numeric'],batch['public_state'],channel=1,position=batch['position'],segments=batch['segments'])
            points=selected.nonzero();decisions+=len(points)
            old_unknown=batch['config_unknown'][selected].reshape(-1,6,21)
            for name,field in (('item',1),('ability',2),('tera',4)):
                excluded[name]+=int((old_unknown[...,field]&~unknown[selected][...,field]).sum())
            torch.manual_seed(1701+index*args.shards+args.shard)
            for chunk in points.split(args.context_batch):
                ii,jj=chunk.T
                marginals=configuration_marginals(task,public[ii,jj],batch['preview'][ii,jj],args.samples)
                truth=batch['configuration'][ii,jj].reshape(-1,6,21);hidden_config=unknown[ii,jj]
                for name,field in (('item',1),('ability',2),('tera',4)):
                    keep=hidden_config[:,:,field]&(truth[:,:,field]>0)
                    metrics[name].add(marginals[name][keep],truth[:,:,field][keep])
                move_truth=truth[:,:,17:];hidden=hidden_config[:,:,17:]&(move_truth>0)
                revealed=((~hidden_config[:,:,17:])&(move_truth>0)).nonzero()
                m=marginals['moves'];m[:,:,~task.config_allowed[17]]=-torch.inf
                if len(revealed):
                    i,j,k=revealed.T;m[i,j,move_truth[i,j,k]]=-torch.inf
                order=m.argsort(dim=-1,descending=True,stable=True)[...,:5]
                metrics['moves'].add_order(order[:,:,None,:].expand(-1,-1,4,-1)[hidden],move_truth[hidden])
                hidden_move_sets+=int(hidden.any(-1).sum())
        print(json.dumps({'shard':args.shard,'player_games':records,'decision_points':decisions,'seconds':round(time.time()-started,1)}),flush=True)
    result={'checkpoint_step':checkpoint['steps'],'checkpoint_epoch':checkpoint['epoch'],'player_games':records,'decision_points':decisions,
            'metrics':{k:v.result() for k,v in metrics.items()},'action_phases':{k:v.result() for k,v in phases.items()},
            'seconds':time.time()-started,'device':args.device,'hidden_move_sets':hidden_move_sets,
            'excluded_public_reveals_missing_from_old_mask':dict(excluded)}
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps({'shard':args.shard,'metrics':result['metrics']}),flush=True)


def main(args):
    if args.shard is not None:return evaluate_shard(args)
    destination=args.directory/'training'/'topk.json';jobs=[];started=time.time()
    with tempfile.TemporaryDirectory(prefix='opponent_topk_') as temporary:
        for gpu in range(args.shards):
            output=Path(temporary)/f'{gpu}.json';log=(args.directory/'training'/f'topk_gpu{gpu}.log').open('w')
            command=[sys.executable,__file__,'--directory',str(args.directory),'--shard',str(gpu),'--shards',str(args.shards),'--device',f'cuda:{gpu}','--games',str(args.games),'--batch',str(args.batch),'--context-batch',str(args.context_batch),'--samples',str(args.samples),'--output',str(output)]
            jobs.append((output,subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT),log))
        results=[]
        for output,job,log in jobs:
            code=job.wait();log.close()
            if code:raise RuntimeError(f'Evaluation failed; see {log.name}')
            results.append(json.loads(output.read_text()))
        result={k:results[0][k] for k in ('checkpoint_step','checkpoint_epoch')}
        for key in ('player_games','decision_points','hidden_move_sets'):result[key]=sum(r[key] for r in results)
        for section in ('metrics','action_phases'):
            result[section]={}
            for name in results[0][section]:
                total=TopK();total.count=sum(r[section][name]['count'] for r in results)
                total.hits=[sum(r[section][name]['hits'][i] for r in results) for i in range(3)]
                result[section][name]=total.result();del result[section][name]['hits']
        result.update(configuration_samples=args.samples,seconds=time.time()-started,gpus=args.shards,
            dataset='Locked locally generated test games with complete ground truth; every decision of each selected player-game, with no 64-step truncation.',
            configuration_scope='All six Pokemon at every pre-action decision, including preview and forced switches. Only still-hidden initial attributes are scored. Public disclosures/mutations, including effect-source messages, remove fields from subsequent scoring.',
            prediction_scope='Public-history neural configuration prediction with sampled prefixes, typed masks and no duplicate moves; no true hidden prefixes or joint particle filtering.',
            action_scope='Conditional action-head diagnostic given the actor own true configuration and legal action features; actual submitted actions are labels. Preview scores the lead.',
            history_scope='Full query games; at most 32 prior-game support steps. Future query events are causally masked.',
            move_metric='Micro recall@k over all true unrevealed moves, using one shared ranked list per Pokemon and excluding revealed moves.',
            move_top1_upper_bound=result['hidden_move_sets']/result['metrics']['moves']['count'],
            excluded_public_reveals_missing_from_old_mask={name:sum(r['excluded_public_reveals_missing_from_old_mask'][name] for r in results) for name in ('item','ability','tera')})
        destination.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=ROOT/'artifacts/opponent_joint');p.add_argument('--shard',type=int);p.add_argument('--shards',type=int,default=2);p.add_argument('--device',default='cuda:0');p.add_argument('--games',type=int,default=8192);p.add_argument('--batch',type=int,default=16);p.add_argument('--context-batch',type=int,default=256);p.add_argument('--samples',type=int,default=64);p.add_argument('--output',type=Path);main(p.parse_args())
