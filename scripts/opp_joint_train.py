"""Continue predictive opponent-model training for full mixed-data epochs."""
import argparse,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from arena.opponent.joint.tokens import Vocabulary
from arena.opponent.joint.network import JointNetwork
from arena.opponent.joint.training_data import EpisodeDataset,collate_episodes
from arena.opponent.joint.training_objective import TrainingTask,to_device
from arena.opponent.joint.metrics import Scores


def loader(dataset,batch,workers,sampler=None):
    return DataLoader(dataset,batch_size=batch,sampler=sampler,num_workers=workers,
                      collate_fn=collate_episodes,pin_memory=True)


@torch.inference_mode()
def validate(task,data,device):
    task.eval();scores=Scores();oracle=Scores();oracle_top1_sum=0.;auxiliary={};batches=0
    with torch.random.fork_rng(devices=[torch.device(device)]):
        torch.manual_seed(505)
        for batch in data:
            batch=to_device(batch,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=task.model(batch,samples=4);logp,selected=task.probabilities(batch,out)
                _,parts=task(batch,samples=4)
            scores.add_many(logp.exp().cpu().numpy(),batch['target'][selected].cpu().numpy())
            if batch['source']=='controlled':
                valid=batch['teacher_valid'][selected];p=batch['teacher'][selected][valid].cpu().numpy();y=batch['target'][selected][valid].cpu().numpy()
                oracle.add_many(p,y);oracle_top1_sum+=p.max(-1).sum()
            for key,value in parts.items():
                if key not in ('nll','decisions'):auxiliary[key]=auxiliary.get(key,0.)+float(value)
            batches+=1
    task.train();result=scores.result();result['decisions']=result.pop('count')
    result['auxiliary_loss']={key:value/batches for key,value in auxiliary.items()}
    if oracle.n:
        result['oracle_reference']=oracle.result();result['oracle_reference']['expected_top1']=float(oracle_top1_sum/oracle.n)
        result['oracle_reference']['information']='true generator policy, including hidden strategy parameters; not an attainable guarantee under the model information boundary'
    return result


def run(args):
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    device=f'cuda:{local}';torch.cuda.set_device(local);torch.set_num_threads(3)
    if world>1:dist.init_process_group('nccl',device_id=torch.device(device))
    torch.manual_seed(101+rank);np.random.seed(101+rank);random.seed(101+rank)
    base=args.directory;output=base/'training';vocab=Vocabulary(base/'catalog.json');sources=['human','controlled']
    saved=torch.load(output/'state.pt',map_location='cpu',weights_only=False) if (output/'state.pt').exists() else None
    model=JointNetwork.load(output/'model.pt',device);task=TrainingTask(model,vocab).to(device)
    optimizer=torch.optim.AdamW(task.parameters(),lr=args.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=2,threshold=args.min_delta,threshold_mode='abs',min_lr=args.min_lr)
    if saved:
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        if args.resume_lr is not None:
            for group in optimizer.param_groups:group['lr']=args.resume_lr
        initial=saved['initial'];step=saved['step'];start_epoch=saved['epoch'];start_cursor=saved['cursor'];counts=saved['counts'][rank];games=saved['games'][rank]
        best=saved['best'];best_step=saved['best_step'];significant_best=saved['significant_best'];stale=saved['stale'];history=saved['history'];elapsed=saved['seconds']
        torch.set_rng_state(saved['rng'][rank]['cpu']);torch.cuda.set_rng_state(saved['rng'][rank]['cuda']);np.random.set_state(saved['rng'][rank]['numpy']);random.setstate(saved['rng'][rank]['python'])
    else:
        initial=json.loads((output/'training.json').read_text());step=initial['steps'];start_epoch=start_cursor=0;counts={s:0 for s in sources};games={s:0 for s in sources}
        best=initial['best_validation_selection_score'];best_step=step;significant_best=best;stale=0;history=list(initial['validation_history']);elapsed=0.
    origin=initial['steps'];started=time.time()
    datasets={s:EpisodeDataset(output/'data',s,'train',training=True) for s in sources}
    samplers={s:DistributedSampler(datasets[s],num_replicas=world,rank=rank,shuffle=True,seed=2028) for s in sources}
    batches={s:math.ceil(len(samplers[s])/args.batch) for s in sources}
    schedule=[s for _,s in sorted(((j+.5)/n,s) for s,n in batches.items() for j in range(n))]
    # Traverse both datasets once while retaining a 2:1 task balance by
    # default; an explicit weight supports targeted optimization experiments.
    source_weight={'human':1.,'controlled':args.controlled_weight if args.controlled_weight is not None else batches['human']/(2*batches['controlled'])}
    validation_sources=[sources[rank]] if world==2 else sources
    validations={s:loader(EpisodeDataset(output/'data',s,'validation',limit=args.validation_games),32,0) for s in validation_sources}
    wrapped=DDP(task,device_ids=[local],find_unused_parameters=True) if world>1 else task
    meters={s:{} for s in sources};validation_every=math.ceil(len(schedule)/4);epoch_fraction=float(start_epoch);reason='running'

    def gather(value):
        if world==1:return [value]
        values=[None]*world;dist.all_gather_object(values,value);return values

    def emit(row):
        if rank==0:
            print(json.dumps(row),flush=True)
            with (output/'learning_curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

    def checkpoint(epoch,cursor,update_scheduler=True):
        nonlocal best,best_step,significant_best,stale,meters,epoch_fraction
        epoch_fraction=epoch+cursor/len(schedule)
        combined={k:v for r in gather({s:validate(task,validations[s],device) for s in validation_sources}) for k,v in r.items()}
        score=sum(v['nll'] for v in combined.values())
        if update_scheduler and step-origin>=args.warmup:scheduler.step(score)
        if score<significant_best-args.min_delta:significant_best=score;stale=0
        elif update_scheduler:stale+=1
        improved=score<best
        if improved:best=score;best_step=step
        totals=gather({'counts':counts,'games':games,'meters':meters,'rng':{'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate()},'peak_GB':torch.cuda.max_memory_allocated()/1e9})
        training={}
        for source in sources:
            sums={}
            for r in totals:
                for k,v in r['meters'][source].items():sums[k]=sums.get(k,0.)+v
            if sums:
                training[source]={k:v/sums['updates'] for k,v in sums.items() if k not in ('updates','decisions','nll_sum')}
                training[source]['nll']=sums['nll_sum']/sums['decisions'];training[source]['decisions']=int(sums['decisions'])
        seconds=elapsed+time.time()-started;lr=optimizer.param_groups[0]['lr']
        row={'step':step,'epoch':epoch_fraction,'seconds':initial['seconds']+seconds,'lr':lr,'validation':combined,'training_interval':training,'selection_score':score,'stale_validations':stale}
        history.append(row);emit(row)
        if rank==0:
            if improved:model.save(output/'model.pt',steps=step,epoch=epoch_fraction,validation=combined,selection_score=score)
            next_epoch,next_cursor=(epoch+1,0) if cursor==len(schedule) else (epoch,cursor)
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'initial':initial,
                        'step':step,'epoch':next_epoch,'cursor':next_cursor,'counts':[r['counts'] for r in totals],'games':[r['games'] for r in totals],
                        'rng':[r['rng'] for r in totals],'best':best,'best_step':best_step,'significant_best':significant_best,'stale':stale,'history':history,'seconds':seconds},output/'state.pt')
            result={'steps':step,'seconds':initial['seconds']+seconds,'continuation_seconds':seconds,'gpus':world,'batch_per_gpu':args.batch,'samples':args.samples,
                    'parameters':sum(p.numel() for p in model.parameters()),'best_checkpoint_step':best_step,'epochs_completed':epoch_fraction,'planned_epochs':args.epochs or None,
                    'optimizer':'AdamW; warmup then validation-driven ReduceLROnPlateau; gradient clipping 1',
                    'optimizer_config':{'initial_lr':args.lr,'current_lr':lr,'min_lr':args.min_lr,'warmup_steps':args.warmup,'weight_decay':.01,'seed':101},
                    'resume_note':'The original 3600-step run saved weights only; optimizer was initialized once at continuation start. state.pt now preserves optimizer, scheduler, cursor and RNG.',
                    'epoch_definition':'one complete pass over each source; long games use one random query crop per epoch',
                    'source_loss_weights':source_weight,'training_dataset_player_games':{s:len(datasets[s]) for s in sources},'best_validation_selection_score':best,
                    'validation_history':history,'history':initial['history'],'supervision':initial['supervision'],'peak_allocated_GB_by_rank':[r['peak_GB'] for r in totals],
                    'status':reason,'initial_training_steps':origin,'min_delta':args.min_delta}
            result['scored_decisions']={s:initial['scored_decisions'][s]+sum(r['counts'][s] for r in totals) for s in sources}
            result['player_game_samples']={s:initial['player_game_samples'][s]+sum(r['games'][s] for r in totals) for s in sources}
            (output/'training.json').write_text(json.dumps(result,indent=2))
            (output/'progress.json').write_text(json.dumps({k:result[k] for k in ('status','steps','epochs_completed','planned_epochs','scored_decisions','player_game_samples','best_checkpoint_step','optimizer_config')},indent=2))
            from opp_joint_plot import plot
            plot(output)
        if world>1:dist.barrier()
        meters={s:{} for s in sources}

    if not saved:
        emit({'event':'continuation_start','step':step,'epochs':args.epochs,'batch_per_gpu':args.batch,'lr':args.lr,'source_batches_per_epoch':batches,'source_loss_weights':source_weight,'validation_games_per_source':args.validation_games})
        checkpoint(0,0,False)
    else:
        emit({'event':'resume','step':step,'epoch':start_epoch+start_cursor/len(schedule),'planned_epochs':args.epochs or None,'lr':optimizer.param_groups[0]['lr'],'batch_per_gpu':args.batch,'source_loss_weights':source_weight})
        if rank==0:
            for name in ('training.json','progress.json'):
                path=output/name;record=json.loads(path.read_text());record.update(status='running',planned_epochs=args.epochs or None);path.write_text(json.dumps(record,indent=2))
    task.train()
    epoch=start_epoch
    while not args.epochs or epoch<args.epochs:
        cursor=start_cursor if epoch==start_epoch else 0
        seen={s:schedule[:cursor].count(s) for s in sources};iters={}
        for s in sources:
            samplers[s].set_epoch(epoch+1)
            remaining=list(samplers[s])[seen[s]*args.batch:]
            iters[s]=iter(loader(datasets[s],args.batch,args.workers,remaining))
        for index in range(cursor,len(schedule)):
            update_started=time.perf_counter();source=schedule[index];raw=next(iters[source]);data_wait=time.perf_counter()-update_started
            games[source]+=len(raw['games']);batch=to_device(raw,device)
            step+=1;continued=step-origin
            if continued<=args.warmup:
                for group in optimizer.param_groups:group['lr']=args.lr*continued/args.warmup
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=wrapped(batch,samples=args.samples);loss=loss*source_weight[source]
            loss.backward();gradient=torch.nn.utils.clip_grad_norm_(task.parameters(),1.);optimizer.step()
            values={k:float(v) for k,v in metrics.items()};count=int(values.pop('decisions'));counts[source]+=count
            meter=meters[source];meter['updates']=meter.get('updates',0)+1;meter['decisions']=meter.get('decisions',0)+count;meter['nll_sum']=meter.get('nll_sum',0.)+values.pop('nll')*count
            for k,v in values.items():meter[k]=meter.get(k,0.)+v
            if step%50==0:
                emit({'step':step,'epoch':epoch+(index+1)/len(schedule),'source':source,'lr':optimizer.param_groups[0]['lr'],'seconds':initial['seconds']+elapsed+time.time()-started,'gradient_norm':float(gradient),'batch_nll':float(metrics['nll']),'data_wait_seconds':data_wait,'update_seconds':time.perf_counter()-update_started})
            if (index+1)%validation_every==0 or index+1==len(schedule):
                checkpoint(epoch,index+1)
                # Confirm a plateau after LR reductions and a full epoch at
                # the LR floor; an epoch budget is not evidence of convergence.
                floor_trained=len(history)>=5 and all(r.get('lr',args.lr)<=args.min_lr*1.001 for r in history[-5:])
                if epoch_fraction>=args.min_epochs and stale>=8 and floor_trained:
                    reason='validation_plateau';break
        if reason=='validation_plateau':break
        epoch+=1
    if reason=='running':reason='epoch_budget_completed'
    if rank==0:
        for name in ('training.json','progress.json'):
            path=output/name;record=json.loads(path.read_text());record['status']=reason;path.write_text(json.dumps(record,indent=2))
        print(json.dumps({'finished':reason,'step':step,'epochs_completed':epoch_fraction,'best_step':best_step,'best_validation':best}),flush=True)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=ROOT/'artifacts/opponent_joint');p.add_argument('--epochs',type=int,default=0,help='Target epoch count; 0 continues until a validation plateau');p.add_argument('--min-epochs',type=int,default=5);p.add_argument('--batch',type=int,default=256);p.add_argument('--workers',type=int,default=4);p.add_argument('--samples',type=int,default=4);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--resume-lr',type=float,help='Set a new learning rate while preserving Adam moments');p.add_argument('--controlled-weight',type=float);p.add_argument('--min-lr',type=float,default=1e-5);p.add_argument('--warmup',type=int,default=200);p.add_argument('--min-delta',type=float,default=.002);p.add_argument('--validation-games',type=int,default=512);run(p.parse_args())
