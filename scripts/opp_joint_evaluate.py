"""Calibrate on validation, then test the standalone neural training stage."""
import argparse,json,pickle,sys,time
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from arena.opponent.joint.tokens import Vocabulary
from arena.opponent.joint.network import JointNetwork,NetworkConfig
from arena.opponent.joint.training_objective import TrainingTask,to_device
from arena.opponent.joint.training_evaluation import evaluation_loader,evaluate,HumanFrequency


def count_file(job):
    path,moves=job;counts=defaultdict(Counter)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32,columns=['payload']):
        for payload in batch['payload'].to_pylist():
            row=pickle.loads(payload)
            for y,phase,active,preview in zip(row['target'],row['phase'],row['active'],row['preview']):
                if y<0 or phase>=3:continue
                key=int(y) if y<moves else int(preview[y-moves])+moves if y<moves+6 else -1
                counts[(int(phase==0),int(active[0]))][key]+=1;counts[(int(phase==0),-1)][key]+=1
    return dict(counts)


def frequency(base,vocab):
    path=base/'training/frequency.pkl'
    if path.exists():return pickle.loads(path.read_bytes())
    counts=defaultdict(Counter);files=sorted((base/'training/data/human/train').glob('*.parquet'))
    with ProcessPoolExecutor(4) as pool:
        for partial in pool.map(count_file,[(str(p),len(vocab.move_ids)*2) for p in files]):
            for k,v in partial.items():counts[k].update(v)
    path.write_bytes(pickle.dumps(dict(counts),protocol=5));return dict(counts)


@torch.inference_mode()
def calibrate(task,data,device):
    values=[];targets=[];changes=[];labels=[]
    for i,raw in enumerate(data):
        batch=to_device(raw,device);torch.manual_seed(904+i)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=task.model(batch,samples=4);logp,selected=task.probabilities(batch,out)
        # Controlled legal-action widths vary between batches.
        if batch['source']=='controlled':logp=torch.nn.functional.pad(logp,(0,16-logp.shape[-1]),value=-1e9)
        values.append(logp);targets.append(batch['target'][selected]);valid=selected&(batch['change']>=0)
        if valid.any():changes.append(out['change'][valid].float());labels.append(batch['change'][valid])
    def fit(x,y):
        def objective(logtemp):return float(torch.nn.functional.cross_entropy(x/np.exp(logtemp),y))
        opt=minimize_scalar(objective,bounds=(np.log(.3),np.log(4.)),method='bounded',options={'xatol':1e-3})
        return {'temperature':float(np.exp(opt.x)),'before_nll':objective(0.),'after_nll':float(opt.fun),'count':len(y)}
    result={'action':fit(torch.cat(values),torch.cat(targets))}
    if changes:
        x=torch.cat(changes);y=torch.cat(labels);result['change']=fit(x,y)
        probability=1-(x/result['change']['temperature']).softmax(-1)[:,0]
        result['change']['threshold_at_validation_fpr_1pct']=float(torch.quantile(probability[y==0],.99))
    return result


def main(args):
    torch.set_num_threads(3);base=args.directory;output=base/'training';vocab=Vocabulary(base/'catalog.json')
    if args.frequency_only:
        count=frequency(base,vocab);print(json.dumps({'contexts':len(count),'frequency_baseline':'training data only'}),flush=True);return
    checkpoint=torch.load(output/'model.pt',map_location='cpu',weights_only=True)
    model=JointNetwork.load(output/'model.pt',args.device);task=TrainingTask(model,vocab).to(args.device).eval()
    started=time.time();validation=evaluation_loader(output/'data',args.source,'validation',args.calibration_games,args.batch)
    calibration=calibrate(task,validation,args.device)
    (output/f'calibration_{args.source}.json').write_text(json.dumps(calibration,indent=2))
    baseline=HumanFrequency(frequency(base,vocab),len(vocab.move_ids)) if args.source=='human' else json.loads((base/'traditional.json').read_text())
    result=evaluate(task,evaluation_loader(output/'data',args.source,'test',args.games,args.batch),args.device,
                    temperature=calibration['action']['temperature'],change_temperature=calibration.get('change',{}).get('temperature',1.),change_threshold=calibration.get('change',{}).get('threshold_at_validation_fpr_1pct',.5),baseline=baseline)
    print(json.dumps({'source':args.source,'methods':result['methods']}),flush=True)
    ablation=evaluate(task,evaluation_loader(output/'data',args.source,'test',args.ablation_games,args.batch),args.device,
                      temperature=calibration['action']['temperature'],change_temperature=calibration.get('change',{}).get('temperature',1.),change_threshold=calibration.get('change',{}).get('threshold_at_validation_fpr_1pct',.5),ablation=True,attribute_batches=0)
    result['memory_and_sampling_ablation']=ablation
    result.update(source=args.source,checkpoint_step=checkpoint['steps'],calibration=calibration,seconds=time.time()-started,device=args.device)
    (output/f'test_{args.source}.json').write_text(json.dumps(result,indent=2));print(json.dumps({'source':args.source,'seconds':result['seconds'],'memory_comparison':ablation['cross_game_paired']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=ROOT/'artifacts/opponent_joint');p.add_argument('--source',choices=['human','controlled'],default='human');p.add_argument('--device',default='cuda:0');p.add_argument('--games',type=int,default=8192);p.add_argument('--ablation-games',type=int,default=2048);p.add_argument('--calibration-games',type=int,default=1024);p.add_argument('--batch',type=int,default=24);p.add_argument('--frequency-only',action='store_true');main(p.parse_args())
