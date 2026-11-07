"""Collect the specified 400k-game budget with independent player groups."""
import argparse,json,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
BUDGET={'train':240000,'calibration':30000,'test':70000,'supplement':60000}

def run(args):
 out=args.out.resolve(); started=time.time()
 (out/'budget.json').write_text(json.dumps({'requested_games':BUDGET,'records':'gzip JSONL; initial teams, engine seed and commands losslessly reconstruct full state trajectories; player requests and public/private protocol retained','providers':{'rule':'state-conditioned rule policy','search':'two-step abstract minimax','learned':'behavioral clone trained on previous controlled choices','random':'random legal actions','adaptive':'history-conditioned changes'},'scope':'controlled policies; not human performance'},indent=2))
 jobs=[(role,g) for role,n in BUDGET.items() for g in range(n//100)]
 if args.groups:jobs=[(role,g) for role,g in jobs if g<args.groups]
 jobs.sort(key=lambda p:(p[1],list(BUDGET).index(p[0])))
 completed={k:0 for k in BUDGET};decisions=0
 def generate(role,group):
  path=out/'data'/role/f'group-{group:05}.jsonl.gz'
  if path.exists():return {'role':role,'group':group,'games':100,'decisions':0,'existing':True}
  p=subprocess.run(['node',str(ROOT/'scripts/node/opp_joint_generate.js'),str(out),role,str(group),'100'],capture_output=True,text=True)
  if p.returncode:raise RuntimeError(f'{role}/{group}: {p.stderr[-2500:]}')
  return json.loads(p.stdout)
 with ThreadPoolExecutor(args.workers) as executor:
  tasks=[executor.submit(generate,*job) for job in jobs]
  for task in as_completed(tasks):
   row=task.result();completed[row['role']]+=row['games'];decisions+=row['decisions']
   if sum(completed.values())%1000==0:
    status={'games':completed,'new_decisions':decisions,'seconds':time.time()-started}
    (out/'generation-progress.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
 (out/'generation-progress.json').write_text(json.dumps({'games':completed,'new_decisions':decisions,'seconds':time.time()-started,'complete':True},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'artifacts/opponent_joint');p.add_argument('--workers',type=int,default=12);p.add_argument('--groups',type=int,default=0);run(p.parse_args())
