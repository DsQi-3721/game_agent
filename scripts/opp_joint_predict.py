"""Read one legal observer history JSON and write one joint opponent belief JSON."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from arena.opponent.joint import JointOpponentModel


def main(args):
    torch.set_num_threads(2);data=json.loads(args.input.read_text());initial=data['initial'];directory=args.directory
    if initial.get('format','gen9ou')!='gen9ou':raise ValueError('This checkpoint and catalogue are trained for gen9ou')
    model=JointOpponentModel(args.model or directory/'training/adaptation/model.pt',directory/'catalog.json',initial['observer_team'],initial['opponent_preview'],observer=initial['observer'],game_id=initial['game_id'],initial_events=initial['public_events'],particles=args.particles,device=args.device,seed=initial.get('seed',83),opponent_id=initial.get('opponent_id'))
    try:
        for window in data['windows']:
            belief=model.update(window['public_events'],window.get('our_submitted_action'),private_events=window.get('observer_private_events'),event_id=window.get('event_id'),sequence=window.get('sequence'))
            if not belief['inference_cost'].get('observation_applied',False):break
        result=json.dumps(model.dossier(),ensure_ascii=False,indent=2,allow_nan=False)
        if args.output:args.output.write_text(result)
        else:print(result)
    finally:model.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--directory',type=Path,default=ROOT/'artifacts/opponent_joint');p.add_argument('--model',type=Path);p.add_argument('--device',default='cuda:1');p.add_argument('--particles',type=int,default=1024);main(p.parse_args())
