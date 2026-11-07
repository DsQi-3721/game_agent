"""Prepare, run and evaluate action-only SFT and real-game PPO as explicit stages."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import gzip
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def sft(cfg, resume=None):
    from arena.s1.train_hf import train
    from arena.s1.action_training import Journal
    root = Path(cfg['root']); params = dict(cfg['sft'])
    if (root/'sft/checkpoints.jsonl').exists() and not resume:
        raise FileExistsError('SFT already has checkpoints; use --resume to continue')
    params.pop('devices')
    params.pop('battle_eval_every')
    params['lr'] = params.pop('learning_rate')
    journal = Journal(root/'sft',cfg,'sft',resume=bool(resume)) if int(os.getenv('RANK','0')) == 0 else None
    def record(event):
        scalar = {k:v for k,v in event.items() if isinstance(v,(int,float,bool))}
        for group, values in event.get('valid_action_metrics',{}).items():
            scalar.update({f'valid/{group}/{k}':v for k,v in values.items()})
        journal.log(scalar)
        if 'valid_ce' in event or (event.get('step',0) and event['step'] % 50 == 0): journal.plot()
    try:
        return train(str(root/'data/train.jsonl'),str(root/'data/valid.jsonl'),cfg['base_model'],str(root/'sft'),
            init_adapter=cfg['init_adapter'],max_tokens=cfg['max_tokens'],seed=cfg['seed'],eval_n=0,
            until_converged=True,action_only=True,permute_each_visit=True,max_epochs=params['epochs'],
            resume_from=resume,event_callback=record if journal else None,family_mass=cfg['data']['source_mass'],
            log=lambda s: print(s,flush=True),**params)
    finally:
        if journal: journal.close()


def load_battle_policies(cfg, adapter, reference=None):
    import torch
    from arena.s1.ppo import ActionPolicy
    from arena.s1.ppo_battles import PolicyBatcher
    os.environ['ARENA_SHOWDOWN_URL'] = cfg['ppo']['showdown_url']
    torch.manual_seed(cfg['seed'])
    policy = ActionPolicy.load(cfg['base_model'],adapter,cfg['ppo']['device'],cfg['max_tokens'],
                              temperature=cfg['ppo']['temperature'],batch_size=cfg['ppo']['microbatch_size'])
    warm = reference or Path(cfg['root'])/'sft'
    frozen = ActionPolicy.load(cfg['base_model'],str(warm),cfg['ppo']['device'],cfg['max_tokens'],trainable=False,
                              temperature=cfg['ppo']['temperature'],batch_size=cfg['ppo']['microbatch_size'])
    params = cfg['ppo']
    batches = [PolicyBatcher(p,params['inference_batch_size'],params['inference_wait_ms']) for p in (policy,frozen)]
    return policy, batches


def ppo(cfg, resume=None):
    from arena.s1.ppo import PPOTrainer, BattleStop
    from arena.s1.ppo_battles import collect_round, evaluate_battles
    from arena.s1.action_training import Journal
    root = Path(cfg['root']); out = root/'ppo'; out.mkdir(parents=True,exist_ok=True)
    if not (root/'sft/s1_train_meta.json').exists():
        raise FileNotFoundError('PPO requires a completed, selected SFT adapter')
    if (out/'status.json').exists() and not resume:
        raise FileExistsError('PPO already has a run; use --resume with its checkpoint')
    policy, (actor,frozen) = load_battle_policies(cfg,resume or str(root/'sft'))
    p = cfg['ppo']; trainer = PPOTrainer(policy,p,cfg['seed'])
    stop = BattleStop(p['actor_lr'],p['min_actor_lr'],p['min_rounds'],p['plateau_checks'],p['min_score_gain'])
    round_id, env_steps, games, best_adapter = 0,0,0,str(root/'sft')
    if resume:
        state = trainer.resume(resume)
        round_id, env_steps, games = state['round'],state['env_steps'],state['games']
        best_adapter = state['best_adapter']; stop = BattleStop(**state['stop'])
    journal = Journal(out,cfg,'ppo',resume=bool(resume))
    def dev(round_id):
        result = asyncio.run(evaluate_battles(actor,frozen,cfg,out/'dev'/f'round_{round_id:04d}'))
        score = sum(cfg['evaluation']['score_weights'][k]*r['score'] for k,r in result.items())
        journal.log({'round':round_id,'step':trainer.step,'env_steps':env_steps,'dev_score':score,
                     **{'dev_'+k:v['win_rate'] for k,v in result.items()}})
        journal.plot()
        return stop.observe(score,round_id)
    reason = 'round_limit'
    try:
        if not resume: dev(0)
        for round_id in range(round_id+1,p['max_rounds']+1):
            directory = out/'rollouts'/f'round_{round_id:04d}'
            write(out/'status.json',{'state':'collecting','round':round_id,'env_steps':env_steps})
            started = time.monotonic()
            episodes,rejected = asyncio.run(collect_round(actor,frozen,cfg,round_id,directory))
            count = sum(len(e['steps']) for e in episodes); env_steps += count; games += len(episodes)
            with gzip.open(directory/'episodes.jsonl.gz','wt') as f:
                for e in episodes+rejected: f.write(json.dumps(e)+'\n')
            collection_time = time.monotonic()-started
            update = trainer.update(episodes)
            event = {'round':round_id,'env_steps':env_steps,'games':games,**update,
                'collection_seconds':collection_time,'update_seconds':time.monotonic()-started-collection_time,
                'decisions_per_s':count/collection_time,'invalid_episodes':len(rejected)}
            for opponent in sorted(set(e['opponent'] for e in episodes)):
                subset = [e for e in episodes if e['opponent']==opponent]
                event['rollout_win_'+opponent] = sum(e['reward']==1 for e in subset)/len(subset)
                event['rollout_games_'+opponent] = len(subset)
            journal.log(event)
            checkpoint = out/'checkpoints'/f'round_{round_id:04d}'
            action = None
            if round_id % p['eval_every'] == 0 or round_id == p['max_rounds']:
                action = dev(round_id)
                trainer.optimizer.param_groups[0]['lr'] = stop.learning_rate
                if action['new_best']: best_adapter = str(checkpoint)
            trainer.save(checkpoint,{'round':round_id,'env_steps':env_steps,'games':games,
                                     'stop':asdict(stop),'best_adapter':best_adapter})
            write(out/'status.json',{'state':'round_completed','round':round_id,'env_steps':env_steps,
                                     'checkpoint':str(checkpoint),'best_adapter':best_adapter})
            if action and action['plateau']:
                reason = 'development_plateau'; break
        write(out/'result.json',{'stopping_reason':reason,'round':round_id,'games':games,'env_steps':env_steps,
            'best_adapter':best_adapter,'best_round':stop.best_round,'best_dev_score':stop.best_score,
            'formal_evaluation':'pending explicit final command'})
        write(out/'status.json',{'state':'stopped','reason':reason,'best_adapter':best_adapter})
    finally:
        actor.close(); frozen.close(); journal.close()


def dev(cfg, adapter, out, reference=None):
    from arena.s1.ppo_battles import evaluate_battles
    policy,(actor,frozen) = load_battle_policies(cfg,adapter,reference)
    try: return asyncio.run(evaluate_battles(actor,frozen,cfg,out))
    finally: actor.close(); frozen.close()


def pending_sft_battle(checkpoints, done, every, completed):
    if not checkpoints: return None
    if not done: return checkpoints[0]
    if completed is not None:
        return next((r for r in checkpoints if r['step']==completed['best_step'] and r['step'] not in done),None)
    candidates = [r for r in checkpoints if r['step'] >= max(done)+every]
    return candidates[-1] if candidates else None


def watch_sft(cfg, config_path):
    from arena.s1.action_training import Journal
    root = Path(cfg['root']); out = root/'sft/battles'
    out.mkdir(parents=True,exist_ok=True)
    def records(path):
        if not path.exists(): return []
        return [json.loads(l) for l in path.read_text().splitlines(keepends=True) if l.endswith('\n')]
    points = out/'points.jsonl'
    done = {r['step'] for r in records(points)}
    journal = Journal(out,cfg,'sft-battles',resume=bool(done))
    try:
        while True:
            checkpoints = records(root/'sft/checkpoints.jsonl')
            meta = root/'sft/s1_train_meta.json'
            completed = None
            if meta.exists():
                try: completed = json.loads(meta.read_text())
                except json.JSONDecodeError: pass
            entry = pending_sft_battle(checkpoints,done,cfg['sft']['battle_eval_every'],completed)
            if completed is not None and entry is None and completed['best_step'] not in done:
                entry = {'step':completed['best_step'],'epoch':completed['epochs'],'adapter':str(root/'sft')}
            if entry:
                directory = out/f'step_{entry["step"]:07d}'
                command = [sys.executable,str(Path(__file__).resolve()),'dev','--config',config_path,
                           '--adapter',entry['adapter'],'--reference',cfg['init_adapter'],'--out',str(directory)]
                subprocess.run(command,check=True)
                result = json.loads((directory/'result.json').read_text())
                point = {'step':entry['step'],'epoch':entry['epoch'],'adapter':entry['adapter'],'opponents':result}
                with points.open('a') as f: f.write(json.dumps(point)+'\n')
                journal.log({'step':entry['step'],**{'dev_'+k:r['win_rate'] for k,r in result.items()}})
                journal.plot(); done.add(entry['step'])
            elif completed is not None:
                write(out/'status.json',{'state':'completed','evaluated_steps':sorted(done)})
                break
            else: time.sleep(5)
    finally: journal.close()


def final(cfg, adapter):
    from arena.s1.ppo_battles import evaluate_battles
    import gc
    import torch
    out = Path(cfg['root'])/'final'
    if (out/'comparison.json').exists(): raise FileExistsError(str(out))
    reports = {}
    for name,path in [('warmstart',str(Path(cfg['root'])/'sft')),('ppo',adapter)]:
        policy,(actor,frozen) = load_battle_policies(cfg,path)
        try: reports[name] = asyncio.run(evaluate_battles(actor,frozen,cfg,out/name,final=True))
        finally: actor.close(); frozen.close()
        del policy,actor,frozen; gc.collect(); torch.cuda.empty_cache()
    comparison = {}
    for opponent in ('kakuna','foulplay'):
        base,new = reports['warmstart'][opponent],reports['ppo'][opponent]
        delta = new['win_rate']-base['win_rate']
        lo = delta-((new['win_rate']-new['win_ci95'][0])**2+(base['win_ci95'][1]-base['win_rate'])**2)**.5
        hi = delta+((new['win_ci95'][1]-new['win_rate'])**2+(base['win_rate']-base['win_ci95'][0])**2)**.5
        comparison[opponent] = {'win_difference':delta,'ci95':[lo,hi],
            'latency_mean_ratio':new['latency_mean']/base['latency_mean'],
            'latency_p90_ratio':new['latency_p90']/base['latency_p90']}
    write(out/'comparison.json',{'arms':reports,'comparison':comparison,
        'improvement_demonstrated':comparison['kakuna']['ci95'][0]>0,
        'latency_passed':all(r['latency_mean_ratio']<=cfg['evaluation']['latency_ratio_limit'] and
                            r['latency_p90_ratio']<=cfg['evaluation']['latency_ratio_limit'] for r in comparison.values())})


def plan(cfg, config_path):
    root = Path(cfg['root'])
    data = json.loads((root/'data/validation.json').read_text())
    if not data['passed']: raise ValueError('Dataset preparation failed')
    if not Path(cfg['base_model'],'config.json').exists() or not Path(cfg['init_adapter'],'adapter_config.json').exists():
        raise FileNotFoundError('Missing base model or initial adapter')
    (root/'protocol.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    commands = {stage:f'{sys.executable} scripts/s1_action_train.py {stage} --config {config_path}' for stage in ('sft','ppo')}
    commands['final'] = f'{sys.executable} scripts/s1_action_train.py final --adapter <best_adapter> --config {config_path}'
    write(root/'preparation.json',{'state':'prepared','formal_training_started':False,'commands':commands,
        'sft_effective_batch':len(cfg['sft']['devices'])*cfg['sft']['batch_size']*cfg['sft']['grad_accum'],
        'data':{k:v for k,v in data.items() if k not in ('dev_groups','dev_rosters')},
        'verification':{name:json.loads((root/'verification'/name/'result.json').read_text())
                        for name in ('gpu','live') if (root/'verification'/name/'result.json').exists()}})
    print(json.dumps(commands,indent=2))


def run_training(cfg, config_path):
    """Run supervised warmup and PPO in sequence, preserving each stage's log."""
    root = Path(cfg['root']); root.mkdir(parents=True,exist_ok=True)
    status = {'state':'running','pid':os.getpid(),'started_at':time.time(),'completed_stages':[]}
    for stage in ('sft','ppo'):
        status.update(stage=stage,stage_started_at=time.time())
        write(root/'training_status.json',status)
        command = [sys.executable,'-u',str(Path(__file__).resolve()),stage,'--config',str(config_path)]
        with (root/f'{stage}.log').open('a') as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(state='failed',returncode=result.returncode,finished_at=time.time())
            write(root/'training_status.json',status)
            raise SystemExit(result.returncode)
        status['completed_stages'].append(stage)
    status.update(state='completed',finished_at=time.time())
    write(root/'training_status.json',status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','run','sft','ppo','dev','final','watch-sft'])
    parser.add_argument('--config',default='configs/training/s1_action_ppo.yaml')
    parser.add_argument('--resume'); parser.add_argument('--adapter'); parser.add_argument('--out')
    parser.add_argument('--reference')
    args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text())
    os.chdir(ROOT)
    if args.stage=='plan': plan(cfg,args.config)
    elif args.stage=='run': run_training(cfg,args.config)
    elif args.stage=='sft':
        if 'LOCAL_RANK' not in os.environ:
            out = Path(cfg['root'])/'sft'
            if (out/'checkpoints.jsonl').exists() and not args.resume:
                raise FileExistsError('SFT already has checkpoints; use --resume to continue')
            if args.resume: (out/'s1_train_meta.json').unlink(missing_ok=True)
            env = {**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,cfg['sft']['devices'])),
                   'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'true','RAYON_NUM_THREADS':'4'}
            command = [sys.executable,'-m','torch.distributed.run','--standalone',
                f'--nproc_per_node={len(cfg["sft"]["devices"])}',str(Path(__file__).resolve()),*sys.argv[1:]]
            watcher = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'watch-sft','--config',args.config],
                                       start_new_session=True)
            try:
                subprocess.run(command,env=env,check=True)
                if watcher.wait(): raise RuntimeError('SFT battle validation failed; inspect sft/battles')
            finally:
                if watcher.poll() is None:
                    os.killpg(watcher.pid,signal.SIGINT)
                    try: watcher.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(watcher.pid,signal.SIGKILL); watcher.wait()
        else: sft(cfg,args.resume)
    elif args.stage=='ppo': ppo(cfg,args.resume)
    elif args.stage=='watch-sft': watch_sft(cfg,args.config)
    else:
        if not args.adapter: parser.error('--adapter is required for evaluation')
        if args.stage=='dev':
            if args.reference: cfg['ppo']['parallel_matches']=2
            dev(cfg,args.adapter,args.out or str(Path(cfg['root'])/'manual_dev'),args.reference)
        else: final(cfg,args.adapter)


if __name__=='__main__':
    main()
