"""Run the controlled full-battle evaluation of persistent HyMem strategy."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import sys

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('baseline', 'strategic')
OPPONENTS = ('kakuna', 'foulplay')
ADAPTER = Path('/data/wangxinqi/game_agent_runs/s1_policy_retraining/round1/convergence/model/checkpoints/step_0008000')
ENGINE = Path('/data/wangxinqi/game_agent_runs/hymem_strategy/native_build/engine')
POOL = Path('/data/wangxinqi/teams/gen9ou_eval12')
DEVELOPMENT_POOL = Path('/data/wangxinqi/game_agent_runs/s1_policy_retraining/teams/ours')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def configuration(out, opponent, arm, team, games, *, preflight=False):
    cfg = yaml.safe_load((ROOT / 'configs/experiments/hymem_strategic_functional_ou.yaml').read_text())
    player = cfg['agents'][0]
    player['name'] = arm
    player['params'].update(student_url='http://127.0.0.1:8117', student_model='s1-policy-step8000',
        mode='flat', history_events=8, ask_hard=False, debias=True, tier0=False,
        strategic_planning=arm == 'strategic', s1_timeout_s=20.0, shadow_rate=0.0,
        lookahead_url=f'http://127.0.0.1:{8106 if arm == "baseline" else 8107}',
        strategy_memory_path=str(out / ('preflight/memory' if preflight else 'memory') / opponent / 'experiences.jsonl'))
    opposing = yaml.safe_load((ROOT / f'configs/experiments/strong_{opponent}_ou.yaml').read_text())['agents'][-1]
    opposing['external'].update(stall_timeout_s=3600, battle_timeout_s=3600)
    if opponent == 'kakuna':
        opposing['external']['env'].update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1')
    cfg['agents'] = [player, opposing]
    cfg.update(name=f'hymem_strategy_{opponent}_{arm}_t{team:02d}', seed=62107 + team)
    cfg['teams'].update(mode='mirror', pool=str(DEVELOPMENT_POOL if preflight else out / 'teams'),
                        per_team=games, offset=team * games)
    cfg['evaluation'].update(n_battles=games, anchors=[opponent])
    return cfg


def prepare(out, games_per_team=2):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        return
    if not (ADAPTER / 'adapter_model.safetensors').exists():
        raise FileNotFoundError(ADAPTER)
    (out / 'teams').mkdir()
    teams = sorted(POOL.glob('*.txt'))
    if len(teams) != 12:
        raise ValueError('Expected the fixed twelve-team evaluation pool')
    for path in teams:
        shutil.copyfile(path, out / 'teams' / path.name)
    services = {
        'student': {'url': 'http://127.0.0.1:8117', 'health': '/health', 'model': 's1-policy-step8000',
            'cwd': str(ROOT), 'env': {'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '2'},
            'command': [sys.executable, '-u', str(ROOT / 'scripts/s1_serve.py'), '--backend', 'hf',
                '--model', '/data/models/Qwen3.5-2B', '--adapter', str(ADAPTER), '--port', '8117',
                '--merge', '--model-id', 's1-policy-step8000', '--batch-size', '8',
                '--temperature', '1.0', '--action-temperature', '1.55']},
        'planner': {'url': 'http://127.0.0.1:8113', 'health': '/v1/models', 'model': 'Qwen3.5-9B',
            'cwd': str(ROOT), 'env': {'CUDA_VISIBLE_DEVICES': '1', 'OMP_NUM_THREADS': '2',
                'VLLM_CACHE_ROOT': str(out / 'cache/vllm'), 'TORCHINDUCTOR_CACHE_DIR': str(out / 'cache/inductor')},
            'command': [str(Path(sys.executable).with_name('vllm')), 'serve', '/data/models/Qwen3.5-9B',
                '--served-model-name', 'Qwen3.5-9B', '--host', '127.0.0.1', '--port', '8113',
                '--max-model-len', '16384', '--max-num-seqs', '4', '--max-num-batched-tokens', '2048',
                '--gpu-memory-utilization', '0.39', '--reasoning-parser', 'qwen3', '--language-model-only']},
    }
    for arm, port in (('baseline', 8106), ('strategic', 8107)):
        services[f'search_{arm}'] = {
            'url': f'http://127.0.0.1:{port}', 'health': '/health', 'cwd': str(ROOT / 'external/foulplay'),
            'env': {'PYTHONPATH': str(ENGINE), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
            'command': [str(ROOT / 'external/foulplay/venv/bin/python'),
                str(ROOT / 'external/foulplay/lookahead_server.py'), '--port', str(port), '--workers', '8']}
    schedule, preflight = [], []
    for opponent in OPPONENTS:
        for team in range(12):
            for arm in (ARMS if team % 2 == 0 else tuple(reversed(ARMS))):
                directory = f'{opponent}/team{team:02d}/{arm}'
                (out / directory).mkdir(parents=True)
                cfg = configuration(out, opponent, arm, team, games_per_team)
                (out / directory / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
                schedule.append(dict(opponent=opponent, arm=arm, team=team, games=games_per_team,
                                     primary_games=games_per_team, directory=directory))
        for arm in ARMS:
            directory = f'preflight/{opponent}/{arm}'
            (out / directory).mkdir(parents=True)
            cfg = configuration(out, opponent, arm, 10, 1, preflight=True)
            (out / directory / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
            preflight.append(dict(opponent=opponent, arm=arm, team=10, games=1, directory=directory))
    expected = games_per_team * len(teams) * len(ARMS) * len(OPPONENTS)
    protocol = {'created_at': datetime.now().astimezone().isoformat(), 'expected_games': expected,
        'expected_recorded_games': expected, 'games_per_team': games_per_team,
        'primary_selection': 'first completed games within each team, opponent and arm; retain every additional game',
        'games_per_arm_opponent': games_per_team * len(teams), 'services': services, 'student_adapter': str(ADAPTER),
        'student_input': 'full legal-action distribution and eight completed decisions, including single-option decisions',
        'student_decode': 'mean forward/reverse probabilities followed by joint argmax, action temperature 1.55',
        'search': {'samples': 8, 'milliseconds_per_sample': 700, 'workers_per_arm': 8,
            'service_isolation': 'one identical process per arm',
            'selection': 'highest mean value with full hypothesis coverage and at least sixteen visits per hypothesis',
            'persistent_trees': True, 'wait_seconds': 2.5, 'waiting_budget_seconds_per_game': 120},
        'strategic_search': {'general_iterations': .75, 'plan_iterations': .25},
        'planner': {'model': 'Qwen3.5-9B', 'configuration': 'configs/hymem/strategic.yaml',
                    'background_only': True},
        'opponent_information': 'same public observations and configuration prior; no personalized dossier supplied',
        'experience': 'starts empty for each opponent; persists across all its strategic games, including supplementary games; preflight stores separate',
        'teams': [path.name for path in teams], 'parallel_pairs': 1,
        'scheduling': 'baseline and strategic run concurrently; alternate opponents within each fixed team block',
        'latency': 'action request received to final accepted action sent; team preview excluded',
        'primary': 'equal-weight win-rate difference across Kakuna and Foul Play',
        'passing': 'primary difference and its 95% lower bound positive, mean and p90 latency ratios <= 1.10, strategic loop complete',
        'runtime_history': 'shared HyMem policy history observes every decision and records the action actually submitted',
        'preflight': preflight}
    write_json(out / 'protocol.json', protocol)
    write_json(out / 'schedule.json', schedule)
    write_json(out / 'status.json', {'state': 'prepared', 'expected_games': expected, 'completed_games': 0})


async def run(out):
    from hymem_strategy_evaluation_results import summarize, write_report, preflight_quality
    protocol = json.loads((out / 'protocol.json').read_text())
    schedule = json.loads((out / 'schedule.json').read_text())
    children, running = [], {}
    phase, finished = 'starting_services', []
    offsets, outcomes, seen_counts = {}, {}, {}

    def status(**extra):
        for item in schedule:
            path = out / item['directory'] / f"s1exp_{item['arm']}.jsonl"
            if not path.exists():
                continue
            with path.open() as file:
                file.seek(offsets.get(str(path), 0))
                while True:
                    position = file.tell(); line = file.readline()
                    if not line or not line.endswith('\n'):
                        file.seek(position); break
                    if '"type": "outcome"' in line:
                        row = json.loads(line)
                        index = seen_counts.get(str(path), 0)
                        seen_counts[str(path)] = index + 1
                        primary = index < item.get('primary_games', item['games'])
                        outcomes[(item['arm'], row['battle_tag'])] = (item['opponent'], bool(row['won']), primary)
                offsets[str(path)] = file.tell()
        counts = {arm: {opponent: {'games': sum(a == arm and o == opponent and primary for (a, _), (o, _, primary) in outcomes.items()),
                                  'wins': sum(a == arm and o == opponent and w and primary for (a, _), (o, w, primary) in outcomes.items())}
                        for opponent in OPPONENTS} for arm in ARMS}
        write_json(out / 'status.json', {'state': phase, 'pid': os.getpid(),
            'updated_at': datetime.now().astimezone().isoformat(), 'expected_games': protocol['expected_games'],
            'expected_recorded_games': sum(item['games'] for item in schedule),
            'completed_games': sum(primary for _, _, primary in outcomes.values()),
            'completed_recorded_games': len(outcomes), 'completed_blocks': len(finished),
            'running': running, 'arms': counts, **extra})

    async def launch(command, cwd, env, logfile):
        with logfile.open('a') as log:
            process = await asyncio.create_subprocess_exec(*command, cwd=cwd,
                env={**os.environ, **env}, stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        children.append(process)
        return process

    async def service(name, config):
        process = await launch(config['command'], config['cwd'], config['env'], out / f'service_{name}.log')
        write_json(out / f'service_{name}.json', {**config, 'pid': process.pid})
        async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
            for _ in range(450):
                if process.returncode is not None:
                    raise RuntimeError(f'{name} service exited: {out / f"service_{name}.log"}')
                try:
                    response = await client.get(config['url'] + config['health'])
                    if response.status_code == 200:
                        body = response.json()
                        ready = (body.get('model') == config['model'] if name == 'student' else
                                 any(m['id'] == config['model'] for m in body.get('data', [])) if name == 'planner'
                                 else body.get('ok'))
                        if ready:
                            print(f'{name} ready pid={process.pid}', flush=True); return
                except (httpx.HTTPError, ValueError):
                    pass
                await asyncio.sleep(2)
        raise TimeoutError(f'{name} service startup timed out')

    async def match(item):
        directory = out / item['directory']; path = directory / 'report.json'
        if not path.exists():
            env = {'ARENA_SHOWDOWN_URL': 'http://127.0.0.1:8100', 'ARENA_OPPONENT_MODEL': 'prior',
                'ARENA_STREAM_DIR': str(directory / 'streams'), 'VLLM_API_KEY': 'local',
                'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            process = await launch([sys.executable, '-m', 'arena.cli', 'run', str(directory / 'config.yaml'),
                                   '--run-dir', str(directory), '--emit-events'], ROOT, env, directory / 'run.log')
            running[item['directory']] = process.pid; status()
            print(f"start {item['directory']} pid={process.pid}", flush=True)
            code = await process.wait()
            running.pop(item['directory'])
            if code:
                raise RuntimeError(f"Match exited {code}: {item['directory']}")
        report = json.loads(path.read_text()); result = report['agents'][item['arm']]
        if result['games'] != item['games'] or report.get('aborted'):
            raise RuntimeError(f"Incomplete match: {item['directory']} {result['games']}/{item['games']}")
        finished.append(item); status()
        print(f"finish {item['directory']} {result['wins']}/{result['games']}", flush=True)

    async def pulse():
        while True:
            status(); await asyncio.sleep(30)

    monitor = asyncio.create_task(pulse())
    try:
        await asyncio.gather(*(service(name, config) for name, config in protocol['services'].items()))
        phase = 'preflight'; status()
        for opponent in OPPONENTS:
            await asyncio.gather(*(match(item) for item in protocol['preflight'] if item['opponent'] == opponent))
        check = preflight_quality(out, protocol)
        write_json(out / 'preflight/quality.json', check)
        if not check['passed']:
            raise RuntimeError(f"Preflight failed: {check}")
        finished.clear(); phase = 'battles'; status()

        for team in range(12):
            for opponent in OPPONENTS:
                await asyncio.gather(*(match(item) for item in schedule
                                       if item['opponent'] == opponent and item['team'] == team))
        phase = 'summarizing'; status()
        result = summarize(out)
        write_json(out / 'summary.json', result)
        write_report(out / 'results.md', result)
        from plot_hymem_strategy_evaluation import plot
        plot(out, result)
        phase = 'completed'; status(passed=result['passed'], summary=str(out / 'summary.json'))
    except BaseException as error:
        phase = 'failed'; status(error=f'{type(error).__name__}: {error}')
        raise
    finally:
        monitor.cancel()
        for process in reversed(children):
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), 20)
                except asyncio.TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL); await process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--games-per-team', type=int, default=2)
    args = parser.parse_args()
    prepare(args.out, args.games_per_team)
    if args.action == 'run':
        asyncio.run(run(args.out))


if __name__ == '__main__':
    main()
