"""Prepare player-view action supervision and fixed development teams."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import yaml


def read(path):
    with open(path) as file:
        yield from map(json.loads, file)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def action_key(aid, request):
    from poke_env.data import to_id_str
    if aid.startswith('move:'):
        return 'move_' + aid[5:].replace('+tera', '_tera')
    if aid.startswith('switch:'):
        mon = next(m for m in request['side']['pokemon']
                   if to_id_str(m['ident'].split(': ', 1)[-1]) == aid[7:])
        return 'switch_' + to_id_str(mon['details'].split(',')[0])
    raise ValueError(f'unsupported battle action: {aid}')


def export_teacher(cfg):
    from arena.observation.pokeenv_replay import battle_states
    from arena.observation.trace import request_phase
    from arena.agents.system1.decider import FastDecider, RouteConfig
    from arena.s1.policy_data import history_for
    from arena.state.views import legal_options

    root = Path(cfg['root']) / 'teacher'
    if (root / 'files.txt').exists():
        raise FileExistsError(f'teacher export already exists: {root}')
    (root / 'streams').mkdir(parents=True, exist_ok=True)
    (root / 'trajectories').mkdir(exist_ok=True)
    source = Path(cfg['data']['quality_root']) / 'teacher_recovery'
    meta = {r['battle_tag']: r for r in read(source / 'recovered_source_manifest.jsonl')}
    count = decisions = 0
    with (root / 'observations.jsonl').open('w') as rows_out, (root / 'files.txt').open('w') as names:
        for game in read(source / 'kakuna_player_streams.jsonl'):
            tag = game['battle_tag']; info = meta[tag]
            stream = root / 'streams' / f'{tag}.jsonl'
            with stream.open('w') as f:
                f.write(json.dumps({'k': 'meta', 'battle_tag': tag, 'username': info['teacher_username']})+'\n')
                f.write(json.dumps({'k': 'in', 'data': '\n'.join(game['player_lines'])})+'\n')
            choices = iter(game['accepted_choices'])
            decider = FastDecider(None, RouteConfig(mode='flat', debias=False, ask_hard=False,
                                                  history_events=cfg['history_events']))
            request_ids, keys, played, requests = [], [], [], []
            for req, battle in battle_states(game['player_lines'], battle_tag=tag,
                    username=info['teacher_username'], fmt='gen9ou'):
                phase = request_phase(req).value
                if phase in ('wait', 'unknown'):
                    continue
                accepted = next(choices)
                if phase == 'team_preview':
                    assert accepted['action_id'].startswith('team:')
                    continue
                state, qs, facts = decider.build_request(battle, legal_options(battle))
                legal = [f.key for f in facts]
                chosen = action_key(accepted['action_id'], req)
                if chosen not in legal:
                    raise ValueError(f'{tag}:{req["rqid"]}: accepted action missing: {chosen}')
                rows_out.write(json.dumps({'id': f'{info["group"]}:kakuna:{req["rqid"]}:best',
                    'group': info['group'], 'battle_tag': tag, 'state': state, 'question': qs['best'],
                    'keys': legal, 'played_key': chosen, 'request': req, 'step': len(keys),
                    'source': 'kakuna_executed', 'split': info['split']}, ensure_ascii=False)+'\n')
                history_for(battle, cfg['history_events']).acted(chosen)
                request_ids.append(req['rqid']); keys.append(legal); played.append(chosen)
                requests.append(deepcopy(req))
            if next(choices, None) is not None:
                raise ValueError(f'unconsumed accepted choices: {tag}')
            path = root / 'trajectories' / f'{tag}.json'
            write(path, {'source_stream': str(stream), 'request_ids': request_ids,
                         'legal_keys': keys, 'played_keys': played, 'requests': requests})
            names.write(str(path)+'\n'); count += 1; decisions += len(keys)
    result = {'battles': count, 'decisions': decisions}
    write(root / 'export.json', result); print(json.dumps(result), flush=True)


def roster(path):
    from poke_env.teambuilder.teambuilder import Teambuilder
    from poke_env.data import to_id_str
    return tuple(sorted(to_id_str(m.species or m.nickname) for m in Teambuilder.parse_showdown_team(Path(path).read_text())))


def refresh(job):
    from arena.s1.policy_replay import rebuild_historical
    return list(rebuild_historical(job))


def prepare(cfg, workers):
    from arena.s1.policy_data import full_action_target
    from arena.s1.prompts import render
    from arena.s1.scorer import _chat_ids_batch
    from transformers import AutoTokenizer
    import math

    root = Path(cfg['root']); out = root / 'data'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'validation.json').exists():
        raise FileExistsError(f'Prepared dataset already exists: {out}')
    quality = Path(cfg['data']['quality_root'])
    dispositions = {r['id']: r for r in read(quality / 'sample_manifest.jsonl')}
    sources = list(read(quality / 'source_manifest.jsonl'))
    native_sources = {r['group']: r for r in sources if r['source'].startswith('student_')}
    teams = sorted(Path(cfg['data']['teams']).glob('*.txt'))
    dev_rosters = {roster(teams[i]) for i in cfg['data']['dev_team_indices']}
    for split in ('train', 'dev'):
        dest = root / 'teams' / split; dest.mkdir(parents=True, exist_ok=True)
        for team in teams:
            if (roster(team) in dev_rosters) == (split == 'dev'):
                shutil.copyfile(team, dest / team.name)
    dev_groups = {r['group'] for r in sources if (r['source'].startswith('historical_') and r['split'] == 'valid')
                  or tuple(sorted(r['roster'])) in dev_rosters
                  or tuple(sorted(r.get('opponent_roster', []))) in dev_rosters}
    # Keep complete native matchups on one side of the split, including both player views.
    def matchup(r):
        return tuple(sorted((tuple(sorted(r['roster'])), tuple(sorted(r['opponent_roster'])))))
    dev_pairs = {matchup(r) for r in native_sources.values() if r['group'] in dev_groups}
    dev_groups.update(r['group'] for r in native_sources.values() if matchup(r) in dev_pairs)
    native = {(r['battle_tag'], r['request']['rqid']): (r['state'], r['question'])
              for r in read(cfg['data']['native'])}
    grouped, rows, excluded = defaultdict(list), [], Counter()
    for split in ('train', 'valid'):
        for row in read(Path(cfg['data']['policy']) / f'{split}.jsonl'):
            disposition = dispositions[row['id']]['action_supervision']
            if disposition.startswith('exclude'):
                excluded[disposition] += 1; continue
            row['split'] = 'valid' if row['group'] in dev_groups else 'train'
            row['source'] = dispositions[row['id']]['source']
            row['family'] = 'native' if row['source'].startswith('student_') else 'historical'
            if row['family'] == 'native':
                tag, rqid = row['request'].rsplit(':', 1)
                row['state'], row['question'] = native[tag, int(rqid)]
                rows.append(row)
            else:
                grouped[row['provenance']['record']].append(row)
    del native
    with ProcessPoolExecutor(workers) as pool:
        for chunk in pool.map(refresh, grouped.values(), chunksize=8):
            rows.extend(chunk)
    labels = {Path(r['file']).stem: r for r in read(root / 'teacher/labels.jsonl')}
    for row in read(root / 'teacher/observations.jsonl'):
        if len(row['keys']) < 2:
            excluded['teacher_single_option'] += 1; continue
        lab = labels[row['battle_tag']]; i = row['step']
        target, removed = full_action_target({}, lab['probs'][i], row['keys'], index_keys=lab['action_keys'][i])
        row.update(target=target, purpose='action', qid='best', teacher='kakuna-34',
            family='kakuna_executed', split='valid' if row['group'] in dev_groups else 'train',
            provenance={'record': lab['file'], 'step': i, 'tier': 'native_player_view',
                        'teacher_history_complete': True, 'teacher_removed_mass': removed})
        row.pop('request'); rows.append(row)
    # Compare full visible contexts directly. Validation takes precedence over a duplicate in training.
    def context(row):
        return json.dumps([row['state'], row['question']], sort_keys=True, separators=(',', ':'))
    valid_context = {context(r) for r in rows if r['split'] == 'valid'}
    retained, seen = [], {'train': set(), 'valid': set()}
    for row in rows:
        ctx = context(row); split = row['split']
        if split == 'train' and ctx in valid_context:
            excluded['train_valid_context_overlap'] += 1; continue
        identity = (ctx, tuple(round(x, 4) for x in row['target']))
        if identity in seen[split]:
            excluded['same_context_and_target'] += 1; continue
        seen[split].add(identity); retained.append(row)
    del rows, seen, valid_context, grouped
    counts = {s: Counter(r['family'] for r in retained if r['split'] == s) for s in ('train', 'valid')}
    os.environ.setdefault('TOKENIZERS_PARALLELISM','true')
    os.environ.setdefault('RAYON_NUM_THREADS','6')
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model'])
    stats, lengths, ids, groups = defaultdict(Counter), [], set(), defaultdict(set)
    action_counts = defaultdict(Counter)
    handles = {s: (out / f'{s}.jsonl').open('w') for s in counts}
    try:
        for start in range(0, len(retained), 128):
            chunk = retained[start:start+128]
            prompts = [render(r['state'], r['question']) for r in chunk]
            tokens = _chat_ids_batch(tokenizer, [p[0] for p in prompts])
            for row, (_, keys), encoded in zip(chunk, prompts, tokens):
                target = row['target']; split = row['split']; family = row['family']
                assert row['id'] not in ids; ids.add(row['id'])
                assert keys == row['keys'] and len(keys) == len(target)
                assert all(math.isfinite(x) and x >= 0 for x in target) and abs(sum(target)-1) < 1e-5
                assert row['provenance']['teacher_history_complete']
                assert 'resources' in row['state'] and 'recent_decisions' in row['state']
                if len(encoded) > cfg['max_tokens']:
                    raise ValueError(f'{row["id"]}: {len(encoded)} tokens exceeds configured limit')
                n = sum(counts[split].values())
                row['weight'] = cfg['data']['source_mass'][family] * n / counts[split][family]
                row['token_length'] = len(encoded)
                row['teacher_action'] = keys[max(range(len(keys)), key=target.__getitem__)]
                from arena.s1.action_training import action_category
                action_counts[split][action_category(row['teacher_action'])] += 1
                groups[split].add(row['group']); lengths.append(len(encoded))
                stats[split][row['source']] += 1
                handles[split].write(json.dumps(row, ensure_ascii=False)+'\n')
            if start % 8192 == 0:
                print(json.dumps({'encoded': min(start+128, len(retained)), 'total': len(retained)}), flush=True)
    finally:
        for f in handles.values(): f.close()
    assert not groups['train'] & groups['valid']
    lengths.sort()
    report = {'passed': True, 'rows': {s: dict(c) for s,c in stats.items()},
              'families': {s: dict(c) for s,c in counts.items()}, 'excluded': dict(excluded),
              'battles': {s:len(g) for s,g in groups.items()},
              'teacher_actions': {s:dict(c) for s,c in action_counts.items()},
              'train_valid_battle_overlap': 0, 'train_valid_context_overlap': 0,
              'max_tokens': lengths[-1], 'median_tokens': lengths[len(lengths)//2],
              'dev_rosters': list(dev_rosters), 'dev_groups': sorted(dev_groups),
              'source_mass': cfg['data']['source_mass'], 'train_teams': len(teams)-len(dev_rosters),
              'dev_teams': len(dev_rosters), 'supervision': 'full legal action distribution only'}
    write(out / 'validation.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('dev_groups','dev_rosters')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['export-teacher', 'prepare'])
    p.add_argument('--config', default='configs/training/s1_action_ppo.yaml')
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args(); cfg = yaml.safe_load(Path(a.config).read_text())
    if a.command == 'export-teacher': export_teacher(cfg)
    else: prepare(cfg, a.workers)


if __name__ == '__main__':
    main()
