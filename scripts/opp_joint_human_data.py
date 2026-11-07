"""Build one temporal Gen9 OU human dataset from local public replay archives."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from arena.opponent.joint.human_data import (
    public_lines, windows, executed_actions, revealed_facts, identifier,
    training_example, read_game, player_history,
)

SCHEMA = pa.schema([
    ('battle_id', pa.string()), ('source', pa.string()), ('source_split', pa.string()),
    ('split', pa.string()), ('format', pa.string()), ('month', pa.string()),
    ('upload_time', pa.int64()), ('players', pa.list_(pa.string())),
    ('ratings', pa.list_(pa.int32())), ('rules', pa.list_(pa.string())),
    ('preview', pa.list_(pa.list_(pa.string()))), ('public', pa.list_(pa.string())),
])


def select_games(args):
    frames, paths = [], {}
    for source in ('metamon', 'pokechamp'):
        root = ROOT / 'artifacts/opponent_data' / source
        manifest = json.loads((root / 'manifest.json').read_text())
        paths[source] = [f['path'] for f in manifest['files']]
        frame = pq.read_table(root / 'battles.parquet').to_pandas()
        frame['source'] = source
        frame['source_split'] = frame.source_file.map(
            lambda i: Path(paths[source][i]).name.split('-')[0])
        frames.append(frame)
    meta, poke = frames
    upstream_test = set(poke.loc[poke.source_split == 'test', 'identity'])
    all_rows = pd.concat(frames, ignore_index=True)
    overlap = len(meta) + len(poke) - all_rows.identity.nunique()
    # Prefer Metamon's stable pseudonyms/timestamps for duplicate battles.
    conflict_ids = set(all_rows.loc[all_rows.has_conflict, 'identity'])
    pool = all_rows.drop_duplicates('identity', keep='first')
    excluded_test = int(pool.identity.isin(upstream_test).sum())
    pool = pool[~pool.identity.isin(conflict_ids | upstream_test)].copy()
    pool = pool[pool.month >= args.start_month]
    pool['split'] = np.where(pool.month < args.validation_month, 'train',
                            np.where(pool.month < args.test_month, 'validation', 'test'))
    selected = []
    for split, count in [('train', args.train), ('validation', args.validation), ('test', args.test)]:
        part = pool[pool.split == split]
        # Source/month stratification prevents first-file or recent-only sampling.
        group_sizes = part.groupby(['source', 'month']).size()
        allocations = group_sizes * min(count, len(part)) / len(part)
        sizes = np.floor(allocations).astype(int)
        for key in (allocations - sizes).sort_values(ascending=False).index[:min(count, len(part)) - sizes.sum()]:
            sizes.loc[key] += 1
        for key, size in sizes.items():
            group = part[(part.source == key[0]) & (part.month == key[1])]
            selected.append(group.sample(n=int(size), random_state=211))
    chosen = pd.concat(selected, ignore_index=True)
    summary = {
        'source_gen9ou_games': {s: len(f) for s, f in zip(('metamon', 'pokechamp'), frames)},
        'cross_source_duplicate_games': overlap, 'unique_gen9ou_games': all_rows.identity.nunique(),
        'identity_conflict_games_excluded': len(conflict_ids),
        'pokechamp_official_test_games_reserved': excluded_test,
        'eligible_after_date_and_holdout': pool.groupby('split').size().to_dict(),
        'selected': chosen.groupby('split').size().to_dict(),
    }
    return chosen, paths, summary


def convert_file(source, path, selected, out):
    selected = {int(r['source_row']): r for r in selected}
    counts, event_counts, species, moves = Counter(), Counter(), Counter(), Counter()
    buffers, writers, index_rows = {}, {}, []
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        indices = [i for i in range(batch.num_rows) if offset + i in selected]
        for i, row in zip(indices, batch.take(pa.array(indices, type=pa.int64())).to_pylist()):
            info = selected[offset + i]
            text = row['log' if source == 'metamon' else 'text']
            lines = public_lines(text, info['winner_side'])
            # Public archives also tag custom room tournaments as gen9ou.
            # An empty rated marker identifies ladder battles, unlike
            # "|rated|Tournament battle"; historical OU ban changes are retained.
            if info['platform'] != 'showdown' or not any(l in ('|rated|', '|rated') for l in lines):
                counts['excluded_unverified_tournament_or_challenge'] += 1
                continue
            preview, ratings, rules = [[], []], [None, None], []
            for line in text.splitlines():
                p = line.split('|')
                if len(p) > 3 and p[1] == 'poke' and p[2] in ('p1', 'p2'):
                    preview[int(p[2][1])-1].append(identifier(p[3].split(',')[0]))
                if len(p) > 5 and p[1] == 'player' and p[2] in ('p1', 'p2') and p[5].isdigit():
                    ratings[int(p[2][1])-1] = int(p[5])
                if len(p) > 2 and p[1] == 'rule':
                    rules.append(p[2])
            if any(len(p) != 6 for p in preview):
                counts['excluded_incomplete_preview'] += 1
                continue
            if not any(l.startswith('|turn|') for l in lines):
                counts['excluded_no_turn'] += 1
                continue
            split = info['split']
            record = {'battle_id': info['identity'], 'source': source,
                      'source_split': info['source_split'], 'split': split, 'format': 'gen9ou',
                      'month': info['month'], 'upload_time': None if pd.isna(info['uploadtime']) else int(info['uploadtime']),
                      'players': [info['p1'], info['p2']], 'ratings': ratings,
                      'rules': rules, 'preview': preview, 'public': lines}
            file_name = source + '-' + Path(path).stem + '.parquet'
            if split not in writers:
                target = Path(out) / split / file_name
                target.parent.mkdir(parents=True, exist_ok=True)
                writers[split] = pq.ParquetWriter(target, SCHEMA, compression='zstd')
                buffers[split] = []
            record_row = counts['rows:' + split]
            buffers[split].append(record)
            counts['rows:' + split] += 1
            if len(buffers[split]) >= 128:
                writers[split].write_table(pa.Table.from_pylist(buffers[split], schema=SCHEMA))
                buffers[split].clear()
            counts['games'] += 1
            counts['source:' + source] += 1
            counts['month:' + info['month']] += 1
            event_counts.update(l.split('|')[1] for l in lines)
            species.update(s for team in preview for s in team)
            facts = revealed_facts(record)
            counts.update('fact:' + f['field'] + ':' + f['scope'] for f in facts)
            for window in windows(record):
                actions, reasons, tera = executed_actions(record, window)
                counts['player_windows:' + split] += 2
                for action, reason, tera_type in zip(actions, reasons, tera):
                    counts['action:' + (action['kind'] if action else 'missing')] += 1
                    if action and action['kind'] == 'move':
                        moves[action['value']] += 1
                    if tera_type:
                        counts['tera_observations'] += 1
                    if reason == 'prevented_or_cancelled':
                        counts['prevented_or_cancelled'] += 1
            rating = max((r for r in ratings if r is not None), default=0)
            counts['rating:' + ('unknown' if not rating else str(min(1800, rating // 200 * 200)))] += 1
            for side, player in zip(('p1', 'p2'), record['players']):
                index_rows.append({'battle_id': record['battle_id'], 'side': side, 'player_id': player,
                                   'split': split, 'month': record['month'], 'upload_time': record['upload_time'],
                                   'file': split + '/' + file_name, 'row': record_row})
        offset += batch.num_rows
    for split, writer in writers.items():
        if buffers[split]:
            writer.write_table(pa.Table.from_pylist(buffers[split], schema=SCHEMA))
        writer.close()
    return dict(counts), dict(event_counts), dict(species), dict(moves), index_rows


def history_index(rows, out):
    frame = pd.DataFrame(rows)
    # Ordering means historical replay availability, not inferred battle start time.
    frame['order'] = frame.upload_time.fillna(
        pd.to_datetime(frame.month + '-01', utc=True).astype('int64') // 10**9).astype('int64')
    frame = frame.sort_values(['player_id', 'order', 'battle_id'])
    previous_ids, previous_sides = [], []
    previous_splits = []
    current_player, previous, block_previous, block_order = None, None, None, None
    for row in frame.itertuples():
        if row.player_id != current_player:
            current_player, previous, block_previous, block_order = row.player_id, None, None, None
        if row.order != block_order:
            block_previous, block_order = previous, row.order
        previous_ids.append(block_previous[0] if block_previous else None)
        previous_sides.append(block_previous[1] if block_previous else None)
        previous_splits.append(block_previous[2] if block_previous else None)
        previous = (row.battle_id, row.side, row.split)
    frame['previous_battle_id'] = previous_ids
    frame['previous_side'] = previous_sides
    frame['previous_split'] = previous_splits
    frame['history_basis'] = np.where(frame.upload_time.notna(), 'strictly_earlier_upload', 'strictly_earlier_month')
    frame = frame.drop(columns='order')
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), out / 'players.parquet', compression='zstd')
    sizes = frame.groupby('player_id').size()
    ranks = {'train': 0, 'validation': 1, 'test': 2}
    bad = sum(ranks[p] > ranks[s] for p, s in zip(frame.previous_split, frame.split) if p is not None)
    assert bad == 0
    return {'player_ids': len(sizes), 'players_with_multiple_games': int((sizes >= 2).sum()),
            'players_with_at_least_five_games': int((sizes >= 5).sum()),
            'player_games_with_earlier_history': int(frame.previous_battle_id.notna().sum()),
            'earlier_history_by_split': frame[frame.previous_battle_id.notna()].groupby('split').size().to_dict(),
            'history_from_future_split': bad,
            'identity_scope': 'Source-local stable account IDs; no Metamon pseudonym to PokéChamp username linkage. No claim of strict unseen-player holdout.'}


def main(args):
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    selected, paths, census = select_games(args)
    print(json.dumps({'selection': census}), flush=True)
    counts, events, species, moves = Counter(), Counter(), Counter(), Counter()
    rows = []
    tasks = list(selected.groupby(['source', 'source_file']))
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(convert_file, source, paths[source][file], group.to_dict('records'), str(args.out))
                for (source, file), group in tasks]
        for number, job in enumerate(as_completed(jobs), 1):
            c, e, s, m, index = job.result()
            counts.update(c); events.update(e); species.update(s); moves.update(m); rows.extend(index)
            print(json.dumps({'files': number, 'of': len(tasks), 'games': counts['games'],
                              'seconds': round(time.time() - started, 1)}), flush=True)
    active_files = {row['file'] for row in rows}
    for path in args.out.glob('*/*.parquet'):
        if path.relative_to(args.out).as_posix() not in active_files:
            path.unlink()
    histories = history_index(rows, args.out)
    index = pq.read_table(args.out / 'players.parquet', filters=[('split', '=', 'validation')]).to_pandas()
    entry = index[index.previous_battle_id.notna()].iloc[0]
    game = read_game(args.out, entry.file, int(entry.row))
    window = next(w for w in windows(game) if w['phase'] == 'turn')
    example = training_example(game, window, entry.side)
    example['input']['past_games'] = player_history(args.out, entry.battle_id, entry.side, limit=2)
    (args.out / 'example.json').write_text(json.dumps(example, ensure_ascii=False, indent=2))
    summary = {
        'sources': {'metamon': 'https://huggingface.co/datasets/jakegrigsby/metamon-raw-replays',
                    'pokechamp': 'https://huggingface.co/datasets/milkkarten/pokechamp'},
        'selection': census, 'counts': dict(counts), 'distinct_preview_species': len(species),
        'distinct_executed_moves': len(moves), 'cross_game': histories,
        'temporal_split': {'start_month': args.start_month, 'validation_month': args.validation_month,
                           'test_month': args.test_month, 'time_semantics': 'replay availability/upload month; PokéChamp source month where exact upload time is absent'},
        'supervision': {'public_event_target': True, 'executed_action_partial_label': True,
                        'repeated_known_attributes_not_counted_as_new_labels': True,
                        'submitted_action_truth': False, 'complete_hidden_team_truth': False,
                        'strategy_parameter_truth': False, 'private_information_state_truth': False,
                        'metamon_imputed_hidden_fields_used': False},
        'evaluation_scope': 'Temporal holdout. All copies of official PokéChamp test battle IDs are reserved outside this corpus. Test data is not used to fit a model or construct priors.',
        'battle_scope': 'Showdown Gen9 OU rated ladder only. Custom tournaments/challenges are excluded; the historical rules and date are retained instead of imposing present-day bans on old games.',
        'training_use': 'human_examples yields public-prefix inputs and separate event/partial-fact labels. Requires observed-event/partial-label objectives; do not feed this into the synthetic true-C/X action loss unchanged.',
        'seconds': round(time.time() - started, 1),
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (args.out / 'vocabulary_observed.json').write_text(json.dumps({'preview_species_counts': species,
        'executed_move_counts': moves, 'event_counts': events}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/opponent_joint/human')
    p.add_argument('--train', type=int, default=240000)
    p.add_argument('--validation', type=int, default=30000)
    p.add_argument('--test', type=int, default=70000)
    p.add_argument('--start-month', default='2024-01')
    p.add_argument('--validation-month', default='2026-01')
    p.add_argument('--test-month', default='2026-03')
    p.add_argument('--workers', type=int, default=6)
    main(p.parse_args())
