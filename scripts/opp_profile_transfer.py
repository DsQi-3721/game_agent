"""Player-specific behavior probabilities on fixed new-team query games."""
import argparse
import json
import multiprocessing
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from arena.opponent.joint.data import games
from arena.opponent.joint.filter import systematic
from arena.opponent.joint.strategy import behavior_opportunities, BEHAVIORS
from arena.opponent.joint.tokens import ident
from opp_baseline_topk import aligned_private
from opp_joint_public import identity, score
from opp_experiment_utils import BASE, DEX, TYPES, side_of, roster, start_game

OUT = BASE / 'training/experiments/profile_transfer'
HISTORY = (0, 1, 3, 5)
BEHAVIOR = {'switch': 2, 'setup': 12, 'recovery': 11, 'tera': 3}


def compact(game, player):
    observer = 'p2' if side_of(game, player) == 'p1' else 'p1'
    result = {k: game[k] for k in ('id', 'players', 'round', 'public', 'private', 'commands')}
    result['teams'] = {observer: game['teams'][observer]}
    result['decisions'] = []
    for row in game['decisions']:
        entry = {k: row[k] for k in ('side', 'cutoff', 'actions', 'features', 'target', 'distribution')}
        entry['request'] = {k: row['request'][k] for k in ('teamPreview', 'forceSwitch') if k in row['request']}
        entry['truth'] = {k: row['truth'][k] for k in ('family', 'change')}
        result['decisions'].append(entry)
    return result


def select_group(group):
    source = list(games(BASE / f'data/test/group-{group:05}.jsonl.gz'))
    preferred = group // 7 % 20
    for offset in range(4):
        player = f'test-{group}-{(preferred + 5 * offset) % 20}'
        sequence = sorted((g for g in source if player in g['players']), key=lambda g: g['round'])
        support = sequence[:5]
        prior_rosters = {tuple(sorted(map(ident, roster(g, player)))) for g in support}
        for query in sequence[5:]:
            if tuple(sorted(map(ident, roster(query, player)))) in prior_rosters: continue
            rows = [r for r in query['decisions'] if r['side'] == side_of(query, player)
                    and not r['request'].get('teamPreview')]
            if len(rows) < 10 or any(r['target'] < 0 for r in rows[:10]): continue
            selected = support + [query]
            labels = [r['truth'] for g in selected for r in g['decisions'] if r['side'] == side_of(g, player)]
            families = {r['family'] for r in labels}
            if len(families) != 1 or any(r['change'] != 'keep' for r in labels): continue
            family = next(iter(families))
            entry = dict(player=player, group=group, role='test', family=family,
                         scenario='known' if group % 7 == 0 else 'unseen_team',
                         support=[g['id'] for g in support], query=query['id'], query_round=query['round'],
                         query_roster=roster(query, player), support_rosters=[roster(g, player) for g in support])
            with (OUT / 'cases' / f'{player}.pkl').open('wb') as f:
                pickle.dump({g['id']: compact(g, player) for g in selected}, f, protocol=5)
            return entry
    raise ValueError(f'No eligible fixed new-team query for group {group}')


def prepare(workers):
    import pyarrow.parquet as pq
    groups = [g for g in range(84, 254) if g % 7 in (0, 1)]
    assert len(groups) == 50
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        entries = list(pool.map(select_group, groups))
    index = pq.read_table(BASE / 'training/data/index.parquet', columns=['split', 'player_id']).to_pandas()
    trained = set(index.loc[index.split == 'train', 'player_id'])
    assert not trained.intersection(e['player'] for e in entries)
    protocol = dict(cohort=entries, model='training/adaptation/model.pt', histories=list(HISTORY),
        particles=1024, configurations=128, branches=32, move_budget=32, time_budget_per_call=60,
        query_decisions=10, early_decisions=5, probability_floor=1e-12,
        primary_population='40 non-random held-out players; 10 random controllers are a separate negative control.',
        selection='One player per independent test group, balanced across five controller families and known/unseen compositions. First five games form support; earliest subsequent new-roster game with ten labelled decisions is fixed before prediction.',
        controls='Correct player history; common prior; another player from the same family and composition scenario. Donor is fixed across history lengths, matched by effective history length and public team-type frequencies.',
        carry='Only weighted Z is inherited. Fresh C/X, encoder memory, previous action and change history. Identical initial candidate pools and RNG state across query conditions.',
        prediction='Predict, obtain observed resolution, update. Model weights remain fixed. Family, true action lists and controller distributions are evaluation-only.',
        action_metrics='Move identity or switch destination, with Tera variants marginalized. Expected NLL and KL use the logged controller distribution; TV includes predicted actions absent from true support. Submitted-action NLL and Top-1 are also reported.',
        profiles='Situation-conditioned behavior probabilities from current_action_tendencies. Score switch/setup/recovery/Tera when both that behavior and an alternative are legally available in an ordinary decision. Standardized dossier opportunity subsets are also reported.',
        profile_metric='Per-player MAE for each behavior, then equal average over available behaviors and players. Full query decisions are retained for action metrics.',
        missing='Missing teacher distributions are omitted only from distribution scores. Invalid model predictions receive NLL -log(1e-12), TV=1, profile error=1 and zero Top-1; failures remain in denominators.',
        uncertainty='Equal-player means and paired bootstrap confidence intervals, 2000 player resamples. Fixed donor assignment. Primary comparison is H=5 correct versus each control; all H=0/1/3/5 results are retained.')
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(dict(players=len(entries), families={f:sum(e['family']==f for e in entries) for f in sorted({e['family'] for e in entries})})), flush=True)


def teacher_scores(before, target):
    valid = before['prediction_valid']; predicted = defaultdict(float)
    for action in before['prediction']['actions']:
        predicted[identity(action['action'])] += action['probability']
    mass = sum(predicted.values())
    predicted = {a: p / mass for a, p in predicted.items()}
    result = dict(valid_prediction=valid, action_mass_error=abs(mass - 1),
                  nll=None, top1=None, expected_nll=None, kl=None, tv=None, unsupported_teacher_mass=None,
                  profiles={}, card_profiles={})
    if target['target'] >= 0:
        actual = score(predicted, identity(target['actions'][target['target']]['key']), valid)
        result.update(nll=actual['nll'], top1=actual['hits'][0])
    q = np.asarray(target['distribution'], dtype=float)
    teacher_valid = bool(np.isfinite(q).all() and (q >= 0).all() and abs(q.sum() - 1) < 1e-5)
    result['teacher_valid'] = teacher_valid
    if not teacher_valid: return result
    q = q / q.sum()
    truth = defaultdict(float)
    for action, probability in zip(target['actions'], q): truth[identity(action['key'])] += float(probability)
    positive = {a: p for a, p in truth.items() if p > 0}
    entropy = -sum(p * np.log(p) for p in positive.values())
    ce = -sum(p * np.log(max(predicted.get(a, 0.) if valid else 0., 1e-12)) for a, p in positive.items())
    result.update(expected_nll=float(ce), kl=float(ce - entropy), teacher_entropy=float(entropy),
                  tv=float(sum(abs(predicted.get(a, 0.) - truth.get(a, 0.)) for a in predicted.keys() | truth.keys()) / 2) if valid else 1.,
                  unsupported_teacher_mass=float(sum(p for a, p in positive.items() if predicted.get(a, 0.) == 0)) if valid else 1.,
                  predicted_actions=dict(predicted) if valid else {}, teacher_actions=dict(truth))
    features = np.asarray(target['features'], dtype=float)
    ordinary = not bool(target['request'].get('forceSwitch'))
    result['ordinary_choice'] = ordinary and len(truth) > 1
    tendencies = before['opponent']['strategy']['current_action_tendencies']
    for name, column in BEHAVIOR.items():
        if ordinary and (features[:, column] > 0).any() and (features[:, column] == 0).any():
            actual = float(q @ features[:, column]); prediction = float(tendencies[name]) / mass if valid else None
            result['profiles'][name] = dict(predicted=prediction, truth=actual,
                absolute_error=abs(prediction - actual) if valid else 1.)
    for name, eligible in behavior_opportunities(features, 1 if ordinary else 2).items():
        if eligible:
            column = BEHAVIORS[name][1]; key = next(k for k, v in BEHAVIOR.items() if v == column)
            result['card_profiles'][name] = result['profiles'][key]
    return result


def replay(model, game, entry, label, limit=None, scoring=False):
    private = aligned_private(game); i = step = window = 0; output = []; started = time.monotonic()
    while i < len(game['decisions']) and (limit is None or step < limit):
        j = i + 1
        while j < len(game['decisions']) and game['decisions'][j]['cutoff'] == game['decisions'][i]['cutoff']: j += 1
        start = game['decisions'][i]['cutoff']
        end = game['decisions'][j]['cutoff'] if j < len(game['decisions']) else len(game['public'])
        target = next((r for r in game['decisions'][i:j] if r['side'] == model.side and not r['request'].get('teamPreview')), None)
        row = None
        if target:
            row = dict(player=entry['player'], family=entry['family'], scenario=entry['scenario'], game=game['id'], step=step)
            if scoring: row.update(teacher_scores(model.query(), target))
        our = next((command for side, command in game['commands'][i:j] if side == model.observer), None)
        after = model.update(game['public'][start:end], our, private_events=private[model.observer][start:end]); calls = 1
        while model.pending is not None:
            after = model.resume(); calls += 1
            if calls % 5 == 0:
                print(json.dumps(dict(player=entry['player'], game=game['id'], label=label, window=window, calls=calls, phase=model.pending['phase'] if model.pending else None)), flush=True)
        valid = after['prediction_valid'] and after['inference_cost'].get('observation_applied', False)
        if row is not None:
            row['valid_update'] = valid; output.append(row); step += 1
        if window % 10 == 0 or not valid:
            print(json.dumps(dict(player=entry['player'], game=game['id'], label=label, window=window, valid=valid, seconds=round(time.monotonic()-started, 1))), flush=True)
        i = j; window += 1
    return output


def load_cases(entry):
    with (OUT / 'cases' / f"{entry['player']}.pkl").open('rb') as f: return pickle.load(f)


def support_job(entry):
    torch.set_num_threads(1); source = load_cases(entry); model = None; roots = {}; records = []
    try:
        for n, game_id in enumerate(entry['support'], 1):
            model = start_game(model, source[game_id], entry, 17131 + entry['group'])
            records.extend(replay(model, source[game_id], entry, f'support_{n}'))
            if n in HISTORY:
                roots[n] = dict(z=model.z.detach().cpu().clone(), weights=torch.tensor(model.weights.copy()),
                                effective_decisions=sum(r['valid_update'] for r in records))
                torch.save(dict(player=entry['player'], roots=roots, records=records), OUT / 'memory' / f"{entry['player']}.pt")
        (OUT / 'support' / f"{entry['player']}.json").write_text(json.dumps(dict(player=entry['player'], records=records, complete=True)))
        return entry['player']
    finally:
        if model is not None: model.close()


def match():
    from scipy.optimize import linear_sum_assignment
    entries = json.loads((OUT / 'protocol.json').read_text())['cohort']; assignments = {}; details = []
    roots = {e['player']: torch.load(OUT / 'memory' / f"{e['player']}.pt", weights_only=True)['roots'] for e in entries}
    for family, scenario in sorted({(e['family'], e['scenario']) for e in entries}):
        cohort = [e for e in entries if (e['family'], e['scenario']) == (family, scenario)]
        cost = np.zeros((len(cohort), len(cohort))); observations = {}
        for h in HISTORY[1:]:
            types = []; length = []
            for e in cohort:
                counts = np.zeros(len(TYPES))
                for team in e['support_rosters'][:h]:
                    for species in team:
                        for t in DEX[ident(species)]['types']: counts[TYPES.index(t)] += 1
                types.append(counts / np.linalg.norm(counts)); length.append(roots[e['player']][h]['effective_decisions'])
            types = np.stack(types); length = np.array(length); observations[h] = types, length
            cost += (abs(np.log1p(length)[:, None] - np.log1p(length)[None, :]) + 1 - types @ types.T) / 3
        np.fill_diagonal(cost, np.inf); a, b = linear_sum_assignment(cost)
        for i, j in zip(a, b):
            player, donor = cohort[i]['player'], cohort[j]['player']; assignments[player] = donor
            for h, (types, length) in observations.items():
                details.append(dict(player=player, donor=donor, family=family, scenario=scenario, history=h,
                    effective_decisions=int(length[i]), donor_effective_decisions=int(length[j]),
                    public_type_cosine=float(types[i] @ types[j])))
    (OUT / 'matching.json').write_text(json.dumps(dict(assignments=assignments, details=details), indent=2) + '\n')
    print(json.dumps(dict(matched_players=len(assignments), same_family=True, same_scenario=True, self_matches=sum(p==d for p,d in assignments.items()))), flush=True)


def query_job(entry):
    torch.set_num_threads(1); game = load_cases(entry)[entry['query']]
    donor = json.loads((OUT / 'matching.json').read_text())['assignments'][entry['player']]
    roots = {c: torch.load(OUT / 'memory' / f'{p}.pt', weights_only=True)['roots'] for c, p in [('correct', entry['player']), ('wrong', donor)]}
    template = start_game(None, game, entry, 43117 + entry['group']); records = []
    try:
        for condition, h in [('none', 0)] + [(c, h) for h in HISTORY[1:] for c in ('correct', 'wrong')]:
            model = template.fork()
            try:
                if h:
                    root = roots[condition][h]; rng = np.random.default_rng(8513 + entry['group'])
                    indices = rng.permutation(systematic(root['weights'].numpy(), model.n, rng))
                    model.z = root['z'][indices].to(model.device).clone(); model.start_z = model.z.clone()
                    model.path_start_z = model.z.float().cpu().tolist(); model.cached = None; model.games_seen = h + 1
                rows = replay(model, game, entry, f'{condition}_{h}', limit=10, scoring=True)
                for row in rows: row.update(condition=condition, history=h)
                records.extend(rows)
            finally: model.close()
            (OUT / 'query' / f"{entry['player']}.json").write_text(json.dumps(dict(player=entry['player'], complete=len(records)==70, records=records), allow_nan=False))
        return entry['player']
    finally: template.close()


def estimate(values):
    arrays = {p: np.asarray(v, dtype=float) for p, v in values.items() if len(v)}
    if not arrays: return dict(value=None, players=0, count=0, interval_95=None)
    means = np.array([v.mean() for v in arrays.values()]); rng = np.random.default_rng(2027)
    draws = means[rng.integers(len(means), size=(2000, len(means)))].mean(1)
    return dict(value=float(means.mean()), players=len(means), count=sum(len(v) for v in arrays.values()),
                interval_95=np.quantile(draws, [.025, .975]).tolist() if len(means)>1 else None)


def player_values(rows, metric):
    values = defaultdict(list)
    if metric == 'profile_mae':
        behavior = defaultdict(lambda: defaultdict(list))
        for r in rows:
            for name, result in r['profiles'].items(): behavior[r['player']][name].append(result['absolute_error'])
        for player, fields in behavior.items(): values[player] = [float(np.mean(v)) for v in fields.values()]
    else:
        for r in rows:
            if metric in BEHAVIOR:
                if metric in r['profiles']: values[r['player']].append(r['profiles'][metric]['absolute_error'])
            elif metric in BEHAVIORS:
                if metric in r['card_profiles']: values[r['player']].append(r['card_profiles'][metric]['absolute_error'])
            elif r.get(metric) is not None: values[r['player']].append(r[metric])
    return dict(values)


def comparison(a, b, metric):
    av, bv = player_values(a, metric), player_values(b, metric)
    assert av.keys() == bv.keys() and all(len(av[p]) == len(bv[p]) for p in av)
    return estimate({p: [float(np.mean(av[p]) - np.mean(bv[p]))] for p in av})


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protocol = json.loads((OUT / 'protocol.json').read_text()); rows = []; support = []
    for entry in protocol['cohort']:
        q = json.loads((OUT / 'query' / f"{entry['player']}.json").read_text()); assert q['complete']
        rows.extend(q['records'])
        support.extend(json.loads((OUT / 'support' / f"{entry['player']}.json").read_text())['records'])
    summary = {}; paired = {}; metrics = ('expected_nll', 'kl', 'tv', 'nll', 'top1', 'profile_mae', 'unsupported_teacher_mass', *BEHAVIOR)
    for scope in ('personalized', 'random', 'all'):
        population = [r for r in rows if scope == 'all' or (r['family'] == 'random') == (scope == 'random')]
        summary[scope] = {}; paired[scope] = {}
        for horizon in (5, 10):
            summary[scope][horizon] = {}; paired[scope][horizon] = {}
            for h in HISTORY:
                selections = {c: [r for r in population if r['step'] < horizon and
                    (r['condition'], r['history']) == (('none', 0) if c == 'none' or h == 0 else (c, h))]
                    for c in ('correct', 'none', 'wrong')}
                summary[scope][horizon][h] = {c: {m: estimate(player_values(rr, m)) for m in metrics} for c, rr in selections.items()}
                paired[scope][horizon][h] = {c: {m: comparison(selections['correct'], selections[c], m) for m in metrics} for c in ('none', 'wrong')}
    card = {}; quality = {}
    for condition, h in sorted({(r['condition'], r['history']) for r in rows}):
        selected = [r for r in rows if (r['condition'], r['history']) == (condition, h)]
        quality[f'{condition}_{h}'] = dict(decisions=len(selected), invalid_predictions=sum(not r['valid_prediction'] for r in selected),
            failed_updates=sum(not r['valid_update'] for r in selected), missing_teachers=sum(not r['teacher_valid'] for r in selected),
            missing_actions=sum(r['nll'] is None for r in selected), maximum_action_mass_error=max(r['action_mass_error'] for r in selected))
        card[f'{condition}_{h}'] = {m: estimate(player_values([r for r in selected if r['family'] != 'random'], m)) for m in BEHAVIORS}
    profiles = []
    for entry in protocol['cohort']:
        data = [r for r in rows if r['player'] == entry['player'] and r['condition'] == 'correct' and r['history'] == 5]
        fields = {}
        for name in BEHAVIOR:
            values = [r['profiles'][name] for r in data if name in r['profiles']]
            if values:
                fields[name] = dict(opportunities=len(values), predicted_probability=float(np.mean([r['predicted'] for r in values])) if all(r['predicted'] is not None for r in values) else None,
                                    teacher_probability=float(np.mean([r['truth'] for r in values])))
        profiles.append(dict(player=entry['player'], family=entry['family'], query=entry['query'], behaviors=fields))
    subgroups = {}
    for field in ('family', 'scenario'):
        subgroups[field] = {}
        for value in sorted({r[field] for r in rows}):
            selected = {c: [r for r in rows if r[field] == value and (r['condition'], r['history']) == (('none', 0) if c == 'none' else (c, 5))]
                        for c in ('correct', 'none', 'wrong')}
            subgroups[field][value] = dict(metrics={c: {m: estimate(player_values(rr, m)) for m in metrics} for c, rr in selected.items()},
                paired_correct_minus_control={c: {m: comparison(selected['correct'], selected[c], m) for m in metrics} for c in ('none', 'wrong')})
    result = dict(protocol=protocol, matching=json.loads((OUT / 'matching.json').read_text()), summary=summary, subgroups=subgroups,
        paired_correct_minus_control=paired, dossier_opportunities=card, query_quality=quality,
        memory_extraction=dict(players=50, games=250, decisions=len(support), valid_updates=sum(r['valid_update'] for r in support)),
        player_profiles=profiles)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, metric, title in zip(axes, ('kl', 'profile_mae'), ('Action distribution KL', 'Behavior probability MAE')):
        for condition, label, color in [('correct', 'Correct player', '#16704a'), ('none', 'No history', '#777777'), ('wrong', 'Other player, same family', '#b94e37')]:
            points = [summary['personalized'][10][h][condition][metric] for h in HISTORY]
            values = np.array([r['value'] for r in points]); bounds = np.array([r['interval_95'] for r in points])
            ax.plot(HISTORY, values, 'o-', color=color, label=label); ax.fill_between(HISTORY, bounds[:, 0], bounds[:, 1], color=color, alpha=.12)
        ax.set(xlabel='Previous games', ylabel=title + ' (lower is better)', xticks=HISTORY); ax.grid(alpha=.2)
    axes[0].legend(fontsize=9); fig.suptitle('Player profiles on new teams: 40 held-out players, first 10 decisions')
    fig.tight_layout(); fig.savefig(OUT / 'profile_transfer.png', dpi=180); fig.savefig(OUT / 'profile_transfer.pdf'); plt.close(fig)
    print(json.dumps(dict(summary=summary['personalized'], paired=paired['personalized'], quality=quality), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'support', 'match', 'query', 'report'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    for directory in ('cases', 'memory', 'support', 'query'): (OUT / directory).mkdir(parents=True, exist_ok=True)
    if args.command == 'prepare': prepare(args.workers)
    elif args.command == 'match': match()
    elif args.command == 'report': report()
    else:
        entries = json.loads((OUT / 'protocol.json').read_text())['cohort']
        selected = [e for e in entries if not (OUT / args.command / f"{e['player']}.json").exists()
                    or not json.loads((OUT / args.command / f"{e['player']}.json").read_text()).get('complete')]
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(support_job if args.command == 'support' else query_job, e) for e in selected]):
                print('completed', args.command, future.result(), flush=True)
