"""Evaluate the unmodified Foul Play tracker and native set sampler."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy, deepcopy
from functools import lru_cache
import itertools
import json
import logging
from pathlib import Path
import pickle
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE/'repo'), str(ROOT/'scripts')]
sys.path.append(str(HERE/'venv/lib/python3.12/site-packages'))
from opp_baseline_topk import BASE, OUT, Scores, FIELDS
from fp.battle.state import Battle, LastUsedMove
from fp.battle.protocol import process_battle_updates
from fp.battle.helpers import normalize_name
from fp.config import FoulPlayConfig
from fp.data import pokedex
from fp.modes.standard_battle import StandardBattleMode
from fp.search.standard_battles import sample_pokemon
from fp.battle import inference
from fp.battle import helpers, state
from fp.data.sets import base as set_base

NATIVE_DAMAGE = inference.poke_engine_get_damage_rolls
LAST_DAMAGE_BATTLE = None
DAMAGE_CACHE = {}
NATIVE_STATS = helpers.calculate_stats


@lru_cache(maxsize=32768)
def stat_values(base, level, ivs, evs, nature):
    return NATIVE_STATS(dict(base),level,ivs,evs,nature)


def cached_stats(base_stats, level, ivs=(31,)*6, evs=(85,)*6, nature='serious'):
    return stat_values(tuple(base_stats.items()),level,tuple(ivs),tuple(evs),nature).copy()


def cached_damage(battle, move1, move2, first):
    # _do_check changes only these opponent fields between candidate sets.
    # Tera variants often produce exactly the same damage query.
    global LAST_DAMAGE_BATTLE
    if battle is not LAST_DAMAGE_BATTLE:
        DAMAGE_CACHE.clear(); LAST_DAMAGE_BATTLE = battle
    p = battle.opponent.active
    key = (p.ability,p.item,p.hp,p.max_hp,tuple(p.stats.items()),p.nature,tuple(p.evs),move1,move2,first)
    if key not in DAMAGE_CACHE: DAMAGE_CACHE[key] = NATIVE_DAMAGE(battle,move1,move2,first)
    return DAMAGE_CACHE[key]


def initialize(samples, save=False):
    global TEMPLATE, SAMPLES, SAVE
    SAMPLES = samples
    SAVE = save
    logging.disable(logging.CRITICAL)
    FoulPlayConfig.pokemon_format = 'gen9ou'
    inference.poke_engine_get_damage_rolls = cached_damage
    for module in (helpers,state,set_base): module.calculate_stats = cached_stats
    # Showdown preview aliases (e.g. Keldeo-Resolute) need the same mapping
    # as the existing offline Foul Play bridge.
    for entry in pokedex.values():
        if entry.get('changesFrom') and not entry.get('otherFormes'):
            entry['otherFormes'] = [entry['changesFrom']]
    names = set(json.loads((BASE/'catalog.json').read_text())['bySpecies'])
    TEMPLATE = StandardBattleMode()
    TEMPLATE.initialize_team_preview_datasets('gen9ou', names, '')


def fresh_mode(names):
    mode = copy(TEMPLATE)
    for attr in ('team_datasets','smogon_sets'):
        dataset = copy(getattr(TEMPLATE,attr))
        def included(k): return any(k.startswith(n) or n.startswith(k) for n in names)
        dataset.pkmn_sets = {k:list(v) for k,v in dataset.pkmn_sets.items() if included(k)}
        dataset.raw_pkmn_sets = {k:v for k,v in dataset.raw_pkmn_sets.items() if included(k)}
        if attr == 'team_datasets': dataset.raw_pkmn_moves = {k:v for k,v in dataset.raw_pkmn_moves.items() if included(k)}
        setattr(mode,attr,dataset)
    return mode


def distribution(pokemon, mode, cache):
    fields = ('name','base_name','mega_name','level','item','removed_item','ability',
              'terastallized','tera_type','can_have_choice_item','impossible_items','impossible_abilities')
    key = pickle.dumps(([getattr(pokemon,k) for k in fields], tuple(pokemon.speed_range), [m.name for m in pokemon.moves],
                        len(mode.team_datasets.get_pkmn_sets_from_pkmn_name(pokemon)),
                        len(mode.smogon_sets.get_pkmn_sets_from_pkmn_name(pokemon))))
    if key in cache: return cache[key]
    counts = {name:Counter() for name in FIELDS}
    # All 64 draws condition on the same observation. Reuse native candidate
    # filtering; sampled sets/moves and their probabilities remain unchanged.
    methods = []
    for obj,name in [(mode.team_datasets,'get_all_remaining_sets'),
                     (mode.team_datasets,'get_all_possible_move_combinations'),
                     (mode.smogon_sets,'get_all_remaining_trait_combinations')]:
        original = getattr(obj,name); memo = {}
        def cached(p, *args, original=original, memo=memo):
            k = (p.name,p.item,p.ability,p.tera_type,tuple(m.name for m in p.moves),
                 tuple(tuple(vars(a).items()) for a in args))
            if k not in memo: memo[k] = original(p,*args)
            return memo[k]
        methods.append((obj,name,original)); setattr(obj,name,cached)
    for _ in range(SAMPLES):
        sample = deepcopy(pokemon)
        sample_pokemon(sample,mode)
        for move in sample.moves: counts['moves'][move.name] += 1
        for name,value in [('item',sample.item),('ability',sample.ability),('tera',sample.tera_type)]:
            if value and value not in ('unknownitem','typeless'): counts[name][value] += 1
    for obj,name,original in methods: setattr(obj,name,original)
    cache[key] = counts
    return counts


def evaluate_game(game, side, scores):
    opponent = side
    observer = 'p2' if side == 'p1' else 'p1'
    case = game['cases'][side]
    names = {normalize_name(line.split('|')[3].split(',')[0]) for line in game['public'][:case['cutoffs'][0]] if line.startswith('|poke|')}
    mode = fresh_mode(names)
    battle = Battle(game['id'])
    battle.pokemon_format = 'gen9ou'; battle.generation = battle.format_spec.generation
    battle.battle_type = battle.format_spec.battle_type; battle.mode = mode
    battle.opponent.name = opponent
    # Only the observer's own private request/configuration enter the tracker.
    # The modeled player's request, submitted action and hidden truth never do.
    first = next(d['request'] for d in game['decisions'] if d['side']==observer)
    battle.user.team_dict = [{**m,'species':normalize_name(m['species']),'nature':normalize_name(m['nature'])} for m in game['teams'][observer]]
    battle.user.initialize_first_turn_user_from_json(first)
    preview = [line.split('|')[3] for line in game['public'][:case['cutoffs'][0]] if line.startswith(f'|poke|{side}|')]
    battle.initialize_team_preview(preview,'gen9ou')
    order = list(battle.opponent.reserve)
    battle.request_json = first
    cutoff = case['cutoffs'][0]; t = 0; cache = {}
    for end, decisions in itertools.groupby(game['decisions'],key=lambda d:d['cutoff']):
        decisions = list(decisions)
        ours = next((d for d in decisions if d['side']==observer),None)
        if ours is not None:
            battle.request_json = ours['request']
            battle.force_switch = bool(ours['request'].get('forceSwitch'))
        battle.msg_list = game['private'][observer][cutoff:end]
        process_battle_updates(battle)
        cutoff = end
        if any(d['side']==side for d in decisions):
            predictions = [distribution(mon,mode,cache) for mon in order]
            scores.add(predictions,case['truth'],case['hidden'][t]); t += 1
        if ours is not None and not ours['request'].get('teamPreview'):
            battle.user.update_from_request_json(ours['request'])
            action = ours['selected']
            move = 'switch '+normalize_name(action['species']) if action['kind']=='switch' else action.get('move','none')
            battle.user.last_selected_move = LastUsedMove(battle.user.active.name,move,battle.turn)
    assert t == len(case['cutoffs'])
    return t, len(cache)


def evaluate_group(path):
    destination=OUT/'foulplay_groups'/f'{path.stem}.json'
    if SAVE and destination.exists(): return json.loads(destination.read_text())
    random.seed(2027+int(path.stem))
    scores=Scores(); n=0; decisions=0; states=0; started=time.time()
    for game in pickle.loads(path.read_bytes()):
        for side in game['cases']:
            try:
                d,s = evaluate_game(game,side,scores)
            except Exception as exc:
                raise RuntimeError(f'{game["id"]} {side}: {exc}') from exc
            n+=1; decisions+=d; states+=s
    result=dict(player_games=n,decision_points=decisions,sampled_states=states,metrics=scores.result(),seconds=time.time()-started)
    if SAVE: destination.write_text(json.dumps(result))
    return result


def main(args):
    global OUT
    OUT=args.output
    paths=sorted((OUT/'cases').glob('*.pkl'))
    if args.groups: paths=paths[:args.groups]
    started=time.time(); totals=dict(player_games=0,decision_points=0,sampled_states=0)
    scores=Scores()
    (OUT/'foulplay_groups').mkdir(exist_ok=True)
    failures=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(args.samples,not args.groups)) as pool:
        futures={pool.submit(evaluate_group,path):path for path in paths}
        for i,future in enumerate(as_completed(futures)):
            try: r=future.result()
            except Exception as exc:
                failures.append(dict(group=futures[future].stem,error=str(exc)))
                print(json.dumps(failures[-1]),flush=True);continue
            for k in totals: totals[k]+=r[k]
            for name,m in r['metrics'].items():
                scores.count[name]+=m['count']; scores.empty[name]+=m['empty_predictions']
                scores.hits[name]=[a+b for a,b in zip(scores.hits[name],m['hits'])]
            print(json.dumps(dict(groups=i+1,seconds=round(time.time()-started,1),**totals)),flush=True)
    if failures:
        (OUT/'foulplay_errors.json').write_text(json.dumps(failures,indent=2))
        raise RuntimeError(f'{len(failures)} groups failed; successful groups are saved for resuming.')
    result=dict(method='Foul Play native tracker + set sampler',category='traditional',source='https://github.com/pmariglia/foul-play',
                commit='8f541f8',samples=args.samples,**totals,metrics=scores.result()|{'action':None},seconds=time.time()-started,
                input='Public history plus observer own legitimate private configuration/requests; no opponent hidden labels or future events.',
                prior='Official cached Showdown/Smogon/replay set tables bundled with the local Foul Play reproduction; not refit on this test set.',
                action='Unsupported: MCTS decisions are not opponent-action predictions.')
    if not args.groups:
        reference=json.loads((OUT/'cases.json').read_text())
        assert result['decision_points']==reference['decisions']
        assert scores.count==reference['counts']
    destination=OUT/('foulplay_probe.json' if args.groups else 'foulplay.json')
    destination.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=12);p.add_argument('--samples',type=int,default=64);p.add_argument('--groups',type=int,default=0)
    p.add_argument('--output',type=Path,default=OUT)
    main(p.parse_args())
