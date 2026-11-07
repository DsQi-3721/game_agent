"""Shared player and replay helpers for opponent-model experiments."""
import json
from pathlib import Path
from arena.opponent.joint.filter import JointOpponentModel

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/opponent_joint'
DEX = json.loads((ROOT / 'data/knowledge/dex_gen9.json').read_text())['species']
TYPES = sorted({t for mon in DEX.values() for t in mon['types']})

def side_of(game, player):
    return 'p' + str(game['players'].index(player) + 1)

def roster(game, player):
    side = side_of(game, player)
    return [l.split('|')[3].split(',')[0] for l in game['public'] if l.startswith('|poke|' + side + '|')]

def start_game(model, game, entry, seed):
    side = side_of(game, entry['player']); observer = 'p2' if side == 'p1' else 'p1'
    initial = game['public'][:game['decisions'][0]['cutoff']]; preview = roster(game, entry['player'])
    if model is None:
        return JointOpponentModel(BASE / 'training/adaptation/model.pt', BASE / 'catalog.json',
            game['teams'][observer], preview, observer=observer, particles=1024, configurations=128,
            branches=32, move_budget=32, time_budget=60, device=f"cuda:{entry['group'] % 2}", seed=seed,
            game_id=game['id'], opponent_id=entry['player'], initial_events=initial)
    model.new_game(game['teams'][observer], preview, game['id'], initial_events=initial, observer=observer)
    return model

def windows(game,observer):
    private=[];cursor=0
    for line in game['public']:
        while game['private'][observer][cursor].startswith('|-hint|') and game['private'][observer][cursor]!=line:cursor+=1
        private.append(game['private'][observer][cursor]);cursor+=1
    starts=[i for i,d in enumerate(game['decisions']) if not i or d['cutoff']!=game['decisions'][i-1]['cutoff']]
    for k,i in enumerate(starts):
        j=starts[k+1] if k+1<len(starts) else len(game['decisions'])
        a=game['decisions'][i]['cutoff'];z=game['decisions'][j]['cutoff'] if j<len(game['decisions']) else len(game['public'])
        yield dict(events=game['public'][a:z],private_events=private[a:z],
            our_command=next((c for s,c in game['commands'][i:j] if s==observer),None),
            event_id=f'{game["id"]}:{k}')
