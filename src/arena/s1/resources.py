"""Recorded battle resources and their persistent, player-visible history."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

from poke_env.data import to_id_str

from arena.observation.tracker import mon_key


def pp_band(pp, maximum):
    if pp is None or maximum is None or maximum <= 0:
        return None
    ratio = pp / maximum
    return int(ratio >= .5) + int(ratio >= .25) + int(ratio > 0)


def resources(battle, obs):
    own = {}
    for row in obs.own_team:
        mon = battle.team.get(row['ident'])
        cur, maximum = row['hp']
        entry = {'species': row['species'], 'active': row['active'], 'fainted': row['fainted'],
                 'hp_fraction': row.get('hp_fraction', cur / maximum if maximum else 0. if row['fainted'] else None),
                 'status': row['status'], 'item': deepcopy(row['item']),
                 'ability': row.get('ability') if row.get('ability') not in ('', 'unknownability') else None,
                 'tera_type': row.get('tera_type'),
                 'terastallized': row.get('terastallized'), 'moves': {}}
        if 'stats' not in obs.unrecorded:
            entry['hp'] = list(row['hp'])
            entry['stats'] = row.get('stats')
        if mon is not None:
            entry.update(level=mon.level, types=[t.name.lower() for t in mon.types if t],
                         base_stats=dict(mon.base_stats))
        if 'recorded_types' in row:
            entry.update(types=row['recorded_types'], last_observed_effect=row['last_observed_effect'],
                         knowledge=row['knowledge'])
        if row.get('active'):
            entry.update(boosts=deepcopy(row.get('boosts', {})), effects=deepcopy(row.get('effects', [])))
        for move in row['moves']:
            pp, maximum = move.get('pp'), move.get('maxpp')
            entry['moves'][move['id']] = {'pp': pp, 'max_pp': maximum, 'band': pp_band(pp, maximum),
                                         'knowledge': move.get('pp_knowledge', 'unrecorded')}
        own[row['ident']] = entry
    opposing = {}
    for row in obs.opponent_revealed_team:
        mon = battle.opponent_team.get(row['ident'])
        entry = {key: deepcopy(row.get(key)) for key in ('species', 'active', 'fainted', 'level', 'hp_pct',
            'hp_knowledge', 'hp_seen_turn', 'status', 'item', 'ability', 'terastallized', 'boosts', 'effects')}
        entry['moves'] = {m['id']: {'observed_pp_used': m.get('pp_used')} for m in row['revealed_moves']}
        if mon is not None:
            entry.update(types=[t.name.lower() for t in mon.types if t], base_stats=dict(mon.base_stats))
        opposing[row['ident']] = entry
    source = getattr(battle, '_arena_source_state', None)
    previous_moves = {'own': None, 'opponent': None}
    if source is not None:
        active = next((m for m in opposing.values() if m['active']), None)
        if active is not None:
            recorded = source['opponent_active_pokemon']
            active['hp_fraction'] = recorded.get('hp_pct')
            active['last_observed_effect'] = recorded.get('effect')
        for side, field in (('own', 'player_prev_move'), ('opponent', 'opponent_prev_move')):
            move = (source.get(field) or {}).get('name')
            previous_moves[side] = move if move != 'nomove' else None
    else:
        for event in getattr(battle, '_replay_data', []):
            if len(event) > 3 and event[1] == 'move':
                side = 'own' if event[2].startswith(battle.player_role) else 'opponent'
                previous_moves[side] = to_id_str(event[3])
    return {'own': own, 'opponent': opposing, 'previous_moves': previous_moves, 'tera': deepcopy(obs.tera),
            'side_conditions': deepcopy(obs.side_conditions), 'weather': deepcopy(obs.weather),
            'field_conditions': deepcopy(obs.field_conditions), 'incoming': deepcopy(obs.slot_conditions),
            'unrecorded': list(obs.unrecorded)}


class ResourceLedger:
    """Retain observed consumption and resource events beyond the recent decision window."""

    def __init__(self):
        self.cursor = 0
        self.turn = 0
        self.moves = {}
        self.hp = {}
        self.hp_changes = {}
        self.pp = {}
        self.pp_changes = {}
        self.events = {}
        self.statuses = {}

    @staticmethod
    def fraction(condition):
        token = condition.split()[0]
        if '/' in token:
            numerator, denominator = token.split('/')
            return float(numerator) / float(denominator) if float(denominator) else None
        return 0. if token == '0' else None

    def ingest(self, prefix):
        persistent = {'faint', '-terastallize', '-enditem', '-item', '-ability', '-status', '-curestatus',
                      '-sidestart', '-sideend', '-weather', '-fieldstart', '-fieldend', '-fail', '-immune'}
        for index in range(self.cursor, len(prefix)):
            parts = prefix[index]
            if isinstance(parts, str):
                parts = parts.split('|')
            if len(parts) < 2:
                continue
            kind = parts[1]
            if kind == 'turn':
                self.turn = int(parts[2])
                continue
            if len(parts) < 3:
                continue
            actor = mon_key(parts[2])
            if kind == 'move' and len(parts) > 3:
                self.moves.setdefault(actor, Counter())[to_id_str(parts[3])] += 1
            if kind in ('switch', 'drag', 'replace') and len(parts) > 4:
                self.hp[actor] = self.fraction(parts[4])
            if kind in ('-damage', '-heal', '-sethp') and len(parts) > 3:
                pairs = [(parts[2], parts[3])]
                if kind == '-sethp' and len(parts) > 5 and parts[4].startswith(('p1', 'p2')):
                    pairs.append((parts[4], parts[5]))
                for who, condition in pairs:
                    key, current = mon_key(who), self.fraction(condition)
                    previous = self.hp.get(key)
                    if previous is not None and current is not None:
                        change = self.hp_changes.setdefault(key, Counter())
                        change['lost_fraction'] += max(0., previous - current)
                        change['restored_fraction'] += max(0., current - previous)
                    self.hp[key] = current
            if kind == '-status' and len(parts) > 3:
                self.statuses.setdefault(actor, set()).add(parts[3])
            if kind in persistent:
                detail = '|'.join([kind, actor, *parts[3:]])
                event = self.events.setdefault(detail, {'first_turn': self.turn, 'count': 0})
                event.update(last_turn=self.turn, last_sequence=index)
                event['count'] += 1
        self.cursor = len(prefix)

    def observe(self, current, prefix):
        self.ingest(prefix)
        for ident, mon in {**current['own'], **current.get('opponent', {})}.items():
            status = mon.get('status')
            if status and status != 'fnt':
                self.statuses.setdefault(ident, set()).add(status.lower())
            if not prefix:
                hp = mon.get('hp_fraction')
                if hp is not None:
                    before = self.hp.get(ident)
                    if before is not None:
                        change = self.hp_changes.setdefault(ident, Counter())
                        change['lost_fraction'] += max(0., before - hp)
                        change['restored_fraction'] += max(0., hp - before)
                    self.hp[ident] = hp
        for ident, mon in current['own'].items():
            for move, value in mon['moves'].items():
                key, pp = f'{ident}/{move}', value['pp']
                before = self.pp.get(key)
                if pp is not None and before is not None:
                    change = self.pp_changes.setdefault(key, Counter())
                    change['decrease'] += max(0, before - pp)
                    change['restore'] += max(0, pp - before)
                self.pp[key] = pp
        return {'move_executions': {key: dict(value) for key, value in self.moves.items()},
                'observed_hp_changes': {key: {k: round(v, 6) for k, v in value.items()}
                                        for key, value in self.hp_changes.items()},
                'recorded_pp_changes': {key: dict(value) for key, value in self.pp_changes.items() if any(value.values())},
                'statuses_seen': {key: sorted(value) for key, value in self.statuses.items()},
                'resource_events': deepcopy(self.events),
                'event_coverage': 'player_protocol' if prefix else 'decision states only; between-request events unrecorded'}


def policy_resources(battle, observation):
    current = resources(battle, observation)
    ledger = getattr(battle, '_arena_policy_resources', None)
    if ledger is None:
        ledger = battle._arena_policy_resources = ResourceLedger()
    current['history'] = ledger.observe(current, getattr(battle, '_replay_data', []))
    return current
