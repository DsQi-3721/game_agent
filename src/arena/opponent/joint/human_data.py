"""Human replay observations, with partial labels rather than reconstructed truth.

Examples predict a public resolution window. Executed actions are diagnostics;
they are not substituted for unobserved submissions or a complete legal mask.
"""
import re
from pathlib import Path

import pyarrow.parquet as pq


def identifier(value):
    return re.sub('[^a-z0-9]', '', value.lower())


ROOM_EVENTS = {
    '', 'j', 'join', 'l', 'leave', 'n', 'name', 'c', 'c:', 'chat', 't:', ':',
    'raw', 'html', 'uhtml', 'uhtmlchange', 'inactive', 'inactiveoff', 'player',
    'message', '-message', 'debug', 'error', 'request', 'badge', 'init', 'title',
}


def public_lines(text, winner_side=None):
    """Keep complete battle protocol fields; remove chat and account-name fields."""
    result = []
    for line in text.splitlines():
        parts = line.split('|')
        if len(parts) < 2 or parts[1].lower() in ROOM_EVENTS:
            continue
        if parts[1] == 'win':
            line = '|win|' + (winner_side or 'unknown')
        line = re.sub(r'(?<=\|)(p[12]): [^|]*', r'\1', line)
        result.append(line)
    return result


def windows(game):
    """Both players predict from the same pre-resolution boundary."""
    lines = game['public']
    starts = []
    for i, line in enumerate(lines):
        if line.startswith('|teampreview'):
            starts.append((i + 1, 0, 'preview'))
        elif line.startswith('|turn|'):
            starts.append((i + 1, int(line.split('|')[2]), 'turn'))
    for index, (cutoff, turn, phase) in enumerate(starts):
        end = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        yield {'cutoff': cutoff, 'end': end, 'turn': turn, 'phase': phase}


def executed_actions(game, window):
    """First executed move or early voluntary switch, not later pivot choices.

Missing/cancelled intent remains null. Replacement switches and called moves
remain in the event target even when they do not supply a primary action label.
"""
    actions = [None, None]
    reasons = ['unobserved', 'unobserved']
    tera = [None, None]
    move_seen = False
    displaced = set()
    for i in range(window['cutoff'], window['end']):
        p = game['public'][i].split('|')
        if len(p) < 3 or p[2][:2] not in ('p1', 'p2'):
            continue
        side = int(p[2][1]) - 1
        kind = p[1]
        if kind == '-terastallize':
            tera[side] = identifier(p[3])
        if kind in ('drag', 'replace'):
            displaced.add(side)
        if kind in ('cant', 'faint') and actions[side] is None:
            reasons[side] = 'prevented_or_cancelled'
        if kind == 'move':
            move_seen = True
        if actions[side] is not None or side in displaced:
            continue
        if kind == 'move' and not any(x.startswith('[from]') for x in p[4:]):
            actions[side] = {'kind': 'move', 'value': identifier(p[3]), 'event_index': i}
            reasons[side] = 'executed_only'
        elif kind == 'switch' and not move_seen and not any(x.startswith('[from]') for x in p[4:]):
            actions[side] = {'kind': 'lead' if window['phase'] == 'preview' else 'switch',
                             'value': identifier(p[3].split(',')[0]), 'event_index': i}
            reasons[side] = 'executed_only'
    return actions, reasons, tera


def revealed_facts(game):
    """Event-local facts. Item acquisition/consumption separates initial/current.

Subject is the publicly displayed combatant, not an invented true team slot.
Illusion/Transform observations therefore never provide initial-item labels.
"""
    item_changed = set()
    uncertain_identity = set()
    for side, team in zip(('p1', 'p2'), game['preview']):
        if any('zoroark' in s or 'zorua' in s for s in team):
            uncertain_identity.add(side)
    facts = []
    for i, line in enumerate(game['public']):
        p = line.split('|')
        if len(p) < 4 or p[2][:2] not in ('p1', 'p2'):
            continue
        actor, field, value, scope = p[2], None, None, 'current'
        if p[1] == '-transform':
            uncertain_identity.add(actor[:2])
        if p[1] in ('-item', '-enditem'):
            acquisition = p[1] == '-item' and any(x.startswith('[from]') for x in p[4:])
            if acquisition:
                item_changed.add(actor)
            field, value = 'item', identifier(p[3])
            if actor not in item_changed and actor[:2] not in uncertain_identity:
                scope = 'initial'
            if p[1] == '-enditem':
                item_changed.add(actor)
        elif p[1] == '-ability':
            field, value = 'ability', identifier(p[3])
        elif p[1] == '-terastallize':
            field, value = 'tera_type', identifier(p[3])
        elif p[1] == 'move' and not any(x.startswith('[from]') for x in p[4:]):
            field, value = 'executed_move', identifier(p[3])
        if field:
            facts.append({'event_index': i, 'side': actor[:2], 'subject': actor,
                          'field': field, 'value': value, 'scope': scope})
        source_item = next((x[len('[from] item:'):].strip() for x in p[4:]
                            if x.startswith('[from] item:')), None)
        if source_item:
            holder = next((x[len('[of] '):] for x in p[4:] if x.startswith('[of] ')), actor)
            if holder[:2] in ('p1', 'p2'):
                initial = holder not in item_changed and holder[:2] not in uncertain_identity
                facts.append({'event_index': i, 'side': holder[:2], 'subject': holder,
                              'field': 'item', 'value': identifier(source_item),
                              'scope': 'initial' if initial else 'current'})
    # Repeated Leftovers heals or repeated use of an already revealed move do
    # not create new hidden-attribute labels. The full events remain available.
    known, fresh = {}, []
    for fact in facts:
        key = (fact['subject'], fact['field'])
        if fact['field'] == 'executed_move':
            key += (fact['value'],)
        value = (fact['value'], fact['scope'])
        if known.get(key) != value:
            fresh.append(fact)
        known[key] = value
    return fresh


def training_example(game, window, side, facts=None):
    actions, reasons, tera = executed_actions(game, window)
    index = int(side[1]) - 1
    facts = revealed_facts(game) if facts is None else facts
    return {
        'battle_id': game['battle_id'], 'player_id': game['players'][index],
        'input': {'format': game['format'], 'side': side, 'turn': window['turn'],
                  'phase': window['phase'], 'public': game['public'][:window['cutoff']]},
        'labels': {
            'public_events': game['public'][window['cutoff']:window['end']],
            'executed_action': actions[index], 'action_observation': reasons[index],
            'observed_tera_type': tera[index],
            'revealed_facts': [f for f in facts if f['side'] == side and
                               window['cutoff'] <= f['event_index'] < window['end']],
            'submitted_action': None, 'complete_configuration': None,
            'private_information_state': None, 'strategy_parameters': None,
        },
    }


def human_games(directory, split):
    for path in sorted((Path(directory) / split).glob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            yield from batch.to_pylist()


def human_examples(directory, split):
    for game in human_games(directory, split):
        facts = revealed_facts(game)
        for window in windows(game):
            for side in ('p1', 'p2'):
                yield training_example(game, window, side, facts)


def read_game(directory, file, row):
    """Random access using the locations in players.parquet."""
    parquet = pq.ParquetFile(Path(directory) / file)
    for group in range(parquet.num_row_groups):
        size = parquet.metadata.row_group(group).num_rows
        if row < size:
            return parquet.read_row_group(group).slice(row, 1).to_pylist()[0]
        row -= size
    raise IndexError(row)


def player_history(directory, battle_id, side, limit=8):
    """Earlier public games of this source-local account, oldest first.

    The link excludes equal timestamps/months. No past-game hidden labels or
    current-game future events are supplied to the model.
    """
    path = Path(directory) / 'players.parquet'
    current = pq.read_table(path, filters=[('battle_id', '=', battle_id), ('side', '=', side)]).to_pylist()[0]
    rows = pq.read_table(path, filters=[('player_id', '=', current['player_id'])]).to_pylist()
    by_id = {(r['battle_id'], r['side']): r for r in rows}
    history = []
    while current['previous_battle_id'] and len(history) < limit:
        current = by_id[(current['previous_battle_id'], current['previous_side'])]
        game = read_game(directory, current['file'], current['row'])
        history.append({'battle_id': game['battle_id'], 'side': current['side'],
                        'format': game['format'], 'public': game['public']})
    return history[::-1]
