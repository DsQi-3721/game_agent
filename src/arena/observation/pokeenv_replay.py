"""Rebuild the poke-env ``Battle`` a live client held at each request, offline, from one player's stream.

Feeds a player's protocol lines and requests (live capture or simulator replay of a server log) through
poke-env the way ``Player._handle_battle_message`` does, and yields ``(request, battle)`` at every request.
This lets the current consumers (``legal_options``, S1 ``build_state``, renderers, HyMem views) be run on
exact strict-tier positions and checked against the request ground truth, without a server.

The yielded battle is the live object and keeps changing; read what you need before advancing.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

from poke_env.battle.battle import Battle
from poke_env.data import GenData

_LOG = logging.getLogger("arena.observation.pokeenv_replay")
_LOG.setLevel(logging.CRITICAL)

# poke-env Player.MESSAGES_TO_IGNORE plus lines the Player handles itself instead of the battle
_IGNORE = {"t:", "expire", "uhtmlchange", "tempnotify", "tempnotifyoff", "error", "bigerror", "showteam"}


def battle_states(lines: list[str], *, battle_tag: str, username: str, fmt: str) -> Iterator[tuple[dict, Battle]]:
    battle = Battle(battle_tag=battle_tag, username=username, logger=_LOG, gen=GenData.from_format(fmt).gen)
    for line in lines:
        split = line.split("|")
        if len(split) < 2:
            continue
        kind = split[1]
        if kind in _IGNORE:
            continue
        if kind == "request":
            if len(split) > 2 and split[2]:
                req = json.loads("|".join(split[2:]))
                battle.parse_request(req)
                yield req, battle
        elif kind == "win":
            battle.won_by(split[2])
        elif kind == "tie":
            battle.tied()
        else:
            battle.parse_message(split)
