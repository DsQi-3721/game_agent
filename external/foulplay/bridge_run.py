"""Foul Play bridge (runs INSIDE external/foulplay/venv; never on the arena import path).

Foul Play (pmariglia/foul-play: MCTS on the Rust poke-engine; PokéAgent Challenge 2025 Gen 9 OU champion) runs
unmodified through its own ``fp.main.run_foul_play``. This shim only adapts it to our local Showdown server:
  - login: the local server runs with --no-security, so the name is claimed with ``/trn <name>,0,`` instead of
    asking play.pokemonshowdown.com for a guest assertion;
  - no battle timer: Foul Play turns the timer on at every battle start, our other opponents do not, so that
    request is dropped to keep the conditions the same (its own thinking time is fixed by --search-time-ms).
  - formes set at team validation (Keldeo -> Keldeo-Resolute when it knows Secret Sword) are linked back to their
    base species, so Foul Play finds them in its own team: its lookup otherwise raises "Could not find
    keldeoresolute in team_dict" at the first request and the battle never starts.
It challenges ARENA_OPPONENT (``challenger: them`` in the arena config): in accept mode a challenge arriving while
it is still leaving the previous battle would be read and dropped by that loop. Team: ARENA_TEAM_FILE (a
Showdown paste; mirror teams). Search: FOULPLAY_SEARCH_TIME_MS (per sampled state), FOULPLAY_PARALLELISM
(sampled opponent sets searched in parallel), FOULPLAY_THREADS (threads per state).
"""

import asyncio
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "repo"))
os.chdir(os.path.join(HERE, "repo"))

from fp.data import pokedex  # noqa: E402
from fp.websocket_client import PSWebsocketClient  # noqa: E402

for _entry in pokedex.values():            # before run_foul_play snapshots the pokedex for its mutation check
    if _entry.get("changesFrom") and not _entry.get("otherFormes"):
        _entry["otherFormes"] = [_entry["changesFrom"]]

_send = PSWebsocketClient.send_message


async def _local_login(self):
    await self.get_id_and_challstr()
    await _send(self, "", [f"/trn {self.username},0,"])
    await asyncio.sleep(1)
    return self.username


async def _send_without_timer(self, room, message_list):
    message_list = [m for m in message_list if m.strip() != "/timer on"]
    if message_list:
        await _send(self, room, message_list)


PSWebsocketClient.login = _local_login
PSWebsocketClient.send_message = _send_without_timer

sys.argv = ["run.py",
            "--websocket-uri", os.environ["ARENA_SERVER"],
            "--ps-username", os.environ["ARENA_USERNAME"],
            "--bot-mode", "challenge_user",
            "--user-to-challenge", os.environ["ARENA_OPPONENT"],
            "--pokemon-format", os.environ["ARENA_FORMAT"],
            "--team-name", os.environ["ARENA_TEAM_FILE"],
            "--search-time-ms", os.environ.get("FOULPLAY_SEARCH_TIME_MS", "2000"),
            "--search-parallelism", os.environ.get("FOULPLAY_PARALLELISM", "4"),
            "--search-threads", os.environ.get("FOULPLAY_THREADS", "1"),
            "--run-count", os.environ.get("ARENA_N_BATTLES", "1000"),
            "--log-level", os.environ.get("FOULPLAY_LOG_LEVEL", "INFO")]

from fp.main import run_foul_play  # noqa: E402

asyncio.run(run_foul_play())
