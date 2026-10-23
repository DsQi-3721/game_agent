"""Stub external bot for validating the external-opponent bridge end-to-end.

A plain poke-env MaxBasePowerPlayer that logs in as ARENA_USERNAME on the local
Showdown server and accepts challenges from ARENA_OPPONENT. It stands in for
PokeChamp/Metamon so we can exercise the launcher + handshake + A/B orientation +
grey-box trajectory capture on real battles WITHOUT uv / LLM keys / heavy repos.

This is NOT a published baseline -- it is a test fixture for arena.external.
"""

import asyncio
import os
import sys

from poke_env.data import to_id_str
from poke_env.player.baselines import MaxBasePowerPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

USERNAME = os.environ["ARENA_USERNAME"]
OPPONENT = os.environ["ARENA_OPPONENT"]
FMT = os.environ.get("ARENA_FORMAT", "gen9randombattle")
N = int(os.environ.get("ARENA_N_BATTLES", "1000"))


async def main() -> None:
    player = MaxBasePowerPlayer(
        battle_format=FMT,
        account_configuration=AccountConfiguration(USERNAME, None),
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )
    await player.ps_client.wait_for_login()
    print(f"[stub] logged in as {USERNAME}; accepting challenges from {OPPONENT}", flush=True)
    # accept N challenges from our framework's player (one at a time)
    await player.accept_challenges(to_id_str(OPPONENT), N)
    print("[stub] done", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
