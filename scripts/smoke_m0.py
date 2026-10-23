"""M0 smoke test: confirm we can connect to the local Showdown server and play.

Run the local server first:
    cd ../pokemon-showdown && node pokemon-showdown start --no-security

Then:
    python scripts/smoke_m0.py
"""

import asyncio

# Defensive: keep localhost traffic off any system/Clash proxy. Harmless if unset.
import os

for _v in ("ALL_PROXY", "all_proxy", "WS_PROXY", "ws_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(_v, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1," + os.environ.get("NO_PROXY", "")

from poke_env import RandomPlayer  # noqa: E402

# gen9ou needs a team (that's M2); for a pure connection smoke use a teamless random format.
FORMAT = "gen9randombattle"
N_BATTLES = 10


async def main() -> None:
    p1 = RandomPlayer(battle_format=FORMAT, max_concurrent_battles=N_BATTLES)
    p2 = RandomPlayer(battle_format=FORMAT, max_concurrent_battles=N_BATTLES)
    await p1.battle_against(p2, n_battles=N_BATTLES)
    print(f"[M0] {FORMAT}: p1 won {p1.n_won_battles}/{p1.n_finished_battles}")
    assert p1.n_finished_battles == N_BATTLES, "did not finish all battles"
    print("[M0] OK — local Showdown connection works.")


if __name__ == "__main__":
    asyncio.run(main())
