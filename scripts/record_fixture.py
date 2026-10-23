"""Record a mid-battle ``Battle`` snapshot as a pickle fixture for offline tests.

Doubles as the M1 live smoke: it drives a real battle through ``AgentPlayer``
(proving the adapter works end-to-end), capturing the Battle at the first
decision where moves are available.

    python scripts/record_fixture.py
"""

import asyncio
import os
import pickle
from pathlib import Path

for _v in ("ALL_PROXY", "all_proxy", "WS_PROXY", "ws_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(_v, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1," + os.environ.get("NO_PROXY", "")

from poke_env.player.player import Player  # noqa: E402

from arena.agents.adapter import AgentPlayer  # noqa: E402
from arena.agents.baselines import RandomAgent  # noqa: E402
from arena.agents.base import Agent  # noqa: E402

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "battles" / "gen9random_midbattle.pkl"


def _strip_unpicklables(battle):
    """Battle holds a logging.Logger (not picklable). Drop it for the snapshot."""
    battle._logger = None
    return battle


class CapturingAgent(Agent):
    name = "capturing"

    def __init__(self, ctx=None):
        super().__init__(ctx)
        self.captured = False

    async def act(self, battle):
        if not self.captured and battle.available_moves and battle.available_switches:
            try:
                snap = pickle.loads(pickle.dumps(_strip_unpicklables(battle)))
                FIXTURE.parent.mkdir(parents=True, exist_ok=True)
                with open(FIXTURE, "wb") as f:
                    pickle.dump(snap, f)
                self.captured = True
                print(f"[record] captured Battle snapshot -> {FIXTURE}")
            except Exception as e:  # noqa: BLE001
                print(f"[record] pickle failed: {type(e).__name__}: {e}")
        return Player.choose_random_move(battle)


async def main():
    capt = AgentPlayer(CapturingAgent(), battle_format="gen9randombattle", max_concurrent_battles=1)
    opp = AgentPlayer(RandomAgent(), battle_format="gen9randombattle", max_concurrent_battles=1)
    await capt.battle_against(opp, n_battles=3)
    print(f"[M1 live smoke] AgentPlayer adapter ran: capt won {capt.n_won_battles}/{capt.n_finished_battles}")
    print(f"[record] fixture exists: {FIXTURE.exists()}")


if __name__ == "__main__":
    asyncio.run(main())
