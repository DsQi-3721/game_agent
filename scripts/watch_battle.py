"""Run one battle and save a replay you can watch in a browser.

    python scripts/watch_battle.py                     # heuristic vs maxpow, gen9randombattle
    python scripts/watch_battle.py simple_heuristics random gen9ou

Two ways to see the battle:
  1) REPLAY (recommended): this saves an .html replay; open it in a browser to
     watch the full animated battle (loads the Showdown client assets, so keep
     Clash on for that browser tab).
  2) LIVE: open http://localhost:8000 in a browser, log in with any name, then
     open the printed battle URL while the battle is running (bots are fast).
"""

import asyncio
import os
import sys
from pathlib import Path

for _v in ("ALL_PROXY", "all_proxy", "WS_PROXY", "ws_proxy"):
    os.environ.pop(_v, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1," + os.environ.get("NO_PROXY", "")

from arena.agents.baselines import BASELINES  # noqa: E402
from arena.eval.runner import AgentSpec, MatchRunner  # noqa: E402
from arena.teams.manager import AssignmentMode  # noqa: E402

A = sys.argv[1] if len(sys.argv) > 1 else "simple_heuristics"
B = sys.argv[2] if len(sys.argv) > 2 else "max_base_power"
FMT = sys.argv[3] if len(sys.argv) > 3 else "gen9randombattle"
REPLAY_DIR = Path("runs/replays")


def _spec(name):
    cls = BASELINES[name]
    return AgentSpec(name, lambda ctx, cls=cls: cls(ctx))


async def main():
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    runner = MatchRunner()
    mode = AssignmentMode.MIRROR
    res = await runner.run_match(_spec(A), _spec(B), FMT, mode, n_battles=1,
                                 save_replays=str(REPLAY_DIR))
    print(f"\n[watch] {res.a} {res.a_wins}-{res.b_wins} {res.b}  ({res.avg_turns:.0f} turns)")
    replays = sorted(REPLAY_DIR.glob("*.html"), key=lambda p: p.stat().st_mtime)
    if replays:
        latest = replays[-1]
        print(f"[watch] replay saved: {latest}")
        print(f"[watch] open it:      open '{latest.resolve()}'")
    else:
        print("[watch] no replay file found (did the battle finish?)")


if __name__ == "__main__":
    asyncio.run(main())
