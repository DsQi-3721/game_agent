"""Collect real battle positions (pickled ``Battle`` objects) for offline System-1 evaluation.

Baseline agents play each other on the local Showdown server; a recording wrapper pickles the
battle at every decision with 2+ legal options, so the evaluation set covers ordinary turns,
forced switches and endgames from both sides. Needs only the Showdown server (no LLM, no Jev).

    python scripts/collect_decision_snapshots.py --out data/jev_eval/snapshots --battles 6
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pickle
from pathlib import Path

for _v in ("ALL_PROXY", "all_proxy", "WS_PROXY", "ws_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(_v, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1," + os.environ.get("NO_PROXY", "")

from arena.agents.adapter import AgentPlayer  # noqa: E402
from arena.agents.base import Agent  # noqa: E402
from arena.agents.baselines import MaxBasePowerAgent, RandomAgent, SimpleHeuristicsAgent  # noqa: E402
from arena.state.views import legal_options  # noqa: E402


class RecordingAgent(Agent):
    def __init__(self, inner: Agent, out: Path, tag: str):
        super().__init__()
        self.inner, self.out, self.tag = inner, out, tag
        self.name = f"rec_{inner.name}"
        self.n = 0

    async def act(self, battle):
        if len(legal_options(battle)) >= 2:
            logger, battle.logger = battle.logger, None
            try:
                data = pickle.dumps(battle)
            finally:
                battle.logger = logger
            self.n += 1
            name = f"{self.tag}_{battle.battle_tag.split('-')[-1]}_{battle.player_role}_t{battle.turn:02d}_{self.n:04d}.pkl"
            (self.out / name).write_bytes(data)
        return await self.inner.act(battle)


async def play(a: Agent, b: Agent, n: int, fmt: str) -> tuple[int, int]:
    pa = AgentPlayer(a, battle_format=fmt, max_concurrent_battles=1)
    pb = AgentPlayer(b, battle_format=fmt, max_concurrent_battles=1)
    await pa.battle_against(pb, n_battles=n)
    return pa.n_won_battles, pa.n_finished_battles


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/jev_eval/snapshots")
    ap.add_argument("--battles", type=int, default=6, help="heuristics vs max-power battles (both sides recorded)")
    ap.add_argument("--vs-random", type=int, default=2, help="extra heuristics vs random battles")
    ap.add_argument("--format", default="gen9randombattle")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    heur = RecordingAgent(SimpleHeuristicsAgent(), out, "hv")
    maxbp = RecordingAgent(MaxBasePowerAgent(), out, "hv")
    w, n = await play(heur, maxbp, args.battles, args.format)
    print(f"heuristics vs max_base_power: {w}/{n} won; recorded {heur.n + maxbp.n} positions")
    if args.vs_random:
        heur2 = RecordingAgent(SimpleHeuristicsAgent(), out, "hr")
        w, n = await play(heur2, RandomAgent(), args.vs_random, args.format)
        print(f"heuristics vs random: {w}/{n} won; recorded {heur2.n} positions")
    print(f"snapshots -> {out} ({len(list(out.glob('*.pkl')))} files)")


if __name__ == "__main__":
    asyncio.run(main())
