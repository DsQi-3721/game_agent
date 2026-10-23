"""M3 live smoke: an LLM IO agent (DeepSeek by default) plays vs a baseline.

Prereq: put your key in pokemonbot/.env  ->  DEEPSEEK_API_KEY=sk-...
Then:   python scripts/smoke_m3_llm.py
        python scripts/smoke_m3_llm.py deepseek-chat 4

Uses gen9randombattle (no team / teampreview) for the fastest live validation.
"""

import asyncio
import os
import sys

for _v in ("ALL_PROXY", "all_proxy", "WS_PROXY", "ws_proxy"):
    os.environ.pop(_v, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1," + os.environ.get("NO_PROXY", "")

from arena.agents.base import AgentContext  # noqa: E402
from arena.agents.baselines import MaxBasePowerAgent  # noqa: E402
from arena.agents.llm.io import IOAgent  # noqa: E402
from arena.eval.runner import AgentSpec, MatchRunner  # noqa: E402
from arena.llm.cache import ResponseCache  # noqa: E402
from arena.llm.client import LiteLLMClient  # noqa: E402
from arena.llm.model_registry import ModelRegistry  # noqa: E402
from arena.teams.manager import AssignmentMode  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "deepseek-chat"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2


async def main():
    registry = ModelRegistry.load()
    registry.get(MODEL).require_key()  # fail fast with a clear message if missing
    client = LiteLLMClient(registry, ResponseCache())

    runner = MatchRunner()
    res = await runner.run_match(
        AgentSpec(f"llm_io[{MODEL}]", lambda ctx: IOAgent(AgentContext(llm=client), model=MODEL)),
        AgentSpec("max_base_power", lambda ctx: MaxBasePowerAgent(ctx)),
        fmt="gen9randombattle",
        mode=AssignmentMode.MIRROR,  # random format -> no team anyway
        n_battles=N,
    )
    a = res.agent_a
    print(f"\n[M3 live] {res.a} {res.a_wins}-{res.b_wins} {res.b} over {res.n_battles} (avg_turns={res.avg_turns:.1f})")
    print(f"  decisions={a.n_decisions} fallbacks={a.n_fallbacks} "
          f"tokens_in={a.tokens_in} tokens_out={a.tokens_out} cost=${a.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
