"""M5 live smoke: DeepSeek ReAct agent (native tool-calling) with pokedex tool +
passive grounding, vs MaxBasePower. Validates the native tool path end-to-end.

    python scripts/smoke_m5_react.py [model] [n_battles]
"""

import asyncio
import os
import sys

for _v in ("ALL_PROXY", "all_proxy", "WS_PROXY", "ws_proxy"):
    os.environ.pop(_v, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1," + os.environ.get("NO_PROXY", "")

from arena.agents.base import AgentContext  # noqa: E402
from arena.agents.baselines import MaxBasePowerAgent  # noqa: E402
from arena.agents.llm.react import ReActAgent  # noqa: E402
from arena.augment.base import AugmentationRegistry  # noqa: E402
from arena.eval.runner import AgentSpec, MatchRunner  # noqa: E402
from arena.llm.cache import ResponseCache  # noqa: E402
from arena.llm.client import LiteLLMClient  # noqa: E402
from arena.llm.model_registry import ModelRegistry  # noqa: E402
from arena.teams.manager import AssignmentMode  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "deepseek-chat"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1


async def main():
    registry = ModelRegistry.load()
    registry.get(MODEL).require_key()
    client = LiteLLMClient(registry, ResponseCache())
    aug = AugmentationRegistry.from_config({
        "damage_calc": "passive", "type_matchup": "passive",
        "base_stats": "passive", "pokedex": "tool",
    })

    runner = MatchRunner()
    res = await runner.run_match(
        AgentSpec(f"react[{MODEL}]",
                  lambda ctx: ReActAgent(AgentContext(llm=client, augmentations=aug), model=MODEL, max_steps=3)),
        AgentSpec("max_base_power", lambda ctx: MaxBasePowerAgent(ctx)),
        fmt="gen9randombattle", mode=AssignmentMode.MIRROR, n_battles=N,
    )
    a = res.agent_a
    print(f"\n[M5 live] {res.a} {res.a_wins}-{res.b_wins} {res.b} (avg_turns={res.avg_turns:.1f})")
    print(f"  decisions={a.n_decisions} tool_calls={a.n_tool_calls} fallbacks={a.n_fallbacks} "
          f"tokens={a.tokens_in}+{a.tokens_out} cost=${a.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
