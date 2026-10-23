"""End-to-end acceptance test: does the framework behave as expected?

Layered checks with PASS/FAIL assertions:
  L1 offline unit/integration tests (pytest)        — no server, no keys
  L2 server connectivity                            — needs local Showdown
  L3 baseline tournament invariants                 — needs server
  L4 LLM smoke (optional)                            — needs server + DEEPSEEK_API_KEY

    python scripts/acceptance.py            # L1-L3 (+ L4 if key present)
    python scripts/acceptance.py --no-llm   # skip L4

Exits non-zero if any required layer fails.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

for _v in ("ALL_PROXY", "all_proxy", "WS_PROXY", "ws_proxy"):
    os.environ.pop(_v, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1," + os.environ.get("NO_PROXY", "")

PY = sys.executable
ROOT = Path(__file__).resolve().parent.parent
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def server_up() -> bool:
    import socket
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", 8000))
        return True
    except OSError:
        return False
    finally:
        s.close()


def l1_offline() -> None:
    print("\nL1: offline tests (pytest)")
    r = subprocess.run([PY, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    check("pytest offline suite", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else "")


async def l3_baseline_tournament() -> None:
    print("\nL3: baseline tournament invariants (gen9randombattle)")
    from arena.config.schema import ExperimentConfig
    from arena.eval.tournament import Evaluator
    cfg = ExperimentConfig.load(ROOT / "configs/experiments/quick_baselines.yaml")
    cfg.evaluation.n_battles = 6
    ev = Evaluator(cfg, run_dir=ROOT / "runs/_acceptance")
    rep = await ev.run()
    ag = rep["agents"]
    check("report has all agents", set(ag) == {"random", "maxpow", "heuristic"})
    check("matrix populated", len(rep["matrix"]) == 3)
    check("baselines never fall back", all(ag[a]["fallback_rate"] == 0.0 for a in ag))
    check("heuristic beats random (winrate)", ag["heuristic"]["win_rate"] > ag["random"]["win_rate"],
          f"heuristic {ag['heuristic']['win_rate']:.0%} vs random {ag['random']['win_rate']:.0%}")
    check("Elo ordering heuristic>random", ag["heuristic"]["elo"] > ag["random"]["elo"])
    check("Wilson CI present", all(len(ag[a]["wilson_ci"]) == 2 for a in ag))
    check("report.json written", (ROOT / "runs/_acceptance/report.json").exists())


async def l4_llm_smoke() -> None:
    print("\nL4: LLM smoke (DeepSeek, 1 battle)")
    from arena.agents.base import AgentContext
    from arena.agents.baselines import MaxBasePowerAgent
    from arena.agents.llm.io import IOAgent
    from arena.eval.runner import AgentSpec, MatchRunner
    from arena.llm.cache import ResponseCache
    from arena.llm.client import LiteLLMClient
    from arena.llm.model_registry import ModelRegistry
    from arena.teams.manager import AssignmentMode

    registry = ModelRegistry.load()
    client = LiteLLMClient(registry, ResponseCache())
    runner = MatchRunner()
    res = await runner.run_match(
        AgentSpec("io", lambda ctx: IOAgent(AgentContext(llm=client), model="deepseek-chat")),
        AgentSpec("maxpow", lambda ctx: MaxBasePowerAgent(ctx)),
        "gen9randombattle", AssignmentMode.MIRROR, n_battles=1)
    a = res.agent_a
    check("LLM made decisions", a.n_decisions > 0, f"{a.n_decisions} decisions")
    check("LLM parse mostly succeeds", a.n_fallbacks < a.n_decisions,
          f"{a.n_fallbacks}/{a.n_decisions} fallbacks")
    check("tokens tracked", a.tokens_in > 0 and a.tokens_out > 0)
    check("cost computed", a.cost_usd > 0, f"${a.cost_usd:.4f}")


async def main() -> None:
    no_llm = "--no-llm" in sys.argv
    l1_offline()

    if not server_up():
        check("server reachable (localhost:8000)", False,
              "start: cd ../pokemon-showdown && node pokemon-showdown start --no-security")
    else:
        check("server reachable (localhost:8000)", True)
        await l3_baseline_tournament()
        if not no_llm and (os.environ.get("DEEPSEEK_API_KEY") or (ROOT / ".env").exists()):
            try:
                await l4_llm_smoke()
            except Exception as e:  # noqa: BLE001
                check("L4 LLM smoke ran", False, f"{type(e).__name__}: {e}")
        else:
            print("\nL4: skipped (no DEEPSEEK_API_KEY / .env)")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n=== {len(results) - n_fail}/{len(results)} checks passed ===")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    asyncio.run(main())
