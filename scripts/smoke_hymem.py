"""Preflight + one live battle + a per-space summary, for a HyMem-style experiment.

    python scripts/smoke_hymem.py [configs/experiments/hymem_smoke_randbats.yaml]

Checks (fails fast with a hint): Showdown on :8000, the model endpoint answers,
the knowledge base is built. Then runs the experiment like ``arena run`` and
prints decisions, fallbacks, folds and prompt tokens per context space.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


async def main() -> int:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "configs" / "experiments" / "hymem_smoke_randbats.yaml")
    for v in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        os.environ.pop(v, None)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    from arena.config.schema import ExperimentConfig
    from arena.eval.tournament import Evaluator
    from arena.events.emitter import EventEmitter
    from arena.llm.cache import ResponseCache
    from arena.llm.client import LiteLLMClient
    from arena.llm.model_registry import ModelRegistry

    cfg = ExperimentConfig.load(cfg_path)
    if not _port_open("127.0.0.1", 8000):
        print("[FAIL] Showdown not reachable on :8000 -> cd ../pokemon-showdown && node pokemon-showdown start --no-security")
        return 1
    print("[OK] Showdown on :8000")
    if not (ROOT / "data" / "knowledge" / "manifest.json").exists():
        print("[FAIL] knowledge base missing -> python scripts/build_knowledge.py")
        return 1
    print("[OK] knowledge base built")
    client = LiteLLMClient(ModelRegistry.load(), ResponseCache())
    for model in sorted({a.model for a in cfg.agents if a.model}):
        try:
            r = await client.complete(model, [{"role": "user", "content": "Reply with: ok"}], max_tokens=8,
                                      extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            print(f"[OK] model {model}: {r.text.strip()!r} in {r.latency_s:.1f}s")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] model {model}: {type(e).__name__}: {e} -> bash scripts/serve_qwen35_mlx.sh")
            return 1

    run_dir = ROOT / "runs" / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    report = await Evaluator(cfg, llm_client=client, run_dir=run_dir,
                             emitter=EventEmitter.to_file(run_dir / "events.jsonl", cfg.name)).run()
    print(json.dumps({k: {kk: v.get(kk) for kk in ("win_rate", "decisions", "fallback_rate", "tokens")}
                      for k, v in report["agents"].items()}, indent=2))
    for calls in sorted(run_dir.glob("calls_*.jsonl")):
        recs = [json.loads(x) for x in calls.read_text().splitlines() if x.strip()]
        spaces, toks = Counter(), defaultdict(int)
        for r in recs:
            spaces[r["space"]] += 1
            toks[r["space"]] += r.get("prompt_tokens", 0)
        print(f"\n{calls.stem}: " + ", ".join(f"{s}={n} calls/{toks[s]} prompt tok" for s, n in spaces.most_common()))
    print(f"\nanalysis: python scripts/analyze_context.py {run_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
