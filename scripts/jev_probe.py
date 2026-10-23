"""Feasibility probe for the TypeSafe System One API (Jev): key, models, latency, fan-out.

    python scripts/jev_probe.py                 # uses TYPESAFE_API_KEY from the project .env
    python scripts/jev_probe.py --repeats 5 --models jev-1.13.0,jev-1.14.0

1. GET /v1/models (is the key valid, which aliases exist)
2. a one-question request, repeated: round-trip latency floor
3. fan-out: the same state with 1 / 4 / 16 / 64 Nouls in one request (docs: questions run in
   parallel, so latency should stay nearly flat while input tokens grow)
4. one real battle request (the fixture position: best + best_rev + hard), repeated
Writes runs/jev_eval/probe.json. Never prints the key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

from arena.agents.system1.decider import FastDecider, RouteConfig
from arena.llm import typesafe as ts
from arena.state.views import legal_options

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "battles" / "gen9random_midbattle.pkl"
FRUITS = ["apple", "typesafe", "banana", "california", "orange", "vertex", "mango", "calibration"]


def stats(xs: list[float]) -> dict:
    return {"n": len(xs), "mean": round(statistics.mean(xs), 4), "min": round(min(xs), 4), "max": round(max(xs), 4),
            "p50": round(statistics.median(xs), 4)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="jev-1.13.0,jev-1.14.0")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="runs/jev_eval/probe.json")
    args = ap.parse_args()
    client = ts.TypeSafeClient()
    if not client.configured:
        sys.exit("TYPESAFE_API_KEY is not set: paste your key into the project .env (TYPESAFE_API_KEY=...)")
    report: dict = {"base_url": client.base_url}
    try:
        report["models_listed"] = [m.get("name") for m in await client.list_models()]
    except ts.TypeSafeError as e:
        sys.exit(f"key check failed: {e}")
    print("models:", report["models_listed"])

    battle = pickle.loads(FIXTURE.read_bytes())
    options = legal_options(battle)
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        r: dict = {}
        try:
            lat = []
            for _ in range(args.repeats):
                res = await client.system_one("I bought a banana.", {"q": ts.noul("Is a fruit mentioned?")}, model=model)
                lat.append(res.latency_s)
            r["served_by"] = res.model
            r["single_noul"] = {**stats(lat), "p_yes": res.answers["q"].noul, "input_tokens": res.input_tokens}
            fan = {}
            for n in (1, 4, 16, 64):
                qs = {f"q{i}": ts.noul(f"Is `items[{i % len(FRUITS)}]` the name of a fruit?") for i in range(n)}
                lat = []
                for _ in range(max(2, args.repeats // 2)):
                    res = await client.system_one({"items": FRUITS}, qs, model=model)
                    lat.append(res.latency_s)
                fan[n] = {**stats(lat), "input_tokens": res.input_tokens}
            r["fan_out"] = fan
            decider = FastDecider(client, RouteConfig(model=model))
            lat, tok = [], []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                fd = await decider.decide(battle, options)
                lat.append(time.perf_counter() - t0)
                tok.append(fd.input_tokens)
            r["battle_request"] = {**stats(lat), "input_tokens": tok[-1], "choice": fd.key,
                                   "confidence": round(fd.confidence, 3), "margin": round(fd.margin, 3),
                                   "consistent": fd.consistent, "p_hard": fd.p_hard, "accept": fd.accept,
                                   "top": fd.ranked(4), "cost_usd": round(tok[-1] * 0.042e-6, 7)}
        except ts.TypeSafeError as e:
            r["error"] = str(e)
        report[model] = r
        print(model, json.dumps(r, indent=1)[:1600])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    await client.aclose()
    print(f"-> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
