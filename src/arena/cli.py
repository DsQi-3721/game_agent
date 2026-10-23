"""`arena run <experiment.yaml>` — drive a full evaluation from one config file."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path


def _bypass_local_proxy() -> None:
    for v in ("ALL_PROXY", "all_proxy", "WS_PROXY", "ws_proxy"):
        os.environ.pop(v, None)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1," + os.environ.get("NO_PROXY", "")
    # litellm otherwise fetches its price table from GitHub at import (slow / fails offline)
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _print_summary(report: dict) -> None:
    if report.get("dry_run"):
        print(f"[dry-run] {report['experiment']}: {report['estimate']}")
        return
    print(f"\n=== {report['experiment']} | {report['format']} | "
          f"{report['mode']} n={report['n_battles']} ===")
    rows = sorted(report["agents"].items(), key=lambda kv: kv[1]["elo"], reverse=True)
    print(f"{'agent':22} {'elo':>6} {'winrate':>8} {'95% CI':>16} {'fallback':>9} {'cost$':>8}")
    for name, m in rows:
        ci = f"[{m['wilson_ci'][0]:.2f},{m['wilson_ci'][1]:.2f}]"
        print(f"{name:22} {m['elo']:>6.0f} {m['win_rate']:>8.2%} {ci:>16} "
              f"{m['fallback_rate']:>8.1%} {m['cost_usd']:>8.4f}")
    print(f"\ntotal cost ${report['total_cost_usd']:.4f}  tokens {report['total_tokens']}")
    if report.get("aborted"):
        print(f"ABORTED: {report['aborted']}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="arena")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tui", help="launch the interactive terminal UI")
    run = sub.add_parser("run", help="run an experiment config")
    run.add_argument("config")
    run.add_argument("--run-dir", default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--watch", action="store_true",
                     help="print each battle's spectate URL and save replays")
    run.add_argument("--turn-delay", type=float, default=0.0,
                     help="seconds to pause per turn (slows battles so they're watchable live)")
    run.add_argument("--emit-events", action="store_true",
                     help="stream structured progress events to runs/<name>/events.jsonl (for the TUI)")
    run.add_argument("--wandb", action="store_true",
                     help="also log per-agent runs to Weights & Biases (grouped by experiment)")
    args = ap.parse_args()

    if args.cmd == "tui":
        from arena.tui.app import main as tui_main
        tui_main()
        return

    _bypass_local_proxy()
    from arena.agents.registry import is_llm_type
    from arena.config.schema import ExperimentConfig
    from arena.eval.tournament import Evaluator

    cfg = ExperimentConfig.load(args.config)
    if args.dry_run:
        cfg.guardrails.dry_run = True

    client = None
    if any(is_llm_type(a.type) for a in cfg.agents):
        from arena.llm.cache import ResponseCache
        from arena.llm.client import LiteLLMClient
        from arena.llm.model_registry import ModelRegistry
        client = LiteLLMClient(ModelRegistry.load(), ResponseCache())

    run_dir = Path(args.run_dir or f"runs/{cfg.name}")
    if args.watch:
        print(f"[watch] open {('http://localhost:8000')} in a browser and log in with any "
              f"name; click each battle URL below to spectate live.\n")
    emitter = None
    if args.emit_events:
        from arena.events.emitter import EventEmitter
        run_dir.mkdir(parents=True, exist_ok=True)
        emitter = EventEmitter.to_file(run_dir / "events.jsonl", cfg.name)

    metrics_sink = None
    if not args.dry_run:
        from arena.viz.sinks import MultiSink, PlotlySink, WandbSink
        sinks = [PlotlySink(run_dir / "charts.html")]
        if args.wandb:
            sinks.append(WandbSink(group=cfg.name))
        metrics_sink = MultiSink(sinks)

    ev = Evaluator(cfg, llm_client=client, run_dir=run_dir, watch=args.watch,
                   turn_delay=args.turn_delay, emitter=emitter, metrics_sink=metrics_sink)
    report = asyncio.run(ev.run())
    _print_summary(report)
    print(f"\nreport + trajectories -> {run_dir}")
    if args.watch or cfg.evaluation.save_replays:
        print(f"replays -> {run_dir/'replays'} (open the .html in a browser)")


if __name__ == "__main__":
    main()
