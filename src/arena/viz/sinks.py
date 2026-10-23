"""Metrics sinks. ``publish(report, results)`` is called once after a run; it may
return a path (e.g. the charts HTML) surfaced in the RUN_FINISHED event.

  · PlotlySink  — always-on local interactive HTML (tier ii)
  · WandbSink   — optional W&B (tier iii); a no-op if wandb is absent/disabled
  · MultiSink   — fan out to several sinks
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Protocol


class MetricsSink(Protocol):
    def publish(self, report: dict, results: list) -> Optional[Path]: ...


class PlotlySink:
    def __init__(self, out_path: Path | str):
        self.out = Path(out_path)

    def publish(self, report, results):
        try:
            from arena.viz.charts_html import build
            return build(report, self.out)
        except Exception:  # noqa: BLE001  (viz must never break a finished run)
            return None


class WandbSink:
    """One W&B run per agent, grouped by experiment. No-op when unavailable."""

    def __init__(self, group: str, project: str = "pokebotmon"):
        self.group = group
        self.project = project

    def publish(self, report, results):
        if os.environ.get("WANDB_MODE") == "disabled":
            return None
        try:
            import wandb
        except Exception:  # noqa: BLE001
            return None
        try:
            for name, m in report.get("agents", {}).items():
                run = wandb.init(project=self.project, group=self.group, name=name,
                                 job_type="agent", reinit=True,
                                 config={"experiment": report.get("experiment"),
                                         "format": report.get("format")})
                metrics = {k: m[k] for k in
                           ("elo", "win_rate", "wins", "games", "fallback_rate",
                            "tokens", "cost_usd", "tool_calls") if k in m}
                run.summary.update(metrics)
                run.log(metrics)
                run.finish()
        except Exception:  # noqa: BLE001
            return None
        return None


class MultiSink:
    def __init__(self, sinks: list):
        self.sinks = sinks

    def publish(self, report, results):
        charts = None
        for s in self.sinks:
            r = s.publish(report, results)
            if r:
                charts = r
        return charts
