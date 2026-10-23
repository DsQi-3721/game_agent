"""Scan past runs and load decision trajectories (UI-agnostic, testable)."""

from __future__ import annotations

import json
from pathlib import Path


def list_runs(runs_dir: Path | str = "runs") -> list[dict]:
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir(), reverse=True):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        report = d / "report.json"
        trajs = sorted(d.glob("traj_*.jsonl"))
        if not report.exists() and not trajs:
            continue
        info = {
            "name": d.name, "path": str(d),
            "agents": [p.stem[len("traj_"):] for p in trajs],
            "format": "-", "top": "-", "n_agents": 0,
        }
        if report.exists():
            try:
                r = json.loads(report.read_text(encoding="utf-8"))
                ag = r.get("agents", {})
                info["format"] = r.get("format", "-")
                info["n_agents"] = len(ag)
                info["top"] = max(ag, key=lambda n: ag[n].get("elo", 0)) if ag else "-"
            except Exception:  # noqa: BLE001
                pass
        out.append(info)
    return out


def load_turns(traj_path: Path | str, query: str = "") -> list[dict]:
    p = Path(traj_path)
    if not p.exists():
        return []
    q = query.lower().strip()
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if q:
            hay = (str(rec.get("output", "")) + str(rec.get("state", "")) +
                   str(rec.get("parsed_order", "")) + str(rec.get("note", ""))).lower()
            if q not in hay:
                continue
        rows.append(rec)
    return rows
