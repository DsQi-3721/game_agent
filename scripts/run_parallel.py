"""Run an anchored experiment as parallel processes: one per (agent, shard), then merge the results.

    python scripts/run_parallel.py configs/experiments/hymem_main_gpu_ou.yaml \
        --out runs/hymem_main_gpu_ou [--shards 2] [--max-parallel 8]

The tournament plays its pairings one after another, and Showdown serialises the battles of one
pairing, so 4 HyMem agents x 30 battles would run back to back for many hours. Here every agent
(and every shard of its battles) is its own ``arena.cli run`` process with its own Showdown
accounts; the LLM server (vLLM) batches their concurrent requests. Each job's config is written
next to its run directory (``<out>/<agent>/s<k>/config.yaml``); the cost cap is split evenly. With
``teams.per_team`` (mirror teams in blocks) every shard gets the offset of its first battle, so the shards
together play the same team sequence as one run would.
``<out>/merged.json``: wins per agent and anchor over all shards, Wilson 95% intervals.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import yaml


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def plan(cfg: dict, agents: list[str], shards: int) -> list[tuple[str, int, dict]]:
    ev = cfg["evaluation"]
    if ev.get("mode") != "anchored":
        raise SystemExit("only anchored experiments can be split per agent")
    anchors = ev["anchors"]
    by_name = {a["name"]: a for a in cfg["agents"]}
    n = int(ev["n_battles"])
    jobs = []
    budget = (cfg.get("guardrails") or {}).get("max_cost_usd")
    n_jobs = len(agents) * shards
    per_team = int((cfg.get("teams") or {}).get("per_team") or 0)
    for name in agents:
        start = 0
        for k in range(shards):
            nb = n // shards + (1 if k < n % shards else 0)
            start += nb
            if nb == 0:
                continue
            c = copy.deepcopy(cfg)
            c["name"] = f"{cfg['name']}__{name}__s{k}"
            c["seed"] = int(cfg.get("seed", 0)) * 1000 + k
            c["agents"] = [copy.deepcopy(by_name[name])] + [copy.deepcopy(by_name[a]) for a in anchors]
            c["evaluation"] = {**ev, "n_battles": nb}
            if per_team > 0:
                c["teams"] = {**c["teams"], "offset": int(c["teams"].get("offset") or 0) + start - nb}
            if budget is not None:
                c.setdefault("guardrails", {})["max_cost_usd"] = round(budget / n_jobs, 4)
            jobs.append((name, k, c))
    return jobs


def merge(out: Path, cfg: dict, agents: list[str]) -> dict:
    anchors = cfg["evaluation"]["anchors"]
    merged = {}
    for name in agents:
        row = {a: {"wins": 0, "n": 0} for a in anchors}
        reports = sorted(out.glob(f"{name}/s*/report.json"))
        cost, aborted = 0.0, []
        for rp in reports:
            r = json.loads(rp.read_text())
            for a, v in ((r.get("matrix") or {}).get(name) or {}).items():
                if a in row:
                    row[a]["wins"] += v["wins"]
                    row[a]["n"] += v["n"]
            cost += float(((r.get("agents") or {}).get(name) or {}).get("cost_usd") or 0.0)
            if r.get("aborted"):
                aborted.append(f"{rp.parent.name}: {r['aborted']}")
        w, n = sum(v["wins"] for v in row.values()), sum(v["n"] for v in row.values())
        merged[name] = {"vs": {a: {**v, "win_rate": round(v["wins"] / v["n"], 4) if v["n"] else None,
                                   "ci95": wilson(v["wins"], v["n"])} for a, v in row.items()},
                        "wins": w, "n": n, "win_rate": round(w / n, 4) if n else None, "ci95": wilson(w, n),
                        "shards": len(reports), "cost_usd": round(cost, 4), "aborted": aborted}
    (out / "merged.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--agents", default="", help="comma-separated subset (default: every non-anchor agent)")
    ap.add_argument("--shards", type=int, default=1, help="processes per agent (battles split between them)")
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    anchors = cfg["evaluation"]["anchors"]
    agents = [a for a in args.agents.split(",") if a] or [a["name"] for a in cfg["agents"] if a["name"] not in anchors]
    out = Path(args.out)
    if not args.merge_only:
        pending = plan(cfg, agents, args.shards)
        running: list = []
        while pending or running:
            while pending and len(running) < args.max_parallel:
                name, k, c = pending.pop(0)
                d = out / name / f"s{k}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
                log = open(d / "run.log", "w")
                p = subprocess.Popen([sys.executable, "-m", "arena.cli", "run", str(d / "config.yaml"),
                                      "--run-dir", str(d), "--emit-events"], stdout=log, stderr=subprocess.STDOUT)
                running.append((name, k, p, log, time.time()))
                print(f"[parallel] started {name} s{k} ({c['evaluation']['n_battles']} battles) pid {p.pid}",
                      flush=True)
            time.sleep(5)
            for item in list(running):
                name, k, p, log, t0 = item
                if p.poll() is not None:
                    log.close()
                    running.remove(item)
                    print(f"[parallel] finished {name} s{k} rc={p.returncode} in {time.time() - t0:.0f}s", flush=True)
    merged = merge(out, cfg, agents)
    for name, m in merged.items():
        print(f"{name:16s} {m['wins']:3d}/{m['n']:<3d} win {m['win_rate']}  95% CI {m['ci95']}  ${m['cost_usd']}"
              + (f"  ABORTED {m['aborted']}" if m["aborted"] else ""))


if __name__ == "__main__":
    main()
