"""Measure a search allocation on saved development configurations."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
from pathlib import Path
import time

from evidence import aggregate, to_key
from poke_engine import State, monte_carlo_tree_search


def compute(state, milliseconds, threads):
    wall, cpu = time.perf_counter(), time.process_time()
    result = monte_carlo_tree_search(State.from_string(state), milliseconds, threads=threads)
    return {"visits": result.total_visits, "seconds": time.perf_counter() - wall,
            "cpu_seconds": time.process_time() - cpu,
            "ours": {to_key(o.move_choice): {"visits": o.visits, "total_score": o.total_score}
                     for o in result.side_one}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--search-ms", type=int, default=700)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.engine = root / args.engine
    args.out = root / args.out
    logging.disable(logging.CRITICAL)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, args.out.open("w") as output:
        for line in args.engine.read_text().splitlines():
            record = json.loads(line)
            started = time.perf_counter()
            jobs = [pool.submit(compute, sample["state"], args.search_ms, args.threads)
                    for sample in record["samples"]]
            results = [job.result() for job in jobs]
            row = {"id": record["id"], "policy": aggregate(record["samples"], results),
                   "samples": len(results), "search_ms": args.search_ms, "threads": args.threads,
                   "wall_s": time.perf_counter() - started,
                   "worker_wall_s": sum(r["seconds"] for r in results),
                   "worker_cpu_s": sum(r["cpu_seconds"] for r in results),
                   "iterations": sum(r["visits"] for r in results)}
            output.write(json.dumps(row) + "\n")
            output.flush()
            print(record["id"], round(row["wall_s"], 3), round(row["worker_cpu_s"], 3), flush=True)


if __name__ == "__main__":
    main()
