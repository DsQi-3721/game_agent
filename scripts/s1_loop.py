"""One iteration of the System-1 evolution loop:

    collect (optional) -> dataset -> train -> evaluate (base / champion / candidate) -> register -> promote

    python scripts/s1_loop.py --id s1-v1 --base /Users/hoi/project/models/Qwen3.5-0.8B \
        [--collect configs/experiments/s1_collect_randbats.yaml] [--experience "runs/**/s1exp_*.jsonl"] \
        [--backend mlx|hf] [--mode auto] [--epochs 1] [--max-train 1500] [--eval-limit 80]

Collect with the teacher (Jev) at first; once a champion exists, collect with the cascade
(``student_url`` -> teacher) so the teacher labels exactly the turns the student is unsure about,
then train the next student on everything gathered so far.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from arena.s1 import dataset as ds
from arena.s1.evaluate import evaluate
from arena.s1.recorder import load_records
from arena.s1.registry import Registry, decide_promotion


def collect(configs: list[str]) -> None:
    for cfg in configs:
        print(f"[loop] collect: {cfg}", flush=True)
        subprocess.run([sys.executable, "-m", "arena.cli", "run", cfg, "--emit-events"], check=True)


async def score(backend: str, model: str, adapter, valid: list[dict], limit: int) -> dict:
    from arena.s1.scorer import ScorerClient, load_scorer
    client = ScorerClient(load_scorer(backend, model, adapter))
    return await evaluate(client, valid, limit_requests=limit or None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="model id of this iteration, e.g. s1-v1")
    ap.add_argument("--base", required=True, help="base model path (the student's backbone)")
    ap.add_argument("--collect", action="append", default=[], help="arena experiment config(s) to run first")
    ap.add_argument("--experience", action="append", default=None)
    ap.add_argument("--backend", default="mlx", choices=["mlx", "hf"])
    ap.add_argument("--mode", default="auto", choices=["auto", "head", "full"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-train", type=int, default=1500)
    ap.add_argument("--max-per-request", type=int, default=24)
    ap.add_argument("--s2-weight", type=float, default=0.5)
    ap.add_argument("--eval-limit", type=int, default=80, help="max held-out requests to score")
    ap.add_argument("--registry", default="runs/s1_models/registry.json")
    args = ap.parse_args()

    if args.collect:
        collect(args.collect)
    paths = sorted({p for g in (args.experience or ["runs/**/s1exp_*.jsonl"]) for p in glob.glob(g, recursive=True)})
    records = load_records(paths)
    cfg = ds.BuildConfig(s2_weight=args.s2_weight, max_per_request=args.max_per_request)
    examples = ds.build(records, cfg)
    data = Path("data/s1") / args.id
    train_ex = [e for e in examples if e["split"] == "train"]
    valid_ex = [e for e in examples if e["split"] == "valid"]
    ds.write_jsonl(train_ex, data / "train.jsonl")
    ds.write_jsonl(valid_ex, data / "valid.jsonl")
    meta = {"sources": paths, "records": len(records), "config": asdict(cfg), **ds.describe(examples)}
    (data / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[loop] dataset {data}: {meta['by_split']} from {len(paths)} logs", flush=True)

    out = Path("runs/s1_models") / args.id
    if args.backend == "mlx":
        from arena.s1.train_mlx import train
        tmeta = train(str(data / "train.jsonl"), str(data / "valid.jsonl"), args.base, str(out), epochs=args.epochs,
                      max_train=args.max_train, mode=args.mode)
    else:
        from arena.s1.train_hf import train
        tmeta = train(str(data / "train.jsonl"), str(data / "valid.jsonl"), args.base, str(out), epochs=args.epochs,
                      max_train=args.max_train)

    reg = Registry(args.registry)
    champ = reg.champion()
    metrics = {"valid": asyncio.run(score(args.backend, args.base, str(out), valid_ex, args.eval_limit))}
    if champ is None:
        metrics["base_valid"] = asyncio.run(score(args.backend, args.base, None, valid_ex, args.eval_limit))
    entry = reg.add({"id": args.id, "base_model": args.base, "adapter": str(out), "backend": args.backend,
                     "parent": champ["id"] if champ else None, "dataset": str(data), "n_train": tmeta.get("n_train"),
                     "train": {k: v for k, v in tmeta.items() if k != "history"}, "metrics": metrics})
    ok, why = decide_promotion(entry, champ)
    if ok:
        reg.promote(args.id, why)
    print(json.dumps({"id": args.id, "promoted": ok, "reason": why,
                      "valid": metrics["valid"].get("overall"), "base": metrics.get("base_valid", {}).get("overall"),
                      "champion_before": champ["id"] if champ else None}, indent=1), flush=True)
    print(f"[loop] serve: python scripts/s1_serve.py --model {args.base} --adapter {out} --backend {args.backend}")


if __name__ == "__main__":
    main()
