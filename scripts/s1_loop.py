"""One iteration of the System-1 evolution loop:

    collect (optional) -> dataset -> train -> evaluate (base / champion / candidate) -> register -> promote

    python scripts/s1_loop.py --id s1-local --base models/Qwen3.5-2B \
        [--collect configs/experiments/s1_student_ou.yaml] [--experience "runs/**/s1exp_*.jsonl"] \
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


async def _score(backend: str, model: str, adapter, valid: list[dict], limit: int) -> dict:
    from arena.s1.scorer import ScorerClient, load_scorer
    client = ScorerClient(load_scorer(backend, model, adapter))
    return await evaluate(client, valid, limit_requests=limit or None)


def score(backend: str, model: str, adapter, valid: list[dict], limit: int) -> dict:
    """Score one model on ``valid``, then release it (several models are scored in one process)."""
    import gc
    res = asyncio.run(_score(backend, model, adapter, valid, limit))
    gc.collect()
    if backend == "hf":
        import torch
        torch.cuda.empty_cache()
    return res


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
    ap.add_argument("--teachers", default="typesafe", help="backend prefixes whose answers are labels")
    ap.add_argument("--compare-base", action="store_true", help="also score the untrained base (always done "
                    "for the first model)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="hf: examples per forward pass")
    ap.add_argument("--grad-ckpt", action="store_true", help="hf: gradient checkpointing")
    ap.add_argument("--init-adapter", default=None, help="hf: continue this adapter ('champion' = the current one)")
    ap.add_argument("--self-labels", action="store_true", help="teacher-free labels from System 2 (HANDOVER 4.4)")
    ap.add_argument("--won-self-weight", type=float, default=0.0)
    ap.add_argument("--debias-orders", action="store_true")
    args = ap.parse_args()

    if args.collect:
        collect(args.collect)
    paths = sorted({p for g in (args.experience or ["runs/**/s1exp_*.jsonl"]) for p in glob.glob(g, recursive=True)})
    records = load_records(paths)
    cfg = ds.BuildConfig(teachers=tuple(t for t in args.teachers.split(",") if t), s2_weight=args.s2_weight,
                         max_per_request=args.max_per_request, self_labels=args.self_labels,
                         won_self_weight=args.won_self_weight, debias_orders=args.debias_orders)
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
    kw = {k: v for k, v in (("lr", args.lr), ("rank", args.rank)) if v is not None}
    if args.backend == "mlx":
        from arena.s1.train_mlx import train
        tmeta = train(str(data / "train.jsonl"), str(data / "valid.jsonl"), args.base, str(out), epochs=args.epochs,
                      max_train=args.max_train or None, mode=args.mode, **kw)
    else:
        from arena.s1.train_hf import train
        if args.batch_size:
            kw["batch_size"] = args.batch_size
        if args.init_adapter:
            champ0 = Registry(args.registry).champion()
            kw["init_adapter"] = champ0["adapter"] if args.init_adapter == "champion" and champ0 else args.init_adapter
        tmeta = train(str(data / "train.jsonl"), str(data / "valid.jsonl"), args.base, str(out), epochs=args.epochs,
                      max_train=args.max_train or None, grad_ckpt=args.grad_ckpt, log=lambda m: print(m, flush=True), **kw)
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    reg = Registry(args.registry)
    champ = reg.champion()
    metrics = {"valid": score(args.backend, args.base, str(out), valid_ex, args.eval_limit)}
    if champ is None or args.compare_base:
        metrics["base_valid"] = score(args.backend, args.base, None, valid_ex, args.eval_limit)
    if champ is not None:
        # the valid split grows with every round's battles: re-score the champion on THIS held-out set
        metrics["champion_valid"] = score(champ.get("backend", args.backend), champ["base_model"], champ["adapter"],
                                          valid_ex, args.eval_limit)
    entry = reg.add({"id": args.id, "base_model": args.base, "adapter": str(out), "backend": args.backend,
                     "parent": champ["id"] if champ else None, "dataset": str(data), "n_train": tmeta.get("n_train"),
                     "train": {k: v for k, v in tmeta.items() if k not in ("history", "loss_curve")},
                     "metrics": metrics})
    ok, why = decide_promotion(entry, champ, champion_valid=metrics.get("champion_valid"))
    if ok:
        reg.promote(args.id, why)
    print(json.dumps({"id": args.id, "promoted": ok, "reason": why,
                      "valid": metrics["valid"].get("overall"), "base": metrics.get("base_valid", {}).get("overall"),
                      "champion_before": champ["id"] if champ else None}, indent=1), flush=True)
    print(f"[loop] serve: python scripts/s1_serve.py --model {args.base} --adapter {out} --backend {args.backend}")


if __name__ == "__main__":
    main()
