"""Build System-One training data from experience logs.

    python scripts/s1_dataset.py --experience "runs/**/s1exp_*.jsonl" --out data/s1/v1 [--s2-weight 0.5] \
        [--max-per-request 24] [--purposes action,memory]

Writes <out>/train.jsonl, <out>/valid.jsonl, <out>/meta.json (sources, counts, config).
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import asdict
from pathlib import Path

from arena.s1 import dataset as ds
from arena.s1.recorder import load_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experience", action="append", default=None, help="glob(s) of s1exp_*.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--teachers", default="typesafe")
    ap.add_argument("--purposes", default="")
    ap.add_argument("--s2-weight", type=float, default=0.5)
    ap.add_argument("--max-per-request", type=int, default=24)
    ap.add_argument("--valid-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    paths = sorted({p for g in (args.experience or ["runs/**/s1exp_*.jsonl"]) for p in glob.glob(g, recursive=True)})
    if not paths:
        raise SystemExit("no experience logs found")
    records = load_records(paths)
    cfg = ds.BuildConfig(teachers=tuple(args.teachers.split(",")), s2_weight=args.s2_weight,
                         purposes=tuple(args.purposes.split(",")) if args.purposes else None,
                         max_per_request=args.max_per_request or None, valid_frac=args.valid_frac, seed=args.seed)
    examples = ds.build(records, cfg)
    out = Path(args.out)
    ds.write_jsonl([e for e in examples if e["split"] == "train"], out / "train.jsonl")
    ds.write_jsonl([e for e in examples if e["split"] == "valid"], out / "valid.jsonl")
    meta = {"sources": paths, "records": len(records), "config": asdict(cfg), **ds.describe(examples)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
