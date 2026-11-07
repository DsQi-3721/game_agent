"""Build the opponent-model knowledge the agent loads (``data/knowledge/gen9ou_opponent.json.gz``).

    python scripts/opp_build_revealed.py              # once: revealed teams -> parquet (~22 min)
    python scripts/build_opponent_kb.py [--month 2026-10] [--cutoff 1500] [--train-hi 2026-05-31]

Base distribution: Smogon chaos usage of ``--month`` at ``--cutoff`` (what teams hold now); co-occurrence counts:
every revealed team side up to ``--train-hi`` in the ``train`` split of ``arena.s1.identity`` (pinned test sets never
train), recent months weighted more (``--half-life`` months). Pairs that move a log-odds by less than ``--tau`` are
dropped and each species keeps its ``--max-feats`` most frequent features; the benchmark showed this costs nothing
measurable (``scripts/eval_opponent_model.py``, reports/opponent_modeling.md). Legality and formes follow the local
Showdown server.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arena.opponent import build as B  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026-10")
    ap.add_argument("--cutoff", type=int, default=1500)
    ap.add_argument("--train-hi", default="2026-05-31")
    ap.add_argument("--half-life", type=float, default=6.0)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--max-feats", type=int, default=150)
    ap.add_argument("--min-c", type=float, default=5.0)
    ap.add_argument("--out", default=str(ROOT / "data" / "knowledge" / "gen9ou_opponent.json.gz"))
    args = ap.parse_args()

    t0 = time.time()
    tables = B.showdown_tables()
    rv = B.Revealed(B.REVEALED, date_hi=args.train_hi, splits=("train",), half_life_months=args.half_life,
                    aliases=tables["aliases"])
    co = B.cooccurrence(rv)
    sets = B.full_sets(tables["aliases"])
    kb = B.compile_kb(chaos_sources=[(B.chaos_path(args.month, args.cutoff), 1.0)],
                      leads=B.leads_path(args.month, args.cutoff), co=co, tables=tables, sets=sets,
                      min_c=args.min_c, tau=args.tau, max_feats=args.max_feats,
                      provenance={"train": f"revealed teams <= {args.train_hi}, split train ({rv.n_teams} sides)",
                                  "half_life_months": args.half_life, "tau": args.tau, "max_feats": args.max_feats,
                                  "min_c": args.min_c})
    raw = json.dumps(kb, separators=(",", ":")).encode()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wb", compresslevel=9) as f:
        f.write(raw)
    manifest = out.parent / "manifest.json"
    m = json.loads(manifest.read_text()) if manifest.exists() else {}
    m["gen9ou_opponent"] = {"built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "file": out.name,
                            "species": len(kb["species"]), "bytes_json": len(raw), "bytes_gz": out.stat().st_size,
                            **kb["built"]}
    manifest.write_text(json.dumps(m, indent=2))
    print(f"[opponent kb] {len(kb['species'])} species, {len(raw) / 1e6:.1f} MB json, "
          f"{out.stat().st_size / 1e6:.1f} MB gz, {time.time() - t0:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
