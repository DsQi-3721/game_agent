"""Validate every team in the pool against the local Showdown server and move
illegal ones (stale banlists etc.) into a sibling _rejected/ dir with reasons.

    python scripts/prune_teams.py            # all tiers
    python scripts/prune_teams.py gen9ou
"""

import sys
from pathlib import Path

from arena.teams.validator import validate

ROOT = Path(__file__).resolve().parent.parent
TEAMS_DIR = ROOT / "data" / "teams"


def prune_tier(tier: str) -> tuple[int, int]:
    tier_dir = TEAMS_DIR / tier
    files = [p for p in sorted(tier_dir.rglob("*.txt")) if "_rejected" not in p.parts]
    kept = rejected = 0
    reject_dir = tier_dir / "_rejected"
    for p in files:
        res = validate(p.read_text(encoding="utf-8"), tier)
        if res.ok:
            kept += 1
        else:
            reject_dir.mkdir(parents=True, exist_ok=True)
            p.rename(reject_dir / p.name)
            (reject_dir / (p.stem + ".reason.txt")).write_text(res.reason, encoding="utf-8")
            rejected += 1
            print(f"  [REJECT] {tier}/{p.name}: {res.reason.splitlines()[0] if res.reason else '?'}")
    print(f"[{tier}] kept {kept}, rejected {rejected}")
    return kept, rejected


def main():
    tiers = sys.argv[1:] or [p.name for p in sorted(TEAMS_DIR.iterdir()) if p.is_dir()]
    tk = tr = 0
    for t in tiers:
        k, r = prune_tier(t)
        tk += k
        tr += r
    print(f"\nTotal kept {tk}, rejected {tr}")


if __name__ == "__main__":
    main()
