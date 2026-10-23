"""Download Smogon sample teams (via @pkmn/smogon teams API) into the team pool.

Writes one Showdown paste .txt per team under data/teams/<tier>/smogon/.
Overseas host (github.io) — relies on the system proxy (Clash), so do NOT strip
proxy env here.

    python scripts/ingest_teams.py
    python scripts/ingest_teams.py gen9ou gen9uu     # specific tiers
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

from arena.teams.convert import team_to_paste

ROOT = Path(__file__).resolve().parent.parent
TEAMS_DIR = ROOT / "data" / "teams"
API = "https://pkmn.github.io/smogon/data/teams/{fmt}.json"

DEFAULT_TIERS = [
    "gen9ou", "gen9ubers", "gen9uu", "gen9ru", "gen9nu", "gen9pu",
    "gen9lc", "gen9monotype", "gen9doublesou", "gen9vgc2024regh",
]


def _slug(text: str, idx: int) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return f"{idx:02d}_{s or 'team'}"[:60]


def ingest(tier: str) -> int:
    url = API.format(fmt=tier)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            if resp.status != 200:
                print(f"[{tier}] http {resp.status} — skip")
                return 0
            teams = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[{tier}] fetch failed: {type(e).__name__}: {e}")
        return 0

    if not teams:
        print(f"[{tier}] empty")
        return 0

    out_dir = TEAMS_DIR / tier / "smogon"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, team in enumerate(teams):
        try:
            paste = team_to_paste(team)
        except Exception as e:  # noqa: BLE001
            print(f"[{tier}] team {i} convert failed: {e}")
            continue
        (out_dir / f"{_slug(team.get('name', 'team'), i)}.txt").write_text(paste, encoding="utf-8")
        n += 1
    print(f"[{tier}] wrote {n} teams -> {out_dir}")
    return n


def main():
    tiers = sys.argv[1:] or DEFAULT_TIERS
    total = sum(ingest(t) for t in tiers)
    print(f"\nTotal: {total} teams across {len(tiers)} tiers")


if __name__ == "__main__":
    main()
