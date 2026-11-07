"""Load Showdown-paste team files from the on-disk pool.

Layout: data/teams/<tier>/<source>/*.txt  (source = smogon | pokechamp | custom ...)
Each .txt is one team in Showdown paste format.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "teams"


class TeamPool:
    def __init__(self, root: Path | str = _DEFAULT_ROOT):
        self.root = Path(root)

    @lru_cache(maxsize=None)
    def _load(self, tier: str) -> tuple[str, ...]:
        tier_dir = self.root / tier
        if not tier_dir.is_dir():
            return ()
        files = [
            p for p in sorted(tier_dir.rglob("*.txt"))
            if "_rejected" not in p.parts and not p.name.endswith(".reason.txt")
        ]
        teams = [p.read_text(encoding="utf-8").strip() for p in files]
        return tuple(t for t in teams if t)

    def get(self, tier: str) -> list[str]:
        return list(self._load(tier))

    def has(self, tier: str) -> bool:
        return len(self._load(tier)) > 0

    def tiers(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and self._load(p.name))


class DirTeamPool:
    """Every *.txt paste of one directory, sorted by file name, whatever the tier (``teams.pool`` in a config)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        texts = (p.read_text(encoding="utf-8").strip() for p in sorted(self.path.glob("*.txt")))
        self._teams = tuple(t for t in texts if t)

    def get(self, tier: str) -> list[str]:
        return list(self._teams)

    def has(self, tier: str) -> bool:
        return bool(self._teams)
