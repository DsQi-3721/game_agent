"""Lazy, cached access to the knowledge JSON files.

``ARENA_KNOWLEDGE_DIR`` overrides the location (tests point it at a tiny fixture).
A missing file yields ``{}`` so tools degrade to "unknown" instead of crashing.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Optional

_DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "knowledge"

DEX_FILE = "dex_gen9.json"
RANDBATS_FILE = "randbats_gen9.json"
OU_FILE = "gen9ou_sets.json"


def knowledge_dir() -> Path:
    return Path(os.environ.get("ARENA_KNOWLEDGE_DIR") or _DEFAULT_DIR)


@functools.lru_cache(maxsize=None)
def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load(name: str) -> dict:
    return _load(str(knowledge_dir() / name))


def clear_cache() -> None:
    _load.cache_clear()


def dex() -> dict:
    """{moves, abilities, items, species}: id -> entry (with Showdown ``desc``)."""
    return load(DEX_FILE)


def sets_file_for(battle_format: str) -> Optional[str]:
    fmt = (battle_format or "").lower()
    if not fmt.startswith("gen9"):
        return None
    if "randombattle" in fmt:
        return RANDBATS_FILE
    # usage stats exist for OU; other gen9 tiers fall back to them (closest prior)
    return OU_FILE


def species_sets(battle_format: str) -> dict:
    name = sets_file_for(battle_format)
    return (load(name).get("species") or {}) if name else {}
