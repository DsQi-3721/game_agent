"""The compiled opponent knowledge (one JSON per format) and species-name resolution.

Layout of ``gen9ou_opponent.json`` (see ``arena.opponent.build``):

    {"format", "built": {provenance}, "aliases": {forme id: canonical id}, "reveal_rates": {kind: {value: r}},
     "species": {canonical id: {
         "name", "usage", "n",                         usage share and teams behind the base distribution
         "items" | "abilities" | "tera": {id: p}       base distribution (Smogon usage: what teams really hold)
         "moves": {id: p}                              P(move in the set), sums to ~4
         "spreads": [[nature, [6 EVs], p], ...]        unconditional, most common first
         "spreads_by_item": {item: [[nature, evs, p], ...]}   from full sets (Foul Play / Smogon dex)
         "mates": {id: p}                              P(teammate | species)
         "lead": p                                     P(leads | brought)
         "counters": [[species id, n, p_ko_or_switch], ...]
         "nb": {kind: {"N", "n": {x: count}, "pf": {feature: p}, "c": {x: {feature: count}}}}
                                                       co-occurrence counts from revealed teams (see posterior)
     }}}

Feature keys inside ``nb``: ``m:<teammate>``, ``v:<revealed move>``, ``i:<item>``, ``a:<ability>``, ``t:<tera>``.
"""

from __future__ import annotations

import functools
import gzip
import json
from pathlib import Path
from typing import Optional

from poke_env.data import to_id_str

from arena.knowledge import store

KB_FILE = "gen9ou_opponent.json"


def tid(s) -> str:
    return to_id_str(str(s or ""))


@functools.lru_cache(maxsize=8)
def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        gz = p.with_suffix(p.suffix + ".gz")
        if not gz.exists():
            return {}
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def kb_for(battle_format: str = "gen9ou", path: Optional[str] = None) -> dict:
    """The knowledge for a format ({} when missing: the model then falls back to the plain prior)."""
    if path:
        return _load(str(path))
    fmt = (battle_format or "").lower()
    if not fmt.startswith("gen9") or "random" in fmt:
        return {}
    return _load(str(store.knowledge_dir() / KB_FILE))


def clear_cache() -> None:
    _load.cache_clear()


def resolve_species(kb: dict, species: str) -> Optional[str]:
    """Canonical KB id for a species / forme shown in the battle: battle-only and cosmetic formes map to the base
    (Ogerpon-Wellspring-Tera, Mimikyu-Busted, Maushold-Four ...); a forme masked at team preview ("Urshifu-*")
    maps to its most used legal forme."""
    sp = kb.get("species") or {}
    s = tid(species)
    if s in sp:
        return s
    a = (kb.get("aliases") or {}).get(s)
    if a and a in sp:
        return a
    cands = [k for k in sp if k.startswith(s)] if len(s) >= 4 else []
    if cands:
        return max(cands, key=lambda k: sp[k].get("usage", 0.0))
    return None
