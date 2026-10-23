"""Partial-observability helpers.

Single source of truth for "what does the acting player actually know" so the
renderer and (M4) augmentations agree and never leak hidden info in the default
view. ``full_info=True`` callers may look past these.
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.pokemon import Pokemon


def identifier_in(battle, mon: Pokemon) -> Optional[str]:
    """The roster key (e.g. 'p1: Passimian') under which ``mon`` is stored."""
    for pool in (battle.team, battle.opponent_team):
        for ident, p in pool.items():
            if p is mon:
                return ident
    return None


def request_entry(battle, mon: Pokemon) -> Optional[dict]:
    """Our own mon's entry in the last ``|request|`` (None for opponents)."""
    req = getattr(battle, "_last_request", None) or {}
    ident = identifier_in(battle, mon)
    if not ident or not isinstance(req, dict):
        return None
    for entry in (req.get("side") or {}).get("pokemon", []) or []:
        if entry.get("ident") == ident:
            return entry
    return None


def tera_type_of(battle, mon: Pokemon) -> Optional[str]:
    """Tera type name, falling back to the request for our own mons (poke-env does
    not copy ``teraType`` from the request in random battles)."""
    tt = getattr(mon, "tera_type", None)
    if tt is not None:
        return getattr(tt, "name", str(tt)).title()
    entry = request_entry(battle, mon)
    if entry and entry.get("teraType"):
        return str(entry["teraType"]).title()
    return None


def hp_pct(mon: Optional[Pokemon]) -> int:
    """HP as an integer percentage (0 for fainted/unknown)."""
    if mon is None or mon.current_hp_fraction is None:
        return 0
    return int(round(mon.current_hp_fraction * 100))


def revealed_moves(mon: Pokemon) -> list[str]:
    """Moves that have been observed for this Pokemon."""
    return [m.id for m in mon.moves.values()]


def known_item(mon: Pokemon) -> str | None:
    item = mon.item
    if item in (None, "", "unknown_item"):
        return None
    return item


def known_ability(mon: Pokemon) -> str | None:
    return mon.ability or None


def mon_label(mon: Pokemon) -> str:
    hp = f"{round(mon.current_hp_fraction * 100)}%" if mon.current_hp_fraction is not None else "?%"
    status = f" {mon.status.name}" if mon.status else ""
    return f"{mon.species} ({hp}{status})"


def active_boosts(mon: Pokemon) -> str:
    nz = {k: v for k, v in mon.boosts.items() if v}
    if not nz:
        return ""
    return " " + " ".join(f"{'+' if v > 0 else ''}{v} {k}" for k, v in nz.items())
