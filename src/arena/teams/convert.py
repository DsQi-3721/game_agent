"""Convert @pkmn/smogon team JSON sets into Showdown paste text.

The teams API (https://pkmn.github.io/smogon/data/teams/<format>.json) returns
teams as ``{name, author, data: [set, ...]}`` where each set uses Showdown display
names. poke-env's ``Teambuilder`` consumes Showdown paste text, so we render to
that canonical format.
"""

from __future__ import annotations

from typing import Any

_STAT_ORDER = ["hp", "atk", "def", "spa", "spd", "spe"]
_STAT_LABEL = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}


def _stat_line(stats: dict[str, int], default: int) -> str:
    parts = [f"{stats[s]} {_STAT_LABEL[s]}" for s in _STAT_ORDER if stats.get(s, default) != default]
    return " / ".join(parts)


def set_to_paste(s: dict[str, Any]) -> str:
    species = s["species"]
    nickname = s.get("name") or s.get("nickname")
    head = f"{nickname} ({species})" if nickname and nickname != species else species
    gender = s.get("gender")
    if gender in ("M", "F"):
        head += f" ({gender})"
    if s.get("item"):
        head += f" @ {s['item']}"

    lines = [head]
    if s.get("ability"):
        lines.append(f"Ability: {s['ability']}")
    if s.get("level") and s["level"] != 100:
        lines.append(f"Level: {s['level']}")
    if s.get("shiny"):
        lines.append("Shiny: Yes")
    if s.get("teraType"):
        lines.append(f"Tera Type: {s['teraType']}")
    evs = _stat_line(s.get("evs", {}), default=0)
    if evs:
        lines.append(f"EVs: {evs}")
    if s.get("nature"):
        lines.append(f"{s['nature']} Nature")
    ivs = _stat_line(s.get("ivs", {}), default=31)
    if ivs:
        lines.append(f"IVs: {ivs}")
    for mv in s.get("moves", []):
        lines.append(f"- {mv}")
    return "\n".join(lines)


def team_to_paste(team: dict[str, Any]) -> str:
    return "\n\n".join(set_to_paste(s) for s in team["data"])
