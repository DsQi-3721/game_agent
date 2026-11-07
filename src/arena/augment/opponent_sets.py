"""Opponent set inference: what an opposing Pokemon probably runs.

Format prior (randbats: the server's own generator; gen9ou: Smogon usage)
conditioned on everything this Pokemon has revealed in the battle.
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle

from arena.augment.base import Augmentation, AugmentationResult, Tool, function_schema
from arena.augment.calc_util import mon_names, resolve_mon
from arena.knowledge.estimate import is_own
from arena.knowledge.opponent_model import DEFAULT_MODEL
from arena.knowledge.priors import raw_prior
from arena.registry.augments import register_augment


def _fmt(choices, k: int) -> str:
    return ", ".join(f"{c.name} {round(c.p * 100)}%" for c in list(choices)[:k]) or "?"


def sets_text(battle: AbstractBattle, mon, k_moves: int = 8) -> str:
    if is_own(battle, mon):
        return f"{mon.species} is your own Pokemon (full set known)."
    b = DEFAULT_MODEL.belief(battle, mon)
    if b is not None:
        return belief_sets_text(b, k_moves)
    prior = DEFAULT_MODEL.set_prior(battle, mon)
    if prior is None:
        return (f"No set data for {mon.species} in {battle.format} "
                "(knowledge base missing? run scripts/build_knowledge.py).")
    seen = prior.revealed
    src = f"{prior.source} L{prior.level}" + (f", {prior.n} samples" if prior.n else "")
    lines = [f"{prior.name} ({src}):"]
    lines.append(f"  moves: {_fmt(prior.moves, k_moves)}")
    lines.append(f"  item: {_fmt(prior.items, 4)} | ability: {_fmt(prior.abilities, 3)} | tera: {_fmt(prior.tera, 4)}")
    if prior.source != "randbats" and prior.spreads:
        s = prior.spreads[0]
        lines.append(f"  common spread: {s.get('nature')} {'/'.join(map(str, s.get('evs', [])))} "
                     f"({round(s.get('p', 0) * 100)}%)")
    rev = []
    if seen.get("moves"):
        rev.append("moves " + ", ".join(seen["moves"]))
    for key in ("item", "ability", "tera"):
        if seen.get(key):
            rev.append(f"{key} {seen[key]}")
    lines.append("  revealed: " + ("; ".join(rev) if rev else "nothing yet"))
    return "\n".join(lines)


def belief_sets_text(b, k_moves: int = 8) -> str:
    """The team-aware model's view: usage + teammates + co-reveals + in-battle evidence."""
    from arena.opponent.model import _name, spread_text
    pct = lambda d, kind, k: ", ".join(f"{_name(kind, x) if kind != 'tera' else x.title()} {round(p * 100)}%"  # noqa: E731
                                       for x, p in list(d.items())[:k]) or "?"
    lines = [f"{b.name} (usage stats + teammates + what it showed{', ' + str(b.n) + ' teams' if b.n else ''}):"]
    seen = set(b.revealed.get("moves") or ())
    lines.append(f"  moves: {pct({m: p for m, p in b.moves.items() if m not in seen}, 'move', k_moves)}"
                 + (f" (revealed: {', '.join(sorted(seen))})" if seen else ""))
    item = (f"{_name('item', b.revealed['item'])} (revealed)" if b.revealed.get("item")
            else f"none now (had {_name('item', b.original_item)})" if b.original_item else pct(b.items, "item", 4))
    lines.append(f"  item: {item} | ability: {pct(b.abilities, 'ability', 3)} | tera: {pct(b.tera, 'tera', 4)}")
    if b.spreads:
        lines.append("  spreads: " + "; ".join(f"{spread_text(n, e)} {round(p * 100)}%" for n, e, p in b.spreads[:3]))
    if b.notes:
        lines.append("  evidence: " + "; ".join(b.notes[:6]))
    return "\n".join(lines)


class OpponentSetsTool(Tool):
    name = "opponent_sets"

    @property
    def schema(self) -> dict:
        return function_schema(
            self.name,
            "Likely set of an opponent's Pokemon (move/item/ability/tera probabilities from the "
            "format's set data, conditioned on what it has revealed). Also works for species not "
            "yet seen in this battle.",
            {"pokemon": {"type": "string", "description": "opponent Pokemon or any species (default: their active)"}},
        )

    async def call(self, battle: AbstractBattle, arguments: dict) -> str:
        name = arguments.get("pokemon")
        mon = resolve_mon(battle, name, default="theirs", side="theirs")
        if mon is not None:
            return sets_text(battle, mon)
        species = resolve_mon(battle, name, side="mine")     # e.g. a mirror species not revealed yet
        prior = raw_prior(battle.format, species.species if species is not None else (name or ""))
        if prior is None:
            return f"opponent_sets: unknown Pokemon {name!r}. Seen -- {mon_names(battle)}"
        return (f"{prior.name} ({prior.source} L{prior.level}, not seen in this battle):\n"
                f"  moves: {_fmt(prior.moves, 8)}\n  item: {_fmt(prior.items, 4)} | "
                f"ability: {_fmt(prior.abilities, 3)} | tera: {_fmt(prior.tera, 4)}")


@register_augment("opponent_sets")
class OpponentSetsAugmentation(Augmentation):
    name = "opponent_sets"

    def compute(self, battle: AbstractBattle) -> AugmentationResult:
        opp = battle.opponent_active_pokemon
        if opp is None:
            return AugmentationResult(self.name, "")
        return AugmentationResult(self.name, sets_text(battle, opp, k_moves=6))

    def as_tool(self) -> Optional[Tool]:
        return OpponentSetsTool()
