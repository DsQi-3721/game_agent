"""Opponent set priors, conditioned on what has been revealed in this battle.

The prior for a species comes from the format's set distribution (randbats: the
server's own generator; gen9ou: Smogon usage). Conditioning is deliberately
simple and transparent: revealed moves/item/ability/tera become certain (p=1)
and everything else keeps its prior probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from poke_env.data import to_id_str

from arena.knowledge import store


@dataclass(frozen=True)
class Choice:
    id: str
    name: str
    p: float


@dataclass(frozen=True)
class SetPrior:
    species: str
    name: str
    level: int
    source: str                       # "randbats" | "gen9ou" | "none"
    n: int = 0                        # sample size behind the prior (0 = unknown)
    moves: tuple[Choice, ...] = ()
    items: tuple[Choice, ...] = ()
    abilities: tuple[Choice, ...] = ()
    tera: tuple[Choice, ...] = ()
    spreads: tuple[dict, ...] = ()    # {nature, evs[6], ivs[6], p}
    revealed: dict = field(default_factory=dict)  # moves(list)/item/ability/tera actually seen

    def likely_moves(self, k: int = 4) -> list[Choice]:
        """Revealed moves first, then the most probable others, up to ``k``."""
        seen = list(self.revealed.get("moves", []))
        out = [c for c in self.moves if c.id in seen]
        out += [Choice(m, m, 1.0) for m in seen if m not in {c.id for c in out}]
        for c in self.moves:
            if len(out) >= max(k, len(seen)):
                break
            if c.id not in seen:
                out.append(c)
        return out

    def top(self, attr: str) -> Optional[Choice]:
        seq = getattr(self, attr)
        return seq[0] if seq else None


def _choices(rows) -> tuple[Choice, ...]:
    return tuple(Choice(str(r.get("id") or to_id_str(r.get("name", ""))), str(r.get("name", r.get("id"))),
                        float(r.get("p", 0.0))) for r in rows or [])


def raw_prior(battle_format: str, species: str) -> Optional[SetPrior]:
    table = store.species_sets(battle_format)
    entry = table.get(species) or table.get(to_id_str(species))
    if not entry:
        return None
    src = "randbats" if "randombattle" in (battle_format or "") else "gen9ou"
    return SetPrior(
        species=to_id_str(species), name=entry.get("name", species),
        level=int(entry.get("level") or 100), source=src, n=int(entry.get("n") or 0),
        moves=_choices(entry.get("moves")), items=_choices(entry.get("items")),
        abilities=_choices(entry.get("abilities")), tera=_choices(entry.get("tera")),
        spreads=tuple(entry.get("spreads") or ()),
    )


def condition(prior: SetPrior, *, moves=(), item: Optional[str] = None,
              ability: Optional[str] = None, tera: Optional[str] = None) -> SetPrior:
    revealed: dict = {}
    pr = prior
    if moves:
        revealed["moves"] = [to_id_str(m) for m in moves]
    if item:
        iid = to_id_str(item)
        name = next((c.name for c in prior.items if c.id == iid), item)
        pr = replace(pr, items=(Choice(iid, name, 1.0),))
        revealed["item"] = iid
    if ability:
        aid = to_id_str(ability)
        name = next((c.name for c in prior.abilities if c.id == aid), ability)
        pr = replace(pr, abilities=(Choice(aid, name, 1.0),))
        revealed["ability"] = aid
    if tera:
        pr = replace(pr, tera=(Choice(tera, tera, 1.0),))
        revealed["tera"] = tera
    return replace(pr, revealed=revealed)
