"""Pokedex knowledge-base lookup. Primarily a TOOL the agent calls when it needs
facts it doesn't know: species typing/stats/abilities, a move's numbers and
effect, what an ability or item does. Backed by the local Showdown Dex export
(``data/knowledge/dex_gen9.json``, with effect text) and poke-env's GenData as a
fallback -- offline and reproducible.
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.data import GenData, to_id_str

from arena.augment.base import Augmentation, AugmentationResult, Tool, function_schema
from arena.knowledge import store
from arena.registry.augments import register_augment

_KINDS = ("species", "move", "ability", "item")


def _species(e: dict, dex: dict) -> str:
    types = "/".join(e.get("types", []))
    bs = e.get("baseStats", {})
    abil = []
    for a in (e.get("abilities") or {}).values():
        desc = (dex.get("abilities", {}).get(to_id_str(a)) or {}).get("desc", "")
        abil.append(f"{a} ({desc})" if desc else a)
    tier = f"; tier {e['tier']}" if e.get("tier") else ""
    return (f"{e.get('name')}: {types}; base HP{bs.get('hp')}/Atk{bs.get('atk')}/Def{bs.get('def')}/"
            f"SpA{bs.get('spa')}/SpD{bs.get('spd')}/Spe{bs.get('spe')}; abilities: {', '.join(abil)}{tier}")


def _move(e: dict) -> str:
    acc = "always hits" if e.get("accuracy") is True else f"acc {e.get('accuracy')}"
    pr = f", priority {e['priority']:+d}" if e.get("priority") else ""
    desc = f" -- {e['desc']}" if e.get("desc") else ""
    return (f"{e.get('name')}: {e.get('type')} {e.get('category')}, power {e.get('basePower')}, "
            f"{acc}{pr}, pp {e.get('pp')}{desc}")


def lookup(kind: str, query: str, gen: int = 9) -> str:
    q = to_id_str(query or "")
    if not q:
        return "pokedex_lookup: empty query."
    dex = store.dex()
    kinds = [kind] if kind in _KINDS else list(_KINDS)  # "auto": first table that has it
    for k in kinds:
        table = dex.get({"species": "species", "move": "moves", "ability": "abilities", "item": "items"}[k], {})
        e = table.get(q)
        if e:
            if k == "species":
                return _species(e, dex)
            if k == "move":
                return _move(e)
            return f"{e.get('name')} ({k}): {e.get('desc') or 'no description'}"
    # fallback: poke-env's bundled data (no effect text)
    gd = GenData.from_gen(gen)
    if "species" in kinds and q in gd.pokedex:
        return _species(gd.pokedex[q], {})
    if "move" in kinds and q in gd.moves:
        return _move(gd.moves[q])
    return f"No {kind if kind in _KINDS else 'entry'} named {query!r}."


class PokedexTool(Tool):
    name = "pokedex_lookup"

    @property
    def schema(self) -> dict:
        return function_schema(
            self.name,
            "Look up a Pokemon species (typing, base stats, abilities), a move (type, power, "
            "accuracy, effect), an ability or an item (what it does).",
            {"query": {"type": "string", "description": "name to look up, e.g. 'Garchomp', 'Knock Off', 'Leftovers'"},
             "kind": {"type": "string", "enum": ["auto", *_KINDS], "description": "optional; default auto"}},
            required=["query"],
        )

    async def call(self, battle: AbstractBattle, arguments: dict) -> str:
        return lookup((arguments.get("kind") or "auto").lower(), arguments.get("query", ""),
                      getattr(battle, "gen", 9) or 9)


@register_augment("pokedex")
class PokedexAugmentation(Augmentation):
    name = "pokedex"

    def compute(self, battle: AbstractBattle) -> AugmentationResult:
        # Not meaningful as a passive dump; this augmentation is tool-only.
        return AugmentationResult(self.name, "")

    def as_tool(self) -> Optional[Tool]:
        return PokedexTool()
