"""Shared helpers for the game tools: name resolution and estimated damage.

poke-env's ``calculate_damage`` needs both Pokemon to have all six stats; under
partial observability the opponent's are ``None``. ``assumed()`` fills them (and,
optionally, the most likely item/ability) from the opponent model for the
duration of one synchronous calculation and always restores them, so no estimate
ever leaks into the battle state. Never ``await`` inside it.
"""

from __future__ import annotations

import difflib
import math
import re
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.calc import calculate_damage
from poke_env.data import GenData, to_id_str

from arena.knowledge.estimate import STATS, StatEstimate, is_own
from arena.knowledge.opponent_model import DEFAULT_MODEL, OpponentModel
from arena.state.observe import identifier_in

_MY_WORDS = {"me", "my", "mine", "self", "you", "your", "yours", "ours", "our", "own", "ally"}
_OPP_WORDS = {"opp", "opponent", "opponents", "foe", "enemy", "their", "theirs", "them", "target"}
_ACTIVE_WORDS = {"active", "current", "lead", "pokemon", "mon"}


# --- names -------------------------------------------------------------------
def _pool(battle: AbstractBattle) -> list[tuple[str, Pokemon]]:
    return [("your", p) for p in battle.team.values()] + [("opp", p) for p in battle.opponent_team.values()]


def mon_names(battle: AbstractBattle) -> str:
    mine = ", ".join(p.species for p in battle.team.values())
    theirs = ", ".join(p.species for p in battle.opponent_team.values()) or "(none seen)"
    return f"yours: {mine}; opponent's: {theirs}"


def resolve_mon(battle: AbstractBattle, name: Optional[str], default: str = "mine",
                side: str = "any") -> Optional[Pokemon]:
    """Fuzzy name -> Pokemon. ``side``: "mine" | "theirs" | "any" (then ``default`` side first).

    "opp X" / "my X" prefixes pick the side explicitly. Mirror matches put the same
    species on both sides, so a side-bound lookup never falls back to the other side.
    An empty name (or just "active") means that side's active Pokemon.
    """
    tokens = re.findall(r"[a-z0-9]+", (name or "").lower())
    if any(t in _OPP_WORDS for t in tokens):
        side = "theirs"
    elif any(t in _MY_WORDS for t in tokens) and side == "any":
        side = "mine"
    n = "".join(t for t in tokens if t not in _OPP_WORDS | _MY_WORDS | _ACTIVE_WORDS)
    first = side if side != "any" else ("mine" if default == "mine" else "theirs")
    if not n:
        return battle.active_pokemon if first == "mine" else battle.opponent_active_pokemon
    order = {"mine": ["your"], "theirs": ["opp"]}.get(side) or (["your", "opp"] if first == "mine" else ["opp", "your"])
    pool = [(sd, p) for sd in order for s2, p in _pool(battle) if s2 == sd]
    for _, p in pool:
        if to_id_str(p.species) == n:
            return p
    for _, p in pool:
        sid = to_id_str(p.species)
        if sid.startswith(n) or n.startswith(sid):
            return p
    close = difflib.get_close_matches(n, [to_id_str(p.species) for _, p in pool], n=1, cutoff=0.75)
    if close:
        return next(p for _, p in pool if to_id_str(p.species) == close[0])
    return None


def known_or_likely_moves(battle: AbstractBattle, mon: Pokemon, k: int = 4,
                          model: OpponentModel = DEFAULT_MODEL) -> list[Move]:
    """Ours: our moves. Opponent's: revealed moves, then the most likely others."""
    gen = getattr(battle, "gen", 9) or 9
    if is_own(battle, mon):
        return list(mon.moves.values())
    prior = model.set_prior(battle, mon)
    ids = [c.id for c in prior.likely_moves(k)] if prior else [m.id for m in mon.moves.values()]
    out = []
    for mid in ids:
        try:
            out.append(mon.moves.get(mid) or Move(mid, gen=gen))
        except Exception:  # noqa: BLE001  (unknown/renamed move id in a prior)
            continue
    return out


_TERA_WORDS = re.compile(r"\([^)]*\)|\btera(?:stalli[sz]ed|stalli[sz]e)?\b(?!\s*blast)", re.IGNORECASE)


def clean_move_name(name: str) -> tuple[str, bool]:
    """'Tera Close Combat (Steel)' -> ('Close Combat', True)."""
    cleaned = _TERA_WORDS.sub(" ", name or "")
    return " ".join(cleaned.split()), cleaned != (name or "") and bool(re.search(r"tera", name or "", re.I))


def resolve_move(battle: AbstractBattle, mon: Optional[Pokemon], name: str,
                 model: OpponentModel = DEFAULT_MODEL) -> Optional[Move]:
    n = to_id_str(clean_move_name(name)[0])
    if not n:
        return None
    gen = getattr(battle, "gen", 9) or 9
    pool = known_or_likely_moves(battle, mon, k=8, model=model) if mon is not None else []
    for m in pool:
        if m.id == n:
            return m
    moves = GenData.from_gen(gen).moves
    if n in moves:
        return Move(n, gen=gen)
    close = difflib.get_close_matches(n, [m.id for m in pool] + list(moves), n=1, cutoff=0.8)
    return Move(close[0], gen=gen) if close else None


# --- damage ------------------------------------------------------------------
def _needs_stats(mon: Pokemon) -> bool:
    return any(not isinstance((mon.stats or {}).get(k), (int, float)) for k in STATS)


@contextmanager
def assumed(mon: Pokemon, est: StatEstimate, *, item: Optional[str] = None, ability: Optional[str] = None):
    """Temporarily give ``mon`` estimated stats (+ assumed item/ability). Sync only."""
    saved = (mon._stats, mon._item, mon._ability)
    try:
        mon._stats = dict(est.stats)
        if item is not None:
            mon._item = item
        if ability is not None:
            mon._ability = ability
        yield mon
    finally:
        mon._stats, mon._item, mon._ability = saved


@contextmanager
def assumed_tera(mon: Pokemon, type_name: Optional[str]):
    """Temporarily terastallize ``mon`` into ``type_name`` (no-op if None). Sync only."""
    if not type_name or mon.is_terastallized:
        yield mon
        return
    saved = (mon._terastallized, mon._terastallized_type)
    try:
        mon._terastallized, mon._terastallized_type = True, PokemonType.from_name(type_name)
        yield mon
    finally:
        mon._terastallized, mon._terastallized_type = saved


@dataclass(frozen=True)
class DamageEstimate:
    attacker: str
    defender: str
    move: str
    pct_lo: float
    pct_hi: float
    ko_lo: Optional[int]       # hits to KO from current HP (best roll)
    ko_hi: Optional[int]       # ... worst roll
    note: str

    def line(self) -> str:
        if self.pct_hi <= 0:
            return f"{self.attacker} {self.move} -> {self.defender}: no damage"
        if self.ko_lo is None:
            ko = ""
        elif self.ko_lo == self.ko_hi:
            ko = f", {self.ko_lo}HKO" if self.ko_lo > 1 else ", OHKO"
        else:
            ko = f", {self.ko_lo}-{self.ko_hi} hits to KO"
        return (f"{self.attacker} {self.move} -> {self.defender}: "
                f"{self.pct_lo:.0f}-{self.pct_hi:.0f}%{ko}")


# Items a Pokemon shows as it enters the field ("X floats in the air with its Air Balloon!"): one that has been
# on the field with its item still unknown does not hold them, so the estimate never assumes them.
REVEALED_ON_ENTRY = {"airballoon"}


def _assumption(battle, mon, model) -> tuple[Optional[StatEstimate], Optional[str], Optional[str], str]:
    est = model.estimate(battle, mon)
    if est is None:
        return None, None, None, ""
    item = ability = None
    notes = [est.note]
    prior = model.set_prior(battle, mon)
    if prior is not None:
        if mon.item == "unknown_item":       # None = known to hold nothing (knocked off, consumed, popped)
            top = next((c for c in prior.items if c.id not in REVEALED_ON_ENTRY), None)
            if top and top.p >= 0.5:
                item = top.id
                notes.append(f"assumes {top.name}")
        if not mon.ability:
            top = prior.top("abilities")
            if top and top.p >= 0.5:
                ability = top.id
                notes.append(f"assumes {top.name}")
    return est, item, ability, ", ".join(notes)


def damage_estimate(battle: AbstractBattle, attacker: Pokemon, defender: Pokemon, move: Move,
                    model: OpponentModel = DEFAULT_MODEL, *, attacker_tera: Optional[str] = None,
                    defender_tera: Optional[str] = None) -> Optional[DamageEstimate]:
    atk_id, def_id = identifier_in(battle, attacker), identifier_in(battle, defender)
    if atk_id is None or def_id is None or move is None:
        return None
    notes = []
    with ExitStack() as stack:
        hp_est: Optional[StatEstimate] = None
        for mon in (attacker, defender):
            if is_own(battle, mon) or not _needs_stats(mon):
                continue
            est, item, ability, note = _assumption(battle, mon, model)
            if est is None:
                return None
            if mon is defender:
                hp_est = est
            notes.append(f"{mon.species}: {note}")
            stack.enter_context(assumed(mon, est, item=item, ability=ability))
        try:
            stack.enter_context(assumed_tera(attacker, attacker_tera))
            stack.enter_context(assumed_tera(defender, defender_tera))
        except Exception:  # noqa: BLE001  (unknown type name)
            return None
        if attacker_tera:
            notes.append(f"{attacker.species} terastallized {attacker_tera}")
        if defender_tera:
            notes.append(f"{defender.species} terastallized {defender_tera}")
        try:
            lo, hi = calculate_damage(atk_id, def_id, move, battle)
        except Exception:  # noqa: BLE001  (poke-env calc has unhandled edge cases)
            return None
    if is_own(battle, defender):
        max_hp, cur = defender.max_hp or 0, defender.current_hp or 0
    else:
        max_hp = (hp_est.stats["hp"] if hp_est else (defender.stats or {}).get("hp")) or 0
        cur = round((defender.current_hp_fraction or 0) * max_hp)
    if not max_hp:
        return None
    ko_lo = math.ceil(cur / hi) if hi > 0 and cur > 0 else None
    ko_hi = math.ceil(cur / lo) if lo > 0 and cur > 0 else None
    return DamageEstimate(attacker.species, defender.species, move.id,
                          100.0 * lo / max_hp, 100.0 * hi / max_hp, ko_lo, ko_hi, "; ".join(notes))


# --- speed -------------------------------------------------------------------
def boost_mult(stage: int) -> float:
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
