"""Shared text views of a battle for agents with a persistent context.

HyMem and the long-context baselines all read the battle through these views, so
comparisons differ only in *context management*, not in what the model is shown:

- ``legal_options``  -- exactly ``battle.valid_orders``, numbered (moves, tera variants,
  switches); small models answer more reliably with an index than with an id.
- ``render_brief``   -- the static battle brief (task ``q``): format, objective, our six
  full sets, the opponent's team preview. Built once per battle.
- ``render_state``   -- the current snapshot (replaced every decision, never accumulated).
  ``dossier=True`` additionally lists everything revealed about every opponent mon;
  the default keeps only the active one, so retaining bench reveals is the job of the
  agent's own context/memory -- the quantity under study.
- ``entity_sheet``   -- the names an executor needs to call tools correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.data import to_id_str
from poke_env.player.battle_order import BattleOrder

from arena.state.observe import (
    active_boosts,
    hp_pct,
    known_ability,
    known_item,
    revealed_moves,
    tera_type_of,
)


@dataclass(frozen=True)
class Option:
    idx: int            # 1-based number shown to the model
    kind: str           # "move" | "switch"
    id: str             # move id / species id
    tera: bool
    order: BattleOrder
    label: str


def _move_name(m: Move) -> str:
    return (m.entry or {}).get("name", m.id) if hasattr(m, "entry") else m.id


def _types(mon: Pokemon) -> str:
    return "/".join(t.name.title() for t in mon.types if t is not None)


def _status(mon: Pokemon) -> str:
    return f" {mon.status.name}" if mon.status else ""


def legal_options(battle: AbstractBattle) -> list[Option]:
    """Numbered legal actions == ``battle.valid_orders`` (minus default/pass orders)."""
    try:
        orders = list(battle.valid_orders)
    except Exception:  # noqa: BLE001
        orders = []
    moves, teras, switches = [], [], []
    for o in orders:
        inner = getattr(o, "order", None)
        if isinstance(inner, Move):
            (teras if getattr(o, "terastallize", False) else moves).append(o)
        elif isinstance(inner, Pokemon):
            switches.append(o)
    me = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    tera_t = tera_type_of(battle, me) if me is not None else None
    out: list[Option] = []
    for o in moves + teras:
        m: Move = o.order
        acc = "-" if m.accuracy is True else f"{round(float(m.accuracy) * 100)}%"
        pp = f", pp {m.current_pp}/{m.max_pp}" if m.max_pp else ""
        prio = f", priority {m.priority:+d}" if m.priority else ""
        tera = f" + TERASTALLIZE ({tera_t})" if o in teras else ""
        eff = ""
        if opp is not None and m.base_power > 0:   # public typing only (revealed tera included)
            try:
                mult = opp.damage_multiplier(m)
                eff = f", x{mult:g} vs {opp.species}" + (" (NO EFFECT)" if mult == 0 else "")
            except Exception:  # noqa: BLE001
                eff = ""
        label = (f"move {m.id}{tera} -- {_move_name(m)}: {m.type.name.title()} "
                 f"{m.category.name.lower()}, power {m.base_power}, acc {acc}{prio}{pp}{eff}")
        out.append(Option(len(out) + 1, "move", m.id, o in teras, o, label))
    for o in switches:
        p: Pokemon = o.order
        label = f"switch {p.species} ({hp_pct(p)}%{_status(p)}) [{_types(p)}]"
        out.append(Option(len(out) + 1, "switch", to_id_str(p.species), False, o, label))
    return out


def render_options(options: list[Option]) -> str:
    return "\n".join(f"{o.idx}. {o.label}" for o in options)


def _set_line(battle: AbstractBattle, mon: Pokemon) -> str:
    moves = ", ".join(_move_name(m) for m in mon.moves.values()) or "?"
    st = mon.stats or {}
    stats = " ".join(f"{k[:3].title()}{v}" for k, v in st.items() if v and k != "hp") if st else ""
    hp = f"HP{mon.max_hp} " if mon.max_hp else ""
    tera = tera_type_of(battle, mon)
    parts = [f"{mon.species} L{mon.level} [{_types(mon)}]"]
    if mon.item:
        parts.append(f"@ {mon.item}")
    if mon.ability:
        parts.append(f"| ability {mon.ability}")
    if tera:
        parts.append(f"| tera {tera}")
    parts.append(f"| moves: {moves}")
    if stats:
        parts.append(f"| {hp}{stats}")
    return " ".join(parts)


def render_brief(battle: AbstractBattle) -> str:
    """Static per-battle task description (q)."""
    fmt = battle.format or "?"
    lines = [f"Format: {fmt} (singles). You are {battle.player_role}.",
             "Objective: make all of the opponent's Pokemon faint before all of yours faint.",
             "Your team (full sets):"]
    for mon in battle.team.values():
        lines.append(f"- {_set_line(battle, mon)}")
    preview = getattr(battle, "teampreview_opponent_team", None) or []
    if preview:
        lines.append("Opponent team preview: " + ", ".join(sorted(p.species for p in preview)))
    else:
        lines.append("Opponent team: unknown; each Pokemon is revealed when it switches in.")
    return "\n".join(lines)


def _field_line(battle: AbstractBattle) -> str:
    parts = []
    if battle.weather:
        parts.append("weather " + ",".join(w.name.lower() for w in battle.weather))
    if battle.fields:
        parts.append("field " + ",".join(f.name.lower() for f in battle.fields))
    if battle.side_conditions:
        parts.append("your side " + ",".join(
            f"{c.name.lower()}" + (f"x{n}" if isinstance(n, int) and n > 1 else "")
            for c, n in battle.side_conditions.items()))
    if battle.opponent_side_conditions:
        parts.append("opp side " + ",".join(
            f"{c.name.lower()}" + (f"x{n}" if isinstance(n, int) and n > 1 else "")
            for c, n in battle.opponent_side_conditions.items()))
    return " | ".join(parts)


def _opp_reveals(mon: Pokemon) -> str:
    bits = []
    mv = revealed_moves(mon)
    if mv:
        bits.append("moves " + ", ".join(mv))
    if known_item(mon):
        bits.append(f"item {mon.item}")
    if known_ability(mon):
        bits.append(f"ability {mon.ability}")
    if mon.is_terastallized and mon.tera_type is not None:
        bits.append(f"terastallized {mon.tera_type.name.title()}")
    return "; ".join(bits)


def render_state(battle: AbstractBattle, *, dossier: bool = False) -> str:
    """The current snapshot (no action list; see ``legal_options``)."""
    lines = [f"Turn {battle.turn}"]
    fl = _field_line(battle)
    if fl:
        lines[0] += f" | {fl}"
    me = battle.active_pokemon
    if me is not None:
        tera = tera_type_of(battle, me)
        if me.is_terastallized:
            tstate = f" | terastallized {tera}"
        elif battle.can_tera and tera:
            tstate = f" | can terastallize ({tera})"
        else:
            tstate = ""
        lines.append(f"Your active: {me.species} {hp_pct(me)}%{_status(me)}{active_boosts(me)}"
                     f" [{_types(me)}]{tstate}")
    bench = [f"{p.species} {'fainted' if p.fainted else str(hp_pct(p)) + '%'}{_status(p)}"
             for p in battle.team.values() if not p.active]
    if bench:
        lines.append("Your bench: " + ", ".join(bench))
    opp = battle.opponent_active_pokemon
    if opp is not None:
        rev = _opp_reveals(opp)
        lines.append(f"Opp active: {opp.species} L{opp.level} {hp_pct(opp)}%{_status(opp)}"
                     f"{active_boosts(opp)} [{_types(opp)}]" + (f" | revealed: {rev}" if rev else ""))
    seen = [p for p in battle.opponent_team.values() if not p.active]
    if seen:
        if dossier:
            for p in seen:
                rev = _opp_reveals(p)
                lines.append(f"Opp bench: {p.species} {'fainted' if p.fainted else str(hp_pct(p)) + '%'}"
                             f"{_status(p)}" + (f" | revealed: {rev}" if rev else ""))
        else:
            lines.append("Opp bench (seen): " + ", ".join(
                f"{p.species} {'fainted' if p.fainted else str(hp_pct(p)) + '%'}" for p in seen))
    n_unseen = max(0, 6 - len(battle.opponent_team))
    if n_unseen:
        lines.append(f"Opp unrevealed: {n_unseen}")
    if battle.force_switch:
        lines.append("You must switch (your active Pokemon fainted or is being switched out).")
    return "\n".join(lines)


def entity_sheet(battle: AbstractBattle) -> str:
    """Exact names for tool arguments."""
    me = battle.active_pokemon
    lines = []
    if me is not None:
        lines.append(f"Your active: {me.species} (moves: {', '.join(m.id for m in me.moves.values())})")
    bench = [p.species for p in battle.team.values() if not p.active and not p.fainted]
    if bench:
        lines.append("Your bench: " + ", ".join(bench))
    opp = battle.opponent_active_pokemon
    if opp is not None:
        lines.append(f"Opp active: {opp.species}")
    seen = [p.species for p in battle.opponent_team.values() if not p.active]
    if seen:
        lines.append("Opp seen: " + ", ".join(seen))
    return "\n".join(lines)


def option_for(options: list[Option], order: Optional[BattleOrder]) -> Optional[Option]:
    if order is None:
        return None
    for o in options:
        if o.order.message == order.message:
            return o
    return None
