"""Battle state -> text rendering.

``StateRenderer`` is the swappable representation module (a research variable).
``DefaultRenderer`` produces a compact, human-readable partial-observability view
(our side fully known; opponent only as revealed). ``full_info=True`` additionally
surfaces the opponent's team-preview roster. Passive augmentation results (M4) are
appended verbatim.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

from poke_env.battle.abstract_battle import AbstractBattle

from arena.state.observe import (
    active_boosts,
    known_ability,
    known_item,
    mon_label,
    revealed_moves,
    tera_type_of,
)

if TYPE_CHECKING:
    from arena.augment.base import AugmentationResult


class StateRenderer(ABC):
    @abstractmethod
    def render(
        self,
        battle: AbstractBattle,
        *,
        full_info: bool = False,
        passive_augments: Sequence["AugmentationResult"] = (),
    ) -> str: ...


class DefaultRenderer(StateRenderer):
    def render(self, battle, *, full_info=False, passive_augments=()):
        lines: list[str] = []
        lines.append(f"Turn {battle.turn} | {battle.format}")

        field = self._field(battle)
        if field:
            lines.append(field)

        # Our active
        me = battle.active_pokemon
        if me is not None:
            lines.append("")
            lines.append(f"Your active: {mon_label(me)}{active_boosts(me)}")
            if me.ability:
                lines.append(f"  Ability: {me.ability}")
            if known_item(me):
                lines.append(f"  Item: {me.item}")
            tera = tera_type_of(battle, me)
            if me.is_terastallized:
                lines.append(f"  Terastallized: {tera}")
            elif tera is not None and battle.can_tera:
                lines.append(f"  Can Tera -> {tera}")

        # Available actions
        if battle.available_moves:
            lines.append("  Moves:")
            for mv in battle.available_moves:
                pp = f" (pp {mv.current_pp}/{mv.max_pp})" if mv.max_pp else ""
                lines.append(f"    - {mv.id} [{mv.type.name} {mv.category.name.lower()} bp{mv.base_power}]{pp}")
        if battle.available_switches:
            names = ", ".join(mon_label(p) for p in battle.available_switches)
            lines.append(f"  Switches: {names}")

        # Opponent active (partial obs)
        opp = battle.opponent_active_pokemon
        if opp is not None:
            lines.append("")
            lines.append(f"Opponent active: {mon_label(opp)}{active_boosts(opp)}")
            if known_ability(opp):
                lines.append(f"  Ability (revealed): {opp.ability}")
            if known_item(opp):
                lines.append(f"  Item (revealed): {opp.item}")
            mv = revealed_moves(opp)
            if mv:
                lines.append(f"  Revealed moves: {', '.join(mv)}")

        # Benches
        my_bench = [mon_label(p) for p in battle.team.values() if not p.active]
        if my_bench:
            lines.append("")
            lines.append("Your bench: " + ", ".join(my_bench))
        opp_seen = [mon_label(p) for p in battle.opponent_team.values() if not p.active]
        if opp_seen:
            lines.append("Opponent revealed (bench): " + ", ".join(opp_seen))

        if full_info and battle.teampreview_opponent_team:
            roster = ", ".join(sorted(p.species for p in battle.teampreview_opponent_team))
            lines.append(f"[full-info] Opponent full roster: {roster}")

        for aug in passive_augments:
            lines.append("")
            lines.append(f"[{aug.name}]")
            lines.append(aug.text)

        return "\n".join(lines)

    @staticmethod
    def _field(battle: AbstractBattle) -> str:
        parts = []
        if battle.weather:
            parts.append("Weather: " + ",".join(w.name for w in battle.weather))
        if battle.fields:
            parts.append("Field: " + ",".join(f.name for f in battle.fields))
        if battle.side_conditions:
            parts.append("Your side: " + ",".join(c.name for c in battle.side_conditions))
        if battle.opponent_side_conditions:
            parts.append("Opp side: " + ",".join(c.name for c in battle.opponent_side_conditions))
        return " | ".join(parts)
