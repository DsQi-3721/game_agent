"""Ability augmentation: surface the active mons' abilities (ours; opp if revealed)."""

from __future__ import annotations

from poke_env.battle.abstract_battle import AbstractBattle

from arena.augment.base import Augmentation, AugmentationResult
from arena.registry.augments import register_augment


@register_augment("ability")
class AbilityAugmentation(Augmentation):
    name = "ability"

    def compute(self, battle: AbstractBattle) -> AugmentationResult:
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if me is None or opp is None:
            return AugmentationResult(self.name, "")

        lines = []
        if me.ability:
            lines.append(f"Your ability: {me.ability}")
        if opp.ability:
            lines.append(f"Opponent ability (revealed): {opp.ability}")
        elif opp.possible_abilities:
            poss = ", ".join(opp.possible_abilities.values()) if isinstance(opp.possible_abilities, dict) else ", ".join(opp.possible_abilities)
            lines.append(f"Opponent possible abilities: {poss}")

        return AugmentationResult(self.name, "\n".join(lines))
