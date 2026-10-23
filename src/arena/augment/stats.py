"""Base-stat / speed comparison augmentation (active vs opponent active)."""

from __future__ import annotations

from poke_env.battle.abstract_battle import AbstractBattle

from arena.augment.base import Augmentation, AugmentationResult
from arena.registry.augments import register_augment

_ORDER = ["hp", "atk", "def", "spa", "spd", "spe"]


@register_augment("base_stats")
class StatsAugmentation(Augmentation):
    name = "base_stats"

    def compute(self, battle: AbstractBattle) -> AugmentationResult:
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if me is None or opp is None:
            return AugmentationResult(self.name, "")

        ms, os_ = me.base_stats, opp.base_stats
        line = "  ".join(f"{s.upper()} {ms[s]}/{os_[s]}" for s in _ORDER)
        spe_me, spe_opp = ms["spe"], os_["spe"]
        if spe_me > spe_opp:
            spd = "You are faster (by base speed; ignores items/boosts)."
        elif spe_opp > spe_me:
            spd = "Opponent is faster (by base speed; ignores items/boosts)."
        else:
            spd = "Equal base speed (speed tie)."

        text = f"Base stats you/opp:  {line}\n{spd}"
        return AugmentationResult(self.name, text, data={"me": ms, "opp": os_})
