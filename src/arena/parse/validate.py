"""Legality check + fallback for a chosen order.

``battle.valid_orders`` is the ground truth of legal singles orders this turn.
If the parsed order is missing or illegal, fall back to a safe legal move and
report it so behavioral metrics (illegal-action / fallback rate) can be tracked.
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player


def validate_or_fallback(
    order: Optional[BattleOrder], battle: AbstractBattle
) -> tuple[BattleOrder, bool]:
    """Return (order_to_play, fallback_used)."""
    try:
        legal = {o.message for o in battle.valid_orders}
    except Exception:  # noqa: BLE001
        legal = set()

    if order is not None and (not legal or order.message in legal):
        return order, False

    return Player.choose_random_move(battle), True
