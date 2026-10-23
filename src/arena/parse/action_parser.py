"""Parse free-form LLM text into a poke-env ``BattleOrder``.

Strategy: enumerate this turn's legal candidates (available moves + switches),
then find which candidate id the model named. We prefer an explicit ``Action:``
directive line, falling back to scanning the whole reply. ``to_id_str`` collapses
text to lowercase-alnum, so contiguous ids (move ids / species ids already in that
form) match as substrings. Longest id first avoids prefix collisions.
"""

from __future__ import annotations

import re
from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move import Move
from poke_env.data import to_id_str
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

_ACTION_LINE = re.compile(r"(?:action|choice|decision|final)\s*[:=\-]\s*(.+)", re.IGNORECASE)
_TERA = re.compile(r"\b(terastalli[sz]e|tera)\b(?!\s*blast)", re.IGNORECASE)


def _maybe_tera(order: BattleOrder, segment: str, battle: AbstractBattle) -> BattleOrder:
    """Upgrade a move to its tera variant when the directive asks for it and it is legal."""
    inner = getattr(order, "order", None)
    if isinstance(inner, Move) and battle.can_tera and _TERA.search(segment or ""):
        return Player.create_order(inner, terastallize=True)
    return order


def _candidates(battle: AbstractBattle) -> list[tuple[str, str, BattleOrder]]:
    cands: list[tuple[str, str, BattleOrder]] = []
    for m in battle.available_moves:
        cands.append(("move", m.id, Player.create_order(m)))
    for p in battle.available_switches:
        cands.append(("switch", to_id_str(p.species), Player.create_order(p)))
    # longest id first so e.g. "icebeam" wins over a stray "ice"
    cands.sort(key=lambda c: len(c[1]), reverse=True)
    return cands


def _scan(segment: str, cands: list[tuple[str, str, BattleOrder]], prefer_kind: Optional[str]) -> Optional[BattleOrder]:
    norm = to_id_str(segment)
    if not norm:
        return None
    pool = [c for c in cands if prefer_kind is None or c[0] == prefer_kind]
    for _kind, cid, order in pool:
        if cid and cid in norm:
            return order
    return None


def parse_action(text: str, battle: AbstractBattle) -> tuple[Optional[BattleOrder], str]:
    """Return (order or None, parse_note)."""
    cands = _candidates(battle)
    if not cands:
        return None, "no_candidates"

    # 1) explicit "Action: ..." directive (last one wins)
    directives = _ACTION_LINE.findall(text or "")
    if directives:
        seg = directives[-1]
        kind = "switch" if "switch" in seg.lower() else ("move" if "move" in seg.lower() else None)
        order = _scan(seg, cands, kind) or _scan(seg, cands, None)
        if order is not None:
            # tera only from the explicit directive (prose may say "I won't tera")
            return _maybe_tera(order, seg, battle), "directive"

    # 2) scan whole reply
    order = _scan(text or "", cands, None)
    if order is not None:
        return order, "scan"

    return None, "unparsed"
