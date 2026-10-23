"""Raw battle-log search (tool only).

Returns raw Showdown protocol lines -- deliberately noisy, exactly like a raw tool
observation in the paper's executor. Agents that isolate contexts (HyMem) only
call it from the executor, which distils what matters before anything reaches
the planner.
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.data import to_id_str

from arena.augment.base import Augmentation, AugmentationResult, Tool, function_schema
from arena.registry.augments import register_augment
from arena.state.battle_log import NOISE_KINDS

MAX_LINES = 60


def search_log(battle: AbstractBattle, *, turn_from: Optional[int] = None, turn_to: Optional[int] = None,
               pokemon: str = "", keyword: str = "", max_lines: int = 40) -> list[str]:
    turn, out = 0, []
    mon_id, kw = to_id_str(pokemon or ""), (keyword or "").lower()
    for ev in getattr(battle, "_replay_data", None) or []:
        if len(ev) < 2:
            continue
        if ev[1] == "turn":
            try:
                turn = int(ev[2])
            except (IndexError, ValueError):
                pass
        if ev[1] in NOISE_KINDS:
            continue
        if turn_from is not None and turn < turn_from:
            continue
        if turn_to is not None and turn > turn_to:
            continue
        line = "|".join(ev)
        if mon_id and mon_id not in to_id_str(line):
            continue
        if kw and kw not in line.lower():
            continue
        out.append(line)
    return out[-max(1, min(int(max_lines or 40), MAX_LINES)):]


class BattleLogTool(Tool):
    name = "battle_log"

    @property
    def schema(self) -> dict:
        return function_schema(
            self.name,
            "Search the raw battle log (Showdown protocol lines) by turn range, Pokemon and/or "
            "keyword, e.g. to find what an opposing Pokemon used earlier or how much damage a move did.",
            {"turn_from": {"type": "integer"}, "turn_to": {"type": "integer"},
             "pokemon": {"type": "string", "description": "only lines mentioning this Pokemon"},
             "keyword": {"type": "string", "description": "only lines containing this text, e.g. 'move' or 'Knock Off'"},
             "max_lines": {"type": "integer", "description": f"at most {MAX_LINES}"}},
        )

    async def call(self, battle: AbstractBattle, arguments: dict) -> str:
        def _int(v):
            try:
                return int(v) if v not in (None, "") else None
            except (TypeError, ValueError):
                return None

        lines = search_log(battle, turn_from=_int(arguments.get("turn_from")),
                           turn_to=_int(arguments.get("turn_to")),
                           pokemon=str(arguments.get("pokemon") or ""),
                           keyword=str(arguments.get("keyword") or ""),
                           max_lines=_int(arguments.get("max_lines")) or 40)
        return "\n".join(lines) if lines else "battle_log: no matching lines."


@register_augment("battle_log")
class BattleLogAugmentation(Augmentation):
    name = "battle_log"

    def compute(self, battle: AbstractBattle) -> AugmentationResult:
        return AugmentationResult(self.name, "")  # tool-only: raw logs are never injected passively

    def as_tool(self) -> Optional[Tool]:
        return BattleLogTool()
