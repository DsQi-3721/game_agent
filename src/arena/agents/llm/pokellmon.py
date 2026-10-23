"""PokeLLMon: a native reimplementation of git-disl/PokeLLMon's three strategies.

Faithful to the paper "PokeLLMon: A Human-Parity Agent for Pokemon Battles"
(Hu et al., 2024), reusing our own renderer / llm client / augment system:

  1. Knowledge-augmented generation -- force-inject the knowledge augmentations
     (type matchup, damage calc, ability, base stats, pokedex) into the prompt
     regardless of the experiment's augment config, to counter hallucination.
  2. Consistent action generation -- sample N reasoned answers and majority-vote,
     with an explicit anti-"panic switching" rule that suppresses switch
     oscillation when we already switched the previous turn.
  3. In-context feedback ("in-context RL") -- derive plain-text feedback from the
     previous turn's state delta (damage dealt/taken, no-effect moves, opponent
     switches) and feed it back into the next prompt.

The official repo lives under ``external/pokellmon`` as a reference oracle; this
class is the framework-native port that routes through ``ctx.llm`` (cache + cost).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

from arena.agents.base import AgentContext
from arena.agents.llm.base import LLMAgent
from arena.parse.action_parser import parse_action
from arena.parse.validate import validate_or_fallback
from arena.registry.agents import register_agent
from arena.registry.param import Param

# Knowledge sources PokeLLMon grounds on; injected passively every turn.
_KNOWLEDGE = ("type_matchup", "damage_calc", "ability", "base_stats", "pokedex")

POKELLMON_SYSTEM = (
    "You are PokeLLMon, a human-parity competitive Pokemon (Pokemon Showdown) "
    "battle AI. Ground every claim in the provided knowledge blocks (type "
    "matchups, damage estimates, abilities) -- do NOT rely on memory that "
    "contradicts them. Reason about the matchup step by step, then commit to one "
    "legal action. Avoid panic switching: only switch when it is clearly better "
    "than attacking, and do not oscillate between switches on consecutive turns."
)

POKELLMON_INSTRUCTION = (
    "Think step by step about type matchups, damage, speed, hazards and status, "
    "using the knowledge blocks above. Then end your reply with exactly one line:\n"
    "  Action: <move_id>        (to use one of the listed Moves, use its id)\n"
    "  Action: switch <species> (to switch to one of the listed Switches)\n"
    "Choose only from the Moves and Switches listed above."
)


@dataclass
class _TurnMemory:
    """Last-turn snapshot per battle, used to synthesize text feedback."""

    my_species: Optional[str] = None
    my_hp: float = 1.0
    opp_species: Optional[str] = None
    opp_hp: float = 1.0
    action_msg: Optional[str] = None  # the order.message we played last turn


@register_agent(
    "llm_pokellmon",
    is_llm=True,
    needs_model=True,
    label="LLM - PokeLLMon (native variant)",
    params=[
        Param("n_samples", int, 3, help="consistency votes per decision", label="Samples"),
        Param("temperature", float, 0.7, help="sampling temperature", label="Temperature"),
    ],
)
class PokeLLMonAgent(LLMAgent):
    name = "llm_pokellmon"

    def __init__(self, ctx: Optional[AgentContext] = None, *, n_samples: int = 3,
                 temperature: float = 0.7, **kw):
        kw.setdefault("system_prompt", POKELLMON_SYSTEM)
        kw.setdefault("instruction", POKELLMON_INSTRUCTION)
        kw["temperature"] = temperature
        super().__init__(ctx, **kw)
        self.n_samples = max(1, n_samples)
        self._mem: dict[str, _TurnMemory] = {}
        self._knowledge = self._build_knowledge()

    # --- (1) knowledge-augmented generation ------------------------------
    @staticmethod
    def _build_knowledge():
        """An augment registry with the knowledge sources forced to PASSIVE."""
        try:
            from arena.augment.base import AugmentationRegistry
            from arena.registry.augments import entries as aug_entries

            augs = [cls() for cls in aug_entries().values()]
            modes = {n: "passive" for n in _KNOWLEDGE}
            return AugmentationRegistry(augs, modes)
        except Exception:  # noqa: BLE001  (knowledge is best-effort, never fatal)
            return None

    def render_state(self, battle: AbstractBattle) -> str:
        passive = []
        if self._knowledge is not None:
            passive = list(self._knowledge.passive(battle))
        # merge any *extra* passive augments the experiment configured on top
        if self.ctx.augmentations is not None:
            have = {a.name for a in passive}
            for a in self.ctx.augmentations.passive(battle):
                if a.name not in have:
                    passive.append(a)
        return self.renderer.render(
            battle, full_info=self.ctx.full_info, passive_augments=passive
        )

    # --- (3) in-context feedback -----------------------------------------
    def _feedback(self, battle: AbstractBattle) -> str:
        mem = self._mem.get(battle.battle_tag)
        if mem is None or mem.action_msg is None:
            return ""
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        lines: list[str] = []
        switched = "switch" in mem.action_msg
        if opp is not None and opp.species == mem.opp_species:
            drop = mem.opp_hp - _hp(opp)
            if not switched and drop > 0.02:
                lines.append(
                    f"Your last action dropped {opp.species} by ~{drop * 100:.0f}% HP."
                )
            elif not switched and drop <= 0.0:
                lines.append(
                    "Your last move did little or no damage -- reconsider the type matchup."
                )
        elif opp is not None and mem.opp_species and opp.species != mem.opp_species:
            lines.append(f"Opponent switched in {opp.species}.")
        if me is not None and me.species == mem.my_species:
            took = mem.my_hp - _hp(me)
            if took > 0.25:
                lines.append(f"You took heavy damage (~{took * 100:.0f}% HP) last turn.")
        if not lines:
            return ""
        return "Last turn feedback:\n" + "\n".join(f"- {x}" for x in lines)

    def _remember(self, battle: AbstractBattle, action_msg: Optional[str]) -> None:
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        self._mem[battle.battle_tag] = _TurnMemory(
            my_species=(me.species if me is not None else None),
            my_hp=_hp(me),
            opp_species=(opp.species if opp is not None else None),
            opp_hp=_hp(opp),
            action_msg=action_msg,
        )

    # --- (2) consistent action generation --------------------------------
    def _vote(self, orders: list[BattleOrder], battle: AbstractBattle) -> Optional[BattleOrder]:
        votes: Counter = Counter()
        rep: dict[str, BattleOrder] = {}
        for o in orders:
            if o is None:
                continue
            votes[o.message] += 1
            rep[o.message] = o
        if not votes:
            return None
        top = votes.most_common(1)[0][0]
        # anti-panic-switch: if we switched last turn and the vote wants to switch
        # again, prefer the most-voted *move* (suppresses switch oscillation).
        mem = self._mem.get(battle.battle_tag)
        last_was_switch = mem is not None and mem.action_msg and "switch" in mem.action_msg
        if last_was_switch and "switch" in top:
            move_votes = {m: c for m, c in votes.items() if "switch" not in m}
            if move_votes:
                top = max(move_votes, key=move_votes.get)
        return rep[top]

    async def act(self, battle: AbstractBattle) -> BattleOrder:
        state = self.render_state(battle)
        feedback = self._feedback(battle)
        user = state + (f"\n\n{feedback}" if feedback else "") + f"\n\n{self.instruction}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

        async def sample(i: int):
            return await self.ctx.llm.complete(
                self.model, messages, temperature=self.temperature, sample_idx=i
            )

        responses = await asyncio.gather(*(sample(i) for i in range(self.n_samples)))

        orders: list[BattleOrder] = []
        texts: list[str] = []
        for resp in responses:
            self.account(resp)
            texts.append(resp.text)
            order, _note = parse_action(resp.text, battle)
            orders.append(order)

        chosen = self._vote(orders, battle)
        final, fallback = validate_or_fallback(chosen, battle)
        self.n_decisions += 1
        self.n_fallbacks += int(fallback)
        self._remember(battle, final.message if final else None)

        if self.ctx.logger is not None:
            self.ctx.logger.log_turn(
                battle_tag=battle.battle_tag, turn=battle.turn, state_text=state,
                prompt=user, raw_llm_output="\n---\n".join(texts), tool_calls=[],
                parsed_order=(final.message if final else None), fallback_used=fallback,
                note=f"samples={self.n_samples}",
            )
        return final


def _hp(mon) -> float:
    """HP fraction in [0,1], defensively (fainted/None -> 0)."""
    if mon is None:
        return 0.0
    frac = getattr(mon, "current_hp_fraction", None)
    return float(frac) if frac is not None else 0.0
