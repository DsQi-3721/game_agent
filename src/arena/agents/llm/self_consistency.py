"""Self-consistency agent: sample N reasoned answers, majority-vote the action.

Samples run concurrently and each carries a distinct ``sample_idx`` so the cache
keeps them separate (and never collapses temperature>0 samples).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

from arena.agents.base import AgentContext
from arena.agents.llm.base import LLMAgent
from arena.agents.llm.prompts import COT_INSTRUCTION
from arena.parse.action_parser import parse_action
from arena.parse.validate import validate_or_fallback
from arena.registry.agents import register_agent
from arena.registry.param import Param


@register_agent("llm_self_consistency", is_llm=True, needs_model=True, label="LLM - Self-Consistency",
                params=[Param("n_samples", int, 5, help="votes per decision", label="Samples"),
                        Param("temperature", float, 0.7, help="sampling temperature", label="Temperature")])
class SelfConsistencyAgent(LLMAgent):
    name = "llm_self_consistency"

    def __init__(self, ctx: Optional[AgentContext] = None, *, n_samples: int = 5,
                 temperature: float = 0.7, **kw):
        kw.setdefault("instruction", COT_INSTRUCTION)
        kw["temperature"] = temperature
        super().__init__(ctx, **kw)
        self.n_samples = n_samples

    async def act(self, battle: AbstractBattle) -> BattleOrder:
        state = self.render_state(battle)
        messages = self.build_messages(state)

        async def sample(i: int):
            return await self.ctx.llm.complete(
                self.model, messages, temperature=self.temperature, sample_idx=i
            )

        responses = await asyncio.gather(*(sample(i) for i in range(self.n_samples)))

        votes: Counter = Counter()
        rep: dict = {}
        texts = []
        for resp in responses:
            self.account(resp)
            texts.append(resp.text)
            order, _note = parse_action(resp.text, battle)
            if order is not None:
                votes[order.message] += 1
                rep[order.message] = order

        chosen = rep[votes.most_common(1)[0][0]] if votes else None
        final, fallback = validate_or_fallback(chosen, battle)
        self.n_decisions += 1
        self.n_fallbacks += int(fallback)
        if self.ctx.logger is not None:
            self.ctx.logger.log_turn(
                battle_tag=battle.battle_tag, turn=battle.turn, state_text=state,
                prompt=messages[-1]["content"], raw_llm_output="\n---\n".join(texts),
                tool_calls=[], parsed_order=(final.message if final else None),
                fallback_used=fallback, note=f"vote={dict(votes)}",
            )
        return final
