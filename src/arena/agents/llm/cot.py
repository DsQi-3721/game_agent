"""Chain-of-Thought agent: same single-shot loop as IO but prompts the model to
reason step by step before emitting the action line."""

from __future__ import annotations

from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

from arena.agents.base import AgentContext
from arena.agents.llm.base import LLMAgent
from arena.agents.llm.prompts import COT_INSTRUCTION
from arena.registry.agents import register_agent
from arena.registry.param import Param


@register_agent("llm_cot", is_llm=True, needs_model=True, label="LLM - Chain-of-Thought",
                params=[Param("temperature", float, 0.0, help="sampling temperature", label="Temperature")])
class CoTAgent(LLMAgent):
    name = "llm_cot"

    def __init__(self, ctx: Optional[AgentContext] = None, **kw):
        kw.setdefault("instruction", COT_INSTRUCTION)
        super().__init__(ctx, **kw)

    async def act(self, battle: AbstractBattle) -> BattleOrder:
        state = self.render_state(battle)
        messages = self.build_messages(state)
        resp = await self.ctx.llm.complete(
            self.model, messages, temperature=self.temperature, sample_idx=0
        )
        self.account(resp)
        return self.finalize(resp.text, battle, state=state, prompt=messages[-1]["content"])
