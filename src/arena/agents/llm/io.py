"""Single-shot IO agent: render state, ask the LLM once, parse the action."""

from __future__ import annotations

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

from arena.agents.llm.base import LLMAgent
from arena.registry.agents import register_agent
from arena.registry.param import Param


@register_agent("llm_io", is_llm=True, needs_model=True, label="LLM - IO (single-shot)",
                params=[Param("temperature", float, 0.0, help="sampling temperature", label="Temperature")])
class IOAgent(LLMAgent):
    name = "llm_io"

    async def act(self, battle: AbstractBattle) -> BattleOrder:
        state = self.render_state(battle)
        messages = self.build_messages(state)
        resp = await self.ctx.llm.complete(
            self.model, messages, temperature=self.temperature, sample_idx=0
        )
        self.account(resp)
        return self.finalize(
            resp.text, battle, state=state, prompt=messages[-1]["content"]
        )
