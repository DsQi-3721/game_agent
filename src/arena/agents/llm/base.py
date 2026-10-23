"""Shared scaffolding for LLM agents: render -> prompt -> LLM -> parse -> validate.

Subclasses (IO / CoT / self-consistency / ReAct) override how they call the LLM;
the render/parse/validate/log/metrics plumbing lives here so it stays consistent.
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

from arena.agents.base import Agent, AgentContext
from arena.agents.llm.prompts import ACTION_INSTRUCTION, SYSTEM
from arena.parse.action_parser import parse_action
from arena.parse.validate import validate_or_fallback
from arena.state.renderer import DefaultRenderer, StateRenderer


class LLMAgent(Agent):
    name = "llm"

    def __init__(
        self,
        ctx: Optional[AgentContext] = None,
        *,
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        renderer: Optional[StateRenderer] = None,
        system_prompt: str = SYSTEM,
        instruction: str = ACTION_INSTRUCTION,
    ):
        super().__init__(ctx)
        if self.ctx.llm is None:
            raise ValueError("LLM agents require ctx.llm to be set")
        self.model = model
        self.temperature = temperature
        self.renderer = renderer or self.ctx.renderer or DefaultRenderer()
        self.system_prompt = system_prompt
        self.instruction = instruction
        # rolling behavioral/cost metrics (consumed by eval until M6 logger lands)
        self.n_decisions = 0
        self.n_fallbacks = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_usd = 0.0

    # --- building blocks -------------------------------------------------
    def render_state(self, battle: AbstractBattle) -> str:
        passive = []
        if self.ctx.augmentations is not None:  # M4
            passive = self.ctx.augmentations.passive(battle)
        return self.renderer.render(battle, full_info=self.ctx.full_info, passive_augments=passive)

    def build_messages(self, state: str) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{state}\n\n{self.instruction}"},
        ]

    def account(self, resp) -> None:
        self.tokens_in += resp.prompt_tokens
        self.tokens_out += resp.completion_tokens
        self.cost_usd += resp.cost_usd

    def finalize(self, text: str, battle: AbstractBattle, *, state: str, prompt: str,
                 tool_calls=None) -> BattleOrder:
        order, note = parse_action(text, battle)
        final, fallback = validate_or_fallback(order, battle)
        self.n_decisions += 1
        self.n_fallbacks += int(fallback)
        if self.ctx.logger is not None:
            self.ctx.logger.log_turn(
                battle_tag=battle.battle_tag,
                turn=battle.turn,
                state_text=state,
                prompt=prompt,
                raw_llm_output=text,
                tool_calls=tool_calls or [],
                parsed_order=(order.message if order else None),
                fallback_used=fallback,
                note=note,
            )
        return final
