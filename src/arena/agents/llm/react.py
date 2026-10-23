"""ReAct agent: reason, optionally call tools to gather info, then act.

Works with both tool-calling protocols (native function-calling or text protocol)
via ``make_protocol``. Tools come from TOOL-mode augmentations (e.g. pokedex).
Loops up to ``max_steps``; stops as soon as the model emits an ``Action:`` line.
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

from arena.agents.base import AgentContext
from arena.agents.llm.base import LLMAgent
from arena.llm.tool_calling import make_protocol
from arena.parse.action_parser import parse_action
from arena.parse.validate import validate_or_fallback
from arena.registry.agents import register_agent
from arena.registry.param import Param


@register_agent("llm_react", is_llm=True, needs_model=True, label="LLM - ReAct (tool-use)",
                params=[Param("max_steps", int, 4, help="reasoning/tool loops per decision", label="Max steps"),
                        Param("temperature", float, 0.0, help="sampling temperature", label="Temperature")])
class ReActAgent(LLMAgent):
    name = "llm_react"

    def __init__(self, ctx: Optional[AgentContext] = None, *, max_steps: int = 4,
                 supports_tools: Optional[bool] = None, **kw):
        super().__init__(ctx, **kw)
        self.max_steps = max_steps
        self._supports_tools = supports_tools
        self.n_tool_calls = 0  # cumulative across decisions

    def _resolve_supports_tools(self) -> bool:
        if self._supports_tools is not None:
            return self._supports_tools
        reg = getattr(self.ctx.llm, "registry", None)
        if reg is not None:
            try:
                return reg.get(self.model).supports_tools
            except Exception:  # noqa: BLE001
                pass
        return True

    async def act(self, battle: AbstractBattle) -> BattleOrder:
        tools = self.ctx.augmentations.tools() if self.ctx.augmentations is not None else []
        tool_map = {t.name: t for t in tools}
        proto = make_protocol(self._resolve_supports_tools())

        state = self.render_state(battle)
        docs = proto.tool_docs(tools)
        user = state + (f"\n\n{docs}" if docs else "") + f"\n\n{self.instruction}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]
        complete_tools = proto.complete_kwargs(tools).get("tools")

        transcript: list[str] = []
        order = None
        n_tool_calls = 0
        for _step in range(self.max_steps):
            resp = await self.ctx.llm.complete(
                self.model, messages, tools=complete_tools,
                temperature=self.temperature, sample_idx=0,
            )
            self.account(resp)
            transcript.append(resp.text)

            order, _note = parse_action(resp.text, battle)
            calls = proto.extract_calls(resp) if tools else []
            if order is not None or not calls:
                break

            results = []
            for c in calls:
                tool = tool_map.get(c.name)
                if tool is None:
                    results.append(f"unknown tool '{c.name}'")
                    continue
                try:
                    results.append(await tool.call(battle, c.arguments))
                except Exception as e:  # noqa: BLE001
                    results.append(f"error: {type(e).__name__}: {e}")
            n_tool_calls += len(calls)
            self.n_tool_calls += len(calls)
            proto.append_turn(messages, resp, calls, results)

        final, fallback = validate_or_fallback(order, battle)
        self.n_decisions += 1
        self.n_fallbacks += int(fallback)
        if self.ctx.logger is not None:
            self.ctx.logger.log_turn(
                battle_tag=battle.battle_tag, turn=battle.turn, state_text=state,
                prompt=user, raw_llm_output="\n---\n".join(transcript),
                tool_calls=[], parsed_order=(final.message if final else None),
                fallback_used=fallback, note=f"tool_calls={n_tool_calls}",
            )
        return final
