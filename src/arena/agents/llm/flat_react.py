"""Flat ReAct over the whole battle: the paper's ReAct baseline (Table 1).

Every event, state, reply and raw tool output stays in one growing conversation;
only a sliding window at the context budget drops the oldest decisions.
"""

from __future__ import annotations

from arena.agents.llm.longctx import PersistentContextAgent
from arena.registry.agents import register_agent
from arena.registry.param import Param


@register_agent(
    "llm_flat_react", is_llm=True, needs_model=True, label="LLM - Flat ReAct (persistent context)",
    params=[Param("max_steps", int, 4, help="model turns per decision (last is answer-only)", label="Max steps"),
            Param("obs_format", str, "raw", choices=["raw", "digest"], help="battle events as raw protocol or digest",
                  label="Observation"),
            Param("max_context_tokens", int, 0, help="sliding-window budget (0 = 90% of the model window)",
                  label="Context budget"),
            Param("temperature", float, 0.7, label="Temperature")],
)
class FlatReActAgent(PersistentContextAgent):
    name = "llm_flat_react"

    def __init__(self, ctx=None, *, max_context_tokens: int = 0, **kw):
        super().__init__(ctx, max_context_tokens=max_context_tokens or None, **kw)
