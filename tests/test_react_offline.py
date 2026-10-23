"""M5 offline: ReAct loop drives tool calls then acts, both protocols (mock)."""

import pickle
from pathlib import Path

import pytest

from arena.agents.base import AgentContext
from arena.agents.llm.react import ReActAgent
from arena.augment.base import AugmentationRegistry
from arena.llm.client import LLMResponse, MockLLMClient, ToolCall

FIXTURE = Path(__file__).parent / "fixtures" / "battles" / "gen9random_midbattle.pkl"


@pytest.fixture(scope="module")
def battle():
    with open(FIXTURE, "rb") as f:
        return pickle.load(f)


def _ctx(responder):
    return AgentContext(
        llm=MockLLMClient(responder),
        augmentations=AugmentationRegistry.from_config({"pokedex": "tool"}),
    )


async def test_react_text_protocol(battle):
    target = battle.available_moves[0].id
    seen = {"tool_calls": 0}

    def responder(messages, tools, idx):
        last = messages[-1]["content"]
        if "Tool results:" in last:
            return f"Now I know enough.\nAction: {target}"
        seen["tool_calls"] += 1
        return 'Let me check.\nTool: pokedex_lookup {"kind": "species", "query": "Garganacl"}'

    agent = ReActAgent(_ctx(responder), model="mock", supports_tools=False, max_steps=4)
    order = await agent.act(battle)
    assert order.message in {o.message for o in battle.valid_orders}
    assert agent.n_fallbacks == 0
    assert seen["tool_calls"] >= 1  # the model actually invoked a tool


async def test_react_native_protocol(battle):
    target = battle.available_moves[0].id
    n = {"calls": 0}

    def responder(messages, tools, idx):
        n["calls"] += 1
        if n["calls"] == 1:
            return LLMResponse(text="", tool_calls=[
                ToolCall(name="pokedex_lookup", arguments={"kind": "species", "query": "Garganacl"}, id="c1")
            ])
        return LLMResponse(text=f"Action: {target}")

    agent = ReActAgent(_ctx(responder), model="mock", supports_tools=True, max_steps=4)
    order = await agent.act(battle)
    assert order.message in {o.message for o in battle.valid_orders}
    assert agent.n_fallbacks == 0
    assert n["calls"] == 2  # tool turn, then action turn


async def test_react_degrades_without_tools(battle):
    target = battle.available_moves[1].id
    # no augmentations -> behaves like single-shot
    ctx = AgentContext(llm=MockLLMClient(lambda m, t, i: f"Action: {target}"))
    agent = ReActAgent(ctx, model="mock", supports_tools=False)
    order = await agent.act(battle)
    assert order.message in {o.message for o in battle.valid_orders}
