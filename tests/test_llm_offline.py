"""M3 offline: full LLM decision chain with a mock client (no key/network)."""

import pickle
from pathlib import Path

import pytest
from poke_env.player.battle_order import BattleOrder

from arena.agents.base import AgentContext
from arena.agents.llm.io import IOAgent
from arena.llm.client import MockLLMClient
from arena.parse.action_parser import parse_action

FIXTURE = Path(__file__).parent / "fixtures" / "battles" / "gen9random_midbattle.pkl"


@pytest.fixture(scope="module")
def battle():
    with open(FIXTURE, "rb") as f:
        return pickle.load(f)


def test_parse_action_directive(battle):
    target = battle.available_moves[1].id
    order, note = parse_action(f"I'll go aggressive.\nAction: {target}", battle)
    assert order is not None
    assert order.message in {o.message for o in battle.valid_orders}
    assert note == "directive"


def test_parse_action_unparsed(battle):
    order, note = parse_action("the weather looks nice today", battle)
    assert order is None and note == "unparsed"


async def test_io_agent_legal_move(battle):
    target = battle.available_moves[0].id
    mock = MockLLMClient(lambda messages, tools, idx: f"Reasoning...\nAction: {target}")
    agent = IOAgent(AgentContext(llm=mock), model="mock-model")
    order = await agent.act(battle)
    assert isinstance(order, BattleOrder)
    assert order.message in {o.message for o in battle.valid_orders}
    assert agent.n_fallbacks == 0


async def test_io_agent_fallback_on_garbage(battle):
    mock = MockLLMClient(lambda messages, tools, idx: "no idea")
    agent = IOAgent(AgentContext(llm=mock), model="mock-model")
    order = await agent.act(battle)
    # still returns a legal order, but via fallback
    assert order.message in {o.message for o in battle.valid_orders}
    assert agent.n_fallbacks == 1
