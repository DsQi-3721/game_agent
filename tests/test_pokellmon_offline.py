"""Offline: PokeLLMon native agent -- consistency vote, knowledge injection,
anti-panic-switch, fallback. Mock LLM, no key/network."""

import pickle
from pathlib import Path

import pytest
from poke_env.player.battle_order import BattleOrder

from arena.agents.base import AgentContext
from arena.agents.llm.pokellmon import PokeLLMonAgent, _TurnMemory
from arena.llm.client import MockLLMClient

FIXTURE = Path(__file__).parent / "fixtures" / "battles" / "gen9random_midbattle.pkl"


@pytest.fixture(scope="module")
def battle():
    with open(FIXTURE, "rb") as f:
        return pickle.load(f)


async def test_pokellmon_majority_vote_legal(battle):
    target = battle.available_moves[0].id
    # all 3 samples agree -> that move wins the vote, no fallback
    mock = MockLLMClient(lambda messages, tools, idx: f"Reasoning...\nAction: {target}")
    agent = PokeLLMonAgent(AgentContext(llm=mock), model="mock-model", n_samples=3)
    order = await agent.act(battle)
    assert isinstance(order, BattleOrder)
    assert order.message in {o.message for o in battle.valid_orders}
    assert agent.n_fallbacks == 0
    assert agent.n_decisions == 1


async def test_pokellmon_fallback_on_garbage(battle):
    mock = MockLLMClient(lambda messages, tools, idx: "no idea what to do")
    agent = PokeLLMonAgent(AgentContext(llm=mock), model="mock-model", n_samples=2)
    order = await agent.act(battle)
    assert order.message in {o.message for o in battle.valid_orders}
    assert agent.n_fallbacks == 1


async def test_pokellmon_injects_knowledge(battle):
    """render_state must contain at least one forced knowledge block."""
    mock = MockLLMClient(lambda messages, tools, idx: "Action: noop")
    agent = PokeLLMonAgent(AgentContext(llm=mock), model="mock-model")
    state = agent.render_state(battle)
    # the knowledge registry forces these passive; at least one should render
    assert any(f"[{k}]" in state for k in ("type_matchup", "damage_calc", "base_stats",
                                           "ability", "pokedex"))


def test_pokellmon_anti_panic_switch(battle):
    """If we switched last turn and the vote is tied switch/move, prefer the move."""
    agent = PokeLLMonAgent(AgentContext(llm=MockLLMClient(lambda *a: "")), model="m")
    move = battle.available_moves[0]
    switch = battle.available_switches[0] if battle.available_switches else None
    if switch is None:
        pytest.skip("fixture has no available switch this turn")
    from poke_env.player.player import Player
    move_order = Player.create_order(move)
    switch_order = Player.create_order(switch)
    # pretend we switched last turn
    agent._mem[battle.battle_tag] = _TurnMemory(action_msg=switch_order.message)
    # vote: one switch, one move (tie) -> anti-panic prefers the move
    chosen = agent._vote([switch_order, move_order], battle)
    assert "switch" not in chosen.message


def test_pokellmon_feedback_after_switch_is_quiet(battle):
    """No phantom damage feedback when last action was a switch."""
    agent = PokeLLMonAgent(AgentContext(llm=MockLLMClient(lambda *a: "")), model="m")
    opp = battle.opponent_active_pokemon
    agent._mem[battle.battle_tag] = _TurnMemory(
        opp_species=(opp.species if opp else None), opp_hp=1.0,
        action_msg="/choose switch foo",
    )
    fb = agent._feedback(battle)
    assert "did little or no damage" not in fb
