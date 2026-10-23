"""M4 offline: augmentations, passive injection, CoT & self-consistency (mock)."""

import pickle
from pathlib import Path

import pytest

from arena.agents.base import AgentContext
from arena.agents.llm.cot import CoTAgent
from arena.agents.llm.self_consistency import SelfConsistencyAgent
from arena.augment.base import AugmentationRegistry, AugmentMode
from arena.augment.damage import DamageAugmentation
from arena.augment.pokedex import PokedexTool
from arena.augment.stats import StatsAugmentation
from arena.augment.type_matchup import TypeMatchupAugmentation
from arena.llm.client import MockLLMClient
from arena.state.renderer import DefaultRenderer

FIXTURE = Path(__file__).parent / "fixtures" / "battles" / "gen9random_midbattle.pkl"


@pytest.fixture(scope="module")
def battle():
    with open(FIXTURE, "rb") as f:
        return pickle.load(f)


def test_individual_augmentations(battle):
    assert TypeMatchupAugmentation().compute(battle).text
    assert StatsAugmentation().compute(battle).text
    # damage may or may not produce rows depending on opp stat knowledge; must not crash
    DamageAugmentation().compute(battle)


def test_registry_passive_and_tool(battle):
    reg = AugmentationRegistry.from_config({
        "type_matchup": "passive", "base_stats": "passive",
        "damage_calc": "passive", "ability": "passive", "pokedex": "tool",
    })
    passive = reg.passive(battle)
    names = {r.name for r in passive}
    assert "type_matchup" in names and "base_stats" in names
    tools = reg.tools()
    assert any(t.name == "pokedex_lookup" for t in tools)


def test_passive_injected_into_render(battle):
    reg = AugmentationRegistry.from_config({"type_matchup": "passive"})
    txt = DefaultRenderer().render(battle, passive_augments=reg.passive(battle))
    assert "[type_matchup]" in txt


async def test_pokedex_tool(battle):
    out = await PokedexTool().call(battle, {"kind": "species", "query": "Garchomp"})
    assert "Garchomp" in out and "base" in out.lower()


async def test_cot_agent(battle):
    target = battle.available_moves[0].id
    mock = MockLLMClient(lambda m, t, i: f"Step 1... Action: {target}")
    agent = CoTAgent(AgentContext(llm=mock), model="mock")
    order = await agent.act(battle)
    assert order.message in {o.message for o in battle.valid_orders}


async def test_self_consistency_majority(battle):
    mv0 = battle.available_moves[0].id
    mv1 = battle.available_moves[1].id
    # 3 votes for mv0, 2 for mv1 -> mv0 wins
    seq = [mv0, mv0, mv0, mv1, mv1]
    mock = MockLLMClient(lambda m, t, i: f"Action: {seq[i]}")
    agent = SelfConsistencyAgent(AgentContext(llm=mock), model="mock", n_samples=5)
    order = await agent.act(battle)
    expected = {o.message for o in battle.valid_orders if mv0 in o.message}
    assert order.message in expected
