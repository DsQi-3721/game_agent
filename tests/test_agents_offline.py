"""M1: agents are decidable offline against a recorded Battle (no server)."""

import pickle
from pathlib import Path

import pytest
from poke_env.player.battle_order import BattleOrder

from arena.agents.baselines import BASELINES, make_baseline

FIXTURE = Path(__file__).parent / "fixtures" / "battles" / "gen9random_midbattle.pkl"


@pytest.fixture(scope="module")
def battle():
    assert FIXTURE.exists(), f"missing fixture {FIXTURE}; run scripts/record_fixture.py"
    with open(FIXTURE, "rb") as f:
        return pickle.load(f)


def test_fixture_is_a_decision_point(battle):
    # The snapshot was captured where a real choice exists.
    assert battle.available_moves, "fixture has no available moves"
    assert battle.valid_orders, "fixture exposes no valid orders"


@pytest.mark.parametrize("name", sorted(BASELINES))
async def test_baseline_returns_legal_order(battle, name):
    agent = make_baseline(name)
    order = await agent.act(battle)
    assert isinstance(order, BattleOrder), f"{name} returned {type(order)}"
    legal = {o.message for o in battle.valid_orders}
    assert order.message in legal, f"{name} chose illegal order {order.message!r}"
