"""M2 offline tests: renderer output + team pool integrity (no server)."""

import pickle
from pathlib import Path

import pytest
from poke_env.teambuilder.teambuilder import Teambuilder

from arena.state.renderer import DefaultRenderer
from arena.teams.pool import TeamPool

FIXTURE = Path(__file__).parent / "fixtures" / "battles" / "gen9random_midbattle.pkl"


@pytest.fixture(scope="module")
def battle():
    with open(FIXTURE, "rb") as f:
        return pickle.load(f)


def test_renderer_partial_obs(battle):
    txt = DefaultRenderer().render(battle)
    assert "Your active:" in txt
    assert "Opponent active:" in txt
    assert "Moves:" in txt
    # partial observability: no leak of an opponent full roster line by default
    assert "[full-info]" not in txt
    # enum names render cleanly, not python repr
    assert "(pokemon type) object" not in txt


def test_pool_excludes_rejected_and_parses():
    pool = TeamPool()
    tiers = pool.tiers()
    assert "gen9ou" in tiers, f"gen9ou missing from pool: {tiers}"
    for tier in tiers:
        for paste in pool.get(tier):
            # every pooled team must be parseable to packed (rejected/.reason files excluded)
            packed = Teambuilder.join_team(Teambuilder.parse_showdown_team(paste))
            assert packed and "|" in packed
