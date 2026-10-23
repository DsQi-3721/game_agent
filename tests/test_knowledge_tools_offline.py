"""Knowledge layer + game tools (offline, tiny fixture knowledge base, no node)."""

import copy
import pickle
from pathlib import Path

import pytest
from poke_env.battle.side_condition import SideCondition
from poke_env.data import GenData
from poke_env.stats import compute_raw_stats

from arena.augment.base import AugmentationRegistry
from arena.augment.calc_util import damage_estimate, known_or_likely_moves, resolve_mon
from arena.augment.damage import DamageAugmentation
from arena.augment.switch_eval import hazard_text
from arena.knowledge import store
from arena.knowledge.opponent_model import DEFAULT_MODEL
from arena.knowledge.priors import condition, raw_prior
from arena.registry.augments import entries

FIXTURE = Path(__file__).parent / "fixtures" / "battles" / "gen9random_midbattle.pkl"
KNOWLEDGE = Path(__file__).parent / "fixtures" / "knowledge_mini"
TOOLS = ["damage_calc", "type_matchup", "pokedex", "matchup", "speed_check",
         "opponent_sets", "switch_eval", "battle_log"]


@pytest.fixture(autouse=True)
def mini_knowledge(monkeypatch):
    monkeypatch.setenv("ARENA_KNOWLEDGE_DIR", str(KNOWLEDGE))
    store.clear_cache()
    yield
    store.clear_cache()


@pytest.fixture
def battle():
    with open(FIXTURE, "rb") as f:
        return copy.deepcopy(pickle.load(f))


def test_randbats_estimate_matches_formula(battle):
    opp = battle.opponent_active_pokemon
    est = DEFAULT_MODEL.estimate(battle, opp)
    expect = compute_raw_stats("garganacl", [85] * 6, [31] * 6, 80, "hardy", GenData.from_gen(9))
    assert est.exact and [est.stats[k] for k in ("hp", "atk", "def", "spa", "spd", "spe")] == expect
    own = DEFAULT_MODEL.estimate(battle, battle.active_pokemon)
    assert own.exact and own.stats["hp"] == 302


def test_prior_conditioning(battle):
    prior = raw_prior("gen9randombattle", "garganacl")
    assert prior.level == 80 and prior.top("items").id == "leftovers"
    cond = condition(prior, moves=["bodypress"], item="Leftovers")
    assert cond.likely_moves(4)[0].id == "bodypress" and len(cond.likely_moves(4)) == 4
    assert cond.items[0].p == 1.0 and cond.revealed["item"] == "leftovers"
    assert raw_prior("gen9ou", "greattusk").source == "gen9ou"


def test_damage_both_directions_restores_hidden_state(battle):
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    cc = next(m for m in me.moves.values() if m.id == "closecombat")
    out = damage_estimate(battle, me, opp, cc)
    assert out is not None and 60 < out.pct_lo < out.pct_hi < 100 and out.ko_lo == 2
    back = [damage_estimate(battle, opp, me, m) for m in known_or_likely_moves(battle, opp)]
    assert any(d is not None and d.pct_hi > 0 for d in back)
    assert all(v is None for v in opp.stats.values())       # estimates never leak
    assert opp.item == "unknown_item" and opp.ability is None


def test_damage_passive_is_no_longer_empty(battle):
    text = DamageAugmentation().compute(battle).text
    assert "closecombat -> garganacl" in text and "Opponent's known/likely moves on you" in text


async def test_every_tool_answers(battle):
    reg = AugmentationRegistry.from_config({n: "tool" for n in TOOLS})
    tools = {t.name: t for t in reg.tools()}
    assert set(tools) == {"damage_calc", "type_matchup", "pokedex_lookup", "matchup", "speed_check",
                          "opponent_sets", "switch_eval", "battle_log"}
    args = {"damage_calc": {"attacker": "garganacl", "defender": "passimian"},
            "type_matchup": {"attack": "Water"}, "pokedex_lookup": {"query": "Purifying Salt"},
            "matchup": {}, "speed_check": {}, "opponent_sets": {"pokemon": "kingambit"},
            "switch_eval": {}, "battle_log": {"keyword": "switch"}}
    for name, tool in tools.items():
        out = await tool.call(battle, args[name])
        assert out and "error" not in out.lower(), (name, out)
    assert "can't be statused" in await tools["pokedex_lookup"].call(battle, {"query": "Purifying Salt"})
    bad = await tools["damage_calc"].call(battle, {"attacker": "pikachu"})
    assert "Valid names" in bad and "garganacl" in bad
    assert "you move first" in await tools["speed_check"].call(battle, {})


def test_resolve_mon_fuzzy(battle):
    assert resolve_mon(battle, "Garganal").species == "garganacl"
    assert resolve_mon(battle, "opp").species == "garganacl"
    assert resolve_mon(battle, "").species == "passimian"
    assert resolve_mon(battle, "vika").species == "vikavolt"
    assert resolve_mon(battle, "zzzz") is None


def test_hazard_text(battle):
    battle._side_conditions[SideCondition.STEALTH_ROCK] = 1
    battle._side_conditions[SideCondition.SPIKES] = 2
    dmg, text = hazard_text(battle, resolve_mon(battle, "mightyena"))  # Dark, grounded, no Boots
    assert text.startswith("Stealth Rock -12%") and "Spikes x2" in text and round(dmg, 1) == 29.2
    dmg, text = hazard_text(battle, resolve_mon(battle, "vikavolt"))   # holds Heavy-Duty Boots
    assert dmg == 0 and "Heavy-Duty Boots" in text


def test_tool_constructors_do_no_io(monkeypatch, battle):
    def boom(path):
        raise AssertionError("constructor touched the knowledge files")

    monkeypatch.setattr(store, "load", boom)
    reg = AugmentationRegistry([cls() for cls in entries().values()], {n: "tool" for n in TOOLS})
    assert len(reg.tools()) == len(TOOLS)


async def test_missing_knowledge_degrades(monkeypatch, tmp_path, battle):
    monkeypatch.setenv("ARENA_KNOWLEDGE_DIR", str(tmp_path))
    store.clear_cache()
    reg = AugmentationRegistry.from_config({"opponent_sets": "tool", "damage_calc": "tool"})
    tools = {t.name: t for t in reg.tools()}
    assert "No set data" in await tools["opponent_sets"].call(battle, {})
    assert "closecombat" in await tools["damage_calc"].call(battle, {})  # generic estimate still works


async def test_side_aware_resolution_for_mirror_matches(battle):
    from arena.augment.opponent_sets import OpponentSetsTool

    assert resolve_mon(battle, "passimian", side="theirs") is None        # only we have one
    assert resolve_mon(battle, "opp passimian") is None
    assert resolve_mon(battle, "my passimian").species == "passimian"
    assert resolve_mon(battle, "garganacl", side="mine") is None
    assert resolve_mon(battle, "opp active").species == "garganacl"
    out = await OpponentSetsTool().call(battle, {"pokemon": "Passimian"})
    assert "not seen in this battle" in out and "your own" not in out    # prior, never our own set
