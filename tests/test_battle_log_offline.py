"""Battle-log distillation (offline): protocol events -> perspective-aware turn digests."""

import copy
import pickle
from pathlib import Path

import pytest

from arena.state.battle_log import LogDigester, events_since

FIXTURE = Path(__file__).parent / "fixtures" / "battles" / "gen9random_midbattle.pkl"

TURN1 = [
    ["", "move", "p2a: Garganacl", "Salt Cure", "p1a: Passimian"],
    ["", "-damage", "p1a: Passimian", "215/302"],
    ["", "-start", "p1a: Passimian", "Salt Cure"],
    ["", "move", "p1a: Passimian", "Close Combat", "p2a: Garganacl"],
    ["", "-supereffective", "p2a: Garganacl"],
    ["", "-damage", "p2a: Garganacl", "35/100"],
    ["", "-unboost", "p1a: Passimian", "def", "1"],
    ["", "-damage", "p1a: Passimian", "177/302", "[from] Salt Cure"],
    ["", "-heal", "p2a: Garganacl", "41/100", "[from] item: Leftovers"],
    ["", ""], ["", "upkeep"], ["", "t:", "1700000000"],
    ["", "turn", "2"],
]

TURN2 = [
    ["", "-terastallize", "p2a: Garganacl", "Water"],
    ["", "move", "p1a: Passimian", "Close Combat", "p2a: Garganacl"],
    ["", "-crit", "p2a: Garganacl"],
    ["", "-damage", "p2a: Garganacl", "0 fnt"],
    ["", "faint", "p2a: Garganacl"],
    ["", "-sidestart", "p1: AgentPlayer 1", "move: Stealth Rock"],
    ["", "upkeep"],
    ["", "switch", "p2a: Kingambit", "Kingambit, L80, M", "100/100"],
    ["", "turn", "3"],
]


@pytest.fixture
def battle():
    with open(FIXTURE, "rb") as f:
        return copy.deepcopy(pickle.load(f))


def _primed(battle):
    """Digester that already consumed the fixture's start-of-battle events."""
    dg = LogDigester()
    events, cursor = events_since(battle, 0)
    results, _ = dg.ingest(events, battle)
    return dg, cursor, results


def test_start_of_battle_digest(battle):
    _, cursor, results = _primed(battle)
    assert cursor == len(battle._replay_data)
    text = "\n".join(r.render() for r in results)
    assert "your sent out Passimian (100%)" in text and "opp sent out Garganacl (100%)" in text
    assert "rule" not in text and "AgentPlayer" not in text  # protocol noise dropped


def test_turn_digest_merges_effects_and_tracks_hp(battle):
    dg, cursor, _ = _primed(battle)
    battle._replay_data.extend(TURN1)
    events, cursor2 = events_since(battle, cursor)
    assert len(events) == len(TURN1)
    results, facts = dg.ingest(events, battle)
    assert [r.turn for r in results] == [1]  # events before "|turn|2" belong to turn 1
    lines = results[0].lines
    assert lines[0] == ("opp Garganacl used Salt Cure on your Passimian; "
                        "your Passimian 100%->71%; Salt Cure on your Passimian")
    assert lines[1].startswith("your Passimian used Close Combat on opp Garganacl; super effective; "
                               "opp Garganacl 100%->35%; your Passimian def -1")
    assert lines[2] == "your Passimian took damage 71%->59% (Salt Cure)"
    assert lines[3] == "opp Garganacl healed 35%->41% (Leftovers)"
    kinds = {(f.kind, f.text) for f in facts}
    assert ("opp_move", "opp Garganacl has Salt Cure") in kinds
    assert ("opp_item", "opp Garganacl holds Leftovers") in kinds
    # re-requested decision: nothing new
    again, cursor3 = events_since(battle, cursor2)
    assert again == [] and cursor3 == cursor2


def test_faint_tera_hazard_facts_dedupe(battle):
    dg, cursor, _ = _primed(battle)
    battle._replay_data.extend(TURN1)
    events, cursor = events_since(battle, cursor)
    dg.ingest(events, battle)
    battle._replay_data.extend(TURN2)
    events, _ = events_since(battle, cursor)
    results, facts = dg.ingest(events, battle)
    text = "\n".join(results[0].lines)
    assert "opp Garganacl terastallized into Water" in text
    assert "critical hit" in text and "opp Garganacl fainted" in text
    assert "your side: Stealth Rock set" in text
    assert "opp sent out Kingambit (100%)" in text
    kinds = [f.kind for f in facts]
    assert {"tera", "faint", "hazard"} <= set(kinds)
    assert next(f for f in facts if f.kind == "faint").text == "opp Garganacl fainted (turn 2)"
    # the same events again produce no duplicate facts
    _, facts_again = dg.ingest(events, battle)
    assert not [f for f in facts_again if f.kind in ("tera", "faint")]
