"""Shared views + answer parsing (offline): numbered legal options, brief, state."""

import copy
import pickle
from pathlib import Path

import pytest

from arena.parse.action_parser import parse_action
from arena.parse.choice import parse_choice
from arena.state.views import entity_sheet, legal_options, option_for, render_brief, render_options, render_state

FIXTURE = Path(__file__).parent / "fixtures" / "battles" / "gen9random_midbattle.pkl"


@pytest.fixture(scope="module")
def battle():
    with open(FIXTURE, "rb") as f:
        return pickle.load(f)


def test_options_match_valid_orders(battle):
    opts = legal_options(battle)
    assert {o.order.message for o in opts} == {o.message for o in battle.valid_orders}
    assert [o.idx for o in opts] == list(range(1, len(opts) + 1))
    kinds = [o.kind for o in opts]
    assert kinds.index("switch") > max(i for i, k in enumerate(kinds) if k == "move")  # moves first
    tera = [o for o in opts if o.tera]
    assert tera and all("TERASTALLIZE (Fighting)" in o.label for o in tera)
    text = render_options(opts)
    assert text.splitlines()[0].startswith("1. move ")


def test_forced_switch_options(battle):
    b = copy.deepcopy(battle)
    b._force_switch = True
    b._available_moves = []
    opts = legal_options(b)
    assert opts and all(o.kind == "switch" for o in opts)
    assert "You must switch" in render_state(b)


def test_brief_and_state(battle):
    brief = render_brief(battle)
    for mon in battle.team.values():
        assert mon.species in brief
    assert "tera Fighting" in brief and "unknown" in brief  # randbats: no preview
    state = render_state(battle)
    assert "can terastallize (Fighting)" in state and "Opp active: garganacl" in state
    assert "Opp unrevealed: 5" in state
    assert "passimian" in entity_sheet(battle)


def test_parse_choice_variants(battle):
    opts = legal_options(battle)
    first_move = next(o for o in opts if o.kind == "move" and not o.tera)
    assert parse_choice("2", opts) is opts[1]
    assert parse_choice("Option 3. whatever", opts) is opts[2]
    sw = next(o for o in opts if o.kind == "switch")
    assert parse_choice(f"switch {sw.id.title()}", opts) is sw
    picked = parse_choice(f"{first_move.id} with tera", opts)
    assert picked.tera and picked.id == first_move.id
    assert parse_choice(f"{first_move.idx} tera", opts).tera
    assert parse_choice("Close Combat", opts).id == "closecombat"
    assert parse_choice("no idea", opts) is None
    assert option_for(opts, sw.order) is sw


def test_action_parser_tera_only_from_directive(battle):
    order, note = parse_action("Action: closecombat terastallize", battle)
    assert note == "directive" and order.message.endswith("terastallize")
    order, _ = parse_action("I will not tera yet.\nAction: closecombat", battle)
    assert not order.message.endswith("terastallize")
