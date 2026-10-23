"""UI-M2 offline: logo render, Draft->config round-trip, headless TUI smoke (Pilot)."""

import pytest

from arena.config.schema import AgentConfig, ExperimentConfig
from arena.tui.config_model import Draft
from arena.tui.widgets.logo import render_logo


def test_logo_renders():
    g = render_logo()
    assert g is not None  # rich Group; pyfiglet wordmark + pokeball


def test_draft_to_config_and_yaml_roundtrip(tmp_path):
    d = Draft(name="t", format="gen9ou", n_battles=6, eval_mode="anchored",
              anchors=["maxpow"], teams_mode="mirror",
              augmentations={"damage_calc": "passive", "type_matchup": "off"})
    d.agents = [
        AgentConfig(name="maxpow", type="max_base_power"),
        AgentConfig(name="react", type="llm_react", model="deepseek-chat", params={"max_steps": 3}),
    ]
    assert d.validate() is None
    cfg = d.to_config()
    assert cfg.format == "gen9ou" and cfg.evaluation.n_battles == 6
    # "off" augment is dropped
    assert cfg.augmentations == {"damage_calc": "passive"}
    path = d.write_yaml(tmp_path)
    reloaded = ExperimentConfig.load(path)
    assert {a.name for a in reloaded.agents} == {"maxpow", "react"}
    assert reloaded.evaluation.anchors == ["maxpow"]


def test_draft_validation():
    assert Draft().validate() == "Add at least one agent"
    d = Draft(eval_mode="anchored")
    d.agents = [AgentConfig(name="a", type="random")]
    assert "anchor" in d.validate()


async def test_tui_mounts_and_config_shows_registered_agents():
    from arena.tui.app import ArenaApp
    from arena.tui.screens.config_screen import ConfigScreen

    app = ArenaApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("space")        # skip splash
        await pilot.pause()
        assert isinstance(app.screen, ConfigScreen)
        # registry-driven: the config screen discovered all agent types
        assert set(app.screen._agents) >= {"random", "llm_react", "llm_io"}
        # the add/start controls exist
        assert app.screen.query_one("#add_agent") is not None
        assert app.screen.query_one("#start") is not None


async def test_monitor_feeds_events():
    from pathlib import Path

    from textual.app import App
    from textual.widgets import DataTable

    from arena.events.schema import Event, EventType
    from arena.tui.screens.monitor_screen import MonitorScreen

    class _Host(App):
        def on_mount(self):
            self.push_screen(MonitorScreen("x", Path("runs/x")))

    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        scr = app.screen
        ev = lambda t, **d: Event(t, ts=0.0, seq=0, run="x", data=d)
        scr.feed(ev(EventType.RUN_STARTED, agents=["a", "b"], total_battles=4,
                    format="gen9randombattle", mode="round_robin", n_pairings=1, n_battles=4))
        scr.feed(ev(EventType.PAIRING_STARTED, a="a", b="b", index=0))
        scr.feed(ev(EventType.TURN, agent="a", battle_tag="battle-gen9randombattle-1",
                    turn=3, parsed_order="/choose move 1", fallback=False, n_tool_calls=2))
        scr.feed(ev(EventType.MATCH_FINISHED, a="a", b="b", a_wins=3, b_wins=1, ties=0,
                    avg_turns=20.0, totals={"a": {"wins": 3, "games": 4}, "b": {"wins": 1, "games": 4}}))
        scr.feed(ev(EventType.METRIC, elo={"a": 1030.0, "b": 970.0},
                    win_rate={"a": 0.75, "b": 0.25}, cost_usd={}, completed_pairings=1, total_pairings=1))
        await pilot.pause()
        assert scr.query_one("#standings", DataTable).row_count == 2
        assert scr.cur_tag == "battle-gen9randombattle-1"
        assert scr.done == 4


async def test_tui_add_agent_flow():
    from textual.widgets import Input, Select

    from arena.tui.app import ArenaApp

    app = ArenaApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        scr = app.screen
        scr.query_one("#agent_type", Select).value = "max_base_power"
        scr._rebuild_params()
        scr.query_one("#agent_name", Input).value = "mp"
        await pilot.pause()
        await pilot.click("#add_agent")
        await pilot.pause()
        assert any(a.name == "mp" and a.type == "max_base_power" for a in scr.draft.agents)


async def test_config_handles_blank_select():
    """agent_type can fire a Changed(BLANK) on init; must not KeyError."""
    from textual.widgets import Select

    from arena.tui.app import ArenaApp

    app = ArenaApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        scr = app.screen
        scr.query_one("#agent_type", Select).set_options([])  # value -> BLANK
        assert scr._cur_type() in scr._agents                 # falls back to a real type
        scr._rebuild_params()                                 # must not raise (was KeyError)


async def test_tui_offers_hymem_with_param_form():
    """The HyMem agent and its params auto-appear in the configurator (registry-driven)."""
    from textual.widgets import Checkbox, Input, Select

    from arena.tui.app import ArenaApp

    app = ArenaApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        scr = app.screen
        assert {"llm_hymem", "llm_flat_react", "llm_summary"} <= set(scr._agents)
        scr.query_one("#agent_type", Select).value = "llm_hymem"
        scr._rebuild_params()
        await pilot.pause()
        assert scr.query_one("#param_planner_steps", Input).value == "4"
        assert scr.query_one("#param_think_iso", Checkbox).value is False
        ablation = scr.query_one("#param_ablation", Select)
        assert ablation.value == "full"
        model = scr.query_one("#param_model", Select)
        model.value = "qwen3.5-4b-mlx"
        scr.query_one("#agent_name", Input).value = "hymem"
        await pilot.pause()
        scr._add_agent()                      # the long form can push the button off-screen
        await pilot.pause()
        added = [a for a in scr.draft.agents if a.name == "hymem"]
        assert added and added[0].type == "llm_hymem" and added[0].model == "qwen3.5-4b-mlx"
        assert added[0].params["planner_steps"] == 4 and added[0].params["ablation"] == "full"

    from arena.agents.base import AgentContext
    from arena.config.schema import AgentConfig
    from arena.llm.client import MockLLMClient
    from arena.registry.agents import build_spec

    spec = build_spec(AgentConfig(name="hymem", type="llm_hymem", model="qwen3.5-4b-mlx", params=added[0].params))
    agent = spec.build(AgentContext(llm=MockLLMClient(lambda m, t, i: "")))
    assert agent.cfg.planner_steps == 4 and agent.cfg.ablation == "full"
