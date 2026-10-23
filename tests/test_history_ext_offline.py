"""UI-M5 offline: history scan, trajectory load, screen, extensibility (auto-register)."""

import json
from pathlib import Path

from arena.tui.history import list_runs, load_turns


def _make_run(root: Path, name: str):
    d = root / name
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps({
        "format": "gen9ou",
        "agents": {"a": {"elo": 1020}, "b": {"elo": 980}},
    }), encoding="utf-8")
    (d / "traj_a.jsonl").write_text(
        json.dumps({"agent": "a", "battle_tag": "battle-x-1", "turn": 1,
                    "parsed_order": "/choose move 1", "output": "use earthquake",
                    "fallback": False, "n_tool_calls": 2}) + "\n"
        + json.dumps({"agent": "a", "battle_tag": "battle-x-1", "turn": 2,
                      "parsed_order": "/switch 2", "output": "pivot out",
                      "fallback": True, "n_tool_calls": 0}) + "\n",
        encoding="utf-8")
    return d


def test_list_runs(tmp_path):
    _make_run(tmp_path, "exp1")
    (tmp_path / "_ui").mkdir()  # underscore dirs ignored
    runs = list_runs(tmp_path)
    assert len(runs) == 1
    r = runs[0]
    assert r["name"] == "exp1" and r["format"] == "gen9ou"
    assert r["agents"] == ["a"] and r["top"] == "a"


def test_load_turns_and_search(tmp_path):
    d = _make_run(tmp_path, "exp1")
    turns = load_turns(d / "traj_a.jsonl")
    assert len(turns) == 2
    filtered = load_turns(d / "traj_a.jsonl", "earthquake")
    assert len(filtered) == 1 and filtered[0]["turn"] == 1


async def test_trajectory_screen(tmp_path):
    from textual.app import App
    from textual.widgets import DataTable

    from arena.tui.screens.trajectory_screen import TrajectoryScreen

    _make_run(tmp_path, "exp1")

    class _Host(App):
        def on_mount(self):
            self.push_screen(TrajectoryScreen(str(tmp_path)))

    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen.query_one("#runs", DataTable).row_count == 1


def test_extensibility_new_agent_autoregisters():
    """A newly decorated agent appears in the registry without touching the UI."""
    from arena.agents.base import Agent
    from arena.config.schema import AgentConfig
    from arena.registry.agents import build_spec, entries, register_agent
    from arena.registry.param import Param

    @register_agent("llm_dummy_test", is_llm=True, needs_model=True,
                    params=[Param("k", int, 2)])
    class _Dummy(Agent):
        def __init__(self, ctx=None, *, model=None, k=2):
            super().__init__(ctx)
            self.model = model
            self.k = k

        async def act(self, battle):  # pragma: no cover - not run here
            return None

    e = entries()
    assert "llm_dummy_test" in e and e["llm_dummy_test"].params[0].name == "k"
    spec = build_spec(AgentConfig(name="d", type="llm_dummy_test", model="deepseek-chat",
                                  params={"k": 5}))
    from arena.agents.base import AgentContext
    agent = spec.build(AgentContext())
    assert agent.model == "deepseek-chat" and agent.k == 5
