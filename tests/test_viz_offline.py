"""UI-M4 offline: Plotly HTML, terminal charts, sinks, results screen."""

import json
from pathlib import Path

from rich.console import Console

from arena.viz.charts_html import build
from arena.viz.sinks import PlotlySink, WandbSink
from arena.viz.tui_charts import bar_chart, heatmap

REPORT = {
    "experiment": "t", "format": "gen9randombattle", "mode": "round_robin", "n_battles": 4,
    "agents": {
        "a": {"win_rate": 0.75, "wilson_ci": [0.4, 0.95], "wins": 3, "games": 4, "elo": 1030.0,
              "fallback_rate": 0.0, "decisions": 80, "tool_calls": 5, "tokens": 1000, "cost_usd": 0.1},
        "b": {"win_rate": 0.25, "wilson_ci": [0.05, 0.6], "wins": 1, "games": 4, "elo": 970.0,
              "fallback_rate": 0.02, "decisions": 80, "tool_calls": 0, "tokens": 0, "cost_usd": 0.0},
    },
    "matrix": {"a": {"b": {"wins": 3, "losses": 1, "n": 4}},
               "b": {"a": {"wins": 1, "losses": 3, "n": 4}}},
    "total_cost_usd": 0.1, "total_tokens": 1000, "aborted": None,
}


def test_charts_html(tmp_path):
    out = build(REPORT, tmp_path / "charts.html")
    html = out.read_text()
    assert out.exists() and "Plotly" in html and "Win-rate matrix" in html


def test_tui_charts_render():
    c = Console(width=80, record=True)
    c.print(heatmap(REPORT))
    c.print(bar_chart(REPORT, "elo", "Elo"))
    text = c.export_text()
    assert "a" in text and "b" in text and "█" in text


def test_plotly_sink(tmp_path):
    p = PlotlySink(tmp_path / "charts.html").publish(REPORT, [])
    assert p is not None and Path(p).exists()


def test_wandb_sink_graceful(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert WandbSink("grp").publish(REPORT, []) is None  # no exception, no-op


async def test_results_screen(tmp_path):
    from textual.app import App
    from textual.widgets import DataTable

    from arena.tui.screens.results_screen import ResultsScreen

    (tmp_path / "report.json").write_text(json.dumps(REPORT), encoding="utf-8")

    class _Host(App):
        def on_mount(self):
            self.push_screen(ResultsScreen(tmp_path))

    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen.query_one("#board", DataTable).row_count == 2
