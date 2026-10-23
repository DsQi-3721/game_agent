"""Results / leaderboard screen: in-terminal charts + open Plotly HTML / W&B."""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, Static

from arena.viz.tui_charts import bar_chart, heatmap


class ResultsScreen(Screen):
    BINDINGS = [
        ("c", "open_charts", "Charts"),
        ("w", "open_wandb", "W&B"),
        ("escape", "app.pop_screen", "Back"),
    ]
    CSS = """
    ResultsScreen { background: $background; }
    #board { height: 9; border: round $primary; border-title-color: $accent; }
    #charts { height: 1fr; }
    #left { width: 1fr; border: round $secondary; border-title-color: $accent; padding: 0 1; }
    #right { width: 1fr; border: round $secondary; border-title-color: $accent; padding: 0 1; }
    """

    def __init__(self, run_dir: Path | str):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.report: dict = {}

    def compose(self) -> ComposeResult:
        yield Header()
        t = DataTable(id="board", cursor_type="row", zebra_stripes=True)
        t.border_title = "Leaderboard"
        t.add_columns("Agent", "Win%", "95% CI", "Elo", "Fallback%", "Tools", "$")
        yield t
        with Horizontal(id="charts"):
            with VerticalScroll(id="left") as left:
                left.border_title = "Win-rate matrix"
                yield Static(id="heatmap")
            with VerticalScroll(id="right") as right:
                right.border_title = "Elo"
                yield Static(id="bars")
        yield Footer()

    def on_mount(self) -> None:
        path = self.run_dir / "report.json"
        if not path.exists():
            self.query_one("#heatmap", Static).update("(no report.json)")
            return
        self.report = json.loads(path.read_text(encoding="utf-8"))
        self._fill()

    def _fill(self) -> None:
        ag = self.report.get("agents", {})
        board = self.query_one("#board", DataTable)
        for n in sorted(ag, key=lambda k: -ag[k].get("elo", 1000)):
            m = ag[n]
            ci = f"[{m['wilson_ci'][0]:.2f},{m['wilson_ci'][1]:.2f}]"
            board.add_row(n, f"{m['win_rate']:.0%}", ci, f"{m['elo']:.0f}",
                          f"{m.get('fallback_rate', 0):.0%}", str(m.get("tool_calls", 0)),
                          f"{m.get('cost_usd', 0):.3f}")
        self.query_one("#heatmap", Static).update(heatmap(self.report))
        self.query_one("#bars", Static).update(bar_chart(self.report, "elo", "Elo"))

    def action_open_charts(self) -> None:
        p = self.run_dir / "charts.html"
        if p.exists():
            webbrowser.open(p.resolve().as_uri())
        else:
            self.app.bell()

    def action_open_wandb(self) -> None:
        webbrowser.open("https://wandb.ai/home")
