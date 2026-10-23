"""History + trajectory viewer: browse past runs and per-turn agent decisions."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Label, RichLog, Select

from arena.tui.history import list_runs, load_turns


class TrajectoryScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back")]
    CSS = """
    TrajectoryScreen { background: $background; }
    #runs { height: 9; border: round $primary; border-title-color: $accent; }
    #picker { height: 3; }
    #agent { width: 32; }
    #search { width: 1fr; }
    #turns { height: 1fr; border: round $surface-lighten-2; border-title-color: $accent; }
    """

    def __init__(self, runs_dir: str = "runs"):
        super().__init__()
        self.runs_dir = runs_dir
        self.runs: list[dict] = []
        self.cur: dict | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        rt = DataTable(id="runs", cursor_type="row", zebra_stripes=True)
        rt.border_title = "History"
        rt.add_columns("Run", "Format", "Agents", "Top (Elo)")
        yield rt
        with Horizontal(id="picker"):
            yield Select([], id="agent", prompt="Select agent")
            yield Input(placeholder="Search... (enter)", id="search")
        turns = RichLog(id="turns", highlight=True, markup=True, wrap=True, max_lines=6000)
        turns.border_title = "Trajectory"
        yield turns
        yield Footer()

    def on_mount(self) -> None:
        self.runs = list_runs(self.runs_dir)
        rt = self.query_one("#runs", DataTable)
        for r in self.runs:
            rt.add_row(r["name"], r["format"], str(r["n_agents"]), r["top"])
        if self.runs:
            self._select_run(0)

    def _select_run(self, idx: int) -> None:
        if not (0 <= idx < len(self.runs)):
            return
        self.cur = self.runs[idx]
        agents = self.cur["agents"]
        self.query_one("#agent", Select).set_options([(a, a) for a in agents])
        self.query_one("#turns", RichLog).clear()
        if agents:
            self.query_one("#agent", Select).value = agents[0]

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "runs":
            self._select_run(event.cursor_row)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "agent" and event.value is not Select.BLANK:
            self._render_turns(str(event.value))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search":
            sel = self.query_one("#agent", Select).value
            if sel is not Select.BLANK:
                self._render_turns(str(sel), event.value)

    def _render_turns(self, agent: str, query: str = "") -> None:
        if not self.cur:
            return
        log = self.query_one("#turns", RichLog)
        log.clear()
        path = f"{self.cur['path']}/traj_{agent}.jsonl"
        turns = load_turns(path, query)
        log.write(f"[b]{agent}[/]  .  {len(turns)} turns" + (f"  (filter: {query})" if query else ""))
        for rec in turns:
            fb = "  [#d13b27][fallback][/]" if rec.get("fallback") else ""
            tools = f"  tools x{rec.get('n_tool_calls', 0)}" if rec.get("n_tool_calls") else ""
            log.write(f"\n[#3c6fd6]-- {rec.get('battle_tag')} . T{rec.get('turn')}[/]"
                      f" -> [b]{rec.get('parsed_order')}[/]{fb}{tools}")
            out = (rec.get("output") or "").strip()
            if out:
                log.write(out[:600])
