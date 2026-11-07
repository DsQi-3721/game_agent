"""Live monitor: standings + current-battle detail + global decision stream.

Fed by the event stream (app tails events.jsonl). Three panels:
  · standings  — per-agent W-L / Win% / Elo / $ (from METRIC + MATCH_FINISHED)
  · battle     — current pairing + latest turn detail (from PAIRING + TURN)
  · log        — scrolling decision/result stream
'o' opens the current battle's Showdown spectate page in a browser.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, ProgressBar, RichLog, Static

from arena.events.schema import Event, EventType

# NES palette (Rich renderables don't read $tokens, so hardcode)
C_GREEN = "#5ab552"
C_YELLOW = "#f8c537"
C_RED = "#d13b27"
C_BLUE = "#3c6fd6"
C_CREAM = "#fcf7e8"


class MonitorScreen(Screen):
    BINDINGS = [
        ("o", "open_battle", "Open battle"),
        ("escape", "app.pop_screen", "Back"),
        ("r", "results", "Results"),
    ]
    CSS = """
    MonitorScreen { background: $background; }
    #top { height: 11; }
    #left { width: 1fr; border: round $primary; padding: 0 1; }
    #right { width: 1fr; border: round $secondary; padding: 0 1; }
    #left { border-title-color: $accent; }
    #right { border-title-color: $accent; }
    #standings { height: 1fr; }
    #prog { height: 1; margin: 1 0; }
    #log { height: 1fr; border: round $surface-lighten-2; }
    """

    def __init__(self, name: str, run_dir: Path):
        super().__init__()
        self.exp = name
        self.run_dir = Path(run_dir)
        self.total = 1
        self.done = 0
        self.cur_tag: str | None = None
        self.totals: dict[str, dict] = {}
        self.elo: dict[str, float] = {}
        self.winrate: dict[str, float] = {}
        self.cost: dict[str, float] = {}
        self._pairing_line = ""
        self._cols: list = []
        self._rows: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top"):
            with Vertical(id="left") as left:
                left.border_title = "Standings"
                t = DataTable(id="standings", cursor_type="none", zebra_stripes=True)
                self._cols = t.add_columns("Agent", "W-L", "Win%", "Elo", "$")
                yield t
            with Vertical(id="right") as right:
                right.border_title = "Battle"
                yield Static("Waiting to start...", id="battle")
        yield ProgressBar(id="prog", total=100, show_eta=False)
        yield RichLog(id="log", highlight=True, markup=True, max_lines=4000, wrap=True)
        yield Footer()

    # --- event sink ----------------------------------------------------
    def feed(self, ev: Event) -> None:
        log = self.query_one("#log", RichLog)
        d = ev.data
        if ev.type is EventType.RUN_STARTED:
            self.total = d.get("total_battles", 0) or 1
            self.query_one("#prog", ProgressBar).update(total=self.total)
            for a in d.get("agents", []):
                self.totals[a] = {"wins": 0, "games": 0}
            self._refresh_standings()
            log.write(f"[{C_GREEN}]> RUN[/] {d.get('format')} . {d.get('mode')} . "
                      f"{d.get('n_pairings')} pairings x {d.get('n_battles')} battles")
        elif ev.type is EventType.PAIRING_STARTED:
            self._battle(f"[b]{d['a']}[/]  vs  [b]{d['b']}[/]", "-- waiting for turn 1 --")
            log.write(f"[{C_BLUE}]> {d['a']} vs {d['b']}[/]")
        elif ev.type is EventType.TURN:
            self.cur_tag = d.get("battle_tag") or self.cur_tag
            fb = f"  [{C_RED}][fallback][/]" if d.get("fallback") else ""
            tools = f"  tools x{d['n_tool_calls']}" if d.get("n_tool_calls") else ""
            self._battle(None, f"T{d.get('turn')}  [b]{d.get('agent')}[/] -> "
                               f"{d.get('parsed_order')}{fb}{tools}")
            log.write(f"  T{d.get('turn'):<3} {d.get('agent'):<18} -> {d.get('parsed_order')}{fb}{tools}")
        elif ev.type is EventType.MATCH_FINISHED:
            self.done += d["a_wins"] + d["b_wins"] + d["ties"]
            self.totals = d.get("totals", self.totals)
            self.query_one("#prog", ProgressBar).update(progress=self.done)
            self._refresh_standings()
            log.write(f"[{C_GREEN}][OK][/] {d['a']} [b]{d['a_wins']}-{d['b_wins']}[/] {d['b']}  "
                      f"(avg {d['avg_turns']}t)")
        elif ev.type is EventType.METRIC:
            self.elo = d.get("elo", {})
            self.winrate = d.get("win_rate", {})
            self.cost = d.get("cost_usd", {})
            self._refresh_standings()
        elif ev.type is EventType.RUN_FINISHED:
            log.write(f"[{C_GREEN}][DONE][/] report -> {d.get('report_path')}")
            self._battle("Done", "r = results . esc = back")
        elif ev.type is EventType.ERROR:
            log.write(f"[{C_RED}][ERROR][/] {d.get('message')}")

    def _battle(self, pairing: str | None, turn: str | None) -> None:
        w = self.query_one("#battle", Static)
        if pairing is not None:
            self._pairing_line = pairing
        tag = f"\n[{C_CREAM}]{self.cur_tag}[/]" if self.cur_tag else ""
        w.update(f"{self._pairing_line}{tag}\n\n{turn or ''}")

    def _refresh_standings(self) -> None:
        # In-place keyed updates (no clear/rebuild) so the table never flickers.
        t = self.query_one("#standings", DataTable)
        for n in self.totals:
            tot = self.totals.get(n, {})
            wl = f"{tot.get('wins', 0)}-{tot.get('games', 0) - tot.get('wins', 0)}"
            wr = f"{self.winrate.get(n, 0) * 100:.0f}%" if n in self.winrate else "-"
            elo = f"{self.elo.get(n, 1000):.0f}" if n in self.elo else "-"
            cost = f"{self.cost.get(n, 0):.3f}" if self.cost.get(n) else "-"
            if n not in self._rows:
                t.add_row(n, wl, wr, elo, cost, key=n)
                self._rows.add(n)
            else:
                for col, val in zip(self._cols[1:], (wl, wr, elo, cost)):
                    t.update_cell(n, col, val)

    def action_open_battle(self) -> None:
        if self.cur_tag:
            from arena import showdown
            webbrowser.open(f"{showdown.http_url()}/{self.cur_tag}")

    def action_results(self) -> None:
        from arena.tui.screens.results_screen import ResultsScreen
        self.app.push_screen(ResultsScreen(self.run_dir))
