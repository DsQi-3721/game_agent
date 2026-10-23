"""ArenaApp — POKEBOTMON terminal UI entry point.

Splash → Config → (launch eval subprocess) → Monitor. The eval runs as a separate
process; the app only tails runs/<name>/events.jsonl, so the UI stays smooth and
the eval survives the TUI closing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual import work
from textual.app import App

from arena.events.reader import EventReader
from arena.tui.config_model import Draft
from arena.tui.launcher import build_argv
from arena.tui.screens.config_screen import ConfigScreen
from arena.tui.screens.monitor_screen import MonitorScreen
from arena.tui.screens.splash_screen import SplashScreen
from arena.tui.theme import FAMICOM


class ArenaApp(App):
    TITLE = "POKEBOTMON"
    SUB_TITLE = "Pokemon Showdown Agent Arena"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.register_theme(FAMICOM)
        self.theme = "famicom"
        self.proc: asyncio.subprocess.Process | None = None
        self.monitor: MonitorScreen | None = None

    def on_mount(self) -> None:
        self.push_screen(ConfigScreen())
        self.push_screen(SplashScreen())   # fades, then dismisses to reveal Config

    # called by ConfigScreen's Start button
    def start_eval(self, draft: Draft) -> None:
        run_dir = Path("runs") / draft.name
        cfg_path = draft.write_yaml(run_dir)
        events = run_dir / "events.jsonl"
        try:
            events.unlink()          # fresh stream per launch
        except FileNotFoundError:
            pass
        self.monitor = MonitorScreen(draft.name, run_dir)
        self.push_screen(self.monitor)
        self._run_eval(cfg_path, run_dir)
        self._tail(events)

    @work(group="eval")
    async def _run_eval(self, cfg_path: Path, run_dir: Path) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            *build_argv(cfg_path, run_dir),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        _, err = await self.proc.communicate()
        if self.proc.returncode not in (0, None) and self.monitor is not None:
            self.monitor.query_one("#m_log").write(
                f"[red]subprocess exited {self.proc.returncode}[/]\n{(err or b'').decode()[-800:]}")

    @work(exclusive=True, group="tail")
    async def _tail(self, events: Path) -> None:
        async for ev in EventReader(events):
            if self.monitor is not None:
                self.monitor.feed(ev)


def main() -> None:
    ArenaApp().run()


if __name__ == "__main__":
    main()
