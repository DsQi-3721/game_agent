"""Startup splash: fade the POKEBOTMON logo in, hold, fade out, then continue."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Middle
from textual.screen import Screen
from textual.widgets import Static

from arena.tui.widgets.logo import render_logo


class SplashScreen(Screen):
    """Auto-dismisses after a smooth fade; any key skips immediately."""

    CSS = """
    SplashScreen { background: $surface; }
    #logo { width: auto; height: auto; }
    """

    def __init__(self, hold: float = 0.8, fade: float = 0.45):
        super().__init__()
        self._hold = hold
        self._fade = fade
        self._done = False

    def compose(self) -> ComposeResult:
        with Middle():
            with Center():
                yield Static(render_logo(), id="logo")

    def on_mount(self) -> None:
        logo = self.query_one("#logo")
        logo.styles.opacity = 0.0
        logo.styles.animate("opacity", 1.0, duration=self._fade)
        self.set_timer(self._fade + self._hold, self._fade_out)

    def _fade_out(self) -> None:
        if self._done:
            return
        logo = self.query_one("#logo")
        logo.styles.animate("opacity", 0.0, duration=self._fade,
                            on_complete=self._finish)

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        self.dismiss()

    def on_key(self, event) -> None:
        # any key skips the splash
        if not self._done:
            self._done = True
            self.dismiss()
