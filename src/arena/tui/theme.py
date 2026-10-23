"""POKEBOTMON TUI theme — a modern, low-saturation dark palette with a muted
Famicom-red accent. Surface sits clearly above the background so input boxes are
visible. Registered + activated in ArenaApp.
"""

from __future__ import annotations

from textual.theme import Theme

# muted / modern palette (lower saturation than raw NES)
PRIMARY = "#d2553f"     # muted famicom red  -> primary (buttons / titles)
CREAM = "#ece7da"       # soft bone white    -> foreground / accent
BLUE = "#6b91cf"        # muted blue         -> secondary / info
YELLOW = "#d8b24c"      # muted gold         -> warning
GREEN = "#6fae6a"       # muted green        -> success
BG = "#23232b"          # modern neutral-dark background (not pure black)
SURFACE = "#33333e"     # clearly lighter than BG  -> input/select boxes are visible
PANEL = "#3c3c48"       # panel / raised surface

FAMICOM = Theme(
    name="famicom",
    primary=PRIMARY,
    secondary=BLUE,
    accent=CREAM,
    foreground=CREAM,
    background=BG,
    surface=SURFACE,
    panel=PANEL,
    success=GREEN,
    warning=YELLOW,
    error="#c0473a",
    dark=True,
    luminosity_spread=0.10,
    text_alpha=0.96,
    variables={
        "footer-key-foreground": YELLOW,
        "footer-description-foreground": CREAM,
        "block-cursor-background": PRIMARY,
        "block-cursor-foreground": CREAM,
        "button-color-foreground": "#1b1b22",
        "border": PRIMARY,
        "input-cursor-background": PRIMARY,
    },
)

# win-rate / value ramp, low -> high = red -> green (used by tui_charts)
NES_RAMP = ["#d2553f", "#d98b3e", "#d8b24c", "#8cbf66", "#6fae6a"]
