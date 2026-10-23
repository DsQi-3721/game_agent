"""POKEBOTMON brand logo: a round 3D-shaded Poké Ball + a 3d_diagonal wordmark.

The ball is drawn with solid blocks whose per-character color encodes both the
region (red top / white bottom / black band + button) and sphere shading, so it
reads as a round, lit Poké Ball. The wordmark is forced onto one line.
"""

from __future__ import annotations

import math

from rich.align import Align
from rich.console import Group
from rich.text import Text

# dark -> light ramps for the lit sphere
_RED = ["#6b0000", "#9e0000", "#d10000", "#ff2b2b", "#ff7a7a"]
_WHITE = ["#9aa0a6", "#c2c7cc", "#e3e7ea", "#f5f7f9", "#ffffff"]
_BLACK = "#141414"


def _ball_text(R: int = 13) -> Text:
    t = Text()
    band = 1.2          # half-height of the black band
    br = 4.3            # outer button radius (concentric rings inside)
    lx, ly, lz = -0.5, -0.55, 0.67   # light from upper-left-front
    n = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / n, ly / n, lz / n

    for y in range(-R, R + 1):
        for x in range(-2 * R, 2 * R + 1):
            fx = x / 2.0                  # chars are ~2x tall -> squash x for roundness
            d = math.hypot(fx, y)
            if d > R + 0.3:
                t.append(" ")
                continue
            if d <= br:                   # center button: concentric white/black rings
                if d < 1.6:
                    col = _WHITE[4]       # white core
                elif d < 2.5:
                    col = _BLACK          # inner black ring
                elif d < 3.4:
                    col = _WHITE[3]       # white ring
                else:
                    col = _BLACK          # outer black ring
                t.append("█", style=col)
                continue
            if abs(y) <= band:            # black band
                t.append("█", style=_BLACK)
                continue
            nx, ny = fx / R, y / R        # sphere shading
            nz = math.sqrt(max(0.0, 1 - nx * nx - ny * ny))
            b = max(0.0, nx * lx + ny * ly + nz * lz)
            ramp = _RED if y < 0 else _WHITE
            col = ramp[min(4, int(b * 5))]
            if d > R - 0.7:               # crisp darker rim
                col = _BLACK
            t.append("█", style=col)
        t.append("\n")
    return t


def _wordmark() -> Text:
    try:
        import pyfiglet
        art = pyfiglet.figlet_format("POKEBOTMON", font="3d_diagonal", width=400)
        return Text(art.rstrip("\n"), style="bold #ffcb05", no_wrap=True)  # Pokémon gold
    except Exception:  # noqa: BLE001
        return Text("P O K E B O T M O N", style="bold #ffcb05")


def render_logo() -> Group:
    return Group(
        Align.center(_ball_text()),
        Text(""),
        Align.center(_wordmark()),
    )
