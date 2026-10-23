"""Tier (i): always-available in-terminal charts (Rich renderables).

A colored win-rate matrix heatmap and horizontal bar charts — render inside the
results screen via a Static, so there's a visual even without Plotly/W&B.
"""

from __future__ import annotations

from rich.table import Table
from rich.text import Text

# win-rate ramp, low -> high = red -> green (NES palette)
_RAMP = ["#d13b27", "#e08a1e", "#f8c537", "#7ec850", "#5ab552"]


def _agents_by_elo(report: dict) -> list[str]:
    ag = report.get("agents", {})
    return sorted(ag, key=lambda n: -ag[n].get("elo", 1000))


def heatmap(report: dict) -> Table:
    agents = _agents_by_elo(report)
    mat = report.get("matrix", {})
    t = Table(show_lines=False, expand=False)
    t.add_column("", style="bold")
    for b in agents:
        t.add_column(b[:8], justify="center")
    for a in agents:
        cells = [Text(a[:12], style="bold")]
        for b in agents:
            cell = mat.get(a, {}).get(b)
            if a == b or not cell or not cell.get("n"):
                cells.append(Text("·", style="dim"))
                continue
            wr = cell["wins"] / cell["n"]
            color = _RAMP[min(4, int(wr * 5))]
            cells.append(Text(f"{cell['wins']}-{cell['losses']}", style=f"bold white on {color}"))
        t.add_row(*cells)
    return t


def bar_chart(report: dict, key: str = "elo", title: str = "Elo", width: int = 30) -> Text:
    ag = report.get("agents", {})
    names = _agents_by_elo(report)
    vals = [ag[n].get(key, 0) for n in names]
    if not vals:
        return Text("(no data)")
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1
    out = Text()
    for n, v in zip(names, vals):
        fill = int((v - lo) / span * width) if key == "elo" else int(v * width)
        fill = max(1, min(width, fill))
        out.append(f"{n[:14]:<14} ", style="#fcf7e8")
        out.append("█" * fill, style="#5ab552")
        out.append(f" {v:.0f}\n" if key == "elo" else f" {v:.0%}\n")
    return out
