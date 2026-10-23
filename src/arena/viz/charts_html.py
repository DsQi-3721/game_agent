"""Tier (ii): a self-contained interactive Plotly HTML from report.json.

Three figures in one page: win-rate matrix heatmap, leaderboard bar with Wilson-CI
error bars, and cost/tokens. Opened in a browser; no service required.
"""

from __future__ import annotations

from pathlib import Path


def _agents_by_elo(report: dict) -> list[str]:
    ag = report.get("agents", {})
    return sorted(ag, key=lambda n: -ag[n].get("elo", 1000))


def _heatmap(report):
    import plotly.graph_objects as go

    agents = _agents_by_elo(report)
    mat = report.get("matrix", {})
    z, text = [], []
    for a in agents:
        zrow, trow = [], []
        for b in agents:
            cell = mat.get(a, {}).get(b)
            if a == b or not cell or not cell.get("n"):
                zrow.append(None)
                trow.append("")
            else:
                zrow.append(cell["wins"] / cell["n"])
                trow.append(f"{cell['wins']}-{cell['losses']}")
        z.append(zrow)
        text.append(trow)
    fig = go.Figure(go.Heatmap(
        z=z, x=agents, y=agents, text=text, texttemplate="%{text}",
        colorscale="RdYlGn", zmin=0, zmax=1, colorbar=dict(title="win rate")))
    fig.update_layout(title="Win-rate matrix (row beats column)", height=420,
                      yaxis=dict(autorange="reversed"))
    return fig


def _leaderboard(report):
    import plotly.graph_objects as go

    ag = report.get("agents", {})
    names = _agents_by_elo(report)
    y = [ag[n]["win_rate"] for n in names]
    lo = [ag[n]["win_rate"] - ag[n]["wilson_ci"][0] for n in names]
    hi = [ag[n]["wilson_ci"][1] - ag[n]["win_rate"] for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=y, marker_color="#3b82f6",
        error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo)))
    fig.update_layout(title="Win rate + 95% Wilson CI", height=380,
                      yaxis=dict(title="win rate", range=[0, 1]))
    return fig


def _cost(report):
    import plotly.graph_objects as go

    ag = report.get("agents", {})
    names = _agents_by_elo(report)
    fig = go.Figure([
        go.Bar(name="cost $", x=names, y=[ag[n].get("cost_usd", 0) for n in names],
               marker_color="#f59e0b"),
        go.Bar(name="fallback %", x=names,
               y=[ag[n].get("fallback_rate", 0) * 100 for n in names], marker_color="#ef4444",
               yaxis="y2"),
    ])
    fig.update_layout(title="Cost & fallback rate", height=360, barmode="group",
                      yaxis=dict(title="cost $"),
                      yaxis2=dict(title="fallback %", overlaying="y", side="right"))
    return fig


def build(report: dict, out: Path | str) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figs = [_heatmap(report), _leaderboard(report), _cost(report)]
    parts = [f.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False))
             for i, f in enumerate(figs)]
    title = report.get("experiment", "arena")
    out.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title} · charts</title>"
        f"<style>body{{font-family:sans-serif;margin:24px;background:#0b0b0b;color:#eee}}</style>"
        f"</head><body><h2>{title}</h2>{''.join(parts)}</body></html>",
        encoding="utf-8")
    return out
