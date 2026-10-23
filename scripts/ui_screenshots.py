"""Capture SVG screenshots of the TUI (splash logo + config) without a terminal.

    python scripts/ui_screenshots.py
    open runs/_ui/*.svg     # view the logo / config in a browser
"""

import asyncio
from pathlib import Path

from arena.tui.app import ArenaApp

OUT = Path("runs/_ui")


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    app = ArenaApp()
    async with app.run_test(size=(140, 60)) as pilot:
        app.screen.query_one("#logo").styles.opacity = 1.0   # skip fade for the shot
        await pilot.pause(0.5)
        app.save_screenshot(str(OUT / "01_splash.svg"))    # POKEBOTMON logo
        await pilot.press("space")
        await pilot.pause()
        app.save_screenshot(str(OUT / "02_config.svg"))    # configurator
        # add an agent so the roster shows
        from textual.widgets import Input, Select
        scr = app.screen
        scr.query_one("#agent_type", Select).value = "llm_react"
        scr._rebuild_params()
        scr.query_one("#agent_name", Input).value = "react"
        await pilot.pause()
        await pilot.click("#add_agent")
        await pilot.pause()
        app.save_screenshot(str(OUT / "03_config_with_agent.svg"))

        # monitor with synthetic events (no server needed)
        from pathlib import Path

        from arena.events.schema import Event, EventType
        from arena.tui.screens.monitor_screen import MonitorScreen
        mon = MonitorScreen("demo", Path("runs/demo"))
        await app.push_screen(mon)
        await pilot.pause()
        e = lambda t, **d: Event(t, ts=0.0, seq=0, run="demo", data=d)
        mon.feed(e(EventType.RUN_STARTED, agents=["random", "maxpow", "react"], total_battles=12,
                   format="gen9randombattle", mode="round_robin", n_pairings=3, n_battles=4))
        mon.feed(e(EventType.PAIRING_STARTED, a="react", b="maxpow", index=2))
        for i, mv in enumerate(["earthquake", "switch garchomp", "icebeam"], 1):
            mon.feed(e(EventType.TURN, agent="react", battle_tag="battle-gen9randombattle-7",
                       turn=i, parsed_order=f"/choose move {mv}", fallback=False, n_tool_calls=i % 2))
        mon.feed(e(EventType.MATCH_FINISHED, a="react", b="maxpow", a_wins=4, b_wins=0, ties=0,
                   avg_turns=18.0, totals={"random": {"wins": 1, "games": 8}, "maxpow": {"wins": 4, "games": 8},
                                           "react": {"wins": 7, "games": 8}}))
        mon.feed(e(EventType.METRIC, elo={"react": 1064.0, "maxpow": 994.0, "random": 912.0},
                   win_rate={"react": 0.875, "maxpow": 0.5, "random": 0.125},
                   cost_usd={"react": 0.118}, completed_pairings=2, total_pairings=3))
        await pilot.pause()
        app.save_screenshot(str(OUT / "04_monitor.svg"))

        # results screen with a demo report
        import json

        from arena.tui.screens.results_screen import ResultsScreen
        from arena.viz.charts_html import build
        rep = {
            "experiment": "demo", "format": "gen9randombattle", "mode": "round_robin", "n_battles": 4,
            "agents": {
                "react": {"win_rate": 0.875, "wilson_ci": [0.53, 0.98], "wins": 7, "games": 8,
                          "elo": 1064.0, "fallback_rate": 0.0, "tool_calls": 12, "tokens": 23000, "cost_usd": 0.118},
                "maxpow": {"win_rate": 0.5, "wilson_ci": [0.3, 0.7], "wins": 4, "games": 8,
                           "elo": 994.0, "fallback_rate": 0.0, "tool_calls": 0, "tokens": 0, "cost_usd": 0.0},
                "random": {"win_rate": 0.125, "wilson_ci": [0.02, 0.47], "wins": 1, "games": 8,
                           "elo": 912.0, "fallback_rate": 0.0, "tool_calls": 0, "tokens": 0, "cost_usd": 0.0},
            },
            "matrix": {
                "react": {"maxpow": {"wins": 4, "losses": 0, "n": 4}, "random": {"wins": 3, "losses": 1, "n": 4}},
                "maxpow": {"react": {"wins": 0, "losses": 4, "n": 4}, "random": {"wins": 4, "losses": 0, "n": 4}},
                "random": {"react": {"wins": 1, "losses": 3, "n": 4}, "maxpow": {"wins": 0, "losses": 4, "n": 4}},
            },
            "total_cost_usd": 0.118, "total_tokens": 23000, "aborted": None,
        }
        Path("runs/demo").mkdir(parents=True, exist_ok=True)
        (Path("runs/demo") / "report.json").write_text(json.dumps(rep), encoding="utf-8")
        build(rep, Path("runs/demo/charts.html"))
        await app.push_screen(ResultsScreen(Path("runs/demo")))
        await pilot.pause()
        app.save_screenshot(str(OUT / "06_results.svg"))

        # trajectory / history viewer over the real runs/ dir
        from arena.tui.screens.trajectory_screen import TrajectoryScreen
        await app.push_screen(TrajectoryScreen("runs"))
        await pilot.pause()
        tscr = app.screen
        # show a run that has real LLM trajectories
        for i, r in enumerate(tscr.runs):
            if r["agents"]:
                tscr._select_run(i)
                tscr.query_one("#agent", Select).value = r["agents"][0]
                break
        await pilot.pause()
        app.save_screenshot(str(OUT / "07_trajectory.svg"))
    print(f"screenshots -> {OUT}/*.svg  (open in a browser)")


if __name__ == "__main__":
    asyncio.run(main())
