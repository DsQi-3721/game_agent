"""End-to-end: drive the real TUI (headless Pilot) through a tiny live eval.

Launches the actual subprocess, tails real events.jsonl, and confirms the monitor
populates. Needs the local Showdown server up. Saves a live monitor screenshot.

    python scripts/tui_e2e.py
"""

import asyncio
import os

for _v in ("ALL_PROXY", "all_proxy", "WS_PROXY", "ws_proxy"):
    os.environ.pop(_v, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1," + os.environ.get("NO_PROXY", "")

from arena.config.schema import AgentConfig  # noqa: E402
from arena.tui.app import ArenaApp  # noqa: E402


async def main():
    app = ArenaApp()
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        await pilot.press("space")          # skip splash
        await pilot.pause()
        d = app.screen.draft                # configure directly
        d.name = "tui_e2e"
        d.format = "gen9randombattle"
        d.eval_mode = "round_robin"
        d.n_battles = 1
        d.agents = [AgentConfig(name="r1", type="random"),
                    AgentConfig(name="r2", type="random")]
        app.start_eval(d)
        for _ in range(200):                # wait up to ~100s for completion
            await pilot.pause(0.5)
            m = app.monitor
            if m and m.total and m.done >= m.total:
                break
        app.save_screenshot("runs/_ui/05_monitor_live.svg")
        m = app.monitor
        print(f"[e2e] battles done {m.done}/{m.total}  standings rows="
              f"{m.query_one('#standings').row_count}")


if __name__ == "__main__":
    asyncio.run(main())
