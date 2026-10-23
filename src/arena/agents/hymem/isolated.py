"""Isolated-reasoning space: a fresh sub-session for one hard question.

C^iso = (P^iso + brief, d, M_epi, M_wm, M_tool, tau_k(H^p)) plus the current state:
partially open at ingress (it reads memory and the recent planner history),
strictly closed at egress (only the <return> block goes back). It may delegate
calculations to a fresh executor; those results stay inside this session.
Thinking mode can be enabled for this space alone (``iso_thinking``), so the
model deliberates only where the deliberation never reaches the planner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arena.agents.hymem import prompts as P
from arena.agents.hymem.messages import IsoReturn, clip, parse_planner, parse_return, strip_think
from arena.state.views import render_options, render_state

if TYPE_CHECKING:
    from arena.agents.hymem.agent import HyMemAgent
    from arena.agents.hymem.session import BattleSession


class IsolatedReasoner:
    def __init__(self, agent: "HyMemAgent"):
        self.a = agent

    async def run(self, sess: "BattleSession", battle, options, description: str, prompt: str) -> IsoReturn:
        a, cfg = self.a, self.a.cfg
        recent = sess.history_text(sess.entries[-cfg.tau_k:]) if cfg.tau_k > 0 else ""
        user = P.ISO_USER.format(
            memory=(a.memory_text(sess, "iso") if cfg.memory_enabled else "") or P.MEMORY_EMPTY,
            history=recent or P.HISTORY_EMPTY, state=render_state(battle, dossier=cfg.dossier),
            options=render_options(options), description=description, prompt=prompt)
        messages = [{"role": "system", "content": P.ISO_SYSTEM.format(brief=sess.brief)},
                    {"role": "user", "content": user}]
        last_text = ""
        for turn in range(cfg.max_iso_turns):
            if turn > 0 and turn == cfg.max_iso_turns - 1:
                messages[-1]["content"] += "\n\n" + P.ISO_TIMEOUT
            resp = await a.call("iso", messages, sess=sess)
            last_text = strip_think(resp.text).strip()
            ret = parse_return(resp.text)
            if ret and ret["conclusion"]:
                return IsoReturn(description, prompt, ret["conclusion"], ret["evidence"],
                                 ret["confidence"], ret["notes"], success=True)
            tasks = [d for d in parse_planner(resp.text)[0] if d.kind == "task"]
            if tasks and turn < cfg.max_iso_turns - 1:
                r = await a.executor.run(sess, battle, tasks[0].text, parent="iso")
                sess.trigger.record_tools(r.n_tool_calls)
                sess.stats.tool_calls += r.n_tool_calls
                sess.store.exec_log.append({"task": r.task, "status": r.status, "findings": r.findings,
                                            "window": sess.store.folds, "parent": "iso"})
                follow = r.render() + "\n\n" + P.ISO_CONTINUE
            else:
                follow = P.ISO_CONTINUE
            messages += [{"role": "assistant", "content": last_text or "(no output)"},
                         {"role": "user", "content": follow}]
        return IsoReturn(description, prompt, conclusion=f"(incomplete) {clip(last_text, 300)}",
                         confidence=0.2, success=False)
