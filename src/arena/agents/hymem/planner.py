"""Planner space: the only persistent context, and the per-decision control loop.

Every planner call is rebuilt as [system: P^p + battle brief] [user: memory +
H^p (typed entries since the last fold) + the CURRENT state + numbered legal
actions + instruction]. The state is replaced, never accumulated; H^p only grows
through bounded typed returns; folds rewrite memory and reset H^p to a recent
buffer. Raw tool output and isolated deliberation have no path in here
(paper Eq. 4) unless the ``no_typed_isolation`` ablation is on.

Transition rule per planner step (paper Eq. 16, reference ``run``):
  <answer>     -> commit this decision (optionally one low-confidence replan)
  <to_memory/> -> fold now
  <task>       -> router -> executor (or isolated session when complex) -> <RESULTS>
  <isoReason>  -> isolated session -> <RETURN>
Several task/isoReason directives in one reply run concurrently.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Optional

from arena.agents.hymem import prompts as P
from arena.agents.hymem.messages import (
    Entry,
    ExecResult,
    IsoReturn,
    clip,
    note_entry,
    parse_planner,
    strip_think,
)
from arena.parse.choice import parse_choice
from arena.state.views import Option, render_options, render_state

if TYPE_CHECKING:
    from arena.agents.hymem.agent import HyMemAgent
    from arena.agents.hymem.session import BattleSession


def estimate_tokens(messages: list[dict]) -> int:
    """Reference fallback estimate: characters / 3.5."""
    return int(sum(len(str(m.get("content") or "")) for m in messages) / 3.5)


class Planner:
    def __init__(self, agent: "HyMemAgent"):
        self.a = agent

    # --- context -----------------------------------------------------------------
    def messages(self, sess: "BattleSession", battle, options: list[Option], instruction: str,
                 *, with_options: bool = True) -> list[dict]:
        cfg = self.a.cfg
        memory = self.a.memory_text(sess, "planner") if cfg.memory_enabled else ""
        passive = self.a.passive_text(battle)
        user = P.PLANNER_USER.format(
            memory=memory or P.MEMORY_EMPTY,
            history=sess.history_text() or P.HISTORY_EMPTY,
            state=render_state(battle, dossier=cfg.dossier, extended=cfg.state_view == "obs1"),
            passive=f"\n{passive}\n" if passive else "",
            options=render_options(options) if with_options else "(omitted)",
            instruction=instruction)
        return [{"role": "system", "content": P.PLANNER_SYSTEM.format(brief=sess.brief)},
                {"role": "user", "content": user}]

    def _overflow_guard(self, sess, battle, options, instruction, msgs, window) -> list[dict]:
        """no_mm ablation: history grows unmanaged; only a hard cut at the window."""
        while estimate_tokens(msgs) > 0.95 * window and len(sess.entries) > 1:
            sess.entries.pop(0)
            self.a.metrics["hp_truncations"] += 1
            msgs = self.messages(sess, battle, options, instruction)
        return msgs

    @staticmethod
    def task_line(battle) -> str:
        return (f"{battle.format} singles battle, you are {battle.player_role}; objective: make all of the "
                f"opponent's Pokemon faint before yours do.")

    # --- fold ---------------------------------------------------------------------
    async def fold(self, sess: "BattleSession", battle, options, reason: str) -> None:
        a, cfg = self.a, self.a.cfg
        before_entries, before_tokens = len(sess.entries), sess.last_planner_tokens
        cut, d0 = len(sess.entries), sess.decision_idx      # entries added while folding stay unsummarised
        t_wall = time.time()
        summary_msgs = (self.messages(sess, battle, options, P.FOLD_SUMMARY_REQUEST, with_options=False)
                        if cfg.fold_mode in ("two_stage", "typed") else None)
        outcome = await a.memory.fold(
            sess.store, call=lambda space, m: a.call(space, m, sess=sess), reason=reason,
            task=self.task_line(battle), summary_messages=summary_msgs, history_text=sess.history_text())
        sess.keep_recent(cfg.recent_buffer, cut=cut, decision=d0)
        sess.trigger.reset(sess.planner_turns_total)
        sess.iso_used_window = 0
        sess.stats.folds.append(reason)
        after_tokens = estimate_tokens(self.messages(sess, battle, options, P.INSTR_ANSWER_ONLY))
        event = {"decision": sess.decision_idx, "turn": battle.turn, "reason": reason, "mode": outcome.mode,
                 "summary_ok": outcome.summary_ok, "extracted": outcome.extracted,
                 "entries_before": before_entries, "entries_after": len(sess.entries),
                 "tokens_before": before_tokens, "tokens_after_reset": after_tokens,
                 "seconds": round(outcome.seconds, 2), "t_start": round(t_wall, 3)}
        sess.store.fold_events.append(event)
        a.metrics["folds"] += 1
        a.log_event(sess, "fold", event)

    # --- dispatch -------------------------------------------------------------------
    def _iso_budget(self, sess, iso_this_decision: int) -> bool:
        cfg = self.a.cfg
        return (cfg.iso_enabled and sess.iso_used_window < cfg.max_iso_per_window
                and iso_this_decision < cfg.max_iso_per_decision)

    async def dispatch(self, sess, battle, options, work, iso_this: int) -> int:
        """Route and run task/iso directives concurrently; append typed returns. -> isos used."""
        a, cfg = self.a, self.a.cfg
        hint_iso = any(d.kind == "iso" for d in work)
        plan: list[tuple[str, str, str]] = []
        for d in work:
            if d.kind == "iso" and self._iso_budget(sess, iso_this):
                plan.append(("iso", d.text, d.description))
                iso_this += 1
                sess.iso_used_window += 1
                continue
            if d.kind == "iso":                         # ablation demotion / budget exhausted
                plan.append(("exec", d.text or d.description, ""))
                continue
            if cfg.iso_enabled and cfg.router != "off" and self._iso_budget(sess, iso_this):
                route = await a.router.route(d.text, hint_iso=hint_iso,
                                             llm_call=lambda m: a.text("router", m, sess=sess))
                if route.target == "iso":
                    plan.append(("iso", d.text, f"Research: {clip(d.text, 40)}"))
                    iso_this += 1
                    sess.iso_used_window += 1
                    sess.stats.routed_to_iso += 1
                    continue
            plan.append(("exec", d.text, ""))

        async def run_one(kind: str, text: str, desc: str):
            if kind == "exec":
                return await a.executor.run(sess, battle, text, parent="planner")
            return await a.iso.run(sess, battle, options, desc, text)

        sem = asyncio.Semaphore(max(1, cfg.max_parallel))

        async def guarded(p):
            async with sem:
                return await run_one(*p)

        results = await asyncio.gather(*(guarded(p) for p in plan), return_exceptions=True)
        for (kind, text, desc), r in zip(plan, results):
            if isinstance(r, asyncio.CancelledError):
                raise r
            if isinstance(r, BaseException):
                r = (ExecResult(task=text, findings="", status="failure", gaps=f"error: {type(r).__name__}")
                     if kind == "exec" else IsoReturn(desc, text, f"(failed: {type(r).__name__})", success=False))
            self._record(sess, battle, kind, r)
        return iso_this

    def _record(self, sess, battle, kind: str, r) -> None:
        a, cfg, st = self.a, self.a.cfg, sess.stats
        if kind == "exec":
            sess.trigger.record_executor(r.success, r.n_tool_calls)
            st.exec_calls += 1
            st.tool_calls += r.n_tool_calls
            a.metrics["exec_calls"] += 1
            sess.store.exec_log.append({"task": r.task, "status": r.status, "findings": r.findings,
                                        "window": sess.store.folds, "parent": "planner"})
            text = r.render()
            if not cfg.typed_isolation and r.raw_observations:   # ablation: raw trace reaches the planner
                text += "\nRAW TOOL OUTPUT:\n" + "\n".join(r.raw_observations)
            sess.add(Entry("results", text, sess.decision_idx, battle.turn))
        else:
            sess.trigger.record_executor(r.success, 0)
            st.iso_calls += 1
            a.metrics["iso_calls"] += 1
            sess.store.iso_log.append({"description": r.description, "conclusion": r.conclusion,
                                       "confidence": r.confidence, "success": r.success,
                                       "window": sess.store.folds})
            sess.add(Entry("return", r.render(), sess.decision_idx, battle.turn))

    # --- the decision loop -----------------------------------------------------------
    async def decide(self, sess: "BattleSession", battle, options: list[Option]) -> tuple[Optional[Option], Optional[float], str]:
        """-> (chosen option or None, confidence, last raw planner output)."""
        a, cfg, st = self.a, self.a.cfg, sess.stats
        deadline = time.monotonic() + cfg.decision_budget_s
        window = a.context_window()
        iso_this, replanned, mode, last_text = 0, False, "normal", ""
        replan_info: tuple = ()
        for step in range(cfg.planner_steps):
            steps_left = cfg.planner_steps - step
            answer_only = steps_left == 1 or time.monotonic() > deadline
            if not answer_only and await a.answer_now(sess, battle, options, step):
                answer_only = True                   # a controller judged the facts sufficient
            if cfg.memory_enabled and not answer_only:
                reason = sess.trigger.should_fold(context_tokens=sess.last_planner_tokens, window=window,
                                                  global_turn=sess.planner_turns_total)
                if reason:
                    await self.fold(sess, battle, options, reason)
            if answer_only:
                instruction = P.INSTR_ANSWER_ONLY
            elif mode == "replan":
                instruction = P.INSTR_REPLAN.format(conf=replan_info[0], answer=replan_info[1], steps=steps_left)
                mode = "normal"                     # the replan prompt is shown once
            elif cfg.iso_enabled:
                iso_left = max(0, min(cfg.max_iso_per_window - sess.iso_used_window,
                                      cfg.max_iso_per_decision - iso_this))
                instruction = P.INSTR_NORMAL.format(steps=steps_left, iso=iso_left)
            else:
                instruction = P.INSTR_NO_ISO.format(steps=steps_left)
            msgs = self.messages(sess, battle, options, instruction)
            if not cfg.memory_enabled:
                msgs = self._overflow_guard(sess, battle, options, instruction, msgs, window)
            resp = await a.call("planner", msgs, sess=sess)
            last_text = resp.text
            sess.trigger.record_planner_turn()
            sess.planner_turns_total += 1
            sess.last_planner_tokens = resp.prompt_tokens or estimate_tokens(msgs)
            st.planner_calls += 1
            st.planner_prompt_tokens.append(sess.last_planner_tokens)
            a.metrics["planner_calls"] += 1

            directives, analysis = parse_planner(resp.text)
            st.directives += [d.kind for d in directives] or ["none"]
            if analysis and cfg.keep_planner_analysis:
                sess.add(Entry("analysis", f"<ANALYSIS>{clip(analysis, 400)}</ANALYSIS>",
                               sess.decision_idx, battle.turn))

            answer = next((d for d in directives if d.kind == "answer"), None)
            if answer is not None:
                opt = parse_choice(answer.text, options)
                if opt is None:
                    st.parse_failures += 1
                    sess.add(note_entry(sess.decision_idx, battle.turn,
                                        f"'{clip(answer.text, 60)}' is not a legal action; answer with a number "
                                        f"from the list."))
                    if answer_only:
                        break
                    continue
                sess.trigger.record_confidence(answer.confidence)
                if (cfg.replan and not replanned and not answer_only and answer.confidence is not None
                        and answer.confidence < cfg.replan_threshold):
                    replanned, mode = True, "replan"
                    replan_info = (answer.confidence, f"{opt.idx} ({opt.label.split(' -- ')[0]})")
                    a.metrics["replans"] += 1
                    continue
                return opt, answer.confidence, last_text
            if answer_only:
                # last resort: a legal action named anywhere in the reply
                opt = parse_choice(strip_think(resp.text), options)
                if opt is not None:
                    return opt, None, last_text
                st.parse_failures += 1
                break
            if any(d.kind == "fold" for d in directives) and cfg.memory_enabled \
                    and sess.trigger.can_fold(sess.planner_turns_total):
                await self.fold(sess, battle, options, "active_fold")
                continue
            work = [d for d in directives if d.kind in ("task", "iso")][: cfg.max_parallel]
            if not work:
                st.parse_failures += 1
                sess.add(note_entry(sess.decision_idx, battle.turn,
                                    "No valid directive in the last reply. Use exactly one of <task>, "
                                    "<isoReason>, <to_memory/>, <answer>."))
                continue
            iso_this = await self.dispatch(sess, battle, options, work, iso_this)
        return None, None, last_text
