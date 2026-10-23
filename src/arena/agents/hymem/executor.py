"""Executor space: a fresh, bounded tool-use loop per planner task.

C^e = (P^e, d, M_tool) plus an entity sheet so tool arguments use exact names.
Raw tool observations stay inside this context: each passes through phi(o|d)
before the model sees it, and the loop ends with a schema-bounded <results>
block (sigma). The context is discarded on return; only ``ExecResult`` leaves.
Reference behaviours kept: diminishing-returns early exit (``exec_early_exit``)
and a forced summary when the turn budget runs out.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Optional

from arena.agents.hymem import prompts as P
from arena.agents.hymem.distill import distill
from arena.agents.hymem.messages import ExecResult, clip, parse_results
from arena.llm.tool_calling import make_protocol
from arena.state.views import entity_sheet

if TYPE_CHECKING:
    from arena.agents.hymem.agent import HyMemAgent
    from arena.agents.hymem.session import BattleSession


def _low_info(obs: str) -> bool:
    s = (obs or "").strip()
    return len(s) < 120 and not any(ch.isdigit() for ch in s)


def detect_diminishing(window: list[str]) -> Optional[str]:
    """Reference O5 detector over the last two observations."""
    if len(window) < 2:
        return None
    a, b = window[-2], window[-1]
    if a[:200] == b[:200] and len(a) >= 50:
        return "loop"
    if _low_info(a) and _low_info(b):
        return "low_info"
    if "error" in a.lower() and "error" in b.lower():
        return "consecutive_errors"
    return None


def _tool_ok(name: str, out: str) -> bool:
    head = (out or "").strip().lower()
    return bool(head) and not (head.startswith(f"{name}:") or head.startswith("error") or head.startswith("no "))


class Executor:
    def __init__(self, agent: "HyMemAgent"):
        self.a = agent

    async def run(self, sess: "BattleSession", battle, task: str, parent: str = "planner") -> ExecResult:
        a, cfg = self.a, self.a.cfg
        tools = a.executor_tools()
        tool_map = {t.name: t for t in tools}
        proto = make_protocol(a.supports_tools(), final_hint=P.EXECUTOR_FINAL_HINT,
                              continue_hint=P.EXECUTOR_CONTINUE_HINT)
        docs = proto.tool_docs(tools)
        tm = a.memory_text(sess, "executor")
        user = P.EXECUTOR_USER.format(task=task, tool_memory=f"\n{tm}\n" if tm else "",
                                      entities=entity_sheet(battle), tool_docs=f"\n{docs}" if docs else "")
        messages = [{"role": "system", "content": P.EXECUTOR_SYSTEM}, {"role": "user", "content": user}]
        complete_tools = proto.complete_kwargs(tools).get("tools")
        n_tools, raw_obs, window = 0, [], []

        async def llm_text(msgs):
            return (await a.call("distill", msgs, sess=sess)).text

        for _turn in range(cfg.execute_turns):
            resp = await a.call("executor", messages, tools=complete_tools, sess=sess)
            calls = proto.extract_calls(resp) if tools else []
            parsed = parse_results(resp.text)
            if parsed and not calls:
                return self._result(task, parsed, n_tools, raw_obs)
            if not calls:
                messages += [{"role": "assistant", "content": resp.text or "(no output)"},
                             {"role": "user", "content": P.EXECUTOR_NUDGE}]
                continue
            calls = calls[: cfg.max_tool_calls_per_turn]
            outputs = []
            for c in calls:
                tool = tool_map.get(c.name)
                t_tool, t_wall = time.perf_counter(), time.time()
                if tool is None:
                    out = f"{c.name}: unknown tool. Available: {', '.join(tool_map)}"
                else:
                    try:
                        out = await tool.call(battle, c.arguments or {})
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:  # noqa: BLE001  (a tool bug must not kill the decision)
                        out = f"error: {type(e).__name__}: {e}"
                tool_dt = time.perf_counter() - t_tool
                sess.stats.tool_time_s += tool_dt
                ok = _tool_ok(c.name, out)
                n_tools += 1
                a.n_tool_calls += 1
                sess.store.tool_calls.append({"tool": c.name, "args": c.arguments, "ok": ok,
                                              "window": sess.store.folds})
                raw_obs.append(f"[{c.name}] {out}")
                t_dist = time.perf_counter()
                obs, _ = await distill(out, task, mode=cfg.distill, max_chars=cfg.distill_max_chars,
                                       llm_call=llm_text if cfg.distill == "llm" else None)
                dist_dt = time.perf_counter() - t_dist
                sess.stats.distill_time_s += dist_dt
                a.log_event(sess, "tool", {"tool": c.name, "t_start": round(t_wall, 3),
                                           "latency_s": round(tool_dt, 5), "distill_s": round(dist_dt, 5),
                                           "out_chars": len(out or ""), "obs_chars": len(obs or ""),
                                           "ok": ok, "parent": parent})
                outputs.append(obs)
                window.append(obs)
            proto.append_turn(messages, resp, calls, outputs)
            reason = detect_diminishing(window)
            if reason:
                messages.append({"role": "user", "content": P.EXECUTOR_EARLY_EXIT.format(reason=reason)})
                break

        if cfg.exec_synthesis == "llm":
            last = messages[-1]
            if last["role"] == "user":           # text-protocol results / nudge / early-exit note
                if "Write your <results> block now" not in (last.get("content") or ""):
                    last["content"] = (last.get("content") or "") + "\n\n" + P.EXECUTOR_TIMEOUT
            else:                                # native protocol: after tool messages
                messages.append({"role": "user", "content": P.EXECUTOR_TIMEOUT})
            resp = await a.call("executor", messages, sess=sess)
            parsed = parse_results(resp.text)
            if parsed:
                return self._result(task, parsed, n_tools, raw_obs)
            if resp.text.strip():
                return self._result(task, {"findings": clip(resp.text, 700), "status": "partial"}, n_tools, raw_obs)
        if raw_obs:   # rule-based sigma: the last distilled observations
            findings = "\n".join(clip(o, 280) for o in window[-3:])
            return self._result(task, {"findings": findings, "status": "partial",
                                       "gaps": "analyst did not write a results block"}, n_tools, raw_obs)
        return self._result(task, {"findings": "", "status": "failure", "gaps": "no tool was called"},
                            n_tools, raw_obs)

    @staticmethod
    def _result(task: str, parsed: dict, n_tools: int, raw_obs: list) -> ExecResult:
        return ExecResult(task=task, findings=parsed.get("findings", ""), evidence=parsed.get("evidence", ""),
                          sources=parsed.get("sources", ""), status=parsed.get("status", "success"),
                          gaps=parsed.get("gaps", ""), n_tool_calls=n_tools, raw_observations=list(raw_obs))
