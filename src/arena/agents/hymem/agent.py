"""HyMemAgent: the battle-facing shell around the four context spaces.

Per decision: ingest the new protocol events as typed <TURN_RESULT> entries
(phi on the environment's raw trace), run the planner loop, validate the chosen
option (random legal fallback, like every other agent), record a <DECISION>
entry and per-decision telemetry. Robust by construction: a single legal option
skips the LLM, a server re-request of the same decision never re-plans, and any
error or timeout degrades to a legal fallback instead of stalling the battle.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

from arena.agents.base import AgentContext
from arena.agents.hymem.config import ABLATIONS, AUX_SPACES, DEFAULT_TOOLS, HyMemConfig
from arena.agents.hymem.executor import Executor
from arena.agents.hymem.isolated import IsolatedReasoner
from arena.agents.hymem.memory import MemoryManager, render_memory, render_tool_memory
from arena.agents.hymem.messages import Entry, clip, decision_entry, note_entry
from arena.agents.hymem.planner import Planner
from arena.agents.hymem.router import Router
from arena.agents.hymem.session import BattleSession, DecisionStats, SessionStore
from arena.agents.llm.base import LLMAgent
from arena.llm.client import LLMResponse
from arena.llm.spaces import complete_in_space, model_spec
from arena.logging.calls import CallLogger
from arena.parse.validate import validate_or_fallback
from arena.registry.agents import register_agent
from arena.registry.param import Param
from arena.state.battle_log import events_since
from arena.state.views import legal_options, option_for, render_brief, render_state


@register_agent(
    "llm_hymem", is_llm=True, needs_model=True, label="LLM - HyMem (hierarchical context)",
    params=[
        Param("planner_steps", int, 4, help="planner turns per decision (last is answer-only)", label="Planner steps"),
        Param("execute_turns", int, 3, help="executor tool-loop turns (paper: 10)", label="Executor turns"),
        Param("ablation", str, "full", choices=list(ABLATIONS), help="paper ablations", label="Ablation"),
        Param("fold_mode", str, "two_stage", choices=["two_stage", "direct"], label="Fold mode"),
        Param("router", str, "rule", choices=["off", "rule", "llm"], label="Task router"),
        Param("think_iso", bool, False, help="thinking mode inside isolated reasoning", label="Iso thinking"),
        Param("replan", bool, False, help="verify once when answer confidence is low", label="Replan"),
        Param("aux_model", str, "", help="registry model for memory/router/distill (empty = same)", label="Aux model"),
        Param("tools", str, "", help="comma list of executor tools (empty = default set)", label="Executor tools"),
        Param("config_path", str, "", help="YAML with HyMem params", label="Config YAML"),
    ],
)
class HyMemAgent(LLMAgent):
    name = "llm_hymem"

    def __init__(self, ctx: Optional[AgentContext] = None, *, model: str = "qwen3.5-4b-mlx",
                 supports_tools: Optional[bool] = None, **params):
        super().__init__(ctx, model=model)
        self.cfg = HyMemConfig.from_params(**params)
        self._supports_tools = supports_tools
        self.sessions = SessionStore(self.cfg)
        self.call_log = CallLogger.beside(self.ctx.logger, level=self.cfg.log_calls)
        self.memory = MemoryManager(self.cfg)
        self.router = Router(self.cfg.router)
        self.executor = Executor(self)
        self.iso = IsolatedReasoner(self)
        self.planner = Planner(self)
        self._tools = None
        self.n_tool_calls = 0
        self.metrics: dict = defaultdict(float)

    # --- infrastructure used by the spaces ------------------------------------------
    def supports_tools(self) -> bool:
        if self._supports_tools is not None:
            return self._supports_tools
        spec = model_spec(self.ctx.llm, self.model)
        return spec.supports_tools if spec is not None else True

    def context_window(self) -> int:
        spec = model_spec(self.ctx.llm, self.model)
        return int(self.cfg.context_window or (spec.context_window if spec and spec.context_window else 32768))

    def executor_tools(self) -> list:
        if self._tools is None:
            names = self.cfg.tools
            tools: list = []
            if names is None and self.ctx.augmentations is not None:
                tools = self.ctx.augmentations.tools()
            if not tools:
                from arena.registry.augments import entries
                reg = entries()
                for n in names or DEFAULT_TOOLS:
                    if n in reg:
                        t = reg[n]().as_tool()
                        if t is not None:
                            tools.append(t)
            self._tools = tools
        return self._tools

    def memory_text(self, sess: BattleSession, space: str) -> str:
        """The memory each space reads: planner / iso get everything, the executor only the
        tool memory (paper's asymmetric injection). Subclasses may route it differently."""
        if space == "executor":
            return render_tool_memory(sess.store)
        return render_memory(sess.store)

    async def answer_now(self, sess: BattleSession, battle: AbstractBattle, options: list, step: int) -> bool:
        """Planner hook: True turns this planner step into an answer-only step (default: never)."""
        return False

    def passive_text(self, battle: AbstractBattle) -> str:
        if self.ctx.augmentations is None:
            return ""
        return "\n".join(f"[{r.name}]\n{r.text}" for r in self.ctx.augmentations.passive(battle))

    async def call(self, space: str, messages: list[dict], *, tools=None,
                   sess: Optional[BattleSession] = None) -> LLMResponse:
        sc = self.cfg.space(space)
        model = self.cfg.aux_model if (space in AUX_SPACES and self.cfg.aux_model) else self.model
        t_wall = time.time()
        resp, latency = await complete_in_space(self.ctx.llm, model, messages, sc, tools=tools)
        self.account(resp)
        self.metrics["llm_calls"] += 1
        self.metrics[f"prompt_tokens_{space}"] += resp.prompt_tokens
        self.metrics[f"llm_time_s_{space}"] += latency
        if sess is not None:
            st = sess.stats
            st.llm_calls += 1
            st.prompt_tokens += resp.prompt_tokens
            st.completion_tokens += resp.completion_tokens
            st.by_space[space] = st.by_space.get(space, 0) + resp.prompt_tokens
            st.llm_time_by_space[space] = st.llm_time_by_space.get(space, 0.0) + latency
        if self.call_log is not None:
            self.call_log.log(battle_tag=sess.tag if sess else "", decision=sess.decision_idx if sess else 0,
                              space=space, messages=messages, output=resp.text, reasoning=resp.reasoning,
                              prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                              latency_s=latency, t_start=t_wall, cached_tokens=resp.cached_tokens,
                              finish_reason=resp.finish_reason)
        return resp

    async def text(self, space: str, messages: list[dict], *, sess: Optional[BattleSession] = None) -> str:
        return (await self.call(space, messages, sess=sess)).text

    def log_event(self, sess: BattleSession, kind: str, data: dict) -> None:
        if self.call_log is not None:
            self.call_log.log(battle_tag=sess.tag, decision=sess.decision_idx, space=f"event:{kind}",
                              messages=[], output="", extra=data)

    # --- decision ------------------------------------------------------------------
    def _ingest(self, sess: BattleSession, battle: AbstractBattle) -> None:
        events, sess.cursor = events_since(battle, sess.cursor)
        if not events:
            return
        results, facts = sess.digester.ingest(events, battle)
        for r in results:
            sess.add(Entry("turn", r.render(), sess.decision_idx, r.turn))
        sess.store.key_facts.extend(facts)

    async def act(self, battle: AbstractBattle) -> BattleOrder:
        sess = self.sessions.get_or_create(battle.battle_tag, lambda: render_brief(battle))
        sess.in_flight += 1
        try:
            return await self._act(sess, battle)
        finally:
            self.sessions.release(sess)

    async def _act(self, sess: BattleSession, battle: AbstractBattle) -> BattleOrder:
        t0 = time.monotonic()
        t_wall = time.time()
        options = legal_options(battle)
        req = getattr(battle, "_last_request", None) or {}
        key = (req.get("rqid") if isinstance(req, dict) else None, len(getattr(battle, "_replay_data", []) or []))

        if sess.last_key == key and sess.last_order is not None:
            # the server re-requested the same decision (our order was rejected): no re-planning
            alt = next((o for o in options if o.order.message != sess.last_order), None)
            final, fallback = validate_or_fallback(alt.order if alt else None, battle)
            sess.stats = DecisionStats(t_start=t_wall)   # nothing new to account for this re-request
            sess.add(note_entry(sess.decision_idx, battle.turn,
                                f"The server rejected '{sess.last_order}'; played '{final.message}' instead."))
            sess.last_order = final.message
            self.metrics["recalls"] += 1
            return self._finish(sess, battle, options, final, fallback, None, t0, note="recall", recall=True)

        sess.decision_idx += 1
        sess.stats = DecisionStats(t_start=t_wall)
        self._ingest(sess, battle)

        if len(options) <= 1:   # nothing to decide (recharge, last Pokemon, ...)
            final, fallback = validate_or_fallback(options[0].order if options else None, battle)
            sess.last_key, sess.last_order = key, final.message
            self.metrics["single_option"] += 1
            return self._finish(sess, battle, options, final, fallback, None, t0, note="single_option")

        opt, conf, raw, error = await self._decide(sess, battle, options)
        final, fallback = validate_or_fallback(opt.order if opt else None, battle)
        sess.last_key, sess.last_order = key, final.message
        return self._finish(sess, battle, options, final, fallback, conf, t0, raw=raw, error=error)

    async def _decide(self, sess: BattleSession, battle: AbstractBattle,
                      options: list) -> tuple[Optional[object], Optional[float], str, str]:
        """-> (option or None, confidence, raw planner output, error). Subclasses put a fast
        path in front of the planner here (see ``arena.agents.system1``)."""
        try:
            opt, conf, raw = await asyncio.wait_for(self.planner.decide(sess, battle, options),
                                                    timeout=self.cfg.decision_budget_s * 2)
            return opt, conf, raw, ""
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            self.metrics["timeouts"] += 1
            return None, None, "", "decision_timeout"
        except Exception as e:  # noqa: BLE001  (never stall a battle on an agent bug or API error)
            self.metrics["errors"] += 1
            return None, None, "", f"{type(e).__name__}: {clip(str(e), 200)}"

    def _finish(self, sess, battle, options, final: BattleOrder, fallback: bool, conf, t0: float, *,
                raw: str = "", error: str = "", note: str = "", recall: bool = False) -> BattleOrder:
        opt = option_for(options, final)
        label = opt.label.split(" -- ")[0] if opt else final.message
        if not recall:
            sess.add(decision_entry(sess.decision_idx, battle.turn, label, conf))
        self.n_decisions += 1
        self.n_fallbacks += int(fallback)
        latency = time.monotonic() - t0
        st = sess.stats
        m = self.metrics
        m["decisions"] += 1
        m["fallbacks"] += int(fallback)
        m["parse_failures"] += st.parse_failures
        m["routed_to_iso"] += st.routed_to_iso
        m["planner_ctx_tokens_sum"] += sum(st.planner_prompt_tokens)
        if st.planner_prompt_tokens:
            m["planner_ctx_tokens_max"] = max(m.get("planner_ctx_tokens_max", 0), max(st.planner_prompt_tokens))
        m["decision_latency_s_sum"] += latency
        m["decision_latency_s_max"] = max(m.get("decision_latency_s_max", 0.0), latency)
        extra = {
            "decision": sess.decision_idx, "planner_calls": st.planner_calls,
            "planner_prompt_tokens": st.planner_prompt_tokens[-1] if st.planner_prompt_tokens else 0,
            "planner_prompt_tokens_max": max(st.planner_prompt_tokens) if st.planner_prompt_tokens else 0,
            "llm_calls": st.llm_calls, "prompt_tokens": st.prompt_tokens, "completion_tokens": st.completion_tokens,
            "exec_calls": st.exec_calls, "iso_calls": st.iso_calls, "tool_calls": st.tool_calls,
            "routed_to_iso": st.routed_to_iso, "parse_failures": st.parse_failures, "folds": len(st.folds),
            "fold_reasons": ",".join(st.folds), "directives": ",".join(st.directives),
            "confidence": conf if conf is not None else -1.0, "latency_s": round(latency, 2),
            "hp_entries": len(sess.entries), "fallback": fallback, "error": error, "note": note,
            "prompt_tokens_by_space": dict(st.by_space),
            "t_start": round(st.t_start, 3), "t_end": round(time.time(), 3),
            "llm_time_by_space": {k: round(v, 3) for k, v in st.llm_time_by_space.items()},
            "tool_time_s": round(st.tool_time_s, 4), "distill_time_s": round(st.distill_time_s, 4),
        }
        extra.update(st.extra)
        if self.ctx.logger is not None:
            self.ctx.logger.log_turn(
                battle_tag=battle.battle_tag, turn=battle.turn, state_text=render_state(battle),
                prompt=sess.history_text()[-4000:], raw_llm_output=raw, tool_calls=[],
                parsed_order=final.message, fallback_used=fallback,
                note=note or (error and f"error={error}") or f"directives={extra['directives']}", extra=extra)
        return final

    # --- lifecycle / metrics -------------------------------------------------------------
    def on_battle_end(self, battle: AbstractBattle) -> None:
        sess = self.sessions.close(battle.battle_tag)
        if sess is None:
            return
        self.metrics["battles"] += 1
        try:
            self._ingest(sess, battle)          # complete the key facts with the final events
        except Exception:  # noqa: BLE001
            pass
        self.log_event(sess, "battle_end", {
            "won": battle.won, "turns": battle.turn, "decisions": sess.decision_idx,
            "planner_turns": sess.planner_turns_total, "folds": sess.store.folds,
            "fold_events": sess.store.fold_events,
            "key_facts": [{"turn": kf.turn, "kind": kf.kind, "text": kf.text} for kf in sess.store.key_facts],
        })

    def extra_metrics(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.metrics.items()}
