"""Agents built on the System-1 layer.

``jev_fast``        pure System 1: every decision is one typed request. With ``student_url`` the
                    request first goes to a local student and only escalates to the teacher (Jev)
                    when the student's gate fails -- the cascade of the learning loop.
``llm_hymem_dual``  System 1 in front of and inside HyMem:
                    * in front: Tier 0 rule -> student -> teacher; confident answers are played,
                      otherwise HyMem plans with the System-1 ranking as a hint under a time cap and
                      the System-1 answer is played if it runs out of time (always a move in ~1 s);
                    * inside (``controller``): shared option facts in the planner prompt, step
                      control (answer now vs research), a System-One task router and executor;
                    * memory: a System-One router classifies / checks / places HyMem's memory items
                      per context (parallel request), optionally fed back to System 1 (fast_memory);
                    * folds run in the background, never inside a capped decision.
Both agents record their System-One traffic, decisions and outcomes (``s1exp_<agent>.jsonl``)
for the learning loop (``arena.s1``).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import time
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from arena.agents.base import Agent, AgentContext
from arena.agents.hymem.agent import HyMemAgent
from arena.agents.hymem.memory import render_memory, render_tool_memory
from arena.agents.hymem.messages import clip
from arena.agents.hymem.planner import Planner
from arena.agents.system1.controller import S1Controller, S1Executor, S1Router, facts_block
from arena.agents.system1.decider import FastDecider, FastDecision, RouteConfig, obvious_choice
from arena.agents.system1.facts import build_option_facts, build_state, option_key
from arena.agents.system1.memory_router import (
    MemoryRouter,
    RoutedMemory,
    RouterConfig,
    collect_items,
    memory_version,
)
from arena.llm.typesafe import TypeSafeClient
from arena.parse.validate import validate_or_fallback
from arena.registry.agents import AGENT_REGISTRY, register_agent
from arena.registry.param import Param
from arena.s1.context import s1_meta
from arena.s1.recorder import ExperienceRecorder, RecordingClient
from arena.state.views import legal_options, option_for

S1_PARAMS = [
    Param("jev_model", str, "jev-1.13.0", help="TypeSafe model id (pin a version when tuning thresholds)",
          label="Jev model"),
    Param("min_confidence", float, 0.5, help="act fast only above this Choice confidence", label="Min confidence"),
    Param("min_margin", float, 0.15, help="act fast only when top-1 beats top-2 by this much", label="Min margin"),
    Param("max_hard", float, 0.6, help="escalate when P(hard turn) is above this", label="Max P(hard)"),
    Param("require_consistency", bool, True, help="escalate when reordering the options changes the answer",
          label="Order consistency"),
    Param("debias", bool, True, help="ask with both option orders and average", label="Debias order"),
    Param("ask_hard", bool, True, help="ask the 'hard turn' question", label="Ask hard-turn"),
    Param("s1_timeout_s", float, 8.0, help="timeout of one System-One request", label="S1 timeout (s)"),
    Param("mode", str, "split", choices=["split", "flat"],
          help="question layout: attack-or-switch + best move / switch-in, or one flat Choice", label="Layout"),
    Param("min_branch_confidence", float, 0.3, help="split: |2 P(attack) - 1| needed to act fast",
          label="Min attack/switch confidence"),
    Param("teacher", bool, True, help="use the TypeSafe API (Jev) as the teacher tier", label="Teacher (Jev)"),
    Param("student_url", str, "", help="TypeSafe-compatible local student server, tried before the teacher",
          label="Student URL"),
    Param("student_model", str, "", help="model name sent to the student server", label="Student model"),
    Param("student_min_confidence", float, 0.5, label="Student min confidence"),
    Param("student_min_margin", float, 0.15, label="Student min margin"),
    Param("student_min_branch_confidence", float, 0.3, label="Student min attack/switch confidence"),
    Param("record", bool, True, help="write the System-1 experience log (s1exp_<agent>.jsonl)", label="Record"),
]
TIER0 = Param("tier0", bool, False, help="play a sure first-strike KO without asking any model", label="Tier-0 rule")

_BATTLE: contextvars.ContextVar = contextvars.ContextVar("s1_battle", default=None)


@dataclass
class S1Config:
    jev_model: str = "jev-1.13.0"
    min_confidence: float = 0.5
    min_margin: float = 0.15
    max_hard: float = 0.6
    require_consistency: bool = True
    debias: bool = True
    ask_hard: bool = True
    s1_timeout_s: float = 8.0
    mode: str = "split"
    min_branch_confidence: float = 0.3
    tier0: bool = False
    teacher: bool = True
    student_url: str = ""
    student_model: str = ""
    student_min_confidence: float = 0.5
    student_min_margin: float = 0.15
    student_min_branch_confidence: float = 0.3
    record: bool = True
    # dual-process only
    fast_path: bool = True
    slow_hint: bool = True
    slow_budget_s: float = 0.0
    memory_router: str = "off"
    fast_memory: bool = False
    mem_timeout_s: float = 20.0
    shared_facts: bool = False
    step_control: bool = False
    s1_router: bool = False
    s1_executor: bool = False

    @classmethod
    def pop_from(cls, params: dict) -> "S1Config":
        names = {f.name for f in fields(cls)}
        return cls(**{k: params.pop(k) for k in list(params) if k in names})

    def route_config(self, tier: str = "teacher") -> RouteConfig:
        if tier == "student":
            return RouteConfig(model=self.student_model or "local", min_confidence=self.student_min_confidence,
                               min_margin=self.student_min_margin, max_hard=self.max_hard,
                               require_consistency=self.require_consistency, debias=self.debias,
                               ask_hard=self.ask_hard, mode=self.mode,
                               min_branch_confidence=self.student_min_branch_confidence)
        return RouteConfig(model=self.jev_model, min_confidence=self.min_confidence, min_margin=self.min_margin,
                           max_hard=self.max_hard, require_consistency=self.require_consistency,
                           debias=self.debias, ask_hard=self.ask_hard, mode=self.mode,
                           min_branch_confidence=self.min_branch_confidence)


class S1Stack:
    """The System-1 tiers of one agent: an optional local student and an optional teacher (Jev),
    both recorded. ``decide`` is the cascade: student -> teacher; the first tier whose gate passes
    answers. Every teacher answer on a turn the student was unsure about is a label for the next
    student (active learning comes for free)."""

    def __init__(self, cfg: S1Config, recorder: Optional[ExperienceRecorder], *, teacher_client=None,
                 student_client=None):
        self.cfg = cfg
        self.teacher = None
        if cfg.teacher:
            self.teacher = RecordingClient(teacher_client or TypeSafeClient(model=cfg.jev_model), recorder,
                                           backend=f"typesafe:{cfg.jev_model}", role="teacher")
        self.student = None
        if student_client is not None or cfg.student_url:
            inner = student_client or TypeSafeClient(base_url=cfg.student_url, api_key="local",
                                                     model=cfg.student_model or "local")
            self.student = RecordingClient(inner, recorder, backend=f"local:{cfg.student_model or cfg.student_url}",
                                           role="student")
        if self.teacher is None and self.student is None:
            raise ValueError("System 1 needs a teacher (teacher: true) or a student (student_url)")
        self.deciders = [(tier, FastDecider(client, cfg.route_config(tier)))
                         for tier, client in (("student", self.student), ("teacher", self.teacher)) if client]
        self.primary = self.teacher or self.student          # controller / memory-router client

    async def decide(self, battle, options, *, extra_facts=None) -> tuple[Optional[FastDecision], str, list]:
        """-> (decision of the accepting tier, or the last tier's for hint / fallback; accepting tier
        or ""; [(tier, FastDecision | None, error)])."""
        tried = []
        for tier, decider in self.deciders:
            try:
                with s1_meta(purpose="action", tier=tier):
                    fd = await asyncio.wait_for(decider.decide(battle, options, extra_facts=extra_facts),
                                                timeout=self.cfg.s1_timeout_s)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001  (the next tier, or System 2, decides)
                tried.append((tier, None, f"{type(e).__name__}: {clip(str(e), 160)}"))
                continue
            tried.append((tier, fd, ""))
            if fd.accept:
                return fd, tier, tried
        last = next((fd for _t, fd, _e in reversed(tried) if fd is not None), None)
        return last, "", tried


def _fallback_order(battle: AbstractBattle, how: str) -> BattleOrder:
    if how == "heuristic":
        from poke_env.player.baselines import SimpleHeuristicsPlayer
        return SimpleHeuristicsPlayer.choose_singles_move(battle)[0]
    if how == "max_damage":
        options = legal_options(battle)
        facts = [f for f in build_option_facts(battle, options) if f.exp_damage is not None]
        if facts:
            return max(facts, key=lambda f: f.exp_damage).option.order
    return Player.choose_random_move(battle)


def _request_key(battle: AbstractBattle) -> tuple:
    req = getattr(battle, "_last_request", None) or {}
    return (battle.battle_tag, req.get("rqid") if isinstance(req, dict) else None,
            len(getattr(battle, "_replay_data", []) or []))


def _tier_telemetry(tried: list) -> dict:
    out = {}
    for tier, fd, err in tried:
        if fd is not None:
            out.update({f"{tier}_{k}": v for k, v in fd.telemetry().items()})
        if err:
            out[f"{tier}_error"] = err
    return out


def _key_of(options, order: BattleOrder) -> Optional[str]:
    opt = option_for(options, order)
    return option_key(opt) if opt is not None else None


@register_agent(
    "jev_fast", is_llm=True, needs_model=False, label="System 1 - Jev / student fast decisions",
    params=S1_PARAMS + [TIER0, Param("fallback", str, "random", choices=["random", "max_damage", "heuristic"],
                                     help="order on API error / timeout", label="Fallback"),
                        Param("escalate", str, "none", choices=["none", "heuristic"],
                              help="who plays the turns the gate escalates (a cheap stand-in for System 2)",
                              label="Escalate to")],
)
class JevFastAgent(Agent):
    """Pure System 1: one typed request per decision (student -> teacher cascade when configured)."""

    name = "jev_fast"

    def __init__(self, ctx: Optional[AgentContext] = None, *, client=None, student_client=None,
                 fallback: str = "random", escalate: str = "none", **params):
        super().__init__(ctx)
        self.s1 = S1Config.pop_from(params)
        if params:
            raise TypeError(f"unknown jev_fast params: {sorted(params)}")
        self.fallback = fallback
        self.escalate = escalate
        self.recorder = ExperienceRecorder.beside(self.ctx.logger) if self.s1.record else None
        self.stack = S1Stack(self.s1, self.recorder, teacher_client=client, student_client=student_client)
        self.client = self.stack.primary
        self.n_decisions = 0
        self.n_fallbacks = 0
        self.tokens_in = 0
        self.cost_usd = 0.0
        self.metrics: dict = defaultdict(float)
        self._last: dict = {}
        self._n: dict = defaultdict(int)

    @property
    def agent_name(self) -> str:
        return self.ctx.logger.agent_name if self.ctx.logger is not None else self.name

    async def act(self, battle: AbstractBattle) -> BattleOrder:
        self._n[battle.battle_tag] += 1
        decision_id = f"{battle.battle_tag}:{self._n[battle.battle_tag]}"
        with s1_meta(agent=self.agent_name, battle_tag=battle.battle_tag, decision_id=decision_id, turn=battle.turn):
            return await self._act(battle, decision_id)

    async def _act(self, battle: AbstractBattle, decision_id: str) -> BattleOrder:
        t0, t_wall = time.perf_counter(), time.time()
        options = legal_options(battle)
        key = _request_key(battle)
        fd: Optional[FastDecision] = None
        tried: list = []
        error, note, tier = "", "", ""
        if key in self._last:            # the server rejected our order: play the next-ranked option
            prev, ranked = self._last[key]
            by_msg = {o.order.message: o for o in options}
            order = [by_msg[msg] for msg in ranked if msg in by_msg] + options
            alt = next((o for o in order if o.order.message != prev), None)
            final, fallback = validate_or_fallback(alt.order if alt else None, battle)
            note = "recall"
            self.metrics["recalls"] += 1
        elif len(options) <= 1:
            final, fallback = validate_or_fallback(options[0].order if options else None, battle)
            note = "single_option"
        elif self.s1.tier0 and (sure := obvious_choice(battle, build_option_facts(battle, options))) is not None:
            final, fallback = validate_or_fallback(sure.option.order, battle)
            note = "tier0"
            self.metrics["tier0"] += 1
        else:
            fd, tier, tried = await self.stack.decide(battle, options)
            error = next((e for _t, f, e in tried if f is None and e), "")
            if error and fd is None:
                self.metrics["errors"] += 1
            if fd is not None and not fd.accept and self.escalate == "heuristic":
                final, fallback = _fallback_order(battle, "heuristic"), False
                note = "escalated"
                self.metrics["escalated"] += 1
            elif fd is not None:
                final, fallback = validate_or_fallback(fd.option.order, battle)
            else:
                final, fallback = _fallback_order(battle, self.fallback), True
        self.n_decisions += 1
        self.n_fallbacks += int(fallback)
        ranked = []
        for t, f, _e in tried:
            if f is None:
                continue
            self.metrics[f"{t}_requests"] += 1
            self.metrics[f"{t}_accepted"] += int(f.accept)
            if t == "teacher":
                self.tokens_in += f.input_tokens
                self.cost_usd += f.input_tokens * 0.042e-6
        if fd is not None:
            by_key = {f.key: f.option.order.message for f in fd.facts}
            ranked = [by_key[k] for k, _ in fd.ranked(len(fd.probabilities))]
            m = self.metrics
            m["s1_requests"] += 1
            m["s1_latency_s_sum"] += fd.latency_s
            m["s1_latency_s_max"] = max(m.get("s1_latency_s_max", 0.0), fd.latency_s)
            m["s1_would_accept"] += int(fd.accept)
            m["s1_consistent"] += int(fd.consistent)
        self._last[key] = (final.message, ranked)
        if len(self._last) > 64:
            self._last.pop(next(iter(self._last)))
        latency = time.perf_counter() - t0
        self.metrics["decisions"] += 1
        self.metrics["decision_latency_s_sum"] += latency
        self.metrics["decision_latency_s_max"] = max(self.metrics.get("decision_latency_s_max", 0.0), latency)
        route = note or (f"fast_{tier}" if tier else ("unsure_" + tried[-1][0] if tried else "none"))
        if self.recorder is not None:
            self.recorder.decision(route=route, final_key=_key_of(options, final), fallback=fallback,
                                   s1_keys={t: (f.key if f else None) for t, f, _e in tried},
                                   latency_s=round(latency, 4), n_options=len(options))
        if self.ctx.logger is not None:
            extra = {"latency_s": round(latency, 4), "t_start": round(t_wall, 3), "t_end": round(time.time(), 3),
                     "fallback": fallback, "error": error, "note": note, "n_options": len(options),
                     "route": route, "decision_id": decision_id}
            if fd is not None:
                extra.update(fd.telemetry())
            extra.update(_tier_telemetry(tried))
            opt = option_for(options, final)
            self.ctx.logger.log_turn(
                battle_tag=battle.battle_tag, turn=battle.turn,
                state_text=json.dumps(build_state(battle), ensure_ascii=False) if fd is not None else "",
                prompt="", raw_llm_output=json.dumps(fd.probabilities) if fd is not None else "",
                parsed_order=final.message, fallback_used=fallback,
                note=note or (error and f"error={error}") or (opt.label.split(" -- ")[0] if opt else ""),
                extra=extra)
        return final

    def on_battle_end(self, battle: AbstractBattle) -> None:
        if self.recorder is not None:
            with s1_meta(agent=self.agent_name, battle_tag=battle.battle_tag):
                self.recorder.outcome(won=bool(battle.won), turns=battle.turn,
                                      decisions=self._n.get(battle.battle_tag, 0))
        self._n.pop(battle.battle_tag, None)

    def extra_metrics(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.metrics.items()}


class _DualPlanner(Planner):
    """HyMem planner for the dual-process agent: a fold never runs inside a decision.

    Measured on the Mac: a fold (planner <summary> + three extractions) takes 60-110 s, so a fold
    that fires inside a time-capped System-2 decision is cancelled half-way, the trigger stays armed
    and every later slow turn restarts it -- System 2 never decides. Folds requested while
    deciding are therefore deferred and run in the background right after the decision (one at a
    time); with the memory router on, memory is re-routed after every completed fold."""

    async def fold(self, sess, battle, options, reason: str) -> None:
        if getattr(sess, "deciding", False):
            if not getattr(sess, "pending_fold", ""):
                sess.pending_fold = reason
            return
        await super().fold(sess, battle, options, reason)
        if self.a.mrouter is not None:
            await self.a.reroute(sess, battle)


DUAL_PARAMS = [
    Param("fast_path", bool, True, help="answer confident turns with System 1", label="Fast path"),
    Param("slow_hint", bool, True, help="show the System-1 ranking to the planner", label="Slow hint"),
    Param("slow_budget_s", float, 0.0, help="time cap of System 2 (0 = HyMem's own budget); on timeout the "
          "System-1 answer is played", label="System-2 budget (s)"),
    Param("memory_router", str, "off", choices=["off", "jev"], help="route memory items with System One",
          label="Memory router"),
    Param("fast_memory", bool, False, help="give System 1 the routed memory as known facts", label="Fast memory"),
    Param("mem_timeout_s", float, 20.0, help="timeout of one memory-routing request", label="Memory timeout (s)"),
    Param("shared_facts", bool, False, help="show the planner the option facts System 1 decides from",
          label="Shared facts"),
    Param("step_control", bool, False, help="System One decides whether the planner answers right away",
          label="Step control"),
    Param("s1_router", bool, False, help="System One routes planner tasks (executor vs isolated reasoning)",
          label="S1 task router"),
    Param("s1_executor", bool, False, help="System One plans the executor's tool calls (no executor LLM)",
          label="S1 executor"),
]


@register_agent(
    "llm_hymem_dual", is_llm=True, needs_model=True, label="LLM - HyMem + System 1 dual process",
    params=list(AGENT_REGISTRY["llm_hymem"].params) + S1_PARAMS + DUAL_PARAMS
    + [Param("tier0", bool, True, help="play a sure first-strike KO without asking any model", label="Tier-0 rule")],
)
class HyMemDualAgent(HyMemAgent):
    """System 1 answers confident turns and steers HyMem's machinery; HyMem plans the rest."""

    name = "llm_hymem_dual"

    def __init__(self, ctx: Optional[AgentContext] = None, *, model: str = "qwen3.5-4b-mlx",
                 jev_client=None, student_client=None, **params):
        params.setdefault("tier0", True)
        self.s1 = S1Config.pop_from(params)
        super().__init__(ctx, model=model, **params)
        self.recorder = ExperienceRecorder.beside(self.ctx.logger) if self.s1.record else None
        self.stack = S1Stack(self.s1, self.recorder, teacher_client=jev_client, student_client=student_client)
        self.jev = self.stack.primary
        self.mrouter = (MemoryRouter(self.jev, RouterConfig(model=self.s1.jev_model))
                        if self.s1.memory_router == "jev" else None)
        self.controller = S1Controller(self.jev, self.s1.jev_model)
        self.planner = _DualPlanner(self)
        if self.s1.s1_router:
            self.router = S1Router(self.jev, self.s1.jev_model, battle_fn=_BATTLE.get)
        if self.s1.s1_executor:
            self.executor = S1Executor(self)
        self._hints: dict[str, str] = {}
        self._ranking: dict[str, list] = {}

    @property
    def agent_name(self) -> str:
        return self.ctx.logger.agent_name if self.ctx.logger is not None else self.name

    # --- memory views ------------------------------------------------------------------
    def memory_text(self, sess, space: str) -> str:
        routed: Optional[RoutedMemory] = getattr(sess, "routed", None)
        if self.mrouter is None or routed is None:
            return super().memory_text(sess, space)
        return routed.render(space)

    async def reroute(self, sess, battle) -> None:
        """After a fold the memory was rewritten: route it again before the planner continues."""
        await self._route_memory(sess, battle, collect_items(sess.store), memory_version(sess.store))
        if getattr(sess, "routed", None) is not None:
            self._record_routed(sess, sess.routed)

    def _record_routed(self, sess, routed: RoutedMemory) -> None:
        st = sess.stats.extra
        st.update(routed.stats())
        st["mem_chars_planner"] = len(routed.render("planner"))
        st["mem_chars_planner_rule"] = len(render_memory(sess.store))
        st["mem_chars_executor"] = len(routed.render("executor"))
        st["mem_chars_executor_rule"] = len(render_tool_memory(sess.store))
        if getattr(sess, "mem_error", ""):
            st["mem_error"] = sess.mem_error

    def passive_text(self, battle: AbstractBattle) -> str:
        base = super().passive_text(battle)
        hint = self._hints.get(battle.battle_tag, "")
        shared = facts_block(battle, legal_options(battle)) if (self.s1.shared_facts and hint) else ""
        return "\n".join(x for x in (base, shared, hint) if x)

    # --- System 1 inside the planner -----------------------------------------------------------
    async def answer_now(self, sess, battle, options, step: int) -> bool:
        if not self.s1.step_control or step != 0:
            return False
        try:
            mode, p_answer = await asyncio.wait_for(
                self.controller.step_mode(battle, options, ranking=self._ranking.get(battle.battle_tag)),
                timeout=self.s1.s1_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001  (the planner proceeds normally)
            sess.stats.extra["s1_step_error"] = f"{type(e).__name__}: {clip(str(e), 120)}"
            return False
        sess.stats.extra["s1_step"] = mode
        sess.stats.extra["s1_p_answer_now"] = round(p_answer, 4)
        go = mode == "answer_now" and p_answer >= self.controller.min_answer_now
        self.metrics["step_answer_now"] += int(go)
        return go

    # --- the dual-process decision ----------------------------------------------------------
    async def _route_memory(self, sess, battle, items: list, version: tuple) -> None:
        """One memory-routing request (its own request: at ~140 questions / ~26k tokens it takes
        1-4 s, so riding on the ~0.4 s decision request would slow every fast turn down)."""
        try:
            with s1_meta(purpose="memory"):
                routed = await asyncio.wait_for(self.mrouter.route(build_state(battle), items, version),
                                                timeout=self.s1.mem_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001  (keep the previous / HyMem's own memory view)
            self.metrics["mem_errors"] += 1
            sess.mem_error = f"{type(e).__name__}: {clip(str(e), 120)}"
            return
        sess.routed = routed
        self.metrics["mem_routes"] += 1
        self.metrics["mem_items_sum"] += len(routed.items)
        self.metrics["mem_kept_sum"] += sum(1 for r in routed.items if r.kept)

    def _schedule_fold(self, sess, battle, options) -> None:
        reason = getattr(sess, "pending_fold", "")
        running = getattr(sess, "fold_task", None)
        if not reason or (running is not None and not running.done()):
            return
        sess.pending_fold = ""
        self.metrics["background_folds"] += 1

        async def run():
            t0 = time.perf_counter()
            try:
                with s1_meta(purpose="memory"):
                    await self.planner.fold(sess, battle, options, reason)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001  (memory stays as it was; the trigger asks again later)
                self.metrics["fold_errors"] += 1
            self.metrics["background_fold_s_sum"] += time.perf_counter() - t0
        sess.fold_task = asyncio.ensure_future(run())

    async def _decide(self, sess, battle, options):
        token = _BATTLE.set(battle)
        try:
            with s1_meta(agent=self.agent_name, battle_tag=sess.tag, decision_id=f"{sess.tag}:{sess.decision_idx}",
                         turn=battle.turn):
                return await self._decide_dual(sess, battle, options)
        finally:
            _BATTLE.reset(token)

    async def _decide_dual(self, sess, battle, options):
        c, st, m = self.s1, sess.stats, self.metrics
        if c.tier0:
            sure = obvious_choice(battle, build_option_facts(battle, options))
            if sure is not None:
                st.extra["route"] = "tier0"
                m["tier0_decisions"] += 1
                return sure.option, 1.0, "", ""
        fd: Optional[FastDecision] = None
        if self.mrouter is not None:
            version = memory_version(sess.store)
            routed = getattr(sess, "routed", None)
            pending = getattr(sess, "route_task", None)
            items = collect_items(sess.store)
            if items and (routed is None or routed.version != version) and (pending is None or pending.done()):
                # runs alongside the fast decision; a fast turn leaves it finishing in the background
                sess.route_task = asyncio.ensure_future(self._route_memory(sess, battle, items, version))
        if c.fast_path:
            routed = getattr(sess, "routed", None)
            extra_facts = routed.fast_facts() if (c.fast_memory and routed is not None) else None
            fd, tier, tried = await self.stack.decide(battle, options, extra_facts=extra_facts)
            st.extra.update(_tier_telemetry(tried))
            for t, f, err in tried:
                if f is None:
                    m["s1_errors"] += 1
                    st.extra["s1_error"] = err
                    continue
                m[f"{t}_requests"] += 1
                m[f"{t}_accepted"] += int(f.accept)
            if fd is not None:
                st.extra.update(fd.telemetry())
                m["s1_requests"] += 1
                m["s1_latency_s_sum"] += fd.latency_s
                if tier:
                    st.extra["route"] = f"fast_{tier}" if len(self.stack.deciders) > 1 else "fast"
                    m["fast_decisions"] += 1
                    return fd.option, fd.confidence, "", ""
        pending = getattr(sess, "route_task", None)
        if pending is not None and not pending.done():
            t_wait = time.perf_counter()
            await asyncio.shield(pending)          # System 2 reads the routed memory
            st.extra["mem_wait_s"] = round(time.perf_counter() - t_wait, 3)
        if getattr(sess, "routed", None) is not None:
            self._record_routed(sess, sess.routed)
        st.extra["route"] = "slow"
        m["slow_decisions"] += 1
        if fd is not None and c.slow_hint:
            self._hints[battle.battle_tag] = fd.hint(options)
            self._ranking[battle.battle_tag] = [[k, round(p, 3)] for k, p in fd.ranked(3)]
        budget = c.slow_budget_s or self.cfg.decision_budget_s * 2
        sess.deciding = True
        try:
            opt, conf, raw = await asyncio.wait_for(self.planner.decide(sess, battle, options), timeout=budget)
            if opt is None and fd is not None:
                st.extra["route"] = "slow_failed_fast"
                m["slow_failed_fast"] += 1
                return fd.option, fd.confidence, raw, ""
            return opt, conf, raw, ""
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            m["timeouts"] += 1
            if fd is not None:
                st.extra["route"] = "slow_timeout_fast"
                return fd.option, fd.confidence, "", "slow_timeout"
            return None, None, "", "decision_timeout"
        except Exception as e:  # noqa: BLE001
            m["errors"] += 1
            err = f"{type(e).__name__}: {clip(str(e), 200)}"
            if fd is not None:
                st.extra["route"] = "slow_error_fast"
                return fd.option, fd.confidence, "", err
            return None, None, "", err
        finally:
            sess.deciding = False
            self._hints.pop(battle.battle_tag, None)
            self._ranking.pop(battle.battle_tag, None)
            self._schedule_fold(sess, battle, options)

    def _finish(self, sess, battle, options, final: BattleOrder, fallback: bool, conf, t0: float, **kw) -> BattleOrder:
        out = super()._finish(sess, battle, options, final, fallback, conf, t0, **kw)
        if self.recorder is not None and not kw.get("recall"):
            ex = sess.stats.extra
            route = ex.get("route", kw.get("note") or "")
            final_key = _key_of(options, final)
            with s1_meta(agent=self.agent_name, battle_tag=sess.tag, decision_id=f"{sess.tag}:{sess.decision_idx}",
                         turn=battle.turn):
                self.recorder.decision(route=route, final_key=final_key, fallback=fallback,
                                       s2_key=final_key if route == "slow" else None,
                                       s1_keys={t: ex.get(f"{t}_s1_choice") for t in ("student", "teacher")},
                                       latency_s=round(time.monotonic() - t0, 3), n_options=len(options))
        return out

    def on_battle_end(self, battle: AbstractBattle) -> None:
        sess = self.sessions.get(battle.battle_tag)
        decisions = sess.decision_idx if sess is not None else 0
        super().on_battle_end(battle)
        if self.recorder is not None:
            with s1_meta(agent=self.agent_name, battle_tag=battle.battle_tag):
                self.recorder.outcome(won=bool(battle.won), turns=battle.turn, decisions=decisions)
