"""Fast decision and deep look-ahead running concurrently (``llm_hymem_fused``).

Each decision: the Tier-0 rule; otherwise the local student and a look-ahead search of the same position START
TOGETHER. The search runs in a separate service that mirrors the battle from the messages we forward
(``lookahead.py``; Foul Play's tracking and poke-engine search in Foul Play's own environment). The student answers
in ~0.3 s, the search in ~1 s.

The fast system routes the deep work: on critical turns -- the student's gate rejects, or a forced switch -- the
decision may wait for the search (at most ``wait_s``, within a per-battle ``think_budget_s``); on confident turns
it acts at once and the search finishes in the background (recorded for learning). The two answers are fused
log-linearly over the legal options: log p = (1 - w) log p_student + w log p_search (``fuse_w``).

HyMem's memory keeps running in the background (rule key facts, memory routing) and feeds the student's facts.
With ``async_analysis``, its planner interprets search evidence and schedules conditional computations for
later decisions. ``decision_rule=mean_q`` selects the highest current search value in both experimental arms.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, fields
from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle

from arena.agents.fusion.lookahead import LookaheadClient, team_export
from arena.agents.hymem.strategic_execution import compile_execution
from arena.agents.hymem.coordination import AsyncCoordinator, live_conditions
from arena.agents.system1.agents import HyMemDualAgent, _tier_telemetry
from arena.agents.system1.decider import obvious_choice
from arena.agents.system1.facts import build_option_facts, build_state, option_key
from arena.agents.system1.memory_router import collect_items, memory_version
from arena.knowledge.opponent_model import DEFAULT_MODEL
from arena.observation.capture import CURRENT_REQUEST
from arena.opponent.model import export_beliefs
from arena.registry.agents import AGENT_REGISTRY, register_agent
from arena.registry.param import Param
from arena.s1.context import s1_meta


@dataclass
class FusionConfig:
    lookahead_url: str = "http://127.0.0.1:8104"
    search_ms: int = 700
    samples: int = 8
    fuse_w: float = 0.5
    wait_policy: str = "critical"        # critical | always | never
    wait_s: float = 2.5
    think_budget_s: float = 120.0
    la_beliefs: bool = True              # send our opponent model's beliefs to weight the sets the search samples
    la_belief_mode: str = "reweight"     # reweight | mix | direct (external/foulplay/lookahead_server.py)
    decision_rule: str = "fusion"        # fusion | mean_q
    async_analysis: bool = False         # research ablation: background HyMem on/off
    strategic_planning: bool = False
    strategy_memory_path: str = "runs/hymem_strategy/experiences.jsonl"
    analysis_interval: int = 2           # source decisions between background calls

    @classmethod
    def pop_from(cls, params: dict) -> "FusionConfig":
        return cls(**{f.name: params.pop(f.name) for f in fields(cls) if f.name in params})


FUSION_PARAMS = [
    Param("lookahead_url", str, "http://127.0.0.1:8104", help="look-ahead service (external/foulplay/"
          "lookahead_server.py); empty = student only", label="Look-ahead URL"),
    Param("search_ms", int, 700, help="search time per sampled opponent set", label="Search (ms)"),
    Param("samples", int, 8, help="opponent sets sampled per search", label="Samples"),
    Param("fuse_w", float, 0.5, help="weight of the search in the fused log-probabilities (0 = student only)",
          label="Search weight"),
    Param("wait_policy", str, "critical", choices=["critical", "always", "never"],
          help="when a decision may wait for its search", label="Wait policy"),
    Param("wait_s", float, 2.5, help="longest wait for the search on one decision", label="Wait (s)"),
    Param("think_budget_s", float, 120.0, help="total waiting allowed per battle", label="Think budget (s)"),
    Param("la_beliefs", bool, True, help="weight the opponent sets the search samples by our opponent model "
          "(only with the team model; the plain prior leaves Foul Play's own sampling)", label="Search beliefs"),
    Param("la_belief_mode", str, "reweight", choices=["reweight", "mix", "direct"],
          help="re-weight Foul Play's candidate sets, or draw sets from our beliefs (direct / half: mix)",
          label="Belief mode"),
    Param("decision_rule", str, "fusion", choices=["fusion", "mean_q"],
          help="numeric foreground rule shared by the coordination experiment", label="Decision rule"),
    Param("async_analysis", bool, False, help="HyMem schedules conditional searches in the background",
          label="Asynchronous analysis"),
    Param("strategic_planning", bool, False, help="Persistent HyMem strategy in the background", label="Strategic planning"),
    Param("strategy_memory_path", str, "runs/hymem_strategy/experiences.jsonl", help="Conditional experience store", label="Strategic memory"),
    Param("analysis_interval", int, 2, help="source decisions between background analysis requests",
          label="Analysis interval"),
]


def fuse(p_student: dict, p_search: dict, keys: list[str], w: float, eps: float = 1e-3) -> dict:
    """Log-linear pooling over ``keys``; w = 0 is the student, w = 1 the search."""
    logit = {k: (1 - w) * math.log(p_student.get(k, 0.0) + eps) + w * math.log(p_search.get(k, 0.0) + eps)
             for k in keys}
    top = max(logit.values())
    z = sum(math.exp(v - top) for v in logit.values())
    return {k: math.exp(v - top) / z for k, v in logit.items()}


@register_agent(
    "llm_hymem_fused", is_llm=True, needs_model=True, label="LLM - HyMem + student + look-ahead, concurrent",
    params=list(AGENT_REGISTRY["llm_hymem_dual"].params) + FUSION_PARAMS,
)
class HyMemFusedAgent(HyMemDualAgent):
    name = "llm_hymem_fused"

    def __init__(self, ctx=None, *, lookahead_client=None, dossier_provider=None, **params):
        self.fz = FusionConfig.pop_from(params)
        if self.fz.strategic_planning and self.fz.async_analysis:
            raise ValueError("Use persistent strategy or immediate comparisons for one battle")
        super().__init__(ctx, **params)
        self.la: Optional[LookaheadClient] = lookahead_client or (
            LookaheadClient(self.fz.lookahead_url) if self.fz.lookahead_url else None)
        self._budget: dict[str, float] = {}
        self._started: set[str] = set()
        self._coordinators: dict[str, AsyncCoordinator] = {}
        self._strategy_tasks = set()
        self._strategy_loop = None
        self.dossier_provider = dossier_provider
        if self.fz.strategic_planning:
            from arena.agents.hymem.strategic_memory import StrategicExperienceStore
            self.strategic_memory = StrategicExperienceStore(self.fz.strategy_memory_path)

    def _ingest(self, sess, battle):
        super()._ingest(sess, battle)
        if self.fz.strategic_planning:
            from arena.agents.hymem.strategic_coordinator import StrategicCoordinator
            from arena.agents.hymem.strategic_state import snapshot
            if not hasattr(sess, 'strategy'):
                sess.strategy = StrategicCoordinator(self, sess, self.strategic_memory, battle.player_role)
                self._strategy_loop = asyncio.get_running_loop()
            card = self.dossier_provider(battle) if self.dossier_provider else None
            sess.strategy.boundary(snapshot(battle, sess.decision_idx), battle._replay_data, card)

    async def flush_background(self):
        async def drain():
            while self._strategy_tasks:
                await asyncio.gather(*list(self._strategy_tasks))
        if self._strategy_loop is None:
            return
        if asyncio.get_running_loop() is self._strategy_loop:
            await drain()
        else:
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(drain(), self._strategy_loop))

    # --- the battle stream reaches the look-ahead service before the client acts on it ---------------------
    def on_battle_message(self, battle_tag: str, text: str, player) -> None:
        if self.la is None:
            return
        if battle_tag not in self._started and "|init|battle" in text:
            team = team_export(getattr(player, "_team", None))
            if team:
                self._started.add(battle_tag)
                self.la.start(battle_tag, player.username, team)
        self.la.message(battle_tag, text)

    # --- one decision ------------------------------------------------------------------------------------------
    async def _decide_dual(self, sess, battle, options):
        c, z, st, m = self.s1, self.fz, sess.stats, self.metrics
        if c.tier0:
            sure = obvious_choice(battle, build_option_facts(battle, options))
            if sure is not None:
                st.extra["route"] = "tier0"
                m["tier0_decisions"] += 1
                return sure.option, 1.0, "", ""
        if self.mrouter is not None:          # background memory routing, as in the dual agent
            version = memory_version(sess.store) + (battle.turn,)
            routed, pending = getattr(sess, "routed", None), getattr(sess, "route_task", None)
            items = collect_items(sess.store)
            if items and (routed is None or routed.version != version) and (pending is None or pending.done()):
                sess.route_task = asyncio.ensure_future(self._route_memory(sess, battle, items, version))

        tag, rqid = CURRENT_REQUEST.get((None, None))
        coordinator, case, task = None, None, None
        samples, search_ms, threads = z.samples, z.search_ms, 1
        strategic = getattr(sess, "strategy", None)
        execution = compile_execution(strategic.plan, strategic.state, [option_key(o) for o in options]) if strategic else None
        if execution:
            st.extra.update(strategy_plan_id=execution["plan_id"], strategy_stage=execution["stage"],
                            strategy_candidates=execution["candidate_actions"])
        if z.async_analysis:
            if sess.tag not in self._coordinators:
                self._coordinators[sess.tag] = AsyncCoordinator(self, sess, z.analysis_interval)
            coordinator = self._coordinators[sess.tag]
            task = coordinator.consume(live_conditions(battle), sess.decision_idx,
                                       [option_key(o) for o in options])
            if task is not None:
                st.extra.update(coord_task_id=task["id"], coord_actions=task["actions"],
                                coord_source_decision=task["source_decision"],
                                coord_age_decisions=sess.decision_idx - task["source_decision"])
            state = build_state(battle)
            case = {"decision": sess.decision_idx, "turn": battle.turn, "rqid": rqid, "state": state,
                    "current_conditions": live_conditions(battle),
                    "own_species": [p.species for p in battle.team.values()],
                    "opponent_species": state["opponent"].get("team_preview", [p.species for p in battle.opponent_team.values()]),
                    "legal_actions": [option_key(o) for o in options], "student": {},
                    "recent_feedback": [entry.render() for entry in sess.entries if entry.kind in ("turn", "decision")][-4:]}
        analysis_ready, analysis_sent = False, False

        def offer_analysis(result):
            nonlocal analysis_sent
            if coordinator is not None and analysis_ready and not analysis_sent and result:
                analysis_sent = True
                coordinator.observe(case, result)

        search = None
        if self.la is not None and rqid is not None and tag in self._started:
            beliefs = None
            if z.la_beliefs and DEFAULT_MODEL.name == "team":
                try:
                    beliefs = export_beliefs(battle) or None
                except Exception:  # noqa: BLE001  (no beliefs = Foul Play's own sampling)
                    beliefs = None
                m["la_beliefs"] += int(beliefs is not None)
            detail = ({"evidence": True, "task": task, "threads": threads, "progressive": True}
                      if z.async_analysis or z.decision_rule == "mean_q" or strategic else {})
            if execution:
                detail["strategic_plan"] = execution
            search = asyncio.ensure_future(self.la.analyze(tag, rqid, search_ms, samples, beliefs=beliefs,
                                                           belief_mode=z.la_belief_mode, **detail))
            search.add_done_callback(self._record_search(sess, battle, rqid, offer_analysis, coordinator, task))
            m["searches"] += 1

        routed = getattr(sess, "routed", None)
        extra_facts = routed.fast_facts() if (c.fast_memory and routed is not None) else None
        if execution:
            extra_facts = list(extra_facts or []) + execution["student_facts"]
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
            if search is not None and not search.done():
                self.la.guide(tag, rqid, fd.probabilities)
                st.extra["s1_guidance_sent"] = True
            if case is not None:
                case["student"] = dict(fd.probabilities)
                case["action_facts"] = {f.key: f.fields for f in fd.facts}
        analysis_ready = True
        if search is not None and search.done() and not search.cancelled() and search.exception() is None:
            offer_analysis(search.result())

        critical = fd is None or not fd.accept or bool(getattr(battle, "force_switch", False))
        st.extra["critical"] = critical
        res = None
        if search is not None:
            left = self._budget.setdefault(battle.battle_tag, z.think_budget_s)
            may_wait = z.wait_policy == "always" or (z.wait_policy == "critical" and critical)
            if not search.done() and may_wait and left > 0:
                t_w = time.perf_counter()
                await asyncio.wait({search}, timeout=min(z.wait_s, left))
                waited = time.perf_counter() - t_w
                self._budget[battle.battle_tag] = left - waited
                st.extra["la_wait_s"] = round(waited, 3)
                m["la_wait_s_sum"] += waited
                m["la_waits"] += 1
            if search.done() and not search.cancelled():
                res = search.result()
            elif z.async_analysis or z.decision_rule == "mean_q" or strategic:
                res = self.la.preview(tag, rqid)
            if res is not None:
                offer_analysis(res)

        keys = [option_key(o) for o in options]
        by_key = {option_key(o): o for o in options}
        p_s = dict(fd.probabilities) if fd is not None else {}
        p_l = {k: v["share"] for k, v in ((res or {}).get("policy") or {}).items()}
        if not p_l:
            if fd is None:
                return None, None, "", "no student answer and no look-ahead"
            st.extra["route"] = "fast"
            m["fast_decisions"] += 1
            return fd.option, fd.confidence, "", ""
        if z.decision_rule == "mean_q":
            values = {k: v["score"] for k, v in res["policy"].items()
                      if k in by_key and v["score"] is not None
                      and v.get("min_sample_visits", 16) >= 16 and v.get("coverage", 1) >= .999}
            if values:
                best = max(values, key=values.get)
                st.extra.update(route="search_value", la_value=values[best], la_top=max(p_l, key=p_l.get),
                                la_ms=res.get("ms"), la_compute=res.get("compute"),
                                la_stage=res.get("stage", "complete"),
                                coord_comparison_used=bool(res.get("comparison")),
                                la_value_top=sorted(values.items(), key=lambda kv: -kv[1])[:5])
                m["value_decisions"] += 1
                m["value_flips"] += int(fd is not None and best != fd.key)
                return by_key[best], None, "", ""
        fused = fuse(p_s, p_l, keys, z.fuse_w if fd is not None else 1.0)
        best = max(fused, key=fused.get)
        flip = fd is not None and best != fd.key
        st.extra.update({"route": "fused_flip" if flip else ("fused" if fd is not None else "search_only"),
                         "la_top": max(p_l, key=p_l.get), "la_ms": res.get("ms"), "fused_p": round(fused[best], 4),
                         "la_policy": {k: round(v, 4) for k, v in sorted(p_l.items(), key=lambda kv: -kv[1])[:5]}})
        m["fused_decisions"] += 1
        m["fused_flips"] += int(flip)
        return by_key[best], fused[best], "", ""

    def _record_search(self, sess, battle, rqid, offer_analysis=None, coordinator=None, task=None):
        """The search result for learning, also when it arrives after the decision."""
        decision_id = f"{sess.tag}:{sess.decision_idx}"
        turn = battle.turn
        decision = sess.decision_idx

        def done(fut: asyncio.Future) -> None:
            if fut.cancelled() or fut.exception() is not None:
                if coordinator is not None:
                    coordinator.complete(task, None, decision)
                return
            res = fut.result()
            if coordinator is not None:
                coordinator.complete(task, res, decision)
            if not res:
                self.metrics["search_failures"] += 1
                return
            self.metrics["search_latency_s_sum"] += res.get("latency_s", 0.0)
            strategy = getattr(sess, "strategy", None)
            if strategy is not None:
                strategy.search_result(decision, res)
            if offer_analysis is not None:
                offer_analysis(res)
            if self.recorder is not None:
                with s1_meta(agent=self.agent_name, battle_tag=sess.tag, decision_id=decision_id, turn=turn):
                    self.recorder.write({"type": "lookahead", "agent": self.agent_name, "battle_tag": sess.tag,
                                         "decision_id": decision_id, "rqid": rqid, "turn": turn,
                                         "policy": res.get("policy"), "opp_policy": res.get("opp_policy"),
                                         "ms": res.get("ms"), "latency_s": res.get("latency_s"),
                                         "compute": res.get("compute"), "task": res.get("task"),
                                         "stage": res.get("stage", "complete"), "comparison": res.get("comparison"),
                                         "evidence": res.get("evidence"), "tree": res.get("tree"),
                                         "strategic_plan": res.get("strategic_plan"),
                                         "plan_progress": res.get("plan_progress")})
        return done

    def _finish(self, sess, battle, options, final, fallback, conf, t0, **kw):
        if self.fz.strategic_planning and not kw.get('recall'):
            key = next((option_key(o) for o in options if o.order.message == final.message), None)
            if key:
                sess.strategy.action(key)
        out = super()._finish(sess, battle, options, final, fallback, conf, t0, **kw)
        tag, rqid = CURRENT_REQUEST.get((None, None))
        if self.la is not None and rqid is not None and tag in self._started:
            key = next((option_key(o) for o in options if o.order.message == out.message), None)
            if key:
                self.la.action(tag, rqid, key)
        return out

    def on_battle_end(self, battle: AbstractBattle) -> None:
        coordinator = self._coordinators.pop(battle.battle_tag, None)
        if coordinator is not None:
            coordinator.close()
        sess = self.sessions.get(battle.battle_tag)
        super().on_battle_end(battle)
        if self.fz.strategic_planning and sess is not None:
            sess.strategy.close(battle.won)
        if self.la is not None:
            self.la.close(battle.battle_tag)
        self._started.discard(battle.battle_tag)
        self._budget.pop(battle.battle_tag, None)

    def extra_metrics(self) -> dict:
        out = super().extra_metrics()
        if self.la is not None:
            out["lookahead_client_errors"] = self.la.errors
        return out
