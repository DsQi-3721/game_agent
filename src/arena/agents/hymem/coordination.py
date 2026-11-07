"""HyMem's asynchronous planner for conditional search tasks.

A task changes how a future position is computed, not which old action is played.
Completed tasks are published only when the foreground starts another decision.
"""
from __future__ import annotations

import asyncio
from collections import Counter, deque
from copy import deepcopy
import json
import time

from arena.agents.hymem.evidence import compact, parse_answer
from arena.agents.hymem.messages import Entry, ExecResult
from arena.agents.hymem.session import DecisionStats, stats_scope


PLANNER = """You are HyMem's background battle analyst. The student and search keep playing while
 you reason. Schedule at most TWO concrete action comparisons for the CURRENT active matchup.
 Each task must name 2 or 3 distinct actions from legal_actions, using their exact keys.
 The executor samples the CURRENT position, searches all legal actions, then spends the remaining
 budget comparing your actions plus the fresh search leader. All sampled configurations keep their
 original weights. The engine returns measured action scores and pairwise differences.
 Ask which action or short continuation is better, not whether an unobserved item or move is true.
 The executor does not change hidden sets or force a hypothetical opponent action.

 Tasks may run during the next four decisions. They are bound to the current active pair, surviving
 counts, boosts, status, field and side conditions. Choose our_hp and opponent_hp intervals in [0,1]
 that make your question relevant after the next turn. For recovery questions, use an appropriate
 HP upper bound; for a matchup comparison independent of HP, use [0,1].
 Read recent_comparisons and recent_feedback. Each task executes ONCE and is then completed.
 A new task list replaces all queued tasks; an empty list cancels them. Propose a new comparison
 only when the evidence still leaves a consequential choice unresolved. Order tasks by importance.
 Search scores are estimates, not calibrated win probabilities or proof of a guaranteed knockout.
 Return only JSON. Keep finding and each question to one short sentence:
 {"finding":"...","tasks":[{"question":"Compare ...","actions":["move_x","switch_y"],
 "our_hp":[0,1],"opponent_hp":[0,1]}]}.
"""


def live_conditions(battle):
    def names(values):
        return sorted(getattr(k, "name", str(k)) for k in (values or {}))

    def mon(p):
        return {"status": getattr(getattr(p, "status", None), "name", None),
                "boosts": {k: v for k, v in (getattr(p, "boosts", {}) or {}).items() if v},
                "effects": names(getattr(p, "effects", {})),
                "tera": bool(getattr(p, "is_terastallized", False))}

    def side(values):
        return {getattr(k, "name", str(k)): v if getattr(k, "name", str(k)) in
                ("SPIKES", "TOXIC_SPIKES") else True for k, v in (values or {}).items()}

    ours, opposing = battle.active_pokemon, battle.opponent_active_pokemon
    return {"our": getattr(ours, "species", "") or "",
            "opponent": getattr(opposing, "species", "") or "",
            "our_left": sum(not p.fainted for p in battle.team.values()),
            "opponent_left": 6 - sum(p.fainted for p in battle.opponent_team.values()),
            "our_hp": float(getattr(ours, "current_hp_fraction", 0) or 0),
            "opponent_hp": float(getattr(opposing, "current_hp_fraction", 0) or 0),
            "context": {"our": mon(ours), "opponent": mon(opposing),
                        "weather": names(getattr(battle, "weather", {})),
                        "field": names(getattr(battle, "fields", {})),
                        "our_side": side(getattr(battle, "side_conditions", {})),
                        "opponent_side": side(getattr(battle, "opponent_side_conditions", {}))}}


def task_validity(task, position, decision, legal_actions):
    if decision > task["expires_decision"]:
        return "expired"
    anchor = task["anchor"]
    if any(anchor[k] != position[k] for k in ("our", "opponent", "our_left", "opponent_left", "context")):
        return "context_changed"
    if not set(task["actions"]).issubset(legal_actions):
        return "actions_unavailable"
    if any(not task[k][0] <= position[k] <= task[k][1] for k in ("our_hp", "opponent_hp")):
        return "waiting_for_hp"
    return "ready"


def planner_messages(case, search, tasks):
    policy = search.get("policy", {})
    ranked = sorted((k for k in policy if policy[k]["score"] is not None),
                    key=lambda k: -policy[k]["score"])
    samples = []
    best_actions, top_scores = Counter(), []
    for sample in search.get("evidence", []):
        values = {k: v["score"] for k, v in sample["policy"].items() if v["score"] is not None}
        if values:
            best_actions[max(values, key=values.get)] += 1
        if ranked and ranked[0] in values:
            top_scores.append(values[ranked[0]])
        samples.append({**sample, "policy": dict(sorted(sample["policy"].items(),
                       key=lambda kv: -(kv[1]["score"] or 0))[:3])})
    result = {"ranked_actions": [{"action": k, **policy[k]} for k in ranked[:6]],
              "sample_best_action_counts": dict(best_actions),
              "top_action_score_range": [min(top_scores), max(top_scores)] if top_scores else None,
              "top_two_mean_score_gap": policy[ranked[0]]["score"] - policy[ranked[1]]["score"]
                                        if len(ranked) > 1 else None,
              "sampled_hypotheses": samples, "executed_task": search.get("task"),
              "compute": search.get("compute"), "comparison": search.get("comparison")}
    candidates = set(ranked[:4]) | set(sorted(case.get("student", {}), key=case.get("student", {}).get,
                                            reverse=True)[:2])
    context = {**case, "action_facts": {k: v for k, v in case.get("action_facts", {}).items() if k in candidates},
               "current_tasks": tasks}
    return [{"role": "system", "content": PLANNER},
            {"role": "user", "content": json.dumps(compact(context), ensure_ascii=False)
             + "\n<RESULTS>\n" + json.dumps(compact(result), ensure_ascii=False) + "\n</RESULTS>"}]


class AsyncCoordinator:
    def __init__(self, agent, sess, interval=2):
        self.agent, self.sess, self.interval = agent, sess, interval
        self.pending = None
        self.worker = None
        self.ready = deque()
        self.tasks = deque()
        self.running = {}
        self.feedback = deque(maxlen=4)
        self.last_started = -interval
        self.sequence = 0
        self.closed = False

    def observe(self, case, search):
        if self.closed or not search or not search.get("policy"):
            return
        case = {**case, "recent_comparisons": list(self.feedback)}
        self.pending = (deepcopy(case), deepcopy(search))
        if self.worker is None or self.worker.done():
            if case["decision"] - self.last_started >= self.interval:
                self.worker = asyncio.create_task(self._run())

    async def _run(self):
        self.sess.in_flight += 1
        try:
            while self.pending is not None and not self.closed:
                case, search = self.pending
                if case["decision"] - self.last_started < self.interval:
                    break
                self.pending = None
                self.last_started = case["decision"]
                self.sequence += 1
                ident = f"{self.sess.tag}:analysis:{self.sequence}"
                started = time.time()
                error, answer = None, None
                try:
                    with stats_scope(DecisionStats(t_start=started)):
                        response = await self.agent.call("planner", planner_messages(case, search, list(self.tasks)),
                                                         sess=self.sess)
                        text = response.text.strip()
                        if text.endswith("`") and not text.endswith("```"):
                            text = text[:-1].rstrip()
                        answer = parse_answer(text)
                        if len(answer["tasks"]) > 2:
                            raise ValueError("expected at most two comparisons")
                        for task in answer["tasks"]:
                            actions = task["actions"]
                            if (not 2 <= len(actions) <= 3 or len(set(actions)) != len(actions)
                                    or not set(actions).issubset(case["legal_actions"])):
                                raise ValueError("comparison needs 2 or 3 distinct legal actions")
                            for key in ("our_hp", "opponent_hp"):
                                lo, hi = task[key]
                                if not 0 <= lo <= hi <= 1:
                                    raise ValueError("HP condition must be within [0,1]")
                    self.ready.append({"id": ident, "source_decision": case["decision"],
                                       "source_turn": case["turn"], "anchor": case["current_conditions"],
                                       "ready_at": time.time(), **answer})
                    self.agent.metrics["coord_completed"] += 1
                except asyncio.CancelledError:
                    raise
                except (ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
                    self.agent.metrics["coord_parse_errors"] += 1
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    self.agent.metrics["coord_call_errors"] += 1
                self.agent.log_event(self.sess, "coord_analysis", {
                    "id": ident, "source_decision": case["decision"], "source_turn": case["turn"],
                    "started_at": started, "completed_at": time.time(), "answer": answer, "error": error})
        finally:
            self.agent.sessions.release(self.sess)

    def _event(self, task, status, decision):
        self.agent.log_event(self.sess, "coord_task", {
            "id": task["id"], "status": status, "decision": decision})

    def consume(self, position, decision, legal_actions):
        while self.ready:
            publication = self.ready.popleft()
            for task in self.tasks:
                self._event(task, "replaced", decision)
            self.tasks.clear()
            for number, task in enumerate(publication["tasks"]):
                self.tasks.append({**task, "id": f"{publication['id']}:{number}",
                                   "source_decision": publication["source_decision"],
                                   "source_turn": publication["source_turn"], "ready_at": publication["ready_at"],
                                   "anchor": publication["anchor"],
                                   "expires_decision": publication["source_decision"] + 4})
            result = ExecResult(task="compare current legal actions", findings=publication["finding"],
                                evidence=json.dumps(publication["tasks"], ensure_ascii=False), sources=publication["id"])
            self.sess.add(Entry("results", result.render(), decision, publication["source_turn"]))
            self.agent.log_event(self.sess, "coord_published", {
                "id": publication["id"], "decision": decision, "tasks": len(self.tasks)})
        chosen, remaining = None, deque()
        for task in self.tasks:
            status = task_validity(task, position, decision, legal_actions)
            if status in ("expired", "context_changed", "actions_unavailable"):
                self._event(task, status, decision)
            elif chosen is None and status == "ready":
                chosen = task
            else:
                remaining.append(task)
        self.tasks = remaining
        if chosen is not None:
            self.running[chosen["id"]] = chosen
            self.agent.metrics["coord_used"] += 1
            self.agent.log_event(self.sess, "coord_used", {
                "task": chosen, "decision": decision, "conditions": position,
                "age_decisions": decision - chosen["source_decision"],
                "ready_to_use_s": time.time() - chosen["ready_at"]})
        return chosen

    def complete(self, task, result, decision):
        if task is None or self.running.pop(task["id"], None) is None:
            return
        comparison = (result or {}).get("comparison")
        status = "completed" if comparison else "not_completed"
        feedback = {"id": task["id"], "question": task["question"], "actions": task["actions"],
                    "decision": decision, "status": status, "comparison": comparison}
        self.feedback.append(feedback)
        self.agent.log_event(self.sess, "coord_result", feedback)

    def close(self):
        self.closed = True
        self.pending = None
        self.tasks.clear()
        self.running.clear()
        if self.worker is not None and not self.worker.done():
            self.worker.cancel()
