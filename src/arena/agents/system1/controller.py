"""System 1 inside System 2: typed decisions that steer HyMem's own machinery.

The fast path answers *what to play*. These components answer the multiple-choice questions
HyMem otherwise settles by free-text generation or fixed rules, each in one typed request:

  shared facts   the code-computed option facts System 1 reads are shown to the planner too,
                 so routine damage / speed / switch-in numbers no longer need an executor trip
  step control   before the first planner step: answer now / research / deliberate? A confident
                 "answer now" turns the step into an answer-only step (one short planner call)
  S1Router       a planner <task>: tool look-up (executor) or deliberation (isolated reasoning)?
                 -- replaces the rule / LLM complexity router
  S1Executor     a planner <task>: which analysis tools, with which arguments? (function calling
                 over closed sets: tool menu + battle entities). Tools run in code; their outputs
                 are distilled by rule into the <RESULTS> block -- no executor LLM turns at all

All requests carry ``s1_meta(purpose=...)`` so the experience log knows what each answer steered.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from arena.agents.hymem.distill import distill
from arena.agents.hymem.messages import ExecResult, clip
from arena.agents.hymem.router import Route
from arena.agents.system1.facts import build_option_facts, build_state
from arena.llm import typesafe as ts
from arena.s1.context import s1_meta
from arena.state.views import Option

if TYPE_CHECKING:
    from arena.agents.hymem.session import BattleSession

STEP_Q = ("The planner of `you` must pick this turn's action from `options`. What should it do before "
          "answering?")
STEP_OPTIONS = {
    "answer_now": "The facts in `options` and `ranking` are enough to choose well; answer now.",
    "research": "A specific calculation or look-up that is not in `options` (for example a teammate's matchup "
                "against the opponent, or the opponent's likely set) would change the choice.",
    "deliberate": "The choice depends on predicting the opponent's next action or on a plan over several turns; "
                  "think it through before answering.",
}
ROUTE_Q = "How should the analysis task in `task` be handled?"
ROUTE_OPTIONS = {
    "lookup": "A calculation or look-up the battle tools answer directly: damage, speed, type matchup, likely "
              "sets, switch-in safety, the battle log.",
    "deliberate": "A judgement that needs several reasoning steps: predicting the opponent, comparing plans over "
                  "several turns, deciding whether to sacrifice or set up.",
}
TOOL_Q = "Does the analyst need the tool in `tool` to answer the task in `task`?"
ARG_Q = "Which value should the tool argument in `argument` take for the task in `task`?"


def facts_block(battle, options: list[Option], *, max_chars: int = 2400) -> str:
    """The option facts System 1 decides from, one line per legal action (planner numbering)."""
    lines = []
    for f in build_option_facts(battle, options):
        body = "; ".join(str(v) for k, v in f.fields.items() if k != "action")
        lines.append(f"{f.option.idx} ({f.option.label.split(' -- ')[0]}): {body}")
    st = build_state(battle)
    head = [x for x in (st.get("speed"), st.get("threat")) if x]
    text = "[system1_facts]\n" + ("\n".join(head) + "\n" if head else "") + "\n".join(lines)
    return clip(text, max_chars)


class S1Controller:
    def __init__(self, client, model: str, *, min_answer_now: float = 0.5):
        self.client = client
        self.model = model
        self.min_answer_now = min_answer_now

    async def step_mode(self, battle, options: list[Option], ranking: Optional[list] = None) -> tuple[str, float]:
        facts = {f.key: f.text for f in build_option_facts(battle, options)}
        state = {**build_state(battle), "options": facts, "ranking": ranking or []}
        with s1_meta(purpose="step_control"):
            res = await self.client.system_one(state, {"step": ts.choice(STEP_Q, STEP_OPTIONS)}, model=self.model)
        a = res.answers["step"]
        top = max(a.probabilities, key=a.probabilities.get) if a.probabilities else "deliberate"
        return top, float(a.probabilities.get("answer_now", 0.0))


class S1Router:
    """Drop-in for ``hymem.router.Router``: ``await route(task, hint_iso=..., llm_call=...)``."""

    def __init__(self, client, model: str, *, battle_fn):
        self.client = client
        self.model = model
        self.battle_fn = battle_fn            # () -> current battle (state for the question)

    async def route(self, task: str, hint_iso: bool = False, llm_call=None) -> Route:
        battle = self.battle_fn()
        state = {**(build_state(battle) if battle is not None else {}), "task": task}
        try:
            with s1_meta(purpose="route_task"):
                res = await self.client.system_one(state, {"route": ts.choice(ROUTE_Q, ROUTE_OPTIONS)},
                                                   model=self.model)
        except Exception:  # noqa: BLE001  (an unanswered routing question defaults to the executor)
            return Route("executor", 0.5, "fallback", "System One unavailable")
        p = res.answers["route"].probabilities
        p_iso = float(p.get("deliberate", 0.0)) + (0.1 if hint_iso else 0.0)
        target = "iso" if p_iso >= 0.5 else "executor"
        return Route(target, max(p_iso, 1 - p_iso), "s1", f"P(deliberate)={p_iso:.2f}")


def _entities(battle) -> dict[str, list[str]]:
    mine = [f"your {p.species}" for p in battle.team.values() if not p.fainted]
    theirs = [f"opp {p.species}" for p in battle.opponent_team.values() if not p.fainted]
    moves = sorted({m.id for p in battle.team.values() for m in p.moves.values()}
                   | {m.id for p in battle.opponent_team.values() for m in p.moves.values()})
    return {"mine": mine, "theirs": theirs, "moves": moves}


# which battle entities a tool argument ranges over (others are left to the tool's default)
ARG_DOMAINS = {"mine": ("mine",), "candidate": ("mine",), "attacker": ("mine", "theirs"),
               "theirs": ("theirs",), "defender": ("theirs", "mine"), "pokemon": ("theirs", "mine"),
               "query": ("mine", "theirs", "moves")}


class S1Executor:
    """Drop-in for ``hymem.executor.Executor``: ``await run(sess, battle, task, parent)``."""

    def __init__(self, agent, *, max_tools: int = 3, min_use: float = 0.5):
        self.a = agent
        self.max_tools = max_tools
        self.min_use = min_use

    def _questions(self, battle, task: str, tools: list) -> tuple[dict, dict]:
        ents = _entities(battle)
        qs, args_of = {}, {}
        for t in tools:
            fn = (t.schema() if callable(t.schema) else t.schema)["function"]
            qs[f"use_{t.name}"] = ts.noul({"question": TOOL_Q, "task": task,
                                           "tool": {"name": fn["name"], "description": fn["description"]}})
            args_of[t.name] = []
            for pname, spec in (fn.get("parameters") or {}).get("properties", {}).items():
                domain = [x for d in ARG_DOMAINS.get(pname, ()) for x in ents[d]]
                if not domain:
                    continue
                crit = {"default": spec.get("description", "the tool's default")}
                crit.update({v: None for v in domain[:200]})
                qid = f"arg_{t.name}__{pname}"
                qs[qid] = ts.choice({"question": ARG_Q, "task": task, "argument": {"tool": fn["name"], "name": pname,
                                                                                    "description": spec.get("description", "")}},
                                    crit)
                args_of[t.name].append((pname, qid))
        return qs, args_of

    async def run(self, sess: "BattleSession", battle, task: str, parent: str = "planner") -> ExecResult:
        a = self.a
        tools = list(a.executor_tools())
        qs, args_of = self._questions(battle, task, tools)
        with s1_meta(purpose="tool_plan"):
            res = await a.jev.system_one(build_state(battle), qs, model=a.s1.jev_model)
        use = sorted(((res.answers[f"use_{t.name}"].noul or 0.0, t) for t in tools if f"use_{t.name}" in res.answers),
                     key=lambda x: -x[0])
        chosen = [t for p, t in use if p >= self.min_use][: self.max_tools]
        if not chosen and use and use[0][0] >= self.min_use / 2:
            chosen = [use[0][1]]
        outputs, raw_obs = [], []
        for t in chosen:
            args = {}
            for pname, qid in args_of.get(t.name, []):
                ans = res.answers.get(qid)
                if ans is not None and ans.choice and ans.choice != "default":
                    args[pname] = ans.choice
            t0, t_wall = time.perf_counter(), time.time()
            try:
                out = await t.call(battle, args)
            except Exception as e:  # noqa: BLE001  (a tool bug must not kill the decision)
                out = f"error: {type(e).__name__}: {e}"
            dt = time.perf_counter() - t0
            ok = bool(out) and not out.startswith("error")
            sess.stats.tool_time_s += dt
            sess.store.tool_calls.append({"tool": t.name, "args": args, "ok": ok, "window": sess.store.folds,
                                          "planned_by": "s1"})
            a.n_tool_calls += 1
            a.log_event(sess, "tool", {"tool": t.name, "t_start": round(t_wall, 3), "latency_s": round(dt, 5),
                                       "ok": ok, "parent": parent, "planned_by": "s1", "args": args})
            obs, _ = await distill(out, task, mode="rule", max_chars=max(240, 700 // max(1, len(chosen))))
            raw_obs.append(f"[{t.name}] {out}")
            outputs.append(f"[{t.name} {', '.join(f'{k}={v}' for k, v in args.items()) or 'defaults'}] {obs}")
        if not outputs:
            return ExecResult(task=task, findings="", status="failure", gaps="no tool fits this task",
                              n_tool_calls=0)
        return ExecResult(task=task, findings="\n".join(outputs), status="success", n_tool_calls=len(outputs),
                          raw_observations=raw_obs)
