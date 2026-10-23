"""System-1 decision: one typed Jev request per battle decision.

Two question layouts (``RouteConfig.mode``), both sent as ONE request (questions are evaluated
in parallel, so extra ones cost tokens, not time):

  flat   best / best_rev   Choice over every legal option (both option orders)
  split  stay              Noul: keep the active Pokemon in and attack, or switch?
         best_move / _rev  Choice over the moves only
         best_switch / _rev  Choice over the switch-ins only
         (code combines them: P(move k) = P(stay) * P(k | move), P(switch j) = (1 - P(stay)) * P(j | switch))
  both   hard              Noul: does this turn hinge on reading the opponent or on a multi-turn plan?

Every Choice is asked in both option orders (Jev leans toward the first option); the two answers
are averaged and their agreement is a routing signal. ``split`` exists because a flat list lets
the five switch-ins outvote the moves (measured: Jev switched on half of all non-forced turns).
``gate`` turns the answer into a routing decision: act now when the debiased answer is confident,
clearly ahead of the runner-up, order-consistent and (split) the attack/switch call is clear;
otherwise hand the decision, with the ranking as a hint, to System 2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle

from arena.agents.system1.facts import OptionFacts, build_option_facts, build_state
from arena.llm import typesafe as ts
from arena.state.observe import hp_pct
from arena.state.views import Option

BEST_INSTRUCTIONS = (
    "You control `you` in this Pokemon battle. Pick the action for this turn that gives `you` the best "
    "chance to win the whole battle. Each option lists facts computed by a damage calculator: type "
    "effectiveness, estimated damage as % of the target's max HP with a knock-out verdict, who moves first, "
    "and for switches how hard the opponent's active Pokemon hits the incoming Pokemon. Knocking the "
    "opponent out before it moves is usually best. Never pick a move that has no effect. Switch when your "
    "active Pokemon is likely to be knocked out (see `threat`) and a teammate takes the hit much better. "
    "Terastallizing can be done once per battle; it is worth it when it secures a knock-out or survival."
)
MOVE_INSTRUCTIONS = (
    "You control `you` in this Pokemon battle and your active Pokemon attacks this turn. Pick the move that "
    "gives `you` the best chance to win the whole battle. Each option lists facts computed by a damage "
    "calculator: type effectiveness, estimated damage as % of the target's max HP with a knock-out verdict, "
    "and who moves first. Knocking the opponent out before it moves is usually best. Never pick a move that has "
    "no effect. Terastallizing can be done once per battle; it is worth it when it secures a knock-out or survival."
)
SWITCH_INSTRUCTIONS = (
    "You control `you` in this Pokemon battle and bring in a teammate this turn. Pick the teammate that gives "
    "`you` the best chance to win the whole battle. Each option lists how hard the opponent's active Pokemon hits "
    "it as it comes in, entry hazard damage, its speed, and how hard it hits back from next turn. Prefer a "
    "teammate that takes little damage and threatens the opponent; avoid one that would be knocked out on entry."
)
STAY_QUESTION = "Should `you` keep the active Pokemon in and attack this turn, rather than switch to a teammate?"
STAY_TRUE = ("Attacking is better: the active Pokemon can deal solid damage or a knock-out before it is knocked "
             "out, or no teammate takes the opponent's attacks clearly better.")
STAY_FALSE = ("Switching is better: the active Pokemon is likely to be knocked out this turn without doing much, "
              "or cannot damage the opponent, and a teammate takes the opponent's attacks much better.")
HARD_INSTRUCTIONS = (
    "Is this a hard turn for `you`, where the best action is not clear from the facts and depends on "
    "predicting whether the opponent will switch or attack, or on a plan over several turns?"
)
HARD_TRUE = ("Several options are close, or the right choice depends on reading the opponent, preserving a "
             "key Pokemon for later, setting up, or scouting.")
HARD_FALSE = ("One option is clearly best from the facts given, for example a safe knock-out or the only move "
              "that does real damage.")


@dataclass
class RouteConfig:
    model: str = "jev-1.13.0"
    min_confidence: float = 0.5    # Choice confidence of the debiased distribution
    min_margin: float = 0.15       # top-1 minus top-2 probability
    max_hard: float = 0.6          # P(hard turn) above which System 2 takes over
    require_consistency: bool = True
    debias: bool = True            # also ask with the options reversed
    ask_hard: bool = True
    mode: str = "split"            # "split" (attack-or-switch, then best move / switch-in) | "flat"
    min_branch_confidence: float = 0.3   # split: |2 P(stay) - 1| needed to act fast


@dataclass
class FastDecision:
    option: Option
    key: str
    probabilities: dict            # debiased (mean of both orders when ``debias``)
    p_fwd: dict
    p_rev: dict
    confidence: float
    margin: float
    consistent: bool
    p_hard: Optional[float]
    accept: bool
    reason: str
    reason_code: str
    latency_s: float
    input_tokens: int
    model: str
    facts: list = field(default_factory=list)
    extra_answers: dict = field(default_factory=dict)   # answers to piggy-backed questions
    branch: str = ""               # split: "move" | "switch" ("" in flat mode)
    p_stay: Optional[float] = None # split: P(attack rather than switch)
    branch_confidence: float = 1.0
    choice_confidence: float = 0.0
    branch_keys: list = field(default_factory=list)     # option keys of the answered Choice, in asked order

    def ranked(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])[:k]

    def hint(self, options: list[Option], k: int = 3) -> str:
        """System-1 suggestion for the System-2 prompt (cf. TypeSafe's skill-suggestion pattern)."""
        by_key = {f.key: f.option for f in self.facts}
        parts = []
        for key, p in self.ranked(k):
            o = by_key.get(key)
            if o is not None:
                parts.append(f"{o.idx} ({o.label.split(' -- ')[0]}) p={p:.2f}")
        hard = f"; P(hard turn)={self.p_hard:.2f}" if self.p_hard is not None else ""
        return (f"[system1_suggestion]\nA fast decision model ranked the legal actions: {', '.join(parts)}"
                f"{hard}. It was not sure enough to act alone ({self.reason}). Use it as a hint and check it; "
                f"ignore it if your own analysis disagrees.")

    def telemetry(self) -> dict:
        return {"s1_model": self.model, "s1_choice": self.key, "s1_confidence": round(self.confidence, 4),
                "s1_margin": round(self.margin, 4), "s1_consistent": self.consistent,
                "s1_p_hard": -1.0 if self.p_hard is None else round(self.p_hard, 4),
                "s1_accept": self.accept, "s1_reason": self.reason_code, "s1_latency_s": round(self.latency_s, 4),
                "s1_input_tokens": self.input_tokens, "s1_n_options": len(self.probabilities),
                "s1_branch": self.branch, "s1_p_stay": -1.0 if self.p_stay is None else round(self.p_stay, 4),
                "s1_branch_confidence": round(self.branch_confidence, 4),
                "s1_choice_confidence": round(self.choice_confidence, 4),
                "s1_top": [[k, round(p, 4)] for k, p in self.ranked(5)]}


def obvious_choice(battle: AbstractBattle, facts: list[OptionFacts]) -> Optional[OptionFacts]:
    """Tier 0 (no model): a sure knock-out before the opponent can act.

    Sure = the whole damage range KOs, the move cannot miss, we act first (speed or priority),
    and the target is already damaged so Focus Sash / Sturdy cannot save it. Spending the tera
    is a strategic call, so tera options never qualify; among several, prefer a move without
    recoil or stat drops."""
    opp = battle.opponent_active_pokemon
    if opp is None or hp_pct(opp) >= 100:
        return None
    sure = []
    for f in facts:
        mv = f.option.order.order if f.option.kind == "move" and not f.option.tera else None
        if mv is None or not f.ko_now or not f.fields.get("damage", "").endswith("guaranteed KO this turn"):
            continue
        order = f.fields.get("order", "")
        if not (mv.accuracy is True or float(mv.accuracy) >= 1.0) or not ("move first" in order or "priority" in order):
            continue
        drawback = bool(getattr(mv, "recoil", 0)) or any(v < 0 for v in (getattr(mv, "self_boost", None) or {}).values())
        sure.append((drawback, -(f.exp_damage or 0.0), f))
    return min(sure, key=lambda t: t[:2])[2] if sure else None


def choice_confidence(probs: dict) -> float:
    n = len(probs)
    if n <= 1:
        return 1.0
    return max(0.0, (max(probs.values()) - 1 / n) / (1 - 1 / n))


def _margin(probs: dict) -> float:
    ps = sorted(probs.values(), reverse=True)
    return ps[0] - (ps[1] if len(ps) > 1 else 0.0) if ps else 0.0


def _normalise(probs: dict, keys: list[str]) -> dict:
    out = {k: max(0.0, float(probs.get(k, 0.0))) for k in keys}
    z = sum(out.values())
    return {k: v / z for k, v in out.items()} if z > 0 else {k: 1 / len(keys) for k in keys}


class FastDecider:
    def __init__(self, client, cfg: Optional[RouteConfig] = None):
        self.client = client
        self.cfg = cfg or RouteConfig()

    def build_request(self, battle: AbstractBattle, options: list[Option], *,
                      extra_facts: Optional[list[str]] = None) -> tuple[dict, dict, list[OptionFacts]]:
        facts = build_option_facts(battle, options)
        state = build_state(battle, extra_facts=extra_facts)
        questions: dict = {}

        def ask(qid: str, instructions: str, group: list[OptionFacts]) -> None:
            if not group:
                return
            criteria = {f.key: f.fields for f in group}
            questions[qid] = ts.choice(instructions, criteria)
            if self.cfg.debias and len(group) > 1:
                questions[qid + "_rev"] = ts.choice(instructions, dict(reversed(list(criteria.items()))))

        if self.cfg.mode == "flat":
            ask("best", BEST_INSTRUCTIONS, facts)
        else:
            moves = [f for f in facts if f.option.kind == "move"]
            switches = [f for f in facts if f.option.kind == "switch"]
            ask("best_move", MOVE_INSTRUCTIONS, moves)
            ask("best_switch", SWITCH_INSTRUCTIONS, switches)
            if moves and switches:
                questions["stay"] = ts.noul(
                    {"question": STAY_QUESTION,
                     "your_attacks": [f.text for f in moves if not f.option.tera],
                     "switch_ins": [f.text for f in switches]},
                    true=STAY_TRUE, false=STAY_FALSE)
        if self.cfg.ask_hard:
            questions["hard"] = ts.noul(HARD_INSTRUCTIONS, true=HARD_TRUE, false=HARD_FALSE)
        return state, questions, facts

    def gate(self, confidence: float, margin: float, consistent: bool,
             p_hard: Optional[float], branch_confidence: float = 1.0) -> tuple[bool, str, str]:
        """-> (act fast?, reason code, human-readable reason)."""
        c = self.cfg
        if c.require_consistency and not consistent:
            return False, "order_inconsistent", "its answer changed when the options were reordered"
        if branch_confidence < c.min_branch_confidence:
            return False, "unclear_switch", f"attack-or-switch call unclear ({branch_confidence:.2f})"
        if confidence < c.min_confidence:
            return False, "low_confidence", f"confidence {confidence:.2f} < {c.min_confidence}"
        if margin < c.min_margin:
            return False, "low_margin", f"margin {margin:.2f} < {c.min_margin}"
        if p_hard is not None and p_hard > c.max_hard:
            return False, "hard_turn", f"P(hard turn) {p_hard:.2f} > {c.max_hard}"
        return True, "confident", "confident"

    @staticmethod
    def _debiased(res, qid: str, keys: list[str]) -> tuple[dict, dict, dict]:
        fwd = _normalise(res.answers[qid].probabilities, keys) if qid in res.answers else \
            {k: 1 / len(keys) for k in keys}
        rev = _normalise(res.answers[qid + "_rev"].probabilities, keys) if qid + "_rev" in res.answers else fwd
        return fwd, rev, {k: (fwd[k] + rev[k]) / 2 for k in keys}

    async def decide(self, battle: AbstractBattle, options: list[Option], *,
                     extra_facts: Optional[list[str]] = None,
                     extra_questions: Optional[dict] = None) -> FastDecision:
        state, questions, facts = self.build_request(battle, options, extra_facts=extra_facts)
        if extra_questions:
            clash = set(extra_questions) & set(questions)
            if clash:
                raise ValueError(f"question ids already used: {sorted(clash)}")
            questions.update(extra_questions)
        t0 = time.perf_counter()
        res = await self.client.system_one(state, questions, model=self.cfg.model)
        latency = res.latency_s or (time.perf_counter() - t0)
        hard = res.answers.get("hard")
        p_hard = hard.noul if hard is not None else None

        branch, p_stay, branch_conf = "", None, 1.0
        if self.cfg.mode == "flat":
            keys = [f.key for f in facts]
            fwd, rev, probs = self._debiased(res, "best", keys)
            joint, branch_keys = probs, keys
        else:
            mkeys = [f.key for f in facts if f.option.kind == "move"]
            skeys = [f.key for f in facts if f.option.kind == "switch"]
            if mkeys and skeys:
                stay = res.answers.get("stay")
                p_stay = stay.noul if stay is not None and stay.noul is not None else 0.5
                branch_conf = abs(2 * p_stay - 1)
            else:
                p_stay = 1.0 if mkeys else 0.0
            mf, mr, pm = self._debiased(res, "best_move", mkeys) if mkeys else ({}, {}, {})
            sf, sr, ps = self._debiased(res, "best_switch", skeys) if skeys else ({}, {}, {})
            joint = {**{k: p_stay * v for k, v in pm.items()}, **{k: (1 - p_stay) * v for k, v in ps.items()}}
            branch = "move" if p_stay >= 0.5 else "switch"
            fwd, rev, probs, branch_keys = (mf, mr, pm, mkeys) if branch == "move" else (sf, sr, ps, skeys)
        key = max(branch_keys, key=lambda k: probs[k])
        consistent = max(branch_keys, key=lambda k: fwd[k]) == max(branch_keys, key=lambda k: rev[k])
        choice_conf, margin = choice_confidence(probs), _margin(probs)
        conf = min(choice_conf, branch_conf)
        accept, code, reason = self.gate(choice_conf, margin, consistent, p_hard, branch_conf)
        option = next(f.option for f in facts if f.key == key)
        extra = {q: a for q, a in res.answers.items() if extra_questions and q in extra_questions}
        return FastDecision(option=option, key=key, probabilities=joint, p_fwd=fwd, p_rev=rev,
                            confidence=conf, margin=margin, consistent=consistent, p_hard=p_hard,
                            accept=accept, reason=reason, reason_code=code, latency_s=latency,
                            input_tokens=res.input_tokens, model=res.model, facts=facts, extra_answers=extra,
                            branch=branch, p_stay=p_stay, branch_confidence=branch_conf,
                            choice_confidence=choice_conf, branch_keys=branch_keys)
