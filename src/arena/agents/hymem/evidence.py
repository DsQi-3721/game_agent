"""HyMem's offline decision-evidence task: planner -> computation -> typed conclusion.

The planner chooses a comparison and allocates a fixed search budget among sampled
states. The executor returns numeric evidence; the final call receives no referee
scores or future battle events. The result can later feed the asynchronous agent.
"""
from __future__ import annotations

import json

from arena.agents.hymem.messages import strip_think


PLANNER = """You are HyMem's Pokemon battle research planner. Identify the consequential disagreement
between a fast student and a search. Ask ONE specific question that a calculation can answer.
Choose 2-3 legal actions to compare, and select 1-3 sample_ids whose further investigation would most
resolve this question. The executor splits the entire additional budget evenly among those IDs.
Choose contrasting or decision-sensitive samples rather than enumerating all samples. The hypothesis
weights stay fixed: allocating computation does not make a hypothesis more likely.
The sampled opposing sets are hypotheses, not revealed facts. State and action facts are player-visible;
damage estimates have assumptions. Search share is visit frequency, not calibrated win probability.
Return the requested JSON. Keep the question and rationale concise."""

ANALYST = """You are HyMem's evidence-based Pokemon decision analyst. Resolve the requested comparison
using the returned search and simulation evidence. Choose the legal action with the strongest supported
whole-battle value. Root search scores estimate continuation value, so a short-term HP gain need not be best.
Student probabilities are a prior, not a veto of a well-supported search result. Per-sample disagreements
show which hypotheses matter; their weights are unchanged by additional search time.
Turn outcomes are conditional on the named opposing action and sampled set, not predictions of the actual
opponent. The executor already ranks the actions by continuation score and computes their gaps.
Use that ordering: rank 1 is the highest score. Prefer its action unless a specific computed mechanism
provides contrary evidence; student confidence alone is not contrary evidence. A score advantage
supports an action, but does not establish WHY it is better. Add a mechanism only when the conditional
turn results explicitly demonstrate it. HP loss is a fraction of team members' maximum HP; positive means
damage and negative means healing. A multi-hit KO estimate is not an immediate KO. Search scores are
not calibrated win probabilities. If the tactical explanation is unknown, say so in unresolved.
Do not invent a continuation beyond the simulated turn. Cite only the evidence you actually use.
The numeric comparison is attached by code; do not copy or recompute numbers in your explanation.
Return a conclusion of at most three sentences, its applicability conditions, and unresolved question as JSON."""


def compact(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def context(case):
    return {"state": case["state"], "legal_action_facts": case["facts"],
            "student_distribution": case["student"], "student_choice": case["student_choice"]}


def plan_messages(case, evidence):
    return [{"role": "system", "content": PLANNER},
            {"role": "user", "content": json.dumps(compact({**context(case), "search": evidence}), ensure_ascii=False)}]


def plan_schema(case, n):
    return {"type": "object", "properties": {
        "question": {"type": "string"}, "rationale": {"type": "string"},
        "candidates": {"type": "array", "items": {"type": "string", "enum": list(case["facts"])},
                       "minItems": 2, "maxItems": 3},
        "sample_ids": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": n - 1},
                       "minItems": 1, "maxItems": min(3, n)}},
        "required": ["question", "rationale", "candidates", "sample_ids"], "additionalProperties": False}


def conclusion_messages(case, plan, tools):
    evidence = tools["evidence"]
    # Keep the executor's factual return bounded by the actual comparison task.
    samples = []
    for s in evidence["samples"]:
        samples.append({"evidence_id": f"sample:{s['sample']}", **s,
                        "policy": {k: v for k, v in s["policy"].items() if k in plan["candidates"]}})
    branches = [{"evidence_id": f"turn:{b['sample']}", **b} for b in tools["branches"]]
    result = {"question": plan["question"], "aggregate": score_table(evidence["policy"], case["facts"]),
              "samples": samples, "conditional_turns": branches}
    return [{"role": "system", "content": ANALYST},
            {"role": "user", "content": json.dumps(compact(context(case)), ensure_ascii=False)
             + "\n<RESULTS>\n" + json.dumps(compact(result), ensure_ascii=False) + "\n</RESULTS>"}]


def score_table(policy, legal):
    ranked = sorted((k for k, v in policy.items() if k in legal and v["score"] is not None),
                    key=lambda k: -policy[k]["score"])
    best = policy[ranked[0]]["score"]
    return [{"rank": i + 1, "action": k, "score": policy[k]["score"],
             "gap_to_best": best - policy[k]["score"], "visits": policy[k]["visits"],
             "share": policy[k]["share"]} for i, k in enumerate(ranked)]


def conclusion_schema(case, tools):
    ids = ["aggregate"] + [f"sample:{s['sample']}" for s in tools["evidence"]["samples"]]
    ids += [f"turn:{s['sample']}" for s in tools["branches"]]
    return {"type": "object", "properties": {
        "action": {"type": "string", "enum": list(case["facts"])},
        "conclusion": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string", "enum": ids}, "minItems": 1},
        "conditions": {"type": "string"}, "unresolved": {"type": "string"}},
        "required": ["action", "conclusion", "evidence", "conditions", "unresolved"], "additionalProperties": False}


def parse_answer(text):
    text = strip_think(text).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)
