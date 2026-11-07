"""Persistent conditional plans whose milestones are evaluated against observed game state."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import operator

from arena.agents.hymem.evidence import parse_answer
from arena.agents.hymem.strategic_state import SETUP, RECOVERY, HAZARDS, REMOVAL, PIVOT


INTENTS = {"chip", "finish", "setup", "recover", "pivot", "set_hazards", "remove_hazards"}
METRICS = {"hp", "alive", "active", "status", "item", "ability", "known_move", "boost_atk", "boost_spa",
           "boost_spe", "boost_def", "tera_available", "hazards", "pokemon_left"}
PROGRESS_METRICS = {"hp", "alive", "active", "status", "boost_atk", "boost_spa", "boost_spe", "boost_def",
                    "tera_available", "hazards", "pokemon_left"}
OPS = {"eq": operator.eq, "ne": operator.ne, "le": operator.le, "ge": operator.ge,
       "lt": operator.lt, "gt": operator.gt, "contains": lambda actual, value: value in actual}

PLANNING = """You are HyMem's persistent strategic planner, working in the background while S1
and search act. Select ONE winning route: a finisher/attrition anchor and one or two preparation phases
that make its endgame possible. Read the entire team, early evidence, previous plan and conditional
experiences. A stall team can win by poison, hazards and recovery; do not invent a setup sweeper.

The execution_catalog lists each member's real capabilities. Choose a route and ONE compatible goal
for each preparation phase, in execution order. Goals are condition IDs from the condition_catalog.
The final phase is automatically added: use the finisher until the opposing team is defeated. Explain
why the preparation enables this finisher. A chip threshold is a target to test, not a guaranteed KO.
Choose an opposing check that actually obstructs this finisher. Do not choose goals already achieved.

The finisher is essential; protected_members are optional additional indispensable resources, not
all helpers. Requirements are persistent own resources that must remain available, never a current
active pair or an enemy that must stay alive. Leave requirements empty unless essential. Conditional
branches are optional temporary execution changes, for example recovery if the actor is at low HP.
Do not add an always-true branch that replaces the main route. Stage progress is tested on actual
observations, while search tests feasibility. Ordinary switches and small HP changes preserve the plan.
Continue an adequate existing plan. Revise after critical resource loss, a completed plan, or evidence
that contradicts the winning route. Cite supplied evidence IDs and mark unconfirmed opponent claims
as uncertainties. Return concise JSON matching the schema.
"""

def catalogs(state):
    from poke_env.battle.move import Move
    routes, conditions = {}, {}

    def add(side, mon, metric, op, value):
        name = f"c{len(conditions)}"
        conditions[name] = {"side": side, "pokemon": mon, "metric": metric, "op": op, "value": value}

    for side in ("own", "opponent"):
        add(side, "", "hazards", "eq", 0)
        add(side, "", "hazards", "ge", 1)
        add(side, "", "tera_available", "eq", True)
        for threshold in (0, 2):
            add(side, "", "pokemon_left", "le", threshold)
        for name, mon in state[side].items():
            for value in (True, False):
                add(side, name, "alive", "eq", value)
            add(side, name, "active", "eq", True)
            add(side, name, "active", "eq", False)
            for threshold in (.35, .6):
                add(side, name, "hp", "ge" if side == "own" else "le", threshold)
                if side == "own":
                    add(side, name, "hp", "le", threshold)
            if side != "own" or not mon["alive"]:
                continue
            moves = set(mon.get("moves", []))
            available = {"pivot"}
            if any(Move(move, gen=9).category.name != "STATUS" for move in moves) or moves & {"toxic", "willowisp", "leechseed"}:
                available.update(("chip", "finish"))
            for intent, known in (("setup", SETUP), ("recover", RECOVERY), ("set_hazards", HAZARDS),
                                  ("remove_hazards", REMOVAL)):
                if moves & known:
                    available.add(intent)
            for intent in sorted(available):
                routes[f"{name}:{intent}"] = {"actor": name, "intent": intent, "known_moves": sorted(moves)}
            if "setup" in available:
                for stat in ("atk", "spa", "spe", "def"):
                    boosting = [mid for mid in moves & SETUP if (Move(mid, gen=9).boosts or {}).get(stat, 0) > 0
                                or ((Move(mid, gen=9).entry.get("self") or {}).get("boosts") or {}).get(stat, 0) > 0]
                    if boosting:
                        add(side, name, "boost_" + stat, "ge", 1)
    for route in routes.values():
        actor, intent = route["actor"], route["intent"]
        def suitable(c):
            if intent in ("chip", "finish"):
                return c["side"] == "opponent" and (c["metric"] == "hp" or
                    (c["metric"] == "alive" and c["value"] is False))
            if intent == "setup":
                return c["side"] == "own" and c["pokemon"] == actor and c["metric"].startswith("boost_")
            if intent == "recover":
                return c["side"] == "own" and c["pokemon"] == actor and c["metric"] == "hp" and c["op"] == "ge"
            if intent == "pivot":
                return c["side"] == "own" and c["pokemon"] == actor and c["metric"] == "active"
            return c["side"] == ("own" if intent == "remove_hazards" else "opponent") and c["metric"] == "hazards" and c["op"] == ("eq" if intent == "remove_hazards" else "ge")
        route["goal_conditions"] = [key for key, c in conditions.items() if suitable(c)]
    routes = {key: route for key, route in routes.items() if route["goal_conditions"]}
    return routes, conditions


def plan_schema(packet):
    state = packet["state"]
    routes, conditions = catalogs(state)
    requirement_ids = [key for key, c in conditions.items() if holds(c, state) is True
                       and c["side"] == "own" and (c["metric"] == "tera_available" or
                           c["metric"] == "alive" and c["value"] is True)]
    def array(items, maximum):
        return {"type": "array", "items": items, "maxItems": maximum}
    def enum(values):
        return {"type": "string", "enum": list(values)}
    def obj(properties):
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    text = {"type": "string", "maxLength": 220}
    branch_conditions = [key for key, c in conditions.items() if c["metric"] in
                         ("hp", "alive", "hazards")]
    branch = obj({"route": enum(routes), "when": {**array(enum(branch_conditions), 2), "minItems": 1}})
    preparations = []
    for key, route in routes.items():
        if route["intent"] == "finish":
            continue
        goals = [goal for goal in route["goal_conditions"] if holds(conditions[goal], state) is not True]
        if goals:
            preparations.append(obj({"route": enum([key]), "goal": enum(goals), "branches": array(branch, 1)}))
    evidence = enum(e["id"] for e in packet["evidence"]) if packet["evidence"] else {"type": "string"}
    plan = obj({"win_condition": text, "finisher": enum(k for k in routes if k.endswith(":finish")),
                "protected_members": array(enum(name for name, p in state["own"].items() if p["alive"]), 1),
                "evidence": array(evidence, 3), "uncertainties": array(text, 2),
                "experience_refs": array(enum(r["id"] for r in packet.get("prior_experiences", [])), 3)
                    if packet.get("prior_experiences") else array({"type": "string"}, 0),
                "requirements": array(enum(requirement_ids), 2),
                "preparation": {**array({"anyOf": preparations}, 2), "minItems": 1}})
    previous = packet.get("previous_plan")
    alternatives = [obj({"operation": enum(["revise" if previous else "create"]), "reason": text, "plan": plan})]
    if previous and previous.get("status") == "active":
        alternatives.append(obj({"operation": enum(["continue"]), "reason": text}))
    return {"type": "json_schema", "json_schema": {"name": "persistent_strategy", "strict": True,
                                                    "schema": {"anyOf": alternatives}}}


def condition_value(condition, state):
    metric = condition["metric"]
    side = condition["side"]
    if metric == "tera_available":
        used = state["tera"][side].get("used")
        return None if used is None else not used
    if metric == "pokemon_left":
        return state["pokemon_left"][side]
    if metric == "hazards":
        return sum(c.get("layers", 1) for c in state["side_conditions"].get(side, [])
                   if c["condition"] in ("stealthrock", "spikes", "toxicspikes", "stickyweb"))
    mon = state[side].get(condition.get("pokemon"))
    if mon is None:
        return None
    if metric == "hp":
        return mon.get("hp") if mon.get("hp_knowledge") == "current" else None
    if metric.startswith("boost_"):
        return mon.get("boosts", {}).get(metric[6:], 0)
    if metric == "known_move":
        return mon["moves"]
    if metric in ("item", "ability"):
        fact = mon.get(metric) or {}
        return fact.get("value") if fact.get("status") == "known" else "" if fact.get("status") == "none" else None
    return mon.get(metric)


def holds(condition, state):
    value = condition_value(condition, state)
    return None if value is None else bool(OPS[condition["op"]](value, condition["value"]))


def all_hold(conditions, state):
    return all(holds(c, state) is True for c in conditions)


def validate_condition(c, state, *, observed=False):
    metric, op = c["metric"], c["op"]
    if metric not in METRICS or op not in OPS:
        raise ValueError("condition metric or operator is not executable")
    side = c["side"]
    if side not in ("own", "opponent"):
        raise ValueError("unknown condition side")
    if metric not in ("tera_available", "hazards", "pokemon_left") and c.get("pokemon") not in state[side]:
        raise ValueError("condition references a Pokemon outside the visible teams")
    numeric = metric in {"hp", "hazards", "pokemon_left"} or metric.startswith("boost_")
    if numeric and (not isinstance(c["value"], (int, float)) or isinstance(c["value"], bool)):
        raise ValueError("numeric condition requires a numeric value")
    if numeric and op == "contains":
        raise ValueError("numeric conditions use comparison operators")
    if metric in ("known_move", "status", "item", "ability") and not isinstance(c["value"], str):
        raise ValueError("categorical condition requires a string")
    if metric == "hp" and not 0 <= c["value"] <= 1:
        raise ValueError("fraction must lie between zero and one")
    if metric in ("alive", "active", "tera_available") and (not isinstance(c["value"], bool) or op not in ("eq", "ne")):
        raise ValueError("boolean condition requires a boolean equality")
    if metric == "known_move" and op != "contains":
        raise ValueError("revealed moves use contains")
    if metric in ("item", "ability", "status") and op not in ("eq", "ne"):
        raise ValueError("categorical facts use equality")


@dataclass
class PersistentPlan:
    id: str
    source_decision: int
    definition: dict
    completed: list[str] = field(default_factory=list)
    status: str = "active"
    changes: list[dict] = field(default_factory=list)
    last_review_decision: int = 0

    def advance(self, state):
        changes = []
        dead = [name for name in self.definition["members"] if not state["own"][name]["alive"]]
        broken = [c for c in self.definition.get("requirements", []) if holds(c, state) is False]
        if dead or broken:
            if self.status != "needs_revision":
                changes.append({"kind": "invalidated", "members_lost": dead, "requirements_broken": broken})
            self.status = "needs_revision"
        else:
            for stage in self.definition["stages"]:
                dependencies = all(s in self.completed for s in stage["depends_on"])
                values = [holds(c, state) for c in stage["complete_when"]]
                maintained = any(c["metric"] != "active" for c in stage["complete_when"])
                if stage["id"] in self.completed and ((maintained and False in values) or not dependencies):
                    self.completed.remove(stage["id"])
                    changes.append({"kind": "reopened", "stage": stage["id"]})
                if dependencies and stage["id"] not in self.completed and all(v is True for v in values):
                    self.completed.append(stage["id"])
                    changes.append({"kind": "completed", "stage": stage["id"]})
            active = self.stage
            if active and (not state["own"][active["actor"]]["alive"]
                           or any(holds(c, state) is True for c in active["invalidate_when"])):
                if self.status != "needs_revision":
                    changes.append({"kind": "invalidated", "stage": active["id"]})
                self.status = "needs_revision"
            elif len(self.completed) == len(self.definition["stages"]):
                self.status = "complete"
            elif self.status == "complete":
                self.status = "active"
        self.changes.extend({"decision": state["decision"], **c} for c in changes)
        return changes

    @property
    def stage(self):
        return next((s for s in self.definition["stages"] if s["id"] not in self.completed
                     and all(d in self.completed for d in s["depends_on"])), None)

    def execution(self, state):
        stage = self.stage
        if self.status != "active" or not stage or not all_hold(stage["when"], state):
            return None
        branch = next((b for b in stage["branches"] if all_hold(b["when"], state)), None)
        return {**deepcopy(stage), **deepcopy(branch or {}), "plan_id": self.id,
                "branch": stage["branches"].index(branch) if branch else None}

    def to_dict(self):
        return {**deepcopy(self.definition), "id": self.id, "source_decision": self.source_decision,
                "last_review_decision": self.last_review_decision, "status": self.status,
                "completed": list(self.completed), "active_stage": (self.stage or {}).get("id"),
                "recent_changes": deepcopy(self.changes[-8:])}


def parse_plan(text, packet, previous=None, sequence=1):
    answer = parse_answer(text) if isinstance(text, str) else deepcopy(text)
    state = packet["state"]
    operation = answer["operation"]
    if operation == "continue":
        if previous is None or previous.status != "active":
            raise ValueError("only an active existing plan can continue")
        plan = deepcopy(previous); plan.last_review_decision = state["decision"]
        return plan, answer
    if operation not in ("create", "revise"):
        raise ValueError("expected create, continue or revise")
    definition = answer["plan"]
    routes, conditions = catalogs(state)
    def expand(values):
        return [deepcopy(conditions[value]) if isinstance(value, str) else value for value in values]
    if "preparation" in definition:
        finisher = routes[definition.pop("finisher")]
        definition["members"] = list(dict.fromkeys([finisher["actor"], *definition.pop("protected_members")]))
        stages = []
        for i, preparation in enumerate(definition.pop("preparation")):
            goal = conditions[preparation["goal"]]
            route = routes[preparation["route"]]
            stages.append({"id": f"prepare_{i + 1}", "objective":
                           f"{route['actor']} {route['intent']}: {goal['side']}.{goal['pokemon'] or 'team'}."
                           f"{goal['metric']} {goal['op']} {goal['value']}",
                           "route": preparation["route"],
                           "target": goal["pokemon"] if goal["side"] == "opponent" else "",
                           "depends_on": [stages[-1]["id"]] if stages else [],
                           "complete_when": [goal], "branches": preparation["branches"]})
        stages.append({"id": "finish", "objective": definition["win_condition"], **finisher,
                       "target": "", "depends_on": [stages[-1]["id"]],
                       "complete_when": [{"side": "opponent", "pokemon": "", "metric": "pokemon_left",
                                          "op": "eq", "value": 0}]})
        definition["strategic_hypothesis"] = definition["win_condition"]
        definition["win_condition"] = "; then ".join(s["objective"] for s in stages[:-1]) + (
            f"; finish with {finisher['actor']}")
        stages[-1]["objective"] = f"Use {finisher['actor']} to defeat the remaining opposing team"
        definition["stages"] = stages
    definition["requirements"] = expand(definition.get("requirements", []))
    for stage in definition["stages"]:
        for route in [stage] + stage.get("branches", []):
            if "route" in route:
                entry = routes[route.pop("route")]
                route.update(actor=entry["actor"], intent=entry["intent"])
            route["when"] = expand(route.get("when", []))
        stage["complete_when"] = expand(stage["complete_when"])
        stage["invalidate_when"] = expand(stage.get("invalidate_when", []))
    if not definition["win_condition"] or not definition["members"]:
        raise ValueError("plan requires a winning route and its essential members")
    if any(name not in state["own"] or not state["own"][name]["alive"] for name in definition["members"]):
        raise ValueError("winning member must be alive on our team")
    supplied = {e["id"] for e in packet["evidence"]}
    if not set(definition.get("evidence", [])).issubset(supplied):
        raise ValueError("plan evidence is absent from the supplied player-visible history")
    if not 1 <= len(definition["stages"]) <= 3:
        raise ValueError("plan needs one to three executable stages")
    definition.setdefault("requirements", [])
    definition.setdefault("uncertainties", [])
    for c in definition["requirements"]:
        validate_condition(c, state, observed=True)
    if not set(definition.get("experience_refs", [])).issubset({r["id"] for r in packet.get("prior_experiences", [])}):
        raise ValueError("experience reference is absent from the retrieved memory")
    prior = set()
    for stage in definition["stages"]:
        stage.setdefault("depends_on", []); stage.setdefault("when", [])
        stage.setdefault("invalidate_when", []); stage.setdefault("branches", [])
        if stage["id"] in prior or not set(stage["depends_on"]).issubset(prior):
            raise ValueError("stage dependencies must refer to earlier distinct stages")
        prior.add(stage["id"])
        if not stage["complete_when"]:
            raise ValueError("stage needs an observable completion condition")
        for c in stage["complete_when"]:
            validate_condition(c, state, observed=True)
            if c["metric"] not in PROGRESS_METRICS:
                raise ValueError("stage progress must have a game-state and engine interpretation")
        for c in stage["when"] + stage["invalidate_when"]:
            validate_condition(c, state)
        for route in [stage] + stage["branches"]:
            if route["actor"] not in state["own"] or route["intent"] not in INTENTS:
                raise ValueError("execution route needs a known actor and supported intent")
            if route.get("target") and route["target"] not in state["opponent"]:
                raise ValueError("execution target is outside the opposing team")
            for c in route.get("when", []):
                validate_condition(c, state)
    plan = PersistentPlan(f"{state['battle']}:plan:{sequence}", state["decision"], definition,
                          last_review_decision=state["decision"])
    plan.advance(state)
    if plan.status == "needs_revision":
        raise ValueError("new plan already violates its prerequisites")
    return plan, answer


def compact_context(packet):
    context = deepcopy(packet)
    context.pop('player_stream', None)
    context['evidence'] = [{k: e[k] for k in ('id', 'turn', 'protocol')} for e in packet['evidence']]
    for row in context.get('prior_experiences', []):
        actions = row['method']['actions']
        row['method'].update(action_count=len(actions), actions=actions[-8:])
        row['exceptions'] = row['exceptions'][-4:]
    for side in ('own', 'opponent'):
        for mon in context['state'][side].values():
            mon.pop('ident', None); mon.pop('pp', None)
    return context


def planner_messages(packet):
    context = compact_context(packet)
    routes, conditions = catalogs(packet["state"])
    context["mechanics"] = mechanics_view(packet["state"])
    context["execution_catalog"] = {key: {"goal_conditions": value["goal_conditions"]}
                                    for key, value in routes.items()}
    context["condition_catalog"] = {key: f"{c['side']}.{c['pokemon'] or 'team'}.{c['metric']} {c['op']} {c['value']}"
                                    for key, c in conditions.items()}
    return [{"role": "system", "content": PLANNING},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False, separators=(",", ":"))}]


def reflection_messages(packet):
    context = compact_context(packet)
    context['mechanics'] = mechanics_view(packet['state'])
    return [{'role': 'system', 'content': (
        'Analyze the whole Pokemon battle, not only the active pair. In at most 180 words compare the living '
        'team members as endgame anchors, choose one winning route, identify its main opposing check and '
        'state one or two preparations that enable it. HP is a fraction: 0.94 is 94%, not low HP. '
        'Use the supplied mechanics as authoritative rules. A type or ability has only its listed effects. '
        'Use only the listed moves and observed facts; unconfirmed opponent details are hypotheses. '
        'Explain why the route is viable; preserve an adequate previous plan. If a key member died, '
        'choose a living alternative. Do not give a move command or JSON.' )},
        {'role': 'user', 'content': json.dumps(context, ensure_ascii=False, separators=(',', ':'))}]


def mechanics_view(state):
    from arena.knowledge.store import dex
    data = dex()
    types, moves, abilities, items = {}, {}, {}, {}
    for side in ("own", "opponent"):
        for name, mon in state[side].items():
            species = data["species"].get(name, {})
            types[name] = {"types": [mon["tera_type"]] if mon.get("tera") and mon.get("tera_type")
                          else species.get("types", []), "base_stats": species.get("baseStats", {})}
            for move in mon["moves"]:
                entry = data["moves"].get(move, {})
                moves[move] = {k: entry[k] for k in ("type", "category", "basePower", "desc") if k in entry}
            for field, target, group in (("ability", abilities, "abilities"), ("item", items, "items")):
                value = mon.get(field) or {}
                if value.get("status") == "known" and value.get("value"):
                    key = value["value"]
                    target[key] = data[group].get(key, {}).get("desc", "unknown")
    return {"pokemon": types, "moves": moves, "abilities": abilities, "items": items}
