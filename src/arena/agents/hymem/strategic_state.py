"""Player-visible strategic state, causal event archive and goal-directed retrieval."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field

from poke_env.data import to_id_str

from arena.observation.builder import observe


SETUP = {"swordsdance", "nastyplot", "calmmind", "dragondance", "quiverdance", "bulkup",
         "irondefense", "shellsmash", "agility", "shiftgear", "tidyup", "curse", "bellydrum",
         "torchsong", "takeheart", "coil", "victorydance", "clangoroussoul", "tailglow"}
RECOVERY = {"recover", "roost", "slackoff", "softboiled", "wish", "rest", "synthesis", "shoreup"}
HAZARDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb", "ceaselessedge", "stoneaxe"}
REMOVAL = {"rapidspin", "defog", "mortalspin", "tidyup", "courtchange"}
PIVOT = {"uturn", "voltswitch", "flipturn", "chillyreception", "partingshot", "teleport"}
PRIORITY = {"suckerpunch", "extremespeed", "bulletpunch", "aquajet", "iceshard", "machpunch",
            "thunderclap", "grassyglide", "jetpunch", "shadowsneak"}
EVENT_KINDS = {"move", "switch", "drag", "replace", "-damage", "-heal", "-sethp", "faint",
               "-item", "-enditem", "-ability", "-terastallize", "-status", "-curestatus",
               "-boost", "-unboost", "-clearboost", "-clearallboost", "-sidestart", "-sideend",
               "-weather", "-fieldstart", "-fieldend", "-immune", "-fail", "-miss", "win", "tie"}
KEY_KINDS = EVENT_KINDS - {"switch", "drag", "-damage", "-heal", "-sethp", "-boost", "-unboost"}


def _mon(row, own):
    hp = row.get("hp", [0, 0])
    fraction = (hp[0] / hp[1] if hp[1] else (0.0 if row.get("fainted") else None)) if own else (
        row["hp_pct"] / 100 if row.get("hp_pct") is not None else None)
    moves = row.get("moves" if own else "revealed_moves", [])
    ability = row.get("ability")
    return {"species": row["species"], "ident": row["ident"], "active": row["active"],
            "alive": not row["fainted"], "hp": fraction,
            "hp_knowledge": "current" if own else row.get("hp_knowledge", "unknown"),
            "moves": [m["id"] for m in moves], "pp": {m["id"]: m["pp"] for m in moves if "pp" in m},
            "status": row.get("status"), "boosts": row.get("boosts", {}), "item": row.get("item"),
            "ability": ability if isinstance(ability, dict) else {"status": "known" if ability else "unknown", "value": ability},
            "tera": row.get("terastallized"), "tera_type": row.get("tera_type")}


def snapshot(battle, decision):
    obs = observe(battle)
    own = {p["species"]: _mon(p, True) for p in obs.own_team}
    opposing = {p["species"]: _mon(p, False) for p in obs.opponent_revealed_team}
    for entry in obs.opponent_preview:
        name = to_id_str(entry["details"].split(",")[0])
        opposing.setdefault(name, {"species": name, "active": False, "alive": True, "hp": None,
                                   "hp_knowledge": "unknown", "moves": [], "boosts": {},
                                   "item": {"status": "unknown"}, "ability": {"status": "unknown"}})
    return {"battle": battle.battle_tag, "decision": decision, "turn": battle.turn,
            "rqid": obs.request_id, "own": own, "opponent": opposing,
            "tera": deepcopy(obs.tera), "pokemon_left": dict(obs.pokemon_left),
            "side_conditions": deepcopy(obs.side_conditions), "weather": deepcopy(obs.weather),
            "field": deepcopy(obs.field_conditions), "event_end": obs.event_sequence}


def profile_view(card):
    """Read the existing joint dossier without promoting inferred sets or tendencies to observations."""
    if not card:
        return {"observed": [], "inferred": [], "unknown": ["personal opponent tendencies"]}
    observed = [{"species": p["species"], "facts": deepcopy(p.get("confirmed", {}))}
                for p in card.get("team", []) if p.get("confirmed")]
    inferred, unknown = [], []
    if card.get("prediction_valid"):
        for key in ("behavior_profile", "current_action_tendencies", "strategy_change"):
            if card.get(key):
                inferred.append({"kind": key, "value": deepcopy(card[key]),
                                 "source": "opponent_dossier", "games_observed": card.get("games_observed")})
        for member in card.get("team", []):
            if member.get("hidden_configuration"):
                inferred.append({"species": member["species"], "kind": "configuration_hypotheses",
                                 "value": deepcopy(member["hidden_configuration"])})
    else:
        unknown.append(card.get("unavailable_reason") or "opponent prediction unavailable")
    return {"observed": observed, "inferred": inferred, "unknown": unknown}


@dataclass
class StrategicHistory:
    battle: str
    player_side: str
    cursor: int = 0
    turn: int = 0
    events: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    _causes: list[str] = field(default_factory=list)
    _species: dict[str, str] = field(default_factory=dict)

    def ingest(self, prefix):
        added = []
        for index in range(self.cursor, len(prefix)):
            parts = prefix[index].split("|") if isinstance(prefix[index], str) else list(prefix[index])
            if len(parts) < 2:
                continue
            kind = parts[1]
            if kind == "turn":
                self.turn = int(parts[2]); self._causes = []
                continue
            if kind not in EVENT_KINDS:
                continue
            ident = f"{self.battle}:event:{index}"
            actor = parts[2] if len(parts) > 2 else ""
            subject = to_id_str(actor.split(":", 1)[-1]) if actor.startswith(("p1", "p2")) else ""
            if subject:
                pokemon_ident = actor[:2] + ':' + subject
                if kind in ('switch', 'drag', 'replace') and len(parts) > 3:
                    self._species[pokemon_ident] = to_id_str(parts[3].split(',')[0])
                subject = self._species.get(pokemon_ident, subject)
            event = {"id": ident, "sequence": index, "turn": self.turn, "kind": kind,
                     "side": ("own" if actor.startswith(self.player_side) else "opponent")
                             if actor.startswith(("p1", "p2")) else "field",
                     "subject": subject, "protocol": "|".join(parts), "causal_context": list(self._causes)}
            if kind in ("move", "switch", "drag"):
                self._causes.append(ident)
            self.events.append(event); added.append(event)
        self.cursor = len(prefix)
        return added

    def record_action(self, state, action, plan_id=None, phase=None, expected=None):
        self.decisions.append({"decision": state["decision"], "turn": state["turn"], "action": action,
                               "plan_id": plan_id, "phase": phase, "before": deepcopy(state),
                               "expected": deepcopy(expected)})

    def retrieve(self, subjects=(), evidence_ids=(), limit=32):
        subjects, evidence_ids = {to_id_str(s) for s in subjects if s}, set(evidence_ids)
        recent = self.events[-12:]
        anchored = [e for e in self.events if e["id"] in evidence_ids]
        ranked = []
        seen_facts = set()
        for e in self.events:
            relevant = e['subject'] in subjects or any(s in to_id_str(e["protocol"]) for s in subjects)
            if e["kind"] not in KEY_KINDS and not relevant:
                continue
            fact = (e["kind"], e["subject"], e["protocol"])
            if fact in seen_facts:
                continue
            seen_facts.add(fact)
            priority = 3 if relevant else 2 if e["kind"] in ("faint", "-terastallize", "-enditem", "-immune") else 1
            ranked.append((priority, e))
        ranked.sort(key=lambda item: (-item[0], item[1]["sequence"]))
        chosen = {e["id"]: e for e in anchored}
        counterexamples = [e for e in self.events if e["kind"] in ("-immune", "-fail", "faint", "-enditem")
                           and (e['subject'] in subjects or any(s in to_id_str(e["protocol"]) for s in subjects))]
        chosen.update({e["id"]: e for e in counterexamples[-8:]})
        for _, event in ranked:
            if len(chosen) >= limit:
                break
            chosen[event["id"]] = event
        for event in recent:
            chosen[event["id"]] = event
        # The context of a failure/damage observation remains retrievable with that observation.
        by_id = {e["id"]: e for e in self.events}
        for event in list(chosen.values()):
            for cause in event["causal_context"]:
                if cause in by_id:
                    chosen[cause] = by_id[cause]
        return sorted(chosen.values(), key=lambda e: e["sequence"])

    def package(self, state, *, plan=None, card=None, memories=(), hymem=None):
        plan = plan or {}
        subjects = list(plan.get("members", []))
        for stage in plan.get("stages", []):
            subjects.extend([stage.get("actor", ""), stage.get("target", "")])
        evidence = self.retrieve(subjects, plan.get("evidence", []))
        routes = [{"member": name, "setup": sorted(set(p["moves"]) & SETUP),
                   "priority": sorted(set(p["moves"]) & PRIORITY), "knowledge": "candidate_to_evaluate"}
                  for name, p in state["own"].items() if p["alive"]]
        threats = [{"member": name, "revealed_moves": p["moves"], "hp": p["hp"],
                    "hp_knowledge": p["hp_knowledge"]} for name, p in state["opponent"].items() if p["alive"]]
        return {"state": deepcopy(state), "evidence": deepcopy(evidence), "archive_events": len(self.events),
                "winning_route_candidates": routes, "threats": threats, "opponent_profile": profile_view(card),
                "prior_experiences": deepcopy(list(memories)), "previous_plan": deepcopy(plan),
                "hymem": deepcopy(hymem or {}),
                "open_questions": ["Which winning route remains viable under the sampled opponent configurations?",
                                   "Which resources and opposing checks determine its transition to the endgame?"]}

    def to_dict(self):
        return asdict(self)
