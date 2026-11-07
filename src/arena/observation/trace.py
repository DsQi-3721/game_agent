"""Request phases, stable action identities and decision traces (Gen 9 singles).

``request_phase`` and ``selectable_actions`` read the request JSON exactly as Showdown builds it
(``sim/side.ts`` getRequestData, ``sim/pokemon.ts`` getMoveRequestData / getMoves):

- 0-PP moves arrive with ``disabled: true``; a hard lock (Outrage, recharge, ...) arrives as a single move
  without pp fields plus ``trapped``, and no ``canTerastallize`` -- so "cannot tera now" is not "tera used".
- ``trapped`` forbids switching; ``maybeTrapped`` / ``maybeDisabled`` / ``maybeLocked`` mean the choice may
  be refused by the server ([Unavailable choice] + an updated request), so those options stay selectable
  and carry the flag.
- a Revival Blessing request is a ``forceSwitch`` whose active has ``reviving``; its targets are the fainted
  allies, and the accepted choice is still written ``switch N`` -- the identity here is ``revive:<name>``.

``build_traces`` joins the replayed player stream (requests), the server inputLog (accepted choices) and
the client capture (agent proposal, sent message, server errors) into one ``DecisionTrace`` per decision.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from poke_env.data import to_id_str

from arena.observation.capture import choice_attempts
from arena.observation.schema import ActionOption, DecisionTrace, Phase


def request_phase(req: dict) -> Phase:
    if not isinstance(req, dict):
        return Phase.UNKNOWN
    if req.get("wait"):
        return Phase.WAIT
    if req.get("teamPreview"):
        return Phase.TEAM_PREVIEW
    if req.get("forceSwitch"):
        side = (req.get("side") or {}).get("pokemon") or []
        return Phase.REVIVAL if any(p.get("active") and p.get("reviving") for p in side) else Phase.FORCE_SWITCH
    if req.get("active"):
        return Phase.MOVE
    return Phase.UNKNOWN


def ident_name(ident: str) -> str:
    """``p1: Great Tusk`` / ``p1a: Great Tusk`` -> ``greattusk`` (protocol identity of a team member)."""
    return to_id_str(ident.split(":", 1)[1] if ":" in ident else ident)


def fainted(mon: dict) -> bool:
    return str(mon.get("condition", "")).endswith(" fnt") or str(mon.get("condition", "")) == "0 fnt"


def selectable_actions(req: dict) -> list[ActionOption]:
    """Every choice the request allows (singles). Flags say which ones the server may still refuse."""
    phase = request_phase(req)
    side = (req.get("side") or {}).get("pokemon") or []
    if phase == Phase.TEAM_PREVIEW:
        return [ActionOption("team:*", "team", "team", details={"size": len(side),
                                                               "choose": req.get("maxChosenTeamSize") or len(side)})]
    opts: list[ActionOption] = []
    active = (req.get("active") or [{}])[0] if phase == Phase.MOVE else {}
    if phase == Phase.MOVE:
        maybe = {k: True for k in ("maybeDisabled", "maybeLocked") if active.get(k)}
        for i, m in enumerate(active.get("moves") or [], 1):
            if m.get("disabled") is True:
                continue
            details = {k: m[k] for k in ("pp", "maxpp", "target", "disabled") if k in m}
            details.update(maybe)
            opts.append(ActionOption(f"move:{m['id']}", f"move {i}", "move", m.get("move", m["id"]), details))
            if active.get("canTerastallize"):
                opts.append(ActionOption(f"move:{m['id']}+tera", f"move {i} terastallize", "move",
                                         f"{m.get('move', m['id'])} + Tera {active['canTerastallize']}", details))
    can_switch = phase in (Phase.FORCE_SWITCH, Phase.MOVE) and not active.get("trapped")
    for pos, mon in enumerate(side, 1):
        if phase == Phase.REVIVAL:
            if fainted(mon):
                opts.append(ActionOption(f"revive:{ident_name(mon['ident'])}", f"switch {pos}", "revive",
                                         mon.get("details", "")))
        elif can_switch and not mon.get("active") and not fainted(mon):
            opts.append(ActionOption(f"switch:{ident_name(mon['ident'])}", f"switch {pos}", "switch",
                                     mon.get("details", ""), {"maybeTrapped": True} if active.get("maybeTrapped") else {}))
    return opts


def choice_action_id(choice: str, req: dict) -> Optional[str]:
    """A submitted (``/choose move 2``, ``/team 312546``) or accepted (``move saltcure terastallize``,
    ``team 3, 1, 2, 5, 4, 6``) choice -> stable action id under this request."""
    c = choice.strip()
    for prefix in ("/choose ", "/"):
        if c.startswith(prefix):
            c = c[len(prefix):]
            break
    kind, _, rest = c.partition(" ")
    side = (req.get("side") or {}).get("pokemon") or []
    if kind == "team":
        digits = [int(d) for d in re.findall(r"\d", rest)]
        try:
            return "team:" + ",".join(ident_name(side[d - 1]["ident"]) for d in digits)
        except IndexError:
            return None
    if kind == "move":
        parts = rest.split()
        if not parts:
            return None
        moves = ((req.get("active") or [{}])[0].get("moves")) or []
        mv = parts[0]
        if mv.isdigit():
            if not 1 <= int(mv) <= len(moves):
                return None
            mv = moves[int(mv) - 1]["id"]
        return f"move:{to_id_str(mv)}" + ("+tera" if "terastallize" in parts[1:] else "")
    if kind == "switch":
        target = rest.strip()
        verb = "revive" if request_phase(req) == Phase.REVIVAL else "switch"
        if target.isdigit():
            if not 1 <= int(target) <= len(side):
                return None
            return f"{verb}:{ident_name(side[int(target) - 1]['ident'])}"
        # by name (poke-env sends the species): Showdown matches the nickname or the species
        for mon in side:
            if to_id_str(target) in (ident_name(mon["ident"]), to_id_str(mon.get("details", "").split(",")[0])):
                return f"{verb}:{ident_name(mon['ident'])}"
        return None
    if kind in ("pass", "default", "shift", "undo"):
        return kind
    return None


def _player_requests(lines: list[str]) -> list[dict]:
    """Requests of one player's stream with their position among protocol lines and the game turn."""
    out, n_protocol, turn = [], 0, 0
    for line in lines:
        if line.startswith("|request|"):
            body = line[len("|request|"):]
            if body:
                out.append({"req": json.loads(body), "at": n_protocol, "turn": turn})
        elif line.startswith("|error|"):
            continue
        else:
            n_protocol += 1
            if line.startswith("|turn|"):
                turn = int(line.split("|")[2])
    return out


def _stale(attempt: dict) -> bool:
    """Refused by the server because the request had already been answered (server/room-battle.ts: "Sorry,
    too late to make a different move; the next turn has already started")."""
    return "too late" in (attempt.get("error") or "")


def build_traces(battle_id: str, side: str, replay_lines: list[str], input_log: list[str],
                 capture: Optional[list[dict]] = None,
                 trajectory: Optional[list[dict]] = None) -> tuple[list[DecisionTrace], dict]:
    """One DecisionTrace per decision request of ``side``; returns (traces, alignment stats).

    A request with ``update: true`` replaces the pending request of the same decision (the server re-sent
    it after an [Unavailable choice]); every other non-wait request is a new decision answered by the next
    ``>side`` line of the inputLog. ``trajectory``: the agent's per-decision log records of this battle
    (``TrajectoryLogger``, joined by rqid) -- the raw model output, the parsed order and whether the agent
    fell back -- so a proposal can be followed from model output to the executed choice."""
    decisions: list[dict] = []
    for r in _player_requests(replay_lines):
        if request_phase(r["req"]) == Phase.WAIT:
            continue
        if r["req"].get("update") and decisions:
            decisions[-1]["superseded"].append(decisions[-1]["req"].get("rqid"))
            decisions[-1].update(req=r["req"])
            continue
        decisions.append({**r, "superseded": []})
    choices = [ln[len(side) + 2:] for ln in input_log if ln.startswith(f">{side} ")]
    acts: dict = {}
    for rec in capture or []:
        if rec.get("k") == "act":
            acts.setdefault(rec.get("rqid"), []).append(rec)
    agent_log = {r.get("rqid"): r for r in trajectory or [] if r.get("battle_tag") == battle_id and "rqid" in r}
    attempts: dict = {}
    for a in choice_attempts(capture or []):
        attempts.setdefault(a["rqid"], []).append(a)
    traces = []
    for i, d in enumerate(decisions):
        req, rqid = d["req"], d["req"].get("rqid")
        rqids = [*d["superseded"], rqid]
        phase = request_phase(req)
        options = selectable_actions(req)
        ids = [o.action_id for o in options]
        t = DecisionTrace(battle_id=battle_id, player_side=side, request_id=rqid, request_phase=phase,
                          decision_index=i, actual_turn=d["turn"], selectable_actions=ids)
        my_acts = [a for q in rqids for a in acts.get(q, [])]
        sent = [a for q in rqids for a in attempts.get(q, [])]
        if my_acts:
            first_act = my_acts[0]
            proposed = choice_action_id(first_act.get("order") or "", req) if first_act.get("order") else None
            t.proposed_action = {"order": first_act.get("order"), "agent": first_act.get("agent"),
                                 "action_id": proposed, **({"policy": first_act["policy"]} if "policy" in first_act else {})}
            legal = proposed is not None and (proposed in ids or (phase == Phase.TEAM_PREVIEW
                                                                  and proposed.startswith("team:")))
            t.validated_action = {"action_id": proposed, "legal": legal}
            if first_act.get("substitute"):
                t.substitute = {"action_id": proposed, "reason": first_act["substitute"]}
            rec = next((agent_log[q] for q in rqids if q in agent_log), None)
            if rec is not None:
                t.proposed_action["raw"] = rec.get("output")
                t.proposed_action["parsed"] = rec.get("parsed_order")
                t.proposed_action["agent_fallback"] = bool(rec.get("fallback"))
        # an attempt the server turned away as "too late" belonged to a request already answered (it arrived
        # after the next request): it is kept, marked stale, and never taken for this decision's submission
        live = [a for a in sent if not _stale(a)]
        if sent:
            last = live[-1] if live else None
            t.submitted_action = {"message": last["data"] if last else None,
                                  "action_id": choice_action_id(last["data"], req) if last else None,
                                  "attempts": [{"message": a["data"], "error": a["error"],
                                                **({"stale": True} if _stale(a) else {})} for a in sent]}
        t.server_feedback = [a["error"] for a in sent if a["error"]]
        if i < len(choices):
            t.accepted_choice = {"choice": choices[i], "action_id": choice_action_id(choices[i], req)}
        nxt = decisions[i + 1]["at"] if i + 1 < len(decisions) else None
        t.resolved_events = {"start": d["at"], "end": nxt}
        if live and t.proposed_action and not t.substitute \
                and t.submitted_action["action_id"] != t.proposed_action["action_id"]:
            own = len(my_acts) == len(live) and my_acts[-1].get("order") == live[-1]["data"]
            refused = [a for a in live if a["error"]]
            reason = ("agent re-decided after the server refused" if own and refused
                      else "client sent its own order" if not own else "agent changed its order")
            t.substitute = {"action_id": t.submitted_action["action_id"], "reason": reason}
        traces.append(t)
    stats = {"decisions": len(decisions), "choices": len(choices),
             "aligned": len(decisions) == len(choices) or len(decisions) == len(choices) + 1,
             "updated_requests": sum(len(d["superseded"]) for d in decisions)}
    return traces, stats


def phase_of(battle) -> Phase:
    """The phase of the request a poke-env battle is answering (poke-env keeps the last request)."""
    return request_phase(getattr(battle, "_last_request", None) or {})


REVIVAL_NOTE = ("Revival Blessing: choose a fainted Pokemon to revive; it returns to your team with half its HP "
                "and does not switch in")


def action_constraints(req: dict) -> dict:
    """Why the request offers fewer actions than usual, as the request states it (empty if nothing applies):
    a hard lock, moves that cannot be used now (no PP / disabled, with the reason when one was derived),
    switching blocked or possibly blocked, a move that may turn out disabled or locked."""
    if request_phase(req) is not Phase.MOVE:
        return {}
    a = (req.get("active") or [{}])[0]
    moves = a.get("moves") or []
    out: dict = {}
    if len(moves) == 1 and "pp" not in moves[0]:
        mid = moves[0]["id"]
        out["locked_into"] = {"recharge": "recharge (must recharge this turn)",
                              "struggle": "struggle (no usable move left)"}.get(mid, mid)
    derived = ((req.get("derived") or {}).get("disabled")) or {}
    unusable = {m["id"]: ("no PP left" if m.get("pp") == 0 else derived.get(m["id"], "disabled this turn"))
                for m in moves if m.get("disabled") is True}
    if unusable:
        out["unusable_moves"] = unusable
    if a.get("trapped"):
        out["switching"] = "trapped: you cannot switch out"
    elif a.get("maybeTrapped"):
        out["switching"] = "you may be trapped: a switch can be refused"
    if a.get("maybeDisabled") or a.get("maybeLocked"):
        out["warning"] = "a move may turn out to be disabled or locked when chosen"
    return out
