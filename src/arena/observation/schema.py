"""Unified observation semantics, schema ``obs-1``.

Two record types, linked by ``(battle_id, side, request_id)``:

- ``PlayerObservation`` -- what the acting player may know when one request arrives. This is model input
  and nothing else: teacher targets live in dataset examples that point at an observation by request id.
- ``DecisionTrace`` -- what happened to the decision for that request: proposal -> validation ->
  substitution -> submission -> server feedback -> the choice the simulator accepted -> resolved events.

Knowledge. Every non-trivial value is a ``Fact`` that says how the player knows it (``Knowledge``) and
whether there is a value at all (``Status``). The statuses keep apart what the old pipelines conflated:

- ``NONE`` (known to be absent: item knocked off, no status, tera already spent) vs ``UNKNOWN`` (may exist,
  this player has not seen it) vs ``UNRECORDED`` (the source never recorded it, e.g. a replay converter).
- a value of ``0`` / ``False`` / ``[]`` is a ``KNOWN`` value, never a stand-in for unknown or unrecorded.
- unavailable *now* (``request`` says so) vs exhausted for the battle (derived from public events).
- current vs original (types, ability, forme, item) and held now vs held before: kept as separate facts.
- actual game turn (``|turn|``) vs decision index (n-th request this player answered).

``HIDDEN`` facts never appear in a PlayerObservation; the class exists so omniscient analysis records
(e.g. the server's team sheets) can say so explicitly.

Provenance. Each observation carries where it came from (live capture, simulator replay of a server log,
Metamon/replay reconstruction, synthetic), the rules pin it was produced under, the simulator / client /
converter versions, and the data tier: ``strict`` (exact player stream at a pinned simulator and rule
set), ``reconstructed`` (rebuilt from public replays; some fields estimated or unrecorded, rules of the
time unknown unless dated) or ``synthetic``. Reconstructed data may never enter a strict evaluation set.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "obs-1"


class Knowledge(str, Enum):
    OWN_PRIVATE = "own_private"      # own team sheet or private request: exact
    PUBLIC_START = "public_start"    # shown to both players at the start by the format (team preview)
    REVEALED = "revealed"            # shown publicly during the battle by a protocol event
    DERIVED = "derived"              # certain consequence of public information and the rules
    ESTIMATED = "estimated"          # probability / usage / experience; carries confidence or candidates
    RECONSTRUCTED = "reconstructed"  # rebuilt from a replay or converter; not guaranteed exact
    HIDDEN = "hidden"                # the player may not know it; never part of a PlayerObservation


class Status(str, Enum):
    KNOWN = "known"            # ``value`` holds the value (0, False and [] are values)
    NONE = "none"              # known to be absent
    UNKNOWN = "unknown"        # may exist; not known to this player
    UNRECORDED = "unrecorded"  # the source did not record it; says nothing about the battle


class Source(str, Enum):
    LIVE = "live_stream"                 # client capture (arena.observation.capture)
    SIM_REPLAY = "sim_replay"            # server log replayed through the simulator (strict tier)
    METAMON = "metamon_reconstructed"    # Metamon parsed replays / self-play (UniversalState)
    POKECHAMP = "pokechamp_replay"       # public replay logs (spectator view)
    SYNTHETIC = "synthetic"              # generated questions / positions


class Tier(str, Enum):
    STRICT = "strict"
    RECONSTRUCTED = "reconstructed"
    SYNTHETIC = "synthetic"


class Phase(str, Enum):
    """What the request asks for. Derived from the request JSON, never from poke-env flags alone."""
    TEAM_PREVIEW = "team_preview"    # teamPreview: choose order / leads
    MOVE = "move"                    # active: choose a move (or switch, if not trapped)
    FORCE_SWITCH = "force_switch"    # forceSwitch: replace a fainted / ejected / pivoting active
    REVIVAL = "revival"              # forceSwitch while reviving: choose a fainted ally to revive
    WAIT = "wait"                    # wait: nothing to choose
    UNKNOWN = "unknown"


@dataclass
class Fact:
    value: Any = None
    status: Status = Status.UNKNOWN
    knowledge: Optional[Knowledge] = None
    since: Optional[int] = None          # event sequence number where it became known (None: from request)
    turn: Optional[int] = None           # game turn it was observed
    confidence: Optional[float] = None   # ESTIMATED / RECONSTRUCTED only
    candidates: Optional[list] = None    # possible values when only a range or set is known

    @classmethod
    def known(cls, value, knowledge: Knowledge, **kw) -> "Fact":
        return cls(value=value, status=Status.KNOWN, knowledge=knowledge, **kw)

    @classmethod
    def none(cls, knowledge: Knowledge, **kw) -> "Fact":
        return cls(status=Status.NONE, knowledge=knowledge, **kw)

    @classmethod
    def unknown(cls, candidates: Optional[list] = None) -> "Fact":
        return cls(status=Status.UNKNOWN, candidates=candidates)

    @classmethod
    def unrecorded(cls) -> "Fact":
        return cls(status=Status.UNRECORDED)


@dataclass
class Provenance:
    source: Source
    tier: Tier
    format: str                              # Showdown format id, e.g. gen9ou
    rules: Optional[str] = None              # rules pin, e.g. data/rules/showdown-1636bf6.json#gen9ou
    rules_epoch: Optional[str] = None        # for historical data: battle date, or "unknown"
    simulator_version: Optional[str] = None  # Showdown commit the battle ran on / was replayed with
    client_version: Optional[str] = None     # poke-env version (live) or None
    converter_version: Optional[str] = None  # repo commit (+dirty) of the code that built the record
    raw_record: Optional[str] = None         # path / id of the raw source (server log, replay id, parquet row)
    platform: str = "local"                  # local | showdown-replays | metamon-selfplay | ...
    battle_id: str = ""
    player_side: str = ""                    # p1 | p2
    notes: list[str] = field(default_factory=list)


@dataclass
class ActionOption:
    """One selectable action of a request, with a stable identity independent of slot order.

    ``action_id``: ``move:<moveid>``, ``move:<moveid>+tera``, ``switch:<species-id>``, ``revive:<species-id>``,
    ``team:<order>``, ``pass``. ``choice`` is the exact string to submit (``move 2 terastallize``)."""
    action_id: str
    choice: str
    kind: str                        # move | switch | revive | team | pass
    label: str = ""
    details: dict = field(default_factory=dict)  # pp/maxpp/target/disabled flags kept as the request says


@dataclass
class PlayerObservation:
    provenance: Provenance
    request_id: Optional[int]                 # server rqid (None if the source has no request ids)
    request_phase: Phase
    actual_turn: int                          # last |turn| seen (0 before the first turn)
    decision_index: int                       # n-th decision request of this player in the battle
    event_sequence: int                       # number of public protocol lines included
    own_team: list[dict] = field(default_factory=list)                 # all six, fainted included
    opponent_preview: list[dict] = field(default_factory=list)         # team preview (OU); [] if none
    opponent_revealed_team: list[dict] = field(default_factory=list)   # public dossiers
    active_positions: dict = field(default_factory=dict)               # {"p1a": ident, "p2a": ident}
    weather: Optional[dict] = None            # {"weather", "since_turn", "source", "turns_left": {min, max}}
    field_conditions: list[dict] = field(default_factory=list)
    side_conditions: dict = field(default_factory=dict)                # {"own": [...], "opponent": [...]}
    slot_conditions: dict = field(default_factory=dict)
    observable_effects_and_counters: list[dict] = field(default_factory=list)
    public_history: Optional[dict] = None     # reference into the battle's event log: {"log": id, "end": seq}
    selectable_actions: list[ActionOption] = field(default_factory=list)
    derived_estimates: list[dict] = field(default_factory=list)       # each with premises and knowledge
    tera: dict = field(default_factory=dict)          # {"own": {used, by, type, available_now}, "opponent": {...}}
    pokemon_left: dict = field(default_factory=dict)  # {"own": n, "opponent": n}
    unrecorded: list[str] = field(default_factory=list)  # fields the source could not provide (not "none")
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlayerObservation":
        """Inverse of ``to_dict`` (also after a JSON round trip: enums come back from their values)."""
        p = dict(d["provenance"])
        p["source"], p["tier"] = Source(p["source"]), Tier(p["tier"])
        return cls(**{**d, "provenance": Provenance(**p), "request_phase": Phase(d["request_phase"]),
                      "selectable_actions": [ActionOption(**a) for a in d.get("selectable_actions") or []]})


@dataclass
class DecisionTrace:
    battle_id: str
    player_side: str
    request_id: Optional[int]
    request_phase: Phase
    decision_index: int
    actual_turn: int
    selectable_actions: list[str] = field(default_factory=list)  # action ids offered by the request
    proposed_action: Optional[dict] = None   # agent output: {"order": ..., "agent": ..., "raw": ...}
    validated_action: Optional[dict] = None  # {"action_id": ..., "legal": bool, "reasons": [...]}
    substitute: Optional[dict] = None        # {"action_id": ..., "reason": ...} when not the agent's choice
    submitted_action: Optional[dict] = None  # {"message": "/choose ...", "action_id": ...}
    server_feedback: list[str] = field(default_factory=list)   # |error| lines answering this request
    accepted_choice: Optional[dict] = None   # {"choice": "move saltcure", "action_id": ...} from inputLog
    resolved_events: Optional[dict] = None   # {"start": seq, "end": seq} of the events this choice caused
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionTrace":
        return cls(**{**d, "request_phase": Phase(d["request_phase"])})


RULES_DIR = Path(__file__).resolve().parents[3] / "data" / "rules"


def rules_pin(simulator_version: str, fmt: str) -> str | None:
    """``data/rules/showdown-<commit7>.json#<format>`` if that rule table was extracted for this simulator
    commit (``scripts/node/showdown_rules.js``) and covers the format; None otherwise (rules not pinned)."""
    words = (simulator_version or "").split()
    commit = words[-1][:7] if words else ""
    path = RULES_DIR / f"showdown-{commit}.json"
    if not commit or not path.is_file() or fmt not in json.loads(path.read_text()):
        return None
    return f"data/rules/{path.name}#{fmt}"
