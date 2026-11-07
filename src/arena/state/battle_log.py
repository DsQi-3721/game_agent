"""Showdown battle-protocol log -> compact, perspective-aware turn digests.

The raw protocol (``battle._replay_data``) is the environment's execution trace:
positional, noisy, full of tokens that do not matter for a decision (timestamps,
upkeep, chat, animation hints). Agents that keep a persistent context consume it
through this module instead of appending it verbatim:

- ``events_since(battle, cursor)`` -- the protocol lines added since the previous
  decision. Cursor based, so a re-requested decision sees an empty slice.
- ``LogDigester.ingest(events, battle)`` -- deterministic distillation into one line
  per action, from the acting player's perspective ("your" / "opp"), HP shown as
  before->after percentages; plus the sparse long-lived ``KeyFact``s (opponent
  reveals, KOs, hazards, tera) used as memory fallbacks and as key facts for
  context-dilution analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from poke_env.data import GenData, to_id_str

from arena.observation.events import OBSERVATION_ONLY, ROLE, is_ignorable
from arena.observation.tracker import effect_holder

# protocol message kinds the turn narrative leaves out: presentation / chat / timing / out of scope, and the
# format, preview and request lines the PlayerObservation reads (``arena.observation.events`` lists every
# kind with its role and the reason it may be left out)
NOISE_KINDS = {k for k in ROLE if is_ignorable(k)} | OBSERVATION_ONLY | {"seed", "-center", "-combine", "-waiting"}

# attachments merged into the action line that caused them
_ATTACH = {
    "-supereffective": "super effective",
    "-resisted": "resisted",
    "-crit": "critical hit",
    "-miss": "missed",
    "-fail": "failed",
    "-immune": "no effect",
    "-notarget": "no target",
    "-ohko": "one-hit KO",
    "-nothing": "nothing happened",
}


def events_since(battle, cursor: int) -> tuple[list[list[str]], int]:
    """New protocol events since ``cursor`` and the advanced cursor."""
    data = getattr(battle, "_replay_data", None) or []
    cursor = max(0, min(cursor, len(data)))
    return [list(e) for e in data[cursor:]], len(data)


@dataclass
class TurnResult:
    """Typed environment return: what happened during one battle turn."""

    turn: int
    lines: list[str] = field(default_factory=list)

    def render(self) -> str:
        body = "\n".join(f"- {x}" for x in self.lines) if self.lines else "- (nothing notable)"
        return f"<TURN_RESULT turn={self.turn}>\n{body}\n</TURN_RESULT>"


@dataclass
class KeyFact:
    """A rule-extracted fact. ``scope``: "event" (something that happened: true for good) or "state" (true
    while it lasts: an item held, a hazard on a side, a status); a state fact gets ``ended_turn`` when an
    event makes it false, and the change itself is recorded as an event fact."""
    turn: int
    kind: str   # opp_move | opp_item | opp_item_lost | opp_ability | tera | faint | hazard | hazard_removed | status
    text: str
    scope: str = "event"
    ended_turn: Optional[int] = None

    @property
    def current(self) -> bool:
        return self.scope == "event" or self.ended_turn is None


def _strip_prefix(effect: str) -> str:
    for p in ("move: ", "ability: ", "item: "):
        if effect.startswith(p):
            return effect[len(p):]
    return effect


def _hp_of(cond: str) -> tuple[Optional[float], Optional[str]]:
    """'212/302 par' -> (0.70, 'par'); '0 fnt' -> (0.0, 'fnt')."""
    if not cond:
        return None, None
    parts = cond.split()
    status = parts[1] if len(parts) > 1 else None
    if parts[0] == "0":
        return 0.0, "fnt"
    num, _, den = parts[0].partition("/")
    try:
        return float(num) / float(den), status
    except (ValueError, ZeroDivisionError):
        return None, status


def _tags(ev: list[str]) -> dict[str, str]:
    out = {}
    for f in ev[2:]:
        if f.startswith("[") and "]" in f:
            k, _, v = f[1:].partition("]")
            out[k.strip()] = v.strip()
    return out


class LogDigester:
    """Stateful (per battle) digester: tracks HP to report deltas and dedupes facts."""

    def __init__(self):
        self.hp: dict[str, float] = {}      # 'p2: Garganacl' -> last known HP fraction
        self._facts: set[tuple] = set()
        self._live: dict[tuple, KeyFact] = {}    # state facts still true, by what they are about
        self._layers: dict[tuple, int] = {}      # (side, condition) -> layers set so far
        self._lost: dict[str, str] = {}          # Pokemon -> item it last lost
        self._transformed: set[str] = set()      # Pokemon currently transformed (its moves are borrowed)
        self._as_one: set[str] = set()           # Pokemon that showed As One (its Unnerve line is part of it)
        self._ending: list[tuple] = []           # (label, fact) ended in the current slice

    # --- naming -----------------------------------------------------------
    @staticmethod
    def _key(ident: str) -> tuple[str, str, str]:
        """'p1a: Passimian' -> ('p1', 'Passimian', 'p1: Passimian')."""
        head, _, name = ident.partition(": ")
        side = head[:2]
        name = name.strip()
        return side, name, f"{side}: {name}"

    def _who(self, battle, side: str) -> str:
        return "your" if side == getattr(battle, "player_role", None) else "opp"

    def _display(self, battle, key: str, name: str) -> str:
        mon = (battle.team.get(key) if key.startswith(str(battle.player_role)) else None) \
            or battle.opponent_team.get(key)
        species = getattr(mon, "species", None)
        if species and to_id_str(name) != species:
            return f"{name} ({species})"
        return name

    def _mon(self, battle, ident: str) -> tuple[str, str, str]:
        side, name, key = self._key(ident)
        return self._who(battle, side), self._display(battle, key, name), key

    def _fact(self, facts: list, label: Optional[int], kind: str, text: str, dedupe: tuple,
              state_key: Optional[tuple] = None) -> Optional[KeyFact]:
        """Queue a fact; ``label`` may be None (pre-marker) and ``text`` may contain ``{turn}`` -- both are
        resolved once the whole slice has been seen. ``state_key``: the fact is a current state (an item
        held, a hazard, a status) that a later event can end; an earlier live fact on the same key ends."""
        if dedupe in self._facts and state_key is None:
            return None
        if state_key is not None and state_key in self._live:
            if self._live[state_key].text == text:
                return None
            self._end(state_key, label, facts)
        self._facts.add(dedupe)
        kf = KeyFact(-1, kind, text, scope="state" if state_key is not None else "event")
        facts.append((label, kf))
        if state_key is not None:
            self._live[state_key] = kf
        return kf

    def _end(self, state_key: tuple, label: Optional[int], facts: list, kind: str = "", text: str = "") -> None:
        """End a live state fact (its turn is resolved with the slice) and record the change as an event."""
        kf = self._live.pop(state_key, None)
        if kf is None:
            return
        self._ending.append((label, kf))
        if kind:
            facts.append((label, KeyFact(-1, kind, text)))

    def _reveal_from(self, facts, label, battle, ev: list[str], tags: dict) -> None:
        """'[from] item: Life Orb' / '[from] ability: Rough Skin' reveal who holds what; the holder follows
        ``tracker.effect_holder`` (``[of]`` for effects on someone else, the line's Pokemon otherwise)."""
        src = tags.get("from", "")
        if not src.startswith(("item: ", "ability: ")):
            return
        holder = effect_holder(ev[1] if len(ev) > 1 else "", ev, tags)
        if not holder:
            return
        who, name, key = self._mon(battle, holder)
        if who != "opp":
            return
        if src.startswith("item: "):
            item = src[len("item: "):]
            if self._lost.get(key) == item:        # Sitrus Berry: -enditem [eat], then -heal [from] item: ...
                return
            self._fact(facts, label, "opp_item", f"opp {name} holds {item}", ("item", key, item),
                       state_key=("item", key))
        else:
            ab = src[len("ability: "):]
            self._fact(facts, label, "opp_ability", f"opp {name} has ability {ab}", ("ability", key, ab),
                       state_key=("ability", key))

    # --- main pass --------------------------------------------------------
    def ingest(self, events: list[list[str]], battle) -> tuple[list[TurnResult], list[KeyFact]]:
        labelled: list[tuple[Optional[int], str]] = []
        facts: list[tuple] = []           # (label | None, KeyFact with a {turn} template)
        self._ending = []
        label: Optional[int] = None
        first_marker: Optional[int] = None
        cur: Optional[list[str]] = None
        cur_label: Optional[int] = None

        def flush():
            nonlocal cur
            if cur:
                labelled.append((cur_label, "; ".join(cur)))
            cur = None

        def emit(text: str):
            flush()
            labelled.append((label, text))

        for ev in events:
            if len(ev) < 2:
                continue
            kind = ev[1]
            if kind in NOISE_KINDS:
                continue
            tags = _tags(ev)

            if kind == "turn":
                flush()
                try:
                    label = int(ev[2])
                except (IndexError, ValueError):
                    continue
                if first_marker is None:
                    first_marker = label
                continue

            if kind == "move" and len(ev) >= 4:
                flush()
                who, name, _ = self._mon(battle, ev[2])
                move = ev[3]
                line = f"{who} {name} used {move}"
                if len(ev) >= 5 and ev[4] and not ev[4].startswith("[") and ev[4] != ev[2]:
                    twho, tname, _ = self._mon(battle, ev[4])
                    line += f" on {twho} {tname}"
                cur, cur_label = [line], label
                if "miss" in tags:
                    cur.append("missed")
                if who == "opp" and "from" not in tags and self._key(ev[2])[2] not in self._transformed:
                    self._fact(facts, label, "opp_move", f"opp {name} has {move}", ("move", name, move))
                continue

            if kind in _ATTACH:
                if cur is not None:
                    note = _ATTACH[kind]
                    if kind == "-immune" and len(ev) >= 3:
                        twho, tname, _ = self._mon(battle, ev[2])
                        note = f"{twho} {tname} immune"
                    if note not in cur:
                        cur.append(note)
                elif kind == "-immune" and len(ev) >= 3:
                    who, name, _ = self._mon(battle, ev[2])
                    emit(f"{who} {name} was immune")
                continue

            if kind in ("switch", "drag", "replace") and len(ev) >= 3:
                side = ev[2][:2]
                self._transformed = {k for k in self._transformed if not k.startswith(side)}
                who, name, key = self._mon(battle, ev[2])
                frac, _ = _hp_of(ev[4] if len(ev) >= 5 else "")
                if frac is not None:
                    self.hp[key] = frac
                hp = f" ({round(frac * 100)}%)" if frac is not None else ""
                verb = {"switch": "sent out", "drag": "was forced to send out",
                        "replace": "revealed (Illusion)"}[kind]
                emit(f"{who} {verb} {name}{hp}")
                continue

            if kind in ("-damage", "-heal", "-sethp") and len(ev) >= 4:
                who, name, key = self._mon(battle, ev[2])
                frac, status = _hp_of(ev[3])
                before = self.hp.get(key)
                if frac is not None:
                    self.hp[key] = frac
                after_s = "fainted" if status == "fnt" else (f"{round(frac * 100)}%" if frac is not None else "?")
                delta = f"{round(before * 100)}%->{after_s}" if before is not None else after_s
                src = tags.get("from", "")
                self._reveal_from(facts, label, battle, ev, tags)
                if kind == "-heal" and src == "move: Revival Blessing":
                    self._fact(facts, label, "revived", f"{who} {name} was revived (Revival Blessing, turn {{turn}})",
                               ("revived", key, label))
                if kind == "-damage" and not src and cur is not None:
                    cur.append(f"{who} {name} {delta}")
                else:
                    verb = {"-damage": "took damage", "-heal": "healed", "-sethp": "HP set"}[kind]
                    why = f" ({_strip_prefix(src)})" if src else ""
                    emit(f"{who} {name} {verb} {delta}{why}")
                continue

            if kind == "faint" and len(ev) >= 3:
                who, name, key = self._mon(battle, ev[2])
                self.hp[key] = 0.0
                emit(f"{who} {name} fainted")
                self._fact(facts, label, "faint", f"{who} {name} fainted (turn {{turn}})",
                           ("faint", key))
                self._end(("status", key), label, facts)
                continue

            if kind in ("-status", "-curestatus") and len(ev) >= 4:
                who, name, key = self._mon(battle, ev[2])
                st = ev[3].upper()
                self._reveal_from(facts, label, battle, ev, tags)
                if kind == "-status":
                    emit(f"{who} {name} is now {st}")
                    self._fact(facts, label, "status", f"{who} {name} is {st}", ("status", key, st),
                               state_key=("status", key))
                else:
                    emit(f"{who} {name} cured of {st}")
                    self._end(("status", key), label, facts)
                continue

            if kind == "-cureteam" and len(ev) >= 3:
                who, name, _ = self._mon(battle, ev[2])
                emit(f"{who} team cured of status ({name})")
                side = ev[2][:2]
                for k in [k for k in self._live if k[0] == "status" and k[1].startswith(side)]:
                    self._end(k, label, facts)
                continue

            if kind in ("-setboost", "-swapboost", "-invertboost", "-copyboost", "-clearpositiveboost") and len(ev) >= 3:
                who, name, _ = self._mon(battle, ev[2])
                what = {"-setboost": f"{ev[3] if len(ev) > 3 else ''} set to {ev[4] if len(ev) > 4 else ''}",
                        "-swapboost": "stat changes swapped", "-invertboost": "stat changes inverted",
                        "-copyboost": "copied stat changes", "-clearpositiveboost": "positive stat changes cleared"}[kind]
                emit(f"{who} {name} {what}".strip())
                continue

            if kind == "-hitcount" and len(ev) >= 4:
                note = f"hit {ev[3]} times"
                if cur is not None:
                    cur.append(note)
                else:
                    emit(note)
                continue

            if kind == "-block" and len(ev) >= 4:
                who, name, _ = self._mon(battle, ev[2])
                note = f"blocked by {who} {name}'s {_strip_prefix(ev[3])}"
                self._reveal_from(facts, label, battle, [*ev[:3], *ev[4:], f"[from] {ev[3]}"],
                                  {**tags, "from": ev[3]} if ev[3].startswith(("ability: ", "item: ")) else tags)
                if cur is not None:
                    cur.append(note)
                else:
                    emit(note)
                continue

            if kind in ("-boost", "-unboost") and len(ev) >= 5:
                who, name, _ = self._mon(battle, ev[2])
                sign = "+" if kind == "-boost" else "-"
                text = f"{who} {name} {ev[3]} {sign}{ev[4]}"
                if cur is not None:
                    cur.append(text)
                else:
                    emit(text)
                continue

            if kind in ("-clearboost", "-clearallboost", "-clearnegativeboost") and len(ev) >= 3:
                if kind == "-clearallboost":
                    emit("all stat changes were cleared")
                else:
                    who, name, _ = self._mon(battle, ev[2])
                    emit(f"{who} {name} stat changes cleared")
                continue

            if kind == "-weather" and len(ev) >= 3:
                if "upkeep" in tags:
                    continue
                w = ev[2]
                emit("weather ended" if w == "none" else f"weather is now {w}")
                continue

            if kind in ("-fieldstart", "-fieldend") and len(ev) >= 3:
                cond = _strip_prefix(ev[2])
                emit(f"field: {cond} {'started' if kind == '-fieldstart' else 'ended'}")
                continue

            if kind in ("-sidestart", "-sideend") and len(ev) >= 4:
                side = ev[2][:2]
                who = self._who(battle, side)
                cond = _strip_prefix(ev[3])
                if kind == "-sidestart":
                    emit(f"{who} side: {cond} set")
                    layers = self._layers.get((side, cond), 0) + 1
                    self._layers[(side, cond)] = layers
                    text = f"{who} side has {cond}" + (f" x{layers}" if cond in ("Spikes", "Toxic Spikes") else "")
                    self._fact(facts, label, "hazard", text, ("side", side, cond, layers),
                               state_key=("side", side, cond))
                else:
                    how = _strip_prefix(tags.get("from", "")) or "ended"
                    emit(f"{who} side: {cond} ended" + (f" ({how})" if how != "ended" else ""))
                    self._layers.pop((side, cond), None)
                    self._end(("side", side, cond), label, facts, "hazard_removed",
                              f"{cond} removed from {who} side ({how}, turn {{turn}})")
                continue

            if kind == "-swapsideconditions":
                emit("side conditions swapped (Court Change)")
                me = getattr(battle, "player_role", "p1")
                for k in [k for k in self._live if k[0] == "side"]:
                    self._end(k, label, facts)
                swapped = {(("p2" if sd == "p1" else "p1"), c): n for (sd, c), n in self._layers.items()}
                self._layers = swapped
                for (sd, c), n in swapped.items():
                    w = "your" if sd == me else "opp"
                    self._fact(facts, label, "hazard", f"{w} side has {c}" + (f" x{n}" if c in ("Spikes", "Toxic Spikes") else ""),
                               ("side", sd, c, n, "swap", label), state_key=("side", sd, c))
                continue

            if kind in ("-start", "-end") and len(ev) >= 4:
                who, name, _ = self._mon(battle, ev[2])
                eff = _strip_prefix(ev[3])
                if "silent" in tags and kind == "-end":
                    continue
                self._reveal_from(facts, label, battle, ev, tags)
                text = f"{who} {name}: {eff} {'started' if kind == '-start' else 'ended'}"
                if cur is not None and kind == "-start":
                    cur.append(f"{eff} on {who} {name}")
                else:
                    emit(text)
                continue

            if kind == "-terastallize" and len(ev) >= 4:
                who, name, _ = self._mon(battle, ev[2])
                emit(f"{who} {name} terastallized into {ev[3]}")
                self._fact(facts, label, "tera", f"{who} {name} terastallized into {ev[3]}",
                           ("tera", who, name))
                continue

            if kind in ("-item", "-enditem") and len(ev) >= 4:
                who, name, key = self._mon(battle, ev[2])
                item = ev[3]
                src = tags.get("from", "")
                if kind == "-item":
                    emit(f"{who} {name} has {item}" + (f" ({_strip_prefix(src)})" if src else ""))
                    self._lost.pop(key, None)
                    if who == "opp":
                        self._fact(facts, label, "opp_item", f"opp {name} holds {item}", ("item", key, item),
                                   state_key=("item", key))
                else:
                    how = "eaten" if "eat" in tags else (_strip_prefix(src) if src else "used up")
                    emit(f"{who} {name} lost {item} ({how})")
                    self._end(("item", key), label, facts)
                    self._lost[key] = item
                    self._fact(facts, label, "item_lost", f"{who} {name} lost {item} ({how}, turn {{turn}})",
                               ("lost", key, item, label))
                continue

            if kind == "-ability" and len(ev) >= 4:
                who, name, key = self._mon(battle, ev[2])
                emit(f"{who} {name} ability {ev[3]}")
                if ev[3] == "As One":
                    self._as_one.add(key)
                elif ev[3] == "Unnerve" and key in self._as_one:
                    continue                        # As One announces Unnerve with itself
                if who == "opp":
                    self._fact(facts, label, "opp_ability", f"opp {name} has ability {ev[3]}",
                               ("ability", key, ev[3]), state_key=("ability", key))
                continue

            if kind == "-endability" and len(ev) >= 3:
                who, name, key = self._mon(battle, ev[2])
                emit(f"{who} {name}'s ability stopped working")
                continue

            if kind == "cant" and len(ev) >= 4:
                who, name, _ = self._mon(battle, ev[2])
                reason = _strip_prefix(ev[3])
                emit(f"{who} {name} could not move ({reason})")
                continue

            if kind in ("-activate", "-singleturn", "-singlemove") and len(ev) >= 4:
                who, name, _ = self._mon(battle, ev[2]) if ev[2][:1] == "p" else ("", ev[2], "")
                eff = _strip_prefix(ev[3])
                src = ev[3]
                if src.startswith(("item: ", "ability: ")):
                    self._reveal_from(facts, label, battle, ev[:3], {"from": src, **({"of": tags["of"]} if "of" in tags else {})})
                text = f"{who} {name}: {eff}".strip()
                if cur is not None:
                    cur.append(text)
                else:
                    emit(text)
                continue

            if kind in ("detailschange", "-formechange") and len(ev) >= 4:
                who, name, key = self._mon(battle, ev[2])
                forme = ev[3].split(",")[0]
                emit(f"{who} {name} changed form to {forme}")
                live = self._live.get(("ability", key))
                if live is not None:              # Tera Shift -> Tera Shell: the new forme's ability differs
                    dex = GenData.from_gen(getattr(battle, "gen", 9) or 9).pokedex.get(to_id_str(forme)) or {}
                    allowed = {to_id_str(a) for a in (dex.get("abilities") or {}).values()}
                    if allowed and to_id_str(live.text.rsplit(" has ability ", 1)[-1]) not in allowed:
                        self._end(("ability", key), label, facts)
                continue

            if kind == "-transform" and len(ev) >= 4:
                who, name, key = self._mon(battle, ev[2])
                twho, tname, _ = self._mon(battle, ev[3])
                emit(f"{who} {name} transformed into {tname}")
                self._transformed.add(key)
                continue

            if kind == "-mustrecharge" and len(ev) >= 3:
                who, name, _ = self._mon(battle, ev[2])
                emit(f"{who} {name} must recharge")
                continue

            if kind == "-prepare" and len(ev) >= 4:
                who, name, _ = self._mon(battle, ev[2])
                emit(f"{who} {name} is charging {ev[3]}")
                continue

            if kind == "-hint" and len(ev) >= 3:
                emit(f"(hint: {ev[2]})")
                continue

            if kind == "win" and len(ev) >= 3:
                emit(f"battle over: {ev[2]} won")
                continue

            if kind == "tie":
                emit("battle over: tie")
                continue
            # anything else: ignored (cosmetic / rare)

        flush()
        default = (first_marker - 1) if first_marker else int(getattr(battle, "turn", 0) or 0)
        grouped: dict[int, list[str]] = {}
        for lab, text in labelled:
            grouped.setdefault(lab if lab is not None else default, []).append(text)
        results = [TurnResult(t, lines) for t, lines in sorted(grouped.items())]
        resolved = []
        for lab, kf in facts:
            kf.turn = lab if lab is not None else default
            kf.text = kf.text.replace("{turn}", str(kf.turn))
            resolved.append(kf)
        for lab, kf in self._ending:
            kf.ended_turn = lab if lab is not None else default
        return results, resolved
