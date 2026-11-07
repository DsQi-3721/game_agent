"""Public-event tracking that poke-env's Battle does not keep, from the battle's own protocol lines.

poke-env holds the current state (HP, status, boosts, effects, revealed moves, side conditions); what it
drops, and what the observation needs, is provenance and history:

- the last move each Pokemon used (and on which turn);
- item provenance: revealed (and how), lost (knocked off / eaten / popped / tricked away, on which turn) --
  so "no item" and "unknown item" stay different facts, and a lost item is never reported as held;
- ability reveals, attributed to the right Pokemon (``[of]`` names the holder for ``[from] item:`` /
  ``[from] ability:`` effects on another Pokemon: Rocky Helmet, Rough Skin, Intimidate ...);
- Terastallization per side (which Pokemon, which type, which turn): the request stops showing it once
  that Pokemon faints, the side's Tera stays spent;
- when each volatile effect, weather, terrain, field and side condition started, and who set weather /
  terrain (their duration depends on the setter's item);
- slot conditions announced in public (Future Sight / Doom Desire, Wish, Healing Wish / Lunar Dance).

``tracker_for(battle)`` keeps one tracker per battle and feeds it only the lines added since the last call.
Pokemon are keyed ``"p1: Name"`` (position letter dropped). Illusion is followed through ``|replace|``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from poke_env.data import to_id_str

SLOT_TURNS = {"wish": 1, "futuresight": 2, "doomdesire": 2}    # resolves at the end of turn set + n
WEATHER_ITEMS = {"raindance": "damprock", "sunnyday": "heatrock", "sandstorm": "smoothrock", "snow": "icyrock",
                 "snowscape": "icyrock", "hail": "icyrock"}


def effect_holder(kind: str, ev: list[str], tags: dict) -> Optional[str]:
    """Whose ``[from] item: / ability:`` effect a protocol line reports. ``[of]`` names the effect's source:
    the holder for effects that hit someone else (Rocky Helmet, Rough Skin, Flame Body, weather / terrain
    setters, Mirror Armor), but the attacker for effects on the holder itself (Dry Skin / Water Absorb
    heals, Levitate immunity, an item or ability activating)."""
    subject = ev[2] if len(ev) > 2 and ev[2][:1] == "p" else None
    if kind in ("-heal", "-immune", "-activate") or not tags.get("of"):
        return subject
    return tags["of"]


def mon_key(ident: str) -> str:
    """``p2a: Ting-Lu`` -> ``p2: Ting-Lu``."""
    side, _, name = ident.partition(": ")
    return f"{side[:2]}: {name}" if name else ident


def _tags(parts: list[str]) -> dict[str, str]:
    out = {}
    for p in parts:
        if p.startswith("[") and "]" in p:
            k, _, v = p[1:].partition("]")
            out[k] = v.strip()
    return out


def _strip(effect: str) -> str:
    for prefix in ("move: ", "ability: ", "item: "):
        if effect.startswith(prefix):
            return effect[len(prefix):]
    return effect


@dataclass
class MonHistory:
    last_move: Optional[str] = None
    last_move_turn: Optional[int] = None
    moved_since_entry: bool = False        # has used a move since its latest switch-in (Fake Out, First Impression)
    item: Optional[str] = None             # last item known to be held (None: no item info)
    item_turn: Optional[int] = None
    item_how: str = ""                     # how it was revealed
    lost_item: Optional[str] = None        # item known to be gone
    lost_turn: Optional[int] = None
    lost_how: str = ""
    lost_seq: Optional[int] = None         # event index of the loss (orders it against other events)
    ability: Optional[str] = None
    ability_turn: Optional[int] = None
    tera: Optional[str] = None
    tera_turn: Optional[int] = None
    effects: dict = field(default_factory=dict)   # effect id -> {"turn": start turn, ...details}
    forme: Optional[str] = None            # species / forme as last shown (switch, detailschange): permanent
    temp_forme: Optional[str] = None       # -formechange (Relic Song, Hunger Switch ...): reverts on leaving
    hp_turn: Optional[int] = None          # turn of the last public HP report for this Pokemon


@dataclass
class Tracker:
    player_role: str = ""
    cursor: int = 0
    turn: int = 0
    mons: dict = field(default_factory=dict)      # mon_key -> MonHistory
    tera: dict = field(default_factory=dict)      # side -> {"mon": key, "type": t, "turn": n}
    weather: dict = field(default_factory=dict)   # {"id", "turn", "source", "setter"}
    fields: dict = field(default_factory=dict)    # field id -> {"turn", "source", "setter"}
    sides: dict = field(default_factory=dict)     # (side, condition id) -> {"turn": first start}
    slots: dict = field(default_factory=dict)     # side -> {condition id -> {"turn", "by"}}
    active: dict = field(default_factory=dict)    # side -> (mon_key shown as active, turn it came in)
    preview: dict = field(default_factory=dict)   # side -> [(details, holds an item)] from |poke|
    last_user: Optional[str] = None               # mon_key of the latest |move| line (sets weather, screens ...)
    after_upkeep: bool = False                    # between |upkeep| and the next |turn|: end-of-turn switch-ins
    seq: int = 0                                  # index of the event being processed
    _last_slot: Optional[tuple] = None            # (user, side, condition) of a slot move just announced
    entered: dict = field(default_factory=dict)   # mon_key -> event index of its latest switch-in

    @property
    def clock(self) -> int:
        """The turn a timed effect starting now counts from: residuals tick at the end of each turn, so an
        effect set before turn 1 or after this turn's upkeep (a switch-in replacing a fainted Pokemon) loses
        its first turn only at the end of the next one."""
        return self.turn + 1 if self.turn == 0 or self.after_upkeep else self.turn

    @property
    def now(self) -> int:
        """The same clock for a decision being made now: a request after this turn's upkeep (replacing a
        fainted Pokemon) comes after the turn's residuals already counted down."""
        return self.turn + 1 if self.after_upkeep else self.turn

    def mon(self, ident: str) -> MonHistory:
        return self.mons.setdefault(mon_key(ident), MonHistory())

    # --- protocol -------------------------------------------------------------------------------
    def feed(self, lines: list[list[str]]) -> None:
        for ev in lines:
            self.seq += 1
            if len(ev) < 2:
                continue
            k = ev[1]
            tags = _tags(ev[3:])
            if k != "-fail":
                self._last_slot = self._last_slot if k == "move" else None
            if k == "turn" and len(ev) > 2:
                self.turn = int(ev[2])
                self.after_upkeep = False
                self.expire_slots(self.turn)
            elif k == "upkeep":
                self.after_upkeep = True
            elif k == "poke" and len(ev) > 3:
                self.preview.setdefault(ev[2], []).append((ev[3], len(ev) > 4 and ev[4] == "item"))
            elif k == "clearpoke":
                self.preview = {}
            elif k == "move" and len(ev) > 3:
                self.last_user = mon_key(ev[2])
                if "from" not in tags or tags.get("from", "").startswith("lockedmove"):
                    h = self.mon(ev[2])
                    h.last_move, h.last_move_turn = to_id_str(ev[3]), self.turn
                    h.moved_since_entry = True
                    self._slot_move(ev[2], to_id_str(ev[3]))
            elif k == "replace" and len(ev) > 2:      # Illusion broken: this stint's events were the real one's
                side = ev[2][:2]
                old, since = self.active.get(side, (None, self.turn))
                new = mon_key(ev[2])
                if old and old != new:
                    o, n = self.mon(old), self.mon(new)
                    if o.last_move_turn is not None and o.last_move_turn >= since:
                        n.last_move, n.last_move_turn, o.last_move, o.last_move_turn = o.last_move, o.last_move_turn, None, None
                    n.effects, o.effects = o.effects, {}
                    if o.item_turn is not None and o.item_turn >= since:
                        n.item, n.item_turn, n.item_how, o.item, o.item_turn = o.item, o.item_turn, o.item_how, None, None
                    n.hp_turn = self.turn               # its HP was the one shown under the disguise
                self.active[side] = (new, since)
            elif k == "-item" and len(ev) > 3:
                h = self.mon(ev[2])
                h.item, h.item_turn, h.item_how = to_id_str(ev[3]), self.turn, tags.get("from", "revealed")
                h.lost_item = None
            elif k == "-enditem" and len(ev) > 3:
                h = self.mon(ev[2])
                h.lost_item, h.lost_turn, h.lost_seq = to_id_str(ev[3]), self.turn, self.seq
                h.lost_how = ("eaten" if "eat" in tags else tags.get("from", "") or "consumed")
                h.item = None
            elif k == "-ability" and len(ev) > 3:
                h = self.mon(ev[2])
                h.ability, h.ability_turn = to_id_str(ev[3]), self.turn
            elif k == "-terastallize" and len(ev) > 3:
                h = self.mon(ev[2])
                h.tera, h.tera_turn = ev[3], self.turn
                self.tera[ev[2][:2]] = {"mon": mon_key(ev[2]), "type": ev[3], "turn": self.turn}
            elif k in ("-start", "-end") and len(ev) > 3:
                h = self.mon(ev[2])
                eff = to_id_str(_strip(ev[3]))
                if k == "-start":
                    detail = {"turn": self.turn}
                    if eff in ("encore", "disable") and len(ev) > 4 and not ev[4].startswith("["):
                        detail["move"] = to_id_str(_strip(ev[4]))
                    elif eff == "encore" and h.last_move:
                        detail["move"] = h.last_move
                    h.effects[eff] = detail
                    if eff in ("futuresight", "doomdesire"):       # announced on the user; lands on the foe
                        foe = "p2" if ev[2].startswith("p1") else "p1"
                        self.slots.setdefault(foe, {})[eff] = {"turn": self.turn, "by": mon_key(ev[2])}
                else:
                    h.effects.pop(eff, None)
                    if eff in ("futuresight", "doomdesire"):
                        self.slots.get(ev[2][:2], {}).pop(eff, None)
            elif k == "-weather" and len(ev) > 2:
                w = to_id_str(ev[2])
                if w == "none":
                    self.weather = {}
                elif "upkeep" not in tags and w != self.weather.get("id"):
                    setter = mon_key(tags["of"]) if tags.get("of") else self.last_user
                    self.weather = {"id": w, "turn": self.turn, "clock": self.clock, "seq": self.seq,
                                    "source": tags.get("from", "") or "move", "setter": setter,
                                    **self._setter_item(setter)}
            elif k in ("-fieldstart", "-fieldend") and len(ev) > 2:
                f = to_id_str(_strip(ev[2]))
                if k == "-fieldstart":
                    setter = mon_key(tags["of"]) if tags.get("of") else self.last_user
                    self.fields[f] = {"turn": self.turn, "clock": self.clock, "seq": self.seq,
                                      "source": tags.get("from", "") or "move", "setter": setter,
                                      **self._setter_item(setter)}
                else:
                    self.fields.pop(f, None)
            elif k in ("-sidestart", "-sideend") and len(ev) > 3:
                key = (ev[2][:2], to_id_str(_strip(ev[3])))
                if k == "-sidestart":
                    self.sides.setdefault(key, {"turn": self.turn, "clock": self.clock, "seq": self.seq,
                                                "setter": self.last_user, **self._setter_item(self.last_user)})
                else:
                    self.sides.pop(key, None)
            elif k == "-heal" and tags.get("from", "") in ("move: Wish", "move: Healing Wish", "move: Lunar Dance"):
                self.slots.get(ev[2][:2], {}).pop(to_id_str(tags["from"][6:]), None)
            elif k in ("switch", "drag") and len(ev) > 2:
                prev = self.active.get(ev[2][:2])
                if prev and prev[0] in self.mons:
                    self.mons[prev[0]].temp_forme = None
                self.active[ev[2][:2]] = (mon_key(ev[2]), self.turn)
                self.entered[mon_key(ev[2])] = self.seq
                h = self.mon(ev[2])
                h.effects.clear()
                h.moved_since_entry = False
                if len(ev) > 3:
                    h.forme = ev[3].split(",")[0]
                if len(ev) > 4:
                    h.hp_turn = self.turn
            elif k == "detailschange" and len(ev) > 3:
                h = self.mon(ev[2])
                h.forme, h.temp_forme = ev[3].split(",")[0], None
            elif k == "-formechange" and len(ev) > 3:
                self.mon(ev[2]).temp_forme = ev[3].split(",")[0]
            elif k == "-transform" and len(ev) > 3:          # Transform / Imposter: looks like the target now
                target = self.mons.get(mon_key(ev[3]))
                self.mon(ev[2]).temp_forme = (target.temp_forme or target.forme) if target else ev[3].split(": ", 1)[-1]
            elif k == "-fail" and len(ev) > 2 and self._last_slot and self._last_slot[0] == mon_key(ev[2]):
                side, cond = self._last_slot[1], self._last_slot[2]
                self.slots.get(side, {}).pop(cond, None)          # the Wish / Healing Wish failed: no slot
            elif k == "faint" and len(ev) > 2:
                h = self.mon(ev[2])
                h.effects.clear()
                h.temp_forme = None
            elif k in ("-damage", "-heal", "-sethp") and len(ev) > 3:
                self.mon(ev[2]).hp_turn = self.turn
            # [from] item: / [from] ability: on other events reveal the holder ([of] when given)
            if k not in ("-item", "-enditem", "-ability") and len(ev) > 2 and "from" in tags:
                src = tags["from"]
                holder = effect_holder(k, ev, tags)
                if holder and src.startswith("item: "):
                    h = self.mon(holder)
                    if h.lost_item != to_id_str(src[6:]):
                        h.item, h.item_turn, h.item_how = to_id_str(src[6:]), self.turn, f"[from] {src}"
                elif holder and src.startswith("ability: "):
                    h = self.mon(holder)
                    h.ability, h.ability_turn = to_id_str(src[9:]), self.turn

    def expire_slots(self, now: int) -> None:
        """Drop slot conditions whose turn has passed (they resolve even when nothing is shown: a Wish on
        a full-HP target, Future Sight on a fainted slot)."""
        for side_slots in self.slots.values():
            for cond in [c for c, v in side_slots.items() if now > v["turn"] + SLOT_TURNS.get(c, 99)]:
                side_slots.pop(cond)

    def _setter_item(self, setter: Optional[str]) -> dict:
        """What the public record says the setter held when the condition started (for its duration):
        {"setter_item": id or None, "setter_item_public": whether the record shows it}."""
        h = self.mons.get(setter) if setter else None
        if h is None:
            return {"setter_item": None, "setter_item_public": False}
        if h.item:
            return {"setter_item": h.item, "setter_item_public": True}
        if h.lost_item:                     # lost before this event: it held nothing
            return {"setter_item": None, "setter_item_public": True}
        return {"setter_item": None, "setter_item_public": False}

    def _slot_move(self, ident: str, move: str) -> None:
        side = ident[:2]
        self._last_slot = None
        if move in ("wish", "healingwish", "lunardance") and move not in self.slots.get(side, {}):
            self.slots.setdefault(side, {})[move] = {"turn": self.turn, "by": mon_key(ident)}
            self._last_slot = (mon_key(ident), side, move)


def tracker_for(battle) -> Tracker:
    """The battle's tracker, advanced over the protocol lines poke-env stored since the last call."""
    tr = getattr(battle, "_arena_tracker", None)
    if tr is None:
        tr = Tracker(player_role=battle.player_role or "")
        try:
            battle._arena_tracker = tr
        except AttributeError:        # a slotted Battle without a __dict__: rebuild every time
            pass
    data = getattr(battle, "_replay_data", None) or []
    if tr.cursor < len(data):
        tr.feed(data[tr.cursor:])
        tr.cursor = len(data)
    return tr
