"""In-battle evidence about the opponent's hidden sets, from the public protocol lines only.

Reveals (``-item``, ``[from] item:``, ``-ability``, ``-terastallize``, moves) are recorded as facts. What makes
this module worth having is the evidence that reveals nothing outright -- the things that did NOT happen:

- no ``Air Balloon`` message when it came in               -> not Air Balloon
- a Protosynthesis / Quark Drive user came in without sun / Electric Terrain and nothing activated
                                                            -> not Booster Energy
- an entry ability that always announces itself (Intimidate, Pressure, a weather or terrain setter when that
  weather is not up, ...) was not announced                 -> not that ability
- Stealth Rock on its side: hurt on entry -> not Heavy-Duty Boots; not hurt -> Heavy-Duty Boots (or Magic Guard)
- two different moves in one stint on the field             -> no Choice item
- a status move (not Trick / Switcheroo)                    -> no Assault Vest (and very unlikely a Choice item)
- dealt damage without Life Orb recoil                      -> not Life Orb
- hit by a contact move without Rocky Helmet damage         -> not Rocky Helmet
- below full HP at the end of a turn without a Leftovers heal -> not Leftovers / Black Sludge
- a whole turn on the field without being statused by its orb -> not Toxic Orb / Flame Orb
- fell to half HP or less without eating a Sitrus Berry     -> not Sitrus Berry
- came back with a third more HP than it left with -> Regenerator; with the same HP -> not Regenerator
- a Trick / Switcheroo that handed us an item -> the item it started with

Each piece of evidence is a likelihood multiplier on one value (``lik_item``, ``lik_ability``); ``"*"`` is the
multiplier for every value not listed (positive evidence: "Heavy-Duty Boots, or nothing else"). Hard
exclusions are not exactly 0 so a protocol corner case we did not model can be outweighed later.

``evidence_for(battle)`` keeps one tracker per battle (like ``arena.observation.tracker``) and feeds it the lines
poke-env stored since the last call; ``EvidenceTracker`` also runs on any list of split protocol lines
(spectator logs, offline evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from poke_env.data import to_id_str

from arena.knowledge import store
from arena.observation.tracker import _tags, effect_holder, mon_key

NO = 0.02            # a hard "it would have shown": not 0, so an unmodelled corner case can be outweighed
CHOICE = ("choiceband", "choicescarf", "choicespecs")
PIVOT_ITEMS = ("trick", "switcheroo")
# abilities that always announce themselves on entry ("|-ability|..." or a [from] ability: line). Conditions
# (the weather is not already up ...) are checked in ``_entry_abilities``.
ANNOUNCE = {"intimidate", "pressure", "unnerve", "moldbreaker", "teravolt", "turboblaze", "neutralizinggas",
            "airlock", "cloudnine", "drizzle", "drought", "sandstream", "snowwarning", "electricsurge",
            "grassysurge", "mistysurge", "psychicsurge", "download", "dauntlessshield", "intrepidsword"}
WEATHER_OF = {"drizzle": "raindance", "drought": "sunnyday", "orichalcumpulse": "sunnyday",
              "sandstream": "sandstorm", "snowwarning": "snowscape"}
TERRAIN_OF = {"electricsurge": "electricterrain", "hadronengine": "electricterrain", "grassysurge": "grassyterrain",
              "mistysurge": "mistyterrain", "psychicsurge": "psychicterrain"}
STATUS_IMMUNE_ABILITIES = {"comatose", "purifyingsalt", "leafguard", "shieldsdown", "waterveil", "waterbubble",
                           "thermalexchange", "immunity", "pastelveil", "magicguard", "poisonheal", "guts",
                           "toxicboost", "flareboost", "quickfeet", "marvelscale"}
NO_RECOIL_ABILITIES = {"magicguard", "sheerforce"}
CONTACT_PROTECT_ITEMS = {"protectivepads"}


def _hp(s: str) -> Optional[float]:
    """'45/100 psn' -> 45.0 (percent); '0 fnt' -> 0."""
    s = (s or "").split(" ")[0]
    if s in ("0", "0fnt"):
        return 0.0
    num, _, den = s.partition("/")
    try:
        return 100.0 * float(num) / float(den) if den else float(num)
    except ValueError:
        return None


@dataclass
class MonEvidence:
    side: str
    key: str
    species: str = ""
    moves: list = field(default_factory=list)         # revealed moves, first use order
    item: Optional[str] = None                        # held now (revealed)
    first_item: Optional[str] = None                  # the item it started with, once known
    item_gone: bool = False                           # known to hold nothing now
    ability: Optional[str] = None
    tera: Optional[str] = None
    lik_item: dict = field(default_factory=dict)
    lik_ability: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)         # (attr, value) -> reason
    hp: float = 100.0
    hp_out: Optional[float] = None                    # HP shown when it last left the field
    status: Optional[str] = None
    stint_moves: set = field(default_factory=set)
    stint_item_event: bool = False
    entries: int = 0
    healblock: bool = False
    version: int = 0

    def held(self) -> Optional[str]:
        return None if self.item_gone else self.item

    def item_unknown(self) -> bool:
        return self.item is None and not self.item_gone and self.first_item is None


@dataclass
class SideStyle:
    """How the opponent has been playing (counts over the battle so far)."""
    turns: int = 0
    switches: int = 0                 # voluntary switches (not replacing a fainted Pokemon)
    attacks: int = 0
    status_moves: int = 0
    setup: int = 0
    hazards: int = 0
    tera_turn: Optional[int] = None


SETUP = {"swordsdance", "dragondance", "nastyplot", "calmmind", "quiverdance", "bulkup", "shellsmash", "tidyup",
         "victorydance", "agility", "rockpolish", "irondefense", "coil", "bellydrum", "curse", "growth", "geomancy",
         "shiftgear", "filletaway", "clangoroussoul", "noretreat", "cosmicpower", "takeheart", "workup", "honeclaws"}
HAZARDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb", "stoneaxe", "ceaselessedge"}


class EvidenceTracker:
    def __init__(self, opp_side: str):
        self.opp = opp_side                       # "p1" | "p2": the side being modelled
        self.mons: dict[str, MonEvidence] = {}
        self.active: dict[str, Optional[str]] = {}    # side -> mon key
        self.weather: Optional[str] = None
        self.fields: set = set()
        self.sides: dict[str, set] = {"p1": set(), "p2": set()}
        self.turn = 0
        self.after_upkeep = False
        self.turn_lines: list = []
        self.windows: dict[str, dict] = {}        # mon key -> entry window being observed
        self.move_win: Optional[dict] = None      # the opponent's move being resolved
        self.our_move: Optional[dict] = None      # our move being resolved (contact checks)
        self.style = SideStyle()
        self.orders: list[dict] = []              # per turn: which active moved first (same priority bracket)
        self.hits: list[dict] = []                # the opponent's first-move hits on our active, exact damage
        self.our_hp: dict[str, tuple] = {}        # our Pokemon -> (current, max) HP points (own view only)
        self._turn_moves: list = []
        self._turn_noise = False                  # Quick Claw / Custap / Quick Draw / a switch before the moves
        self.cursor = 0
        self.version = 0
        self.preview: list[str] = []              # opponent's species at team preview

    # --- helpers --------------------------------------------------------------------------------------------
    def mon(self, ident: str) -> MonEvidence:
        k = mon_key(ident)
        m = self.mons.get(k)
        if m is None:
            m = self.mons[k] = MonEvidence(side=k[:2], key=k)
        return m

    def is_opp(self, ident: str) -> bool:
        return ident[:2] == self.opp

    def lik(self, m: MonEvidence, attr: str, value: str, mult: float, why: str) -> None:
        d = m.lik_item if attr == "item" else m.lik_ability
        if (attr, value) in m.notes and value != "*":
            return
        d[value] = d.get(value, 1.0) * mult
        m.notes[(attr, value)] = why
        m.version += 1
        self.version += 1

    def only(self, m: MonEvidence, attr: str, value: str, mult: float, why: str) -> None:
        """Positive evidence: every value but ``value`` is multiplied by ``mult``."""
        d = m.lik_item if attr == "item" else m.lik_ability
        if (attr, value) in m.notes:
            return
        for k in list(d):
            if k not in (value, "*"):
                d[k] *= mult
        d["*"] = d.get("*", 1.0) * mult
        d[value] = d.get(value, 1.0)
        m.notes[(attr, value)] = why
        m.version += 1
        self.version += 1

    def _reveal_item(self, m: MonEvidence, item: str, *, held: bool = True) -> None:
        item = to_id_str(item)
        if held:
            if m.first_item is None and not m.stint_item_event and m.item is None and not m.item_gone:
                m.first_item = item
            m.item, m.item_gone = item, False
        m.version += 1
        self.version += 1

    def _possible_abilities(self, m: MonEvidence) -> list[str]:
        sp = (store.dex().get("species") or {}).get(to_id_str(m.species)) or {}
        return [to_id_str(a) for a in (sp.get("abilities") or {}).values()]

    def _types(self, m: MonEvidence) -> list[str]:
        sp = (store.dex().get("species") or {}).get(to_id_str(m.species)) or {}
        return [t.lower() for t in sp.get("types") or []]

    @staticmethod
    def _move(mid: str) -> dict:
        return (store.dex().get("moves") or {}).get(mid) or {}

    # --- protocol -------------------------------------------------------------------------------------------
    def feed(self, lines: list[list[str]]) -> None:
        for ev in lines:
            if len(ev) < 2:
                continue
            self._line(ev)

    def _close_windows(self, side: Optional[str] = None) -> None:
        for k in [k for k in self.windows if side is None or k[:2] == side]:
            self._close_entry(self.windows.pop(k))

    def _line(self, ev: list[str]) -> None:
        k = ev[1]
        tags = _tags(ev[3:])
        self.turn_lines.append(ev)
        if k in ("move", "turn", "upkeep", "cant"):
            self._close_windows()
        if k in ("move", "turn", "upkeep", "switch", "drag", ""):
            self._close_move()
        if k == "turn" and len(ev) > 2:
            self._speed_order()
            self.turn = int(ev[2])
            self._turn_moves, self._turn_noise = [], False
            self.after_upkeep = False
            self.turn_lines = [ev]
            self.style.turns = self.turn
        elif k == "poke" and len(ev) > 3 and ev[2] == self.opp:
            self.preview.append(to_id_str(ev[3].split(",")[0]))
        elif k == "upkeep":
            self._residual()
            self.after_upkeep = True
        elif k in ("switch", "drag", "replace") and len(ev) > 3:
            if self.turn > 0 and not self.after_upkeep and len(self._turn_moves) < 2:
                self._turn_noise = True
            self._switch(ev, k)
        elif k == "move" and len(ev) > 3:
            self._move_line(ev, tags)
            if "from" not in tags:
                self._turn_moves.append((ev[2][:2], mon_key(ev[2]), to_id_str(ev[3]), self.after_upkeep))
        elif k == "-damage" and len(ev) > 3:
            self._damage(ev, tags)
        elif k == "-heal" and len(ev) > 3:
            m = self.mon(ev[2])
            hp = _hp(ev[3])
            if hp is not None:
                m.hp = hp
            self._our_hp(ev[2], ev[3])
        elif k == "-sethp" and len(ev) > 3:
            hp = _hp(ev[3])
            if hp is not None:
                self.mon(ev[2]).hp = hp
            self._our_hp(ev[2], ev[3])
        elif k == "-item" and len(ev) > 3:
            self._item(ev, tags)
        elif k == "-enditem" and len(ev) > 3:
            m = self.mon(ev[2])
            item = to_id_str(ev[3])
            if m.first_item is None and m.item is None and not m.stint_item_event:
                m.first_item = item
            m.item, m.item_gone = None, True
            m.stint_item_event = True
            m.version += 1
            self.version += 1
        elif k == "-ability" and len(ev) > 3:
            m = self.mon(ev[2])
            if "from" not in tags or tags.get("from", "").startswith("ability:"):
                m.ability = to_id_str(ev[3])
                m.version += 1
                self.version += 1
        elif k == "-terastallize" and len(ev) > 3:
            m = self.mon(ev[2])
            m.tera = to_id_str(ev[3])
            m.version += 1
            self.version += 1
            if self.is_opp(ev[2]):
                self.style.tera_turn = self.turn
            else:
                self._our_tera_turn = self.turn      # our types changed before this turn's moves
        elif k == "-weather" and len(ev) > 2:
            w = to_id_str(ev[2])
            self.weather = None if w == "none" else w
        elif k == "-fieldstart" and len(ev) > 2:
            self.fields.add(to_id_str(ev[2].replace("move: ", "")))
        elif k == "-fieldend" and len(ev) > 2:
            self.fields.discard(to_id_str(ev[2].replace("move: ", "")))
        elif k == "-sidestart" and len(ev) > 3:
            self.sides[ev[2][:2]].add(to_id_str(ev[3].replace("move: ", "")))
        elif k == "-sideend" and len(ev) > 3:
            self.sides[ev[2][:2]].discard(to_id_str(ev[3].replace("move: ", "")))
        elif k == "-status" and len(ev) > 3:
            self.mon(ev[2]).status = ev[3]
        elif k == "-curestatus" and len(ev) > 2:
            self.mon(ev[2]).status = None
        elif k == "-start" and len(ev) > 3:
            if "healblock" in to_id_str(ev[3]):
                self.mon(ev[2]).healblock = True
        elif k == "-end" and len(ev) > 3:
            if "healblock" in to_id_str(ev[3]):
                self.mon(ev[2]).healblock = False
        elif k == "faint" and len(ev) > 2:
            m = self.mon(ev[2])
            m.hp = 0.0
        if self.move_win is not None and len(ev) > 2 and ev[2][:2] != self.opp:
            if k == "-crit":
                self.move_win["crit"] = True
            elif k == "-hitcount":
                self.move_win["multi"] = True
            elif k in ("-enditem", "-item"):
                self.move_win["our_item_event"] = True
        if k in ("-activate", "-enditem") and len(self._turn_moves) < 2 and any(
                x in "|".join(ev) for x in ("Quick Claw", "Custap Berry", "Quick Draw")):
            self._turn_noise = True
        # [from] item: / [from] ability: reveal the holder
        if k not in ("-item", "-enditem", "-ability") and len(ev) > 2 and "from" in tags:
            src = tags["from"]
            holder = effect_holder(k, ev, tags)
            if holder and src.startswith("item: "):
                m = self.mon(holder)
                item = to_id_str(src[6:])
                if not (m.item_gone and m.first_item == item):
                    self._reveal_item(m, item)
                if self.move_win and self.move_win["key"] == m.key and item == "lifeorb":
                    self.move_win["recoil"] = True
                if self.our_move and item == "rockyhelmet" and mon_key(holder) == self.our_move.get("target"):
                    self.our_move["helmet"] = True
            elif holder and src.startswith("ability: "):
                m = self.mon(holder)
                m.ability = to_id_str(src[9:])
                m.version += 1
                self.version += 1
            for w in self.windows.values():
                if mon_key(holder or "") == w["key"] and src.startswith("ability: "):
                    w["announced"] = True

    # --- events -------------------------------------------------------------------------------------------
    def _switch(self, ev: list[str], k: str) -> None:
        side = ev[2][:2]
        new = self.mon(ev[2])
        species = ev[3].split(",")[0]
        if k != "replace":
            prev_key = self.active.get(side)
            if prev_key and prev_key in self.mons and prev_key != new.key:
                prev = self.mons[prev_key]
                if prev.hp > 0:
                    prev.hp_out = prev.hp
                    if side == self.opp and not self.after_upkeep and k == "switch" and self.turn > 0:
                        self.style.switches += 1
            self._close_windows(side)
        new.species = species
        self.active[side] = new.key
        hp = _hp(ev[4]) if len(ev) > 4 else None
        if len(ev) > 4:
            self._our_hp(ev[2], ev[4])
        if side == self.opp and k != "replace":
            if new.hp_out is not None and hp is not None and new.hp_out < 99.5 and new.ability is None:
                if hp >= new.hp_out + 25 and "regenerator" in self._possible_abilities(new):
                    self.only(new, "ability", "regenerator", 0.05, f"came back with {hp:.0f}% HP after leaving with "
                              f"{new.hp_out:.0f}% (Regenerator)")
                elif abs(hp - new.hp_out) <= 1.5 and "regenerator" in self._possible_abilities(new):
                    self.lik(new, "ability", "regenerator", NO, f"came back with the HP it left with "
                             f"({hp:.0f}%): not Regenerator")
            new.entries += 1
            self.windows[new.key] = {"key": new.key, "balloon": False, "booster": False, "announced": False,
                                     "rocks": "stealthrock" in self.sides.get(side, set()), "rock_hit": False,
                                     "weather": self.weather, "fields": set(self.fields), "first": new.entries == 1}
        if hp is not None:
            new.hp = hp
        new.stint_moves = set()
        new.stint_item_event = False

    def _close_entry(self, w: dict) -> None:
        m = self.mons.get(w["key"])
        if m is None or m.side != self.opp or m.hp <= 0:
            return
        if m.item_unknown() and not w["balloon"]:
            self.lik(m, "item", "airballoon", NO, "no Air Balloon message when it came in")
        abil = self._possible_abilities(m)
        if m.item_unknown() and not w["booster"] and ({"protosynthesis", "quarkdrive"} & set(abil)):
            sun = (w["weather"] or "") in ("sunnyday", "desolateland")
            eterrain = "electricterrain" in w["fields"]
            proto = "protosynthesis" in abil
            if not ((proto and sun) or (not proto and eterrain)):
                self.lik(m, "item", "boosterenergy", NO, "came in without sun / Electric Terrain and its "
                         "Booster Energy did not activate")
        if w["rocks"] and m.item_unknown():
            if w["rock_hit"]:
                self.lik(m, "item", "heavydutyboots", NO, "took Stealth Rock damage on entry")
            elif "magicguard" not in abil:
                self.only(m, "item", "heavydutyboots", 0.03, "took no Stealth Rock damage on entry")
        if m.ability is None and not w["announced"]:
            self._entry_abilities(m, w, abil)

    def _entry_abilities(self, m: MonEvidence, w: dict, abil: list[str]) -> None:
        if "neutralizinggas" in self.fields:
            return
        for a in abil:
            if a not in ANNOUNCE:
                continue
            if a in WEATHER_OF and (w["weather"] or "") == WEATHER_OF[a]:
                continue
            if a in TERRAIN_OF and TERRAIN_OF[a] in w["fields"]:
                continue
            if a in ("dauntlessshield", "intrepidsword") and not w["first"]:
                continue
            self.lik(m, "ability", a, NO, f"{a} would have announced itself when it came in")

    def _move_line(self, ev: list[str], tags: dict) -> None:
        user = self.mon(ev[2])
        mid = to_id_str(ev[3])
        called = "from" in tags and not tags.get("from", "").startswith("lockedmove")
        if user.side == self.opp:
            if not called and mid not in ("struggle", "recharge") and mid not in user.moves:
                user.moves.append(mid)
                user.version += 1
                self.version += 1
            if called or "from" in tags:
                return
            data = self._move(mid)
            cat = data.get("category")
            if mid in SETUP:
                self.style.setup += 1
            if mid in HAZARDS:
                self.style.hazards += 1
            if cat == "Status":
                self.style.status_moves += 1
            elif cat:
                self.style.attacks += 1
            if user.item_unknown() and not user.stint_item_event:
                user.stint_moves.add(mid)
                if len(user.stint_moves) >= 2:
                    for c in CHOICE:
                        self.lik(user, "item", c, NO, "used two different moves without switching out")
                if cat == "Status" and mid not in PIVOT_ITEMS:
                    self.lik(user, "item", "assaultvest", NO, f"used the status move {mid}")
                    for c in CHOICE:
                        self.lik(user, "item", c, 0.1, f"used the status move {mid}")
            if cat in ("Physical", "Special"):
                self.move_win = {"key": user.key, "move": mid, "hit": False, "recoil": False, "fainted": False,
                                 "crit": False, "multi": False}
        else:
            if called:
                return
            data = self._move(mid)
            if "contact" in (data.get("flags") or []) and len(ev) > 4 and ev[4][:2] == self.opp:
                self.our_move = {"target": mon_key(ev[4]), "hit": False, "helmet": False, "user": user.key}

    def _our_hp(self, ident: str, text: str) -> Optional[tuple]:
        """(current, max) HP points of our Pokemon from an exact 'cur/max' (None for percentages)."""
        if ident[:2] == self.opp:
            return None
        num, _, den = (text or "").split(" ")[0].partition("/")
        try:
            cur, mx = float(num), float(den)
        except ValueError:
            return (0.0, self.our_hp.get(mon_key(ident), (0, 0))[1]) if text.startswith("0") else None
        if mx == 100:
            return None
        self.our_hp[mon_key(ident)] = (cur, mx)
        return cur, mx

    def _damage(self, ev: list[str], tags: dict) -> None:
        m = self.mon(ev[2])
        hp = _hp(ev[3])
        src = tags.get("from", "")
        if m.side != self.opp:
            before = self.our_hp.get(m.key)
            after = self._our_hp(ev[2], ev[3])
            mw = self.move_win
            if (not src and mw is not None and before is not None and after is not None and after[0] > 0
                    and "hit_rec" not in mw and len(self._turn_moves) == 1 and self._turn_moves[0][0] == self.opp
                    and not self._turn_noise and self.style.tera_turn != self.turn and not mw.get("crit")
                    and getattr(self, "_our_tera_turn", None) != self.turn and not mw.get("our_item_event")):
                mw["hit_rec"] = {"turn": self.turn, "opp_key": mw["key"], "our_key": m.key, "move": mw["move"],
                                 "damage": before[0] - after[0]}
            if mw is not None:
                mw.setdefault("hit_rec", None)        # only the first damage line of the move counts
        if m.side == self.opp:
            w = self.windows.get(m.key)
            if w is not None and src == "Stealth Rock":
                w["rock_hit"] = True
            if not src and self.our_move and self.our_move["target"] == m.key:
                self.our_move["hit"] = True
            if hp is not None:
                if 0 < hp <= 50 < m.hp and m.item_unknown():
                    m.notes.setdefault(("pending", "sitrus"), self.turn)
                m.hp = hp
        else:
            if not src and self.move_win is not None:
                self.move_win["hit"] = True
            if hp is not None:
                m.hp = hp
        if hp == 0 and self.move_win and self.move_win["key"] == m.key:
            self.move_win["fainted"] = True

    def _item(self, ev: list[str], tags: dict) -> None:
        m = self.mon(ev[2])
        item = to_id_str(ev[3])
        src = tags.get("from", "")
        if src.startswith("move: ") and to_id_str(src[6:]) in PIVOT_ITEMS:
            # Trick / Switcheroo: the receiver shows what the other one held
            other = next((x for x in self.mons.values() if x.side != m.side and x.key == self.active.get(x.side)),
                         None)
            if other is not None and other.side == self.opp and other.first_item is None and not other.item_gone:
                other.first_item = item
                other.notes[("item", "trick")] = f"its {item} was handed over by {to_id_str(src[6:])}"
            m.item, m.item_gone = item, False
            m.stint_item_event = True
            m.version += 1
            self.version += 1
            return
        w = self.windows.get(m.key)
        if w is not None and item == "airballoon":
            w["balloon"] = True
        self._reveal_item(m, item)

    def _close_move(self) -> None:
        mw, self.move_win = self.move_win, None
        if mw is not None and mw.get("hit_rec") and not mw.get("multi") and not mw.get("crit"):
            self.hits.append(mw["hit_rec"])
            self.version += 1
            m = self.mons.get(mw["key"])
            if m is not None:
                m.version += 1
        if mw is not None:
            m = self.mons.get(mw["key"])
            if m is not None and m.item_unknown() and mw["hit"] and not mw["recoil"] and not mw["fainted"] \
                    and m.hp > 0 and not (NO_RECOIL_ABILITIES & set([m.ability] if m.ability else
                                                                    self._possible_abilities(m))):
                self.lik(m, "item", "lifeorb", NO, f"no Life Orb recoil after {mw['move']}")
        om, self.our_move = self.our_move, None
        if om is not None and om["hit"] and not om["helmet"]:
            m = self.mons.get(om["target"])
            if m is not None and m.item_unknown():
                self.lik(m, "item", "rockyhelmet", NO, "no Rocky Helmet damage after a contact move")

    def _speed_order(self) -> None:
        """The two actives' first moves of the turn that just ended, if they tell who is faster: one move each,
        no switch or speed item / ability activation before them, the same base priority (and no ability that
        could change it). The model joins this with the speeds it recorded at the start of the turn."""
        mv = [x for x in self._turn_moves if not x[3]][:2]
        if self.turn <= 0 or self._turn_noise or len(mv) < 2 or mv[0][0] == mv[1][0]:
            return
        pri = []
        for side, key, mid, _late in mv:
            d = self._move(mid)
            pr = d.get("priority", 0) or 0
            if side == self.opp:
                m = self.mons.get(key)
                abil = set([m.ability] if m and m.ability else self._possible_abilities(m) if m else [])
                if (d.get("category") == "Status" and "prankster" in abil) or "galewings" in abil or (
                        "triage" in abil and d.get("flags") and "heal" in d.get("flags")) or mid == "grassyglide":
                    return
            elif mid == "grassyglide" or d.get("category") == "Status":
                return                           # our own ability may change a status move's priority
            pri.append(pr)
        if pri[0] != pri[1]:
            return
        first_side = mv[0][0]
        opp_key = mv[0][1] if first_side == self.opp else mv[1][1]
        self.orders.append({"turn": self.turn, "opp_first": first_side == self.opp, "opp_key": opp_key,
                            "our_key": mv[1][1] if first_side == self.opp else mv[0][1]})
        self.version += 1
        m = self.mons.get(opp_key)
        if m is not None:
            m.version += 1

    def _residual(self) -> None:
        """End of turn: Leftovers / Black Sludge, Toxic / Flame Orb, Sitrus Berry."""
        key = self.active.get(self.opp)
        m = self.mons.get(key) if key else None
        lines = self.turn_lines
        if m is None or m.hp <= 0:
            return
        last_sep = max((i for i, ev in enumerate(lines) if ev[1] == ""), default=None)
        residual = lines[last_sep + 1:] if last_sep is not None else []
        mine = lambda ev: len(ev) > 2 and ev[2][:2] == self.opp and mon_key(ev[2]) == m.key  # noqa: E731
        healed = any(ev[1] == "-heal" and mine(ev) and "item:" in "|".join(ev) for ev in residual)
        weather_hit = any(ev[1] == "-damage" and mine(ev) and _tags(ev[3:]).get("from", "").lower() in
                          ("sandstorm", "hail") for ev in residual)
        terrain_heal = any(ev[1] == "-heal" and mine(ev) and "Grassy Terrain" in "|".join(ev) for ev in residual)
        hp_before = None
        hp_track = m.hp
        for ev in lines[:last_sep] if last_sep is not None else lines:
            if ev[1] in ("-damage", "-heal", "-sethp", "switch", "drag") and mine(ev):
                h = _hp(ev[4] if ev[1] in ("switch", "drag") and len(ev) > 4 else ev[3] if len(ev) > 3 else "")
                if h is not None:
                    hp_before = h
        hp_before = hp_track if hp_before is None else hp_before
        if m.item_unknown() and not healed and not terrain_heal and not m.healblock and (hp_before < 99.5 or weather_hit):
            self.lik(m, "item", "leftovers", NO, f"no Leftovers heal at the end of turn {self.turn}")
            self.lik(m, "item", "blacksludge", NO, f"no Black Sludge heal at the end of turn {self.turn}")
        if m.item_unknown() and m.status is None and "mistyterrain" not in self.fields:
            types = self._types(m)
            abil = set([m.ability] if m.ability else self._possible_abilities(m))
            if not (abil & STATUS_IMMUNE_ABILITIES) and "safeguard" not in self.sides.get(self.opp, set()):
                if "poison" not in types and "steel" not in types:
                    self.lik(m, "item", "toxicorb", NO, f"not poisoned by a Toxic Orb at the end of turn {self.turn}")
                if "fire" not in types:
                    self.lik(m, "item", "flameorb", NO, f"not burned by a Flame Orb at the end of turn {self.turn}")
        for mm in self.mons.values():
            t = mm.notes.pop(("pending", "sitrus"), None)
            if t is not None and mm.side == self.opp and mm.hp > 0 and mm.item_unknown():
                self.lik(mm, "item", "sitrusberry", NO, "fell to half HP or less without eating a Sitrus Berry")


def evidence_for(battle) -> EvidenceTracker:
    """The battle's evidence tracker (opponent side), advanced over the lines poke-env stored since the last call."""
    tr = getattr(battle, "_arena_opp_evidence", None)
    if tr is None:
        role = battle.player_role
        tr = EvidenceTracker("p2" if role == "p1" else "p1")
        if role is None:                     # not known yet: do not pin the wrong side
            tr.feed(getattr(battle, "_replay_data", None) or [])
            return tr
        try:
            battle._arena_opp_evidence = tr
        except AttributeError:
            pass
    data = getattr(battle, "_replay_data", None) or []
    if tr.cursor < len(data):
        tr.feed(data[tr.cursor:])
        tr.cursor = len(data)
    return tr
