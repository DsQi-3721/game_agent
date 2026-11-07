"""Program-truth questions for the memory router: "is `note` still true now?" (state v2, I/O audit batch 6).

Every note is a structured ``Claim`` rendered as text. Its label is computed by ``evaluate(claim, obs)`` on
the PlayerObservation -- a check written independently of the generator's choice of what to claim: the
generator aims for a polarity, the checker decides, and a disagreement is counted and dropped, never
relabelled (``generate`` reports how many). The model is shown the S1 state built from the same observation
(``facts.build_state``), which carries the evidence every family needs: the own team with fainted ones, the
opponent's active, seen bench (HP as last seen), team preview, effects and last moves, items with what was
lost, abilities, Tera used vs available now, hazards with layers, weather and terrain.

Truth values (soft labels): true 0.95, false 0.05, unknown 0.5 --
  true      the state confirms the note (for a benched opponent: consistent with what was last seen)
  false     the state contradicts it
  unknown   the state neither confirms nor contradicts it (an unrevealed item, a plausible unrevealed move,
            an ability among those the species may have)
Every name is normalised to a Showdown id before comparing (Protect / protect, Knock Off / knockoff); HP
claims say "about" when they allow a tolerance (3 points), and are exact otherwise; every note says whose
Pokemon it is about (both teams can field the same species).
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from poke_env.data import GenData, to_id_str

from arena.knowledge import store

TRUE, FALSE, UNKNOWN = 0.95, 0.05, 0.5
LABEL = {"true": TRUE, "false": FALSE, "unknown": UNKNOWN}
HP_TOL = 3
STATUSES = {"brn": ["burned", "is suffering from a burn"], "par": ["paralyzed", "is paralyzed"],
            "psn": ["poisoned", "is poisoned"], "tox": ["badly poisoned", "is badly poisoned (Toxic)"],
            "slp": ["asleep", "is asleep"], "frz": ["frozen", "is frozen"]}
STATS = {"atk": "Attack", "def": "Defense", "spa": "Special Attack", "spd": "Special Defense", "spe": "Speed",
         "accuracy": "accuracy", "evasion": "evasion"}
WEATHERS = {"sunnyday": "harsh sunlight", "raindance": "rain", "sandstorm": "a sandstorm", "snowscape": "snow"}
FIELDS = {"electricterrain": "Electric Terrain", "grassyterrain": "Grassy Terrain", "mistyterrain": "Misty Terrain",
          "psychicterrain": "Psychic Terrain", "trickroom": "Trick Room"}
HAZARDS = {"stealthrock": "Stealth Rock", "spikes": "Spikes", "toxicspikes": "Toxic Spikes", "stickyweb": "Sticky Web",
           "reflect": "Reflect", "lightscreen": "Light Screen", "auroraveil": "Aurora Veil", "tailwind": "Tailwind"}
ITEMS = ["leftovers", "choicescarf", "choiceband", "choicespecs", "heavydutyboots", "lifeorb", "focussash",
         "rockyhelmet", "assaultvest", "boosterenergy", "airballoon", "sitrusberry"]
ABILITIES = ["levitate", "intimidate", "regenerator", "sturdy", "protosynthesis", "multiscale", "unaware", "magicguard"]
DEFAULT_MOVES = ["earthquake", "protect", "uturn", "knockoff", "stealthrock"]
BANDS = {"at full HP": lambda v: v >= 100, "above half HP": lambda v: v > 50, "below half HP": lambda v: v <= 50,
         "in red HP": lambda v: v <= 25, "badly damaged (under a third)": lambda v: v < 34,
         "healthy (over 75%)": lambda v: v > 75}
TAILS = ["", "", "", " Keep this in mind.", " Plan around it.", " This matters for the next turns.",
         " Noted from the battle log.", " Worth tracking.", " so switching is risky.", " which changes the damage math.",
         " -> consider pivoting.", " (confirmed by the log)", " according to the last turn summary.",
         " That limits our options next turn.", " so prioritise a safe KO."]
HEADS = ["", "", "", "", "Note: ", "Observed: ", "From the battle log: ", "Turn {t}: ", "Update (turn {t}): ",
         "Key fact: ", "Summary: ", "Important - ", "[episode] ", "[working] "]


@dataclass
class Claim:
    family: str
    side: str                      # you | opp | field
    predicate: str
    subject: Optional[str] = None  # species id
    value: Any = None
    parts: list = field(default_factory=list)   # compound: sub-claims


# ---- the checker ---------------------------------------------------------------------------------------------

def _own(obs) -> dict:
    return {to_id_str(d["species"]): d for d in obs.own_team if d.get("species")}


def _opp(obs) -> dict:
    return {to_id_str(d["species"]): d for d in obs.opponent_revealed_team}


def _preview(obs) -> set:
    return {to_id_str(p["details"].split(",")[0].removesuffix("-*")) for p in obs.opponent_preview}


def _hp(d: dict, own: bool) -> int:
    if own:
        return round(100 * d["hp"][0] / d["hp"][1]) if d["hp"][1] else 0
    return d["hp_pct"]


def evaluate(c: Claim, obs) -> str:
    """true / false / unknown for a claim, from the observation alone (independent of how it was made)."""
    if c.family == "compound":
        verdicts = [evaluate(p, obs) for p in c.parts]
        return "false" if "false" in verdicts else "unknown" if "unknown" in verdicts else "true"
    own, opp = _own(obs), _opp(obs)
    team = own if c.side == "you" else opp
    p = c.predicate
    d = team.get(c.subject) if c.subject else None
    if p == "fainted":                      # never sent out means never knocked out
        if d is not None and d.get("fainted_unknown"):
            return "unknown"                # a converted replay that lost which opponent fainted
        if d is None and c.side == "opp" and "opp_faint_identity" in (obs.unrecorded or []):
            return "unknown"                # an unseen Pokemon may be the one that fainted
        return "true" if d is not None and d.get("fainted") else "false"
    if p == "brought":
        if c.subject in _preview(obs) or c.subject in opp:
            return "true"
        return "false" if obs.opponent_preview else "unknown"
    if p in ("hp_exact", "hp_about", "hp_band", "status", "boost", "holds", "lost_item", "ability", "used_move",
             "not_used_move", "active"):
        if d is None:                       # a Pokemon this side has not shown
            if p in ("hp_exact", "hp_about"):
                return "false" if c.value < 100 else "unknown"   # an unseen Pokemon is untouched
            if p in ("used_move", "lost_item", "active", "boost", "status"):
                return "false"
            return "true" if p == "not_used_move" else "unknown"
        if d.get("fainted") and p not in ("used_move", "not_used_move", "lost_item"):
            return "unknown" if d.get("fainted_unknown") else "false"   # unknown: may have been revived
    if p == "hp_exact":
        return "true" if _hp(d, c.side == "you") == c.value else "false"
    if p == "hp_about":
        diff = abs(_hp(d, c.side == "you") - c.value)
        return "true" if diff <= HP_TOL else "false" if diff >= 15 else "unknown"
    if p == "hp_band":
        return "true" if BANDS[c.value](_hp(d, c.side == "you")) else "false"
    if p == "active":
        return "true" if d.get("active") else "false"
    if p == "status":
        return "true" if (d.get("status") or "").lower() == c.value else "false"
    if p == "boost":
        if not d.get("active"):
            return "false"
        return "true" if (d.get("boosts") or {}).get(c.value[0], 0) == c.value[1] else "false"
    if p in ("used_move", "not_used_move"):
        used = to_id_str(c.value) in {m["id"] for m in d.get("revealed_moves") or []}
        if p == "used_move":                # not seen yet: it may still be in the set, if it can learn it
            return "true" if used else ("unknown" if can_learn(d["species"], c.value) else "false")
        return "false" if used else "true"  # every move used is public
    if p == "holds":
        item = d["item"]
        if c.side == "you":
            return "true" if (item.get("value") or "") == c.value else "false"
        if item["status"] == "known":
            return "true" if item["value"] == c.value else "false"
        return "false" if item["status"] == "none" else "unknown"
    if p == "lost_item":
        item = d["item"]
        return "true" if item.get("status") == "none" and item.get("lost") == c.value else "false"
    if p == "ability":
        if c.side == "you":
            return "true" if d.get("ability") == c.value else "false"
        ab = d["ability"]
        if ab["status"] == "known":
            return "true" if ab["value"] == c.value else "false"
        return "unknown" if c.value in (ab.get("candidates") or []) else "false"
    if p == "tera_used":
        used = obs.tera["own" if c.side == "you" else "opponent"].get("used")
        return "unknown" if used is None else ("true" if used == c.value else "false")
    if p == "left":
        return "true" if obs.pokemon_left["own" if c.side == "you" else "opponent"] == c.value else "false"
    if p == "weather":
        return "true" if (obs.weather or {}).get("weather") == c.value else "false"
    if p == "field":
        fields = {f["field"] for f in obs.field_conditions}
        if c.value not in fields and fields and "fields_beyond_most_recent" in (obs.unrecorded or []):
            return "unknown"                # a converted replay keeps only the most recent field
        return "true" if c.value in fields else "false"
    if p == "hazard":
        side_key, cond, layers = c.value
        conds = {x["condition"]: x for x in obs.side_conditions[side_key]}
        if cond not in conds:
            partial = conds and "side_conditions_beyond_most_recent" in (obs.unrecorded or [])
            return "unknown" if partial else "false"   # a converted replay keeps one condition per side
        if layers is None:
            return "true"
        known = conds[cond].get("layers")
        return "unknown" if known is None else ("true" if known == layers else "false")
    raise ValueError(f"unknown predicate {p}")


def can_learn(species: str, move: str) -> bool:
    """Whether a species (or the base species of its forme) can have the move at all (Gen 9 learnsets)."""
    data = GenData.from_gen(9)
    sid, mid = to_id_str(species), to_id_str(move)
    for cand in (sid, to_id_str((data.pokedex.get(sid) or {}).get("baseSpecies") or ""),
                 to_id_str((data.pokedex.get(sid) or {}).get("changesFrom") or "")):
        if cand and mid in ((data.learnset.get(cand) or {}).get("learnset") or {}):
            return True
    return False


# ---- the generator -------------------------------------------------------------------------------------------

def _display(sid: str) -> str:
    return (GenData.from_gen(9).pokedex.get(sid) or {}).get("name", sid)


def _move_name(mid: str) -> str:
    return (GenData.from_gen(9).moves.get(to_id_str(mid)) or {}).get("name", mid)


def _item_name(iid: str) -> str:
    return (store.dex().get("items", {}).get(iid) or {}).get("name", iid)


def _ability_name(aid: str) -> str:
    return (store.dex().get("abilities", {}).get(aid) or {}).get("name", aid)


class Generator:
    """Claims about one observation, aiming at a polarity; the text says exactly what the claim says."""

    def __init__(self, obs, state: dict, rng: random.Random, foreign: list[str]):
        self.obs, self.state, self.rng, self.foreign = obs, state, rng, foreign
        self.own, self.opp = _own(obs), _opp(obs)
        self.in_battle = set(self.own) | set(self.opp) | _preview(obs)

    def _name(self, sid: str) -> str:
        return sid if self.rng.random() < 0.5 else _display(sid)

    def _subject(self, side: str, alive: bool = True, active: Optional[bool] = None) -> Optional[str]:
        team = self.own if side == "you" else self.opp
        pool = [s for s, d in team.items() if (not alive or not d.get("fainted"))
                and (active is None or bool(d.get("active")) == active)]
        return self.rng.choice(pool) if pool else None

    @staticmethod
    def _who(side: str) -> str:
        return "Your" if side == "you" else "Opponent's"

    def hp(self, side: str, want: str):
        s = self._subject(side)
        if s is None:
            return None
        hp = _hp((self.own if side == "you" else self.opp)[s], side == "you")
        rng, n = self.rng, self._name(s)
        far = [x for x in range(5, 101, 5) if abs(x - hp) >= 15]
        if rng.random() < 0.5:
            v = hp if want == "true" else (rng.choice(far) if far else None)
            if v is None:
                return None
            return Claim("hp", side, "hp_exact", s, v), rng.choice(
                [f"{self._who(side)} {n} is at {v}%.", f"{n} ({'ours' if side == 'you' else 'theirs'}) has {v}% HP left.",
                 f"HP check: {'you' if side == 'you' else 'opp'} {n} {v}/100."])
        v = max(1, min(100, hp + rng.randint(-HP_TOL, HP_TOL))) if want == "true" else (rng.choice(far) if far else None)
        if v is None:
            return None
        return Claim("hp", side, "hp_about", s, v), rng.choice(
            [f"{self._who(side)} {n} is at about {v}%.", f"{self._who(side)} {n} is around {v}% HP.",
             f"{'you' if side == 'you' else 'opp'} {n}: roughly {v}% left."])

    def hp_band(self, side: str, want: str):
        s = self._subject(side)
        if s is None:
            return None
        hp = _hp((self.own if side == "you" else self.opp)[s], side == "you")
        ok = [b for b, f in BANDS.items() if f(hp) == (want == "true")]
        if not ok:
            return None
        band = self.rng.choice(ok)
        return Claim("hp_band", side, "hp_band", s, band), f"{self._who(side)} {self._name(s)} is {band}."

    def fainted(self, side: str, want: str):
        team = self.own if side == "you" else self.opp
        dead = [s for s, d in team.items() if d.get("fainted")]
        alive = [s for s, d in team.items() if not d.get("fainted")]
        if side == "opp":
            alive += sorted(_preview(self.obs) - set(self.opp))
        pool = dead if want == "true" else alive
        if not pool:
            return None
        s = self.rng.choice(pool)
        n = self._name(s)
        return Claim("fainted", side, "fainted", s), self.rng.choice(
            [f"{self._who(side)} {n} has fainted.", f"{self._who(side)} {n} was knocked out.",
             f"{'Our' if side == 'you' else 'Their'} {n} is out of the battle.",
             f"{self._who(side)} {n} fainted earlier in the battle."])

    def active(self, side: str, want: str):
        s = self._subject(side, active=(want == "true"))
        if s is None:
            return None
        n, whose = self._name(s), ("the opponent's" if side == "opp" else "our")
        return Claim("active", side, "active", s), self.rng.choice(
            [f"{n} is {whose} active Pokemon.", f"{self._who(side)} {n} is on the field now.",
             f"{'Our' if side == 'you' else 'Their'} {n} is the one in battle."])

    def status(self, side: str, want: str):
        s = self._subject(side)
        if s is None:
            return None
        cur = ((self.own if side == "you" else self.opp)[s].get("status") or "").lower()
        if want == "true":
            if cur not in STATUSES:
                return None
            st = cur
        else:
            st = self.rng.choice([k for k in STATUSES if k != cur])
        word, n = self.rng.choice(STATUSES[st]), self._name(s)
        return Claim("status", side, "status", s, st), self.rng.choice(
            [f"{self._who(side)} {n} {word if word.startswith('is') else 'is ' + word}.",
             f"{self._who(side)} {n} got {STATUSES[st][0]}.",
             f"[{'you' if side == 'you' else 'opp'}] {n} status: {st}"])

    def boost(self, side: str, want: str):
        s = self._subject(side, active=True)
        if s is None:
            return None
        ch = (self.own if side == "you" else self.opp)[s].get("boosts") or {}
        if want == "true":
            if not ch:
                return None
            stat, v = self.rng.choice(list(ch.items()))
        else:
            stat = self.rng.choice(["atk", "spa", "spe", "def"])
            v = self.rng.choice([x for x in (-2, -1, 1, 2, 3, 6) if x != ch.get(stat, 0)])
        sign, n = (f"+{v}" if v > 0 else str(v)), self._name(s)
        return Claim("boost", side, "boost", s, (stat, v)), self.rng.choice(
            [f"{self._who(side)} {n} is at {sign} {STATS[stat]}.",
             f"{self._who(side)} {n} has {sign} {STATS[stat]} from boosts."])

    def opp_move(self, side: str, want: str):
        s = self._subject("opp", alive=False)
        if s is None:
            return None
        d = self.opp[s]
        rev = [m["id"] for m in d.get("revealed_moves") or []]
        likely = [to_id_str(m) for m in (self.state.get("opponent", {}).get("active", {}).get("likely_moves") or [])
                  if d.get("active")]
        unseen = [m for m in (likely or DEFAULT_MOVES) if m not in rev]
        plausible = [m for m in unseen if can_learn(s, m)]
        n, rng = self._name(s), self.rng
        if rng.random() < 0.4:                 # "has not used X yet": true unless X was revealed
            pool = unseen if want == "true" else rev if want == "false" else []
            if not pool:
                return None
            mv = rng.choice(pool)
            disp = _move_name(mv) if rng.random() < 0.6 else mv
            return Claim("opp_move", "opp", "not_used_move", s, mv), rng.choice(
                [f"Opponent's {n} has not used {disp} yet.", f"Opponent's {n} hasn't revealed {disp} so far.",
                 f"No {disp} from opp {n} until now."])
        pool = rev if want == "true" else (plausible if want == "unknown" else
                                           [m for m in DEFAULT_MOVES if m not in rev and not can_learn(s, m)])
        if not pool:
            return None
        mv = rng.choice(pool)
        disp = _move_name(mv) if rng.random() < 0.6 else mv
        return Claim("opp_move", "opp", "used_move", s, mv), rng.choice(
            [f"Opponent's {n} has {disp}.", f"Opponent's {n} revealed {disp}.", f"opp {n} used {disp} earlier.",
             f"Watch out: their {n} has shown {disp}."])

    def item(self, side: str, want: str):
        s = self._subject(side)
        if s is None:
            return None
        item = (self.own if side == "you" else self.opp)[s]["item"]
        rng, n = self.rng, self._name(s)
        if side == "opp" and item["status"] == "none" and item.get("lost") and rng.random() < 0.5:
            lost = item["lost"]
            if want == "true":
                return Claim("item", side, "lost_item", s, lost), rng.choice(
                    [f"Opponent's {n} lost its {_item_name(lost)}.", f"Opponent's {n} no longer has {_item_name(lost)}."])
            if want == "false":                # held before, gone now: "holds" is out of date
                return Claim("item", side, "holds", s, lost), rng.choice(
                    [f"Opponent's {n} holds {_item_name(lost)}.", f"[opp] {n} @ {_item_name(lost)}"])
            return None
        known = item.get("value") if item["status"] == "known" else None
        if want == "true":
            if not known:
                return None
            v = known
        elif want == "false":
            if side == "opp" and item["status"] == "unknown":
                return None
            v = rng.choice([x for x in ITEMS if x != known])
        else:
            if side == "you" or item["status"] != "unknown":
                return None
            v = rng.choice(ITEMS)
        return Claim("item", side, "holds", s, v), rng.choice(
            [f"{self._who(side)} {n} holds {_item_name(v)}.", f"{self._who(side)} {n}'s item is {v}.",
             f"[{side}] {n} @ {_item_name(v)}"])

    def ability(self, side: str, want: str):
        s = self._subject(side)
        if s is None:
            return None
        d = (self.own if side == "you" else self.opp)[s]
        rng, n = self.rng, self._name(s)
        if side == "you":
            known, cands = d.get("ability"), []
        else:
            known = d["ability"]["value"] if d["ability"]["status"] == "known" else None
            cands = d["ability"].get("candidates") or []
        if want == "true":
            if not known:
                return None
            v = known
        elif want == "unknown":
            if known or not cands:
                return None
            v = rng.choice(cands)
        else:
            v = rng.choice([a for a in ABILITIES if a != known and a not in cands])
        return Claim("ability", side, "ability", s, v), rng.choice(
            [f"{self._who(side)} {n} has the ability {_ability_name(v)}.", f"{self._who(side)} {n}'s ability: {v}.",
             f"{'Our' if side == 'you' else 'Their'} {n} runs {_ability_name(v)}."])

    def tera(self, side: str, want: str):
        used = self.obs.tera["own" if side == "you" else "opponent"].get("used")
        if used is None:
            return None
        claim_used = used if want == "true" else (not used)
        rng = self.rng
        if side == "you":
            text = rng.choice(["We have already used our Tera.", "No Tera left for us."]) if claim_used else \
                rng.choice(["We can still terastallize this battle.", "Our Tera is still available."])
        else:
            text = rng.choice(["The opponent has already used its Tera.", "Their Tera is spent."]) if claim_used else \
                rng.choice(["The opponent has not terastallized yet.", "The opponent still has its Tera."])
        return Claim("tera", side, "tera_used", None, claim_used), text

    def left(self, side: str, want: str):
        n = self.obs.pokemon_left["own" if side == "you" else "opponent"]
        v = n if want == "true" else self.rng.choice([x for x in range(1, 7) if x != n])
        who = "The opponent has" if side == "opp" else "We have"
        return Claim("left", side, "left", None, v), self.rng.choice(
            [f"{who} {v} Pokemon left.", f"{v} Pokemon remaining on {'their' if side == 'opp' else 'our'} side."])

    def weather(self, side: str, want: str):
        rng = self.rng
        if rng.random() < 0.5:
            cur = (self.obs.weather or {}).get("weather")
            w = cur if want == "true" else rng.choice([x for x in WEATHERS if x != cur])
            if w is None:
                return None
            return Claim("weather", "field", "weather", None, w), rng.choice(
                [f"{WEATHERS.get(w, w).capitalize()} is active.", f"Weather: {WEATHERS.get(w, w)}."])
        cur = [f["field"] for f in self.obs.field_conditions]
        if want == "true":
            if not cur:
                return None
            f = rng.choice(cur)
        else:
            f = rng.choice([x for x in FIELDS if x not in cur])
        return Claim("weather", "field", "field", None, f), rng.choice(
            [f"{FIELDS.get(f, f)} is up.", f"The field is under {FIELDS.get(f, f)}."])

    def hazard(self, side: str, want: str):
        rng = self.rng
        key = "own" if side == "you" else "opponent"
        conds = {c["condition"]: c for c in self.obs.side_conditions[key]}
        whose = "our side" if side == "you" else "the opponent's side"
        layers = None
        if want == "true":
            if not conds:
                return None
            cid = rng.choice(sorted(conds))
            if cid in ("spikes", "toxicspikes") and rng.random() < 0.5:
                layers = conds[cid].get("layers")
        else:
            stacked = [c for c in conds if c in ("spikes", "toxicspikes") and conds[c].get("layers")]
            if stacked and rng.random() < 0.4:
                cid = rng.choice(stacked)
                layers = rng.choice([x for x in (1, 2, 3) if x != conds[cid]["layers"]])
            else:
                cid = rng.choice([x for x in HAZARDS if x not in conds])
        name = HAZARDS.get(cid, cid)
        text = (f"{layers} layer{'s' if layers > 1 else ''} of {name} on {whose}." if layers else
                rng.choice([f"{name} is set on {whose}.", f"{name} up on {whose}."]))
        return Claim("hazard", "field", "hazard", None, (key, cid, layers)), text

    def brought(self, side: str, want: str):
        if not self.obs.opponent_preview:
            return None
        pool = sorted(_preview(self.obs)) if want == "true" else [f for f in self.foreign if f not in self.in_battle]
        if not pool:
            return None
        s = self.rng.choice(pool)
        n = self._name(s)
        return Claim("brought", "opp", "brought", s), self.rng.choice(
            [f"The opponent brought {n}.", f"{n} is on the opponent's team.", f"Their team includes {n}."])

    def wrong_battle(self, side: str, want: str):
        cands = [f for f in self.foreign if f not in self.in_battle]
        if not cands:
            return None
        s = self.rng.choice(cands)
        n = self._name(s)
        mode = self.rng.randrange(3)
        if mode == 0:
            return Claim("wrong_battle", "opp", "hp_exact", s, 40), f"Opponent's {n} is at 40%."
        if mode == 1:
            return Claim("wrong_battle", "opp", "used_move", s, "earthquake"), f"Opponent's {n} revealed Earthquake."
        return Claim("wrong_battle", "opp", "fainted", s), f"Opponent's {n} fainted last turn."


FAMILIES = ["hp", "hp_band", "fainted", "active", "status", "boost", "opp_move", "item", "ability", "tera", "left",
            "weather", "hazard", "brought", "wrong_battle", "compound"]
QUOTA = {  # family -> (true, false, unknown) shares of its examples
    "hp": (0.5, 0.5, 0), "hp_band": (0.5, 0.5, 0), "fainted": (0.5, 0.5, 0), "active": (0.5, 0.5, 0),
    "status": (0.5, 0.5, 0), "boost": (0.5, 0.5, 0), "opp_move": (0.4, 0.3, 0.3), "item": (0.35, 0.35, 0.3),
    "ability": (0.4, 0.3, 0.3), "tera": (0.5, 0.5, 0), "left": (0.5, 0.5, 0), "weather": (0.5, 0.5, 0),
    "hazard": (0.5, 0.5, 0), "brought": (0.5, 0.5, 0), "wrong_battle": (0, 1, 0), "compound": (0.5, 0.5, 0)}


def _one(gen: Generator, fam: str, want: str):
    side = "opp" if gen.rng.random() < 0.65 else "you"
    return getattr(gen, fam)("opp" if fam in ("opp_move", "brought", "wrong_battle") else side, want)


def _compound(gen: Generator, want: str):
    """Two claims in one note: true only if both are true, false if either is false."""
    fams = [f for f in FAMILIES if f not in ("wrong_battle", "compound")]
    a = _one(gen, gen.rng.choice(fams), "true")
    b = _one(gen, gen.rng.choice(fams), "true" if want == "true" else "false")
    if a is None or b is None:
        return None
    (ca, ta), (cb, tb) = a, b
    joiner = gen.rng.choice([" Also, ", "; ", ". Meanwhile "])
    return Claim("compound", ca.side, "and", parts=[ca, cb]), ta.rstrip(".") + joiner + tb


def _claim_dict(c: Claim) -> dict:
    return {"family": c.family, "side": c.side, "predicate": c.predicate, "subject": c.subject,
            "value": list(c.value) if isinstance(c.value, tuple) else c.value,
            **({"parts": [_claim_dict(p) for p in c.parts]} if c.parts else {})}


def generate(items: list[tuple[str, Any, dict]], per_family: int, seed: int = 0) -> tuple[list[dict], dict]:
    """Quota sampling over (battle group, PlayerObservation, S1 state) triples -> (rows, report). Rows:
    {"group", "state", "note", "label", "family", "polarity", "side", "claim"}. The report counts, per family,
    the claims the checker decided differently from the generator's aim (dropped, not relabelled)."""
    rng = random.Random(seed)
    species = sorted({s for _g, obs, _st in items for s in (set(_own(obs)) | set(_opp(obs)) | _preview(obs))})
    out, seen, report = [], set(), Counter()
    for fam, shares in QUOTA.items():
        for want, share in zip(("true", "false", "unknown"), shares):
            need = int(round(per_family * share))
            got, tries = 0, 0
            order = list(range(len(items)))
            rng.shuffle(order)
            while items and got < need and tries < need * 60:
                group, obs, st = items[order[tries % len(order)]]
                tries += 1
                gen = Generator(obs, st, rng, rng.sample(species, min(40, len(species))))
                made = _compound(gen, want) if fam == "compound" else _one(gen, fam, want)
                if made is None:
                    continue
                claim, text = made
                verdict = evaluate(claim, obs)
                if verdict != want:
                    report[f"aim_mismatch:{fam}:{want}->{verdict}"] += 1
                    continue
                head = rng.choice(HEADS).format(t=max(1, obs.actual_turn - rng.randint(0, 3)))
                text = head + text + rng.choice(TAILS)
                if text in seen:
                    continue
                seen.add(text)
                out.append({"group": group, "state": st, "note": text, "label": LABEL[verdict], "family": fam,
                            "polarity": verdict, "side": claim.side, "claim": _claim_dict(claim)})
                got += 1
            report[f"made:{fam}:{want}"] += got
    rng.shuffle(out)
    return out, dict(report)


def stats(rows: list[dict]) -> dict:
    """Coverage / diversity numbers for the report."""
    words = [w for r in rows for w in re.findall(r"[a-z]+", r["note"].lower())]
    shapes = {re.sub(r"[0-9]+|[A-Za-z]+", "x", r["note"]) for r in rows}
    return {"n": len(rows), "battles": len({r["group"] for r in rows}),
            "by_family": dict(Counter(r["family"] for r in rows)), "by_polarity": dict(Counter(r["polarity"] for r in rows)),
            "by_side": dict(Counter(r["side"] for r in rows)),
            "family_x_polarity": dict(Counter(f"{r['family']}:{r['polarity']}" for r in rows)),
            "distinct_notes": len({r["note"] for r in rows}), "distinct_shapes": len(shapes),
            "type_token_ratio": round(len(set(words)) / max(1, len(words)), 4),
            "note_chars_p50": sorted(len(r["note"]) for r in rows)[len(rows) // 2] if rows else 0}
