"""``TeamAwareOpponentModel``: the ``OpponentModel`` protocol (``set_prior`` / ``estimate``) on top of the compiled
knowledge, the battle's evidence tracker and the posterior. It returns the same ``SetPrior`` as the plain prior, so
the damage calc, speed checks, System 1 facts and LLM tools use it unchanged; ``belief`` gives the richer view
(evidence notes, original item) and ``profile`` the team-level view.

One belief per (battle, Pokemon, what is known about it): recomputed only when a reveal or a piece of evidence
changes it, so the many calls per decision cost a dict lookup.
"""

from __future__ import annotations

from typing import Optional

from poke_env.data import to_id_str

from arena.knowledge import store
from arena.knowledge.estimate import StatEstimate, estimate_stats, is_own
from arena.knowledge.priors import Choice, SetPrior
from arena.observation.tracker import mon_key
from arena.opponent.evidence import evidence_for
from arena.opponent.kb import kb_for, resolve_species
from arena.opponent.posterior import DEFAULT_PARAMS, Belief, Params, belief
from arena.state.observe import known_ability, known_item, revealed_moves

MINUS_SPE = {"brave", "quiet", "relaxed", "sassy"}


def _name(kind: str, xid: str) -> str:
    if kind == "tera":
        return xid.title()
    table = {"item": "items", "ability": "abilities", "move": "moves"}[kind]
    e = (store.dex().get(table) or {}).get(xid) or {}
    if xid == "nothing":
        return "No Item"
    return e.get("name", xid)


def _choices(kind: str, d: dict) -> tuple:
    if kind == "tera":
        return tuple(Choice(x.title(), x.title(), round(p, 4)) for x, p in d.items())
    return tuple(Choice(x, _name(kind, x), round(p, 4)) for x, p in d.items())


def _ident(battle, mon) -> Optional[str]:
    return next((k for k, v in battle.opponent_team.items() if v is mon), None)


class TeamAwareOpponentModel:
    name = "team"

    def __init__(self, kb_path: Optional[str] = None, params: Params = DEFAULT_PARAMS, use_evidence: bool = True,
                 use_damage: bool = True):
        self.kb_path, self.params, self.use_evidence, self.use_damage = kb_path, params, use_evidence, use_damage
        self._cache: dict = {}

    # --- speed: what we knew at the start of each turn, joined later with who moved first ---------------------
    def on_request(self, battle) -> None:
        """At a move request: record our active's effective Speed and the opponent active's Speed multipliers
        (boosts, paralysis, Tailwind, its possible abilities under this field) for this turn."""
        if getattr(battle, "force_switch", False) or getattr(battle, "teampreview", False):
            return
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        if me is None or opp is None or not self._kb(battle):
            return
        snaps = getattr(battle, "_arena_speed_snaps", None)
        if snaps is None:
            snaps = {}
            try:
                battle._arena_speed_snaps = snaps
            except AttributeError:
                return
        if battle.turn in snaps:
            return
        try:
            from poke_env.battle.field import Field
            from arena.augment.speed import _mult, speed_band
            ours = speed_band(battle, me)[1]
            abilities = [opp.ability] if opp.ability else sorted(getattr(opp, "possible_abilities", None) or [None])
            mults = [_mult(battle, opp, False, ability=a) for a in abilities]
        except Exception:  # noqa: BLE001
            return
        our_ident = next((k for k, v in battle.team.items() if v is me), None)
        opp_ident = _ident(battle, opp)
        if ours is None or not mults or our_ident is None or opp_ident is None:
            return
        snap = {"our_key": mon_key(our_ident), "opp_key": mon_key(opp_ident), "our_eff": float(ours),
                "our_species": me.species, "m_lo": min(mults), "m_hi": max(mults),
                "trick_room": Field.TRICK_ROOM in (battle.fields or {}), "hits": {}}
        snaps[battle.turn] = snap
        if not self.use_damage:
            return
        try:                                   # damage of its likely moves on our active, per item x attacking stat
            from arena.opponent.damage_evidence import hit_table
            b = self.belief(battle, opp)
            if b is not None:
                seen = set(b.revealed.get("moves") or ())
                likely = [m for m, q in b.moves.items() if m not in seen and q >= 0.1][:4]
                snap["hits"] = hit_table(battle, opp, me, b, list(seen) + likely)
        except Exception:  # noqa: BLE001
            snap["hits"] = {}

    def damage_obs(self, battle, ev, ident: Optional[str]) -> list:
        snaps = getattr(battle, "_arena_speed_snaps", None) or {}
        if not snaps or ev is None or ident is None or not self.use_damage:
            return []
        key = mon_key(ident)
        out = []
        for h in ev.hits:
            sn = snaps.get(h["turn"])
            if h["opp_key"] != key or sn is None or sn["opp_key"] != key or sn["our_key"] != h["our_key"]:
                continue
            tab = (sn.get("hits") or {}).get(h["move"])
            if not tab:
                continue
            out.append({**tab, "damage": h["damage"],
                        "note": f"{h['move']} took {h['damage']:.0f} HP from your {sn['our_species']} on turn {h['turn']}"})
        return out

    def speed_obs(self, battle, ev, ident: Optional[str]) -> list:
        snaps = getattr(battle, "_arena_speed_snaps", None) or {}
        if not snaps or ev is None or ident is None:
            return []
        key = mon_key(ident)
        out = []
        for o in ev.orders:
            sn = snaps.get(o["turn"])
            if o["opp_key"] != key or sn is None or sn["opp_key"] != key or sn["our_key"] != o["our_key"]:
                continue
            how = "before" if o["opp_first"] else "after"
            out.append({**sn, "opp_first": o["opp_first"],
                        "note": f"moved {how} your {sn['our_species']} (Speed {sn['our_eff']:.0f}) on turn {o['turn']}"
                                + (" under Trick Room" if sn["trick_room"] else "")})
        return out

    def _kb(self, battle) -> dict:
        return kb_for(getattr(battle, "format", "") or "", self.kb_path)

    def available(self, battle) -> bool:
        return bool(self._kb(battle))

    def mates(self, battle, kb: dict, sid: str) -> list[str]:
        seen = [p.species for p in (getattr(battle, "teampreview_opponent_team", None) or [])]
        seen += [p.species for p in battle.opponent_team.values()]
        out = []
        for s in seen:
            r = resolve_species(kb, s)
            if r and r != sid and r not in out:
                out.append(r)
        return out[:5] if len(out) > 5 else out

    def belief(self, battle, mon) -> Optional[Belief]:
        if is_own(battle, mon):
            return None
        kb = self._kb(battle)
        if not kb:
            return None
        sid = resolve_species(kb, mon.species)
        if sid is None:
            return None
        ident = _ident(battle, mon)
        ev = evidence_for(battle) if ident and self.use_evidence else None
        me = ev.mons.get(mon_key(ident)) if ev is not None else None
        moves = list(revealed_moves(mon))
        if me is not None:
            moves += [m for m in me.moves if m not in moves]
        item_now = known_item(mon) or (me.held() if me else None)
        item_gone = (mon.item in ("", None)) or bool(me and me.item_gone)
        original = me.first_item if me is not None else None      # the item it started with, once known
        tera = None
        if getattr(mon, "is_terastallized", False) and mon.tera_type is not None:
            tera = mon.tera_type.name.lower()
        elif me is not None and me.tera:
            tera = me.tera
        revealed = {"moves": moves, "item": item_now if not item_gone else None,
                    "ability": to_id_str(known_ability(mon) or (me.ability if me else "") or "") or None, "tera": tera}
        mates = self.mates(battle, kb, sid)
        speed = self.speed_obs(battle, ev, ident)
        hits = self.damage_obs(battle, ev, ident)
        key = (getattr(battle, "battle_tag", ""), ident or mon.species, sid, tuple(mates), tuple(moves),
               revealed["item"], revealed["ability"], tera, original, item_gone, me.version if me else 0,
               len(speed), len(hits))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        notes = list(dict.fromkeys(v for (a, _x), v in (me.notes.items() if me else []) if a in ("item", "ability")))
        b = belief(kb, sid, mates, revealed, lik_item=(me.lik_item if me else None),
                   lik_ability=(me.lik_ability if me else None), original_item=original, item_gone=item_gone,
                   notes=notes, speed_obs=speed, damage_obs=hits, params=self.params)
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = b
        return b

    def set_prior(self, battle, mon) -> Optional[SetPrior]:
        b = self.belief(battle, mon)
        if b is None:
            from arena.knowledge.opponent_model import PRIOR_MODEL
            return PRIOR_MODEL.set_prior(battle, mon)
        return to_set_prior(b, int(getattr(mon, "level", 100) or 100))

    def estimate(self, battle, mon) -> Optional[StatEstimate]:
        return estimate_stats(battle, mon, self.set_prior(battle, mon))


def to_set_prior(b: Belief, level: int = 100) -> SetPrior:
    spreads = []
    for nature, evs, p in b.spreads:
        ivs = [31] * 6
        if nature in MINUS_SPE and evs[5] == 0:
            ivs[5] = 0                       # a minus-Speed nature with no Speed EVs is a Trick Room set
        spreads.append({"nature": nature, "evs": list(evs), "ivs": ivs, "p": p})
    revealed = {}
    if b.revealed.get("moves"):
        revealed["moves"] = list(b.revealed["moves"])
    for k in ("item", "ability"):
        if b.revealed.get(k):
            revealed[k] = b.revealed[k]
    if b.revealed.get("tera"):
        revealed["tera"] = b.revealed["tera"].title()
    return SetPrior(species=b.species, name=b.name, level=level, source="gen9ou+team", n=b.n,
                    moves=_choices("move", b.moves), items=_choices("item", b.items),
                    abilities=_choices("ability", b.abilities), tera=_choices("tera", b.tera),
                    spreads=tuple(spreads), revealed=revealed, notes=tuple(b.notes),
                    original_item=b.original_item)


def export_beliefs(battle, model: Optional[TeamAwareOpponentModel] = None, k_moves: int = 20) -> dict:
    """{species id: {items, abilities, tera: {id: p}, moves: {id: P(in set)}}} for every opposing Pokemon known so far
    (team preview included), keyed by our id and poke-env's -- the look-ahead service weights its sampled sets by it."""
    if model is None:
        from arena.knowledge.opponent_model import team_model
        model = team_model()
    seen = list(battle.opponent_team.values())
    names = {to_id_str(p.species) for p in seen}
    mons = seen + [p for p in (getattr(battle, "teampreview_opponent_team", None) or [])
                   if to_id_str(p.species) not in names]
    out = {}
    for mon in mons:
        b = model.belief(battle, mon)
        if b is None:
            continue
        d = {"items": {x: round(p, 4) for x, p in b.items.items() if p >= 0.005},
             "abilities": {x: round(p, 4) for x, p in b.abilities.items() if p >= 0.005},
             "tera": {x: round(p, 4) for x, p in b.tera.items() if p >= 0.005},
             "moves": {x: round(p, 4) for x, p in list(b.moves.items())[:k_moves]},
             "spreads": [[n, list(e), p] for n, e, p in b.spreads[:6]]}
        kb = model._kb(battle)
        entry = (kb.get("species") or {}).get(b.species)
        if entry is not None and not b.revealed.get("item") and not b.original_item:
            from arena.opponent.posterior import spreads_posterior
            d["spreads_by_item"] = {x: [[n, list(e), p] for n, e, p in
                                        spreads_posterior(entry, {x: 1.0}, None, model.params, keep=4)]
                                    for x, p in b.items.items() if p >= 0.05}
        for key in {b.species, to_id_str(mon.species)}:
            out[key] = d
    return out


STAT_NAMES = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")


def spread_text(nature: str, evs) -> str:
    ev = " / ".join(f"{v} {n}" for v, n in zip(evs, STAT_NAMES) if v)
    return f"{nature.title()} {ev}" if ev else nature.title()


def belief_text(b: Belief, *, k_items: int = 3, k_moves: int = 4, notes: int = 3) -> str:
    """One line for an LLM view: what this Pokemon most likely holds and why."""
    bits = []
    if b.revealed.get("item"):
        bits.append(f"item {_name('item', b.revealed['item'])} (revealed)")
    elif b.original_item:
        bits.append(f"no item now (had {_name('item', b.original_item)})")
    else:
        bits.append("item " + ", ".join(f"{_name('item', x)} {p:.0%}" for x, p in list(b.items.items())[:k_items]))
    if not b.revealed.get("ability") and len(b.abilities) > 1:
        bits.append("ability " + ", ".join(f"{_name('ability', x)} {p:.0%}" for x, p in list(b.abilities.items())[:2]))
    if not b.revealed.get("tera"):
        bits.append("tera " + ", ".join(f"{x.title()} {p:.0%}" for x, p in list(b.tera.items())[:2]))
    seen = set(b.revealed.get("moves") or ())
    other = [(m, p) for m, p in b.moves.items() if m not in seen][:max(0, min(k_moves, 4 - len(seen)) or 0)]
    if other:
        bits.append("unrevealed moves likely " + ", ".join(f"{_name('move', m)} {p:.0%}" for m, p in other))
    if b.spreads:
        n, evs, p = b.spreads[0]
        bits.append(f"spread {spread_text(n, evs)} {p:.0%}")
    line = f"{b.name}: " + "; ".join(bits)
    if b.notes and notes:
        line += " [" + "; ".join(b.notes[:notes]) + "]"
    return line
