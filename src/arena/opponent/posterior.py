"""P(item / ability / Tera / moves / spread | species, teammates, what it revealed, in-battle evidence).

Categorical attributes (item, ability, Tera type), naive Bayes on top of the base distribution:

    log P(x | F) = log P0(x) + sum_f kappa_f * log( P^(f | x) / P(f) ) + const,
    P^(f | x)    = (c[x][f] + beta * P(f)) / (n[x] + beta)            (shrunk towards P(f))

``P0`` is Smogon usage (what teams hold), ``F`` the observed features: the five teammates (known from team preview),
its revealed moves and its other revealed attributes. ``n``, ``c`` and ``P(f)`` are counted over revealed teams
among Pokemon of this species whose value of the attribute is revealed; the bias of what battles reveal cancels in
the ratio if a value is revealed independently of the features. Pairs whose ratio was ~1 are not stored (a missing
pair is neutral); pairs never seen although expected are stored as 0. ``kappa < 1`` tempers the naive-Bayes
double counting of correlated features. Then the in-battle evidence (``arena.opponent.evidence``) multiplies in.

Moves: each move's inclusion odds get the same treatment with likelihood ratios P(f | move in set) /
P(f | not in set), counted over complete (four revealed) movesets; then a common shift makes the expected number of
unrevealed moves equal the free slots.

Spreads: by item (full sets: a Choice Scarf user runs maximum Speed), mixed with the unconditional usage spreads and
weighted by the item posterior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Params:
    beta: float = 30.0                 # shrinkage pseudo-count of the naive-Bayes factors
    kappa_mate: float = 0.6            # teammates are correlated with each other: temper their votes
    kappa_move: float = 0.8
    kappa_attr: float = 1.0
    temp: dict = field(default_factory=lambda: {"item": 1.0, "ability": 1.0, "tera": 1.0, "move": 1.0})
    floor: float = 1e-4                # smallest probability kept for a base value
    spread_by_item: float = 0.7        # weight of the item-specific spreads against the unconditional ones
    damage_floor: float = 0.2          # likelihood of an (item, spread) a hit's exact damage rules out (swept: 0.05 0.2 0.4)
    use_teammates: bool = True
    use_reveals: bool = True           # revealed moves / attributes as features of the others


DEFAULT_PARAMS = Params()


@dataclass
class Belief:
    species: str
    name: str
    items: dict                        # id -> p (sorted, descending)
    abilities: dict
    tera: dict
    moves: dict                        # id -> P(in the set); revealed moves 1.0
    spreads: list                      # [(nature, [6 EVs], p)]
    revealed: dict                     # moves / item / ability / tera actually seen
    notes: list = field(default_factory=list)      # why the evidence moved something
    original_item: Optional[str] = None            # the item it started with when it is gone now
    n: int = 0


def _sorted(d: dict) -> dict:
    return dict(sorted(d.items(), key=lambda kv: -kv[1]))


def _norm(logp: dict, temp: float = 1.0) -> dict:
    if not logp:
        return {}
    top = max(logp.values())
    w = {k: math.exp((v - top) / temp) for k, v in logp.items()}
    z = sum(w.values()) or 1.0
    return _sorted({k: v / z for k, v in w.items()})


def features(mates, revealed: dict, exclude: Optional[str] = None, params: Params = DEFAULT_PARAMS) -> list:
    """[(feature key, kappa)] observed for one Pokemon. ``exclude``: the attribute being predicted."""
    out = []
    if params.use_teammates:
        out += [(f"m:{m}", params.kappa_mate) for m in mates]
    if params.use_reveals:
        out += [(f"v:{m}", params.kappa_move) for m in revealed.get("moves") or () if exclude != f"v:{m}"]
        for kind, letter in (("item", "i"), ("ability", "a"), ("tera", "t")):
            v = revealed.get(kind)
            if v and kind != exclude:
                out.append((f"{letter}:{v}", params.kappa_attr))
    return out


def categorical(entry: dict, kind: str, feats: list, lik: Optional[dict] = None,
                params: Params = DEFAULT_PARAMS) -> dict:
    key = {"item": "items", "ability": "abilities", "tera": "tera"}[kind]
    base = entry.get(key) or {}
    if not base:
        return {}
    logp = {x: math.log(max(p, params.floor)) for x, p in base.items()}
    nb = (entry.get("nb") or {}).get(kind)
    if nb and feats:
        n, pf, cc, beta = nb["n"], nb["pf"], nb["c"], params.beta
        for x in logp:
            cx = cc.get(x)
            if not cx:
                continue
            nx = n.get(x, 0.0)
            s = 0.0
            for f, kap in feats:
                c = cx.get(f)
                p = pf.get(f)
                if c is None or not p:
                    continue
                s += kap * (math.log((c + beta * p) / (nx + beta)) - math.log(p))
            logp[x] += s
    if lik:
        star = lik.get("*", 1.0)
        for x in logp:
            logp[x] += math.log(max(lik.get(x, star), 1e-9))
    return _norm(logp, params.temp.get(kind, 1.0))


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z > -50 else 0.0


def moves_posterior(entry: dict, feats: list, revealed_moves, params: Params = DEFAULT_PARAMS,
                    slots: int = 4) -> dict:
    base = entry.get("moves") or {}
    known = [m for m in revealed_moves or ()]
    out = {m: 1.0 for m in known}
    free = max(0, slots - len(known))
    if free == 0 or not base:
        return _sorted(out)
    nb = (entry.get("nb") or {}).get("move")
    logit = {}
    for m, p in base.items():
        if m in out:
            continue
        p = min(max(p, params.floor), 1 - 1e-4)
        z = math.log(p / (1 - p))
        if nb and feats and nb.get("N"):
            N, n, pf, cc, beta = nb["N"], nb["n"], nb["pf"], nb["c"], params.beta
            nm = n.get(m, 0.0)
            cm = cc.get(m) or {}
            for f, kap in feats:
                c = cm.get(f)
                q = pf.get(f)
                if c is None or not q or f == f"v:{m}":
                    continue
                nf = q * N
                p_in = (c + beta * q) / (nm + beta)
                p_out = (max(nf - c, 0.0) + beta * q) / (max(N - nm, 0.0) + beta)
                z += kap * (math.log(p_in) - math.log(p_out))
        logit[m] = z / params.temp.get("move", 1.0)
    if not logit:
        return _sorted(out)
    target = min(free, len(logit))
    lo, hi = -30.0, 30.0
    for _ in range(50):                  # shift so the expected number of unrevealed moves fills the free slots
        mid = (lo + hi) / 2
        s = sum(_sig(z + mid) for z in logit.values())
        if s > target:
            hi = mid
        else:
            lo = mid
    shift = (lo + hi) / 2
    for m, z in logit.items():
        out[m] = min(1.0, _sig(z + shift))
    return _sorted(out)


def spreads_posterior(entry: dict, items: dict, original_item: Optional[str] = None,
                      params: Params = DEFAULT_PARAMS, keep: int = 8) -> list:
    """[(nature, evs, p)]: sum over items of P(item) * P(spread | item)."""
    uncond = entry.get("spreads") or []
    by_item = entry.get("spreads_by_item") or {}
    weights = {original_item: 1.0} if original_item else items
    agg: dict = {}

    def add(rows, w):
        for nature, evs, p in rows:
            k = (nature, tuple(evs))
            agg[k] = agg.get(k, 0.0) + w * p

    for item, pi in (weights or {}).items():
        rows = by_item.get(item)
        if rows:
            add(rows, pi * params.spread_by_item)
            add(uncond, pi * (1 - params.spread_by_item))
        else:
            add(uncond, pi)
    if not agg:
        add(uncond, 1.0)
    z = sum(agg.values()) or 1.0
    rows = sorted(((n, list(e), v / z) for (n, e), v in agg.items()), key=lambda r: -r[2])[:keep]
    s = sum(r[2] for r in rows) or 1.0
    return [(n, e, round(p / s, 4)) for n, e, p in rows]


_SPE_CACHE: dict = {}


def speed_stat(species: str, nature: str, evs) -> Optional[int]:
    """Level-100 Speed of a spread (31 IVs; 0 with a minus-Speed nature and no Speed EVs: a Trick Room set)."""
    key = (species, nature, tuple(evs))
    if key in _SPE_CACHE:
        return _SPE_CACHE[key]
    from arena.knowledge.estimate import _compute
    ivs = [31] * 6
    if nature in ("brave", "quiet", "relaxed", "sassy") and evs[5] == 0:
        ivs[5] = 0
    st = _compute(species, 100, {"evs": list(evs), "ivs": ivs, "nature": nature}, 9)
    v = st["spe"] if st else None
    _SPE_CACHE[key] = v
    return v


def _speed_ok(spe: float, obs: dict, tol: float = 1.0) -> bool:
    """Could a Pokemon with raw Speed ``spe`` (item multiplier included) have moved as observed?"""
    lo, hi = spe * obs["m_lo"], spe * obs["m_hi"]
    ours = obs["our_eff"]
    opp_faster = obs["opp_first"] != obs.get("trick_room", False)
    return hi + tol >= ours if opp_faster else lo - tol <= ours


def apply_speed(entry: dict, sid: str, items: dict, spreads_for, speed_obs: list, params: Params,
                damage_obs: list = ()) -> tuple:
    """Joint item x spread weights under the observed move orders and exact-damage hits -> (items, spreads)."""
    from arena.opponent.damage_evidence import damage_ok
    joint: dict = {}
    for item, pi in items.items():
        if pi < 0.002:
            joint[(item, None)] = pi
            continue
        mult = 1.5 if item == "choicescarf" else 0.5 if item == "ironball" else 1.0
        for nature, evs, ps in spreads_for(item):
            spe = speed_stat(sid, nature, evs)
            ok = spe is None or all(_speed_ok(spe * mult, o) for o in speed_obs)
            n_bad = sum(not damage_ok(sid, item, nature, evs, o) for o in damage_obs)
            w = pi * ps * (1.0 if ok else 0.02) * (params.damage_floor ** n_bad)
            joint[(item, (nature, tuple(evs)))] = joint.get((item, (nature, tuple(evs))), 0.0) + w
    z = sum(joint.values()) or 1.0
    new_items: dict = {}
    new_spreads: dict = {}
    for (item, sp), w in joint.items():
        new_items[item] = new_items.get(item, 0.0) + w / z
        if sp is not None:
            new_spreads[sp] = new_spreads.get(sp, 0.0) + w / z
    sz = sum(new_spreads.values()) or 1.0
    rows = sorted(((n, list(e), v / sz) for (n, e), v in new_spreads.items()), key=lambda r: -r[2])[:8]
    t = sum(r[2] for r in rows) or 1.0
    return _sorted(new_items), [(n, e, round(p / t, 4)) for n, e, p in rows]


def belief(kb: dict, sid: str, mates, revealed: dict, *, lik_item: Optional[dict] = None,
           lik_ability: Optional[dict] = None, original_item: Optional[str] = None, item_gone: bool = False,
           notes=(), speed_obs=(), damage_obs=(), params: Params = DEFAULT_PARAMS) -> Optional[Belief]:
    """The belief about one Pokemon. ``revealed``: {"moves": [...], "item", "ability", "tera"} (ids, None = unknown);
    ``mates``: canonical ids of its known teammates; ``original_item``: the item it started with if known."""
    entry = (kb.get("species") or {}).get(sid)
    if entry is None:
        return None
    mates = [m for m in mates if m != sid]
    current = revealed.get("item")
    rev = dict(revealed)
    if original_item:
        rev["item"] = original_item                 # a knocked-off / tricked-away Choice Scarf still says how it is built
    if item_gone:
        items = {"nothing": 1.0}
    elif current or original_item:
        items = {current or original_item: 1.0}
    else:
        items = categorical(entry, "item", features(mates, rev, "item", params), lik_item, params)
    abilities = ({rev["ability"]: 1.0} if rev.get("ability")
                 else categorical(entry, "ability", features(mates, rev, "ability", params), lik_ability, params))
    tera = ({rev["tera"]: 1.0} if rev.get("tera")
            else categorical(entry, "tera", features(mates, rev, "tera", params), None, params))
    moves = moves_posterior(entry, features(mates, {k: v for k, v in rev.items()}, None, params),
                            rev.get("moves") or (), params)
    held_for_spreads = original_item or current     # the set was built around the item it started with
    spreads = spreads_posterior(entry, items, held_for_spreads, params)
    if speed_obs or damage_obs:
        basis = {held_for_spreads: 1.0} if held_for_spreads else items
        fit_items, spreads = apply_speed(entry, sid, basis,
                                         lambda it: spreads_posterior(entry, {it: 1.0}, None, params, keep=12),
                                         list(speed_obs), params, list(damage_obs))
        if not held_for_spreads and not item_gone:
            items = fit_items
        notes = list(notes) + [o["note"] for o in list(speed_obs) + list(damage_obs) if o.get("note")]
    return Belief(species=sid, name=entry.get("name", sid), items=items, abilities=abilities, tera=tera, moves=moves,
                  spreads=spreads, revealed={k: v for k, v in revealed.items() if v}, notes=list(notes),
                  original_item=original_item, n=int(entry.get("n") or 0))
