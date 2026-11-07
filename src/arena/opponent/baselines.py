"""Published Pokemon opponent models, re-implemented as baselines for ``arena.opponent`` (offline evaluation only).

Each model answers the same queries as ours: ``dist(sid, kind, mates, revealed)`` -> a distribution over one
attribute (item / ability / tera) or, for ``kind == "move"``, P(move in the set) per move; ``belief(...)`` -> all
of them plus spreads. ``revealed``: {"moves": [...], "item", "ability", "tera"} (ids, None = unknown); ``excluded``:
items the battle has ruled out (for models that use such evidence).

- ``FoulPlayModel`` -- Foul Play (PokéAgent Challenge 2025 gen9ou champion), ``fp/search/standard_battles.py``:
  the opponent's set is drawn UNIFORMLY from the full sets (data.foulplay.cc + Showdown teambuilder sets) still
  consistent with what was revealed and ruled out; with no consistent full set, from Smogon usage combinations.
  Its in-battle exclusions (two moves -> no Choice item, status move -> no Assault Vest, damage without recoil ->
  no Life Orb, below full HP at upkeep -> no Leftovers, hazards -> Heavy-Duty Boots ...) are passed as ``excluded``.
- ``PokeChampModel`` -- PokéChamp (ICML 2025), ``bayesian/team_predictor.py``: counts full configurations
  (item | ability | nature | EVs | moves | Tera) per species over Metamon's replay-derived teams; each configuration
  is weighted by its count times (1 + revealed moves it contains)^2 and the components are marginalized. No
  teammates, no other revealed fields. (Its trained model ships as a pickle; this is the same counting, retrained
  from the team files.)
- ``PolimiBotModel`` -- PolimiBot (De Vito 2026, PoliMi thesis, sections 5.3.2-5.3.5): per-species replay
  frequencies with Dirichlet(0.5) smoothing, fused with the Smogon prior by an adaptive weight
  lambda = max(0.1, 1 - sqrt(c_replay / (c_replay + 0.05 c_smogon))); confirmed fields become point masses;
  moves are re-weighted by co-occurrence with the observed moves (log space; reductions capped at 80%). Its LDA
  archetype model predicts unrevealed teammates, which OU team preview makes moot; its damage-based EV update is
  not used (no model gets damage evidence in these benchmarks).
"""

from __future__ import annotations

import collections
import json
import math
import re
import tarfile
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]
FP_SETS = ROOT / "external" / "foulplay" / "repo" / "fp" / "data" / "pkmn_sets_cache" / "gen9ou"
METAMON_TEAMS = Path("/data/dataset/metamon-teams")
STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def tid(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _norm(d: dict) -> dict:
    z = sum(d.values())
    return {k: v / z for k, v in sorted(d.items(), key=lambda kv: -kv[1])} if z > 0 else {}


def parse_team_text(text: str) -> list[dict]:
    """Showdown paste -> [{species, item, ability, tera, nature, evs[6], moves}] (ids)."""
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        name, _, item = lines[0].partition(" @ ")
        name = re.sub(r"\s+\((M|F)\)$", "", name.strip())
        m = re.search(r"\(([^()]+)\)$", name)
        if m and m.group(1) not in ("M", "F"):
            name = m.group(1)
        s = {"species": tid(name), "item": tid(item) or "nothing", "ability": None, "tera": None, "nature": None,
             "evs": [0] * 6, "moves": []}
        for ln in lines[1:]:
            if ln.startswith("Ability:"):
                s["ability"] = tid(ln[8:])
            elif ln.startswith("Tera Type:"):
                s["tera"] = tid(ln[10:])
            elif ln.startswith("EVs:"):
                for part in ln[4:].split("/"):
                    v, _, stat = part.strip().partition(" ")
                    k = {"HP": 0, "Atk": 1, "Def": 2, "SpA": 3, "SpD": 4, "Spe": 5}.get(stat.strip())
                    if k is not None and v.isdigit():
                        s["evs"][k] = int(v)
            elif ln.endswith("Nature"):
                s["nature"] = tid(ln[:-6])
            elif ln.startswith("- "):
                s["moves"].append(tid(ln[2:]))
        out.append(s)
    return out


def metamon_sets(split: str = "modern_replays_v2", fmt: str = "gen9ou"):
    """Every set of every team file in a Metamon team split (completed teams: revealed + usage-filled)."""
    with tarfile.open(METAMON_TEAMS / split / f"{fmt}.tar.gz") as tar:
        for m in tar:
            if m.isfile():
                yield parse_team_text(tar.extractfile(m).read().decode("utf-8", "replace"))


def _consistent(rev: dict, item, ability, tera, moves, excluded=()) -> bool:
    if rev.get("item") and item != rev["item"]:
        return False
    if rev.get("ability") and ability != rev["ability"]:
        return False
    if rev.get("tera") and tera != rev["tera"]:
        return False
    if not rev.get("item") and item in excluded:
        return False
    return all(m in moves for m in rev.get("moves") or ())


def _marginals(rows, kind: str) -> dict:
    """rows: [(weight, item, ability, tera, moves, nature, evs)] -> distribution of one component."""
    agg: dict = collections.defaultdict(float)
    tot = 0.0
    for w, item, ability, tera, moves, nature, evs in rows:
        tot += w
        if kind == "move":
            for m in moves:
                agg[m] += w
        elif kind == "spread":
            agg[(nature, tuple(evs))] += w
        else:
            agg[{"item": item, "ability": ability, "tera": tera}[kind]] += w
    if tot <= 0:
        return {}
    if kind == "move":
        return dict(sorted(((k, v / tot) for k, v in agg.items()), key=lambda kv: -kv[1]))
    return _norm(agg)


class FoulPlayModel:
    name = "Foul Play"

    def __init__(self, chaos_path: Optional[Path] = None, aliases: Optional[dict] = None):
        full = json.loads((FP_SETS / "pokemon_full_sets.json").read_text())
        ps = json.loads((FP_SETS / "showdown_sets.json").read_text())
        merged: dict = {sp: dict(v) for sp, v in full.items()}
        for src in ("dex", "stats"):                        # Foul Play's get_ps_sets_file: teambuilder sets join
            for sp, sets in (ps.get(src) or {}).items():
                d = merged.setdefault(tid(sp), {})
                for st in sets.values():
                    moves = "|".join(sorted(tid(m if isinstance(m, str) else m[0]) for m in st.get("moves") or []))
                    evs = st.get("evs") or {}
                    key = (f"{tid(st.get('teraType') or '')}|{tid(st.get('ability') or 'noability')}|"
                           f"{tid(st.get('item') or 'none')}|{tid(st.get('nature') or 'serious')}|"
                           f"{','.join(str(int(evs.get(k, 0))) for k in STATS)}|{moves}")
                    d[key] = d.get(key, 0) + 1
        self.sets: dict = {}
        for sp, d in merged.items():
            rows = []
            for key in d:                                   # Foul Play samples uniformly over sets (counts unused)
                p = key.split("|")
                if len(p) < 6:
                    continue
                try:
                    evs = [int(x) for x in p[4].split(",")]
                except ValueError:
                    continue
                rows.append((1.0, p[2], p[1], p[0] or None, frozenset(m for m in p[5:] if m), p[3], evs))
            self.sets[tid(sp)] = rows
        self.aliases = aliases or {}
        self.smogon = {}
        if chaos_path:
            data = json.loads(Path(chaos_path).read_text())["data"]
            for name, v in data.items():
                z = sum((v.get("Abilities") or {}).values()) or 1.0
                self.smogon[tid(name)] = {
                    "item": _norm({tid(k): c for k, c in (v.get("Items") or {}).items() if tid(k)}),
                    "ability": _norm({tid(k): c for k, c in (v.get("Abilities") or {}).items() if tid(k)}),
                    "tera": _norm({tid(k): c for k, c in (v.get("Tera Types") or {}).items() if tid(k)}),
                    "move": {tid(k): c / z for k, c in sorted((v.get("Moves") or {}).items(), key=lambda kv: -kv[1])
                             if tid(k)}}

    def _rows(self, sid: str, rev: dict, excluded=()):
        rows = self.sets.get(sid) or self.sets.get(self.aliases.get(sid, ""), [])
        return [r for r in rows if _consistent(rev, r[1], r[2], r[3], r[4], excluded)]

    def dist(self, sid: str, kind: str, mates=(), rev: Optional[dict] = None, excluded=()) -> dict:
        rev = rev or {}
        key = (sid, kind, tuple(sorted(rev.get("moves") or ())), rev.get("item"), rev.get("ability"), rev.get("tera"),
               tuple(sorted(excluded)))
        cache = self.__dict__.setdefault("_cache", {})
        if key in cache:
            return cache[key]
        if len(cache) > 200000:
            cache.clear()
        cache[key] = self._dist(sid, kind, rev, excluded)
        return cache[key]

    def _dist(self, sid: str, kind: str, rev: dict, excluded) -> dict:
        rows = self._rows(sid, rev, excluded)
        if rows:
            return _marginals(rows, kind)
        sm = (self.smogon.get(sid) or {}).get(kind) or {}
        if kind == "item" and excluded:
            sm = _norm({k: v for k, v in sm.items() if k not in excluded})
        return sm

    def belief(self, sid, mates=(), rev=None, excluded=()) -> dict:
        rows = self._rows(sid, rev or {}, excluded)
        out = {k: self.dist(sid, k, mates, rev, excluded) for k in ("item", "ability", "tera", "move")}
        out["spread"] = _marginals(rows, "spread") if rows else {}
        return out


class PokeChampModel:
    name = "PokéChamp"

    def __init__(self, split: str = "modern_replays_v2", aliases: Optional[dict] = None):
        self.configs: dict = collections.defaultdict(collections.Counter)
        self.n_teams = 0
        for team in metamon_sets(split):
            self.n_teams += 1
            for s in team:
                key = (s["item"], s["ability"], s["tera"], frozenset(m for m in s["moves"] if m), s["nature"],
                       tuple(s["evs"]))
                self.configs[s["species"]][key] += 1
        self.aliases = aliases or {}

    def _rows(self, sid: str, rev: dict):
        cfg = self.configs.get(sid) or self.configs.get(self.aliases.get(sid, ""))
        if not cfg:
            return []
        tot = sum(cfg.values())
        seen = set(rev.get("moves") or ())
        rows = []
        for (item, ability, tera, moves, nature, evs), c in cfg.items():
            bonus = (1 + len(seen & moves)) ** 2 if seen else 1.0      # exactly as in predict_component_probabilities
            rows.append((c / tot * bonus, item, ability, tera, moves, nature, list(evs)))
        return rows

    def dist(self, sid: str, kind: str, mates=(), rev: Optional[dict] = None, excluded=()) -> dict:
        key = (sid, kind, tuple(sorted((rev or {}).get("moves") or ())))
        cache = self.__dict__.setdefault("_cache", {})
        if key not in cache:
            if len(cache) > 200000:
                cache.clear()
            cache[key] = _marginals(self._rows(sid, rev or {}), kind)
        return cache[key]

    def belief(self, sid, mates=(), rev=None, excluded=()) -> dict:
        rows = self._rows(sid, rev or {})
        return {k: _marginals(rows, k) for k in ("item", "ability", "tera", "move", "spread")}

    def spreads_by_item(self, sid, rev, items) -> dict:
        """Its configurations keep item and spread together: the spreads of each item's configurations."""
        rows = self._rows(sid, rev or {})
        out = {}
        for it in items:
            sp = _marginals([r for r in rows if r[1] == it], "spread")
            out[it] = [[n, list(e), round(p, 4)] for (n, e), p in list(sp.items())[:4]]
        return out


class PolimiBotModel:
    name = "PolimiBot"
    ALPHA = 0.5

    def __init__(self, chaos_path: Path, split: str = "modern_replays_v2", aliases: Optional[dict] = None):
        self.counts: dict = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
        self.cooc: dict = collections.defaultdict(collections.Counter)      # (species, move) -> co-move counts
        self.n: collections.Counter = collections.Counter()
        for team in metamon_sets(split):
            for s in team:
                sp = s["species"]
                self.n[sp] += 1
                c = self.counts[sp]
                for k in ("item", "ability", "tera"):
                    if s[k]:
                        c[k][s[k]] += 1
                c["spread"][(s["nature"], tuple(s["evs"]))] += 1
                mv = [m for m in s["moves"] if m]
                for m in mv:
                    c["move"][m] += 1
                    for o in mv:
                        if o != m:
                            self.cooc[(sp, m)][o] += 1
        self.smogon, self.raw = {}, {}
        data = json.loads(Path(chaos_path).read_text())["data"]
        for name, v in data.items():
            z = sum((v.get("Abilities") or {}).values()) or 1.0
            sp = tid(name)
            self.raw[sp] = float(v.get("Raw count") or 0)
            self.smogon[sp] = {"item": {tid(k): c / z for k, c in (v.get("Items") or {}).items() if tid(k)},
                               "ability": {tid(k): c / z for k, c in (v.get("Abilities") or {}).items() if tid(k)},
                               "tera": {tid(k): c / z for k, c in (v.get("Tera Types") or {}).items() if tid(k)},
                               "move": {tid(k): c / z for k, c in (v.get("Moves") or {}).items() if tid(k)}}
        self.aliases = aliases or {}

    def _sp(self, sid: str) -> str:
        return sid if (sid in self.n or sid in self.smogon) else self.aliases.get(sid, sid)

    def _fused(self, sp: str, kind: str) -> dict:
        rep = self.counts[sp][kind] if sp in self.counts else collections.Counter()
        smo = (self.smogon.get(sp) or {}).get(kind) or {}
        cats = set(rep) | set(smo)
        if not cats:
            return {}
        c_rep = float(self.n.get(sp, 0))
        if kind == "move":                                   # moves: per-slot shares (4 per set), as inclusion
            tot = sum(rep.values()) + self.ALPHA * len(cats)
            replay = {k: 4 * (rep.get(k, 0) + self.ALPHA) / tot for k in cats} if c_rep else {}
        else:
            tot = sum(rep.values()) + self.ALPHA * len(cats)
            replay = {k: (rep.get(k, 0) + self.ALPHA) / tot for k in cats} if c_rep else {}
        if kind != "move":
            zs = sum(smo.values())
            smo = {k: v / zs for k, v in smo.items()} if zs else {}
        c_smo = self.raw.get(sp, 0.0)
        lam = max(0.1, 1 - math.sqrt(c_rep / (c_rep + 0.05 * c_smo))) if (c_rep + c_smo) > 0 else 1.0
        if not replay:
            lam = 1.0
        if not smo:
            lam = 0.0
        return {k: lam * smo.get(k, 0.0) + (1 - lam) * replay.get(k, 0.0) for k in cats}

    def dist(self, sid: str, kind: str, mates=(), rev: Optional[dict] = None, excluded=()) -> dict:
        rev = rev or {}
        sp = self._sp(sid)
        if kind != "move":
            if rev.get(kind):
                return {rev[kind]: 1.0}
            return _norm(self._fused(sp, kind))
        base = self._fused(sp, "move")
        seen = [m for m in rev.get("moves") or () if m]
        out = {}
        for m, p in base.items():
            if m in seen:
                continue
            score = math.log(max(p, 1e-9))
            for o in seen:                                   # co-occurrence with each observed move, log space
                co = self.cooc.get((sp, o))
                n_o = self.counts[sp]["move"].get(o, 0) if sp in self.counts else 0
                if not co or not n_o:
                    continue
                ratio = (co.get(m, 0) + self.ALPHA) / (n_o + self.ALPHA) / max(p, 1e-6)
                score += math.log(max(ratio, 0.2))           # negative evidence capped at an 80% reduction
            out[m] = score
        if not out:
            return {m: 1.0 for m in seen}
        top = max(out.values())
        w = {m: math.exp(v - top) for m, v in out.items()}
        free = max(0, 4 - len(seen))
        z = sum(w.values()) or 1.0
        res = {m: min(1.0, free * v / z) for m, v in w.items()}
        res.update({m: 1.0 for m in seen})
        return dict(sorted(res.items(), key=lambda kv: -kv[1]))

    def belief(self, sid, mates=(), rev=None, excluded=()) -> dict:
        sp = self._sp(sid)
        out = {k: self.dist(sid, k, mates, rev) for k in ("item", "ability", "tera", "move")}
        sc = self.counts[sp]["spread"] if sp in self.counts else {}
        out["spread"] = _norm(dict(sc))
        return out
