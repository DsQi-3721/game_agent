"""Offline: compile the opponent knowledge (``arena.opponent.kb``) from local data. Never imported at runtime.

Sources (all on this server, see ``reports/opponent_modeling_resources.md``):

- Smogon chaos usage stats (``/data/wangxinqi/opponent_kb/smogon_stats/<month>/chaos/gen9ou-<cutoff>.json``):
  the base distribution of items, abilities, Tera types, moves, spreads and teammates. They are computed from the
  teams players really submitted, so unlike replays they are not biased towards what a battle happens to reveal.
- Metamon revealed teams (``scripts/opp_build_revealed.py`` -> parquet): ~2M public gen9ou battles with what
  each side revealed. Only co-occurrence statistics are taken from them (item x teammate, item x revealed move,
  move x move in complete movesets ...): under "a value is revealed independently of the teammates" the exposure
  bias cancels in the ratios that ``arena.opponent.posterior`` multiplies onto the base distribution.
- Full sets (Foul Play's data.foulplay.cc cache, Smogon Strategy Dex sets, Showdown teambuilder sets): which
  nature / EVs go with which item (a Choice Scarf user runs maximum Speed).
- The local Showdown server (``../pokemon-showdown``): legality (gen9ou RuleTable) and forme names, so the
  knowledge never contradicts the server we play on.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
KB_DIR = Path("/data/wangxinqi/opponent_kb")
REVEALED = KB_DIR / "derived" / "revealed_gen9ou.parquet"
FP_SETS = ROOT / "external" / "foulplay" / "repo" / "fp" / "data" / "pkmn_sets_cache" / "gen9ou"
SHOWDOWN = ROOT.parent / "pokemon-showdown"
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
KINDS = ("item", "ability", "tera")


def tid(s) -> str:
    return "".join(c for c in str(s or "").lower() if c.isalnum())


# --- the local server: legality and formes ------------------------------------------------------------------
NODE = r"""
const path = require('path');
const {Dex} = require(path.resolve(process.env.SD_DIR, 'dist/sim'));
const rt = Dex.formats.getRuleTable(Dex.formats.get('gen9ou'));
const dex = Dex.forGen(9);
const toID = (s) => ('' + s).toLowerCase().replace(/[^a-z0-9]+/g, '');
const species = {}, aliases = {};
for (const s of dex.species.all()) {
  if (s.isNonstandard && s.isNonstandard !== 'Past') continue;
  let canon = s.name;
  if (s.battleOnly) canon = Array.isArray(s.battleOnly) ? s.battleOnly[0] : s.battleOnly;
  else if (s.changesFrom && s.isCosmeticForme) canon = s.changesFrom;
  const base = dex.species.get(s.baseSpecies);
  if (base && base.cosmeticFormes && base.cosmeticFormes.includes(s.name)) canon = base.name;
  if (toID(canon) !== s.id) aliases[s.id] = toID(canon);
  species[s.id] = {name: s.name, base: toID(s.baseSpecies), banned: rt.isBannedSpecies(s), spe: s.baseStats.spe, types: s.types,
                   stats: s.baseStats, abilities: Object.values(s.abilities).map(toID)};
}
for (const b of Object.values(dex.species.all())) {
  for (const c of (b.cosmeticFormes || [])) aliases[toID(c)] = b.id;
}
const banned = {move: [], item: [], ability: []};
for (const m of dex.moves.all()) if (rt.isBanned('move:' + m.id)) banned.move.push(m.id);
for (const i of dex.items.all()) if (rt.isBanned('item:' + i.id)) banned.item.push(i.id);
for (const a of dex.abilities.all()) if (rt.isBanned('ability:' + a.id)) banned.ability.push(a.id);
const moves = {};
for (const m of dex.moves.all()) moves[m.id] = {cat: m.category, pri: m.priority, type: m.type, bp: m.basePower,
  flags: Object.keys(m.flags || {}), side: m.sideCondition || null, weather: m.weather || null,
  boosts: m.boosts || null, self: (m.self && m.self.boosts) || null, heal: !!m.flags.heal,
  switch: !!m.selfSwitch, force: !!m.forceSwitch, status: m.status || null, target: m.target};
console.log(JSON.stringify({version: require(path.resolve(process.env.SD_DIR, 'package.json')).version,
  species, aliases, banned, moves}));
"""


def showdown_tables(sd_dir: Path = SHOWDOWN, cache: Optional[Path] = KB_DIR / "derived" / "showdown_tables.json") -> dict:
    if cache is not None and cache.exists():
        return json.loads(cache.read_text())
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(NODE)
        js = f.name
    try:
        out = subprocess.run(["node", js], env=dict(os.environ, SD_DIR=str(sd_dir)),
                             capture_output=True, text=True, check=True, timeout=300)
    finally:
        Path(js).unlink(missing_ok=True)
    tables = json.loads(out.stdout.strip().splitlines()[-1])
    tables["banned"] = {k: sorted(v) for k, v in tables["banned"].items()}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(tables))
    return tables


# --- Smogon chaos ----------------------------------------------------------------------------------------------
def chaos_path(month: str, cutoff: int) -> Path:
    return KB_DIR / "smogon_stats" / month / "chaos" / f"gen9ou-{cutoff}.json"


def leads_path(month: str, cutoff: int) -> Path:
    return KB_DIR / "smogon_stats" / month / "leads" / f"gen9ou-{cutoff}.txt"


def load_chaos(sources: list[tuple[Path, float]]) -> dict:
    """Mix monthly chaos files: {species id: {"name", "raw", "usage", field: {key: weighted count}}}; each file's
    counts are scaled by its weight so later months can dominate."""
    out: dict = {}
    for path, w in sources:
        d = json.loads(Path(path).read_text())
        total = sum(v.get("Raw count", 0) for v in d["data"].values()) or 1
        for name, v in d["data"].items():
            sid = tid(name)
            e = out.setdefault(sid, {"name": name, "raw": 0.0, "usage": 0.0, "Abilities": defaultdict(float),
                                     "Items": defaultdict(float), "Tera Types": defaultdict(float),
                                     "Moves": defaultdict(float), "Spreads": defaultdict(float),
                                     "Teammates": defaultdict(float), "Checks and Counters": {}})
            e["raw"] += w * v.get("Raw count", 0)
            e["usage"] += w * v.get("Raw count", 0) / total
            for f in ("Abilities", "Items", "Tera Types", "Moves", "Spreads", "Teammates"):
                for k, c in (v.get(f) or {}).items():
                    e[f][k] += w * float(c)
            for k, c in (v.get("Checks and Counters") or {}).items():
                vals = c if isinstance(c, list) else [c.get("n", 0), c.get("p", 0), c.get("d", 0)]
                old = e["Checks and Counters"].get(k)
                if old is None or vals[0] > old[0]:
                    e["Checks and Counters"][k] = vals
    wsum = sum(w for _p, w in sources) or 1
    for e in out.values():
        e["usage"] /= wsum
    return out


def load_leads(path: Path) -> dict:
    """{species id: raw lead count} from a Smogon leads.txt."""
    out = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) >= 5 and parts[0].isdigit():
            out[tid(parts[1])] = float(parts[3])
    return out


def _dist(counts: dict, banned: set, keep: int, floor: float, norm: Optional[float] = None) -> dict:
    rows = [(tid(k) if k else "", float(v)) for k, v in counts.items()]
    rows = [(k, v) for k, v in rows if k and k not in banned and v > 0]
    z = norm if norm is not None else (sum(v for _k, v in rows) or 1.0)
    rows = sorted(((k, v / z) for k, v in rows), key=lambda kv: -kv[1])
    rows = [(k, p) for k, p in rows if p >= floor][:keep]
    if norm is None:
        s = sum(p for _k, p in rows) or 1.0
        rows = [(k, p / s) for k, p in rows]
    return {k: round(p, 5) for k, p in rows}


def _spreads(counts: dict, keep: int = 12) -> list:
    tot = sum(counts.values()) or 1.0
    rows = []
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:keep]:
        nature, _, evs = k.partition(":")
        try:
            ev = [int(x) for x in evs.split("/")]
        except ValueError:
            continue
        if len(ev) == 6:
            rows.append([tid(nature), ev, round(v / tot, 5)])
    return rows


# --- full sets: spreads that go with an item ---------------------------------------------------------------------
def full_sets(aliases: dict) -> dict:
    """{species: [(item, ability, tera, nature, evs, moves, weight, source)]} from Foul Play's caches and the Smogon
    Strategy Dex. Foul Play's counts are not usage (its sampler ignores them too), so each set is capped."""
    out: dict = defaultdict(list)
    fp = json.loads((FP_SETS / "pokemon_full_sets.json").read_text())
    for sp, sets in fp.items():
        s = aliases.get(tid(sp), tid(sp))
        for key, cnt in sets.items():
            parts = key.split("|")
            if len(parts) < 6:
                continue
            tera, ability, item, nature, evs = parts[:5]
            try:
                ev = [int(x) for x in evs.split(",")]
            except ValueError:
                continue
            out[s].append((tid(item), tid(ability), tid(tera), tid(nature), ev, [tid(m) for m in parts[5:]],
                           min(float(cnt), 10.0), "foulplay"))
    dex = json.loads((KB_DIR / "pkmn_cc" / "sets_gen9ou.json").read_text())
    for sp, sets in dex.items():
        s = aliases.get(tid(sp), tid(sp))
        for st in sets.values():
            items = st.get("item") if isinstance(st.get("item"), list) else [st.get("item")]
            natures = st.get("nature") if isinstance(st.get("nature"), list) else [st.get("nature")]
            evss = st.get("evs") if isinstance(st.get("evs"), list) else [st.get("evs")]
            abil = st.get("ability") if isinstance(st.get("ability"), list) else [st.get("ability")]
            teras = st.get("teratypes") or [None]
            moves = [m if isinstance(m, str) else m[0] for m in st.get("moves") or []]
            k = len(items) * len(natures) * len(evss)
            for it in items:
                for na in natures:
                    for evd in evss:
                        ev = [int((evd or {}).get(x, 0)) for x in STATS]
                        out[s].append((tid(it), tid(abil[0]), tid(teras[0]), tid(na), ev, [tid(m) for m in moves],
                                       5.0 / k, "smogon-dex"))
    return out


def spreads_by_item(sets: list, keep: int = 6) -> dict:
    agg: dict = defaultdict(lambda: defaultdict(float))
    for item, _ab, _te, nature, ev, _mv, w, _src in sets:
        if item and nature:
            agg[item][(nature, tuple(ev))] += w
    out = {}
    for item, d in agg.items():
        tot = sum(d.values())
        out[item] = [[n, list(ev), round(c / tot, 4)] for (n, ev), c in sorted(d.items(), key=lambda kv: -kv[1])[:keep]]
    return out


# --- revealed teams: co-occurrence counts ------------------------------------------------------------------------
def _months_between(a: str, b: str) -> float:
    ya, ma = int(a[:4]), int(a[5:7])
    yb, mb = int(b[:4]), int(b[5:7])
    return (yb - ya) * 12 + (mb - ma)


class Revealed:
    """The revealed-teams table restricted to a date window and splits, as flat numpy arrays (one entry per Pokemon
    of a 6-Pokemon team)."""

    def __init__(self, path: Path = REVEALED, *, date_lo: str = "0000", date_hi: str = "9999",
                 splits: Iterable[str] = ("train",), half_life_months: Optional[float] = 6.0,
                 min_rating: int = -1, aliases: Optional[dict] = None):
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        t = pq.read_table(path, columns=["identity", "split", "date", "rating", "species", "item", "ability", "tera",
                                         "moves"])
        mask = pc.and_(pc.greater_equal(t["date"], date_lo), pc.less_equal(t["date"], date_hi))
        mask = pc.and_(mask, pc.is_in(t["split"], value_set=__import__("pyarrow").array(list(splits))))
        if min_rating > -1:
            mask = pc.and_(mask, pc.greater_equal(t["rating"], min_rating))
        mask = pc.and_(mask, pc.equal(pc.list_value_length(t["species"]), 6))
        t = t.filter(mask)
        self.n_teams = t.num_rows
        self.identity = t["identity"].to_numpy(zero_copy_only=False)
        dates = t["date"].to_numpy(zero_copy_only=False)
        self.rating = t["rating"].to_numpy(zero_copy_only=False)
        if half_life_months:
            ref = max(dates) if len(dates) else date_hi
            ages = np.array([_months_between(d, ref) for d in dates], dtype=np.float64)
            self.team_w = 0.5 ** (ages / half_life_months)
        else:
            self.team_w = np.ones(self.n_teams)
        aliases = aliases or {}
        self.vocab: dict[str, dict[str, int]] = {}
        self.names: dict[str, list[str]] = {}

        def enc(kind: str, values) -> np.ndarray:
            voc = self.vocab.setdefault(kind, {})
            names = self.names.setdefault(kind, [])
            out = np.empty(len(values), dtype=np.int32)
            for j, v in enumerate(values):
                if v is None:
                    out[j] = -1
                    continue
                if kind == "species":
                    v = aliases.get(v, v)
                c = voc.get(v)
                if c is None:
                    c = voc[v] = len(names)
                    names.append(v)
                out[j] = c
            return out

        flat = lambda col: pc.list_flatten(t[col]).to_numpy(zero_copy_only=False)  # noqa: E731
        self.species = enc("species", flat("species")).reshape(-1, 6)
        self.item = enc("item", flat("item")).reshape(-1, 6)
        self.ability = enc("ability", flat("ability")).reshape(-1, 6)
        self.tera = enc("tera", flat("tera")).reshape(-1, 6)
        mv_lists = pc.list_flatten(t["moves"])                     # one list per Pokemon
        lens = pc.list_value_length(mv_lists).to_numpy(zero_copy_only=False)
        self.n_moves = lens.reshape(-1, 6)
        self.move_flat = enc("move", pc.list_flatten(mv_lists).to_numpy(zero_copy_only=False))
        self.move_owner = np.repeat(np.arange(len(lens)), lens)    # flat Pokemon index of each revealed move

    def code(self, kind: str, value: str) -> int:
        return self.vocab.get(kind, {}).get(value, -1)


def _count(keys: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, inv = np.unique(keys, return_inverse=True)
    return uniq, np.bincount(inv, weights=w)


def cooccurrence(rv: Revealed) -> dict:
    """Weighted counts per species for the posterior's naive-Bayes factors.

    Categorical kinds (item / ability / Tera): over Pokemon whose value x is revealed, n[x] and c[x][f] for the
    features f seen with it -- teammates (always), revealed moves, the other revealed attributes.
    Moves: over Pokemon whose four moves are all revealed (a complete moveset, so a move outside it is absent),
    n[m] and c[m][f] with f = teammates, the other moves, revealed attributes; plus N and n[f]."""
    S = rv.species.shape[0]
    sp = rv.species.reshape(-1)
    w = np.repeat(rv.team_w, 6)
    pos = np.tile(np.arange(6), S)
    team = np.repeat(np.arange(S), 6)
    NF = {"m": len(rv.names["species"]), "v": len(rv.names["move"]), "i": len(rv.names["item"]),
          "a": len(rv.names["ability"]), "t": len(rv.names["tera"])}
    off, acc = {}, 0
    for k in "mviat":
        off[k] = acc
        acc += NF[k]
    NFT = acc
    attr = {"item": rv.item.reshape(-1), "ability": rv.ability.reshape(-1), "tera": rv.tera.reshape(-1)}
    letter = {"item": "i", "ability": "a", "tera": "t"}
    mates = np.stack([rv.species[team, k] for k in range(6)], axis=1)        # [mon, 6] species of its team
    out: dict = {}

    def features(mon_idx: np.ndarray, exclude_kind: Optional[str]):
        """(mon index, feature id) pairs for the given Pokemon."""
        rows, cols = [], []
        p = pos[mon_idx]
        for k in range(6):
            sel = p != k
            rows.append(mon_idx[sel])
            cols.append(off["m"] + mates[mon_idx[sel], k])
        has = np.zeros(len(sp), dtype=bool)
        has[mon_idx] = True
        mv_sel = has[rv.move_owner]
        rows.append(rv.move_owner[mv_sel])
        cols.append(off["v"] + rv.move_flat[mv_sel])
        for kind in KINDS:
            if kind == exclude_kind:
                continue
            a = attr[kind][mon_idx]
            sel = a >= 0
            rows.append(mon_idx[sel])
            cols.append(off[letter[kind]] + a[sel])
        return np.concatenate(rows), np.concatenate(cols)

    def decode(f: int) -> str:
        for k in "mviat":
            if off[k] <= f < off[k] + NF[k]:
                kind = {"m": "species", "v": "move", "i": "item", "a": "ability", "t": "tera"}[k]
                return f"{k}:{rv.names[kind][f - off[k]]}"
        raise ValueError(f)

    for kind in KINDS:
        x = attr[kind]
        mon = np.nonzero(x >= 0)[0]
        r, f = features(mon, kind)
        NX = len(rv.names[kind])
        key = (sp[r].astype(np.int64) * NX + x[r]) * NFT + f
        uniq, cnt = _count(key, w[r])
        nkey, ncnt = _count(sp[mon].astype(np.int64) * NX + x[mon], w[mon])
        res: dict = defaultdict(lambda: {"n": {}, "c": defaultdict(dict)})
        for k, c in zip(nkey, ncnt):
            s, xv = divmod(int(k), NX)
            res[rv.names["species"][s]]["n"][rv.names[kind][xv]] = c
        for k, c in zip(uniq, cnt):
            sx, fv = divmod(int(k), NFT)
            s, xv = divmod(sx, NX)
            res[rv.names["species"][s]]["c"][rv.names[kind][xv]][decode(fv)] = c
        out[kind] = res
    # moves: complete movesets only
    mon = np.nonzero(rv.n_moves.reshape(-1) == 4)[0]
    r, f = features(mon, None)
    has = np.zeros(len(sp), dtype=bool)
    has[mon] = True
    msel = has[rv.move_owner]
    m_owner, m_code = rv.move_owner[msel], rv.move_flat[msel]
    # pair each move of a complete set with every feature of the same Pokemon (its other moves included)
    order = np.argsort(r, kind="stable")
    r, f = r[order], f[order]
    starts = np.searchsorted(r, mon)
    ends = np.searchsorted(r, mon, side="right")
    NM = len(rv.names["move"])
    res = defaultdict(lambda: {"n": {}, "c": defaultdict(dict), "N": 0.0, "nf": {}})
    mo_order = np.argsort(m_owner, kind="stable")
    m_owner, m_code = m_owner[mo_order], m_code[mo_order]
    ms, me = np.searchsorted(m_owner, mon), np.searchsorted(m_owner, mon, side="right")
    keys, ws = [], []
    for j in range(4):                       # each complete set has exactly 4 moves
        mv = m_code[ms + j]
        L = ends - starts
        rep_mon = np.repeat(mon, L)
        rep_mv = np.repeat(mv, L)
        idx = np.repeat(starts, L) + (np.arange(L.sum()) - np.repeat(np.cumsum(L) - L, L))
        ff = f[idx]
        keep = ff != off["v"] + rep_mv       # a move is not its own feature
        keys.append((sp[rep_mon[keep]].astype(np.int64) * NM + rep_mv[keep]) * NFT + ff[keep])
        ws.append(w[rep_mon[keep]])
    uniq, cnt = _count(np.concatenate(keys), np.concatenate(ws))
    nkey, ncnt = _count(sp[m_owner].astype(np.int64) * NM + m_code, w[m_owner])
    for k, c in zip(nkey, ncnt):
        s, mv = divmod(int(k), NM)
        res[rv.names["species"][s]]["n"][rv.names["move"][mv]] = c
    for k, c in zip(uniq, cnt):
        sx, fv = divmod(int(k), NFT)
        s, mv = divmod(sx, NM)
        res[rv.names["species"][s]]["c"][rv.names["move"][mv]][decode(fv)] = c
    Nkey, Ncnt = _count(sp[mon].astype(np.int64), w[mon])
    for k, c in zip(Nkey, Ncnt):
        res[rv.names["species"][int(k)]]["N"] = c
    fkey, fcnt = _count(sp[r].astype(np.int64) * NFT + f, w[r])
    for k, c in zip(fkey, fcnt):
        s, fv = divmod(int(k), NFT)
        res[rv.names["species"][s]]["nf"][decode(fv)] = c
    out["move"] = res
    # exposure: Pokemon per species, and revealed counts per value
    tot_key, tot_cnt = _count(sp.astype(np.int64), w)
    out["mons"] = {rv.names["species"][int(k)]: c for k, c in zip(tot_key, tot_cnt)}
    return out


def reveal_rates(co: dict, chaos: dict) -> dict:
    """r(x) = revealed count of x / expected holders of x (base distribution x Pokemon seen), over all species:
    how much more often a battle shows a value than teams hold it (Leftovers heals visibly, Heavy-Duty Boots
    rarely shows). Used to compare true-distribution models with revealed-only test labels."""
    out = {}
    for kind, field in (("item", "Items"), ("ability", "Abilities"), ("tera", "Tera Types")):
        num, den = defaultdict(float), defaultdict(float)
        for s, d in co[kind].items():
            ch = chaos.get(s)
            if ch is None:
                continue
            base = {tid(k): v for k, v in ch[field].items() if tid(k)}
            z = sum(base.values()) or 1.0
            n_mons = co["mons"].get(s, 0.0)
            for x, c in d["n"].items():
                num[x] += c
            for x, v in base.items():
                den[x] += n_mons * v / z
        out[kind] = {x: round(num[x] / den[x], 4) for x in den if den[x] >= 20 and num[x] > 0}
    return out


# --- assembling the knowledge ------------------------------------------------------------------------------------
def _nb_block(d: dict, keep_x: set, min_c: float, kind: str, beta: float = 30.0, tau: float = 0.1,
              max_feats: Optional[int] = None) -> dict:
    """Counts for one species and attribute, keeping only informative pairs: a stored pair moves the log-odds by
    at least ``tau`` at the reference shrinkage ``beta`` (pairs never seen although expected are stored as 0);
    the posterior treats a missing pair as neutral."""
    if kind == "move":
        N = d.get("N", 0.0)
        nf = d["nf"]
    else:
        N = sum(d["n"].values())
        nf = defaultdict(float)
        for _x, fs in d["c"].items():
            for f, c in fs.items():
                nf[f] += c
    if not N:
        return {"N": 0.0, "n": {}, "pf": {}, "c": {}}
    pf = {f: c / N for f, c in nf.items() if c >= min_c}
    if max_feats and len(pf) > max_feats:          # the most frequent features carry almost all the evidence
        pf = dict(sorted(pf.items(), key=lambda kv: -kv[1])[:max_feats])
    n = {x: c for x, c in d["n"].items() if x in keep_x}
    out_c: dict = {}
    used = set()
    for x, nx in n.items():
        cx = d["c"].get(x, {})
        row = {}
        for f, q in pf.items():
            c = cx.get(f, 0.0)
            if kind == "move":
                if f == f"v:{x}":
                    continue
                p_in = (c + beta * q) / (nx + beta)
                p_out = (max(q * N - c, 0.0) + beta * q) / (max(N - nx, 0.0) + beta)
                r = math.log(p_in) - math.log(p_out)
            else:
                r = math.log((c + beta * q) / (nx + beta)) - math.log(q)
            if abs(r) >= tau:
                row[f] = round(c, 1)
                used.add(f)
        if row:
            out_c[x] = row
    return {"N": round(N, 2), "n": {x: round(c, 2) for x, c in n.items()},
            "pf": {f: round(q, 6) for f, q in pf.items() if f in used}, "c": out_c}


def compile_kb(*, chaos_sources: list[tuple[Path, float]], leads: Optional[Path], co: Optional[dict],
               tables: dict, sets: Optional[dict] = None, min_raw: float = 30.0, min_c: float = 3.0,
               tau: float = 0.1, max_feats: Optional[int] = None, provenance: Optional[dict] = None) -> dict:
    chaos = load_chaos(chaos_sources)
    banned = {k: set(v) for k, v in tables["banned"].items()}
    sp_tab, aliases = tables["species"], tables["aliases"]
    lead_counts = load_leads(leads) if leads else {}
    species = {}
    for sid, e in chaos.items():
        sid = aliases.get(sid, sid)
        info = sp_tab.get(sid)
        if info is None or info.get("banned") or e["raw"] < min_raw:
            continue
        z = sum(e["Abilities"].values()) or 1.0
        entry = {
            "name": info["name"], "usage": round(e["usage"], 6), "n": round(e["raw"]),
            "items": _dist(e["Items"], banned["item"], 25, 0.002),
            "abilities": _dist(e["Abilities"], banned["ability"], 4, 0.005),
            "tera": _dist(e["Tera Types"], set(), 19, 0.003),
            "moves": _dist(e["Moves"], banned["move"], 40, 0.01, norm=z),
            "spreads": _spreads(e["Spreads"]),
            "mates": {aliases.get(tid(k), tid(k)): round(v / z, 4)
                      for k, v in sorted(e["Teammates"].items(), key=lambda kv: -kv[1])[:40] if v / z >= 0.01},
            "counters": [[aliases.get(tid(k), tid(k)), round(v[0], 1), round(v[1], 3)]
                         for k, v in sorted(e["Checks and Counters"].items(),
                                            key=lambda kv: -(kv[1][1] - 4 * kv[1][2]))
                         if v[0] >= 20][:8],
        }
        if lead_counts:
            entry["lead"] = round(min(1.0, lead_counts.get(sid, 0.0) / max(e["raw"], 1.0)), 4)
        if sets and sets.get(sid):
            entry["spreads_by_item"] = spreads_by_item(sets[sid])
        if co is not None:
            nb = {}
            for kind, key in (("item", "items"), ("ability", "abilities"), ("tera", "tera"), ("move", "moves")):
                d = co[kind].get(sid)
                if d:
                    nb[kind] = _nb_block(d, set(entry[key]), min_c, kind, tau=tau, max_feats=max_feats)
            entry["nb"] = nb
        species[sid] = entry
    # formes Smogon merges into the base (Maushold-Four, Polteageist-Antique, Tatsugiri-Droopy ...)
    aliases = dict(aliases)
    for sid, info in sp_tab.items():
        base = aliases.get(info.get("base", sid), info.get("base", sid))
        if sid not in species and sid not in aliases and base in species and base != sid:
            aliases[sid] = base
    kb = {"format": "gen9ou", "built": {"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "chaos": [[str(p), w] for p, w in chaos_sources],
                                        "leads": str(leads) if leads else None,
                                        "showdown": tables.get("version"), **(provenance or {})},
          "aliases": {k: v for k, v in aliases.items() if v in species}, "species": species}
    if co is not None:
        kb["reveal_rates"] = reveal_rates(co, {aliases.get(k, k): v for k, v in chaos.items()})
    return kb


def kb_size(kb: dict) -> int:
    return len(json.dumps(kb, separators=(",", ":")))


def write_kb(kb: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kb, separators=(",", ":")), encoding="utf-8")


_TEAM_SPLIT = re.compile(r"\n\s*\n")
