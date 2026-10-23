"""Build the offline knowledge base used by the agent tools (data/knowledge/*.json).

    python scripts/build_knowledge.py [--teams 20000] [--seed 1,2,3,4]

Sources (all local, matched to OUR Showdown server so priors never contradict it):
  - dex_gen9.json        moves / abilities / items / species with Showdown shortDesc
                         text (from ../pokemon-showdown/dist/sim Dex)
  - randbats_gen9.json   per-species set distribution of gen9randombattle, estimated by
                         sampling the server's own seeded random-team generator
  - gen9ou_sets.json     Smogon usage sets (PokeChamp's gen9/ou/sets_<rating>.json),
                         names normalised and filtered against the server's gen9ou
                         RuleTable (e.g. Tera Blast is banned there now)
  - manifest.json        provenance: server version, sample size, seed, rating, date
Runtime code only reads the JSON; node is needed only to (re)build.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SD = ROOT.parent / "pokemon-showdown"
DEFAULT_OU = ROOT / "external" / "pokechamp" / "repo" / "poke_env" / "data" / "static" / "gen9" / "ou"

NODE_SCRIPT = r"""
const path = require('path');
const fs = require('fs');
const {Dex, Teams} = require(path.resolve(process.env.SD_DIR, 'dist/sim'));
const toID = (s) => ('' + s).toLowerCase().replace(/[^a-z0-9]+/g, '');
const out = process.env.OUT_DIR;
const N = parseInt(process.env.N_TEAMS, 10);
const seed = process.env.SEED.split(',').map(Number);
const dex = Dex.forGen(9);

// ---- dex text ---------------------------------------------------------------
const moves = {}, abilities = {}, items = {}, species = {};
for (const m of dex.moves.all()) {
  moves[m.id] = {name: m.name, type: m.type, category: m.category, basePower: m.basePower,
    accuracy: m.accuracy, priority: m.priority, pp: m.pp, desc: m.shortDesc || m.desc || '',
    flags: Object.keys(m.flags || {}), nonstandard: m.isNonstandard || null};
}
for (const a of dex.abilities.all()) abilities[a.id] = {name: a.name, desc: a.shortDesc || a.desc || '', nonstandard: a.isNonstandard || null};
for (const i of dex.items.all()) items[i.id] = {name: i.name, desc: i.shortDesc || i.desc || '', nonstandard: i.isNonstandard || null};
for (const s of dex.species.all()) {
  species[s.id] = {name: s.name, types: s.types, baseStats: s.baseStats, abilities: s.abilities,
    tier: s.tier, weightkg: s.weightkg, nonstandard: s.isNonstandard || null};
}
fs.writeFileSync(path.join(out, 'dex_gen9.json'), JSON.stringify({moves, abilities, items, species}));

// ---- randbats: sample the server's own generator ------------------------------
const agg = {};
const bump = (o, k, by = 1) => { o[k] = (o[k] || 0) + by; };
const gen = Teams.getGenerator('gen9randombattle', seed);
for (let t = 0; t < N; t++) {
  for (const set of gen.getTeam()) {
    const sid = set.speciesId || toID(set.species);
    const a = agg[sid] || (agg[sid] = {name: set.species, n: 0, level: {}, moves: {}, items: {},
                                         abilities: {}, tera: {}, spreads: {}, roles: {}});
    a.n++;
    bump(a.level, set.level);
    for (const mv of set.moves) bump(a.moves, toID(mv));
    bump(a.items, toID(set.item || 'nothing'));
    bump(a.abilities, toID(set.ability));
    bump(a.tera, set.teraType);
    const ev = ['hp','atk','def','spa','spd','spe'].map(k => set.evs[k]);
    const iv = ['hp','atk','def','spa','spd','spe'].map(k => set.ivs[k]);
    bump(a.spreads, JSON.stringify([ev, iv]));
    if (set.role) bump(a.roles, set.role);
  }
}
const nm = (table, id) => (table.get(id) && table.get(id).exists) ? table.get(id).name : id;
const dist = (counts, n, table) => Object.entries(counts)
  .map(([id, c]) => ({id, name: table ? nm(table, id) : id, p: +(c / n).toFixed(4)}))
  .sort((x, y) => y.p - x.p);
const rb = {};
for (const [sid, a] of Object.entries(agg)) {
  rb[sid] = {
    name: a.name, n: a.n,
    level: +Object.entries(a.level).sort((x, y) => y[1] - x[1])[0][0],
    moves: dist(a.moves, a.n, dex.moves), items: dist(a.items, a.n, dex.items),
    abilities: dist(a.abilities, a.n, dex.abilities), tera: dist(a.tera, a.n, null),
    spreads: Object.entries(a.spreads).map(([k, c]) => {
      const [evs, ivs] = JSON.parse(k); return {nature: 'hardy', evs, ivs, p: +(c / a.n).toFixed(4)};
    }).sort((x, y) => y.p - x.p).slice(0, 6),
    roles: dist(a.roles, a.n, null),
  };
}
fs.writeFileSync(path.join(out, 'randbats_gen9.json'),
  JSON.stringify({format: 'gen9randombattle', source: 'sampled Teams.generate', species: rb}));

// ---- gen9ou usage sets, filtered against the server's rule table ---------------
const ouPath = process.env.OU_SETS;
let ouOut = null;
if (ouPath && fs.existsSync(ouPath)) {
  const raw = JSON.parse(fs.readFileSync(ouPath, 'utf8'));
  const rt = Dex.formats.getRuleTable(Dex.formats.get('gen9ou'));
  const legal = (kind, id) => !rt.isBanned(`${kind}:${id}`);
  const norm = (arr, kind, table, renorm) => {
    let rows = (arr || []).map(r => ({id: toID(r.name), name: ('' + r.name).trim(), p: r.percentage / 100}))
      .filter(r => r.id && r.id !== 'other' && (!kind || legal(kind, r.id)));
    if (table) rows = rows.filter(r => table.get(r.id).exists);
    if (renorm) { const s = rows.reduce((t, r) => t + r.p, 0) || 1; rows.forEach(r => r.p = r.p / s); }
    rows.forEach(r => { r.p = +r.p.toFixed(4); if (table) r.name = table.get(r.id).name; });
    return rows.sort((x, y) => y.p - x.p);
  };
  const ou = {}; let dropped = [];
  for (const [sid0, s] of Object.entries(raw)) {
    const sp = dex.species.get(sid0);
    if (!sp.exists || rt.isBannedSpecies(sp)) { dropped.push(sid0); continue; }
    ou[sp.id] = {
      name: sp.name, level: 100,
      moves: norm(s.moves, 'move', dex.moves, false),
      items: norm(s.items, 'item', dex.items, true),
      abilities: norm(s.abilities, 'ability', dex.abilities, true),
      tera: norm(s.tera, null, null, true).map(r => ({id: r.name, name: r.name, p: r.p})),
      spreads: (s.spreads || []).slice(0, 8).map(r => ({nature: toID(r.nature), evs: r.stats,
        ivs: [31,31,31,31,31,31], p: +(r.percentage / 100).toFixed(4)})),
      teammates: (s.teammates || []).slice(0, 8).map(r => ({id: toID(r.name), name: ('' + r.name).trim(), p: +(r.percentage / 100).toFixed(4)})),
    };
  }
  fs.writeFileSync(path.join(out, 'gen9ou_sets.json'),
    JSON.stringify({format: 'gen9ou', source: path.basename(ouPath), species: ou}));
  ouOut = {species: Object.keys(ou).length, dropped_banned_species: dropped.length};
}
console.log(JSON.stringify({version: require(path.resolve(process.env.SD_DIR, 'package.json')).version,
  dex: {moves: Object.keys(moves).length, abilities: Object.keys(abilities).length,
        items: Object.keys(items).length, species: Object.keys(species).length},
  randbats: {species: Object.keys(rb).length, teams: N}, ou: ouOut}));
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--showdown", default=str(DEFAULT_SD), help="local pokemon-showdown checkout (built dist/)")
    ap.add_argument("--teams", type=int, default=20000, help="random teams to sample for randbats priors")
    ap.add_argument("--seed", default="1,2,3,4", help="generator seed (4 ints)")
    ap.add_argument("--ou-rating", default="1500", choices=["0", "1000", "1500", "1825"])
    ap.add_argument("--out", default=str(ROOT / "data" / "knowledge"))
    args = ap.parse_args()

    sd = Path(args.showdown)
    if not (sd / "dist" / "sim" / "index.js").exists():
        print(f"no built Showdown at {sd} (need dist/sim)", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ou_sets = DEFAULT_OU / f"sets_{args.ou_rating}.json"

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(NODE_SCRIPT)
        js = f.name
    env = dict(os.environ, SD_DIR=str(sd), OUT_DIR=str(out), N_TEAMS=str(args.teams),
               SEED=args.seed, OU_SETS=str(ou_sets) if ou_sets.exists() else "")
    t0 = time.time()
    try:
        proc = subprocess.run(["node", js], env=env, capture_output=True, text=True, timeout=1800)
    finally:
        os.unlink(js)
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr)
        return proc.returncode
    stats = json.loads(proc.stdout.strip().splitlines()[-1])
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
        "showdown_dir": str(sd), "showdown_version": stats["version"],
        "randbats": {"format": "gen9randombattle", "teams_sampled": args.teams, "seed": args.seed,
                     **stats["randbats"]},
        "gen9ou": ({"source": str(ou_sets), "rating": args.ou_rating, **stats["ou"]}
                   if stats.get("ou") else None),
        "dex": stats["dex"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
