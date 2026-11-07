"""Lead statistics for OU team preview from public replays (PokéChamp, full Showdown logs).

    python scripts/build_lead_stats.py --min-elo 1600 --exclude /data/wangxinqi/s1v3/t6/valid.jsonl \
        --out data/knowledge/gen9ou_lead_rates.json

Team preview and the leads are public in a spectator log: ``|poke|pX|<species>, ...`` lists both teams and
the first ``|switch|pXa: ...`` of each side is its lead. For every rated gen9ou battle at or above
``--min-elo`` (both sides counted), this counts how often each species is brought and how often it leads.
Battles whose numeric id appears in an ``--exclude`` file (held-out evaluation sets) are skipped. Formes
hidden at preview (``Urshifu-*``) are counted under the base species. The output records its source,
filters, month range and counts, and every skipped / unparsable log is counted rather than dropped silently.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from poke_env.data import to_id_str

ELO_MIN = {"1000-1199": 1000, "1200-1399": 1200, "1400-1599": 1400, "1600-1799": 1600, "1800+": 1800}
POKE = re.compile(r"^\|poke\|(p[12])\|([^|,]+)", re.M)
SWITCH = re.compile(r"^\|(?:switch|drag)\|(p[12])a: [^|]*\|([^|,]+)", re.M)


def species_key(name: str) -> str:
    return to_id_str(name.strip().removesuffix("-*"))


def battle_number(text: str) -> str:
    m = re.match(r"(\d+)", text)
    return m.group(1) if m else text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/dataset/pokechamp/data/*.parquet")
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--min-elo", type=int, default=1600)
    ap.add_argument("--exclude", action="append", default=[], help="jsonl with 'request'/'group' ids to hold out")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    held_out: set[str] = set()
    for path in args.exclude:
        for line in open(path):
            m = re.search(rf"{args.format}-(\d+)", line)
            if m:
                held_out.add(m.group(1))
    team, lead, c, months = Counter(), Counter(), Counter(), Counter()
    for f in sorted(glob.glob(args.data)):
        t = pq.read_table(f, columns=["text", "gamemode", "elo", "battle_id", "month_year"])
        for text, mode, elo, bid, my in zip(*(t[k].to_pylist() for k in ("text", "gamemode", "elo", "battle_id",
                                                                          "month_year"))):
            if mode != args.format:
                continue
            if ELO_MIN.get(elo, 0) < args.min_elo:
                c["below_elo"] += 1
                continue
            if battle_number(bid) in held_out:
                c["held_out"] += 1
                continue
            pokes = POKE.findall(text)
            leads = {}
            for side, sp in SWITCH.findall(text):
                leads.setdefault(side, species_key(sp))
                if len(leads) == 2:
                    break
            sides = {s: [species_key(sp) for p, sp in pokes if p == s] for s in ("p1", "p2")}
            if any(len(v) != 6 for v in sides.values()) or len(leads) != 2:
                c["unparsable"] += 1
                continue
            for s, roster in sides.items():
                ld = leads[s]
                if ld not in roster:          # a forme revealed on switch-in (e.g. hidden at preview)
                    base = next((r for r in roster if ld.startswith(r) or r.startswith(ld)), None)
                    if base is None:
                        c["lead_not_in_preview"] += 1
                        continue
                    ld = base
                team.update(roster)
                lead[ld] += 1
                c["sides"] += 1
            c["battles"] += 1
            months[my] += 1
    rates = {sp: {"team": n, "lead": lead[sp]} for sp, n in team.most_common()}
    out = {"source": "PokéChamp (milkkarten/pokechamp), public Showdown logs", "format": args.format,
           "min_elo": args.min_elo, "held_out_battles": len(held_out), "months": dict(sorted(months.items())),
           "counts": dict(c), "species": rates}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "species"}, indent=1))
    print("top leads:", sorted(((v["lead"] / v["team"], s, v["team"]) for s, v in rates.items() if v["team"] >= 500),
                               reverse=True)[:12])


if __name__ == "__main__":
    main()
