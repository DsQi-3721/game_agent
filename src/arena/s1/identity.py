"""Battle identity and the one split rule shared by every data builder (I/O audit §23).

A battle is identified by the platform it was played on plus that platform's battle id, never by a file name:
the two perspectives of one battle, several files of one replay and the same battle collected by two sources
must fall together, and battles of different servers must stay apart.

  showdown:gen9ou-2553459667          main server. Metamon parsed replays are named "<n>-gen9ou-<id>_..." (the
                                      leading number is Metamon's packaging, the same for both perspectives);
                                      PokéChamp and raw replay ids are "gen9ou-<id>"
  smogtours:gen9ou-736083             tournament server ("smogtours-gen9ou-<id>_...")
  metamon-selfplay:gen9ou-0000021993  Metamon self-play. The id is drawn at random for each perspective
                                      (``metamon/env/wrappers.py``), so the two sides of one game get different
                                      ids and are paired by content (``pair_selfplay``)
  local-<server>:gen9ou-605           our own Showdown servers (strict logs, agent runs). ``server`` names the
                                      installation: ``gpu8100`` (this GPU server, battle counter persisted in
                                      logs/lastbattle.txt) or ``mac`` (the first Jev experience)

``Splits.of``: a pinned test set first (T6, T7, the strict benchmark), otherwise a salted hash of the identity.
Every builder takes its split from here, so no two builders can split the same battle differently.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

METAMON_NAME = re.compile(r"^(?P<src>[a-z0-9]+)-(?P<fmt>gen\d+\w+?)-(?P<id>\d+)_(?P<rating>Unrated|\d+)_(?P<p1>.+?)_vs_"
                          r"(?P<p2>.+?)_(?P<date>[\d:-]+)_(?P<res>WIN|LOSS)\.json(?:\.lz4)?$")
_PREFIX = re.compile(r"^(?P<src>[a-z0-9]+)-(?P<fmt>gen\d+[a-z0-9]+)-(?P<id>\d+)$")
_TAG = re.compile(r"^(?:battle-)?(?P<fmt>gen\d+[a-z0-9]+)-(?P<id>\d+)$")
SPLITS_FILE = Path(__file__).resolve().parents[3] / "data" / "splits" / "s1-v6.json"
RULE = ("battle identity = platform + battle id (perspectives, files and sources of one battle merged); a battle in a "
        "pinned test set belongs to that set, any other battle to valid if md5(salt:identity) < valid_frac, else train")


PLATFORMS = ("showdown", "smogtours", "metamon-selfplay", "local-gpu8100", "local-mac")


def _platform(src: str) -> str:
    if src.isdigit():
        return "showdown"
    if src == "metamon":
        return "metamon-selfplay"
    return src                                   # smogtours, or another named server


def parse_metamon_name(name: str) -> Optional[dict]:
    """Fields of a Metamon trajectory file name, or None if it is not one."""
    m = METAMON_NAME.match(Path(name).name)
    return m.groupdict() if m else None


def battle_identity(ref: str, *, platform: Optional[str] = None) -> str:
    """``ref``: a Metamon file name, a Metamon group prefix ("<n>-gen9ou-<id>", "metamon-gen9ou-<id>"), or a battle
    id / room tag ("gen9ou-123", "battle-gen9ou-605") together with the ``platform`` it was played on
    ("showdown", "local-gpu8100", ...). Raises ValueError when the platform cannot be told."""
    ref = ref.removeprefix("metamon:")
    m = METAMON_NAME.match(Path(ref).name) or _PREFIX.match(ref.split("_")[0])
    if m and m["src"] != "battle" and platform is None:
        return f"{_platform(m['src'])}:{m['fmt']}-{m['id']}"
    t = _TAG.match(ref)
    if t and platform:
        return f"{platform}:{t['fmt']}-{t['id']}"
    raise ValueError(f"no battle identity for {ref!r} (platform={platform!r})")


def pair_selfplay(names: Iterable[str], leads: Callable[[str], tuple[str, str]], window_s: int = 120) -> dict[str, str]:
    """Merge the two perspectives of Metamon self-play games: swapped player names, end times within ``window_s``,
    opposite results and crosswise leads (``leads(name)`` = (own lead, opponent lead) at the first step).
    -> {file name: identity of the pair} for every name (unpaired names keep their own identity)."""
    rows = []
    for n in names:
        f = parse_metamon_name(n)
        if f is None or f["src"] != "metamon":
            continue
        try:
            t = datetime.strptime(f["date"], "%m-%d-%Y-%H:%M:%S")
        except ValueError:
            t = None
        rows.append((n, f, t))
    out = {n: battle_identity(n) for n, _f, _t in rows}
    by_players: dict[frozenset, list] = {}
    for r in rows:
        by_players.setdefault(frozenset((r[1]["p1"], r[1]["p2"])), []).append(r)
    for group in by_players.values():
        for i, (na, fa, ta) in enumerate(group):
            for nb, fb, tb in group[i + 1:]:
                if (fa["p1"], fa["p2"]) != (fb["p2"], fb["p1"]) or fa["res"] == fb["res"] or ta is None or tb is None \
                        or abs((ta - tb).total_seconds()) > window_s:
                    continue
                (own_a, opp_a), (own_b, opp_b) = leads(na), leads(nb)
                if own_a == opp_b and opp_a == own_b:
                    out[na] = out[nb] = min(out[na], out[nb])
    return out


@dataclass
class Splits:
    """The global split rule. ``pinned``: identity -> test set name; everything else is hashed into
    train / valid with ``salt``."""
    salt: str = "s1-v6"
    valid_frac: float = 0.05
    pinned: dict[str, str] = field(default_factory=dict)
    sets: dict[str, dict] = field(default_factory=dict)      # test set name -> description

    def of(self, identity: str) -> str:
        if identity in self.pinned:
            return self.pinned[identity]
        h = int(hashlib.md5(f"{self.salt}:{identity}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return "valid" if h < self.valid_frac else "train"

    @classmethod
    def load(cls, path: Path | str = SPLITS_FILE) -> "Splits":
        d = json.loads(Path(path).read_text())
        pinned = {i: name for name, s in d["sets"].items() for i in s["battles"]}
        return cls(salt=d["salt"], valid_frac=d["valid_frac"], pinned=pinned,
                   sets={k: {x: y for x, y in v.items() if x != "battles"} for k, v in d["sets"].items()})

    def save(self, path: Path | str = SPLITS_FILE) -> None:
        sets = {name: {**self.sets.get(name, {}), "battles": sorted(i for i, s in self.pinned.items() if s == name)}
                for name in sorted(set(self.pinned.values()))}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"rule": RULE, "salt": self.salt, "valid_frac": self.valid_frac, "sets": sets},
                                         indent=1))
