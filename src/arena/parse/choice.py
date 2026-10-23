"""Map a model's answer onto one of the numbered legal options (``state.views``).

Accepts, in order of preference: an option number ("3", "#3", "option 3",
"3. move closecombat"), then a move/species id or name ("switch Garchomp",
"Close Combat"). A "tera"/"terastallize" keyword upgrades a move to its tera
variant when one is legal. Garbage returns ``None`` (caller falls back).
"""

from __future__ import annotations

import re
from typing import Optional

from poke_env.data import to_id_str

from arena.state.views import Option

_NUM = re.compile(r"^\s*(?:option|choice|action)?\s*#?\s*(\d{1,2})(?!\d)", re.IGNORECASE)
_TERA = re.compile(r"\b(terastalli[sz]e|tera)\b(?!\s*blast)", re.IGNORECASE)
_SWITCH = re.compile(r"\bswitch(?:es|ing)?\b", re.IGNORECASE)
_MOVE = re.compile(r"\b(?:move|use|attack)\b", re.IGNORECASE)


def _tera_twin(opt: Option, options: list[Option]) -> Optional[Option]:
    for o in options:
        if o.kind == "move" and o.tera and o.id == opt.id:
            return o
    return None


def parse_choice(text: str, options: list[Option]) -> Optional[Option]:
    if not text or not options:
        return None
    s = text.strip()
    wants_tera = bool(_TERA.search(s))

    m = _NUM.match(s)
    if m:
        i = int(m.group(1))
        if 1 <= i <= len(options):
            opt = options[i - 1]
            if wants_tera and opt.kind == "move" and not opt.tera:
                return _tera_twin(opt, options) or opt
            return opt

    norm = to_id_str(s)
    if not norm:
        return None
    kind = "switch" if _SWITCH.search(s) else ("move" if _MOVE.search(s) else None)
    base = [o for o in options if not o.tera]
    base.sort(key=lambda o: len(o.id), reverse=True)  # "icebeam" before "ice"
    for want in ([kind] if kind else []) + [None]:
        for o in base:
            if want is not None and o.kind != want:
                continue
            if o.id and o.id in norm:
                if wants_tera and o.kind == "move":
                    return _tera_twin(o, options) or o
                return o
    return None
