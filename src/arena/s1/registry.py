"""Model registry of the System-1 loop: every trained student, its data, metrics and status.

``runs/s1_models/registry.json``:
  {"champion": "<id>" | null,
   "models": {"<id>": {"id", "base_model", "adapter", "backend", "parent", "dataset", "n_train",
                        "created", "metrics": {"valid": {...}, "online": {...}}, "status", "notes"}}}

Promotion compares a candidate with the current champion on the SAME held-out set: it must agree
with the teacher more (by ``min_gain``) without getting less calibrated (ECE up by at most
``max_ece_increase``). Online battle results, when recorded, must not be worse either.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional

DEFAULT_PATH = Path("runs/s1_models/registry.json")


class Registry:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"champion": None, "models": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))

    def add(self, entry: dict) -> dict:
        entry = {"created": dt.datetime.now().isoformat(timespec="seconds"), "status": "candidate", "metrics": {},
                 **entry}
        self.data["models"][entry["id"]] = entry
        self.save()
        return entry

    def get(self, model_id: str) -> Optional[dict]:
        return self.data["models"].get(model_id)

    def update(self, model_id: str, **fields) -> dict:
        entry = self.data["models"][model_id]
        for k, v in fields.items():
            if k == "metrics":
                entry.setdefault("metrics", {}).update(v)
            else:
                entry[k] = v
        self.save()
        return entry

    def champion(self) -> Optional[dict]:
        cid = self.data.get("champion")
        return self.data["models"].get(cid) if cid else None

    def promote(self, model_id: str, reason: str) -> None:
        old = self.data.get("champion")
        if old and old in self.data["models"]:
            self.data["models"][old]["status"] = "retired"
        self.data["champion"] = model_id
        self.data["models"][model_id].update(status="champion", promoted_reason=reason)
        self.save()


def decide_promotion(candidate: dict, champion: Optional[dict], *, min_gain: float = 0.01,
                     max_ece_increase: float = 0.02) -> tuple[bool, str]:
    cv = (candidate.get("metrics") or {}).get("valid", {}).get("overall", {})
    if not cv:
        return False, "candidate has no validation metrics"
    if champion is None:
        base = (candidate.get("metrics") or {}).get("base_valid", {}).get("overall", {})
        if base and cv.get("agree", 0) < base.get("agree", 0) + min_gain:
            return False, f"no better than its untrained base ({cv.get('agree')} vs {base.get('agree')})"
        return True, "first model that beats its untrained base" if base else "first model"
    chv = (champion.get("metrics") or {}).get("valid", {}).get("overall", {})
    if cv.get("agree", 0) < chv.get("agree", 0) + min_gain:
        return False, f"agreement {cv.get('agree')} not above champion {chv.get('agree')} + {min_gain}"
    if (cv.get("ece") or 0) > (chv.get("ece") or 0) + max_ece_increase:
        return False, f"calibration worse (ECE {cv.get('ece')} vs {chv.get('ece')})"
    co = (candidate.get("metrics") or {}).get("online", {}).get("win_rate")
    ho = (champion.get("metrics") or {}).get("online", {}).get("win_rate")
    if co is not None and ho is not None and co < ho:
        return False, f"online win rate {co} below champion {ho}"
    return True, f"agreement {cv.get('agree')} > champion {chv.get('agree')}"
