"""Raw per-player stream capture: everything the client received for a battle room and everything it sent.

One JSONL file per (battle, player) under ``AgentPlayer(stream_dir=...)`` or ``$ARENA_STREAM_DIR``::

    {"k": "meta", "seq": 0, ...provenance...}
    {"k": "in",  "seq": 1, "t": 1761.2, "data": ">battle-gen9ou-12\\n|request|{...}"}
    {"k": "act", "seq": 2, "t": ..., "rqid": 7, "turn": 3, "order": "/choose move 1", "agent": "..."}
    {"k": "out", "seq": 3, "t": ..., "rqid": 7, "data": "/choose move 1"}

``in`` records are the websocket messages verbatim (poke-env splits them on newline and ``|``; joining back
is lossless). ``act`` is what the agent returned (with ``substitute`` when the adapter replaced it, e.g. a
random team order); ``out`` is what actually left the client, which also covers poke-env's own default or
random orders after an ``[Invalid choice]``. Together with the server's log (seed, teams, inputLog) this gives
the decision trace: proposed -> submitted -> server feedback -> accepted -> resolved.
"""

from __future__ import annotations

import contextvars
import json
import subprocess
import time
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from poke_env.data import to_id_str

STREAM_SCHEMA = "stream-1"

#: (battle_tag, rqid) of the request the agent is answering; set by AgentPlayer around ``agent.act`` so
#: per-decision logs (``TrajectoryLogger``) can be joined with the stream and the server log by request id
CURRENT_REQUEST: contextvars.ContextVar = contextvars.ContextVar("arena_current_request", default=None)


@lru_cache(maxsize=1)
def code_version() -> dict:
    """Commit of this repository and whether the working tree had uncommitted changes."""
    root = Path(__file__).resolve().parents[3]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                    capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        head, dirty = "", None
    return {"repo_commit": head or None, "repo_dirty": dirty}


@lru_cache(maxsize=1)
def client_version() -> str | None:
    try:
        return metadata.version("poke_env")
    except metadata.PackageNotFoundError:
        return None


class StreamRecorder:
    def __init__(self, root: str | Path, username: str, meta: dict | None = None):
        self.root = Path(root)
        self.username = username
        self.meta = dict(meta or {})
        self._files: dict[str, object] = {}
        self._seq: dict[str, int] = {}
        self._rqid: dict[str, int | None] = {}

    def path(self, tag: str) -> Path:
        return self.root / f"{tag}.{to_id_str(self.username)}.jsonl"

    def _open(self, tag: str):
        f = self._files.get(tag)
        if f is None:
            self.root.mkdir(parents=True, exist_ok=True)
            f = self._files[tag] = open(self.path(tag), "a", encoding="utf-8")
            self._seq[tag] = 0
            self._rqid[tag] = None
            self._write(tag, {"k": "meta", "schema": STREAM_SCHEMA, "battle_tag": tag, "username": self.username,
                              "client": "poke-env", "client_version": client_version(), **code_version(),
                              **self.meta})
        return f

    def _write(self, tag: str, rec: dict) -> None:
        f = self._files[tag]
        rec = {"k": rec.pop("k"), "seq": self._seq[tag], "t": round(time.time(), 3), **rec}
        self._seq[tag] += 1
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()

    def incoming(self, split_messages: list[list[str]]) -> None:
        """A battle-room websocket message, as poke-env split it (first line is ``>battle-...``)."""
        if not split_messages or not split_messages[0] or not split_messages[0][0].startswith(">"):
            return
        tag = split_messages[0][0][1:].strip()
        self._open(tag)
        for m in split_messages[1:]:
            if len(m) > 2 and m[1] == "request" and m[2]:
                try:
                    self._rqid[tag] = json.loads(m[2]).get("rqid")
                except ValueError:
                    pass
        self._write(tag, {"k": "in", "data": "\n".join("|".join(m) for m in split_messages)})

    def outgoing(self, room: str, message: str) -> None:
        if not room.startswith("battle-"):
            return
        self._open(room)
        self._write(room, {"k": "out", "rqid": self._rqid.get(room), "data": message})

    def act(self, tag: str, **fields) -> None:
        self._open(tag)
        self._write(tag, {"k": "act", "rqid": self._rqid.get(tag), **fields})

    def close(self, tag: str) -> None:
        f = self._files.pop(tag, None)
        if f is not None:
            f.close()


def load_stream(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def choice_attempts(records: list[dict]) -> list[dict]:
    """Every choice message the client sent for one battle, in order, with the server's answer:
    ``{"rqid", "data", "error"}`` where ``error`` is the ``|error|`` line that refused it (None: accepted).
    Accepted attempts line up one-to-one with the player's lines in the server inputLog."""
    out: list[dict] = []
    for rec in records:
        if rec.get("k") == "out" and rec.get("data", "").startswith(("/choose", "/team")):
            out.append({"rqid": rec.get("rqid"), "data": rec["data"], "error": None})
        elif rec.get("k") == "in" and out and out[-1]["error"] is None:
            for line in rec["data"].split("\n"):
                if line.startswith("|error|"):
                    out[-1]["error"] = line
                    break
    return out


def rejected_before_accepted(records: list[dict]) -> list[list]:
    """[[k, choice], ...]: refused choices, each tagged with the index k of the accepted choice it preceded
    (the k-th inputLog line of this player), in the server's ``>pX`` input syntax."""
    out, k = [], 0
    for a in choice_attempts(records):
        if a["error"] is None:
            k += 1
        else:
            data = a["data"]
            choice = data[len("/choose "):] if data.startswith("/choose ") else "team " + data[len("/team "):]
            out.append([k, choice])
    return out
