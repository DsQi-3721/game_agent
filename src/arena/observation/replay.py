"""Regenerate each player's stream from a Showdown server log and compare it with a live capture.

The server log (``Config.logchallenges``; ``logs/<YYYY-MM>/<format>/<date>/<id>.log.json``) holds the seed,
both teams and the inputLog of every accepted choice. ``scripts/node/replay_battle.js`` feeds it through the
simulator at the pinned commit and returns, per player, exactly the protocol lines and requests that player
was sent. That makes the server log the strict-tier raw record: every request and private line can be
rebuilt from it, and ``compare`` checks the rebuild against what the client actually received.

Lines the server room adds on its own (joins, chat, timers, ``|t:|`` wall-clock stamps, the room header) are
not part of the simulation and are excluded on both sides; ``|error|`` lines (rejected choices) never reach
the inputLog, so they are counted separately as server feedback instead of compared.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

REPLAY_JS = Path(__file__).resolve().parents[3] / "scripts" / "node" / "replay_battle.js"

# battle-room lines produced by the server/room rather than the simulator, plus the wall-clock stamp
ROOM_ONLY = {
    "init", "deinit", "noinit", "title", "j", "J", "join", "l", "L", "leave", "n", "N", "name", "c", "c:", "chat",
    "raw", "html", "uhtml", "uhtmlchange", "inactive", "inactiveoff", "hidelines", "unlink", "notify",
    "tempnotify", "tempnotifyoff", "expire", "timestamp", ":", "t:", "userstats", "askreg",
}


def kind(line: str) -> str:
    return line.split("|", 2)[1] if line.startswith("|") and line.count("|") >= 1 else ""


def replay_logs(showdown_dir: str | Path, files: list[str | Path], node: str = "node",
                rejected: dict | None = None) -> list[dict]:
    """``rejected``: battle id -> {side: [[k, choice], ...]} refused choices from the client capture
    (``capture.rejected_before_accepted``); without them the replay matches every accepted step, but the
    server's error lines and re-sent requests (and the rqids after them) are missing."""
    if not files:
        return []
    extra = []
    tmp = None
    if rejected:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(rejected, tmp)
        tmp.close()
        extra = ["--rejected", tmp.name]
    try:
        out = subprocess.run([node, str(REPLAY_JS), str(showdown_dir), *extra, *map(str, files)],
                             capture_output=True, text=True, check=True)
    finally:
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)
    return [json.loads(line) for line in out.stdout.splitlines() if line.strip()]


def live_lines(records: list[dict]) -> list[str]:
    """Protocol lines a player received, in order, from a ``capture`` stream (room header lines dropped)."""
    lines: list[str] = []
    for r in records:
        if r.get("k") != "in":
            continue
        parts = r["data"].split("\n")
        lines.extend(parts[1:] if parts and parts[0].startswith(">") else parts)
    return lines


def split_stream(lines: list[str]) -> dict:
    """Separate simulator protocol lines, requests and error feedback; drop room-only lines."""
    protocol, requests, errors, dropped = [], [], [], 0
    for line in lines:
        if not line:
            continue
        k = kind(line)
        if k == "request":
            body = line[len("|request|"):]
            if body:
                requests.append(json.loads(body))
        elif k == "error":
            errors.append(line)
        elif k in ROOM_ONLY or not line.startswith("|"):
            dropped += 1
        else:
            protocol.append(line)
    return {"protocol": protocol, "requests": requests, "errors": errors, "dropped": dropped}


def _first_diff(a: list, b: list) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(live: list[str], replayed: list[str]) -> dict:
    """Line-level comparison of one player's live stream with its simulator replay."""
    a, b = split_stream(live), split_stream(replayed)
    res = {"protocol_live": len(a["protocol"]), "protocol_replay": len(b["protocol"]),
           "requests_live": len(a["requests"]), "requests_replay": len(b["requests"]),
           "errors_live": len(a["errors"]), "errors_replay": len(b["errors"]), "room_lines_dropped": a["dropped"]}
    i = _first_diff(a["protocol"], b["protocol"])
    j = _first_diff(a["requests"], b["requests"])
    res["protocol_equal"] = i is None
    res["requests_equal"] = j is None
    res["errors_equal"] = a["errors"] == b["errors"]
    if i is not None:
        res["protocol_first_diff"] = {"index": i, "live": a["protocol"][i:i + 3], "replay": b["protocol"][i:i + 3]}
    if j is not None:
        res["requests_first_diff"] = {"index": j,
                                      "live": a["requests"][j] if j < len(a["requests"]) else None,
                                      "replay": b["requests"][j] if j < len(b["requests"]) else None}
    return res
