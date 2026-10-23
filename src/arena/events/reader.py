"""Robust async tail of an append-only events JSONL.

Handles: file not yet created (poll), partial last line (buffer until newline),
truncation/rotation (reopen if size shrinks), and resume (skip seq <= from_seq).
Yields Events as they land; stops after RUN_FINISHED/ERROR or on cancellation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

from arena.events.schema import Event, EventType


class EventReader:
    def __init__(self, path: Path | str, from_seq: int = 0, poll: float = 0.15):
        self.path = Path(path)
        self.from_seq = from_seq
        self.poll = poll

    async def __aiter__(self) -> AsyncIterator[Event]:
        buf = ""
        pos = 0
        f = None
        try:
            while True:
                if f is None:
                    if not self.path.exists():
                        await asyncio.sleep(self.poll)
                        continue
                    f = open(self.path, "r", encoding="utf-8")
                    pos = 0

                # truncation/rotation: file shrank -> reopen from start
                try:
                    if self.path.stat().st_size < pos:
                        f.close()
                        f = None
                        buf = ""
                        continue
                except FileNotFoundError:
                    f.close()
                    f = None
                    continue

                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
                if not chunk:
                    await asyncio.sleep(self.poll)
                    continue

                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    ev = Event.from_json(line)
                    if ev is None or ev.seq <= self.from_seq:
                        continue
                    self.from_seq = ev.seq
                    yield ev
                    if ev.type in (EventType.RUN_FINISHED, EventType.ERROR):
                        return
        finally:
            if f is not None:
                f.close()
