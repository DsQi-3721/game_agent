"""Event emitter + sinks. The evaluator holds an EventEmitter and calls helper
methods; sinks fan the event out (JSONL file for the TUI, stdout for piping, or
a no-op when not launched with --emit-events)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Protocol

from arena.events.schema import Event, EventType


class EventSink(Protocol):
    def emit(self, ev: Event) -> None: ...
    def close(self) -> None: ...


class NullSink:
    def emit(self, ev: Event) -> None: ...
    def close(self) -> None: ...


class JsonlEventSink:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def emit(self, ev: Event) -> None:
        self._f.write(ev.to_json() + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:  # noqa: BLE001
            pass


class StdoutEventSink:
    def emit(self, ev: Event) -> None:
        sys.stdout.write(ev.to_json() + "\n")
        sys.stdout.flush()

    def close(self) -> None:
        pass


class EventEmitter:
    def __init__(self, sinks: list[EventSink], run: str, clock=time.time):
        self.sinks = sinks
        self.run = run
        self._seq = 0
        self._clock = clock

    def emit(self, type: EventType, **data) -> None:
        self._seq += 1
        ev = Event(type=type, ts=self._clock(), seq=self._seq, run=self.run, data=data)
        for s in self.sinks:
            try:
                s.emit(ev)
            except Exception:  # noqa: BLE001  (a sink must never break the eval)
                pass

    def close(self) -> None:
        for s in self.sinks:
            s.close()

    # --- convenience helpers -------------------------------------------
    @classmethod
    def to_file(cls, path: Path | str, run: str) -> "EventEmitter":
        return cls([JsonlEventSink(path)], run)

    @classmethod
    def disabled(cls, run: str = "") -> "EventEmitter":
        return cls([NullSink()], run)
