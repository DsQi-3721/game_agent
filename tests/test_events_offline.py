"""UI-M1 offline: event schema, emitter->file->reader, tolerant tail, logger tee."""

import asyncio
from pathlib import Path

from arena.events.emitter import EventEmitter
from arena.events.reader import EventReader
from arena.events.schema import Event, EventType
from arena.logging.trajectory import TrajectoryLogger


def test_event_roundtrip():
    e = Event(EventType.METRIC, ts=1.0, seq=3, run="x", data={"elo": {"a": 1010.0}})
    back = Event.from_json(e.to_json())
    assert back is not None and back.type is EventType.METRIC and back.seq == 3
    assert back.data["elo"]["a"] == 1010.0


def test_from_json_tolerant():
    assert Event.from_json("") is None
    assert Event.from_json('{"type": "metr')  is None  # half-written line
    assert Event.from_json("not json") is None


def test_emitter_writes_jsonl(tmp_path):
    path = tmp_path / "events.jsonl"
    em = EventEmitter.to_file(path, "exp")
    em.emit(EventType.RUN_STARTED, total_battles=4)
    em.emit(EventType.MATCH_FINISHED, a="x", b="y", a_wins=3, b_wins=1)
    em.emit(EventType.RUN_FINISHED, aborted=None)
    em.close()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    seqs = [Event.from_json(l).seq for l in lines]
    assert seqs == [1, 2, 3]


async def test_reader_tails_until_finished(tmp_path):
    path = tmp_path / "events.jsonl"
    em = EventEmitter.to_file(path, "exp")
    em.emit(EventType.RUN_STARTED)
    em.emit(EventType.PAIRING_STARTED, a="x", b="y", index=0)
    em.emit(EventType.RUN_FINISHED)
    em.close()
    got = []
    async for ev in EventReader(path, poll=0.01):
        got.append(ev.type)
    assert got == [EventType.RUN_STARTED, EventType.PAIRING_STARTED, EventType.RUN_FINISHED]


def test_trajectory_logger_tee(tmp_path):
    seen = []
    lg = TrajectoryLogger(tmp_path / "traj.jsonl", "agentA", on_turn=lambda rec: seen.append(rec))
    lg.log_turn(battle_tag="battle-x-1", turn=3, state_text="s", prompt="p",
                raw_llm_output="Action: tackle", tool_calls=[], parsed_order="/choose move 1",
                fallback_used=False, note="")
    lg.close()
    assert len(seen) == 1
    assert seen[0]["agent"] == "agentA" and seen[0]["battle_tag"] == "battle-x-1"
