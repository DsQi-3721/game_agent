"""Experience log of the System-1 loop (one JSONL per agent, beside its trajectory).

Three record types, all carrying the request context (agent, battle_tag, decision_id, purpose):

  s1        one System-One request: state, questions, answers (full distributions), backend,
            served model, role (primary / teacher / student / shadow), latency, input tokens
  decision  what the agent finally played and through which route (tier0 / fast_* / slow / ...),
            plus System 2's own choice when it completed -- the correction signal
  outcome   battle result (won, turns) -- the outcome signal for weighting

``RecordingClient`` wraps any client with a ``system_one`` coroutine (TypeSafe API, the local
student server, a mock) so recording is independent of which model answers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from arena.s1.context import current

SCHEMA = 1


def answer_dict(a) -> dict:
    return {"type": a.type, "choice": a.choice, "probabilities": a.probabilities, "confidence": a.confidence,
            "noul": a.noul, "score": a.score}


class ExperienceRecorder:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")
        self.n = 0

    @classmethod
    def beside(cls, logger, agent_name: Optional[str] = None) -> Optional["ExperienceRecorder"]:
        """``s1exp_<agent>.jsonl`` next to the agent's trajectory log (None without a logger)."""
        if logger is None:
            return None
        return cls(Path(logger.path).parent / f"s1exp_{agent_name or logger.agent_name}.jsonl")

    def write(self, rec: dict) -> None:
        rec = {"schema": SCHEMA, "ts": round(time.time(), 3), **rec}
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()
        self.n += 1

    def decision(self, **fields) -> None:
        self.write({"type": "decision", **current(), **fields})

    def outcome(self, **fields) -> None:
        self.write({"type": "outcome", **current(), **fields})

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:  # noqa: BLE001
            pass


class RecordingClient:
    """Any System-One client + an experience log; behaves exactly like the wrapped client."""

    def __init__(self, inner, recorder: Optional[ExperienceRecorder], backend: str, role: str = "primary"):
        self.inner = inner
        self.recorder = recorder
        self.backend = backend
        self.role = role

    def __getattr__(self, name):          # configured, n_calls, latencies, aclose, ...
        return getattr(self.inner, name)

    async def system_one(self, state, questions, model=None, headers=None):
        t0 = time.perf_counter()
        try:
            res = await self.inner.system_one(state, questions, model=model, headers=headers)
        except Exception as e:  # noqa: BLE001
            if self.recorder is not None:
                self.recorder.write({"type": "s1_error", **current(), "backend": self.backend, "role": self.role,
                                     "error": f"{type(e).__name__}: {str(e)[:200]}",
                                     "latency_s": round(time.perf_counter() - t0, 4)})
            raise
        if self.recorder is not None:
            self.recorder.write({"type": "s1", **current(), "backend": self.backend, "role": self.role,
                                 "model": res.model, "state": state, "questions": questions,
                                 "answers": {q: answer_dict(a) for q, a in res.answers.items()},
                                 "latency_s": round(res.latency_s or time.perf_counter() - t0, 4),
                                 "input_tokens": res.input_tokens})
        return res


def load_records(paths) -> list[dict]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return out
