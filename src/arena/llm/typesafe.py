"""Client for TypeSafe's System One API (Jev): typed questions in, calibrated answers out.

    POST {base}/v1/systemone   {"model", "state", "questions": {id: question}}
    -> {"model", "answers": {id: answer}, "usage": {"input_tokens", "output_tokens"}}

Question types (docs.typesafe.ai/primitives):
  choice -- pick one key of ``criteria`` (<= 255 options): returns choice, probabilities, confidence
  noul   -- probability that a statement is true
  score  -- position on an ordered rubric (<= 10 levels): returns score, probabilities, confidence

Plain httpx (no SDK dependency). ``TYPESAFE_API_KEY`` comes from the environment or the
project ``.env``; ``TYPESAFE_BASE_URL`` can point the same client at any TypeSafe-compatible
server (e.g. a self-hosted kev model), so code written against Jev keeps working there.
The httpx client is created lazily inside the running event loop (poke-env drives agents on
its own loop thread).
"""

from __future__ import annotations

import asyncio
import inspect
import os
import random
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib.parse import urlparse

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
RETRY_STATUS = {429, 529, 500, 502, 503, 504}


def choice(instructions: Any, criteria: dict) -> dict:
    """Choice question. ``criteria``: option name -> description (or None)."""
    if not criteria or len(criteria) > 255:
        raise ValueError("a choice needs 1..255 options")
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def noul(instructions: Any, true: Optional[str] = None, false: Optional[str] = None) -> dict:
    """Yes/no question; optional descriptions of what counts as true / false."""
    q: dict = {"type": "noul", "instructions": instructions}
    if true is not None or false is not None:
        q["criteria"] = {k: v for k, v in (("true", true), ("false", false)) if v is not None}
    return q


def score(instructions: Any, levels: list) -> dict:
    """Ordered rubric; level i scores i."""
    if not 1 <= len(levels) <= 10:
        raise ValueError("a score needs 1..10 levels")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


@dataclass
class Answer:
    type: str
    choice: Optional[str] = None
    probabilities: dict = field(default_factory=dict)
    confidence: Optional[float] = None
    noul: Optional[float] = None
    score: Optional[float] = None

    @property
    def p_top(self) -> float:
        return max(self.probabilities.values()) if self.probabilities else 0.0

    @property
    def margin(self) -> float:
        """Top-1 minus top-2 probability (a second confidence axis the docs recommend)."""
        ps = sorted(self.probabilities.values(), reverse=True)
        return ps[0] - (ps[1] if len(ps) > 1 else 0.0) if ps else 0.0

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])


@dataclass
class SystemOneResult:
    model: str
    answers: dict[str, Answer]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    raw: Optional[dict] = None


def parse_result(data: dict, latency_s: float = 0.0) -> SystemOneResult:
    answers = {}
    for qid, a in (data.get("answers") or {}).items():
        answers[qid] = Answer(type=a.get("type", ""), choice=a.get("choice"),
                              probabilities={str(k): float(v) for k, v in (a.get("probabilities") or {}).items()},
                              confidence=a.get("confidence"), noul=a.get("noul"), score=a.get("score"))
    usage = data.get("usage") or {}
    return SystemOneResult(model=data.get("model", ""), answers=answers,
                           input_tokens=int(usage.get("input_tokens") or 0),
                           output_tokens=int(usage.get("output_tokens") or 0), latency_s=latency_s, raw=data)


class TypeSafeError(RuntimeError):
    def __init__(self, status: int, body: str):
        msg = f"TypeSafe API {status}: {body[:300]}"
        if status == 451:
            msg += " -- TypeSafe does not serve the region this request exits from (check the network path)"
        super().__init__(msg)
        self.status = status


def system_proxy(base_url: str) -> Optional[str]:
    """The OS-configured proxy for ``base_url``'s host, or None.

    urllib / httpx consult the macOS (or Windows) system proxy only when no ``*_proxy`` variable
    is set at all -- and arena exports ``NO_PROXY=localhost`` for the local Showdown websocket,
    which silently turns the system proxy off for every other host as well. Re-apply it here
    (still honouring NO_PROXY for this host); explicit HTTP(S)_PROXY variables are left to httpx.
    """
    env = urllib.request.getproxies_environment()
    if any(k in env for k in ("https", "http", "all")):
        return None
    host = urlparse(base_url).hostname or ""
    try:
        if urllib.request.proxy_bypass(host):
            return None
        if sys.platform == "darwin":
            conf = urllib.request.getproxies_macosx_sysconf()
        elif sys.platform == "win32":
            conf = urllib.request.getproxies_registry()
        else:
            conf = {}
    except Exception:  # noqa: BLE001  (no system proxy information)
        return None
    return conf.get("https") or conf.get("http")


def _load_env() -> None:
    try:   # reuse the project .env loader (keys never live in code or YAML)
        from arena.llm.model_registry import _load_dotenv
        _load_dotenv()
    except Exception:  # noqa: BLE001
        pass


class TypeSafeClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: str = DEFAULT_MODEL, timeout_s: float = 30.0, max_retries: int = 3,
                 transport=None, keepalive_s: float = 600.0):
        _load_env()
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY") or ""
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._transport = transport          # httpx.MockTransport in tests
        self.keepalive_s = keepalive_s
        self._http = None
        self._loop = None
        self.n_calls = 0
        self.input_tokens = 0
        self.latencies: list[float] = []

    @property
    def configured(self) -> bool:
        return bool(self.api_key) or self.base_url != DEFAULT_BASE_URL

    def _client(self):
        import httpx

        loop = asyncio.get_running_loop()
        if self._http is None or self._loop is not loop:
            # keep the TLS connection across decisions: a cold handshake costs ~1-1.5 s through a proxy,
            # and System-2 turns can leave it idle for minutes (httpx's default expiry is 5 s)
            proxy = None if self._transport is not None else system_proxy(self.base_url)
            self._http = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s,
                                           transport=self._transport, proxy=proxy,
                                           limits=httpx.Limits(keepalive_expiry=self.keepalive_s))
            self._loop = loop
        return self._http

    async def system_one(self, state: Any, questions: dict, model: Optional[str] = None,
                         headers: Optional[dict] = None) -> SystemOneResult:
        if not self.configured:
            raise TypeSafeError(401, "TYPESAFE_API_KEY is not set (add it to the project .env)")
        body = {"model": model or self.model, "state": state, "questions": questions}
        hdrs = {"Authorization": f"Bearer {self.api_key or 'EMPTY'}", "Content-Type": "application/json"}
        hdrs.update(headers or {})
        delay = 0.5
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            resp = await self._client().post("/v1/systemone", json=body, headers=hdrs)
            latency = time.perf_counter() - t0
            if resp.status_code == 200:
                res = parse_result(resp.json(), latency)
                self.n_calls += 1
                self.input_tokens += res.input_tokens
                self.latencies.append(latency)
                return res
            if resp.status_code in RETRY_STATUS and attempt < self.max_retries:
                retry_after = resp.headers.get("retry-after")
                try:
                    wait = float(retry_after) if retry_after else delay
                except ValueError:
                    wait = delay
                await asyncio.sleep(wait + random.random() * 0.1)
                delay *= 2
                continue
            raise TypeSafeError(resp.status_code, resp.text)
        raise TypeSafeError(0, "retries exhausted")   # pragma: no cover

    async def list_models(self) -> list[dict]:
        resp = await self._client().get("/v1/models", headers={"Authorization": f"Bearer {self.api_key}"})
        if resp.status_code != 200:
            raise TypeSafeError(resp.status_code, resp.text)
        return list(resp.json().get("models") or [])

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


class ChunkedClient:
    """Split each request into groups of at most ``chunk`` questions (same state) and merge the
    answers -- for local servers that score a whole request in one batch and run out of memory on
    large ones (memory routing: ~140 questions). Answers are per-question, so the result is the same;
    latency is the sum of the parts."""

    def __init__(self, inner, chunk: int):
        self.inner = inner
        self.chunk = chunk

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def system_one(self, state, questions, model=None, headers=None) -> SystemOneResult:
        items = list(questions.items())
        answers, tokens, latency, model_id = {}, 0, 0.0, ""
        for i in range(0, len(items), self.chunk):
            res = await self.inner.system_one(state, dict(items[i:i + self.chunk]), model=model, headers=headers)
            answers.update(res.answers)
            tokens += res.input_tokens
            latency += res.latency_s or 0.0
            model_id = res.model
        return SystemOneResult(model=model_id, answers=answers, input_tokens=tokens, latency_s=latency)


class MockTypeSafeClient:
    """Offline stand-in: ``responder(state, questions) -> {qid: answer dict}`` (sync or async)."""

    configured = True

    def __init__(self, responder: Callable[[Any, dict], Any], latency_s: float = 0.0):
        self.responder = responder
        self.latency_s = latency_s
        self.calls: list[dict] = []
        self.n_calls = 0
        self.input_tokens = 0
        self.latencies: list[float] = []

    async def system_one(self, state, questions, model=None, headers=None) -> SystemOneResult:
        self.calls.append({"state": state, "questions": questions, "model": model})
        out = self.responder(state, questions)
        if inspect.isawaitable(out):
            out = await out
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        self.n_calls += 1
        self.latencies.append(self.latency_s)
        return parse_result({"model": "mock-jev", "answers": out, "usage": {"input_tokens": 0}}, self.latency_s)


def uniform_choice(criteria: dict, pick: Optional[str] = None, p: float = 0.0) -> dict:
    """Helper for mocks/tests: a choice answer favouring ``pick`` with probability ``p``."""
    keys = list(criteria)
    pick = pick or keys[0]
    rest = (1 - p) / max(1, len(keys) - 1) if len(keys) > 1 else 0.0
    probs = {k: (p if k == pick else rest) for k in keys} if len(keys) > 1 else {pick: 1.0}
    n = len(keys)
    conf = (max(probs.values()) - 1 / n) / (1 - 1 / n) if n > 1 else 1.0
    return {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs, "confidence": round(conf, 4)}
