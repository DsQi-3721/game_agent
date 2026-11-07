"""LLM client layer.

A provider-agnostic ``LLMClient`` protocol with two implementations:
  - ``LiteLLMClient``: real models via litellm (vLLM / mlx_lm / DeepSeek / OpenAI / Gemini).
  - ``MockLLMClient``: deterministic, offline; lets the whole decision chain be
    tested without credentials or network.

Must stay async (``litellm.acompletion``) so it never blocks poke-env's POKE_LOOP.

Reasoning models (e.g. Qwen3.5 with thinking on) return their deliberation either
in a separate field (``reasoning_content`` / ``reasoning``) or inline before a
``</think>`` marker. Either way it lands in ``LLMResponse.reasoning`` and is kept
out of ``LLMResponse.text``, so callers only ever parse the final answer.
"""

from __future__ import annotations

import copy
import inspect
import json
import time
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Optional, Protocol, Sequence


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = ""


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False
    model: str = ""
    raw: Any = None
    reasoning: str = ""          # model "thinking", never part of ``text``
    cached_tokens: int = 0       # server-side prompt-cache hits, when reported
    finish_reason: str = ""


class LLMClient(Protocol):
    async def complete(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[Sequence[dict]] = None,
        temperature: float = 0.0,
        sample_idx: int = 0,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
        *,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        extra_body: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse: ...


def deep_merge(base: Optional[dict], over: Optional[dict]) -> dict:
    """Recursive dict merge (``over`` wins); inputs are not mutated."""
    out = copy.deepcopy(base) if base else {}
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def split_think(text: str) -> tuple[str, str]:
    """Split inline thinking off a completion: ``(answer, reasoning)``.

    Qwen-style models with thinking on emit ``...reasoning...</think>answer`` when
    the server runs without a reasoning parser (the opening ``<think>`` is part of
    the prompt). Text without ``</think>`` is returned unchanged.
    """
    if not text or "</think>" not in text:
        return text, ""
    head, _, tail = text.partition("</think>")
    return tail.lstrip(), head.replace("<think>", "", 1).strip()


def _reasoning_of(msg) -> str:
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(msg, attr, None)
        if isinstance(val, str) and val:
            return val
    psf = getattr(msg, "provider_specific_fields", None) or {}
    if isinstance(psf, dict):
        for k in ("reasoning_content", "reasoning"):
            if isinstance(psf.get(k), str) and psf[k]:
                return psf[k]
    return ""


def _cached_tokens_of(usage) -> int:
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    val = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", 0)
    return int(val or 0)


class LiteLLMClient:
    def __init__(self, registry, cache=None):
        # Initialize provider dependencies before the battle event loop starts.
        import_module("litellm")
        self.registry = registry
        self.cache = cache

    async def complete(self, model, messages, tools=None, temperature=0.0,
                       sample_idx=0, seed=None, max_tokens=None, *, top_p=None,
                       presence_penalty=None, extra_body=None, timeout=None) -> LLMResponse:
        import litellm

        spec = self.registry.get(model)
        body = deep_merge(spec.extra_body, extra_body)
        cache_key = None
        if self.cache is not None and self.cache.is_cacheable(temperature):
            extra = {k: v for k, v in (("top_p", top_p), ("presence_penalty", presence_penalty),
                                        ("max_tokens", max_tokens), ("extra_body", body or None))
                     if v is not None}
            cache_key = self.cache.key(spec.litellm_id, messages, tools, sample_idx, extra=extra or None)
            hit = self.cache.get(cache_key)
            if hit is not None:
                hit.cached = True
                return hit

        kwargs: dict[str, Any] = dict(
            model=spec.litellm_id,
            messages=messages,
            temperature=temperature,
        )
        if spec.api_base:
            kwargs["api_base"] = spec.api_base
        # OpenAI-compatible local servers ignore the key but the SDK insists on one.
        api_key = spec.api_key or ("EMPTY" if spec.api_base else None)
        if api_key:
            kwargs["api_key"] = api_key
        if tools:
            kwargs["tools"] = list(tools)
        if seed is not None:
            kwargs["seed"] = seed
        cap = max_tokens if max_tokens is not None else spec.max_tokens
        if cap is not None:
            kwargs["max_tokens"] = cap
        if top_p is not None:
            kwargs["top_p"] = top_p
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if body:
            kwargs["extra_body"] = body
        to = timeout if timeout is not None else spec.timeout_s
        if to is not None:
            kwargs["timeout"] = to
        if spec.max_retries is not None:
            kwargs["max_retries"] = spec.max_retries

        t0 = time.monotonic()
        resp = await litellm.acompletion(**kwargs)
        latency = time.monotonic() - t0

        choice = resp.choices[0]
        msg = choice.message
        tool_calls = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:  # noqa: BLE001
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(name=tc.function.name, arguments=args, id=getattr(tc, "id", "")))

        text, inline_reasoning = split_think(msg.content or "")
        reasoning = _reasoning_of(msg) or inline_reasoning

        usage = getattr(resp, "usage", None)
        ptok = getattr(usage, "prompt_tokens", 0) or 0
        ctok = getattr(usage, "completion_tokens", 0) or 0
        cost = spec.cost(ptok, ctok)  # authoritative price table first
        if cost is None:
            try:
                cost = 0.0 if spec.api_base else float(litellm.completion_cost(completion_response=resp))
            except Exception:  # noqa: BLE001
                cost = 0.0

        out = LLMResponse(
            text=text,
            tool_calls=tool_calls,
            prompt_tokens=ptok,
            completion_tokens=ctok,
            cost_usd=cost,
            latency_s=latency,
            model=spec.litellm_id,
            raw=resp,
            reasoning=reasoning,
            cached_tokens=_cached_tokens_of(usage),
            finish_reason=getattr(choice, "finish_reason", "") or "",
        )
        if cache_key is not None:
            self.cache.put(cache_key, out)
        return out


class MockLLMClient:
    """Offline client. ``responder(messages, tools, sample_idx) -> str | LLMResponse``.

    The responder may also be ``async`` (to simulate latency) or raise (to simulate
    provider errors). Every call is recorded in ``self.calls`` with the keyword
    arguments it received, so tests can assert on what each context space saw.
    """

    def __init__(self, responder: Callable[[list[dict], Optional[Sequence[dict]], int], Any]):
        self.responder = responder
        self.calls: list[dict] = []

    async def complete(self, model, messages, tools=None, temperature=0.0,
                       sample_idx=0, seed=None, max_tokens=None, **kw) -> LLMResponse:
        self.calls.append({"model": model, "messages": [dict(m) for m in messages],
                           "tools": tools, "temperature": temperature,
                           "sample_idx": sample_idx, "max_tokens": max_tokens, **kw})
        out = self.responder(messages, tools, sample_idx)
        if inspect.isawaitable(out):
            out = await out
        if isinstance(out, LLMResponse):
            return out
        text, reasoning = split_think(str(out))
        return LLMResponse(text=text, reasoning=reasoning, model=f"mock/{model}",
                           prompt_tokens=0, completion_tokens=0)
