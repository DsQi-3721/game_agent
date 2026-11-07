"""One LLM call "in a context space": per-space sampling, thinking flag, timeout.

Shared by every agent that runs several kinds of calls (HyMem's planner /
executor / isolated reasoning / fold, the long-context baselines) so that, in a
comparison, all of them reach the model through exactly the same request path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from arena.llm.client import LLMResponse


@dataclass
class SpaceConfig:
    max_tokens: int = 320
    temperature: float = 0.7
    top_p: Optional[float] = 0.8
    presence_penalty: Optional[float] = 0.0
    thinking: bool = False
    timeout_s: Optional[float] = None


def model_spec(client, model: str):
    """The registry ModelSpec behind ``model`` (None for mock clients / unknown)."""
    reg = getattr(client, "registry", None)
    try:
        return reg.get(model) if reg is not None else None
    except Exception:  # noqa: BLE001
        return None


def wants_thinking_flag(client, model: str) -> bool:
    """Only servers that understand ``chat_template_kwargs`` (local Qwen) get it."""
    spec = model_spec(client, model)
    if spec is None:
        return getattr(client, "registry", None) is None   # mock clients: always
    return "chat_template_kwargs" in (spec.extra_body or {})


async def complete_in_space(client, model: str, messages: list[dict], sc: SpaceConfig, *,
                            tools=None, extra_body=None) -> tuple[LLMResponse, float]:
    """-> (response, wall-clock latency)."""
    kw: dict = {"temperature": sc.temperature, "max_tokens": sc.max_tokens}
    if sc.top_p is not None:
        kw["top_p"] = sc.top_p
    if sc.presence_penalty is not None:
        kw["presence_penalty"] = sc.presence_penalty
    if wants_thinking_flag(client, model):
        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(sc.thinking)}}
    if extra_body:
        from arena.llm.client import deep_merge
        kw["extra_body"] = deep_merge(kw.get("extra_body"), extra_body)
    if sc.timeout_s:
        kw["timeout"] = sc.timeout_s
    t0 = time.monotonic()
    resp = await client.complete(model, messages, tools=tools, sample_idx=0, **kw)
    return resp, time.monotonic() - t0
