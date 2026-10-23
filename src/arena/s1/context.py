"""Per-request metadata for System-One calls: who asked, about which decision, for what.

Set with ``with s1_meta(purpose="action", decision_id=...)`` around a call; the recording client
reads it. contextvars travel into tasks created inside the block, so background requests (memory
routing) keep the decision they belong to without threading arguments through every layer.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager

_META: contextvars.ContextVar[dict] = contextvars.ContextVar("s1_meta", default={})


def current() -> dict:
    return dict(_META.get())


@contextmanager
def s1_meta(**fields):
    token = _META.set({**_META.get(), **{k: v for k, v in fields.items() if v is not None}})
    try:
        yield
    finally:
        _META.reset(token)
