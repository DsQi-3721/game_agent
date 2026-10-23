"""LLM response cache.

Key = hash(litellm_id, messages, tools, sample_idx). Only caches deterministic
calls (temperature == 0); self-consistency uses temperature > 0 and is therefore
never collapsed, while distinct ``sample_idx`` values stay distinct in the key.
In-memory for now (per-process); a disk backend can slot in behind the same API.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional, Sequence

from arena.llm.client import LLMResponse


class ResponseCache:
    def __init__(self):
        self._store: dict[str, LLMResponse] = {}

    @staticmethod
    def is_cacheable(temperature: float) -> bool:
        return temperature == 0.0

    @staticmethod
    def key(litellm_id: str, messages: list[dict], tools: Optional[Sequence[dict]], sample_idx: int,
            extra: Optional[dict] = None) -> str:
        """``extra`` carries request knobs that change the output (thinking flag,
        top_p, ...); omitted when empty so pre-existing keys stay stable."""
        payload = [litellm_id, messages, tools, sample_idx]
        if extra:
            payload.append(extra)
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[LLMResponse]:
        return self._store.get(key)

    def put(self, key: str, value: LLMResponse) -> None:
        self._store[key] = value
