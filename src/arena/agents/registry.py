"""Backward-compat shim. The real registry moved to ``arena.registry.agents``
(self-registering decorator + param metadata). Existing imports keep working.
"""

from arena.registry.agents import AGENT_REGISTRY, build_spec, entries, is_llm_type  # noqa: F401

__all__ = ["build_spec", "is_llm_type", "entries", "AGENT_REGISTRY"]
