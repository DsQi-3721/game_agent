"""Self-registering agent registry.

Agents declare themselves with ``@register_agent(...)`` carrying param metadata.
The TUI config builder reads ``entries()`` to populate dropdowns and auto-generate
per-method parameter forms — so adding a new method needs zero UI changes.

``build_spec`` / ``is_llm_type`` keep the exact contract the old hard-coded
``arena.agents.registry`` had (which is now a re-export shim).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arena.config.schema import AgentConfig
from arena.eval.runner import AgentSpec, ExternalAgentSpec
from arena.registry.param import Param


@dataclass
class AgentEntry:
    type_id: str
    cls: type
    is_llm: bool = False
    needs_model: bool = False
    params: list[Param] = field(default_factory=list)
    label: str = ""
    help: str = ""


AGENT_REGISTRY: dict[str, AgentEntry] = {}


def register_agent(type_id: str, *, is_llm: bool = False, needs_model: bool = False,
                   params=(), label: str = "", help: str = ""):
    def deco(cls):
        AGENT_REGISTRY[type_id] = AgentEntry(
            type_id=type_id, cls=cls, is_llm=is_llm, needs_model=needs_model,
            params=list(params), label=label or type_id, help=help or (cls.__doc__ or "").strip().split("\n")[0],
        )
        cls.type_id = type_id
        return cls
    return deco


_loaded = False


def _ensure_loaded() -> None:
    """Import all built-in agent modules once so their decorators run."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    import arena.agents.baselines  # noqa: F401
    import arena.agents.hymem.agent  # noqa: F401
    import arena.agents.llm.cot  # noqa: F401
    import arena.agents.llm.flat_react  # noqa: F401
    import arena.agents.llm.io  # noqa: F401
    import arena.agents.llm.pokellmon  # noqa: F401
    import arena.agents.llm.react  # noqa: F401
    import arena.agents.llm.self_consistency  # noqa: F401
    import arena.agents.llm.summary_ctx  # noqa: F401
    import arena.agents.system1.agents  # noqa: F401
    import arena.agents.fusion.agent  # noqa: F401


def entries() -> dict[str, AgentEntry]:
    _ensure_loaded()
    return dict(AGENT_REGISTRY)


def is_llm_type(type_id: str) -> bool:
    _ensure_loaded()
    return type_id in AGENT_REGISTRY and AGENT_REGISTRY[type_id].is_llm


def build_spec(cfg: AgentConfig):
    _ensure_loaded()
    if cfg.kind == "external":
        if cfg.external is None:
            raise ValueError(f"external agent {cfg.name!r} missing 'external' config")
        return ExternalAgentSpec(name=cfg.name, username=cfg.external.username, cfg=cfg.external)
    if cfg.type not in AGENT_REGISTRY:
        raise KeyError(f"unknown agent type {cfg.type!r}; have {sorted(AGENT_REGISTRY)}")
    entry = AGENT_REGISTRY[cfg.type]
    if entry.needs_model and not cfg.model:
        raise ValueError(f"agent {cfg.name!r} of type {cfg.type} needs a 'model'")
    kwargs = dict(cfg.params)
    if entry.needs_model:
        kwargs["model"] = cfg.model
    return AgentSpec(cfg.name, lambda ctx, cls=entry.cls, kw=kwargs: cls(ctx, **kw))
