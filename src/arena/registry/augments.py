"""Self-registering augmentation registry (mirrors the agent one).

``@register_augment("name")`` on each Augmentation subclass; the TUI lists these
as configurable off/passive/tool toggles, and ``AugmentationRegistry.from_config``
builds from here instead of a hard-coded list.
"""

from __future__ import annotations

AUGMENT_REGISTRY: dict[str, type] = {}


def register_augment(name: str):
    def deco(cls):
        AUGMENT_REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    import arena.augment.ability  # noqa: F401
    import arena.augment.battle_log_tool  # noqa: F401
    import arena.augment.damage  # noqa: F401
    import arena.augment.matchup  # noqa: F401
    import arena.augment.opponent_sets  # noqa: F401
    import arena.augment.pokedex  # noqa: F401
    import arena.augment.speed  # noqa: F401
    import arena.augment.stats  # noqa: F401
    import arena.augment.switch_eval  # noqa: F401
    import arena.augment.type_matchup  # noqa: F401


def entries() -> dict[str, type]:
    _ensure_loaded()
    return dict(AUGMENT_REGISTRY)
