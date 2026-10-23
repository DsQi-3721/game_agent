"""Parameter metadata for self-registered methods.

Drives the TUI config form: each registered agent declares its tunable params
(name/type/default/choices), and the form auto-renders the right input widget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Param:
    name: str
    type: type                                   # int | float | str | bool
    default: Any = None
    choices: Optional[list] = None               # static choices -> dropdown
    choices_provider: Optional[Callable[[], list]] = None  # dynamic choices (e.g. model list)
    help: str = ""
    label: str = ""                              # human label for the TUI form (falls back to name)

    def options(self) -> Optional[list]:
        if self.choices_provider is not None:
            try:
                return list(self.choices_provider())
            except Exception:  # noqa: BLE001  (form must never crash on a bad provider)
                return self.choices
        return self.choices
