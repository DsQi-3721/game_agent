"""Render + read input widgets for an agent's Param metadata (auto-generated form)."""

from __future__ import annotations

from typing import Iterable

from textual.containers import Horizontal
from textual.widgets import Checkbox, Input, Label, Select

from arena.registry.agents import AgentEntry


def param_widgets(entry: AgentEntry, model_names: Iterable[str]):
    """Yield one labeled widget per param (+ a model Select if needs_model)."""
    if entry.needs_model:
        models = list(model_names)
        yield Horizontal(
            Label("Model", classes="pname"),
            Select([(m, m) for m in models], value=(models[0] if models else Select.BLANK),
                   id="param_model", allow_blank=not models),
            classes="prow",
        )
    for p in entry.params:
        opts = p.options()
        if p.type is bool:
            w = Checkbox(value=bool(p.default), id=f"param_{p.name}")
        elif opts:
            w = Select([(str(o), str(o)) for o in opts],
                       value=str(p.default) if p.default is not None else Select.BLANK,
                       id=f"param_{p.name}", allow_blank=False)
        else:
            w = Input(value="" if p.default is None else str(p.default),
                      placeholder=p.help or p.name, id=f"param_{p.name}")
        yield Horizontal(Label(p.label or p.name, classes="pname"), w, classes="prow")


def _coerce(value: str, typ: type):
    if typ is int:
        return int(value)
    if typ is float:
        return float(value)
    if typ is bool:
        return bool(value)
    return value


def read_params(container, entry: AgentEntry) -> tuple[dict, str | None]:
    """Read param widgets back into (params_dict, model_or_None). Raises on bad input."""
    params: dict = {}
    model = None
    if entry.needs_model:
        sel = container.query_one("#param_model", Select)
        model = None if sel.value is Select.BLANK else str(sel.value)
    for p in entry.params:
        try:
            node = container.query_one(f"#param_{p.name}")
        except Exception:  # noqa: BLE001
            continue
        if isinstance(node, Checkbox):
            params[p.name] = bool(node.value)
        else:
            raw = node.value
            if raw is None or raw == "" or raw is Select.BLANK:
                continue
            params[p.name] = _coerce(str(raw), p.type)
    return params, model
