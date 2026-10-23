"""Typed messages crossing HyMem's context boundaries, and their tolerant parsers.

Only these schema-bounded objects may enter the planner context (paper Eq. 4):
``TurnResult`` (environment), ``ExecResult`` (<RESULTS>), ``IsoReturn`` (<RETURN>),
the planner's own ``Decision``s and system ``Note``s. Every field is length-capped,
so the planner grows by bounded increments however verbose the helpers are.

Parsers accept both the paper's tag names and the reference implementation's
(``<isoReason>`` / ``<branch>``, ``<to_memory/>`` / ``<fold_thought>``), tolerate a
missing closing tag at the end of a reply, and ignore any leaked ``<think>`` block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_FLAGS = re.IGNORECASE | re.DOTALL


def clip(text: str, n: int) -> str:
    text = " ".join((text or "").split()) if "\n" not in (text or "") else (text or "").strip()
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=_FLAGS)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text


def _block(tag: str, text: str, allow_open: bool = True) -> list[str]:
    """Contents of every <tag>...</tag>; also a trailing unclosed <tag>... ."""
    out = [m.strip() for m in re.findall(rf"<{tag}>(.*?)</{tag}>", text, _FLAGS)]
    if allow_open:
        tail = re.search(rf"<{tag}>((?:(?!<{tag}>).)*)$", text, _FLAGS)
        if tail and f"</{tag}>".lower() not in tail.group(0).lower():
            out.append(tail.group(1).strip())
    return [x for x in out if x]


# --- planner directives ---------------------------------------------------------
@dataclass
class Directive:
    kind: str                     # task | iso | fold | answer
    text: str = ""                # task / iso objective / answer content
    description: str = ""         # iso: short label
    confidence: Optional[float] = None


def _confidence(text: str) -> Optional[float]:
    m = re.findall(r"<confidence>\s*([0-9]*\.?[0-9]+)", text, _FLAGS)
    if not m:
        return None
    try:
        return max(0.0, min(1.0, float(m[-1])))
    except ValueError:
        return None


def parse_planner(text: str) -> tuple[list[Directive], str]:
    """All directives in a planner reply (answer first, then fold, tasks, isos) + analysis."""
    t = strip_think(text)
    analysis = " ".join(_block("analysis", t))
    out: list[Directive] = []
    answers = _block("answer", t)
    if answers:
        out.append(Directive("answer", answers[-1], confidence=_confidence(t)))
    if re.search(r"<\s*(to_memory|fold_thought|fold)\s*/?\s*>", t, re.IGNORECASE):
        out.append(Directive("fold"))
    for task in _block("task", t, allow_open=False):
        out.append(Directive("task", task))
    for tag in ("isoReason", "branch"):  # case-insensitive match
        for body in _block(tag, t, allow_open=False):
            desc = _block("description", body, allow_open=False)
            prompt = _block("prompt", body, allow_open=False)
            objective = prompt[0] if prompt else re.sub(r"</?\w+>", " ", body).strip()
            label = desc[0] if desc else " ".join(objective.split()[:5])
            out.append(Directive("iso", objective, description=label))
    if not analysis and out:
        first = re.search(r"<(answer|task|isoReason|branch|to_memory|fold)", t, re.IGNORECASE)
        analysis = t[: first.start()].strip() if first else ""
    return out, clip(analysis, 600)


# --- typed returns ----------------------------------------------------------------
_FIELD = re.compile(r"^\s*[-*]?\s*(findings?|evidence|key evidence|sources?|status|gaps?|"
                    r"conclusion|confidence|notes?|assumptions?)\s*[:：]\s*(.*)$", re.IGNORECASE)


_ALIASES = {"finding": "findings", "findings": "findings", "evidence": "evidence",
            "key evidence": "evidence", "source": "sources", "sources": "sources", "status": "status",
            "gap": "gaps", "gaps": "gaps", "conclusion": "conclusion", "confidence": "confidence",
            "note": "notes", "notes": "notes", "assumption": "notes", "assumptions": "notes"}


def _fields(body: str, default: str) -> dict[str, str]:
    """Labelled sections (``FINDINGS: ...``) of a block; unlabeled text -> ``default``."""
    out: dict[str, list[str]] = {}
    key = default
    for line in body.splitlines():
        m = _FIELD.match(line)
        if m:
            key = _ALIASES[m.group(1).lower()]
            if m.group(2).strip():
                out.setdefault(key, []).append(m.group(2).strip())
        elif line.strip():
            out.setdefault(key, []).append(line.strip())
    return {k: "\n".join(v).strip() for k, v in out.items()}


@dataclass
class ExecResult:
    task: str
    findings: str
    evidence: str = ""
    sources: str = ""
    status: str = "success"
    gaps: str = ""
    n_tool_calls: int = 0
    raw_observations: list = field(default_factory=list)   # only for the no_typed_isolation ablation

    @property
    def success(self) -> bool:
        return self.status != "failure" and bool(self.findings.strip())

    def render(self) -> str:
        parts = [f'<RESULTS task="{clip(self.task, 120)}">', f"findings: {clip(self.findings, 700)}"]
        if self.evidence:
            parts.append(f"evidence: {clip(self.evidence, 300)}")
        parts.append(f"status: {self.status}")
        if self.gaps:
            parts.append(f"gaps: {clip(self.gaps, 200)}")
        parts.append("</RESULTS>")
        return "\n".join(parts)


def parse_results(text: str) -> Optional[dict]:
    blocks = _block("results", strip_think(text))
    if not blocks:
        return None
    f = _fields(blocks[-1], "findings")
    status = (f.get("status") or "success").split()[0].lower().strip(".,;")
    if status not in ("success", "partial", "failure"):
        status = "partial"
    return {"findings": f.get("findings", ""), "evidence": f.get("evidence", ""),
            "sources": f.get("sources", ""), "status": status, "gaps": f.get("gaps", "")}


@dataclass
class IsoReturn:
    description: str
    prompt: str
    conclusion: str
    evidence: str = ""
    confidence: float = 0.5
    notes: str = ""
    success: bool = True

    def render(self) -> str:
        parts = [f'<RETURN reason="{clip(self.description, 80)}">',
                 f"conclusion: {clip(self.conclusion, 600)}"]
        if self.evidence:
            parts.append(f"key evidence: {clip(self.evidence, 300)}")
        parts.append(f"confidence: {self.confidence:.2f}")
        if self.notes:
            parts.append(f"notes: {clip(self.notes, 200)}")
        if not self.success:
            parts.append("status: incomplete")
        parts.append("</RETURN>")
        return "\n".join(parts)


def parse_return(text: str) -> Optional[dict]:
    blocks = _block("return", strip_think(text))
    if not blocks:
        return None
    f = _fields(blocks[-1], "conclusion")
    conf = 0.5
    m = re.search(r"([0-9]*\.?[0-9]+)", f.get("confidence", ""))
    if m:
        try:
            conf = max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    return {"conclusion": f.get("conclusion", ""), "evidence": f.get("evidence", ""),
            "confidence": conf, "notes": f.get("notes", "")}


def extract_summary(text: str) -> str:
    blocks = _block("summary", strip_think(text))
    return blocks[-1] if blocks else ""


# --- planner history entries -------------------------------------------------------
@dataclass
class Entry:
    """One item of H^p (the planner's typed history since the last fold)."""

    kind: str          # turn | decision | results | return | note | analysis
    text: str
    decision: int
    turn: int

    def render(self) -> str:
        return self.text


def decision_entry(decision: int, turn: int, label: str, confidence: Optional[float]) -> Entry:
    conf = f" confidence={confidence:.2f}" if confidence is not None else ""
    return Entry("decision", f"<DECISION turn={turn}{conf}>{label}</DECISION>", decision, turn)


def note_entry(decision: int, turn: int, text: str) -> Entry:
    return Entry("note", f"<NOTE>{clip(text, 240)}</NOTE>", decision, turn)
