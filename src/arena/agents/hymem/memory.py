"""Structured memories and the Fold operator Phi.

Three rewritable memories (paper Sec. 3.2; schemas from the reference
``hymem_memory.py``, contents adapted to battles):
  episode  -- task, key events, verified facts (opponent reveals, KOs), progress
  working  -- immediate goal, challenges, hypotheses, failed approaches, next actions
  tool     -- effective strategies, dead ends, derived rules, useful sources
plus the planner's own state summary.

Fold modes:
  two_stage (reference default) -- Stage 1: the planner, in its current full context,
      writes a <summary> handover; Stage 2: three parallel auxiliary-model JSON
      extractions from that summary (+ tool statistics + isolated-reasoning insights).
  direct (paper Eq. 13 literally) -- the three extractions read the raw typed
      histories (H^p, executor and isolated-reasoning logs) and the previous memories.
Every extraction is JSON-repaired, normalised and size-capped; on failure the
previous memory is kept (episode facts are refreshed by rule from the battle log),
so a small model's bad JSON never wipes the agent's state.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from arena.agents.hymem import prompts as P
from arena.agents.hymem.messages import clip, extract_summary

CAPS = {"key_events": 12, "verified_facts": 16, "active_hypotheses": 4, "failed_approaches": 5,
        "next_actions": 3, "effective_strategies": 6, "dead_ends": 5, "derived_rules": 6,
        "useful_sources": 6}


@dataclass
class HyMemStore:
    episode: Optional[dict] = None
    working: Optional[dict] = None
    tool: Optional[dict] = None
    summary: str = ""
    tool_calls: list = field(default_factory=list)    # {tool, args, ok, window}
    exec_log: list = field(default_factory=list)      # {task, status, findings, window}
    iso_log: list = field(default_factory=list)       # {description, conclusion, confidence, success, window}
    key_facts: list = field(default_factory=list)     # KeyFact (rule-extracted from the battle log)
    fold_events: list = field(default_factory=list)
    folds: int = 0

    @property
    def empty(self) -> bool:
        return not (self.episode or self.working or self.tool or self.summary)


# --- JSON repair / normalisation ---------------------------------------------------
def close_truncated_json(text: str) -> Optional[str]:
    """Salvage a JSON object cut off by the token limit: drop the unfinished element and
    close the open brackets. Returns None when nothing usable is left."""
    start = text.find("{")
    if start == -1:
        return None
    s = text[start:]
    stack, in_str, esc, safe = [], False, False, None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                return s[: i + 1]                       # complete object: nothing to salvage
            safe = (i + 1, list(stack))                 # after a closed element
        elif ch == ",":
            safe = (i, list(stack))                     # before the separator
    if safe is None:
        return None
    cut, st = safe
    closers = "".join("}" if c == "{" else "]" for c in reversed(st))
    return s[:cut].rstrip().rstrip(",") + closers


def parse_json(text: str) -> Optional[dict]:
    """json -> fenced block -> outermost {...}, with light repairs (trailing commas, smart
    quotes) and, last, salvage of an object truncated by the token limit."""
    if not text:
        return None
    cands = [text]
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        cands.append(m.group(1))
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        cands.append(text[i:j + 1])
    salvaged = close_truncated_json(text.split("```json")[-1] if "```json" in text else text)
    if salvaged:
        cands.append(salvaged)
    for c in cands:
        for variant in (c, re.sub(r",\s*([}\]])", r"\1", c.replace("“", '"').replace("”", '"'))):
            try:
                out = json.loads(variant)
                if isinstance(out, dict):
                    return out
            except Exception:  # noqa: BLE001
                continue
    return None


def _s(x, n: int) -> str:
    return clip(str(x), n) if x not in (None, "") else ""


def _lst(x) -> list:
    if isinstance(x, list):
        return x
    return [] if x in (None, "", {}) else [x]


def _int(x) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def norm_episode(d) -> Optional[dict]:
    if not isinstance(d, dict):
        return None
    events = []
    for e in _lst(d.get("key_events")):
        if isinstance(e, dict):
            events.append({"turn": _int(e.get("turn", e.get("step"))), "description": _s(e.get("description"), 200),
                           "outcome": _s(e.get("outcome"), 160)})
        elif str(e).strip():
            events.append({"turn": None, "description": _s(e, 200), "outcome": ""})
    facts = []
    for f in _lst(d.get("verified_facts")):
        if isinstance(f, dict) and f.get("fact"):
            facts.append({"fact": _s(f.get("fact"), 200), "turn": _int(f.get("turn")),
                          "confidence": _s(f.get("confidence"), 10) or "medium"})
        elif isinstance(f, str) and f.strip():
            facts.append({"fact": _s(f, 200), "turn": None, "confidence": "medium"})
    out = {"task_description": _s(d.get("task_description"), 300),
           "key_events": [e for e in events if e["description"]][-CAPS["key_events"]:],
           "verified_facts": facts[-CAPS["verified_facts"]:],
           "current_progress": _s(d.get("current_progress"), 300)}
    return out if (out["key_events"] or out["verified_facts"] or out["current_progress"]) else None


def norm_working(d) -> Optional[dict]:
    if not isinstance(d, dict):
        return None
    hyps = []
    for h in _lst(d.get("active_hypotheses")):
        if isinstance(h, dict) and h.get("hypothesis"):
            hyps.append({"hypothesis": _s(h["hypothesis"], 200), "supporting_evidence": _s(h.get("supporting_evidence"), 160),
                         "gaps": _s(h.get("gaps"), 120)})
        elif isinstance(h, str) and h.strip():
            hyps.append({"hypothesis": _s(h, 200), "supporting_evidence": "", "gaps": ""})
    acts = []
    for a in _lst(d.get("next_actions")):
        if isinstance(a, dict) and a.get("description"):
            acts.append({"type": _s(a.get("type"), 20) or "action", "description": _s(a["description"], 200),
                         "rationale": _s(a.get("rationale"), 160)})
        elif isinstance(a, str) and a.strip():
            acts.append({"type": "action", "description": _s(a, 200), "rationale": ""})
    out = {"immediate_goal": _s(d.get("immediate_goal"), 240),
           "current_challenges": _s(d.get("current_challenges") if not isinstance(d.get("current_challenges"), list)
                                    else "; ".join(map(str, d["current_challenges"])), 240),
           "active_hypotheses": hyps[:CAPS["active_hypotheses"]],
           "failed_approaches": [_s(x, 160) for x in _lst(d.get("failed_approaches")) if str(x).strip()][:CAPS["failed_approaches"]],
           "next_actions": acts[:CAPS["next_actions"]]}
    return out if (out["immediate_goal"] or out["active_hypotheses"] or out["next_actions"]) else None


def norm_tool(d) -> Optional[dict]:
    if not isinstance(d, dict):
        return None
    out = {k: [_s(x, 200) for x in _lst(d.get(k)) if str(x).strip()][:CAPS[k]]
           for k in ("effective_strategies", "dead_ends", "derived_rules", "useful_sources")}
    return out if any(out.values()) else None


def merge_tool_memories(old: Optional[dict], new: Optional[dict]) -> Optional[dict]:
    """Reference ``merge_tool_memories``: concatenate, dedupe (latest kept), cap."""
    if not old:
        return new
    if not new:
        return old
    merged = {}
    for k in ("effective_strategies", "dead_ends", "derived_rules", "useful_sources"):
        seen, out = set(), []
        for item in reversed((old.get(k) or []) + (new.get(k) or [])):
            key = item.lower()
            if key not in seen:
                seen.add(key)
                out.append(item)
        merged[k] = list(reversed(out))[-CAPS[k]:]
    return merged


# --- rendering (asymmetric injection) -----------------------------------------------
def render_tool_memory(store: HyMemStore) -> str:
    tm = store.tool or {}
    lines = []
    for title, key in (("Tips", "derived_rules"), ("Worked well", "effective_strategies"),
                       ("Dead ends (avoid)", "dead_ends"), ("Trust", "useful_sources")):
        if tm.get(key):
            lines.append(f"{title}: " + "; ".join(tm[key][:5]))
    return ("Tool experience:\n" + "\n".join(f"- {x}" for x in lines)) if lines else ""


def render_memory(store: HyMemStore, *, include_summary: bool = True) -> str:
    """Planner / isolated-reasoning view: summary + episode + working + tool + iso insights."""
    parts = []
    if include_summary and store.summary:
        parts.append("### Battle state summary\n" + clip(store.summary, 1400))
    ep = store.episode or {}
    if ep:
        lines = []
        if ep.get("verified_facts"):
            lines += [f"- {f['fact']}" + (f" (T{f['turn']})" if f.get("turn") else "") for f in ep["verified_facts"]]
        if ep.get("key_events"):
            lines.append("Milestones:")
            lines += [f"- {'T' + str(e['turn']) + ': ' if e.get('turn') else ''}{e['description']}"
                      + (f" -> {e['outcome']}" if e.get("outcome") else "") for e in ep["key_events"]]
        if ep.get("current_progress"):
            lines.append(f"Progress: {ep['current_progress']}")
        parts.append("### Key findings\n" + "\n".join(lines))
    wm = store.working or {}
    if wm:
        lines = []
        if wm.get("immediate_goal"):
            lines.append(f"Goal: {wm['immediate_goal']}")
        if wm.get("current_challenges"):
            lines.append(f"Challenges: {wm['current_challenges']}")
        for h in wm.get("active_hypotheses", []):
            lines.append(f"Hypothesis: {h['hypothesis']}" + (f" (evidence: {h['supporting_evidence']})"
                                                           if h.get("supporting_evidence") else ""))
        if wm.get("failed_approaches"):
            lines.append("Failed approaches (do NOT repeat): " + "; ".join(wm["failed_approaches"]))
        for a in wm.get("next_actions", []):
            lines.append(f"Planned [{a['type']}]: {a['description']}" + (f" ({a['rationale']})" if a.get("rationale") else ""))
        parts.append("### Current focus\n" + "\n".join(lines))
    tm = render_tool_memory(store)
    if tm:
        parts.append("### " + tm)
    folded_iso = [x for x in store.iso_log if x.get("success") and x.get("window", 0) < store.folds][-3:]
    if folded_iso:
        parts.append("### Earlier isolated-reasoning conclusions\n" + "\n".join(
            f"- [{x['description']}] {clip(x['conclusion'], 200)}" for x in folded_iso))
    return "\n\n".join(parts)


def tool_stats(store: HyMemStore) -> str:
    """Reference ``aggregate_tool_call_stats``: compact per-tool success/failure counts."""
    if not store.tool_calls:
        return "No tool calls recorded."
    agg: dict = {}
    for c in store.tool_calls:
        a = agg.setdefault(c["tool"], {"n": 0, "ok": 0, "good": [], "bad": []})
        a["n"] += 1
        a["ok"] += int(bool(c.get("ok")))
        (a["good"] if c.get("ok") else a["bad"]).append(json.dumps(c.get("args", {}), ensure_ascii=False)[:120])
    lines = []
    for name, a in agg.items():
        lines.append(f"Tool {name}: {a['n']} calls, {a['ok']} ok, {a['n'] - a['ok']} failed")
        if a["good"]:
            lines.append(f"  effective args: {a['good'][-3:]}")
        if a["bad"]:
            lines.append(f"  failed args: {a['bad'][-2:]}")
    return "\n".join(lines)


def rule_episode(store: HyMemStore, previous: Optional[dict]) -> Optional[dict]:
    """Fallback: previous episode memory + verified facts from the battle log."""
    ep = dict(previous or {"task_description": "Win the battle.", "key_events": [], "verified_facts": [],
                           "current_progress": ""})
    have = {f["fact"].lower() for f in ep.get("verified_facts", [])}
    facts = list(ep.get("verified_facts", []))
    events = list(ep.get("key_events", []))
    for kf in store.key_facts:
        if kf.text.lower() in have:
            continue
        have.add(kf.text.lower())
        if kf.kind in ("faint", "tera"):
            events.append({"turn": kf.turn, "description": kf.text, "outcome": ""})
        facts.append({"fact": kf.text, "turn": kf.turn, "confidence": "high"})
    ep["verified_facts"] = facts[-CAPS["verified_facts"]:]
    ep["key_events"] = events[-CAPS["key_events"]:]
    return norm_episode(ep) or previous


# --- the Fold operator -------------------------------------------------------------
@dataclass
class FoldOutcome:
    reason: str
    mode: str
    summary_ok: bool
    extracted: dict           # memory -> True (LLM) / False (fallback)
    seconds: float


class MemoryManager:
    """Phi. ``call(space, messages) -> LLMResponse`` (passed per fold) is the agent's
    space-aware LLM call bound to the battle being folded."""

    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    async def _extract(call, prompt: str) -> Optional[dict]:
        try:
            resp = await call("memory", [{"role": "system", "content": P.MEMORY_SYSTEM},
                                         {"role": "user", "content": prompt}])
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            return None
        return parse_json(resp.text)

    async def fold(self, store: HyMemStore, *, call, reason: str, task: str,
                   summary_messages: Optional[list], history_text: str) -> FoldOutcome:
        t0 = time.monotonic()
        summary_ok = False
        if self.cfg.fold_mode in ("typed", "typed_light"):
            # typed fold: no generative extraction. The memory is the planner's handover summary
            # ("typed" only) + key facts by rule + the executor / isolated-reasoning returns; a
            # System-One memory router classifies, checks and places those items per context
            if self.cfg.fold_mode == "typed" and summary_messages:
                try:
                    resp = await call("fold_summary", summary_messages)
                    summary = extract_summary(resp.text) or resp.text.strip()
                    if summary:
                        store.summary, summary_ok = clip(summary, 2000), True
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    pass
            store.episode = rule_episode(store, store.episode)
            store.folds += 1
            return FoldOutcome(reason, self.cfg.fold_mode, summary_ok,
                               {"episode": False, "working": False, "tool": False}, time.monotonic() - t0)
        if self.cfg.fold_mode == "two_stage" and summary_messages:
            try:
                resp = await call("fold_summary", summary_messages)
                summary = extract_summary(resp.text) or resp.text.strip()
                summary_ok = bool(summary)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                summary = ""
            summary = summary or (store.summary + "\n" + history_text).strip()
            iso = [f"- [{x['description']}] {clip(x['conclusion'], 200)}" for x in store.iso_log[-5:] if x.get("success")]
            source = f"Planner's state summary:\n{clip(summary, 3000)}" + (
                "\n\nIsolated reasoning findings:\n" + "\n".join(iso) if iso else "")
            store.summary = clip(summary, 2000)
        else:
            execs = [f"- {x['task']} -> {x['status']}: {clip(x['findings'], 160)}" for x in store.exec_log[-8:]]
            isos = [f"- [{x['description']}] {clip(x['conclusion'], 200)}" for x in store.iso_log[-5:]]
            source = (f"Previous memory:\n{render_memory(store, include_summary=False) or '(none)'}\n\n"
                      f"Planner history since the last update:\n{clip(history_text, 5000)}\n\n"
                      f"Executor results:\n{chr(10).join(execs) or '(none)'}\n\n"
                      f"Isolated reasoning:\n{chr(10).join(isos) or '(none)'}")
        ep, wm, tm = await asyncio.gather(
            self._extract(call, P.MEMORY_EPISODE.format(task=task, source=source)),
            self._extract(call, P.MEMORY_WORKING.format(task=task, source=source)),
            self._extract(call, P.MEMORY_TOOL.format(task=task, source=source, tool_stats=tool_stats(store))),
        )
        ep, wm, tm = norm_episode(ep), norm_working(wm), norm_tool(tm)
        extracted = {"episode": ep is not None, "working": wm is not None, "tool": tm is not None}
        store.episode = rule_episode(store, ep or store.episode)
        if wm is not None:
            store.working = wm
        store.tool = merge_tool_memories(store.tool, tm)
        store.folds += 1
        return FoldOutcome(reason, self.cfg.fold_mode, summary_ok, extracted, time.monotonic() - t0)
