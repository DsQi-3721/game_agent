"""Memory routing with a System One model: classify, check and place memory items.

HyMem's three memories are written by free-form LLM extraction at fold time and injected by
fixed rules (planner / isolated reasoning see everything, the executor sees the tool memory).
With a small backbone the extractions mix durable facts with stale HP numbers, guesses filed
as "verified", generic advice and spurious tool rules, and every item reaches every context.

The router asks three typed questions per item, all in one request, against the CURRENT
battle state (the same state the fast decision reads, so they can share one request):

  kind_i    Choice  opponent_set | battle_event | plan | hypothesis | lesson | own_team | tool_tip |
                    calc_result | current_numbers | generic
  valid_i   Noul    still true now (not contradicted or outdated by the state)
  useful_i  Noul    helps choose actions in the coming turns, beyond what the state shows

Code then places each kept item by kind (``DESTINATIONS``), drops invalid / number-only /
generic items, ranks by P(useful) and caps every section. Key facts extracted by rule from the
battle log are trusted: they skip ``kind``/``valid`` and are only ranked.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from arena.agents.hymem import prompts as P
from arena.agents.hymem.memory import HyMemStore
from arena.agents.hymem.messages import clip
from arena.llm import typesafe as ts

KINDS = {
    "opponent_set": "A fact about an opponent Pokemon's moves, item, ability, tera type, stats or speed.",
    "battle_event": "Something that already happened in this battle: a knock-out, a switch, a status, a tera, "
                    "a stat change, a hazard being set.",
    "plan": "A goal, an intended next action or a strategy for the coming turns.",
    "hypothesis": "An unconfirmed guess or prediction about the opponent's set or what it will do.",
    "lesson": "What worked or failed in this battle, for example a move that turned out to be resisted.",
    "own_team": "A fact about your own Pokemon's moves, item, ability or stats (your team sheet already has it).",
    "tool_tip": "Advice about how to use the analysis tools (damage_calc, type_matchup, switch_eval, ...).",
    "calc_result": "A calculator result such as damage %, hits to KO or switch-in safety, which the calculator "
                   "can recompute at any time.",
    "current_numbers": "Mainly current HP percentages or other numbers that change every turn.",
    "generic": "General advice that would be true in any battle and says nothing specific about this one.",
}
# where each kind of kept item goes ("fast" = known_facts in the System-1 state)
DESTINATIONS = {
    "opponent_set": ("planner", "iso", "executor", "fast"),
    "battle_event": ("planner", "iso"),
    "plan": ("planner", "iso", "fast"),
    "hypothesis": ("planner", "iso"),
    "lesson": ("planner", "iso", "fast"),
    "own_team": (),
    "tool_tip": ("executor",),
    "calc_result": (),
    "current_numbers": (),
    "generic": (),
}
SECTIONS = [("opponent_set", "Opponent info"), ("lesson", "Lessons from this battle"),
            ("battle_event", "Events so far"), ("plan", "Plans"), ("hypothesis", "Unconfirmed guesses"),
            ("tool_tip", "Tool tips")]
KEY_FACT_KIND = {"opp_move": "opponent_set", "opp_item": "opponent_set", "opp_ability": "opponent_set",
                 "tera": "battle_event", "faint": "battle_event", "hazard": "battle_event",
                 "status": "battle_event", "item_lost": "battle_event", "hazard_removed": "battle_event"}

KIND_Q = "What kind of note is `note`? It was written earlier in this Pokemon battle by `you`'s assistant."
VALID_Q = ("Is `note` still true now, given the current battle in `you`, `opponent` and `field`? Answer no if "
           "the current battle contradicts it or it is out of date, for example the Pokemon, HP, status or "
           "field condition it describes has changed since.")
USEFUL_Q = ("Would `note` help `you` choose actions in the coming turns of this battle, beyond what `you`, "
            "`opponent` and `field` already show?")


@dataclass
class MemoryItem:
    id: str
    text: str
    source: str                 # episode.fact | working.hypothesis | tool.dead_ends | exec | key_fact | ...
    turn: Optional[int] = None
    trusted: bool = False       # rule-extracted from the battle log
    kind_hint: str = ""         # kind for trusted items


@dataclass
class RoutedItem:
    item: MemoryItem
    kind: str
    p_kind: float
    p_valid: Optional[float]
    p_useful: Optional[float]
    dest: tuple = ()
    dropped: str = ""

    @property
    def kept(self) -> bool:
        return not self.dropped


@dataclass
class RouterConfig:
    model: str = "jev-1.13.0"
    # measured on the memory bench (runs/jev_eval/memory): contradicted / other-battle facts score
    # P(valid) <= 0.04 while true facts the state does not mention spread down to ~0.1, so the
    # validity gate sits just above the false ones; 0.5 dropped half of the true facts
    min_valid: float = 0.1
    min_useful: float = 0.3
    # "still true now?" is a check of a factual claim against the state: plans, hypotheses and lessons are
    # neither true nor false, so they are kept or dropped by usefulness only
    valid_kinds: tuple = ("opponent_set", "battle_event")
    caps: dict = field(default_factory=lambda: {"opponent_set": 8, "lesson": 4, "battle_event": 6, "plan": 3,
                                                "hypothesis": 3, "tool_tip": 5})
    fast_cap: int = 6
    cache_kinds: bool = True       # an item's kind never changes: ask it once, re-check validity every time


@dataclass
class RoutedMemory:
    items: list
    latency_s: float = 0.0
    input_tokens: int = 0
    version: tuple = ()
    cfg: RouterConfig = field(default_factory=RouterConfig)

    def for_space(self, space: str) -> list[RoutedItem]:
        out = []
        for kind, _title in SECTIONS:
            pool = [r for r in self.items if r.kept and r.kind == kind and space in r.dest]
            pool.sort(key=lambda r: -(r.p_useful if r.p_useful is not None else 1.0))
            out += pool[: self.cfg.caps.get(kind, 4)]
        return out

    def render(self, space: str) -> str:
        chosen = self.for_space(space)
        parts = []
        for kind, title in SECTIONS:
            lines = [f"- {r.item.text}" + (f" (T{r.item.turn})" if r.item.turn else "")
                     for r in chosen if r.kind == kind]
            if lines:
                parts.append(f"### {title}\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def fast_facts(self) -> list[str]:
        pool = [r for r in self.items if r.kept and "fast" in r.dest]
        pool.sort(key=lambda r: -(r.p_useful if r.p_useful is not None else 1.0))
        return [r.item.text for r in pool[: self.cfg.fast_cap]]

    def stats(self) -> dict:
        kept = [r for r in self.items if r.kept]
        by_kind: dict = {}
        for r in self.items:
            by_kind[r.kind] = by_kind.get(r.kind, 0) + 1
        drops: dict = {}
        for r in self.items:
            if r.dropped:
                drops[r.dropped] = drops.get(r.dropped, 0) + 1
        return {"mem_items": len(self.items), "mem_kept": len(kept), "mem_by_kind": by_kind,
                "mem_dropped": drops, "mem_latency_s": round(self.latency_s, 4),
                "mem_input_tokens": self.input_tokens}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", text.lower()).strip()


# per-source caps (most recent kept): every memory slot stays represented
SOURCE_CAPS = {"key_fact": 16, "episode.fact": 10, "episode.event": 6, "episode.progress": 1, "working.goal": 1,
               "working.challenge": 1, "working.hypothesis": 4, "working.failed": 4, "working.next": 3,
               "tool": 10, "summary": 6, "iso": 3, "exec": 6}
_HEADINGS = [_norm(re.sub(r"^##\s*\d+\.\s*", "", h)) for h in P.FOLD_SUMMARY_REQUEST.splitlines()
             if h.startswith("## ")]


def _is_heading(text: str) -> bool:
    t = _norm(text)
    return any(t[:24] == h[:24] for h in _HEADINGS if h)


def collect_items(store: HyMemStore) -> list[MemoryItem]:
    """Every memory item HyMem would show any context, plus rule key facts, deduplicated."""
    raw: list[tuple[str, str, Optional[int], bool, str]] = []
    for kf in store.key_facts:
        if not getattr(kf, "current", True):     # a state that has since ended (item lost, hazard removed ...)
            continue
        raw.append((kf.text, "key_fact", kf.turn, True, KEY_FACT_KIND.get(kf.kind, "battle_event")))
    ep = store.episode or {}
    for f in ep.get("verified_facts", []):
        raw.append((f["fact"], "episode.fact", f.get("turn"), False, ""))
    for e in ep.get("key_events", []):
        raw.append((e["description"] + (f" -> {e['outcome']}" if e.get("outcome") else ""), "episode.event",
                    e.get("turn"), False, ""))
    if ep.get("current_progress"):
        raw.append((ep["current_progress"], "episode.progress", None, False, ""))
    wm = store.working or {}
    if wm.get("immediate_goal"):
        raw.append((wm["immediate_goal"], "working.goal", None, False, ""))
    if wm.get("current_challenges"):
        raw.append((wm["current_challenges"], "working.challenge", None, False, ""))
    for h in wm.get("active_hypotheses", []):
        raw.append((h["hypothesis"], "working.hypothesis", None, False, ""))
    for x in wm.get("failed_approaches", []):
        raw.append((x, "working.failed", None, False, ""))
    for a in wm.get("next_actions", []):
        raw.append((a["description"], "working.next", None, False, ""))
    for k in ("derived_rules", "effective_strategies", "dead_ends", "useful_sources"):
        for x in (store.tool or {}).get(k, []):
            raw.append((x, f"tool.{k}", None, False, ""))
    for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", store.summary or ""):
        sent = sent.strip(" -*#")
        if len(sent) > 12 and not _is_heading(sent):
            raw.append((clip(sent, 240), "summary", None, False, ""))
    for x in store.iso_log:
        if x.get("success") and x.get("conclusion"):
            raw.append((clip(x["conclusion"], 240), "iso", None, False, ""))
    for x in store.exec_log[-4:]:
        for line in (x.get("findings") or "").splitlines():
            line = line.strip(" -*")
            if len(line) > 12:
                raw.append((clip(line, 240), "exec", None, False, ""))
    seen, groups = set(), {}
    for text, source, turn, trusted, kind in raw:
        key = _norm(text)
        if not key or key in seen:
            continue
        seen.add(key)
        groups.setdefault(source.split(".")[0] if source.startswith("tool.") else source, []).append(
            (text.strip(), source, turn, trusted, kind))
    out: list[MemoryItem] = []
    for group, entries in groups.items():
        for text, source, turn, trusted, kind in entries[-SOURCE_CAPS.get(group, 4):]:
            out.append(MemoryItem(f"m{len(out)}", text, source, turn, trusted, kind))
    return out


class MemoryRouter:
    def __init__(self, client, cfg: Optional[RouterConfig] = None):
        self.client = client
        self.cfg = cfg or RouterConfig()
        self._kinds: dict[str, tuple[str, float]] = {}     # normalised text -> (kind, probability)

    def questions(self, items: list[MemoryItem]) -> dict:
        qs: dict = {}
        for m in items:
            note = {"note": m.text}
            if not m.trusted:
                if not (self.cfg.cache_kinds and _norm(m.text) in self._kinds):
                    qs[f"kind_{m.id}"] = ts.choice({"question": KIND_Q, **note}, KINDS)
                qs[f"valid_{m.id}"] = ts.noul({"question": VALID_Q, **note})
            qs[f"useful_{m.id}"] = ts.noul({"question": USEFUL_Q, **note})
        return qs

    def place(self, item: MemoryItem, kind: str, p_valid: Optional[float],
              p_useful: Optional[float]) -> tuple[tuple, str]:
        c = self.cfg
        dest = DESTINATIONS.get(kind, ())
        if not dest:
            return (), kind
        if p_valid is not None and p_valid < c.min_valid and kind in c.valid_kinds:
            return (), "invalid"
        if p_useful is not None and p_useful < c.min_useful and kind not in ("opponent_set", "battle_event"):
            return (), "not_useful"
        return dest, ""

    def parse(self, items: list[MemoryItem], answers: dict) -> list[RoutedItem]:
        out = []
        for m in items:
            useful = answers.get(f"useful_{m.id}")
            p_useful = useful.noul if useful is not None else None
            if m.trusted:
                kind, p_kind, p_valid = m.kind_hint or "battle_event", 1.0, None
            else:
                ka, va = answers.get(f"kind_{m.id}"), answers.get(f"valid_{m.id}")
                if ka is not None and ka.probabilities:
                    kind, p_kind = max(ka.probabilities.items(), key=lambda kv: kv[1])
                    if self.cfg.cache_kinds and len(self._kinds) < 20000:
                        self._kinds[_norm(m.text)] = (kind, p_kind)
                elif _norm(m.text) in self._kinds:
                    kind, p_kind = self._kinds[_norm(m.text)]
                else:
                    continue
                p_valid = va.noul if va is not None else None
            dest, dropped = self.place(m, kind, p_valid, p_useful)
            out.append(RoutedItem(m, kind, p_kind, p_valid, p_useful, dest, dropped))
        return out

    async def route(self, state, items: list[MemoryItem], version: tuple = ()) -> RoutedMemory:
        if not items:
            return RoutedMemory([], version=version, cfg=self.cfg)
        t0 = time.perf_counter()
        res = await self.client.system_one(state, self.questions(items), model=self.cfg.model)
        return RoutedMemory(self.parse(items, res.answers), latency_s=res.latency_s or time.perf_counter() - t0,
                            input_tokens=res.input_tokens, version=version, cfg=self.cfg)


def memory_version(store: HyMemStore) -> tuple:
    """Changes whenever anything collect_items reads changes."""
    return (store.folds, len(store.key_facts), len(store.exec_log), len(store.iso_log))
