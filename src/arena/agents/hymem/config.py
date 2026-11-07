"""HyMem configuration: budgets, fold triggers, method switches, per-space sampling.

Defaults are sized for a 4B model on a laptop (Qwen3.5-4B via mlx_lm). The paper /
reference values are noted where they differ; ``configs/experiments/*gpu*.yaml``
restore them for full-scale runs.

Flat agent params (YAML ``params:`` or the TUI form) map onto fields by name;
per-space knobs use ``<space>_<field>`` (e.g. ``iso_thinking: true``,
``planner_max_tokens: 400``). ``config_path`` points to a YAML with the same keys.
Unknown keys raise, so typos never silently fall back to defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Optional

import yaml

from arena.llm.spaces import SpaceConfig

ABLATIONS = ("full", "no_iso", "no_mm", "no_typed_isolation")
SPACES = ("planner", "executor", "iso", "fold_summary", "memory", "router", "distill")
# spaces whose calls go to the auxiliary model when one is configured (paper: aux LLM)
AUX_SPACES = ("memory", "router", "distill")

DEFAULT_TOOLS = ("matchup", "damage_calc", "speed_check", "opponent_sets", "switch_eval",
                 "type_matchup", "pokedex", "battle_log")


def _default_spaces() -> dict[str, SpaceConfig]:
    return {
        "planner": SpaceConfig(max_tokens=320),
        "executor": SpaceConfig(max_tokens=256),
        "iso": SpaceConfig(max_tokens=512),
        "fold_summary": SpaceConfig(max_tokens=768, temperature=0.5),
        "memory": SpaceConfig(max_tokens=512, temperature=0.3),   # JSON: low temp, no presence penalty
        "router": SpaceConfig(max_tokens=96, temperature=0.0, top_p=None),
        "distill": SpaceConfig(max_tokens=256, temperature=0.3),
    }


@dataclass
class HyMemConfig:
    # --- budgets ---------------------------------------------------------------
    planner_steps: int = 4              # planner turns per decision; the last is answer-only
    execute_turns: int = 3              # executor tool-loop turns (paper: 10)
    max_iso_per_window: int = 5         # isolated-reasoning sessions per fold window (paper: 5)
    max_iso_per_decision: int = 1
    max_iso_turns: int = 3              # turns per isolated session (paper: 3)
    tau_k: int = 4                      # recent planner entries inherited by an iso session
    max_parallel: int = 3               # tasks/isoReasons dispatched concurrently per turn
    max_tool_calls_per_turn: int = 3
    decision_budget_s: float = 900.0    # wall clock per decision; then answer-only
    # --- fold trigger (paper Eq. 17 + reference constraints) --------------------
    ctx_ratio: float = 0.99             # planner tokens / context window
    n_exec: int = 5                     # executor invocations since the last fold
    n_fail: int = 3                     # consecutive executor failures
    n_turn: int = 8                     # planner turns since the last fold
    n_tool: int = 10                    # tool calls since the last fold
    low_conf: float = 0.3               # "very low" answer confidence ...
    n_low_conf: int = 2                 # ... this many times in a row
    min_turns_between_folds: int = 5    # reference HyMemAgent builds its trigger with 5 (class default 3)
    max_folds: Optional[int] = None     # reference: 3-5 per GAIA task; a battle is far longer
    context_window: Optional[int] = None  # None -> model spec, else 32768
    recent_buffer: int = 4              # H^p entries kept across a fold (~2 exchanges)
    # --- method switches --------------------------------------------------------
    ablation: str = "full"              # full | no_iso | no_mm | no_typed_isolation
    keep_planner_analysis: bool = False
    fold_mode: str = "two_stage"        # two_stage (reference) | direct (paper Eq. 13) | typed | typed_light
    router: str = "rule"                # off | rule | llm
    distill: str = "rule"               # rule | llm   (phi on tool observations)
    distill_max_chars: int = 700
    replan: bool = False                # low-confidence answer -> verify once (reference replan)
    replan_threshold: float = 0.5
    exec_synthesis: str = "llm"         # llm | rule   (results block when the executor runs out)
    dossier: bool = False               # state view lists every opponent reveal (control)
    # Defaults since 2026-10-30 (user decision after the I/O audit): the corrected view and the lead prior. The
    # paper's own runs used state_view=paper and team_preview=random (configs/hymem/gpu_paper.yaml).
    state_view: str = "obs1"            # obs1 (adds the fields the I/O audit found missing; see views) | paper
    team_preview: str = "lead_prior"    # lead_prior (arena.agents.team_preview) | random (paper runs)
    opponent_model: str = ""            # "" (ARENA_OPPONENT_MODEL, default team) | team | prior (arena.knowledge.opponent_model)
    tools: Optional[tuple[str, ...]] = None   # executor tools; None -> experiment TOOL augments or defaults
    aux_model: Optional[str] = None     # memory/router/distill model; None -> same as the agent's
    log_calls: str = "full"             # full | meta | off
    spaces: dict = field(default_factory=_default_spaces)

    # ------------------------------------------------------------------------
    @property
    def iso_enabled(self) -> bool:
        return self.ablation != "no_iso"

    @property
    def memory_enabled(self) -> bool:
        return self.ablation != "no_mm"

    @property
    def typed_isolation(self) -> bool:
        return self.ablation != "no_typed_isolation"

    def space(self, name: str) -> SpaceConfig:
        return self.spaces[name]

    def validate(self) -> "HyMemConfig":
        if self.ablation not in ABLATIONS:
            raise ValueError(f"ablation must be one of {ABLATIONS}, got {self.ablation!r}")
        if self.fold_mode not in ("two_stage", "direct", "typed", "typed_light"):
            raise ValueError(f"fold_mode must be two_stage|direct|typed|typed_light, got {self.fold_mode!r}")
        if self.router not in ("off", "rule", "llm"):
            raise ValueError(f"router must be off|rule|llm, got {self.router!r}")
        if self.distill not in ("rule", "llm"):
            raise ValueError(f"distill must be rule|llm, got {self.distill!r}")
        if self.exec_synthesis not in ("llm", "rule"):
            raise ValueError(f"exec_synthesis must be llm|rule, got {self.exec_synthesis!r}")
        if self.state_view not in ("paper", "obs1"):
            raise ValueError(f"state_view must be paper|obs1, got {self.state_view!r}")
        if self.team_preview not in ("random", "lead_prior"):
            raise ValueError(f"team_preview must be random|lead_prior, got {self.team_preview!r}")
        if self.opponent_model not in ("", "team", "prior"):
            raise ValueError(f"opponent_model must be team|prior, got {self.opponent_model!r}")
        if self.planner_steps < 1 or self.execute_turns < 1 or self.max_iso_turns < 1:
            raise ValueError("planner_steps, execute_turns and max_iso_turns must be >= 1")
        return self

    @classmethod
    def from_params(cls, **params) -> "HyMemConfig":
        """Build from flat params (agent YAML / TUI). Empty strings mean "unset"."""
        params = {k: v for k, v in params.items() if v is not None and v != ""}
        path = params.pop("config_path", None)
        if path:
            p = Path(path)
            if not p.is_absolute() and not p.exists():       # project-relative (configs/hymem/...)
                p = Path(__file__).resolve().parents[4] / p
            loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            params = {**loaded, **params}
        cfg = cls()
        top = {f.name for f in fields(cls)} - {"spaces"}
        spaces = {k: replace(v) for k, v in cfg.spaces.items()}
        space_fields = {f.name for f in fields(SpaceConfig)}
        updates = {}
        for key, val in params.items():
            if key in top:
                if key == "tools" and isinstance(val, str):
                    val = tuple(t.strip() for t in val.split(",") if t.strip()) or None
                elif key == "tools" and val is not None:
                    val = tuple(val)
                updates[key] = val
                continue
            if key.startswith("think_") and key[6:] in spaces:      # think_iso: true
                spaces[key[6:]].thinking = bool(val)
                continue
            space, _, attr = key.partition("_")
            for sp in SPACES:                                       # e.g. fold_summary_max_tokens
                if key.startswith(sp + "_") and key[len(sp) + 1:] in space_fields:
                    space, attr = sp, key[len(sp) + 1:]
                    break
            if space in spaces and attr in space_fields:
                setattr(spaces[space], attr, val)
                continue
            raise ValueError(f"unknown HyMem param {key!r}")
        return replace(cfg, spaces=spaces, **updates).validate()
