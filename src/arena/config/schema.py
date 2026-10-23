"""YAML-driven experiment configuration (pydantic)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ExternalConfig(BaseModel):
    """How to launch an out-of-process opponent (PokeChamp / Metamon).

    The external bot runs in its OWN venv/container, connects to the same local
    Showdown server as ``username``, and accepts challenges from our player. We
    never import its code; the orchestrator only spawns ``launch_cmd`` (or
    ``docker_image``) and watches its stdout for ``ready_regex``.
    """

    launch_cmd: list[str] = Field(default_factory=list)  # e.g. ["bash", "run.sh"]
    cwd: str = "."                         # working dir (e.g. external/pokechamp)
    username: str = ""                     # fixed Showdown login name it uses
    env: dict[str, str] = Field(default_factory=dict)    # extra env (CHECKPOINT, ...)
    docker_image: Optional[str] = None     # if set, run via `docker run` instead
    ready_regex: str = "logged in|listening|ready"       # stdout line = "connected"
    ready_timeout_s: float = 180.0
    stall_timeout_s: float = 420.0         # abort if no battle finishes in this window
    healthy_via: str = "stdout"            # "stdout" | "logfile"


class AgentConfig(BaseModel):
    name: str                              # display id in the matrix / reports
    type: str                              # random | max_base_power | simple_heuristics
                                           # | llm_io | llm_cot | llm_self_consistency | llm_react
                                           # | llm_pokellmon | llm_hymem | llm_flat_react | llm_summary
                                           # | (external) pokechamp | metamon
    kind: str = "internal"                 # "internal" (built in-process) | "external"
    model: Optional[str] = None            # for llm_* types
    params: dict = Field(default_factory=dict)   # n_samples, max_steps, temperature, ...
    external: Optional[ExternalConfig] = None    # required when kind == "external"


class TeamConfig(BaseModel):
    mode: str = "mirror"                   # mirror | pool_random


class EvalConfig(BaseModel):
    mode: str = "round_robin"              # round_robin | anchored
    n_battles: int = 10
    anchors: list[str] = Field(default_factory=list)   # agent names used as fixed baselines
    save_replays: bool = False             # save a watchable .html replay per battle


class GuardrailConfig(BaseModel):
    max_cost_usd: Optional[float] = None
    max_tokens: Optional[int] = None
    max_battles: Optional[int] = None
    dry_run: bool = False


class ExperimentConfig(BaseModel):
    name: str = "experiment"
    format: str = "gen9ou"
    seed: int = 0
    full_info: bool = False
    agents: list[AgentConfig]
    teams: TeamConfig = Field(default_factory=TeamConfig)
    augmentations: dict[str, str] = Field(default_factory=dict)  # name -> off|passive|tool
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)

    @classmethod
    def load(cls, path: Path | str) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        cfg = cls(**data)
        cfg.validate_externals()
        return cfg

    def validate_externals(self) -> None:
        """Externals must carry launch info; a pairing can't be external-vs-external
        (each would need the other to be an in-process Player to battle)."""
        for c in self.agents:
            if c.kind == "external":
                if c.external is None or not (c.external.launch_cmd or c.external.docker_image):
                    raise ValueError(f"external agent {c.name!r} needs external.launch_cmd or docker_image")
                if not c.external.username:
                    raise ValueError(f"external agent {c.name!r} needs external.username")
        ext = {c.name for c in self.agents if c.kind == "external"}
        if len(ext) >= 2:
            # at least one side of every pairing must be in-process; reject if a
            # pairing of two externals is schedulable
            internals = [c.name for c in self.agents if c.kind != "external"]
            if not internals and len(ext) >= 2:
                raise ValueError("need at least one internal agent: external-vs-external is unsupported")
