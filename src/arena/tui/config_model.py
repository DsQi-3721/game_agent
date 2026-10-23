"""In-progress experiment draft -> ExperimentConfig (UI-agnostic, unit-testable)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from arena.config.schema import (
    AgentConfig, EvalConfig, ExperimentConfig, GuardrailConfig, TeamConfig,
)


@dataclass
class Draft:
    name: str = "tui_experiment"
    format: str = "gen9randombattle"
    seed: int = 0
    full_info: bool = False
    agents: list[AgentConfig] = field(default_factory=list)
    teams_mode: str = "mirror"
    augmentations: dict[str, str] = field(default_factory=dict)
    eval_mode: str = "round_robin"
    n_battles: int = 4
    anchors: list[str] = field(default_factory=list)
    max_cost_usd: Optional[float] = None
    save_replays: bool = False

    def to_config(self) -> ExperimentConfig:
        augs = {k: v for k, v in self.augmentations.items() if v and v != "off"}
        return ExperimentConfig(
            name=self.name, format=self.format, seed=self.seed, full_info=self.full_info,
            agents=self.agents,
            teams=TeamConfig(mode=self.teams_mode),
            augmentations=augs,
            evaluation=EvalConfig(mode=self.eval_mode, n_battles=self.n_battles,
                                  anchors=self.anchors, save_replays=self.save_replays),
            guardrails=GuardrailConfig(max_cost_usd=self.max_cost_usd),
        )

    def write_yaml(self, run_dir: Path | str) -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "config.yaml"
        path.write_text(yaml.safe_dump(self.to_config().model_dump(), sort_keys=False,
                                       allow_unicode=True), encoding="utf-8")
        return path

    def validate(self) -> Optional[str]:
        if not self.agents:
            return "Add at least one agent"
        if self.eval_mode == "anchored" and not self.anchors:
            return "Anchored mode needs at least one anchor agent"
        names = [a.name for a in self.agents]
        if len(names) != len(set(names)):
            return "Duplicate agent names"
        return None
