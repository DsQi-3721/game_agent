"""Every shipped HyMem / long-context experiment config loads and builds its agents."""

from pathlib import Path

import pytest

from arena.agents.base import AgentContext
from arena.agents.hymem.config import HyMemConfig
from arena.config.schema import ExperimentConfig
from arena.llm.client import MockLLMClient
from arena.registry.agents import build_spec

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs" / "experiments").glob("hymem_*.yaml")) + \
    [ROOT / "configs" / "experiments" / "longctx_smoke.yaml"]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_experiment_config_builds(path):
    cfg = ExperimentConfig.load(path)
    ctx = AgentContext(llm=MockLLMClient(lambda m, t, i: "<answer>1</answer>"))
    for agent_cfg in cfg.agents:
        agent = build_spec(agent_cfg).build(ctx)
        assert agent is not None


def test_param_files():
    mac = HyMemConfig.from_params(config_path="configs/hymem/mac.yaml")
    gpu = HyMemConfig.from_params(config_path=str(ROOT / "configs" / "hymem" / "gpu.yaml"), ablation="no_mm")
    assert mac.planner_steps == 3 and mac.space("iso").thinking is False
    assert gpu.execute_turns == 10 and gpu.space("iso").thinking and gpu.space("iso").temperature == 1.0
    assert gpu.ablation == "no_mm" and gpu.router == "llm" and gpu.replan
