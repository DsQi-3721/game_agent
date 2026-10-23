"""M6 offline: metrics, guardrails, config load, agent registry, dry-run."""

from dataclasses import dataclass

import pytest

from arena.config.schema import ExperimentConfig
from arena.eval.guardrails import BudgetExceeded, Guardrails
from arena.eval.metrics import elo_from_results, wilson_interval

CONFIG = "configs/experiments/quick_baselines.yaml"


@dataclass
class _R:
    a: str
    b: str
    a_wins: int
    b_wins: int
    ties: int = 0


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(5, 10)
    assert 0 <= lo < 0.5 < hi <= 1
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_elo_orders_dominant_agent_higher():
    results = [_R("strong", "weak", 9, 1), _R("strong", "mid", 7, 3), _R("mid", "weak", 6, 4)]
    elo = elo_from_results(results)
    assert elo["strong"] > elo["mid"] > elo["weak"]


def test_guardrails_caps():
    g = Guardrails(max_battles=2)
    g.battle_done()
    g.check()  # 1 < 2 ok
    g.battle_done()
    with pytest.raises(BudgetExceeded):
        g.check()

    gc = Guardrails(max_cost_usd=0.01)
    gc.record(0.02, 1000)
    with pytest.raises(BudgetExceeded):
        gc.check()


def test_dry_run_estimate_positive():
    est = Guardrails.estimate(n_matchups=6, n_battles=3, avg_turns=30,
                              calls_per_turn=1.5, tokens_per_call=1500,
                              price_in=0.27, price_out=1.10)
    assert est["est_cost_usd"] > 0 and est["battles_total"] == 18


def test_config_loads_and_builds_specs():
    cfg = ExperimentConfig.load(CONFIG)
    assert cfg.format == "gen9randombattle"
    assert {a.name for a in cfg.agents} == {"random", "maxpow", "heuristic"}
    from arena.agents.registry import build_spec
    from arena.agents.base import AgentContext
    spec = build_spec(cfg.agents[0])
    agent = spec.build(AgentContext())
    assert agent is not None


def test_evaluator_pairings_round_robin():
    from arena.eval.tournament import Evaluator
    cfg = ExperimentConfig.load(CONFIG)
    ev = Evaluator(cfg, run_dir="runs/_test")
    pairs = ev._pairings()
    assert len(pairs) == 3  # C(3,2)
