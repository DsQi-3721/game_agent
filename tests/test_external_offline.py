"""Offline: external-opponent plumbing -- A/B orientation (highest-risk),
config validation, registry routing. No server/process launched."""

import pytest

from arena.config.schema import AgentConfig, ExperimentConfig, ExternalConfig
from arena.eval.runner import ExternalAgentSpec, _orient_external
from arena.registry.agents import build_spec, is_llm_type


def test_orient_external_when_external_is_b():
    # our agent = a, external = b. We won 7, lost 2, 1 tie of 10.
    res = _orient_external("gen9ou", "round_robin", "ours", "ext", external_is_a=False,
                           our_wins=7, our_losses=2, n_finished=10, turns=[20] * 10)
    assert res.a == "ours" and res.b == "ext"
    assert res.a_wins == 7 and res.b_wins == 2 and res.ties == 1
    assert res.n_battles == 10


def test_orient_external_when_external_is_a():
    # external = a, our agent = b. Same battle data -> wins must flip slots.
    res = _orient_external("gen9ou", "round_robin", "ext", "ours", external_is_a=True,
                           our_wins=7, our_losses=2, n_finished=10, turns=[20] * 10)
    assert res.a == "ext" and res.b == "ours"
    assert res.a_wins == 2 and res.b_wins == 7 and res.ties == 1


def test_orient_partial_results_denominator():
    # only 4 of 10 battles finished before the external died
    res = _orient_external("gen9ou", "round_robin", "ours", "ext", external_is_a=False,
                           our_wins=3, our_losses=1, n_finished=4, turns=[15] * 4)
    assert res.n_battles == 4 and res.a_wins == 3 and res.b_wins == 1 and res.ties == 0


def test_build_spec_routes_external():
    cfg = AgentConfig(name="metamon", type="metamon", kind="external",
                      external=ExternalConfig(launch_cmd=["bash", "run.sh"], username="MetamonBot"))
    spec = build_spec(cfg)
    assert isinstance(spec, ExternalAgentSpec)
    assert spec.username == "MetamonBot"
    assert is_llm_type("metamon") is False


def test_config_rejects_external_without_launch():
    cfg = ExperimentConfig(
        name="x", format="gen9ou",
        agents=[AgentConfig(name="ours", type="random"),
                AgentConfig(name="ext", type="pokechamp", kind="external",
                            external=ExternalConfig(username="Bot"))],  # no launch_cmd
    )
    with pytest.raises(ValueError, match="launch_cmd"):
        cfg.validate_externals()


def test_config_rejects_external_without_username():
    cfg = ExperimentConfig(
        name="x", format="gen9ou",
        agents=[AgentConfig(name="ours", type="random"),
                AgentConfig(name="ext", type="pokechamp", kind="external",
                            external=ExternalConfig(launch_cmd=["bash", "run.sh"]))],
    )
    with pytest.raises(ValueError, match="username"):
        cfg.validate_externals()


def test_config_rejects_external_vs_external_only():
    cfg = ExperimentConfig(
        name="x", format="gen9ou",
        agents=[AgentConfig(name="e1", type="pokechamp", kind="external",
                            external=ExternalConfig(launch_cmd=["x"], username="b1")),
                AgentConfig(name="e2", type="metamon", kind="external",
                            external=ExternalConfig(launch_cmd=["x"], username="b2"))],
    )
    with pytest.raises(ValueError, match="external-vs-external"):
        cfg.validate_externals()


def test_opponent_actions_extraction():
    from arena.agents.adapter import opponent_actions
    # we are p1; opponent is p2. Mix of our + their lines.
    msgs = [
        [">battle-gen9ou-1"],
        ["", "move", "p1a: Garchomp", "Earthquake", "p2a: Pikachu"],   # ours -> ignored
        ["", "move", "p2a: Pikachu", "Thunderbolt", "p1a: Garchomp"],  # theirs
        ["", "switch", "p2a: Gengar", "Gengar, L50", "100/100"],       # theirs
        ["", "-damage", "p1a: Garchomp", "50/100"],                    # not an action
    ]
    acts = opponent_actions(msgs, "p2")
    assert acts == ["move Thunderbolt", "switch Gengar, L50"]


def test_opponent_actions_no_role():
    from arena.agents.adapter import opponent_actions
    assert opponent_actions([["", "move", "p2a: X", "Tackle"]], "") == []


def test_config_allows_external_with_internal():
    cfg = ExperimentConfig(
        name="x", format="gen9ou",
        agents=[AgentConfig(name="ours", type="random"),
                AgentConfig(name="ext", type="metamon", kind="external",
                            external=ExternalConfig(launch_cmd=["bash", "run.sh"], username="Bot"))],
    )
    cfg.validate_externals()  # must not raise
