"""Focused checks for the new temporal/reward contract against the real JVM."""

import json

import numpy as np
import pytest

from microrts_agent.decisions.env import SkillEnv
from microrts_agent.decisions.policy import discounted_returns
from microrts_agent.decisions.state import decision_prompt


@pytest.mark.parametrize("player", [0, 1])
def test_harvest_cycle_and_state(player):
    env = SkillEnv(opponent="PassiveAI", player=player, skill_ticks=256, gamma=0.97)
    try:
        state = env.reset()
        candidates = env.candidates(state)
        assert json.loads(json.dumps(state)) == state
        assert "action-selection probabilities, not win probabilities" in decision_prompt(
            state, candidates
        )
        for c in candidates:
            if c["skill"] != "wait":
                unit = env.gs.getUnit(c["actor"])
                assert int(unit.getPlayer()) == player
                action = env.skills.next_action(env.gs, c)
                assert unit.getUnitActions(env.gs).contains(action)
                assert env.gs.isUnitActionAllowed(unit, action)
        c = next(c for c in candidates if c["skill"] == "harvest")
        before = int(env.gs.getPlayer(player).getResources())
        result = env.execute(c)
        assert result["skill_end"] == "completed"
        assert int(env.gs.getPlayer(player).getResources()) == before + 1
        assert result["duration"] > 1
        assert result["next_state"]["time"] == result["duration"]
        rewards = np.array(result["raw_tick_rewards"]) @ env.weights
        assert result["reward"] == pytest.approx(rewards.sum())
        assert result["discounted_reward"] == pytest.approx(
            sum(0.97**i * r for i, r in enumerate(rewards))
        )
        assert result["discount"] == pytest.approx(0.97 ** len(rewards))
        assert {a["action"][1] for a in result["issued_actions"]} >= {1, 2, 3}
    finally:
        env.close()


def test_reserved_resources_and_truncation():
    env = SkillEnv(opponent="PassiveAI", skill_ticks=1, max_ticks=1)
    try:
        state = env.reset()
        c = next(c for c in env.candidates(state) if c["skill"] == "build")
        result = env.execute(c)
        assert result["truncated"] and not result["terminated"] and result["done"]
        assert result["next_state"]["time"] == 1  # not an auto-reset observation
        actor = next(u for u in result["next_state"]["units"] if u["id"] == c["actor"])
        assert actor["action"]["remaining_ticks"] > 0
        player = next(p for p in result["next_state"]["players"] if p["id"] == env.player)
        assert player["resources"] == 5 and player["available_resources"] == 0
        assert not any(
            c["skill"] in ("produce", "build") for c in env.candidates(result["next_state"])
        )
        with pytest.raises(RuntimeError, match="reset"):
            env.execute({"skill": "wait", "ticks": 1})
        assert env.reset()["time"] == 0
    finally:
        env.close()


def test_natural_terminal_retained():
    env = SkillEnv(opponent="WorkerRush", max_ticks=3000)
    try:
        env.reset()
        while not env.done:
            result = env.execute({"skill": "wait", "ticks": 64})
        assert result["terminated"] and not result["truncated"]
        assert result["winner"] == 1
        assert result["next_state"]["time"] > 0
        assert not any(u["owner"] == 0 for u in result["next_state"]["units"])
        assert result["reward"] == -10.0
    finally:
        env.close()


def test_smdp_return_bootstraps_only_at_time_limit():
    # Tick rewards [1, 2] then [3], gamma=.5. Terminal ignores a supplied value.
    first = {"discounted_reward": 2.0, "discount": 0.25, "terminated": False}
    last = {"discounted_reward": 3.0, "discount": 0.5, "terminated": True}
    assert discounted_returns([first, last], bootstrap=100.0) == [2.75, 3.0]
    # The same last state at a time limit retains V(next)=4.
    last["terminated"] = False
    assert discounted_returns([first, last], bootstrap=4.0) == [3.25, 5.0]
