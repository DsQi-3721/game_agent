"""Event-driven multi-unit SkillEnv over MicroRTS JNI (with legacy single-skill support)."""

import numpy as np
from jpype import JClass
from jpype.types import JArray, JInt

from microrts_agent.envs.rl_vec_env import MicroRTSRLVecEnv

from .skills import Skills, action_row
from .state import read_state

REWARD_NAMES = (
    "win_loss",
    "resource",
    "worker",
    "base",
    "barracks",
    "attack",
    "light",
    "heavy",
    "ranged",
    "military_score",
)
# Engine reward weights:
# - shaped: dense macro/combat signals (useful for early exploration)
# - win_focused: terminal win/loss dominates (+20/-20) with light shaping + stalemate penalty
# - sparse: pure terminal win/loss (+1/-1)
SHAPED_WEIGHTS = (10.0, 1.0, 0.2, 0.0, 0.2, 1.0, 0.2, 0.2, 0.2, 0.0)
WIN_FOCUSED_WEIGHTS = (20.0, 0.08, 0.04, 0.0, 0.25, 0.08, 0.15, 0.15, 0.15, 0.0)
SPARSE_WEIGHTS = (1.0,) + (0.0,) * 9


def _seed_java_bot(ai_obj, seed):
    """Reset any java.util.Random fields inside a Java AI instance for reproducible rollouts."""
    cls = ai_obj.getClass()
    while cls is not None:
        for field in cls.getDeclaredFields():
            if str(field.getType().getName()) == "java.util.Random":
                field.setAccessible(True)
                rng = field.get(ai_obj)
                if rng is not None:
                    rng.setSeed(seed)
        cls = cls.getSuperclass()


def _skill_completed(gs, pgs, player, candidate, unit, had_cargo):
    kind = candidate["skill"]
    if kind in ("harvest", "return"):
        return had_cargo and int(unit.getResources()) == 0
    if kind == "attack":
        if candidate.get("target_id") is None:
            return not any(int(u.getPlayer()) == 1 - player for u in pgs.getUnits())
        return gs.getUnit(candidate["target_id"]) is None
    if kind == "move":
        return [int(unit.getX()), int(unit.getY())] == candidate["target"]
    if kind in ("build", "produce"):
        product = pgs.getUnitAt(*candidate["target"])
        return (
            product is not None
            and int(product.getPlayer()) == player
            and str(product.getType().name) == candidate["unit_type"]
        )
    return False


class SkillEnv:
    def __init__(
        self,
        map_path="maps/8x8/basesWorkers8x8A.xml",
        opponent="WorkerRush",
        player=0,
        max_ticks=3000,
        skill_ticks=64,
        candidate_limit=12,
        gamma=0.99,
        reward="shaped",
        seed=0,
    ):
        from microrts_agent.registries import ai

        if not (0 < gamma <= 1 and skill_ticks > 0 and max_ticks > 0 and candidate_limit >= 2):
            raise ValueError(
                "Require 0 < gamma <= 1, positive tick budgets, and at least two candidates"
            )
        self.map_path = map_path
        self.player, self.max_ticks, self.skill_ticks = player, max_ticks, skill_ticks
        self.opponent = opponent
        self.candidate_limit, self.gamma = candidate_limit, gamma
        self.seed = seed
        self.reward_mode = reward
        weight_map = {
            "shaped": SHAPED_WEIGHTS,
            "win_focused": WIN_FOCUSED_WEIGHTS,
            "sparse": SPARSE_WEIGHTS,
        }
        if reward not in weight_map:
            raise ValueError(f"Unknown reward mode {reward!r}; choose from {list(weight_map)}")
        self.weights = np.array(weight_map[reward])
        self.timeout_penalty = -6.0 if reward == "win_focused" else (-0.5 if reward == "sparse" else 0.0)
        self.base = MicroRTSRLVecEnv(
            num_selfplay_envs=0,
            num_bot_envs=1,
            partial_obs=False,
            max_steps=max_ticks,
            ai2s=[ai.AI_MAPPING[opponent]],
            map_paths=[map_path],
            reward_weight=self.weights,
            cycle_maps=[],
            jvm_args=["-Djava.awt.headless=true"],
            player=player,
        )
        self.client = self.base.vec_client.clients[0]
        self.unit_types = {t["name"]: t for t in self.base.utt["unitTypes"]}
        self.skills = Skills(self.base.real_utt)
        self.done = False
        self.active_skills = {}
        self.wait_until = {}
        JClass("rts.UnitAction").r.setSeed(seed)

    @property
    def gs(self):
        return self.client.gs

    def state(self):
        return read_state(self.gs, self.player, self.unit_types) | {
            "opponent": self.opponent,
            "skill_horizon": self.skill_ticks,
            "max_ticks": self.max_ticks,
        }

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.client.reset(self.player)
        JClass("rts.UnitAction").r.setSeed(self.seed)
        units = list(self.client.pgs.getUnits())
        if units:
            JClass("rts.units.Unit").next_ID = 1 + max(int(u.getID()) for u in units)
        _seed_java_bot(self.client.ai2, self.seed)
        self.done = False
        self.active_skills = {}
        self.wait_until = {}
        return self.state()

    def candidates(self, state):
        """Legacy global candidate shortlist across all idle units."""
        return self.skills.candidates(state, self.gs, self.candidate_limit)

    def idle_unit_candidates(self, state=None):
        """Return [(unit_dict, candidates), ...] for idle own units needing a decision.

        Units with only a forced wait are automatically put on a short cooldown so we
        never waste a neural forward pass on a 1-option choice.
        """
        if state is None:
            state = self.state()
        now = state["time"]
        live_ids = {u["id"] for u in state["units"] if u["owner"] == self.player}
        self.active_skills = {uid: v for uid, v in self.active_skills.items() if uid in live_ids}
        self.wait_until = {
            uid: t for uid, t in self.wait_until.items() if uid in live_ids and t > now
        }
        decisions = []
        for u in state["units"]:
            if u["owner"] != self.player or u["action"] is not None:
                continue
            uid = u["id"]
            if uid in self.active_skills or uid in self.wait_until:
                continue
            cands = self.skills.unit_candidates(state, self.gs, u, limit=self.candidate_limit)
            if len(cands) <= 1:
                self.wait_until[uid] = now + cands[0].get("ticks", 8)
                continue
            decisions.append((u, cands))
        return decisions

    def step_units(self, assignments, max_skip=None):
        """Register chosen skills for idle units and advance concurrently until the next decision.

        `assignments` maps unit_id -> candidate dict (or an iterable of (unit_id, candidate)).
        All units with active skills issue primitive actions in parallel on every tick.
        """
        if self.done:
            raise RuntimeError("Episode ended; call reset before another decision")
        items = assignments.items() if isinstance(assignments, dict) else assignments
        now = int(self.gs.getTime())
        for uid, cand in items:
            if cand["skill"] == "wait":
                self.wait_until[uid] = now + min(self.skill_ticks, cand.get("ticks", 8))
            else:
                self.active_skills[uid] = {
                    "candidate": cand,
                    "ticks_left": self.skill_ticks,
                    "had_cargo": False,
                }

        budget = max_skip or self.skill_ticks
        rewards = []
        terminated = truncated = False
        max_attack_r = int(self.client.maxAttackRadius)

        for _ in range(budget):
            rows = []
            tick_usage = self.gs.getResourceUsage().clone()
            # Issue next primitive action for EVERY active unit that is ready on this tick.
            for uid in list(self.active_skills):
                entry = self.active_skills[uid]
                unit = self.gs.getUnit(uid)
                if unit is None or entry["ticks_left"] <= 0:
                    del self.active_skills[uid]
                    continue
                entry["had_cargo"] |= int(unit.getResources()) > 0
                if self.gs.getActionAssignment(unit) is None:
                    action = self.skills.next_action(self.gs, entry["candidate"], usage=tick_usage)
                    if action is None:
                        del self.active_skills[uid]
                        continue
                    tick_usage.merge(action.resourceUsage(unit, self.client.pgs))
                    rows.append(action_row(unit, action, self.base.width, max_attack_r))

            response = self.client.gameStep(JArray(JInt, 2)(rows), self.player)
            raw = np.array(response.reward, dtype=float)
            rewards.append(float(raw @ self.weights))

            terminated = bool(self.gs.gameover())
            truncated = not terminated and int(self.gs.getTime()) >= self.max_ticks
            if truncated and self.timeout_penalty != 0.0:
                rewards[-1] += self.timeout_penalty
            if terminated or truncated:
                break

            # Update skill timers and clear completed skills.
            for uid in list(self.active_skills):
                entry = self.active_skills[uid]
                entry["ticks_left"] -= 1
                unit = self.gs.getUnit(uid)
                if (
                    unit is None
                    or entry["ticks_left"] <= 0
                    or _skill_completed(
                        self.gs,
                        self.client.pgs,
                        self.player,
                        entry["candidate"],
                        unit,
                        entry["had_cargo"],
                    )
                ):
                    self.active_skills.pop(uid, None)

            # Event-driven wake-up: stop as soon as ANY own unit is free and ready for orders.
            cur_time = int(self.gs.getTime())
            has_idle = False
            for u in self.client.pgs.getUnits():
                if int(u.getPlayer()) != self.player:
                    continue
                uid = int(u.getID())
                if (
                    self.gs.getActionAssignment(u) is None
                    and uid not in self.active_skills
                    and cur_time >= self.wait_until.get(uid, 0)
                ):
                    has_idle = True
                    break
            if has_idle:
                next_state = self.state()
                if self.idle_unit_candidates(next_state):
                    duration = len(rewards)
                    return {
                        "next_state": next_state,
                        "reward": sum(rewards),
                        "discounted_reward": sum(self.gamma**i * r for i, r in enumerate(rewards)),
                        "duration": duration,
                        "discount": self.gamma**duration,
                        "terminated": False,
                        "truncated": False,
                        "done": False,
                        "winner": None,
                    }

        duration = len(rewards)
        self.done = terminated or truncated
        winner = int(self.gs.winner()) if terminated else None
        return {
            "next_state": self.state(),
            "reward": sum(rewards),
            "discounted_reward": sum(self.gamma**i * r for i, r in enumerate(rewards)),
            "duration": duration,
            "discount": self.gamma**duration,
            "terminated": terminated,
            "truncated": truncated,
            "done": self.done,
            "winner": winner,
        }

    def execute(self, candidate):
        """Legacy single-unit blocking execution (kept for TinyPolicy and unit tests)."""
        if self.done:
            raise RuntimeError("Episode ended; call reset before another decision")
        wait = candidate["skill"] == "wait"
        budget = min(self.skill_ticks, candidate["ticks"]) if wait else self.skill_ticks
        rewards, raw_rewards, issued = [], [], []
        reason = "waited" if wait else "horizon"
        had_cargo = False
        terminated = truncated = False
        for _ in range(budget):
            rows = []
            if not wait:
                unit = self.gs.getUnit(candidate["actor"])
                if unit is None:
                    reason = "actor_lost"
                    break
                had_cargo |= int(unit.getResources()) > 0
                if self.gs.getActionAssignment(unit) is None:
                    action = self.skills.next_action(self.gs, candidate)
                    if action is None:
                        reason = "interrupted"
                        break
                    row = action_row(
                        unit, action, self.base.width, int(self.client.maxAttackRadius)
                    )
                    rows = [row]
                    issued.append(
                        {"time": int(self.gs.getTime()), "actor": candidate["actor"], "action": row}
                    )
            response = self.client.gameStep(JArray(JInt, 2)(rows), self.player)
            raw = np.array(response.reward, dtype=float).copy()
            raw_rewards.append(raw.tolist())
            rewards.append(float(raw @ self.weights))
            terminated = bool(self.gs.gameover())
            truncated = not terminated and int(self.gs.getTime()) >= self.max_ticks
            if terminated or truncated:
                reason = "terminal" if terminated else "time_limit"
                break
            if not wait:
                unit = self.gs.getUnit(candidate["actor"])
                if unit is None:
                    reason = "actor_lost"
                    break
                if _skill_completed(
                    self.gs, self.client.pgs, self.player, candidate, unit, had_cargo
                ):
                    reason = "completed"
                    break
        if not rewards:
            raise ValueError(f"Candidate cannot start in the current state: {candidate}")
        duration = len(rewards)
        self.done = terminated or truncated
        winner = int(self.gs.winner()) if terminated else None
        return {
            "next_state": self.state(),
            "reward": sum(rewards),
            "discounted_reward": sum(self.gamma**i * r for i, r in enumerate(rewards)),
            "duration": duration,
            "discount": self.gamma**duration,
            "tick_rewards": rewards,
            "raw_tick_rewards": raw_rewards,
            "issued_actions": issued,
            "skill_end": reason,
            "terminated": terminated,
            "truncated": truncated,
            "done": self.done,
            "winner": winner,
        }

    def close(self):
        self.base.close()
