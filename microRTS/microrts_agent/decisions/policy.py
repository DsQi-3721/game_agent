"""Replace TinyPolicy.forward with a differentiable decision model to train it."""

import torch
from torch import nn

from .skills import KINDS

TYPES = ("Resource", "Base", "Barracks", "Worker", "Light", "Heavy", "Ranged")


def state_features(state):
    player = state["player"]
    players = {p["id"]: p for p in state["players"]}
    values = [
        state["time"] / 3000,
        players[player]["available_resources"] / 10,
        players[1 - player]["available_resources"] / 10,
    ]
    for owner in (player, 1 - player):
        for kind in TYPES[1:]:
            units = [u for u in state["units"] if u["owner"] == owner and u["type"] == kind]
            values.extend([len(units) / 10, sum(u["hp"] for u in units) / 50])
    return values


def candidate_features(state, c):
    units = {u["id"]: u for u in state["units"]}
    actor = units.get(c.get("actor"))
    target = units.get(c.get("target_id", c.get("resource_id", c.get("base_id"))))
    values = [float(c["skill"] == k) for k in KINDS]
    values += [float(actor is not None and actor["type"] == t) for t in TYPES]
    values += [float(c.get("unit_type") == t) for t in TYPES]
    x, y = (actor["x"], actor["y"]) if actor else (0, 0)
    tx, ty = c.get("target", [target["x"], target["y"]] if target else [x, y])
    values += [
        x / state["width"],
        y / state["height"],
        tx / state["width"],
        ty / state["height"],
        abs(tx - x) / state["width"] + abs(ty - y) / state["height"],
        actor["hp"] / 10 if actor else 0.0,
        actor["resources"] if actor else 0.0,
        target["hp"] / 10 if target else 0.0,
    ]
    return values


class TinyPolicy(nn.Module):
    """Scores each candidate with the same MLP; supports variable option counts.

    forward(state, candidates) -> (logits[K], scalar expected discounted return).
    The value estimate is not a win probability. No text generation is required.
    """

    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(27 + 29, 64), nn.Tanh(), nn.Linear(64, 1))
        self.critic = nn.Sequential(nn.Linear(27, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, state, candidates):
        device = next(self.parameters()).device
        shared = state_features(state)
        options = torch.tensor(
            [shared + candidate_features(state, c) for c in candidates],
            dtype=torch.float32,
            device=device,
        )
        value = self.critic(torch.tensor(shared, dtype=torch.float32, device=device)).squeeze()
        return self.actor(options).squeeze(-1), value


def select(logits, greedy=False):
    distribution = torch.distributions.Categorical(logits=logits)
    index = logits.argmax() if greedy else distribution.sample()
    return int(index), distribution


def discounted_returns(transitions, bootstrap=0.0):
    """SMDP return: R_t + gamma**duration_t * V(next), reset only at true terminal."""
    result = []
    value = float(bootstrap)
    for t in reversed(transitions):
        value = t["discounted_reward"] + t["discount"] * (0.0 if t["terminated"] else value)
        result.append(value)
    return list(reversed(result))
