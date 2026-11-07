"""Opponent model: what we believe about a hidden Pokemon's set and stats.

Every tool asks this one object, so swapping in a better belief model changes all tools at once.

- ``prior``: the format prior conditioned on this battle's reveals (revealed values become certain, the rest
  keeps its usage probability).
- ``team``: ``arena.opponent.model.TeamAwareOpponentModel`` -- current usage stats, teammates and co-reveals
  from ~2M public battles, and in-battle evidence (what did not happen: no Leftovers heal, two different moves
  without switching ...). Falls back to ``prior`` when its knowledge file is missing (offline test fixtures).

``DEFAULT_MODEL`` dispatches to the selected one: ``use_model(name)`` for the current task (an agent's
``opponent_model`` parameter, see ``arena.agents.adapter``), else ``ARENA_OPPONENT_MODEL``, else ``team``.
"""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from typing import Optional, Protocol

from arena.knowledge.estimate import StatEstimate, estimate_stats, is_own
from arena.knowledge.priors import SetPrior, condition, raw_prior
from arena.state.observe import known_ability, known_item, revealed_moves


class OpponentModel(Protocol):
    def set_prior(self, battle, mon) -> Optional[SetPrior]: ...

    def estimate(self, battle, mon) -> Optional[StatEstimate]: ...


class PriorOpponentModel:
    """Format priors conditioned on this battle's reveals."""

    name = "prior"

    def set_prior(self, battle, mon) -> Optional[SetPrior]:
        if is_own(battle, mon):
            return None
        prior = raw_prior(battle.format, mon.species)
        if prior is None:
            return None
        tera = None
        if getattr(mon, "is_terastallized", False) and mon.tera_type is not None:
            tera = mon.tera_type.name.title()
        return condition(prior, moves=revealed_moves(mon), item=known_item(mon),
                         ability=known_ability(mon), tera=tera)

    def estimate(self, battle, mon) -> Optional[StatEstimate]:
        return estimate_stats(battle, mon, self.set_prior(battle, mon))


PRIOR_MODEL = PriorOpponentModel()
_TEAM = None
_SELECTED: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("arena_opponent_model", default=None)
MODEL_NAMES = ("prior", "team")


def team_model():
    global _TEAM
    if _TEAM is None:
        from arena.opponent.model import TeamAwareOpponentModel
        _TEAM = TeamAwareOpponentModel()
    return _TEAM


def selected_name() -> str:
    name = _SELECTED.get() or os.environ.get("ARENA_OPPONENT_MODEL") or "team"
    return name if name in MODEL_NAMES else "team"


def model_by_name(name: str):
    return PRIOR_MODEL if name == "prior" else team_model()


@contextmanager
def use_model(name: Optional[str]):
    """Select the opponent model for the current task (None: keep the default)."""
    if not name:
        yield
        return
    token = _SELECTED.set(name)
    try:
        yield
    finally:
        _SELECTED.reset(token)


class SelectedOpponentModel:
    """``DEFAULT_MODEL``: the model selected for the current task."""

    @property
    def name(self) -> str:
        return selected_name()

    def current(self):
        return model_by_name(selected_name())

    def set_prior(self, battle, mon) -> Optional[SetPrior]:
        return self.current().set_prior(battle, mon)

    def estimate(self, battle, mon) -> Optional[StatEstimate]:
        return self.current().estimate(battle, mon)

    def on_request(self, battle) -> None:
        """Called once per request before the agent decides (``arena.agents.adapter``)."""
        m = self.current()
        if hasattr(m, "on_request"):
            m.on_request(battle)

    def belief(self, battle, mon):
        """The team model's full belief (None under the plain prior or without its knowledge)."""
        m = self.current()
        return m.belief(battle, mon) if hasattr(m, "belief") else None


DEFAULT_MODEL = SelectedOpponentModel()
