"""Opponent model: what we believe about a hidden Pokemon's set and stats.

Every tool asks this one object, so swapping in a better belief model (Bayesian
team prediction from revealed teammates, a learned predictor, ...) changes all
tools at once. The default is the conditioned format prior.
"""

from __future__ import annotations

from typing import Optional, Protocol

from arena.knowledge.estimate import StatEstimate, estimate_stats, is_own
from arena.knowledge.priors import SetPrior, condition, raw_prior
from arena.state.observe import known_ability, known_item, revealed_moves


class OpponentModel(Protocol):
    def set_prior(self, battle, mon) -> Optional[SetPrior]: ...

    def estimate(self, battle, mon) -> Optional[StatEstimate]: ...


class PriorOpponentModel:
    """Format priors conditioned on this battle's reveals."""

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


DEFAULT_MODEL = PriorOpponentModel()
