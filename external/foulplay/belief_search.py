"""Our beliefs in Foul Play's set sampling (inside Foul Play's environment). Importing this module replaces
``fp.search.standard_battles._sample_pokemon``: when the running search has beliefs for a Pokemon (thread-local
``_BELIEFS``, set by the caller around ``prepare_battles``), its set is drawn from our distributions (``direct`` /
``mix``) or from Foul Play's consistent candidates re-weighted to our marginals (``reweight``); otherwise Foul Play
samples as usual. Used by the look-ahead service and the offline search evaluation."""

import random
import threading
from copy import deepcopy

import fp.search.standard_battles as fp_sampling
from fp.data.sets import PokemonMoveset, PredictedPokemonSet
from fp.search.helpers import populate_pkmn_from_set

from belief_sampling import direct_set
from belief_weights import belief_weights

_BELIEFS = threading.local()                 # beliefs of the analysis running in this thread
_FP_SAMPLE = fp_sampling._sample_pokemon
SAMPLING = {"weighted": 0, "fallback": 0, "direct": 0, "direct_failed": 0}


def _cached_weights(key, cands, bel):
    """The weights of one candidate list, computed once per search (its samples share the position)."""
    cache = getattr(_BELIEFS, "cache", None)
    rows = cands()
    if cache is None:
        return belief_weights(rows, bel)
    key = (key[0], key[1], hash(tuple(rows)))           # by content: candidate objects may be rebuilt per call
    if key not in cache:
        cache[key] = belief_weights(rows, bel)
    return cache[key]


def _sample_with_beliefs(pkmn, mode):
    beliefs = getattr(_BELIEFS, "value", None)
    bel = beliefs.get(pkmn.name) if beliefs else None
    if not bel:
        return _FP_SAMPLE(pkmn, mode)
    fp_sampling.pokemon_guaranteed_move(pkmn)
    fp_sampling.set_most_likely_hidden_power(pkmn, mode)
    # 0: a set drawn from our beliefs (belief_mode direct: always; mix: half of the samples)
    how = getattr(_BELIEFS, "mode", "reweight")
    if how == "direct" or (how == "mix" and random.random() < 0.5):
        pps = direct_set(pkmn, bel)
        if pps is not None:
            populate_pkmn_from_set(pkmn, pps, source="beliefs-direct")
            SAMPLING["direct"] += 1
            return
        SAMPLING["direct_failed"] += 1
    # 1: full sets still consistent with what this Pokemon showed (Foul Play's step 1, weighted instead of uniform)
    full = mode.team_datasets.get_all_remaining_sets(pkmn)
    if full and (not pkmn.moves or random.random() < 0.75):
        w = _cached_weights(("full", pkmn.name), lambda: [
            (s.pkmn_set.item, s.pkmn_set.ability, s.pkmn_set.tera_type, s.pkmn_moveset.moves) for s in full], bel)
        populate_pkmn_from_set(pkmn, deepcopy(random.choices(full, weights=w)[0]), source="teamdatasets-full+beliefs")
        SAMPLING["weighted"] += 1
        return
    # 2: item / ability / nature / EVs from a consistent set, moves sampled on top (Foul Play's step 2, weighted)
    part = [s for s in mode.team_datasets.get_pkmn_sets_from_pkmn_name(pkmn)
            if s.pkmn_set.set_makes_sense(pkmn) and s.set_makes_logical_sense()]
    if part:
        w = _cached_weights(("part", pkmn.name), lambda: [
            (s.pkmn_set.item, s.pkmn_set.ability, s.pkmn_set.tera_type, ()) for s in part], bel)
        chosen = deepcopy(random.choices(part, weights=w)[0].pkmn_set)
        moves = fp_sampling.sample_pokemon_moveset_with_known_pkmn_set(pkmn, chosen, mode)
        populate_pkmn_from_set(pkmn, PredictedPokemonSet(pkmn_set=chosen, pkmn_moveset=PokemonMoveset(moves=moves)),
                               source="teamdatasets-partial+beliefs")
        SAMPLING["weighted"] += 1
        return
    SAMPLING["fallback"] += 1
    return _FP_SAMPLE(pkmn, mode)        # 3: Smogon trait combinations, as Foul Play does


fp_sampling._sample_pokemon = _sample_with_beliefs
