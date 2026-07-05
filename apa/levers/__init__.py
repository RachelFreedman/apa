"""
Strategy modules for democratic inference.

- voter_sampling: User sampling strategies
- voter_aggregation: Ranking aggregation strategies
- query_selection: Question selection strategies
- slate_generation: Response generation strategies
"""

from apa.levers.voter_sampling import (
    random_sampling,
    stratified_sampling,
    weighted_sampling,
    temporal_mix_sampling,
)
from apa.levers.voter_aggregation import (
    borda_count,
    plurality,
    copeland,
    instant_runoff,
)
from apa.levers.query_selection import random_subset
from apa.levers.slate_generation import temperature_sampling


# =============================================================================
# Strategy registries
# =============================================================================
# Map strategy names (including the shorthand aliases used by
# apa.config.InferenceConfig) to their lever functions, so a config string
# selects a lever instead of it being hardcoded at the call site.

SAMPLERS = {
    "random": random_sampling,
    "random_sampling": random_sampling,
    "stratified": stratified_sampling,
    "stratified_sampling": stratified_sampling,
    "weighted": weighted_sampling,
    "weighted_sampling": weighted_sampling,
    "temporal_mix": temporal_mix_sampling,
    "temporal_mix_sampling": temporal_mix_sampling,
}

AGGREGATORS = {
    "borda": borda_count,
    "borda_count": borda_count,
    "plurality": plurality,
    "copeland": copeland,
    "instant_runoff": instant_runoff,
    "irv": instant_runoff,
}

GENERATORS = {
    "temperature": temperature_sampling,
    "temperature_sampling": temperature_sampling,
}

SELECTORS = {
    "random_subset": random_subset,
}


def _resolve(registry: dict, name: str, kind: str):
    """Look up a strategy function by name, raising a clear error if unknown."""
    try:
        return registry[name]
    except KeyError:
        raise ValueError(
            f"Unknown {kind} strategy: {name!r}. Available: {sorted(registry)}"
        )


def get_sampler(name: str):
    """Resolve a voter-sampling strategy name to its function."""
    return _resolve(SAMPLERS, name, "sampler")


def get_aggregator(name: str):
    """Resolve a ranking-aggregation strategy name to its function."""
    return _resolve(AGGREGATORS, name, "aggregator")


def get_generator(name: str):
    """Resolve a response-generation strategy name to its function."""
    return _resolve(GENERATORS, name, "generator")


def get_selector(name: str):
    """Resolve a question-selection strategy name to its function."""
    return _resolve(SELECTORS, name, "selector")


__all__ = [
    "random_sampling",
    "stratified_sampling",
    "weighted_sampling",
    "temporal_mix_sampling",
    "borda_count",
    "plurality",
    "copeland",
    "instant_runoff",
    "random_subset",
    "temperature_sampling",
    "SAMPLERS",
    "AGGREGATORS",
    "GENERATORS",
    "SELECTORS",
    "get_sampler",
    "get_aggregator",
    "get_generator",
    "get_selector",
]
