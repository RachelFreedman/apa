"""
Lever: Ranking Aggregation Strategies.

INJECTION POINT: This module controls how individual user rankings
are aggregated into a single democratic ranking. The default is
Borda count, but this can be replaced with other voting rules.

To add a new strategy:
1. Create a new function following the same signature
2. Register it in the STRATEGIES dict
3. Update config to use the new strategy name
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np


# =============================================================================
# Strategy Registry
# =============================================================================

STRATEGIES: dict[str, Callable] = {}


def register_strategy(name: str):
    """Decorator to register an aggregation strategy."""
    def decorator(fn: Callable) -> Callable:
        STRATEGIES[name] = fn
        return fn
    return decorator


# =============================================================================
# Main Interface
# =============================================================================

def lever_aggregate_rankings(
    rankings: dict[str, list[int]],
    config: dict,
) -> list[int]:
    """
    INJECTION POINT: Aggregate multiple user rankings into one.

    This is the main entry point. It dispatches to the appropriate strategy
    based on the config.

    Args:
        rankings: Dict mapping user_id -> ranking (list of response indices,
                  where index 0 is most preferred)
        config: Configuration dict with 'aggregate' key for strategy name

    Returns:
        Aggregated ranking (list of response indices, best first)
    """
    strategy_name = config.get('aggregate', 'borda_count')

    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown aggregation strategy: {strategy_name}. "
                        f"Available: {list(STRATEGIES.keys())}")

    return STRATEGIES[strategy_name](rankings, config)


# =============================================================================
# Strategy Implementations
# =============================================================================

@register_strategy("borda_count")
def borda_count(
    rankings: dict[str, list[int]],
    config: dict,
) -> list[int]:
    """
    Aggregate rankings using Borda count.

    Default strategy: Each voter awards points based on position.
    For k candidates, 1st place gets k-1 points, 2nd gets k-2, etc.

    Args:
        rankings: Dict mapping user_id -> ranking
        config: Not used in this strategy

    Returns:
        Aggregated ranking based on Borda scores
    """
    if not rankings:
        return []

    # Get number of candidates from first ranking
    n_candidates = len(next(iter(rankings.values())))

    # Compute Borda scores
    scores = defaultdict(float)
    for user_id, ranking in rankings.items():
        for position, candidate in enumerate(ranking):
            # Higher score for better (lower) position
            scores[candidate] += (n_candidates - 1 - position)

    # Sort by score (descending)
    sorted_candidates = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return sorted_candidates


@register_strategy("plurality")
def plurality(
    rankings: dict[str, list[int]],
    config: dict,
) -> list[int]:
    """
    Aggregate rankings using plurality voting.

    Only considers each voter's first choice.

    Args:
        rankings: Dict mapping user_id -> ranking
        config: Not used in this strategy

    Returns:
        Aggregated ranking based on first-place votes
    """
    if not rankings:
        return []

    # Count first-place votes
    first_place_counts = defaultdict(int)
    all_candidates = set()

    for user_id, ranking in rankings.items():
        if ranking:
            first_place_counts[ranking[0]] += 1
            all_candidates.update(ranking)

    # Sort by votes (descending), then by candidate id for ties
    sorted_candidates = sorted(
        all_candidates,
        key=lambda x: (first_place_counts[x], -x),
        reverse=True,
    )

    return sorted_candidates


@register_strategy("copeland")
def copeland(
    rankings: dict[str, list[int]],
    config: dict,
) -> list[int]:
    """
    Aggregate rankings using Copeland's method.

    Counts pairwise wins: +1 for each win, -1 for each loss.

    Args:
        rankings: Dict mapping user_id -> ranking
        config: Not used in this strategy

    Returns:
        Aggregated ranking based on Copeland scores
    """
    if not rankings:
        return []

    # Get all candidates
    all_candidates = set()
    for ranking in rankings.values():
        all_candidates.update(ranking)

    candidates = sorted(all_candidates)
    n = len(candidates)

    # Count pairwise preferences
    pairwise_wins = defaultdict(lambda: defaultdict(int))

    for user_id, ranking in rankings.items():
        # For each pair, the one appearing earlier in the ranking is preferred
        pos = {c: i for i, c in enumerate(ranking)}
        for i, c1 in enumerate(candidates):
            for c2 in candidates[i+1:]:
                if c1 in pos and c2 in pos:
                    if pos[c1] < pos[c2]:
                        pairwise_wins[c1][c2] += 1
                    else:
                        pairwise_wins[c2][c1] += 1

    # Compute Copeland scores
    scores = defaultdict(int)
    for i, c1 in enumerate(candidates):
        for c2 in candidates[i+1:]:
            wins_c1 = pairwise_wins[c1][c2]
            wins_c2 = pairwise_wins[c2][c1]

            if wins_c1 > wins_c2:
                scores[c1] += 1
                scores[c2] -= 1
            elif wins_c2 > wins_c1:
                scores[c2] += 1
                scores[c1] -= 1
            # Ties: no change

    # Sort by Copeland score (descending)
    sorted_candidates = sorted(candidates, key=lambda x: scores[x], reverse=True)

    return sorted_candidates


@register_strategy("instant_runoff")
def instant_runoff(
    rankings: dict[str, list[int]],
    config: dict,
) -> list[int]:
    """
    Aggregate rankings using Instant-Runoff Voting (IRV).

    Iteratively eliminates the candidate with fewest first-place votes.

    Args:
        rankings: Dict mapping user_id -> ranking
        config: Not used in this strategy

    Returns:
        Aggregated ranking based on IRV elimination order (reversed)
    """
    if not rankings:
        return []

    # Work with copies to avoid modifying original
    current_rankings = {k: list(v) for k, v in rankings.items()}

    # Get all candidates
    remaining = set()
    for ranking in current_rankings.values():
        remaining.update(ranking)

    elimination_order = []

    while len(remaining) > 1:
        # Count first-place votes
        first_place = defaultdict(int)
        for ranking in current_rankings.values():
            # Find first remaining candidate in this ranking
            for candidate in ranking:
                if candidate in remaining:
                    first_place[candidate] += 1
                    break

        # Find candidate(s) with fewest first-place votes
        min_votes = min(first_place.get(c, 0) for c in remaining)
        to_eliminate = [c for c in remaining if first_place.get(c, 0) == min_votes]

        # Eliminate (break ties arbitrarily)
        eliminated = to_eliminate[0]
        remaining.remove(eliminated)
        elimination_order.append(eliminated)

    # Add last remaining candidate
    if remaining:
        elimination_order.append(remaining.pop())

    # Reverse to get best-first ordering
    return list(reversed(elimination_order))


@register_strategy("schulze")
def schulze(
    rankings: dict[str, list[int]],
    config: dict,
) -> list[int]:
    """
    Aggregate rankings using the Schulze method.

    PLACEHOLDER: This is a stub for future implementation.
    The Schulze method is a Condorcet method that always selects
    the Condorcet winner if one exists.

    Args:
        rankings: Dict mapping user_id -> ranking
        config: Not used in this strategy

    Returns:
        Aggregated ranking
    """
    # TODO: Implement Schulze method
    print("[lever_aggregate] schulze not fully implemented, using copeland")
    return copeland(rankings, config)


@register_strategy("kemeny_young")
def kemeny_young(
    rankings: dict[str, list[int]],
    config: dict,
) -> list[int]:
    """
    Aggregate rankings using the Kemeny-Young method.

    PLACEHOLDER: This is a stub for future implementation.
    Finds the ranking that minimizes total Kendall tau distance
    to all input rankings. NP-hard for general case.

    Args:
        rankings: Dict mapping user_id -> ranking
        config: Not used in this strategy

    Returns:
        Aggregated ranking
    """
    # TODO: Implement Kemeny-Young method (or approximation)
    print("[lever_aggregate] kemeny_young not implemented, using borda_count")
    return borda_count(rankings, config)
