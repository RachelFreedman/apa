"""
Ranking aggregation strategies for democratic voting.

Strategies:
- borda_count: Points based on position (default)
- plurality: Only count first-place votes
- copeland: Pairwise wins/losses
- instant_runoff: Iterative elimination (IRV)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def borda_count(rankings: dict[str, list[int]], config: dict) -> list[int]:
    """
    Aggregate rankings using Borda count.

    Each voter awards points based on position: 1st gets k-1 points, 2nd gets k-2, etc.
    """
    if not rankings:
        return []

    n_candidates = len(next(iter(rankings.values())))

    scores = defaultdict(float)
    for user_id, ranking in rankings.items():
        for position, candidate in enumerate(ranking):
            scores[candidate] += (n_candidates - 1 - position)

    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def plurality(rankings: dict[str, list[int]], config: dict) -> list[int]:
    """Aggregate rankings using plurality voting (only first-choice counts)."""
    if not rankings:
        return []

    first_place_counts = defaultdict(int)
    all_candidates = set()

    for user_id, ranking in rankings.items():
        if ranking:
            first_place_counts[ranking[0]] += 1
            all_candidates.update(ranking)

    return sorted(all_candidates, key=lambda x: (first_place_counts[x], -x), reverse=True)


def copeland(rankings: dict[str, list[int]], config: dict) -> list[int]:
    """Aggregate rankings using Copeland's method (pairwise wins: +1 win, -1 loss)."""
    if not rankings:
        return []

    all_candidates = set()
    for ranking in rankings.values():
        all_candidates.update(ranking)

    candidates = sorted(all_candidates)
    n = len(candidates)

    pairwise_wins = defaultdict(lambda: defaultdict(int))

    for user_id, ranking in rankings.items():
        pos = {c: i for i, c in enumerate(ranking)}
        for i, c1 in enumerate(candidates):
            for c2 in candidates[i+1:]:
                if c1 in pos and c2 in pos:
                    if pos[c1] < pos[c2]:
                        pairwise_wins[c1][c2] += 1
                    else:
                        pairwise_wins[c2][c1] += 1

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

    return sorted(candidates, key=lambda x: scores[x], reverse=True)


def instant_runoff(rankings: dict[str, list[int]], config: dict) -> list[int]:
    """Aggregate rankings using Instant-Runoff Voting (IRV)."""
    if not rankings:
        return []

    current_rankings = {k: list(v) for k, v in rankings.items()}

    remaining = set()
    for ranking in current_rankings.values():
        remaining.update(ranking)

    elimination_order = []

    while len(remaining) > 1:
        first_place = defaultdict(int)
        for ranking in current_rankings.values():
            for candidate in ranking:
                if candidate in remaining:
                    first_place[candidate] += 1
                    break

        min_votes = min(first_place.get(c, 0) for c in remaining)
        to_eliminate = [c for c in remaining if first_place.get(c, 0) == min_votes]

        eliminated = to_eliminate[0]
        remaining.remove(eliminated)
        elimination_order.append(eliminated)

    if remaining:
        elimination_order.append(remaining.pop())

    return list(reversed(elimination_order))
