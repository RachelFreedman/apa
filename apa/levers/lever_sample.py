"""
Lever: User Sampling Strategies.

INJECTION POINT: This module controls how user models are sampled
for democratic voting. The default is random sampling, but this can
be replaced with more sophisticated methods.

To add a new strategy:
1. Create a new function following the same signature
2. Register it in the STRATEGIES dict
3. Update config to use the new strategy name
"""

from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np


# =============================================================================
# Strategy Registry
# =============================================================================

STRATEGIES: dict[str, Callable] = {}


def register_strategy(name: str):
    """Decorator to register a sampling strategy."""
    def decorator(fn: Callable) -> Callable:
        STRATEGIES[name] = fn
        return fn
    return decorator


# =============================================================================
# Main Interface
# =============================================================================

def lever_sample_users(
    all_user_ids: list[str],
    user_metadata: dict[str, Any] | None,
    m: int,
    config: dict,
) -> list[str]:
    """
    INJECTION POINT: Select m users for democratic voting.

    This is the main entry point. It dispatches to the appropriate strategy
    based on the config.

    Args:
        all_user_ids: List of all available user IDs
        user_metadata: Optional dict mapping user_id -> metadata (e.g., century, demographics)
        m: Number of users to sample
        config: Configuration dict with 'sample' key for strategy name

    Returns:
        List of m selected user IDs
    """
    strategy_name = config.get('sample', 'random')

    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown sampling strategy: {strategy_name}. "
                        f"Available: {list(STRATEGIES.keys())}")

    # Ensure we don't try to sample more than available
    m = min(m, len(all_user_ids))

    return STRATEGIES[strategy_name](all_user_ids, user_metadata, m, config)


# =============================================================================
# Strategy Implementations
# =============================================================================

@register_strategy("random")
def random_sampling(
    all_user_ids: list[str],
    user_metadata: dict[str, Any] | None,
    m: int,
    config: dict,
) -> list[str]:
    """
    Sample users uniformly at random.

    Default strategy: Simple random sampling without replacement.

    Args:
        all_user_ids: List of all user IDs
        user_metadata: Not used in this strategy
        m: Number of users to sample
        config: Not used in this strategy

    Returns:
        List of m randomly selected user IDs
    """
    return random.sample(all_user_ids, m)


@register_strategy("stratified")
def stratified_sampling(
    all_user_ids: list[str],
    user_metadata: dict[str, Any] | None,
    m: int,
    config: dict,
) -> list[str]:
    """
    Sample users with stratification by a metadata field.

    Ensures representation from different groups (e.g., centuries).

    Args:
        all_user_ids: List of all user IDs
        user_metadata: Dict mapping user_id -> metadata with 'group' key
        m: Number of users to sample
        config: Config with optional 'stratify_by' key

    Returns:
        List of m users, stratified by group
    """
    if user_metadata is None:
        print("[lever_sample] stratified requires user_metadata, falling back to random")
        return random_sampling(all_user_ids, user_metadata, m, config)

    stratify_by = config.get('stratify_by', 'century')

    # Group users by the stratification field
    groups: dict[Any, list[str]] = {}
    for user_id in all_user_ids:
        if user_id not in user_metadata:
            continue
        group = user_metadata[user_id].get(stratify_by, 'unknown')
        if group not in groups:
            groups[group] = []
        groups[group].append(user_id)

    if not groups:
        print("[lever_sample] No valid groups found, falling back to random")
        return random_sampling(all_user_ids, user_metadata, m, config)

    # Sample proportionally from each group
    n_groups = len(groups)
    per_group = m // n_groups
    remainder = m % n_groups

    selected = []
    for i, (group, users) in enumerate(groups.items()):
        n_to_sample = per_group + (1 if i < remainder else 0)
        n_to_sample = min(n_to_sample, len(users))
        selected.extend(random.sample(users, n_to_sample))

    return selected[:m]


@register_strategy("weighted")
def weighted_sampling(
    all_user_ids: list[str],
    user_metadata: dict[str, Any] | None,
    m: int,
    config: dict,
) -> list[str]:
    """
    Sample users with weights based on metadata.

    PLACEHOLDER: This is a stub for future implementation.
    Could weight by recency, confidence, etc.

    Args:
        all_user_ids: List of all user IDs
        user_metadata: Dict with 'weight' field per user
        m: Number of users to sample
        config: Configuration

    Returns:
        List of m weighted-sampled user IDs
    """
    if user_metadata is None:
        print("[lever_sample] weighted requires user_metadata, falling back to random")
        return random_sampling(all_user_ids, user_metadata, m, config)

    # Get weights
    weights = []
    valid_users = []
    for user_id in all_user_ids:
        if user_id in user_metadata and 'weight' in user_metadata[user_id]:
            weights.append(user_metadata[user_id]['weight'])
            valid_users.append(user_id)

    if not valid_users:
        print("[lever_sample] No weights found, falling back to random")
        return random_sampling(all_user_ids, user_metadata, m, config)

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Sample without replacement
    indices = np.random.choice(
        len(valid_users),
        size=min(m, len(valid_users)),
        replace=False,
        p=weights,
    )

    return [valid_users[i] for i in indices]


@register_strategy("temporal_mix")
def temporal_mix_sampling(
    all_user_ids: list[str],
    user_metadata: dict[str, Any] | None,
    m: int,
    config: dict,
) -> list[str]:
    """
    Sample a mix of users from different time periods.

    Specifically designed for the APA use case where we have
    PRISM users (modern) and historical users (past centuries).

    Args:
        all_user_ids: List of all user IDs
        user_metadata: Dict with 'century' field per user
        m: Number of users to sample
        config: Config with 'historical_ratio' (default 0.5)

    Returns:
        List of m users mixing modern and historical
    """
    if user_metadata is None:
        print("[lever_sample] temporal_mix requires user_metadata, falling back to random")
        return random_sampling(all_user_ids, user_metadata, m, config)

    historical_ratio = config.get('historical_ratio', 0.5)

    # Separate modern and historical users
    modern_users = []
    historical_users = []

    for user_id in all_user_ids:
        if user_id not in user_metadata:
            modern_users.append(user_id)  # Default to modern
            continue

        century = user_metadata[user_id].get('century', 'C021')
        if century == 'C021' or century is None:
            modern_users.append(user_id)
        else:
            historical_users.append(user_id)

    # Calculate split
    n_historical = int(m * historical_ratio)
    n_modern = m - n_historical

    # Adjust if not enough users in either category
    n_historical = min(n_historical, len(historical_users))
    n_modern = min(n_modern, len(modern_users))

    # Fill remainder from whichever has more
    remainder = m - n_historical - n_modern
    if remainder > 0:
        if len(historical_users) > n_historical:
            n_historical += min(remainder, len(historical_users) - n_historical)
            remainder = m - n_historical - n_modern
        if remainder > 0 and len(modern_users) > n_modern:
            n_modern += min(remainder, len(modern_users) - n_modern)

    # Sample from each group
    selected = []
    if n_modern > 0 and modern_users:
        selected.extend(random.sample(modern_users, n_modern))
    if n_historical > 0 and historical_users:
        selected.extend(random.sample(historical_users, n_historical))

    return selected
