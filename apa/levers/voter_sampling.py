"""
User sampling strategies for democratic voting.

Strategies:
- random_sampling: Uniform random sampling
- stratified_sampling: Sample proportionally from groups
- weighted_sampling: Sample based on user weights
- temporal_mix_sampling: Mix modern and historical users
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def random_sampling(
    all_user_ids: list[str],
    user_metadata: dict[str, Any] | None,
    m: int,
    config: dict,
) -> list[str]:
    """Sample users uniformly at random."""
    return random.sample(all_user_ids, min(m, len(all_user_ids)))


def stratified_sampling(
    all_user_ids: list[str],
    user_metadata: dict[str, Any] | None,
    m: int,
    config: dict,
) -> list[str]:
    """Sample users with stratification by a metadata field (e.g., century)."""
    if user_metadata is None:
        return random_sampling(all_user_ids, user_metadata, m, config)

    stratify_by = config.get('stratify_by', 'century')

    groups: dict[Any, list[str]] = {}
    for user_id in all_user_ids:
        if user_id not in user_metadata:
            continue
        group = user_metadata[user_id].get(stratify_by, 'unknown')
        if group not in groups:
            groups[group] = []
        groups[group].append(user_id)

    if not groups:
        return random_sampling(all_user_ids, user_metadata, m, config)

    n_groups = len(groups)
    per_group = m // n_groups
    remainder = m % n_groups

    selected = []
    for i, (group, users) in enumerate(groups.items()):
        n_to_sample = per_group + (1 if i < remainder else 0)
        n_to_sample = min(n_to_sample, len(users))
        selected.extend(random.sample(users, n_to_sample))

    return selected[:m]


def weighted_sampling(
    all_user_ids: list[str],
    user_metadata: dict[str, Any] | None,
    m: int,
    config: dict,
) -> list[str]:
    """Sample users with weights based on metadata 'weight' field."""
    if user_metadata is None:
        return random_sampling(all_user_ids, user_metadata, m, config)

    weights = []
    valid_users = []
    for user_id in all_user_ids:
        if user_id in user_metadata and 'weight' in user_metadata[user_id]:
            weights.append(user_metadata[user_id]['weight'])
            valid_users.append(user_id)

    if not valid_users:
        return random_sampling(all_user_ids, user_metadata, m, config)

    weights = np.array(weights)
    weights = weights / weights.sum()

    indices = np.random.choice(
        len(valid_users),
        size=min(m, len(valid_users)),
        replace=False,
        p=weights,
    )

    return [valid_users[i] for i in indices]


def temporal_mix_sampling(
    all_user_ids: list[str],
    user_metadata: dict[str, Any] | None,
    m: int,
    config: dict,
) -> list[str]:
    """Sample a mix of modern (C021) and historical users."""
    if user_metadata is None:
        return random_sampling(all_user_ids, user_metadata, m, config)

    historical_ratio = config.get('historical_ratio', 0.5)

    modern_users = []
    historical_users = []

    for user_id in all_user_ids:
        if user_id not in user_metadata:
            modern_users.append(user_id)
            continue

        century = user_metadata[user_id].get('century', 'C021')
        if century == 'C021' or century is None:
            modern_users.append(user_id)
        else:
            historical_users.append(user_id)

    n_historical = int(m * historical_ratio)
    n_modern = m - n_historical

    n_historical = min(n_historical, len(historical_users))
    n_modern = min(n_modern, len(modern_users))

    remainder = m - n_historical - n_modern
    if remainder > 0:
        if len(historical_users) > n_historical:
            n_historical += min(remainder, len(historical_users) - n_historical)
            remainder = m - n_historical - n_modern
        if remainder > 0 and len(modern_users) > n_modern:
            n_modern += min(remainder, len(modern_users) - n_modern)

    selected = []
    if n_modern > 0 and modern_users:
        selected.extend(random.sample(modern_users, n_modern))
    if n_historical > 0 and historical_users:
        selected.extend(random.sample(historical_users, n_historical))

    return selected
