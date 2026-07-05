"""
Question selection strategies for historical user training.
"""

from __future__ import annotations

import random

import pandas as pd


def random_subset(
    all_questions: pd.DataFrame,
    n_questions: int,
    config: dict,
) -> pd.DataFrame:
    """Select questions uniformly at random.

    Uses a local ``random.Random(seed)`` so a given ``config['seed']`` selects
    the same questions without seeding (and perturbing) the global RNG. With no
    seed the RNG is entropy-seeded, matching the previous unseeded behavior.
    """
    seed = config.get('seed', None)
    rng = random.Random(seed)

    n_questions = min(n_questions, len(all_questions))
    indices = rng.sample(range(len(all_questions)), n_questions)
    return all_questions.iloc[indices].reset_index(drop=True)
