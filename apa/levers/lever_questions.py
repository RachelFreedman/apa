"""
Lever: Question Selection Strategies.

INJECTION POINT: This module controls how questions are selected
from the PRISM dataset for training historical user models. The
default is random selection, but this can be replaced with more
sophisticated methods.

To add a new strategy:
1. Create a new function following the same signature
2. Register it in the STRATEGIES dict
3. Update config to use the new strategy name
"""

from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np
import pandas as pd


# =============================================================================
# Strategy Registry
# =============================================================================

STRATEGIES: dict[str, Callable] = {}


def register_strategy(name: str):
    """Decorator to register a selection strategy."""
    def decorator(fn: Callable) -> Callable:
        STRATEGIES[name] = fn
        return fn
    return decorator


# =============================================================================
# Main Interface
# =============================================================================

def lever_select_questions(
    all_questions: pd.DataFrame,
    n_questions: int,
    config: dict,
) -> pd.DataFrame:
    """
    INJECTION POINT: Select a subset of questions for historical user training.

    This is the main entry point. It dispatches to the appropriate strategy
    based on the config.

    Args:
        all_questions: DataFrame with all available questions
        n_questions: Number of questions to select
        config: Configuration dict with 'questions' key for strategy name

    Returns:
        DataFrame with selected questions
    """
    strategy_name = config.get('questions', 'random_subset')

    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown question selection strategy: {strategy_name}. "
                        f"Available: {list(STRATEGIES.keys())}")

    # Ensure we don't try to select more than available
    n_questions = min(n_questions, len(all_questions))

    return STRATEGIES[strategy_name](all_questions, n_questions, config)


# =============================================================================
# Strategy Implementations
# =============================================================================

@register_strategy("random_subset")
def random_subset(
    all_questions: pd.DataFrame,
    n_questions: int,
    config: dict,
) -> pd.DataFrame:
    """
    Select questions uniformly at random.

    Default strategy: Simple random sampling without replacement.

    Args:
        all_questions: DataFrame with all questions
        n_questions: Number to select
        config: Optional 'seed' for reproducibility

    Returns:
        DataFrame with selected questions
    """
    seed = config.get('seed', None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    indices = random.sample(range(len(all_questions)), n_questions)
    return all_questions.iloc[indices].reset_index(drop=True)


@register_strategy("diverse_topics")
def diverse_topics(
    all_questions: pd.DataFrame,
    n_questions: int,
    config: dict,
) -> pd.DataFrame:
    """
    Select questions to maximize topic diversity.

    PLACEHOLDER: This is a stub for future implementation.
    Would use clustering or topic modeling to ensure diverse coverage.

    Args:
        all_questions: DataFrame with all questions
        n_questions: Number to select
        config: Configuration options

    Returns:
        DataFrame with selected questions
    """
    # TODO: Implement topic-based diverse selection
    # For now, use random selection
    print("[lever_questions] diverse_topics not implemented, using random_subset")
    return random_subset(all_questions, n_questions, config)


@register_strategy("controversial")
def controversial(
    all_questions: pd.DataFrame,
    n_questions: int,
    config: dict,
) -> pd.DataFrame:
    """
    Select questions that are likely to reveal preference differences.

    PLACEHOLDER: This is a stub for future implementation.
    Would select questions where different models/humans disagree.

    Args:
        all_questions: DataFrame with all questions
        n_questions: Number to select
        config: Configuration options

    Returns:
        DataFrame with selected questions
    """
    # TODO: Implement controversy-based selection
    print("[lever_questions] controversial not implemented, using random_subset")
    return random_subset(all_questions, n_questions, config)


@register_strategy("stratified_by_type")
def stratified_by_type(
    all_questions: pd.DataFrame,
    n_questions: int,
    config: dict,
) -> pd.DataFrame:
    """
    Select questions with stratification by conversation type.

    Ensures representation from different conversation types in PRISM.

    Args:
        all_questions: DataFrame with 'conversation_type' column
        n_questions: Number to select
        config: Configuration options

    Returns:
        DataFrame with selected questions
    """
    if 'conversation_type' not in all_questions.columns:
        print("[lever_questions] No conversation_type column, using random_subset")
        return random_subset(all_questions, n_questions, config)

    # Group by conversation type
    groups = all_questions.groupby('conversation_type')
    n_groups = len(groups)

    # Calculate samples per group
    per_group = n_questions // n_groups
    remainder = n_questions % n_groups

    selected = []
    for i, (group_name, group_df) in enumerate(groups):
        n_to_sample = per_group + (1 if i < remainder else 0)
        n_to_sample = min(n_to_sample, len(group_df))

        indices = random.sample(range(len(group_df)), n_to_sample)
        selected.append(group_df.iloc[indices])

    result = pd.concat(selected, ignore_index=True)

    # If we still don't have enough (due to small groups), fill randomly
    if len(result) < n_questions:
        remaining = all_questions[~all_questions.index.isin(result.index)]
        n_more = n_questions - len(result)
        if len(remaining) >= n_more:
            more_indices = random.sample(range(len(remaining)), n_more)
            result = pd.concat([result, remaining.iloc[more_indices]], ignore_index=True)

    return result.head(n_questions)


@register_strategy("high_agreement")
def high_agreement(
    all_questions: pd.DataFrame,
    n_questions: int,
    config: dict,
) -> pd.DataFrame:
    """
    Select questions where humans showed high agreement.

    PLACEHOLDER: This is a stub for future implementation.
    Would prioritize questions where the preference is clear.

    Args:
        all_questions: DataFrame with all questions
        n_questions: Number to select
        config: Configuration options

    Returns:
        DataFrame with selected questions
    """
    # TODO: Implement agreement-based selection
    print("[lever_questions] high_agreement not implemented, using random_subset")
    return random_subset(all_questions, n_questions, config)


@register_strategy("temporal_relevant")
def temporal_relevant(
    all_questions: pd.DataFrame,
    n_questions: int,
    config: dict,
) -> pd.DataFrame:
    """
    Select questions that are relevant across time periods.

    PLACEHOLDER: This is a stub for future implementation.
    Would filter out questions that are too modern/specific.

    Args:
        all_questions: DataFrame with all questions
        n_questions: Number to select
        config: Configuration options

    Returns:
        DataFrame with selected questions
    """
    # TODO: Implement temporal relevance filtering
    print("[lever_questions] temporal_relevant not implemented, using random_subset")
    return random_subset(all_questions, n_questions, config)
