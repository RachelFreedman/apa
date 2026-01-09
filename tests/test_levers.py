"""
Unit tests for lever modules.
"""

import pytest
import numpy as np
import pandas as pd

from apa.levers.lever_aggregate import (
    lever_aggregate_rankings,
    borda_count,
    plurality,
    copeland,
    instant_runoff,
    STRATEGIES as AGGREGATE_STRATEGIES,
)
from apa.levers.lever_sample import (
    lever_sample_users,
    random_sampling,
    stratified_sampling,
    temporal_mix_sampling,
    STRATEGIES as SAMPLE_STRATEGIES,
)
from apa.levers.lever_questions import (
    lever_select_questions,
    random_subset,
    stratified_by_type,
    STRATEGIES as QUESTION_STRATEGIES,
)


class TestLeverAggregate:
    """Tests for ranking aggregation strategies."""

    def test_borda_count_basic(self):
        """Test basic Borda count."""
        rankings = {
            'user_1': [0, 1, 2],
            'user_2': [0, 2, 1],
            'user_3': [1, 0, 2],
        }

        result = borda_count(rankings, {})

        assert len(result) == 3
        assert result[0] == 0  # 0 has highest Borda score (2+2+1=5)
        assert set(result) == {0, 1, 2}

    def test_borda_count_empty(self):
        """Test Borda count with empty input."""
        result = borda_count({}, {})
        assert result == []

    def test_plurality_basic(self):
        """Test basic plurality voting."""
        rankings = {
            'user_1': [0, 1, 2],
            'user_2': [0, 2, 1],
            'user_3': [1, 0, 2],
            'user_4': [0, 1, 2],
        }

        result = plurality(rankings, {})

        assert result[0] == 0  # 0 has most first-place votes (3)

    def test_copeland_basic(self):
        """Test basic Copeland method."""
        rankings = {
            'user_1': [0, 1, 2],
            'user_2': [0, 2, 1],
            'user_3': [0, 1, 2],
        }

        result = copeland(rankings, {})

        assert result[0] == 0  # 0 beats both 1 and 2 pairwise

    def test_instant_runoff_basic(self):
        """Test basic instant runoff voting."""
        rankings = {
            'user_1': [0, 1, 2],
            'user_2': [1, 0, 2],
            'user_3': [1, 2, 0],
            'user_4': [0, 2, 1],
        }

        result = instant_runoff(rankings, {})

        assert len(result) == 3
        assert set(result) == {0, 1, 2}

    def test_lever_dispatch(self):
        """Test lever dispatches to correct strategy."""
        rankings = {'user_1': [0, 1, 2]}

        result_borda = lever_aggregate_rankings(rankings, {'aggregate': 'borda_count'})
        result_plurality = lever_aggregate_rankings(rankings, {'aggregate': 'plurality'})

        assert len(result_borda) == 3
        assert len(result_plurality) == 3

    def test_lever_invalid_strategy(self):
        """Test lever raises on invalid strategy."""
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            lever_aggregate_rankings({}, {'aggregate': 'nonexistent'})

    def test_strategies_registered(self):
        """Test all expected strategies are registered."""
        expected = {'borda_count', 'plurality', 'copeland', 'instant_runoff', 'schulze', 'kemeny_young'}
        assert expected <= set(AGGREGATE_STRATEGIES.keys())


class TestLeverSample:
    """Tests for user sampling strategies."""

    def test_random_sampling_basic(self):
        """Test basic random sampling."""
        all_users = ['user_1', 'user_2', 'user_3', 'user_4', 'user_5']

        result = random_sampling(all_users, None, 3, {})

        assert len(result) == 3
        assert all(u in all_users for u in result)
        assert len(set(result)) == 3  # No duplicates

    def test_random_sampling_larger_m(self):
        """Test random sampling when m > available users."""
        all_users = ['user_1', 'user_2']

        result = lever_sample_users(all_users, None, 5, {'sample': 'random'})

        assert len(result) == 2  # Capped at available

    def test_stratified_sampling_basic(self):
        """Test basic stratified sampling."""
        all_users = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
        metadata = {
            'u1': {'century': 'C013'},
            'u2': {'century': 'C013'},
            'u3': {'century': 'C017'},
            'u4': {'century': 'C017'},
            'u5': {'century': 'C021'},
            'u6': {'century': 'C021'},
        }

        result = stratified_sampling(all_users, metadata, 6, {'stratify_by': 'century'})

        assert len(result) == 6
        # Should have some from each group
        centuries = [metadata[u]['century'] for u in result if u in metadata]
        assert len(set(centuries)) >= 2

    def test_stratified_no_metadata(self):
        """Test stratified falls back to random without metadata."""
        all_users = ['u1', 'u2', 'u3']
        result = stratified_sampling(all_users, None, 2, {})
        assert len(result) == 2

    def test_temporal_mix_sampling(self):
        """Test temporal mix sampling."""
        all_users = ['modern_1', 'modern_2', 'hist_1', 'hist_2']
        metadata = {
            'modern_1': {'century': 'C021'},
            'modern_2': {'century': 'C021'},
            'hist_1': {'century': 'C013'},
            'hist_2': {'century': 'C017'},
        }

        result = temporal_mix_sampling(all_users, metadata, 4, {'historical_ratio': 0.5})

        assert len(result) == 4

    def test_lever_dispatch(self):
        """Test lever dispatches to correct strategy."""
        all_users = ['u1', 'u2', 'u3']

        result = lever_sample_users(all_users, None, 2, {'sample': 'random'})

        assert len(result) == 2

    def test_strategies_registered(self):
        """Test all expected strategies are registered."""
        expected = {'random', 'stratified', 'weighted', 'temporal_mix'}
        assert expected <= set(SAMPLE_STRATEGIES.keys())


class TestLeverQuestions:
    """Tests for question selection strategies."""

    def test_random_subset_basic(self):
        """Test basic random subset selection."""
        df = pd.DataFrame({
            'question_id': range(10),
            'prompt': [f'q{i}' for i in range(10)],
        })

        result = random_subset(df, 5, {'seed': 42})

        assert len(result) == 5
        assert list(result.columns) == list(df.columns)

    def test_random_subset_reproducible(self):
        """Test random subset is reproducible with seed."""
        df = pd.DataFrame({
            'question_id': range(100),
            'prompt': [f'q{i}' for i in range(100)],
        })

        result1 = random_subset(df, 10, {'seed': 42})
        result2 = random_subset(df, 10, {'seed': 42})

        assert result1['question_id'].tolist() == result2['question_id'].tolist()

    def test_stratified_by_type(self):
        """Test stratified selection by conversation type."""
        df = pd.DataFrame({
            'question_id': range(12),
            'prompt': [f'q{i}' for i in range(12)],
            'conversation_type': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
        })

        result = stratified_by_type(df, 6, {})

        assert len(result) == 6
        # Should have some from each type
        types = set(result['conversation_type'].unique())
        assert len(types) >= 2

    def test_stratified_no_column(self):
        """Test stratified falls back when column missing."""
        df = pd.DataFrame({
            'question_id': range(10),
            'prompt': [f'q{i}' for i in range(10)],
        })

        result = stratified_by_type(df, 5, {})
        assert len(result) == 5

    def test_lever_dispatch(self):
        """Test lever dispatches to correct strategy."""
        df = pd.DataFrame({
            'question_id': range(10),
            'prompt': [f'q{i}' for i in range(10)],
        })

        result = lever_select_questions(df, 3, {'questions': 'random_subset'})

        assert len(result) == 3

    def test_lever_caps_n(self):
        """Test lever caps n_questions at available."""
        df = pd.DataFrame({
            'question_id': range(5),
            'prompt': [f'q{i}' for i in range(5)],
        })

        result = lever_select_questions(df, 100, {'questions': 'random_subset'})

        assert len(result) == 5

    def test_strategies_registered(self):
        """Test all expected strategies are registered."""
        expected = {'random_subset', 'stratified_by_type', 'diverse_topics', 'controversial', 'high_agreement', 'temporal_relevant'}
        assert expected <= set(QUESTION_STRATEGIES.keys())
