"""
Unit tests for strategy modules.
"""

import pytest
import numpy as np
import pandas as pd

from apa.levers.voter_aggregation import (
    borda_count,
    plurality,
    copeland,
    instant_runoff,
)
from apa.levers.query_selection import random_subset, select_by_ids
from apa.levers.voter_sampling import (
    random_sampling,
    stratified_sampling,
    temporal_mix_sampling,
)
from apa.levers.query_selection import random_subset


class TestVoterAggregation:
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


class TestVoterSampling:
    """Tests for user sampling strategies."""

    def test_random_sampling_basic(self):
        """Test basic random sampling."""
        all_users = ['user_1', 'user_2', 'user_3', 'user_4', 'user_5']

        result = random_sampling(all_users, None, 3, {})

        assert len(result) == 3
        assert all(u in all_users for u in result)
        assert len(set(result)) == 3  # No duplicates

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


class TestQuerySelection:
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

    def test_random_subset_caps_n(self):
        """Test random subset caps n_questions at available."""
        df = pd.DataFrame({
            'question_id': range(5),
            'prompt': [f'q{i}' for i in range(5)],
        })

        result = random_subset(df, 100, {})

        assert len(result) == 5


class TestSelectByIds:
    """Tests for select_by_ids question selection."""

    def test_basic_selection(self):
        df = pd.DataFrame({
            'question_id': [10, 20, 30, 40, 50],
            'prompt': ['a', 'b', 'c', 'd', 'e'],
        })
        result = select_by_ids(df, [20, 40])
        assert len(result) == 2
        assert set(result['question_id']) == {20, 40}

    def test_preserves_all_columns(self):
        df = pd.DataFrame({
            'question_id': [1, 2, 3],
            'prompt': ['a', 'b', 'c'],
            'response_1': ['r1a', 'r1b', 'r1c'],
        })
        result = select_by_ids(df, [2])
        assert list(result.columns) == list(df.columns)
        assert result.iloc[0]['prompt'] == 'b'

    def test_missing_ids_raises(self):
        df = pd.DataFrame({'question_id': [1, 2, 3], 'prompt': ['a', 'b', 'c']})
        with pytest.raises(ValueError, match="not found"):
            select_by_ids(df, [1, 99])

    def test_duplicate_rows_for_same_id(self):
        """If a question_id appears multiple times, all rows are returned."""
        df = pd.DataFrame({
            'question_id': [1, 1, 2],
            'prompt': ['a', 'a', 'b'],
            'interaction_id': ['u1', 'u2', 'u3'],
        })
        result = select_by_ids(df, [1])
        assert len(result) == 2
