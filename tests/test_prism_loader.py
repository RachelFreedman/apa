"""
Unit tests for PRISM data loader.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from apa.load_prism import (
    get_user_column,
    get_unique_users,
    load_prism_pairwise,
    PRISMDataset,
)


class TestLoadPrismPairwise:
    """Tests for load_prism_pairwise (guards the tab-delimited CSV parsing)."""

    def _write_tsv(self, path, rows, header):
        lines = ["\t".join(header)]
        for r in rows:
            lines.append("\t".join(str(x) for x in r))
        path.write_text("\n".join(lines) + "\n")

    def test_reads_tab_delimited(self, tmp_path):
        p = tmp_path / "questions_pairwise.csv"
        self._write_tsv(
            p,
            rows=[
                ("u1", "q1", "a", "b"),
                ("u1", "q2", "c", "d"),
                ("u2", "q3", "e", "f"),
            ],
            header=["user_id", "question_id", "response_1", "response_2"],
        )
        df = load_prism_pairwise(p)
        assert len(df) == 3
        # Columns must be split on tabs, not left as one comma-less blob.
        assert set(["user_id", "question_id", "response_1", "response_2"]).issubset(df.columns)
        assert sorted(df["user_id"].unique()) == ["u1", "u2"]

    def test_interaction_id_aliased_to_user_id(self, tmp_path):
        p = tmp_path / "questions_pairwise.csv"
        self._write_tsv(
            p,
            rows=[("i1", "q1", "a", "b"), ("i2", "q2", "c", "d")],
            header=["interaction_id", "question_id", "response_1", "response_2"],
        )
        df = load_prism_pairwise(p)
        assert "user_id" in df.columns
        assert sorted(df["user_id"].unique()) == ["i1", "i2"]

    def test_min_pairs_per_user_filters(self, tmp_path):
        p = tmp_path / "questions_pairwise.csv"
        self._write_tsv(
            p,
            rows=[
                ("u1", "q1", "a", "b"),
                ("u1", "q2", "c", "d"),
                ("u2", "q3", "e", "f"),  # only 1 pair -> filtered out
            ],
            header=["user_id", "question_id", "response_1", "response_2"],
        )
        df = load_prism_pairwise(p, min_pairs_per_user=2)
        assert sorted(df["user_id"].unique()) == ["u1"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_prism_pairwise(tmp_path / "nope.csv")


class TestGetUserColumn:
    """Tests for get_user_column function."""

    def test_user_id_column(self):
        """Test finding user_id column."""
        df = pd.DataFrame({'user_id': [1, 2, 3], 'data': ['a', 'b', 'c']})
        assert get_user_column(df) == 'user_id'

    def test_interaction_id_column(self):
        """Test finding interaction_id column."""
        df = pd.DataFrame({'interaction_id': [1, 2, 3], 'data': ['a', 'b', 'c']})
        assert get_user_column(df) == 'interaction_id'

    def test_user_id_preferred(self):
        """Test user_id is preferred over interaction_id."""
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'interaction_id': [10, 20, 30],
        })
        assert get_user_column(df) == 'user_id'

    def test_no_user_column(self):
        """Test returns None when no user column exists."""
        df = pd.DataFrame({'data': [1, 2, 3]})
        assert get_user_column(df) is None


class TestGetUniqueUsers:
    """Tests for get_unique_users function."""

    def test_with_user_column(self):
        """Test getting unique users."""
        df = pd.DataFrame({'user_id': ['a', 'b', 'a', 'c', 'b']})
        users = get_unique_users(df)
        assert users == ['a', 'b', 'c']

    def test_without_user_column(self):
        """Test returns empty list without user column."""
        df = pd.DataFrame({'data': [1, 2, 3]})
        users = get_unique_users(df)
        assert users == []


class TestPRISMDataset:
    """Tests for PRISMDataset class."""

    def test_init_basic(self):
        """Test basic initialization."""
        embeddings = {
            'response_1_embeddings': np.random.randn(10, 32),
            'response_2_embeddings': np.random.randn(10, 32),
        }
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        dataset = PRISMDataset(embeddings, labels)

        assert len(dataset) == 10
        assert dataset.embedding_dim == 32
        assert dataset.n_users == 1

    def test_init_with_users(self):
        """Test initialization with user IDs."""
        embeddings = {
            'response_1_embeddings': np.random.randn(10, 32),
            'response_2_embeddings': np.random.randn(10, 32),
        }
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        user_ids = np.array(['u1', 'u1', 'u2', 'u2', 'u3', 'u3', 'u1', 'u2', 'u3', 'u1'])

        dataset = PRISMDataset(embeddings, labels, user_ids)

        assert len(dataset) == 10
        assert dataset.n_users == 3
        assert len(dataset.user_to_idx) == 3

    def test_getitem_basic(self):
        """Test getting an item without users."""
        embeddings = {
            'response_1_embeddings': np.random.randn(5, 32),
            'response_2_embeddings': np.random.randn(5, 32),
        }
        labels = np.array([0, 1, 0, 1, 0])

        dataset = PRISMDataset(embeddings, labels)
        item = dataset[0]

        assert 'response_1_embedding' in item
        assert 'response_2_embedding' in item
        assert 'label' in item
        assert item['response_1_embedding'].shape == (32,)
        assert item['response_2_embedding'].shape == (32,)
        assert isinstance(item['label'], torch.Tensor)

    def test_getitem_with_users(self):
        """Test getting an item with users."""
        embeddings = {
            'response_1_embeddings': np.random.randn(5, 32),
            'response_2_embeddings': np.random.randn(5, 32),
        }
        labels = np.array([0, 1, 0, 1, 0])
        user_ids = np.array(['u1', 'u2', 'u1', 'u2', 'u1'])

        dataset = PRISMDataset(embeddings, labels, user_ids)
        item = dataset[0]

        assert 'user_idx' in item
        assert isinstance(item['user_idx'], torch.Tensor)

    def test_embedding_dim_property(self):
        """Test embedding_dim property."""
        for dim in [32, 64, 768]:
            embeddings = {
                'response_1_embeddings': np.random.randn(5, dim),
                'response_2_embeddings': np.random.randn(5, dim),
            }
            labels = np.array([0, 1, 0, 1, 0])

            dataset = PRISMDataset(embeddings, labels)
            assert dataset.embedding_dim == dim
