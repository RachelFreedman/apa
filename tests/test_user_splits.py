"""
Unit tests for user-aware data splitting (FB LoRe protocol).
"""

import pytest
import numpy as np
import torch

from apa.data.user_splits import FBDataSplitter, SplitDataset, UserSplitResult


def create_test_embeddings(n_samples: int, embed_dim: int, n_users: int) -> dict:
    """Create synthetic embeddings for testing."""
    np.random.seed(42)

    # Create user IDs with varying sample counts
    user_ids = []
    samples_per_user = n_samples // n_users
    for i in range(n_users):
        user_ids.extend([f"user_{i}"] * samples_per_user)
    # Add remaining samples to last user
    remaining = n_samples - len(user_ids)
    if remaining > 0:
        user_ids.extend([f"user_{n_users-1}"] * remaining)

    return {
        'response_1_embeddings': np.random.randn(n_samples, embed_dim).astype(np.float32),
        'response_2_embeddings': np.random.randn(n_samples, embed_dim).astype(np.float32),
        'labels': np.random.randint(0, 2, n_samples),
        'user_ids': np.array(user_ids),
    }


class TestSplitDataset:
    """Tests for SplitDataset class."""

    def test_init(self):
        """Test basic initialization."""
        n_samples, embed_dim = 10, 32
        emb1 = np.random.randn(n_samples, embed_dim).astype(np.float32)
        emb2 = np.random.randn(n_samples, embed_dim).astype(np.float32)
        labels = np.random.randint(0, 2, n_samples)
        user_indices = np.zeros(n_samples, dtype=np.int64)

        dataset = SplitDataset(emb1, emb2, labels, user_indices)

        assert len(dataset) == n_samples
        assert dataset.embedding_dim == embed_dim

    def test_getitem(self):
        """Test getting an item."""
        n_samples, embed_dim = 5, 16
        emb1 = np.random.randn(n_samples, embed_dim).astype(np.float32)
        emb2 = np.random.randn(n_samples, embed_dim).astype(np.float32)
        labels = np.array([0, 1, 0, 1, 0])
        user_indices = np.array([0, 0, 1, 1, 0])

        dataset = SplitDataset(emb1, emb2, labels, user_indices)
        item = dataset[2]

        assert 'response_1_embedding' in item
        assert 'response_2_embedding' in item
        assert 'label' in item
        assert 'user_idx' in item
        assert item['response_1_embedding'].shape == (embed_dim,)
        assert item['label'].item() == 0
        assert item['user_idx'].item() == 1

    def test_n_users(self):
        """Test n_users property."""
        n_samples, embed_dim = 10, 32
        emb1 = np.random.randn(n_samples, embed_dim).astype(np.float32)
        emb2 = np.random.randn(n_samples, embed_dim).astype(np.float32)
        labels = np.random.randint(0, 2, n_samples)
        user_indices = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

        dataset = SplitDataset(emb1, emb2, labels, user_indices)

        assert dataset.n_users == 5


class TestFBDataSplitter:
    """Tests for FBDataSplitter class."""

    def test_init(self):
        """Test initialization."""
        embeddings = create_test_embeddings(100, 32, 10)
        splitter = FBDataSplitter(embeddings, seed=42)

        assert splitter.seen_user_ratio == 0.8
        assert splitter.dialog_train_ratio == 0.5
        assert len(splitter.valid_users) == 10

    def test_init_missing_keys(self):
        """Test error on missing keys."""
        embeddings = {
            'response_1_embeddings': np.random.randn(10, 32),
            'response_2_embeddings': np.random.randn(10, 32),
            # missing 'labels' and 'user_ids'
        }

        with pytest.raises(ValueError, match="Missing required key"):
            FBDataSplitter(embeddings)

    def test_split_user_ratio(self):
        """Test that user split ratio is approximately correct."""
        embeddings = create_test_embeddings(200, 32, 20)
        splitter = FBDataSplitter(embeddings, seen_user_ratio=0.8, seed=42)
        result = splitter.split()

        # 80% of 20 users = 16 seen, 4 unseen
        assert result.n_seen_users == 16
        assert result.n_unseen_users == 4

    def test_split_returns_four_datasets(self):
        """Test that split returns all four datasets."""
        embeddings = create_test_embeddings(100, 32, 10)
        splitter = FBDataSplitter(embeddings, seed=42)
        result = splitter.split()

        assert isinstance(result, UserSplitResult)
        assert isinstance(result.train_seen, SplitDataset)
        assert isinstance(result.test_seen, SplitDataset)
        assert isinstance(result.train_unseen, SplitDataset)
        assert isinstance(result.test_unseen, SplitDataset)

    def test_split_user_mappings(self):
        """Test that user mappings are created correctly."""
        embeddings = create_test_embeddings(100, 32, 10)
        splitter = FBDataSplitter(embeddings, seed=42)
        result = splitter.split()

        # Check that mappings have correct size
        assert len(result.seen_user_to_idx) == result.n_seen_users
        assert len(result.unseen_user_to_idx) == result.n_unseen_users

        # Check that indices are 0-indexed and contiguous
        seen_indices = sorted(result.seen_user_to_idx.values())
        unseen_indices = sorted(result.unseen_user_to_idx.values())

        assert seen_indices == list(range(result.n_seen_users))
        assert unseen_indices == list(range(result.n_unseen_users))

    def test_split_no_user_overlap(self):
        """Test that seen and unseen users don't overlap."""
        embeddings = create_test_embeddings(100, 32, 10)
        splitter = FBDataSplitter(embeddings, seed=42)
        result = splitter.split()

        seen_users = set(result.seen_user_to_idx.keys())
        unseen_users = set(result.unseen_user_to_idx.keys())

        assert len(seen_users & unseen_users) == 0

    def test_split_all_users_accounted(self):
        """Test that all valid users are in either seen or unseen."""
        embeddings = create_test_embeddings(100, 32, 10)
        splitter = FBDataSplitter(embeddings, seed=42)
        result = splitter.split()

        seen_users = set(result.seen_user_to_idx.keys())
        unseen_users = set(result.unseen_user_to_idx.keys())
        all_split_users = seen_users | unseen_users

        assert all_split_users == set(splitter.valid_users)

    def test_split_dialog_ratio(self):
        """Test that train/test split per user is approximately 50/50."""
        # Create data where each user has the same number of samples
        n_users = 5
        samples_per_user = 20
        n_samples = n_users * samples_per_user

        embeddings = {
            'response_1_embeddings': np.random.randn(n_samples, 32).astype(np.float32),
            'response_2_embeddings': np.random.randn(n_samples, 32).astype(np.float32),
            'labels': np.random.randint(0, 2, n_samples),
            'user_ids': np.array([f"user_{i // samples_per_user}" for i in range(n_samples)]),
        }

        splitter = FBDataSplitter(embeddings, dialog_train_ratio=0.5, seed=42)
        result = splitter.split()

        # Each seen user should have ~10 train and ~10 test samples
        # Allow some tolerance due to rounding
        train_per_user = samples_per_user * 0.5
        n_seen = result.n_seen_users

        # Total samples should be roughly balanced
        assert len(result.train_seen) == pytest.approx(n_seen * train_per_user, abs=n_seen)
        assert len(result.test_seen) == pytest.approx(n_seen * train_per_user, abs=n_seen)

    def test_split_reproducibility(self):
        """Test that splits are reproducible with same seed."""
        embeddings = create_test_embeddings(100, 32, 10)

        splitter1 = FBDataSplitter(embeddings, seed=42)
        result1 = splitter1.split()

        splitter2 = FBDataSplitter(embeddings, seed=42)
        result2 = splitter2.split()

        # Same users should be in seen/unseen
        assert result1.seen_user_to_idx == result2.seen_user_to_idx
        assert result1.unseen_user_to_idx == result2.unseen_user_to_idx

        # Same samples should be in each split
        assert len(result1.train_seen) == len(result2.train_seen)

    def test_split_different_seeds(self):
        """Test that different seeds produce different splits."""
        embeddings = create_test_embeddings(100, 32, 10)

        splitter1 = FBDataSplitter(embeddings, seed=42)
        result1 = splitter1.split()

        splitter2 = FBDataSplitter(embeddings, seed=123)
        result2 = splitter2.split()

        # Different seeds should produce different seen/unseen splits
        assert result1.seen_user_to_idx != result2.seen_user_to_idx

    def test_get_stats(self):
        """Test get_stats method."""
        embeddings = create_test_embeddings(100, 32, 10)
        splitter = FBDataSplitter(embeddings, seed=42)
        stats = splitter.get_stats()

        assert 'total_users' in stats
        assert 'n_seen_users' in stats
        assert 'n_unseen_users' in stats
        assert 'train_seen_samples' in stats
        assert 'test_seen_samples' in stats
        assert 'train_unseen_samples' in stats
        assert 'test_unseen_samples' in stats
        assert 'embedding_dim' in stats

        assert stats['total_users'] == 10
        assert stats['n_seen_users'] + stats['n_unseen_users'] == 10
        assert stats['embedding_dim'] == 32

    def test_min_samples_per_user(self):
        """Test that users with too few samples are filtered out."""
        # Create data with one user having only 1 sample
        embeddings = {
            'response_1_embeddings': np.random.randn(15, 32).astype(np.float32),
            'response_2_embeddings': np.random.randn(15, 32).astype(np.float32),
            'labels': np.random.randint(0, 2, 15),
            'user_ids': np.array(['u1'] * 5 + ['u2'] * 5 + ['u3'] * 4 + ['u4']),  # u4 has only 1
        }

        splitter = FBDataSplitter(embeddings, min_samples_per_user=2, seed=42)

        # u4 should be filtered out
        assert len(splitter.valid_users) == 3
        assert 'u4' not in splitter.valid_users

    def test_dataloader_compatible(self):
        """Test that split datasets work with DataLoader."""
        from torch.utils.data import DataLoader

        embeddings = create_test_embeddings(100, 32, 10)
        splitter = FBDataSplitter(embeddings, seed=42)
        result = splitter.split()

        loader = DataLoader(result.train_seen, batch_size=8, shuffle=True)
        batch = next(iter(loader))

        assert batch['response_1_embedding'].shape == (8, 32)
        assert batch['response_2_embedding'].shape == (8, 32)
        assert batch['label'].shape == (8,)
        assert batch['user_idx'].shape == (8,)
