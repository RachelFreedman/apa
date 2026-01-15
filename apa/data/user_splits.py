"""
User-aware data splitting for Facebook LoRe replication.

Implements the data splitting strategy from the FB LoRe paper:
- 80% seen users / 20% unseen users
- 50% train dialogs / 50% test dialogs per user

This allows proper evaluation of:
- Seen users on test data (generalization to new prompts)
- Unseen users on test data (generalization to new users after few-shot)
- Train metrics for debugging
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class UserSplitResult:
    """Result of user-aware data splitting."""
    train_seen: "SplitDataset"
    test_seen: "SplitDataset"
    train_unseen: "SplitDataset"
    test_unseen: "SplitDataset"
    seen_user_to_idx: dict[str, int]
    unseen_user_to_idx: dict[str, int]
    n_seen_users: int
    n_unseen_users: int


class SplitDataset(Dataset):
    """
    Dataset for a specific split (train/test x seen/unseen).

    Each item contains embeddings for a preference pair and the label,
    along with a user index relative to this split's user set.
    """

    def __init__(
        self,
        response_1_embeddings: np.ndarray,
        response_2_embeddings: np.ndarray,
        labels: np.ndarray,
        user_indices: np.ndarray,
    ):
        """
        Initialize dataset.

        Args:
            response_1_embeddings: (n_samples, embed_dim) embeddings for response 1
            response_2_embeddings: (n_samples, embed_dim) embeddings for response 2
            labels: (n_samples,) binary labels (0 = prefer response 1, 1 = prefer response 2)
            user_indices: (n_samples,) user indices (0-indexed for this split's users)
        """
        self.response_1_embeddings = torch.tensor(
            response_1_embeddings, dtype=torch.float32
        )
        self.response_2_embeddings = torch.tensor(
            response_2_embeddings, dtype=torch.float32
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.user_indices = torch.tensor(user_indices, dtype=torch.long)

        assert len(self.response_1_embeddings) == len(self.labels)
        assert len(self.response_2_embeddings) == len(self.labels)
        assert len(self.user_indices) == len(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            'response_1_embedding': self.response_1_embeddings[idx],
            'response_2_embedding': self.response_2_embeddings[idx],
            'label': self.labels[idx],
            'user_idx': self.user_indices[idx],
        }

    @property
    def embedding_dim(self) -> int:
        """Dimension of embeddings."""
        return self.response_1_embeddings.shape[1]

    @property
    def n_users(self) -> int:
        """Number of unique users in this split."""
        return int(self.user_indices.max().item()) + 1 if len(self.user_indices) > 0 else 0


class FBDataSplitter:
    """
    User-aware data splitting following Facebook LoRe protocol.

    Splits users into seen (80%) and unseen (20%), then for each user
    splits their samples into train (50%) and test (50%).
    """

    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        seen_user_ratio: float = 0.8,
        dialog_train_ratio: float = 0.5,
        seed: int = 42,
        min_samples_per_user: int = 2,
    ):
        """
        Initialize the data splitter.

        Args:
            embeddings: Dictionary with 'response_1_embeddings', 'response_2_embeddings',
                       'labels', and 'user_ids'
            seen_user_ratio: Fraction of users to use as "seen" (default 0.8)
            dialog_train_ratio: Fraction of each user's dialogs for training (default 0.5)
            seed: Random seed for reproducibility
            min_samples_per_user: Minimum samples required per user to include them
        """
        self.embeddings = embeddings
        self.seen_user_ratio = seen_user_ratio
        self.dialog_train_ratio = dialog_train_ratio
        self.seed = seed
        self.min_samples_per_user = min_samples_per_user

        # Validate embeddings
        required_keys = ['response_1_embeddings', 'response_2_embeddings', 'labels', 'user_ids']
        for key in required_keys:
            if key not in embeddings:
                raise ValueError(f"Missing required key in embeddings: {key}")

        self.response_1_emb = embeddings['response_1_embeddings']
        self.response_2_emb = embeddings['response_2_embeddings']
        self.labels = embeddings['labels']
        self.user_ids = embeddings['user_ids']

        # Build user index
        self._build_user_index()

    def _build_user_index(self) -> None:
        """Build index mapping users to their sample indices."""
        self.user_to_samples: dict[Any, list[int]] = {}
        for idx, user_id in enumerate(self.user_ids):
            if user_id not in self.user_to_samples:
                self.user_to_samples[user_id] = []
            self.user_to_samples[user_id].append(idx)

        # Filter users with too few samples
        self.valid_users = [
            user_id for user_id, samples in self.user_to_samples.items()
            if len(samples) >= self.min_samples_per_user
        ]
        self.valid_users.sort()  # Sort for reproducibility

    def split(self) -> UserSplitResult:
        """
        Perform the user-aware split.

        Returns:
            UserSplitResult with four datasets and user mappings
        """
        rng = np.random.RandomState(self.seed)

        # Shuffle users
        shuffled_users = self.valid_users.copy()
        rng.shuffle(shuffled_users)

        # Split users into seen and unseen
        n_seen = int(len(shuffled_users) * self.seen_user_ratio)
        seen_users = sorted(shuffled_users[:n_seen])
        unseen_users = sorted(shuffled_users[n_seen:])

        # Create user-to-index mappings (0-indexed within each group)
        seen_user_to_idx = {user: idx for idx, user in enumerate(seen_users)}
        unseen_user_to_idx = {user: idx for idx, user in enumerate(unseen_users)}

        # Collect samples for each split
        train_seen_data = self._collect_split_data(seen_users, seen_user_to_idx, 'train', rng)
        test_seen_data = self._collect_split_data(seen_users, seen_user_to_idx, 'test', rng)
        train_unseen_data = self._collect_split_data(unseen_users, unseen_user_to_idx, 'train', rng)
        test_unseen_data = self._collect_split_data(unseen_users, unseen_user_to_idx, 'test', rng)

        return UserSplitResult(
            train_seen=SplitDataset(**train_seen_data),
            test_seen=SplitDataset(**test_seen_data),
            train_unseen=SplitDataset(**train_unseen_data),
            test_unseen=SplitDataset(**test_unseen_data),
            seen_user_to_idx=seen_user_to_idx,
            unseen_user_to_idx=unseen_user_to_idx,
            n_seen_users=len(seen_users),
            n_unseen_users=len(unseen_users),
        )

    def _collect_split_data(
        self,
        users: list[Any],
        user_to_idx: dict[Any, int],
        split: str,
        rng: np.random.RandomState,
    ) -> dict[str, np.ndarray]:
        """
        Collect data for a specific split.

        Args:
            users: List of user IDs for this group (seen or unseen)
            user_to_idx: Mapping from user ID to 0-indexed position
            split: 'train' or 'test'
            rng: Random state for shuffling

        Returns:
            Dictionary with arrays for the dataset
        """
        all_indices = []
        all_user_indices = []

        for user_id in users:
            user_idx = user_to_idx[user_id]
            sample_indices = self.user_to_samples[user_id].copy()

            # Shuffle this user's samples deterministically
            # Use a sub-seed based on user_id to ensure consistent splits across calls
            user_seed = hash(str(user_id)) % (2**31)
            user_rng = np.random.RandomState(user_seed)
            user_rng.shuffle(sample_indices)

            # Split into train and test
            n_train = max(1, int(len(sample_indices) * self.dialog_train_ratio))
            if split == 'train':
                selected_indices = sample_indices[:n_train]
            else:
                selected_indices = sample_indices[n_train:]

            all_indices.extend(selected_indices)
            all_user_indices.extend([user_idx] * len(selected_indices))

        all_indices = np.array(all_indices)
        all_user_indices = np.array(all_user_indices)

        return {
            'response_1_embeddings': self.response_1_emb[all_indices],
            'response_2_embeddings': self.response_2_emb[all_indices],
            'labels': self.labels[all_indices],
            'user_indices': all_user_indices,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the data and splits."""
        result = self.split()

        return {
            'total_users': len(self.valid_users),
            'n_seen_users': result.n_seen_users,
            'n_unseen_users': result.n_unseen_users,
            'train_seen_samples': len(result.train_seen),
            'test_seen_samples': len(result.test_seen),
            'train_unseen_samples': len(result.train_unseen),
            'test_unseen_samples': len(result.test_unseen),
            'embedding_dim': result.train_seen.embedding_dim,
        }
