"""
PRISM dataset loading utilities.

Provides functions for loading and preprocessing the PRISM pairwise preference dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from apa.config import DatasetConfig, HISTORICAL_PREFS_DATA


def get_user_column(df: pd.DataFrame) -> str | None:
    """
    Find the user identifier column in a DataFrame.

    Checks for 'user_id' first (preferred), then 'interaction_id'.

    Args:
        df: DataFrame to check

    Returns:
        Column name if found, None otherwise
    """
    if 'user_id' in df.columns:
        return 'user_id'
    if 'interaction_id' in df.columns:
        return 'interaction_id'
    return None


def get_unique_users(df: pd.DataFrame) -> list[str]:
    """
    Get list of unique user identifiers from DataFrame.

    Args:
        df: DataFrame with user column

    Returns:
        Sorted list of unique user IDs, empty list if no user column
    """
    user_col = get_user_column(df)
    if user_col is None:
        return []
    return sorted(df[user_col].unique().tolist())


def load_prism_pairwise(
    path: Path | str | None = None,
    min_pairs_per_user: int = 0,
) -> pd.DataFrame:
    """
    Load PRISM pairwise preference data.

    Args:
        path: Path to the CSV file. If None, uses default from config.
        min_pairs_per_user: Filter to users with at least this many pairs

    Returns:
        DataFrame with columns including user_id, question_id, etc.
    """
    if path is None:
        config = DatasetConfig()
        path = config.questions_pairwise_path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PRISM pairwise data not found at {path}")

    df = pd.read_csv(path)

    # Ensure user_id column exists
    if 'user_id' not in df.columns and 'interaction_id' in df.columns:
        df['user_id'] = df['interaction_id']

    # Filter by minimum pairs per user if specified
    if min_pairs_per_user > 0:
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_pairs_per_user].index
        df = df[df['user_id'].isin(valid_users)]
        print(f"Filtered to {len(valid_users)} users with >= {min_pairs_per_user} pairs")

    return df


class PRISMDataset(Dataset):
    """
    PyTorch Dataset for PRISM pairwise preference data.

    Provides embeddings and labels for preference learning.
    """

    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        labels: np.ndarray,
        user_ids: np.ndarray | None = None,
    ):
        """
        Initialize dataset.

        Args:
            embeddings: Dict with 'response_1_embeddings' and 'response_2_embeddings'
                       Each of shape (n_samples, embedding_dim)
            labels: Array of shape (n_samples,) with 0 or 1 indicating preference
            user_ids: Optional array of user IDs for each sample
        """
        self.response_1 = torch.tensor(embeddings['response_1_embeddings'], dtype=torch.float32)
        self.response_2 = torch.tensor(embeddings['response_2_embeddings'], dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        assert len(self.response_1) == len(self.response_2) == len(self.labels)

        # Handle user IDs
        if user_ids is not None:
            # Build user_id to index mapping
            unique_users = sorted(set(user_ids))
            self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
            self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
            self.user_indices = torch.tensor(
                [self.user_to_idx[uid] for uid in user_ids],
                dtype=torch.long
            )
            self._n_users = len(unique_users)
        else:
            self.user_to_idx = None
            self.idx_to_user = None
            self.user_indices = None
            self._n_users = 1

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single item.

        Returns:
            Dict with:
                - response_1_embedding: (embed_dim,)
                - response_2_embedding: (embed_dim,)
                - label: () scalar
                - user_idx: () scalar (if user_ids provided)
        """
        item = {
            'response_1_embedding': self.response_1[idx],
            'response_2_embedding': self.response_2[idx],
            'label': self.labels[idx],
        }
        if self.user_indices is not None:
            item['user_idx'] = self.user_indices[idx]
        return item

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.response_1.shape[1]

    @property
    def n_users(self) -> int:
        """Get number of unique users."""
        return self._n_users

    def get_user_data(self, user_id: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all data for a specific user.

        Args:
            user_id: User identifier

        Returns:
            Tuple of (response_1_embeddings, response_2_embeddings, labels)
            for this user
        """
        if self.user_to_idx is None:
            raise ValueError("Dataset was not initialized with user_ids")

        user_idx = self.user_to_idx.get(user_id)
        if user_idx is None:
            raise KeyError(f"User {user_id} not found in dataset")

        mask = self.user_indices == user_idx
        return (
            self.response_1[mask],
            self.response_2[mask],
            self.labels[mask],
        )


def group_embeddings_by_user(
    train_embeddings: list[dict],
    test_embeddings: list[dict],
    device: str | torch.device = "cuda:0",
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Group embeddings by user and compute difference (chosen - rejected).

    This follows the exact logic from LoRe/train_basis.py.

    Args:
        train_embeddings: List of train examples with extra_info containing embeddings
        test_embeddings: List of test examples with extra_info containing embeddings
        device: Device to put tensors on

    Returns:
        Tuple of (train_seen, train_unseen, test_seen, test_unseen)
        Each is a list of tensors, one per user
    """
    from collections import defaultdict
    import time
    from datetime import datetime

    def log(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)

    def process_dataset(dataset, seen_value, split_name):
        split_label = "seen" if seen_value else "unseen"
        log(f"Processing {split_name} {split_label} dataset ({len(dataset)} examples)...")
        start_time = time.time()
        grouped = defaultdict(lambda: {"embeddings": []})
        skipped = 0
        processed = 0

        for idx, example in enumerate(dataset):
            extra_info = example.get("extra_info", {})
            if extra_info.get("seen") == seen_value and extra_info.get("split") == split_name:
                user_id = extra_info.get("user_id")
                if user_id:
                    chosen_emb = extra_info.get("chosen_conv_embedding")
                    rejected_emb = extra_info.get("rejected_conv_embedding")
                    # Skip examples with None embeddings
                    if chosen_emb is None or rejected_emb is None:
                        skipped += 1
                        continue
                    chosen = torch.tensor(chosen_emb, dtype=torch.float32, device=device)
                    rejected = torch.tensor(rejected_emb, dtype=torch.float32, device=device)
                    grouped[user_id]["embeddings"].append(chosen - rejected)
                    processed += 1

            # Log progress every 10% of dataset
            if (idx + 1) % max(1, len(dataset) // 10) == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(dataset) - idx - 1) / rate if rate > 0 else 0
                log(f"  Progress: {idx+1}/{len(dataset)} ({100*(idx+1)/len(dataset):.1f}%) | "
                    f"Processed: {processed} | Skipped: {skipped} | "
                    f"ETA: {remaining:.1f}s")

        # Stack and sort by user_id
        log(f"  Stacking embeddings for {len(grouped)} users...")
        sorted_grouped = []
        count = 0
        for user_id in sorted(grouped.keys()):
            count += len(grouped[user_id]["embeddings"])
            sorted_grouped.append(
                torch.stack(grouped[user_id]["embeddings"]))

        elapsed = time.time() - start_time
        log(f"  Completed {split_name} {split_label}: {count} embeddings from {len(grouped)} users "
            f"({processed} processed, {skipped} skipped) in {elapsed:.1f}s")
        return sorted_grouped

    log("=" * 60)
    log("Grouping embeddings by user...")
    log("=" * 60)
    grouping_start = time.time()

    # Create all 4 groupings
    train_seen = process_dataset(train_embeddings, seen_value=True, split_name="train")
    train_unseen = process_dataset(train_embeddings, seen_value=False, split_name="train")
    test_seen = process_dataset(test_embeddings, seen_value=True, split_name="test")
    test_unseen = process_dataset(test_embeddings, seen_value=False, split_name="test")

    grouping_time = time.time() - grouping_start
    log(f"Embedding grouping completed in {grouping_time:.1f}s ({grouping_time/60:.1f} min)")
    log("=" * 60)

    return train_seen, train_unseen, test_seen, test_unseen
