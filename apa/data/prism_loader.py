"""
PRISM dataset loader for APA project.

Loads the pairwise comparison data from PRISM and provides
PyTorch Dataset classes for training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from apa.config import DatasetConfig, HISTORICAL_PREFS_DATA


def load_prism_pairwise(
    path: Path | None = None,
    n_samples: int | None = None,
) -> pd.DataFrame:
    """
    Load PRISM pairwise comparison data.

    Args:
        path: Path to pairwise CSV (uses default if None)
        n_samples: Limit to first N samples (for testing)

    Returns:
        DataFrame with columns:
            - question_id
            - prompt
            - response_1, response_2
            - response_1_id, response_2_id
            - human_preferred (1 or 2)
            - user_id (from PRISM)
    """
    if path is None:
        path = HISTORICAL_PREFS_DATA / "prism" / "questions_pairwise.csv"

    print(f"Loading PRISM data from {path}")
    df = pd.read_csv(path, sep='\t')

    if n_samples is not None:
        df = df.head(n_samples)

    print(f"Loaded {len(df)} pairwise comparisons")
    return df


def get_user_column(df: pd.DataFrame) -> str | None:
    """Get the user column name from the dataset."""
    if 'user_id' in df.columns:
        return 'user_id'
    elif 'interaction_id' in df.columns:
        return 'interaction_id'
    return None


def get_unique_users(df: pd.DataFrame) -> list[str]:
    """Get list of unique user IDs from the dataset."""
    user_col = get_user_column(df)
    if user_col:
        return sorted(df[user_col].unique().tolist())
    return []


def get_user_preferences(df: pd.DataFrame, user_id: str) -> pd.DataFrame:
    """Get all preferences for a specific user."""
    user_col = get_user_column(df)
    if user_col is None:
        return df
    return df[df[user_col] == user_id]


class PRISMDataset(Dataset):
    """
    PyTorch Dataset for PRISM pairwise preferences.

    Each item contains embeddings for a preference pair and the label.
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
            embeddings: Dictionary with 'response_1_embeddings', 'response_2_embeddings'
            labels: Binary labels (0 = prefer response 1, 1 = prefer response 2)
            user_ids: Optional array of user IDs for each pair
        """
        self.response_1_embeddings = torch.tensor(
            embeddings['response_1_embeddings'], dtype=torch.float32
        )
        self.response_2_embeddings = torch.tensor(
            embeddings['response_2_embeddings'], dtype=torch.float32
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

        if user_ids is not None:
            # Convert user_ids to integer indices
            unique_users = np.unique(user_ids)
            self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
            self.user_indices = torch.tensor(
                [self.user_to_idx[u] for u in user_ids], dtype=torch.long
            )
            self.n_users = len(unique_users)
        else:
            self.user_indices = None
            self.user_to_idx = {}
            self.n_users = 1

        assert len(self.response_1_embeddings) == len(self.labels)
        assert len(self.response_2_embeddings) == len(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            'response_1_embedding': self.response_1_embeddings[idx],
            'response_2_embedding': self.response_2_embeddings[idx],
            'label': self.labels[idx],
        }
        if self.user_indices is not None:
            item['user_idx'] = self.user_indices[idx]
        return item

    @property
    def embedding_dim(self) -> int:
        """Dimension of embeddings."""
        return self.response_1_embeddings.shape[1]


def create_prism_dataset(
    df: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> PRISMDataset:
    """
    Create PRISMDataset from dataframe and embeddings.

    Args:
        df: PRISM pairwise dataframe
        embeddings: Pre-computed embeddings

    Returns:
        PRISMDataset instance
    """
    # Convert human_preferred to binary labels
    # human_preferred='2' means response 2 is preferred (label=1)
    labels = (df['human_preferred'].astype(str) == '2').astype(int).values

    # Get user IDs if available
    user_ids = df['user_id'].values if 'user_id' in df.columns else None

    return PRISMDataset(embeddings, labels, user_ids)
