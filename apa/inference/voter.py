"""
Voter module for democratic inference.

Provides utilities for scoring and ranking responses using
learned user reward models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import numpy as np

from apa.reward.lore_model import LoReRewardModel
from apa.utils.embedding_utils import embed_texts, get_embedding_model


class UserVoter:
    """
    A voter that scores responses based on a learned user preference model.

    Uses the LoRe reward model to compute scores for responses.
    """

    def __init__(
        self,
        user_id: str,
        user_vector: torch.Tensor,
        basis_matrix: torch.Tensor,
        metadata: dict | None = None,
    ):
        """
        Initialize voter.

        Args:
            user_id: Unique identifier for this user
            user_vector: User's preference vector (rank,)
            basis_matrix: Shared basis matrix V (embed_dim, rank)
            metadata: Optional metadata (e.g., century, profile)
        """
        self.user_id = user_id
        self.w = user_vector
        self.V = basis_matrix
        self.metadata = metadata or {}

        # Precompute user's reward direction: V @ w
        self.reward_direction = self.V @ self.w

    def score_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Score response embeddings.

        Args:
            embeddings: (n_responses, embed_dim) tensor

        Returns:
            scores: (n_responses,) tensor
        """
        return embeddings @ self.reward_direction

    def rank_embeddings(self, embeddings: torch.Tensor) -> list[int]:
        """
        Rank response embeddings by score (highest first).

        Args:
            embeddings: (n_responses, embed_dim) tensor

        Returns:
            ranking: List of response indices, best first
        """
        scores = self.score_embeddings(embeddings)
        return torch.argsort(scores, descending=True).tolist()


class VoterPool:
    """
    A pool of user voters for democratic voting.

    Manages loading and accessing multiple user voters.
    """

    def __init__(
        self,
        basis_matrix: torch.Tensor,
    ):
        """
        Initialize voter pool.

        Args:
            basis_matrix: Shared basis matrix V (embed_dim, rank)
        """
        self.V = basis_matrix
        self.voters: dict[str, UserVoter] = {}
        self.embedding_model = None

    def add_voter(
        self,
        user_id: str,
        user_vector: torch.Tensor,
        metadata: dict | None = None,
    ) -> None:
        """
        Add a voter to the pool.

        Args:
            user_id: Unique identifier
            user_vector: User's preference vector (rank,)
            metadata: Optional metadata
        """
        voter = UserVoter(
            user_id=user_id,
            user_vector=user_vector,
            basis_matrix=self.V,
            metadata=metadata,
        )
        self.voters[user_id] = voter

    def load_prism_users(
        self,
        user_vectors_path: Path | str,
        user_mapping_path: Path | str | None = None,
    ) -> None:
        """
        Load PRISM user vectors from checkpoint.

        Args:
            user_vectors_path: Path to W_lore_seen.pt
            user_mapping_path: Optional path to user_to_idx.json
        """
        W = torch.load(user_vectors_path, map_location='cpu')

        if user_mapping_path and Path(user_mapping_path).exists():
            import json
            with open(user_mapping_path, 'r') as f:
                user_to_idx = json.load(f)
            idx_to_user = {v: k for k, v in user_to_idx.items()}
        else:
            idx_to_user = {i: f"prism_user_{i}" for i in range(W.shape[0])}

        for idx in range(W.shape[0]):
            user_id = idx_to_user.get(idx, f"prism_user_{idx}")
            self.add_voter(
                user_id=user_id,
                user_vector=W[idx],
                metadata={'source': 'prism', 'idx': idx},
            )

        print(f"Loaded {W.shape[0]} PRISM user voters")

    def load_historical_users(
        self,
        historical_dir: Path | str,
    ) -> None:
        """
        Load historical user vectors from directory.

        Args:
            historical_dir: Directory containing W_historical_*.pt files
        """
        historical_dir = Path(historical_dir)

        for path in historical_dir.glob("W_historical_*.pt"):
            checkpoint = torch.load(path, map_location='cpu')
            user_id = checkpoint.get('user_id', path.stem)
            w = checkpoint['w']

            self.add_voter(
                user_id=user_id,
                user_vector=w,
                metadata={
                    'source': 'historical',
                    'century': checkpoint.get('century'),
                    'profile': checkpoint.get('user_profile'),
                },
            )

        n_historical = sum(1 for v in self.voters.values()
                         if v.metadata.get('source') == 'historical')
        print(f"Loaded {n_historical} historical user voters")

    def get_voter(self, user_id: str) -> UserVoter | None:
        """Get a specific voter by ID."""
        return self.voters.get(user_id)

    def get_all_user_ids(self) -> list[str]:
        """Get all user IDs in the pool."""
        return list(self.voters.keys())

    def get_user_metadata(self) -> dict[str, dict]:
        """Get metadata for all users."""
        return {uid: v.metadata for uid, v in self.voters.items()}

    def collect_rankings(
        self,
        embeddings: torch.Tensor,
        user_ids: list[str] | None = None,
    ) -> dict[str, list[int]]:
        """
        Collect rankings from multiple voters.

        Args:
            embeddings: (n_responses, embed_dim) tensor
            user_ids: List of user IDs to include (all if None)

        Returns:
            Dict mapping user_id -> ranking (list of indices, best first)
        """
        if user_ids is None:
            user_ids = list(self.voters.keys())

        rankings = {}
        for user_id in user_ids:
            voter = self.voters.get(user_id)
            if voter:
                rankings[user_id] = voter.rank_embeddings(embeddings)

        return rankings

    def embed_responses(
        self,
        responses: list[str],
        query: str | None = None,
    ) -> torch.Tensor:
        """
        Embed responses for scoring.

        Args:
            responses: List of response strings
            query: Optional query to prepend for context

        Returns:
            embeddings: (n_responses, embed_dim) tensor
        """
        if self.embedding_model is None:
            self.embedding_model = get_embedding_model()

        if query:
            texts = [f"{query}\n\n{r}" for r in responses]
        else:
            texts = responses

        embeddings = embed_texts(texts, model=self.embedding_model, show_progress=False)
        return torch.tensor(embeddings, dtype=torch.float32)

    @classmethod
    def from_checkpoint(
        cls,
        lore_checkpoint: Path | str,
        prism_users_path: Path | str | None = None,
        historical_dir: Path | str | None = None,
    ) -> "VoterPool":
        """
        Create a VoterPool from checkpoints.

        Args:
            lore_checkpoint: Path to LoRe model checkpoint
            prism_users_path: Optional path to PRISM user vectors
            historical_dir: Optional directory with historical user vectors

        Returns:
            Initialized VoterPool
        """
        # Load LoRe model to get V matrix
        lore_model = LoReRewardModel.load(str(lore_checkpoint), device='cpu')
        V = lore_model.V.data.clone()

        pool = cls(basis_matrix=V)

        # Load PRISM users if path provided
        if prism_users_path:
            checkpoint_dir = Path(prism_users_path).parent
            user_mapping = checkpoint_dir / "user_to_idx.json"
            pool.load_prism_users(prism_users_path, user_mapping)

        # Load historical users if directory provided
        if historical_dir:
            pool.load_historical_users(historical_dir)

        return pool
