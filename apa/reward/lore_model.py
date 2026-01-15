"""
LoRe: Low-rank Reward Model implementation.

Based on the paper "LoRe: Personalizing LLMs via Low-Rank Reward Modeling"
(arXiv:2504.14439) and the Facebook Research implementation.

The key idea is to decompose the reward function into:
    r(x, user) = <V @ w_user, embed(x)>

Where:
    - V is a shared basis matrix (embed_dim x rank)
    - w_user is a user-specific weight vector (rank,)
    - embed(x) is the embedding of response x

This allows learning personalized rewards with few parameters per user.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoReRewardModel(nn.Module):
    """
    Low-rank Reward Model.

    Learns a shared basis V and user-specific weights W such that:
        reward(embedding, user_idx) = embedding @ V @ W[user_idx]
    """

    def __init__(
        self,
        embed_dim: int,
        rank: int,
        n_users: int,
        alpha: float = 10000.0,
    ):
        """
        Initialize LoRe model.

        Args:
            embed_dim: Dimension of input embeddings
            rank: Rank of the low-rank decomposition (K in the paper)
            n_users: Number of users to model
            alpha: Regularization coefficient
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.rank = rank
        self.n_users = n_users
        self.alpha = alpha

        # Shared basis matrix V: (embed_dim, rank)
        # This captures the shared structure of preferences across users
        self.V = nn.Parameter(torch.randn(embed_dim, rank) * 0.01)

        # User-specific weight matrix W: (n_users, rank)
        # Each row is a user's preference vector in the basis space
        self.W = nn.Parameter(torch.randn(n_users, rank) * 0.01)

    def forward(
        self,
        embeddings: torch.Tensor,
        user_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rewards for embeddings given user indices.

        Args:
            embeddings: (batch_size, embed_dim) response embeddings
            user_indices: (batch_size,) user indices

        Returns:
            rewards: (batch_size,) reward scores
        """
        # Project embeddings to basis space: (batch_size, rank)
        projected = embeddings @ self.V

        # Get user weights: (batch_size, rank)
        user_weights = self.W[user_indices]

        # Compute rewards as dot product: (batch_size,)
        rewards = (projected * user_weights).sum(dim=-1)

        return rewards

    def compute_preference_logits(
        self,
        emb_1: torch.Tensor,
        emb_2: torch.Tensor,
        user_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute preference logits for pairs of responses.

        Uses the Bradley-Terry model: P(1 > 2) = sigmoid(r1 - r2)

        Args:
            emb_1: (batch_size, embed_dim) embeddings for response 1
            emb_2: (batch_size, embed_dim) embeddings for response 2
            user_indices: (batch_size,) user indices

        Returns:
            logits: (batch_size,) logits for preferring response 2 over response 1
        """
        r1 = self.forward(emb_1, user_indices)
        r2 = self.forward(emb_2, user_indices)

        # Logit for preferring response 2
        return r2 - r1

    def compute_loss(
        self,
        emb_1: torch.Tensor,
        emb_2: torch.Tensor,
        user_indices: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Bradley-Terry loss with regularization.

        Args:
            emb_1: (batch_size, embed_dim) embeddings for response 1
            emb_2: (batch_size, embed_dim) embeddings for response 2
            user_indices: (batch_size,) user indices
            labels: (batch_size,) binary labels (1 if response 2 preferred)

        Returns:
            loss: scalar loss value
        """
        logits = self.compute_preference_logits(emb_1, emb_2, user_indices)

        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction='mean'
        )

        # Regularization on W (user weights)
        reg_loss = self.alpha * (self.W ** 2).mean()

        return bce_loss + reg_loss

    def cosine_regularization(self, V_sft: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity regularization toward V_sft.

        Following FB LoRe: regularization = 1 - cosine_similarity(V, V_sft)
        This pulls V toward the pretrained reference direction.

        Args:
            V_sft: Reference basis matrix from pretrained model.
                   Shape should be (embed_dim, rank) or (embed_dim, 1).
                   If V_sft has fewer columns than V, we compare only the
                   first V_sft.shape[1] columns of V.

        Returns:
            Scalar regularization loss (higher = more different from V_sft)
        """
        # Normalize both V and V_sft for cosine similarity
        V_norm = F.normalize(self.V.view(-1), dim=0)

        # If V_sft has fewer columns, tile or compare first K columns
        if V_sft.shape[1] < self.rank:
            # Use only the first V_sft columns from V for comparison
            V_subset = self.V[:, :V_sft.shape[1]]
            V_norm = F.normalize(V_subset.view(-1), dim=0)

        V_sft_norm = F.normalize(V_sft.view(-1), dim=0)

        # Cosine similarity (scalar)
        cos_sim = torch.dot(V_norm, V_sft_norm)

        # Return 1 - cos_sim so that minimizing this maximizes similarity
        return 1.0 - cos_sim

    def compute_loss_alternating(
        self,
        emb_1: torch.Tensor,
        emb_2: torch.Tensor,
        user_indices: torch.Tensor,
        labels: torch.Tensor,
        V_sft: torch.Tensor | None = None,
        alpha: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute loss for FB-style alternating minimization.

        This is similar to compute_loss but:
        - Uses cosine regularization instead of L2 on W
        - Returns detailed metrics for logging
        - Alpha is passed explicitly (for warmup schedule)

        Args:
            emb_1: (batch_size, embed_dim) embeddings for response 1
            emb_2: (batch_size, embed_dim) embeddings for response 2
            user_indices: (batch_size,) user indices
            labels: (batch_size,) binary labels (1 if response 2 preferred)
            V_sft: Reference basis matrix for regularization (optional)
            alpha: Regularization coefficient (0 = no regularization)

        Returns:
            Tuple of (total_loss, metrics_dict) where metrics_dict includes:
            - bce_loss: Binary cross-entropy loss
            - reg_loss: Cosine regularization loss (or 0 if V_sft is None)
            - accuracy: Batch accuracy
        """
        logits = self.compute_preference_logits(emb_1, emb_2, user_indices)

        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction='mean'
        )

        # Cosine regularization (if V_sft provided and alpha > 0)
        if V_sft is not None and alpha > 0:
            reg_loss = self.cosine_regularization(V_sft)
            total_loss = bce_loss + alpha * reg_loss
            reg_loss_value = reg_loss.item()
        else:
            total_loss = bce_loss
            reg_loss_value = 0.0

        # Compute accuracy for logging
        with torch.no_grad():
            preds = (logits > 0).long()
            accuracy = (preds == labels).float().mean().item()

        metrics = {
            'bce_loss': bce_loss.item(),
            'reg_loss': reg_loss_value,
            'accuracy': accuracy,
        }

        return total_loss, metrics

    def get_user_reward_function(self, user_idx: int):
        """
        Get a reward function for a specific user.

        Args:
            user_idx: Index of the user

        Returns:
            Function that takes embeddings and returns rewards
        """
        def reward_fn(embeddings: torch.Tensor) -> torch.Tensor:
            user_indices = torch.full(
                (embeddings.shape[0],), user_idx, dtype=torch.long, device=embeddings.device
            )
            return self.forward(embeddings, user_indices)

        return reward_fn

    def score_responses(
        self,
        embeddings: torch.Tensor,
        user_idx: int,
    ) -> torch.Tensor:
        """
        Score multiple response embeddings for a single user.

        Args:
            embeddings: (n_responses, embed_dim) response embeddings
            user_idx: User index

        Returns:
            scores: (n_responses,) reward scores
        """
        user_indices = torch.full(
            (embeddings.shape[0],), user_idx, dtype=torch.long, device=embeddings.device
        )
        return self.forward(embeddings, user_indices)

    def rank_responses(
        self,
        embeddings: torch.Tensor,
        user_idx: int,
    ) -> list[int]:
        """
        Rank responses by reward for a user (highest first).

        Args:
            embeddings: (n_responses, embed_dim) response embeddings
            user_idx: User index

        Returns:
            ranking: List of response indices, highest reward first
        """
        scores = self.score_responses(embeddings, user_idx)
        ranking = torch.argsort(scores, descending=True).tolist()
        return ranking

    def save(self, path: str) -> None:
        """Save model to disk."""
        torch.save({
            'embed_dim': self.embed_dim,
            'rank': self.rank,
            'n_users': self.n_users,
            'alpha': self.alpha,
            'V': self.V.data,
            'W': self.W.data,
        }, path)
        print(f"Saved LoRe model to {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> "LoReRewardModel":
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            embed_dim=checkpoint['embed_dim'],
            rank=checkpoint['rank'],
            n_users=checkpoint['n_users'],
            alpha=checkpoint['alpha'],
        )
        model.V.data = checkpoint['V']
        model.W.data = checkpoint['W']
        print(f"Loaded LoRe model from {path}")
        return model

    def freeze_basis(self) -> None:
        """Freeze the shared basis V (for learning new user vectors)."""
        self.V.requires_grad = False

    def unfreeze_basis(self) -> None:
        """Unfreeze the shared basis V."""
        self.V.requires_grad = True

    def add_users(self, n_new_users: int) -> None:
        """
        Add new users to the model (for learning historical users).

        Args:
            n_new_users: Number of new users to add
        """
        old_W = self.W.data
        new_W = torch.randn(n_new_users, self.rank) * 0.01

        self.n_users += n_new_users
        self.W = nn.Parameter(torch.cat([old_W, new_W], dim=0))


class LoReTrainer:
    """
    Trainer for LoRe model.

    Handles the training loop with proper batching and optimization.
    """

    def __init__(
        self,
        model: LoReRewardModel,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize trainer.

        Args:
            model: LoRe model to train
            learning_rate: Learning rate for optimizer
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        verbose: bool = True,
    ) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader providing batches
            verbose: Whether to print progress

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            emb_1 = batch['response_1_embedding'].to(self.device)
            emb_2 = batch['response_2_embedding'].to(self.device)
            labels = batch['label'].to(self.device)

            # Handle user indices - default to 0 if not present
            if 'user_idx' in batch:
                user_indices = batch['user_idx'].to(self.device)
            else:
                user_indices = torch.zeros(emb_1.shape[0], dtype=torch.long, device=self.device)

            self.optimizer.zero_grad()
            loss = self.model.compute_loss(emb_1, emb_2, user_indices, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        return avg_loss

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: DataLoader providing batches

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                emb_1 = batch['response_1_embedding'].to(self.device)
                emb_2 = batch['response_2_embedding'].to(self.device)
                labels = batch['label'].to(self.device)

                if 'user_idx' in batch:
                    user_indices = batch['user_idx'].to(self.device)
                else:
                    user_indices = torch.zeros(emb_1.shape[0], dtype=torch.long, device=self.device)

                loss = self.model.compute_loss(emb_1, emb_2, user_indices, labels)
                total_loss += loss.item()

                # Compute accuracy
                logits = self.model.compute_preference_logits(emb_1, emb_2, user_indices)
                preds = (logits > 0).long()
                correct += (preds == labels).sum().item()
                total += labels.shape[0]

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total,
        }
