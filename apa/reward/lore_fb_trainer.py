"""
Facebook-style iteration-based trainer for LoRe.

Implements the training protocol from the FB LoRe paper:
- Alternating minimization (W then V)
- Alpha warmup schedule (linear from 0 to alpha between 20%-80% of iterations)
- Per-iteration logging
- Few-shot learning for unseen users
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from apa.reward.lore_model import LoReRewardModel


class LoReFBTrainer:
    """
    Facebook-style iteration-based trainer for LoRe.

    Key differences from the standard LoReTrainer:
    - Iteration-based instead of epoch-based
    - Alternating minimization (update W then V separately)
    - Cosine similarity regularization toward V_sft
    - Alpha warmup schedule
    """

    def __init__(
        self,
        model: LoReRewardModel,
        V_sft: torch.Tensor,
        learning_rate: float = 0.5,
        alpha: float = 10000.0,
        warmup_start: float = 0.2,
        warmup_end: float = 0.8,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize the FB-style trainer.

        Args:
            model: LoRe model to train
            V_sft: Reference basis matrix from pretrained model (for regularization)
            learning_rate: Learning rate for both optimizers
            alpha: Maximum regularization coefficient (after warmup)
            warmup_start: Fraction of iterations where warmup starts (default 0.2)
            warmup_end: Fraction of iterations where warmup ends (default 0.8)
            device: Device to train on
        """
        self.model = model.to(device)
        self.V_sft = V_sft.to(device)
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.warmup_start = warmup_start
        self.warmup_end = warmup_end
        self.device = device

        # Separate optimizers for alternating minimization
        # Following FB: use Adam with the specified learning rate
        self.optimizer_V = torch.optim.Adam([model.V], lr=learning_rate)
        self.optimizer_W = torch.optim.Adam([model.W], lr=learning_rate)

    def compute_alpha(self, iteration: int, total_iterations: int) -> float:
        """
        Compute current alpha using linear warmup schedule.

        Alpha schedule:
        - 0 to warmup_start: alpha = 0
        - warmup_start to warmup_end: linear interpolation from 0 to alpha
        - warmup_end to end: alpha = self.alpha

        Args:
            iteration: Current iteration number (0-indexed)
            total_iterations: Total number of iterations

        Returns:
            Current alpha value
        """
        start_iter = int(total_iterations * self.warmup_start)
        end_iter = int(total_iterations * self.warmup_end)

        if iteration < start_iter:
            return 0.0
        elif iteration >= end_iter:
            return self.alpha
        else:
            progress = (iteration - start_iter) / (end_iter - start_iter)
            return self.alpha * progress

    def train_iteration(
        self,
        emb_1: torch.Tensor,
        emb_2: torch.Tensor,
        user_indices: torch.Tensor,
        labels: torch.Tensor,
        iteration: int,
        total_iterations: int,
    ) -> dict[str, float]:
        """
        Perform a single training iteration with alternating minimization.

        Algorithm:
        1. Update W (user weights) with V frozen - no regularization
        2. Update V (basis) with W frozen - with cosine regularization

        Args:
            emb_1: (batch_size, embed_dim) embeddings for response 1
            emb_2: (batch_size, embed_dim) embeddings for response 2
            user_indices: (batch_size,) user indices
            labels: (batch_size,) binary labels
            iteration: Current iteration number
            total_iterations: Total number of iterations

        Returns:
            Dictionary with metrics for logging
        """
        self.model.train()
        current_alpha = self.compute_alpha(iteration, total_iterations)

        # Step 1: Update W (freeze V)
        self.model.V.requires_grad = False
        self.model.W.requires_grad = True
        self.optimizer_W.zero_grad()

        loss_w, metrics_w = self.model.compute_loss_alternating(
            emb_1, emb_2, user_indices, labels,
            V_sft=None,  # No regularization for W update
            alpha=0.0,
        )
        loss_w.backward()
        self.optimizer_W.step()

        # Step 2: Update V (freeze W)
        self.model.V.requires_grad = True
        self.model.W.requires_grad = False
        self.optimizer_V.zero_grad()

        loss_v, metrics_v = self.model.compute_loss_alternating(
            emb_1, emb_2, user_indices, labels,
            V_sft=self.V_sft,
            alpha=current_alpha,
        )
        loss_v.backward()
        self.optimizer_V.step()

        # Unfreeze both for next iteration
        self.model.V.requires_grad = True
        self.model.W.requires_grad = True

        return {
            'iteration': iteration,
            'loss_w': loss_w.item(),
            'loss_v': loss_v.item(),
            'bce_loss': metrics_v['bce_loss'],
            'reg_loss': metrics_v['reg_loss'],
            'alpha': current_alpha,
            'accuracy': metrics_v['accuracy'],
        }

    def train(
        self,
        dataloader: DataLoader,
        n_iterations: int,
        log_interval: int = 100,
        log_callback: callable | None = None,
    ) -> list[dict[str, float]]:
        """
        Train for a specified number of iterations.

        Args:
            dataloader: DataLoader providing batches
            n_iterations: Number of training iterations
            log_interval: Log every N iterations
            log_callback: Optional callback function called with metrics dict

        Returns:
            List of logged metrics (one per log_interval)
        """
        logged_metrics = []
        data_iter = self._cycling_iterator(dataloader)

        for iteration in range(n_iterations):
            batch = next(data_iter)

            emb_1 = batch['response_1_embedding'].to(self.device)
            emb_2 = batch['response_2_embedding'].to(self.device)
            user_indices = batch['user_idx'].to(self.device)
            labels = batch['label'].to(self.device)

            metrics = self.train_iteration(
                emb_1, emb_2, user_indices, labels,
                iteration=iteration,
                total_iterations=n_iterations,
            )

            if iteration % log_interval == 0 or iteration == n_iterations - 1:
                logged_metrics.append(metrics)
                if log_callback:
                    log_callback(metrics)

        return logged_metrics

    def _cycling_iterator(self, dataloader: DataLoader) -> Iterator:
        """Create an infinite iterator that cycles through the dataloader."""
        while True:
            for batch in dataloader:
                yield batch

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate accuracy on a dataset.

        Args:
            dataloader: DataLoader providing batches

        Returns:
            Dictionary with accuracy metric
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                emb_1 = batch['response_1_embedding'].to(self.device)
                emb_2 = batch['response_2_embedding'].to(self.device)
                labels = batch['label'].to(self.device)
                user_indices = batch['user_idx'].to(self.device)

                logits = self.model.compute_preference_logits(emb_1, emb_2, user_indices)
                preds = (logits > 0).long()
                correct += (preds == labels).sum().item()
                total += labels.shape[0]

        return {'accuracy': correct / total if total > 0 else 0.0}

    def fewshot_adapt(
        self,
        dataloader: DataLoader,
        n_iterations: int = 500,
        log_interval: int = 100,
        log_callback: callable | None = None,
    ) -> list[dict[str, float]]:
        """
        Few-shot adaptation for unseen users.

        Freezes V and only updates W for the users in the dataloader.
        No regularization is applied during few-shot.

        Args:
            dataloader: DataLoader with unseen user data
            n_iterations: Number of adaptation iterations
            log_interval: Log every N iterations
            log_callback: Optional callback for logging

        Returns:
            List of logged metrics
        """
        self.model.freeze_basis()
        logged_metrics = []
        data_iter = self._cycling_iterator(dataloader)

        # Create optimizer for W only (fresh optimizer for few-shot)
        fewshot_optimizer = torch.optim.Adam([self.model.W], lr=self.learning_rate)

        for iteration in range(n_iterations):
            batch = next(data_iter)

            emb_1 = batch['response_1_embedding'].to(self.device)
            emb_2 = batch['response_2_embedding'].to(self.device)
            user_indices = batch['user_idx'].to(self.device)
            labels = batch['label'].to(self.device)

            self.model.train()
            fewshot_optimizer.zero_grad()

            loss, metrics = self.model.compute_loss_alternating(
                emb_1, emb_2, user_indices, labels,
                V_sft=None,  # No regularization for few-shot
                alpha=0.0,
            )
            loss.backward()
            fewshot_optimizer.step()

            if iteration % log_interval == 0 or iteration == n_iterations - 1:
                log_entry = {
                    'iteration': iteration,
                    'loss': loss.item(),
                    'bce_loss': metrics['bce_loss'],
                    'accuracy': metrics['accuracy'],
                }
                logged_metrics.append(log_entry)
                if log_callback:
                    log_callback(log_entry)

        self.model.unfreeze_basis()
        return logged_metrics

    def reset_user_weights(self, n_users: int | None = None) -> None:
        """
        Reset user weights to random initialization.

        Useful when preparing for few-shot on a new set of users.

        Args:
            n_users: Number of users. If None, keeps current number.
        """
        if n_users is None:
            n_users = self.model.n_users

        self.model.n_users = n_users
        self.model.W = torch.nn.Parameter(
            torch.randn(n_users, self.model.rank, device=self.device) * 0.01
        )
        # Recreate W optimizer with new parameters
        self.optimizer_W = torch.optim.Adam([self.model.W], lr=self.learning_rate)
