"""
LoRe (Low-Rank Reward) model implementation.

This module implements the LoRe reward model for personalized preference learning,
following the Meta AI LoRe paper methodology.

Key components:
- LoReRewardModel: Holds the V basis matrix and provides scoring
- LoReTrainer: Implements alternating minimization training (LoRe_regularized)
- PersonalizeBatch: Few-shot learning for new users given fixed V
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_device() -> torch.device:
    """Get the default device (CUDA if available)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LoReRewardModel:
    """
    LoRe reward model for scoring responses based on learned preferences.

    The model uses a low-rank decomposition where the reward is computed as:
        reward(x) = x @ V @ w

    where V is the shared basis matrix and w is the user-specific weight vector.
    """

    def __init__(self, V: torch.Tensor):
        """
        Initialize the model.

        Args:
            V: Basis matrix of shape (embedding_dim, rank)
        """
        self.V = V

    @classmethod
    def load(cls, checkpoint_path: str, device: str = 'cpu') -> "LoReRewardModel":
        """
        Load a LoRe model from checkpoint.

        Args:
            checkpoint_path: Path to the V matrix checkpoint (.pt file)
            device: Device to load the model on

        Returns:
            Loaded LoReRewardModel instance
        """
        V = torch.load(checkpoint_path, map_location=device)
        if isinstance(V, dict):
            # Handle case where checkpoint is a dict with 'V' key
            V = V.get('V', V.get('basis_matrix', V))
        return cls(V)

    def score(self, embedding: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Score an embedding using user weights.

        Args:
            embedding: Response embedding(s) of shape (embed_dim,) or (n, embed_dim)
            w: User weight vector of shape (rank,)

        Returns:
            Score(s) of shape () or (n,)
        """
        V = self.V.to(embedding.device)
        w = w.to(embedding.device)
        return embedding @ V @ w

    @property
    def rank(self) -> int:
        """Get the rank (number of basis vectors)."""
        return self.V.shape[1]

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.V.shape[0]


class LoReTrainer(nn.Module):
    """
    LoRe trainer implementing alternating minimization.

    This follows the LoRe_regularized implementation from the original codebase.
    Key features:
    - Alternating optimization for W (user weights) and V (basis)
    - Alpha warmup from 20% to 80% of training
    - Cosine similarity regularization toward reference model V_sft
    - Dimension filtering based on softmax threshold
    """

    def __init__(
        self,
        V_sft: torch.Tensor,
        alpha: float,
        num_classes: int,
        num_features: int,
        num_basis_vectors: int,
        num_iterations: int,
        learning_rate: float,
        logits_scale: float = 100.0,
        threshold: float = 1e-2,
        logger: Any = None,
        log_interval: int = 1000,
    ):
        """
        Initialize the trainer.

        Args:
            V_sft: Reference model weights (embedding_dim, 1) - from Skywork last layer
            alpha: Regularization strength
            num_classes: Number of users
            num_features: Embedding dimension (4096 for Llama 3.1 8B)
            num_basis_vectors: Rank K of the decomposition
            num_iterations: Number of training iterations
            learning_rate: Learning rate for Adam optimizer
            logits_scale: Division factor for logits in NLL loss (default 100.0)
            threshold: Threshold for dimension filtering (default 1e-2)
            logger: Optional logger for diagnostics
            log_interval: How often to log during training
        """
        super().__init__()
        device = get_device()

        self.V_sft = V_sft.to(device)
        self.V_sft_norm = F.normalize(self.V_sft, dim=0)
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_basis_vectors = num_basis_vectors
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.logits_scale = logits_scale
        self.threshold = threshold
        self.logger = logger
        self.log_interval = log_interval

        # Training diagnostics storage
        self.training_history = {
            "steps": [],
            "nll_W": [],
            "nll_V": [],
            "reg": [],
            "alpha_curr": [],
            "grad_norm_W": [],
            "grad_norm_V": [],
        }

        # Initialize parameters
        self.W = nn.Parameter(torch.rand(num_classes, num_basis_vectors, device=device))
        self.V = nn.Parameter(torch.randn(num_features, num_basis_vectors, device=device))

    @staticmethod
    def _prepare_batch(X: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch from list of per-user feature tensors.

        Args:
            X: List of length C; X[i] is [m_i, F] tensor for user i

        Returns:
            X_cat: [N, F] concatenated features
            y: [N] class labels (values in 0..C-1)
        """
        x_list, y_list = [], []
        for i, x in enumerate(X):
            x_list.append(x)
            y_list.append(torch.full((x.shape[0],), i, device=x.device, dtype=torch.long))
        X_cat = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        return X_cat, y

    def _forward_from_packed(
        self,
        X_cat: torch.Tensor,
        y: torch.Tensor,
        alpha_curr: float,
    ) -> tuple[torch.Tensor, float, float]:
        """
        Forward pass on packed batch.

        Args:
            X_cat: [N, F] concatenated features
            y: [N] class labels
            alpha_curr: Current alpha for regularization

        Returns:
            nll: Negative log likelihood loss
            reg: Regularization term
            entropy_loss: (unused, for compatibility)
        """
        W_row = F.softmax(self.W, dim=1)    # [C, B]
        Vw = self.V @ W_row.T               # [F, C]

        logits_all = (X_cat @ Vw) / self.logits_scale  # [N, C]
        logits = logits_all.gather(1, y.unsqueeze(1)).squeeze(1)
        nll = -F.logsigmoid(logits).mean()

        # Cosine similarity regularization toward reference
        reg = 0.0
        if alpha_curr > 0:
            V_norm = F.normalize(self.V, dim=0)
            V_sft_norm = F.normalize(self.V_sft, dim=0)
            cos_sim = (V_norm * V_sft_norm).sum(dim=0)
            reg = torch.mean(1 - cos_sim)

        return nll, reg, 0.0

    def _alpha_at_step(self, step: int) -> float:
        """
        Compute alpha with warmup schedule.

        Alpha is 0 for first 20% of training, then linearly increases
        to full alpha value at 80% of training.
        """
        warmup_start = int(0.2 * self.num_iterations)
        warmup_end = int(0.8 * self.num_iterations)
        if step < warmup_start:
            return 0.0
        if step >= warmup_end:
            return float(self.alpha)
        return float(self.alpha) * (step - warmup_start) / (warmup_end - warmup_start)

    def _log(self, msg: str) -> None:
        """Log message using logger if available, otherwise print."""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def train_model(self, X: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Train the model using alternating minimization.

        Args:
            X: List of per-user feature tensors, X[i] is [m_i, F]
               where m_i is number of preference pairs for user i

        Returns:
            W_kept: User weight matrix after filtering, shape [C, B_kept]
            V_kept: Basis matrix after filtering, shape [F, B_kept]
        """
        device = get_device()
        self.to(device)

        X_cat, y = self._prepare_batch(X)
        X_cat = X_cat.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer_W = optim.Adam([self.W], lr=self.learning_rate)
        optimizer_V = optim.Adam([self.V], lr=self.learning_rate)

        # Clear training history
        for key in self.training_history:
            self.training_history[key] = []

        for step in range(self.num_iterations):
            alpha_curr = self._alpha_at_step(step)

            # Update W: freeze V
            optimizer_W.zero_grad()
            nll_W, _, _ = self._forward_from_packed(X_cat, y, alpha_curr=0.0)
            nll_W.backward()
            grad_norm_W = self.W.grad.norm().item() if self.W.grad is not None else 0.0
            optimizer_W.step()

            # Update V: freeze W
            optimizer_V.zero_grad()
            nll_V, reg, _ = self._forward_from_packed(X_cat, y, alpha_curr=alpha_curr)
            total_loss_V = nll_V + alpha_curr * reg
            total_loss_V.backward()
            grad_norm_V = self.V.grad.norm().item() if self.V.grad is not None else 0.0
            optimizer_V.step()

            # Log diagnostics at intervals
            if step % self.log_interval == 0 or step == self.num_iterations - 1:
                self.training_history["steps"].append(step)
                self.training_history["nll_W"].append(nll_W.item())
                self.training_history["nll_V"].append(nll_V.item())
                self.training_history["reg"].append(float(reg))
                self.training_history["alpha_curr"].append(alpha_curr)
                self.training_history["grad_norm_W"].append(grad_norm_W)
                self.training_history["grad_norm_V"].append(grad_norm_V)

                if self.logger and step % self.log_interval == 0:
                    self._log(
                        f"  Step {step:5d}/{self.num_iterations}: "
                        f"NLL={nll_V.item():.4f}, Reg={float(reg):.4f}, "
                        f"Alpha={alpha_curr:.1f}, "
                        f"||grad_W||={grad_norm_W:.4f}, ||grad_V||={grad_norm_V:.4f}"
                    )

            if (step + 1) == self.num_iterations:
                W_sm = F.softmax(self.W, dim=1)
                print(f"W mean per dim: {W_sm.mean(dim=0).detach().cpu().numpy()}")
                print(f"W std  per dim: {W_sm.std(dim=0).detach().cpu().numpy()}")
                with torch.no_grad():
                    V_param_norms = torch.linalg.vector_norm(self.V, ord=2, dim=0)
                print(f"||V[:, i]|| (param): {V_param_norms.detach().cpu().numpy()}")
                print(
                    f"Step {step}: "
                    f"NLL(W)={nll_W.item():.4f}, "
                    f"NLL(V)={nll_V.item():.4f}, "
                    f"Reg={float(reg):.4f}, "
                    f"Alpha={alpha_curr:.4f}, "
                )

        # Filter dimensions based on threshold
        W_probs = F.softmax(self.W, dim=1)           # [C, B]
        max_per_basis = W_probs.max(dim=0).values    # [B]
        print(max_per_basis)
        mask = (max_per_basis >= self.threshold)     # bool[B]

        W_kept = W_probs[:, mask]                    # [C, B_kept]
        V_kept = self.V[:, mask]                     # [F, B_kept]
        num_kept = int(mask.sum().item())
        print(f"Num dimensions kept: {num_kept}/{self.num_basis_vectors} (threshold={self.threshold})")

        print(f"W mean per dim: {W_kept.mean(dim=0).detach().cpu().numpy()}")
        print(f"W std  per dim: {W_kept.std(dim=0).detach().cpu().numpy()}")

        return W_kept, V_kept


class PersonalizeBatch(nn.Module):
    """
    Few-shot personalization for new users with fixed V basis.

    Given a fixed basis matrix V, learns user-specific weight vectors w
    from a small number of preference examples.
    """

    def __init__(
        self,
        num_classes: int,
        num_features: int,
        num_basis_vectors: int,
        num_iterations: int,
        learning_rate: float,
        logits_scale: float = 100.0,
    ):
        """
        Initialize few-shot learner.

        Args:
            num_classes: Number of users to personalize
            num_features: Embedding dimension
            num_basis_vectors: Rank of the basis (from V.shape[1])
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for Adam
            logits_scale: Division factor for logits
        """
        super().__init__()
        device = get_device()

        self.num_classes = num_classes
        self.num_features = num_features
        self.num_basis_vectors = num_basis_vectors
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.logits_scale = logits_scale

        # Initialize user weight vectors
        self.w = nn.ParameterList([
            nn.Parameter(torch.randn(num_basis_vectors, device=device))
            for _ in range(num_classes)
        ])

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, X: list[torch.Tensor], V: torch.Tensor) -> torch.Tensor:
        """
        Compute NLL loss for all users.

        Args:
            X: List of per-user feature tensors
            V: Fixed basis matrix [F, B]

        Returns:
            Total NLL loss
        """
        nll = 0
        for i, x in enumerate(X):
            V_w = V @ F.softmax(self.w[i], dim=0)
            # Ensure x is on same device as V
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=V.device)
            elif x.device != V.device:
                x = x.to(V.device)
            logits = x @ V_w / self.logits_scale
            log_likelihood = torch.log(torch.sigmoid(logits))
            nll += ((-log_likelihood.sum()) / len(x))
        return nll

    def train_model(
        self,
        X: list[torch.Tensor],
        V: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Train user weights with fixed V.

        Args:
            X: List of per-user feature tensors
            V: Fixed basis matrix

        Returns:
            List of learned user weight vectors (softmax applied)
        """
        for j in range(self.num_iterations):
            self.optimizer.zero_grad()
            loss = self.forward(X, V)
            loss.backward()
            self.optimizer.step()

        return [F.softmax(self.w[i], dim=0).detach() for i in range(len(X))]


def evaluate_model(
    X: list[torch.Tensor] | torch.Tensor | np.ndarray,
    V: torch.Tensor,
    w: torch.Tensor,
) -> float:
    """
    Evaluate model accuracy on preference pairs.

    Args:
        X: Feature tensor(s) representing (chosen - rejected) embeddings
           Can be a list of tensors or a single tensor
        V: Basis matrix [F, B]
        w: User weight vector [B]

    Returns:
        Fraction of positive predictions (accuracy)
    """
    if isinstance(X, list):
        X = torch.cat(X, dim=0)
    X = torch.tensor(X, dtype=torch.float32, device=V.device)
    result = X @ V @ w
    num_positive = (result > 0).sum().item()
    fraction_positive = num_positive / result.numel()
    return fraction_positive


def eval_multiple(
    W_list: list[torch.Tensor],
    V_list: list[torch.Tensor],
    test_features: list[torch.Tensor],
) -> list[float]:
    """
    Evaluate accuracy for multiple users.

    Args:
        W_list: List of user weight vectors
        V_list: List of V matrices (usually all the same)
        test_features: List of per-user test feature tensors

    Returns:
        List of accuracies per user
    """
    N = len(test_features)
    accuracies = [
        evaluate_model(test_features[i], V_list[i], W_list[i])
        for i in range(N)
    ]
    average_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"Average accuracy: {average_accuracy:.4f}")
    print(f"Standard deviation of accuracy: {std_accuracy:.4f}")
    return accuracies


def learn_multiple_few_shot(
    train_features: list[torch.Tensor],
    V: torch.Tensor,
    num_iterations: int = 500,
    learning_rate: float = 0.5,
) -> list[torch.Tensor]:
    """
    Learn user weights for multiple users with few-shot data.

    Args:
        train_features: List of per-user training feature tensors
        V: Fixed basis matrix
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate

    Returns:
        List of learned user weight vectors
    """
    device = get_device()
    N = len(train_features)
    num_features = train_features[0].shape[-1] if len(train_features) > 0 else V.shape[0]

    fitw = PersonalizeBatch(
        N, num_features, V.shape[1], num_iterations, learning_rate
    ).to(device)
    W = fitw.train_model(train_features, V)
    return W
