# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Copied from https://github.com/facebookresearch/LoRe/blob/main/utils.py
# for replication purposes.

"""
Facebook LoRe implementation - copied exactly for replication.
Source: https://github.com/facebookresearch/LoRe/blob/main/utils.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_model(X, V, w):
    """
    Evaluate model accuracy on difference vectors.

    Args:
        X: Tensor of difference vectors (n_samples, embed_dim)
        V: Basis matrix (embed_dim, rank)
        w: User weight vector (rank,) - already softmax'd

    Returns:
        Fraction of samples where X @ V @ w > 0
    """
    X = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
    # Ensure all tensors are on the same device as V
    X = X.to(V.device)
    w = w.to(V.device)
    result = X @ V @ w
    num_positive = (result > 0).sum().item()
    fraction_positive = num_positive / result.numel()
    return fraction_positive


def learn_multiple_few_shot(train_features, V, num_iterations=1000, learning_rate=0.01):
    """
    Learn user weights via few-shot adaptation with frozen V.

    Args:
        train_features: List of tensors, one per user
        V: Frozen basis matrix
        num_iterations: Number of training iterations
        learning_rate: Learning rate for Adam optimizer

    Returns:
        List of softmax'd weight vectors, one per user
    """
    N = len(train_features)
    num_features = train_features[0][0].shape[0] if len(train_features[0].shape) > 1 else train_features[0].shape[0]
    fitw = PersonalizeBatch(N, num_features, V.shape[1], num_iterations, learning_rate).to(device)
    W = fitw.train(train_features, V)
    return W


def eval_multiple(W_list, V_list, test_features):
    """
    Evaluate multiple users and return per-user accuracies.

    Args:
        W_list: List of weight vectors (one per user)
        V_list: List of V matrices (one per user, typically all same)
        test_features: List of test tensors (one per user)

    Returns:
        List of accuracies (one per user)
    """
    accuracies = []
    N = len(test_features)
    accuracies = [evaluate_model(test_features[i], V_list[i], W_list[i]) for i in range(N)]
    average_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"Average accuracy: {average_accuracy:.4f}")
    print(f"Standard deviation of accuracy: {std_accuracy:.4f}")
    return accuracies


def solve_regularized(V_sft, alpha, train_features, num_basis_vectors, num_iterations=1000, learning_rate=0.01):
    """
    Train LoRe model with L2 regularization.

    Args:
        V_sft: Reference basis vector for regularization
        alpha: Regularization coefficient
        train_features: List of tensors (one per user)
        num_basis_vectors: Rank K
        num_iterations: Number of training iterations
        learning_rate: Learning rate

    Returns:
        Tuple of (W, V) where W is softmax'd weights
    """
    num_classes = len(train_features)
    num_features = train_features[0][0].shape[0] if len(train_features[0].shape) > 1 else train_features[0].shape[0]
    am = LoRe(V_sft, alpha, num_classes, num_features, num_basis_vectors, num_iterations, learning_rate)
    W, V = am.train(train_features)
    return W, V.detach()


def solve_regularized_simplex(V_sft, alpha, train_features, num_basis_vectors, num_iterations=1000, learning_rate=0.01):
    """
    Train LoRe_regularized model with cosine similarity regularization.

    Args:
        V_sft: Reference basis vector for regularization
        alpha: Regularization coefficient
        train_features: List of tensors (one per user)
        num_basis_vectors: Rank K
        num_iterations: Number of training iterations
        learning_rate: Learning rate

    Returns:
        Tuple of (W, V) where W is softmax'd weights
    """
    num_classes = len(train_features)
    num_features = 4096  # FB hardcodes this for Llama embeddings
    am = LoRe_regularized(V_sft, alpha, num_classes, num_features, num_basis_vectors, num_iterations, learning_rate)
    W, V = am.train(train_features)
    return W, V.detach()


class LoRe_regularized(nn.Module):
    """
    LoRe with cosine similarity regularization and alpha warmup.
    Uses alternating minimization between W and V.
    """

    def __init__(
        self, V_sft, alpha, num_classes, num_features, num_basis_vectors,
        num_iterations, learning_rate
    ):
        super().__init__()
        self.V_sft = V_sft.to(device)
        self.V_sft_norm = F.normalize(self.V_sft, dim=0)
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_basis_vectors = num_basis_vectors
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        self.W = nn.Parameter(torch.rand(num_classes, num_basis_vectors, device=device))
        self.V = nn.Parameter(torch.randn(num_features, num_basis_vectors, device=device))

    @staticmethod
    def _prepare_batch(X):
        """Concatenate all users' data and create index tensor."""
        x_list, y_list = [], []
        for i, x in enumerate(X):
            x_list.append(x)
            y_list.append(torch.full((x.shape[0],), i, device=x.device, dtype=torch.long))
        X_cat = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        return X_cat, y

    def _forward_from_packed(self, X_cat, y, alpha_curr):
        """Forward pass on concatenated data."""
        V_used = self.V
        W_logits = self.W

        W_row = F.softmax(W_logits, dim=1)
        Vw = V_used @ W_row.T

        logits_all = (X_cat @ Vw) / 100.0
        logits = logits_all.gather(1, y.unsqueeze(1)).squeeze(1)
        nll = -F.logsigmoid(logits).mean()

        reg = 0.0
        if alpha_curr > 0:
            V_norm = F.normalize(self.V, dim=0)
            V_sft_norm = F.normalize(self.V_sft, dim=0)
            cos_sim = (V_norm * V_sft_norm).sum(dim=0)
            reg = torch.mean(1 - cos_sim)

        entropy_loss = 0.0

        return nll, reg, entropy_loss

    def forward(self, X, alpha_curr):
        X_cat, y = self._prepare_batch(X)
        return self._forward_from_packed(X_cat, y, alpha_curr)

    def _alpha_at_step(self, step: int) -> float:
        """Linear warmup schedule for alpha."""
        warmup_start = int(0.2 * self.num_iterations)
        warmup_end = int(0.8 * self.num_iterations)
        if step < warmup_start:
            return 0.0
        if step >= warmup_end:
            return float(self.alpha)
        return float(self.alpha) * (step - warmup_start) / (warmup_end - warmup_start)

    def train(self, X):
        """Train with alternating minimization."""
        self.to(device)
        X_cat, y = self._prepare_batch(X)
        X_cat = X_cat.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer_W = optim.Adam([self.W], lr=self.learning_rate)
        optimizer_V = optim.Adam([self.V], lr=self.learning_rate)

        for step in range(self.num_iterations):
            alpha_curr = self._alpha_at_step(step)

            # Step 1: Update W (no regularization)
            optimizer_W.zero_grad()
            nll_W, _, _ = self._forward_from_packed(X_cat, y, alpha_curr=0.0)
            nll_W.backward()
            optimizer_W.step()

            # Step 2: Update V (with regularization)
            optimizer_V.zero_grad()
            nll_V, reg, _ = self._forward_from_packed(X_cat, y, alpha_curr=alpha_curr)
            total_loss_V = nll_V + alpha_curr * reg
            total_loss_V.backward()
            optimizer_V.step()

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

        # Post-training filter: remove low-activation basis vectors
        W_probs = F.softmax(self.W, dim=1)
        max_per_basis = W_probs.max(dim=0).values
        mask = (max_per_basis >= 1e-2)

        W_kept = W_probs[:, mask]
        V_kept = self.V[:, mask]
        num_kept = int(mask.sum().item())
        print(f"Num dimensions kept: {num_kept}/{self.num_basis_vectors} (threshold=1e-2)")

        print(f"W mean per dim: {W_kept.mean(dim=0).detach().cpu().numpy()}")
        print(f"W std  per dim: {W_kept.std(dim=0).detach().cpu().numpy()}")

        return W_kept, V_kept


class LoRe(nn.Module):
    """
    Basic LoRe with L2 regularization toward V_sft.
    """

    def __init__(self, V_sft, alpha, num_classes, num_features, num_basis_vectors, num_iterations, learning_rate):
        super(LoRe, self).__init__()
        self.V_sft = V_sft
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_basis_vectors = num_basis_vectors
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        self.W = nn.Parameter(torch.randn(num_classes, num_basis_vectors))
        self.V = nn.Parameter(torch.randn(num_features, num_basis_vectors))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, X):
        nll = 0
        V_w = self.V @ (F.softmax(self.W, dim=1)).T
        i = 0
        for x in X:
            logits = x @ V_w[:, i] / 100.0
            log_likelihood = torch.log(torch.sigmoid(logits))
            nll += ((-log_likelihood.sum()) / len(x))
            i += 1

        reg = 0
        if self.alpha > 0:
            for j in range(self.num_basis_vectors):
                reg += self.alpha * torch.sum((self.V[:, j] - self.V_sft) ** 2)
        return nll, reg

    def train(self, x):
        self.to(device)
        for j in range(self.num_iterations):
            self.optimizer.zero_grad()
            loss, reg = self.forward(x)
            regularized_loss = loss + reg
            regularized_loss.backward()
            self.optimizer.step()

        return (F.softmax(self.W, dim=1)), self.V


class PersonalizeBatch(nn.Module):
    """
    Few-shot personalization module.
    Learns user-specific weights with frozen V.
    """

    def __init__(self, num_classes, num_features, num_basis_vectors, num_iterations, learning_rate):
        super(PersonalizeBatch, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_basis_vectors = num_basis_vectors
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        self.w = nn.ParameterList([nn.Parameter(torch.randn(num_basis_vectors)) for _ in range(num_classes)])

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, X, V):
        nll = 0
        i = 0
        for x in X:
            V_w = V @ F.softmax(self.w[i], dim=0)
            logits = x @ V_w / 100.0
            log_likelihood = torch.log(torch.sigmoid(logits))
            nll += ((-log_likelihood.sum()) / len(x))
            i += 1
        return nll

    def train(self, X, V):
        for j in range(self.num_iterations):
            self.optimizer.zero_grad()
            loss = self.forward(X, V)
            loss.backward()
            self.optimizer.step()

        return [F.softmax(self.w[i], dim=0).detach() for i in range(len(X))]


def run_regularized(K_list, alpha_list, V_final, train_features, test_features_sparse,
                    train_features_unseen, test_features_sparse_unseen, N, N_unseen, device_str):
    """
    Full training pipeline matching FB's run_regularized().

    Args:
        K_list: List of ranks to train
        alpha_list: List of alpha values (typically just [1e4])
        V_final: V_sft reference vector
        train_features: List[Tensor] - seen users' training data
        test_features_sparse: List[Tensor] - seen users' test data
        train_features_unseen: List[Tensor] - unseen users' training data
        test_features_sparse_unseen: List[Tensor] - unseen users' test data
        N: Number of seen users
        N_unseen: Number of unseen users
        device_str: Device string

    Returns:
        Tuple of 8 arrays: (train_acc, seen_test_acc, unseen_train_acc, unseen_test_acc,
                           train_std, seen_test_std, unseen_train_std, unseen_test_std)
    """
    # Update global device to use the passed device
    global device
    device = torch.device(device_str)

    train_accuracies_joint = []
    seen_user_unseen_prompts_accuracies_joint = []
    few_shot_train_accuracies_few_shot = []
    unseen_user_unseen_prompts_accuracies_few_shot = []
    train_accuracies_joint_std = []
    seen_user_unseen_prompts_accuracies_joint_std = []
    few_shot_train_accuracies_few_shot_std = []
    unseen_user_unseen_prompts_accuracies_few_shot_std = []

    for alpha in alpha_list:
        print("alpha : ", alpha)

        for K in K_list:
            print("Rank : ", K)
            if K == 0:
                # Reference model: use V_sft directly, uniform weights
                V_joint = V_final
                W_joint = [torch.tensor([1.0]).to(device) for i in range(N)]
            else:
                # Train LoRe model
                W_joint, V_joint = solve_regularized_simplex(
                    V_final, alpha, train_features, K,
                    num_iterations=20000, learning_rate=0.5
                )

            print("Train Performance")
            accuracies_train = eval_multiple(W_joint, [V_joint.detach() for i in range(N)], train_features)
            train_accuracies_joint.append(np.mean(accuracies_train))
            train_accuracies_joint_std.append(np.std(accuracies_train))

            print("Seen User Unseen Prompts")
            accuracies_seen_user_unseen_prompts = eval_multiple(
                W_joint, [V_joint.detach() for i in range(N)], test_features_sparse
            )
            seen_user_unseen_prompts_accuracies_joint.append(np.mean(accuracies_seen_user_unseen_prompts))
            seen_user_unseen_prompts_accuracies_joint_std.append(np.std(accuracies_seen_user_unseen_prompts))

            # Few-shot for unseen users
            if K <= 1:
                # No personalization possible with rank 0 or 1
                W_few_shot = [torch.tensor([1.0]).to(device) for i in range(N_unseen)]
            else:
                W_few_shot = learn_multiple_few_shot(
                    train_features_unseen, V_joint.detach(),
                    num_iterations=500, learning_rate=0.5
                )

            print("Few Shot Train Performance")
            accuracies_few_shot_train = eval_multiple(
                W_few_shot, [V_joint.detach() for i in range(N_unseen)], train_features_unseen
            )
            few_shot_train_accuracies_few_shot.append(np.mean(accuracies_few_shot_train))
            few_shot_train_accuracies_few_shot_std.append(np.std(accuracies_few_shot_train))

            print("Unseen User Unseen Prompts")
            accuracies_unseen_user_unseen_prompts = eval_multiple(
                W_few_shot, [V_joint.detach() for i in range(N_unseen)], test_features_sparse_unseen
            )
            unseen_user_unseen_prompts_accuracies_few_shot.append(np.mean(accuracies_unseen_user_unseen_prompts))
            unseen_user_unseen_prompts_accuracies_few_shot_std.append(np.std(accuracies_unseen_user_unseen_prompts))

    fac = 0.25
    train_accuracies_joint = np.array(train_accuracies_joint)
    seen_user_unseen_prompts_accuracies_joint = np.array(seen_user_unseen_prompts_accuracies_joint)
    few_shot_train_accuracies_few_shot = np.array(few_shot_train_accuracies_few_shot)
    unseen_user_unseen_prompts_accuracies_few_shot = np.array(unseen_user_unseen_prompts_accuracies_few_shot)
    train_accuracies_joint_std = fac * np.array(train_accuracies_joint_std)
    seen_user_unseen_prompts_accuracies_joint_std = fac * np.array(seen_user_unseen_prompts_accuracies_joint_std)
    few_shot_train_accuracies_few_shot_std = fac * np.array(few_shot_train_accuracies_few_shot_std)
    unseen_user_unseen_prompts_accuracies_few_shot_std = fac * np.array(unseen_user_unseen_prompts_accuracies_few_shot_std)

    return (
        train_accuracies_joint,
        seen_user_unseen_prompts_accuracies_joint,
        few_shot_train_accuracies_few_shot,
        unseen_user_unseen_prompts_accuracies_few_shot,
        train_accuracies_joint_std,
        seen_user_unseen_prompts_accuracies_joint_std,
        few_shot_train_accuracies_few_shot_std,
        unseen_user_unseen_prompts_accuracies_few_shot_std,
    )
