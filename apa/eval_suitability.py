"""
LoRe dataset suitability evaluation.

This module provides a spectrum of diagnostic metrics — from cheap heuristics to
full LoRe fitting — that predict how well LoRe will learn distinct, predictive
user representations for a new dataset.

Assumed inputs for a new dataset:
  1. Raw user preferences (prompt + chosen + rejected text, grouped by user_id)
  2. Pretrained LoRe bases V (Tensor[embed_dim, K])
  3. The reward model (for embedding computation)

Usage::

    from apa.eval_suitability import embed_preferences, evaluate_suitability
    import torch

    # Step 1: embed raw preferences (skip if you already have embeddings)
    user_pref_embeddings = embed_preferences(user_prefs, model, tokenizer)

    # Step 2: run all metrics
    V = torch.load("models/V_K8.pt")
    results = evaluate_suitability(user_pref_embeddings, V=V)

For PRISM, pre-computed embeddings are available via group_embeddings_by_user()
in load_prism.py — pass the result directly to evaluate_suitability().

Metric tiers
------------
Tier 0 — raw text only (milliseconds):
    annotation_density, prompt_diversity_surface

Tier 1 — embeddings required (minutes, GPU forward pass):
    label_balance, effective_rank, inter_user_agreement, krippendorff_alpha_proxy

Tier 3 — embeddings + pretrained V required:
    basis_activation_variance, fit_user_vectors, fit_quality,
    user_vector_diversity, basis_utilization_entropy

Tier 5 — embeddings + V + held-out splits:
    held_out_accuracy, silhouette_score_metric
"""

from __future__ import annotations

import math
from collections import namedtuple
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

PreferencePair = namedtuple("PreferencePair", ["prompt", "chosen", "rejected"])
"""A single pairwise preference: prompt text, chosen response text, rejected response text."""


# ---------------------------------------------------------------------------
# Tier 0 — raw text metrics (no model needed)
# ---------------------------------------------------------------------------

def annotation_density(
    user_pref_embeddings: list[torch.Tensor],
    K: int,
) -> dict:
    """
    Check whether users have enough preference pairs to reliably fit a K-dim user vector.

    Rule of thumb: a user needs at least 2*K pairs to constrain a K-dimensional vector.

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors.
        K: Rank (number of basis vectors) of the LoRe model.

    Returns:
        Dict with per-user counts and a warning flag.
    """
    counts = [len(u) for u in user_pref_embeddings]
    median_count = float(np.median(counts))
    fraction_below = float(np.mean([c < 2 * K for c in counts]))
    return {
        "n_users": len(counts),
        "min_pairs": int(min(counts)),
        "median_pairs": median_count,
        "mean_pairs": float(np.mean(counts)),
        "fraction_below_2K": fraction_below,
        "warn": median_count < 2 * K,
    }


def prompt_diversity_surface(
    user_prefs: dict[str, list[PreferencePair]],
) -> dict:
    """
    Count unique prompts across the dataset (surface-level, no embedding).

    Low diversity means most users answer the same few questions — user vectors
    will reflect idiosyncratic question reactions, not generalizable preferences.

    Args:
        user_prefs: Mapping from user_id to list of PreferencePair.

    Returns:
        Dict with unique prompt counts and per-user statistics.
    """
    all_prompts: list[str] = []
    per_user_unique: list[int] = []

    for pairs in user_prefs.values():
        prompts = [p.prompt for p in pairs]
        all_prompts.extend(prompts)
        per_user_unique.append(len(set(prompts)))

    n_total_pairs = len(all_prompts)
    n_unique_prompts = len(set(all_prompts))

    return {
        "n_unique_prompts": n_unique_prompts,
        "n_total_pairs": n_total_pairs,
        "prompt_reuse_rate": 1.0 - n_unique_prompts / max(n_total_pairs, 1),
        "mean_unique_prompts_per_user": float(np.mean(per_user_unique)) if per_user_unique else 0.0,
        "median_unique_prompts_per_user": float(np.median(per_user_unique)) if per_user_unique else 0.0,
    }


# ---------------------------------------------------------------------------
# Tier 1 — embedding-based metrics (no V needed)
# ---------------------------------------------------------------------------

def label_balance(user_pref_embeddings: list[torch.Tensor]) -> dict:
    """
    Measure per-user preference direction consistency.

    In the pairwise setting, every example is (chosen - rejected), always
    "positive-label". Consistency is ||mean(pref_vecs)|| / mean(||pref_vecs||).

    - Near 0: user preferences cancel out — either random or strongly varied.
    - Near 1: user always prefers responses in the same embedding direction
              (rubber-stamper, or one dominant preference axis).

    Both extremes are diagnostic: near-0 means the user provides no learnable
    signal; near-1 may mean there is little personalisation to do.

    Returns:
        Dict with per-user consistency scores and population statistics.
    """
    scores = []
    for X in user_pref_embeddings:
        X = X.float()
        mean_vec = X.mean(dim=0)
        mean_norm = mean_vec.norm().item()
        mean_of_norms = X.norm(dim=1).mean().item()
        consistency = mean_norm / (mean_of_norms + 1e-12)
        scores.append(consistency)

    scores_arr = np.array(scores)
    return {
        "mean_consistency": float(scores_arr.mean()),
        "std_consistency": float(scores_arr.std()),
        "median_consistency": float(np.median(scores_arr)),
        "fraction_near_zero": float(np.mean(scores_arr < 0.05)),
        "fraction_near_one": float(np.mean(scores_arr > 0.95)),
        "per_user_consistency": scores,
    }


def effective_rank(
    user_pref_embeddings: list[torch.Tensor],
    threshold: float = 0.90,
) -> dict:
    """
    Estimate the effective rank of the user preference space.

    Builds a [n_users, D] matrix of per-user mean preference vectors and
    computes its SVD. The effective rank is the minimum number of components
    needed to explain `threshold` fraction of variance.

    Low effective rank means user preferences lie in a low-dimensional
    subspace — exactly the structure LoRe assumes.

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors.
        threshold: Cumulative variance fraction to reach (default 0.90).

    Returns:
        Dict with effective rank, singular values, and compression ratio.
    """
    means = torch.stack([X.float().mean(dim=0) for X in user_pref_embeddings])  # [n_users, D]
    # Center
    means = means - means.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(means, full_matrices=False)
    S = S.cpu().float()

    total_var = (S ** 2).sum().item()
    cumvar = (S ** 2).cumsum(dim=0) / (total_var + 1e-12)
    n_components = int((cumvar < threshold).sum().item()) + 1
    n_users = len(user_pref_embeddings)

    return {
        "effective_rank": n_components,
        "variance_threshold": threshold,
        "compression_ratio": n_components / max(n_users, 1),
        "n_users": n_users,
        "singular_values": S.tolist(),
    }


def inter_user_agreement(user_pref_embeddings: list[torch.Tensor]) -> dict:
    """
    Measure pairwise agreement between users via cosine similarity of their
    mean preference vectors.

    High mean similarity: users mostly agree — little room for personalisation.
    Low mean similarity with high variance: mixed bag, some learnable structure.
    Very low mean similarity: users disagree broadly — LoRe must work hard.

    Returns:
        Dict with off-diagonal similarity statistics and clustering proxy.
    """
    means = torch.stack([X.float().mean(dim=0) for X in user_pref_embeddings])
    means_norm = F.normalize(means, dim=1)
    sim = means_norm @ means_norm.T  # [n_users, n_users]

    n = len(user_pref_embeddings)
    mask = ~torch.eye(n, dtype=torch.bool)
    off_diag = sim[mask].cpu().numpy()

    return {
        "mean_pairwise_similarity": float(off_diag.mean()),
        "std_pairwise_similarity": float(off_diag.std()),
        "min_pairwise_similarity": float(off_diag.min()),
        "max_pairwise_similarity": float(off_diag.max()),
        "fraction_high_agreement": float(np.mean(off_diag > 0.5)),
    }


def krippendorff_alpha_proxy(user_pref_embeddings: list[torch.Tensor]) -> dict:
    """
    Proxy for Krippendorff's alpha inter-annotator reliability.

    True Krippendorff's alpha requires annotators to rate the same items.
    Since different users in PRISM-style datasets answer different prompts,
    this function instead computes the fraction of total preference variance
    that is attributable to user identity:

        variance_ratio = between_user_variance / total_variance

    where "variance" is measured in the full embedding space (averaged over
    dimensions).

    - High ratio (> ~0.1): user identity explains a large fraction of preference
      variation — users are genuinely distinct, LoRe can learn to separate them.
    - Low ratio (< ~0.01): user-to-user differences are small relative to
      response-to-response noise — personalisation may not add much.

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors.

    Returns:
        Dict with variance ratio and raw variance values.
    """
    all_prefs = torch.cat([X.float() for X in user_pref_embeddings], dim=0)  # [N, D]
    grand_mean = all_prefs.mean(dim=0)  # [D]

    user_means = torch.stack([X.float().mean(dim=0) for X in user_pref_embeddings])  # [n_users, D]
    between_var = ((user_means - grand_mean) ** 2).mean().item()
    total_var = ((all_prefs - grand_mean) ** 2).mean().item()

    variance_ratio = between_var / (total_var + 1e-12)

    return {
        "variance_ratio": variance_ratio,
        "between_user_variance": between_var,
        "total_variance": total_var,
        "within_user_variance": total_var - between_var,
    }


# ---------------------------------------------------------------------------
# Tier 3 — metrics requiring pretrained V
# ---------------------------------------------------------------------------

def basis_activation_variance(
    user_pref_embeddings: list[torch.Tensor],
    V: torch.Tensor,
) -> dict:
    """
    Check how much each pretrained basis dimension activates on the new dataset.

    Projects all preference embeddings through V: activations = X @ V.
    A basis with near-zero variance across the dataset is "dead" — it doesn't
    respond to any preference variation in the new domain.

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors.
        V: Pretrained basis matrix [D, K].

    Returns:
        Dict with per-basis variances and count of active bases.
    """
    V = V.float()
    all_prefs = torch.cat([X.float() for X in user_pref_embeddings], dim=0)  # [N, D]
    activations = all_prefs @ V  # [N, K]

    variances = activations.var(dim=0).cpu()  # [K]
    max_var = variances.max().item()
    epsilon = 0.01 * max_var if max_var > 0 else 1e-8
    n_active = int((variances > epsilon).sum().item())

    return {
        "per_basis_variance": variances.tolist(),
        "n_active_bases": n_active,
        "n_total_bases": V.shape[1],
        "fraction_active": n_active / V.shape[1],
        "max_variance": max_var,
        "min_variance": variances.min().item(),
    }


def fit_user_vectors(
    user_pref_embeddings: list[torch.Tensor],
    V: torch.Tensor,
) -> torch.Tensor:
    """
    Fit per-user weight vectors via closed-form least squares.

    For each user, solves:  argmin_w ||XV @ w - 1||²
    where XV = X @ V is the projection of preference embeddings into basis space.

    This is a fast closed-form approximation of PersonalizeBatch (which optimises
    NLL via gradient descent). The approximation is sufficient for diagnostic
    metrics; use PersonalizeBatch from train_lore_bases.py for production fitting.

    Users with fewer than 2 pairs are given zero vectors.

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors.
        V: Pretrained basis matrix [D, K].

    Returns:
        W: Tensor of shape [n_users, K] — raw (pre-softmax) user weight vectors.
    """
    V = V.float()
    K = V.shape[1]
    user_ws = []

    for X in user_pref_embeddings:
        X = X.float()
        if len(X) < 2:
            user_ws.append(torch.zeros(K))
            continue
        XV = X @ V  # [n_prefs, K]
        target = torch.ones(len(XV))
        # Closed-form least squares: w = (XV^T XV)^{-1} XV^T 1
        result = torch.linalg.lstsq(XV, target)
        user_ws.append(result.solution.cpu())

    return torch.stack(user_ws)  # [n_users, K]


def fit_quality(
    user_pref_embeddings: list[torch.Tensor],
    V: torch.Tensor,
) -> dict:
    """
    Measure how well closed-form user vectors explain training preferences.

    For each user, computes:
    - accuracy: fraction of pairs where (XV @ w) > 0 (correct preference direction)
    - R²: fraction of variance in the target explained by the linear model

    Users with fewer than 2 pairs are skipped.

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors.
        V: Pretrained basis matrix [D, K].

    Returns:
        Dict with mean/std accuracy and R² across users.
    """
    V = V.float()
    accuracies = []
    r2s = []

    for X in user_pref_embeddings:
        X = X.float()
        if len(X) < 2:
            continue
        XV = X @ V
        target = torch.ones(len(XV))
        result = torch.linalg.lstsq(XV, target)
        w = result.solution
        predictions = XV @ w

        acc = (predictions > 0).float().mean().item()
        accuracies.append(acc)

        ss_res = ((target - predictions) ** 2).sum().item()
        ss_tot = ((target - target.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        r2s.append(r2)

    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_r2": float(np.mean(r2s)),
        "std_r2": float(np.std(r2s)),
        "n_users_evaluated": len(accuracies),
    }


def user_vector_diversity(W: torch.Tensor) -> dict:
    """
    Measure how spread out the fitted user vectors are in basis space.

    High diversity means LoRe has learned to distinguish users. Low diversity
    means all users have similar weights — personalisation isn't helping.

    Args:
        W: Raw user weight matrix [n_users, K] from fit_user_vectors().

    Returns:
        Dict with mean pairwise distance, effective rank of user vector space,
        and eigenvalue spectrum of the user vector covariance.
    """
    W_soft = F.softmax(W.float(), dim=1)  # [n_users, K]
    W_norm = F.normalize(W_soft, dim=1)
    sim = W_norm @ W_norm.T  # [n_users, n_users]

    n = W.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    off_diag_sim = sim[mask].cpu().numpy()
    mean_pairwise_distance = 1.0 - float(off_diag_sim.mean())

    # Effective rank of user vector covariance
    cov = torch.cov(W_soft.T)  # [K, K]
    eigenvalues = torch.linalg.eigvalsh(cov).cpu().float()
    eigenvalues = eigenvalues.clamp(min=0)
    max_eig = eigenvalues.max().item()
    eff_rank = int((eigenvalues > 0.01 * max_eig).sum().item()) if max_eig > 0 else 0

    return {
        "mean_pairwise_distance": mean_pairwise_distance,
        "std_pairwise_distance": float(off_diag_sim.std()),
        "effective_rank": eff_rank,
        "eigenvalues": eigenvalues.tolist(),
    }


def basis_utilization_entropy(W: torch.Tensor) -> dict:
    """
    Measure how uniformly users spread their weight across the K bases.

    High entropy (close to log K): users leverage many different bases —
    the full rank is being utilised.
    Low entropy: most users concentrate on 1-2 bases — effective rank is low,
    and the pretrained bases may not cover the new domain's preference dimensions.

    Args:
        W: Raw user weight matrix [n_users, K] from fit_user_vectors().

    Returns:
        Dict with mean entropy, max possible entropy (log K), and normalised mean.
    """
    W_soft = F.softmax(W.float(), dim=1)  # [n_users, K]
    # Shannon entropy per user: -sum(w_i * log(w_i))
    entropies = -(W_soft * (W_soft + 1e-10).log()).sum(dim=1).cpu().numpy()
    max_entropy = math.log(W.shape[1]) if W.shape[1] > 1 else 1.0

    return {
        "mean_entropy": float(entropies.mean()),
        "std_entropy": float(entropies.std()),
        "max_entropy": max_entropy,
        "normalized_mean_entropy": float(entropies.mean()) / max_entropy,
        "per_user_entropy": entropies.tolist(),
    }


# ---------------------------------------------------------------------------
# Tier 5 — metrics requiring held-out splits and/or full fitting
# ---------------------------------------------------------------------------

def held_out_accuracy(
    user_pref_embeddings: list[torch.Tensor],
    V: torch.Tensor,
    test_frac: float = 0.2,
) -> dict:
    """
    Cross-validate closed-form user vector fitting against held-out preferences.

    For each user, holds out the last `test_frac` fraction of pairs, fits a
    user vector on the rest, and evaluates accuracy on the held-out pairs.
    Users with fewer than 4 pairs are skipped (need at least 1 train + 1 test).

    This is the most faithful fast proxy for what LoRe will achieve in
    production few-shot adaptation.

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors.
        V: Pretrained basis matrix [D, K].
        test_frac: Fraction of pairs to hold out for evaluation (default 0.2).

    Returns:
        Dict with mean/std held-out accuracy and number of users evaluated.
    """
    V = V.float()
    accuracies = []

    for X in user_pref_embeddings:
        X = X.float()
        n = len(X)
        if n < 4:
            continue
        n_test = max(1, int(n * test_frac))
        n_train = n - n_test
        X_train, X_test = X[:n_train], X[n_train:]

        XV_train = X_train @ V
        target_train = torch.ones(n_train)
        result = torch.linalg.lstsq(XV_train, target_train)
        w = result.solution

        XV_test = X_test @ V
        acc = (XV_test @ w > 0).float().mean().item()
        accuracies.append(acc)

    return {
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
        "std_accuracy": float(np.std(accuracies)) if accuracies else float("nan"),
        "n_users_evaluated": len(accuracies),
    }


def silhouette_score_metric(
    W: torch.Tensor,
    n_clusters: int | None = None,
) -> dict:
    """
    Measure how cleanly fitted user vectors cluster in basis space.

    A positive silhouette score means users form distinct groups in the learned
    representation space — LoRe has separated them meaningfully. A score near
    zero or negative means user vectors are spread uniformly or interleaved.

    Args:
        W: Raw user weight matrix [n_users, K] from fit_user_vectors().
        n_clusters: Number of K-means clusters. Defaults to max(2, round(sqrt(n)/2)).

    Returns:
        Dict with silhouette score and the number of clusters used.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    W_soft = F.softmax(W.float(), dim=1).cpu().numpy()
    n = W_soft.shape[0]

    if n_clusters is None:
        n_clusters = max(2, round(math.sqrt(n) / 2))
    n_clusters = min(n_clusters, n - 1)  # can't have more clusters than samples

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(W_soft)
    score = silhouette_score(W_soft, labels)

    return {
        "silhouette_score": float(score),
        "n_clusters_used": n_clusters,
    }


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed_preferences(
    user_prefs: dict[str, list[PreferencePair]],
    model: Any,
    tokenizer: Any,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """
    Embed raw user preferences using the reward model.

    For each pair, embeds f(prompt + chosen) and f(prompt + rejected) and
    stores the difference (chosen - rejected) as the preference vector.
    This matches the representation used during LoRe training.

    Args:
        user_prefs: Mapping from user_id to list of PreferencePair.
        model: Skywork-Reward (or compatible) model with hidden_states output.
        tokenizer: Corresponding tokenizer.
        device: Device string for inference.

    Returns:
        Per-user list of [n_prefs, D] float32 tensors (sorted by user_id).
    """
    from apa.train_lore_bases import _format_for_embedding, _extract_embedding

    result = []
    for user_id in sorted(user_prefs.keys()):
        pairs = user_prefs[user_id]
        diffs = []
        for pair in pairs:
            chosen_text = _format_for_embedding(pair.prompt, pair.chosen, tokenizer)
            rejected_text = _format_for_embedding(pair.prompt, pair.rejected, tokenizer)
            chosen_emb = torch.tensor(
                _extract_embedding(model, tokenizer, chosen_text, device),
                dtype=torch.float32,
            )
            rejected_emb = torch.tensor(
                _extract_embedding(model, tokenizer, rejected_text, device),
                dtype=torch.float32,
            )
            diffs.append(chosen_emb - rejected_emb)
        if diffs:
            result.append(torch.stack(diffs))
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_suitability(
    user_pref_embeddings: list[torch.Tensor],
    V: torch.Tensor | None = None,
    K: int = 8,
    user_prefs: dict[str, list[PreferencePair]] | None = None,
) -> dict:
    """
    Run all applicable suitability metrics and print a summary table.

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors (required).
        V: Pretrained LoRe basis [D, K]. If None, only Tier 0/1 metrics run.
        K: Rank value for annotation_density threshold check.
        user_prefs: Raw preference pairs (dict[user_id, list[PreferencePair]]).
                    Required only for prompt_diversity_surface.

    Returns:
        Flat dict of all computed metric results.
    """
    results: dict = {}

    # --- Tier 0 ---
    results["annotation_density"] = annotation_density(user_pref_embeddings, K)
    if user_prefs is not None:
        results["prompt_diversity"] = prompt_diversity_surface(user_prefs)

    # --- Tier 1 ---
    results["label_balance"] = label_balance(user_pref_embeddings)
    results["effective_rank"] = effective_rank(user_pref_embeddings)
    results["inter_user_agreement"] = inter_user_agreement(user_pref_embeddings)
    results["krippendorff_alpha_proxy"] = krippendorff_alpha_proxy(user_pref_embeddings)

    if V is not None:
        # --- Tier 3 ---
        results["basis_activation_variance"] = basis_activation_variance(user_pref_embeddings, V)
        W = fit_user_vectors(user_pref_embeddings, V)
        results["fit_quality"] = fit_quality(user_pref_embeddings, V)
        results["user_vector_diversity"] = user_vector_diversity(W)
        results["basis_utilization_entropy"] = basis_utilization_entropy(W)

        # --- Tier 5 ---
        results["held_out_accuracy"] = held_out_accuracy(user_pref_embeddings, V)
        results["silhouette_score"] = silhouette_score_metric(W)

    _print_summary(results, K)
    return results


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"


def _flag(value: float, green_range: tuple, yellow_range: tuple) -> str:
    lo_g, hi_g = green_range
    lo_y, hi_y = yellow_range
    if lo_g <= value <= hi_g:
        return f"{_GREEN}OK{_RESET}"
    elif lo_y <= value <= hi_y:
        return f"{_YELLOW}WARN{_RESET}"
    return f"{_RED}FAIL{_RESET}"


def _print_summary(results: dict, K: int) -> None:
    rows = []

    ad = results["annotation_density"]
    flag = f"{_GREEN}OK{_RESET}" if not ad["warn"] else f"{_YELLOW}WARN{_RESET}"
    rows.append(("T0", "annotation_density", f"median={ad['median_pairs']:.1f} pairs/user", flag))

    if "prompt_diversity" in results:
        pd_ = results["prompt_diversity"]
        flag = _flag(pd_["n_unique_prompts"], (10, 1e9), (1, 9))
        rows.append(("T0", "prompt_diversity", f"n_unique={pd_['n_unique_prompts']}", flag))

    lb = results["label_balance"]
    flag = _flag(lb["mean_consistency"], (0.05, 0.95), (0.01, 0.99))
    rows.append(("T1", "label_balance", f"mean_consistency={lb['mean_consistency']:.3f}", flag))

    er = results["effective_rank"]
    flag = _flag(er["compression_ratio"], (0, 0.5), (0, 0.8))
    rows.append(("T1", "effective_rank", f"rank={er['effective_rank']} / {er['n_users']} users", flag))

    ia = results["inter_user_agreement"]
    rows.append(("T1", "inter_user_agreement", f"mean_sim={ia['mean_pairwise_similarity']:.3f}", ""))

    kap = results["krippendorff_alpha_proxy"]
    flag = _flag(kap["variance_ratio"], (0.01, 1.0), (0.001, 1.0))
    rows.append(("T1", "krippendorff_proxy", f"variance_ratio={kap['variance_ratio']:.4f}", flag))

    if "basis_activation_variance" in results:
        bav = results["basis_activation_variance"]
        flag = _flag(bav["fraction_active"], (0.5, 1.0), (0.25, 1.0))
        rows.append(("T3", "basis_activation_var", f"{bav['n_active_bases']}/{bav['n_total_bases']} active", flag))

        fq = results["fit_quality"]
        flag = _flag(fq["mean_accuracy"], (0.55, 1.0), (0.52, 1.0))
        rows.append(("T3", "fit_quality", f"acc={fq['mean_accuracy']:.3f}", flag))

        uvd = results["user_vector_diversity"]
        rows.append(("T3", "user_vec_diversity", f"mean_dist={uvd['mean_pairwise_distance']:.3f}, eff_rank={uvd['effective_rank']}", ""))

        bue = results["basis_utilization_entropy"]
        rows.append(("T3", "basis_entropy", f"norm_entropy={bue['normalized_mean_entropy']:.3f}", ""))

        hoa = results["held_out_accuracy"]
        flag = _flag(hoa["mean_accuracy"], (0.55, 1.0), (0.52, 1.0))
        rows.append(("T5", "held_out_accuracy", f"acc={hoa['mean_accuracy']:.3f} (n={hoa['n_users_evaluated']})", flag))

        ss = results["silhouette_score"]
        flag = _flag(ss["silhouette_score"], (0.0, 1.0), (-0.1, 0.0))
        rows.append(("T5", "silhouette_score", f"score={ss['silhouette_score']:.3f}", flag))

    print("\n=== LoRe Suitability Report ===")
    print(f"{'Tier':<5} {'Metric':<25} {'Value':<45} {'Status'}")
    print("-" * 85)
    for tier, metric, value, status in rows:
        print(f"{tier:<5} {metric:<25} {value:<45} {status}")
    print()
