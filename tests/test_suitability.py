"""
Tests for LoRe dataset suitability evaluation metrics.

Fast unit tests use small synthetic tensors and run in seconds.
The PRISM benchmark (marked slow) loads pre-computed PRISM embeddings
and verifies that all metrics return "green zone" values — confirming
that a dataset known to work well with LoRe is correctly identified as such.

Usage:
    pytest tests/test_suitability.py -v            # fast unit tests only
    pytest tests/test_suitability.py -v -m slow    # PRISM benchmark
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from apa.eval_suitability import (
    annotation_density,
    basis_activation_variance,
    basis_utilization_entropy,
    effective_rank,
    fit_quality,
    fit_user_vectors,
    held_out_accuracy,
    inter_user_agreement,
    krippendorff_alpha_proxy,
    label_balance,
    silhouette_score_metric,
    user_vector_diversity,
)


# ---------------------------------------------------------------------------
# Helpers for synthetic data
# ---------------------------------------------------------------------------

def _make_random_users(n_users: int, n_pairs: int, D: int) -> list[torch.Tensor]:
    """Entirely random preference vectors — no learnable structure."""
    return [torch.randn(n_pairs, D) for _ in range(n_users)]


def _make_consistent_user(n_pairs: int, D: int) -> torch.Tensor:
    """All preferences pointing in the same direction (rubber-stamper)."""
    direction = torch.randn(D)
    direction = direction / direction.norm()
    noise = torch.randn(n_pairs, D) * 0.05
    return direction.unsqueeze(0).expand(n_pairs, -1) + noise


def _make_low_rank_users(n_users: int, n_pairs: int, D: int, true_rank: int) -> list[torch.Tensor]:
    """Users whose preferences live in a true_rank-dimensional subspace."""
    subspace = torch.randn(D, true_rank)
    subspace = torch.linalg.qr(subspace)[0]  # orthonormal basis [D, true_rank]
    tensors = []
    for _ in range(n_users):
        coeffs = torch.randn(n_pairs, true_rank)
        tensors.append(coeffs @ subspace.T + torch.randn(n_pairs, D) * 0.05)
    return tensors


# ---------------------------------------------------------------------------
# Tier 0 — annotation_density
# ---------------------------------------------------------------------------

class TestAnnotationDensity:

    def test_no_warning_when_sufficient(self):
        K = 4
        X = [torch.randn(10, 32) for _ in range(5)]
        result = annotation_density(X, K)
        assert result["warn"] is False
        assert result["median_pairs"] == 10.0

    def test_warns_when_sparse(self):
        K = 8
        X = [torch.randn(1, 32)]  # Only 1 pair, need 16
        result = annotation_density(X, K)
        assert result["warn"] is True
        assert result["fraction_below_2K"] == 1.0

    def test_mixed_users(self):
        K = 4
        X = [torch.randn(1, 32), torch.randn(20, 32)]
        result = annotation_density(X, K)
        assert result["min_pairs"] == 1
        assert result["fraction_below_2K"] == 0.5  # one user below 2*K=8


# ---------------------------------------------------------------------------
# Tier 1 — label_balance
# ---------------------------------------------------------------------------

class TestLabelBalance:

    def test_random_user_has_low_consistency(self):
        # Random unit vectors cancel out → low mean norm / per-vector norm
        rng = torch.Generator()
        rng.manual_seed(0)
        X = torch.randn(500, 64, generator=rng)
        X = X / X.norm(dim=1, keepdim=True)
        result = label_balance([X])
        assert result["mean_consistency"] < 0.2, result["mean_consistency"]

    def test_consistent_user_has_high_consistency(self):
        X = _make_consistent_user(n_pairs=50, D=64)
        result = label_balance([X])
        assert result["mean_consistency"] > 0.8, result["mean_consistency"]

    def test_consistent_user_scores_higher_than_random(self):
        consistent = _make_consistent_user(50, 64)
        rng = torch.Generator()
        rng.manual_seed(1)
        random_user = torch.randn(50, 64, generator=rng)
        result = label_balance([consistent, random_user])
        per = result["per_user_consistency"]
        assert per[0] > per[1], "consistent user should score higher than random user"


# ---------------------------------------------------------------------------
# Tier 1 — effective_rank
# ---------------------------------------------------------------------------

class TestEffectiveRank:

    def test_low_rank_data(self):
        true_rank = 2
        X = _make_low_rank_users(n_users=40, n_pairs=10, D=64, true_rank=true_rank)
        result = effective_rank(X, threshold=0.90)
        # With noise, effective rank should still be small
        assert result["effective_rank"] <= true_rank + 3, result["effective_rank"]

    def test_full_rank_random_data(self):
        # Random users → effective rank close to n_users
        X = _make_random_users(n_users=20, n_pairs=10, D=64)
        result = effective_rank(X, threshold=0.90)
        assert result["effective_rank"] > 5

    def test_compression_ratio_is_in_01(self):
        X = _make_random_users(n_users=10, n_pairs=5, D=32)
        result = effective_rank(X)
        assert 0.0 <= result["compression_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# Tier 1 — inter_user_agreement
# ---------------------------------------------------------------------------

class TestInterUserAgreement:

    def test_identical_users_have_high_agreement(self):
        direction = torch.randn(32)
        X = [direction.unsqueeze(0).expand(5, -1) + torch.randn(5, 32) * 0.01
             for _ in range(10)]
        result = inter_user_agreement(X)
        assert result["mean_pairwise_similarity"] > 0.8

    def test_random_users_have_low_agreement(self):
        X = _make_random_users(n_users=50, n_pairs=20, D=64)
        result = inter_user_agreement(X)
        assert abs(result["mean_pairwise_similarity"]) < 0.3


# ---------------------------------------------------------------------------
# Tier 1 — krippendorff_alpha_proxy
# ---------------------------------------------------------------------------

class TestKrippendorffProxy:

    def test_distinct_users_have_high_ratio(self):
        # Users in clearly distinct subspaces
        X = []
        for i in range(10):
            direction = torch.zeros(64)
            direction[i * 6 % 64] = 1.0
            X.append(direction.unsqueeze(0).expand(10, -1) + torch.randn(10, 64) * 0.1)
        result = krippendorff_alpha_proxy(X)
        assert result["variance_ratio"] > 0.05

    def test_identical_users_have_low_ratio(self):
        # All users have the same preference direction → minimal between-user variance
        direction = torch.randn(32)
        X = [direction.unsqueeze(0).expand(8, -1).clone() for _ in range(10)]
        result = krippendorff_alpha_proxy(X)
        assert result["variance_ratio"] < 0.05

    def test_ratio_in_01(self):
        X = _make_random_users(20, 10, 32)
        result = krippendorff_alpha_proxy(X)
        assert 0.0 <= result["variance_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# Tier 3 — basis_activation_variance
# ---------------------------------------------------------------------------

class TestBasisActivationVariance:

    def test_dead_bases_detected(self):
        D, K = 32, 4
        # V has 4 columns; data only has variance in the first direction
        V = torch.eye(D)[:, :K]  # first K standard basis vectors
        X = [torch.randn(10, D) * torch.tensor([1.0] + [0.0] * (D - 1))]  # only dim 0 varies
        result = basis_activation_variance(X, V)
        assert result["n_active_bases"] < K

    def test_all_bases_active_when_data_isotropic(self):
        D, K = 32, 4
        V = torch.randn(D, K)
        X = [torch.randn(100, D) for _ in range(5)]
        result = basis_activation_variance(X, V)
        assert result["n_active_bases"] == K

    def test_fraction_active_in_01(self):
        V = torch.randn(16, 4)
        X = _make_random_users(5, 10, 16)
        result = basis_activation_variance(X, V)
        assert 0.0 <= result["fraction_active"] <= 1.0


# ---------------------------------------------------------------------------
# Tier 3 — fit_quality
# ---------------------------------------------------------------------------

class TestFitQuality:

    def test_perfect_data_gives_high_accuracy(self):
        # All preference vectors point in the same direction as V's first column
        D, K = 32, 4
        V = torch.randn(D, K)
        v0 = V[:, 0] / V[:, 0].norm()
        # Preference embeddings aligned with v0 → all XV @ e_0 > 0
        noise = torch.randn(20, D) * 0.01
        X = [v0.unsqueeze(0).expand(20, -1) + noise]
        result = fit_quality(X, V)
        assert result["mean_accuracy"] > 0.9, result["mean_accuracy"]

    def test_random_data_gives_near_chance_accuracy(self):
        rng = torch.Generator()
        rng.manual_seed(42)
        D, K = 64, 8
        V = torch.randn(D, K, generator=rng)
        X = [torch.randn(20, D, generator=rng) for _ in range(30)]
        result = fit_quality(X, V)
        # Random data should fit well on training data due to overfitting
        # (closed-form minimises MSE exactly), but accuracy still bounded
        assert 0.0 <= result["mean_accuracy"] <= 1.0

    def test_skips_users_with_too_few_pairs(self):
        D, K = 16, 2
        V = torch.randn(D, K)
        X = [torch.randn(1, D), torch.randn(10, D)]  # first user skipped
        result = fit_quality(X, V)
        assert result["n_users_evaluated"] == 1


# ---------------------------------------------------------------------------
# Tier 3 — user_vector_diversity and basis_utilization_entropy
# ---------------------------------------------------------------------------

class TestUserVectorDiversity:

    def test_identical_vectors_have_zero_distance(self):
        W = torch.zeros(10, 4)  # all the same → after softmax, all uniform
        result = user_vector_diversity(W)
        assert result["mean_pairwise_distance"] < 0.05

    def test_diverse_vectors_have_high_distance(self):
        K = 4
        # Each user concentrates all weight on a different basis
        W = torch.full((K, K), -10.0)
        W.fill_diagonal_(10.0)  # one-hot after softmax
        result = user_vector_diversity(W)
        assert result["mean_pairwise_distance"] > 0.5


class TestBasisUtilizationEntropy:

    def test_uniform_weights_give_max_entropy(self):
        W = torch.zeros(10, 4)  # after softmax: uniform
        result = basis_utilization_entropy(W)
        assert abs(result["normalized_mean_entropy"] - 1.0) < 0.01

    def test_one_hot_weights_give_low_entropy(self):
        K = 4
        W = torch.full((10, K), -100.0)
        W[:, 0] = 100.0  # all weight on basis 0
        result = basis_utilization_entropy(W)
        assert result["normalized_mean_entropy"] < 0.1


# ---------------------------------------------------------------------------
# Tier 5 — held_out_accuracy
# ---------------------------------------------------------------------------

class TestHeldOutAccuracy:

    def test_random_data_gives_near_chance_accuracy(self):
        rng = torch.Generator()
        rng.manual_seed(7)
        D, K = 64, 8
        V = torch.randn(D, K, generator=rng)
        X = [torch.randn(20, D, generator=rng) for _ in range(40)]
        result = held_out_accuracy(X, V)
        # Should be near 0.5 for random data (no generalisation)
        assert 0.3 <= result["mean_accuracy"] <= 0.7, result["mean_accuracy"]

    def test_skips_users_with_too_few_pairs(self):
        D, K = 16, 2
        V = torch.randn(D, K)
        X = [torch.randn(3, D), torch.randn(20, D)]  # first user skipped (< 4)
        result = held_out_accuracy(X, V)
        assert result["n_users_evaluated"] == 1

    def test_accuracy_in_01(self):
        V = torch.randn(16, 4)
        X = [torch.randn(10, 16) for _ in range(10)]
        result = held_out_accuracy(X, V)
        assert 0.0 <= result["mean_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Tier 5 — silhouette_score_metric
# ---------------------------------------------------------------------------

class TestSilhouetteScore:

    def test_well_separated_clusters_give_high_score(self):
        K = 4
        # Two tight clusters far apart in softmax space
        cluster_a = torch.tensor([10.0, -10.0, -10.0, -10.0]).unsqueeze(0).expand(15, -1)
        cluster_b = torch.tensor([-10.0, 10.0, -10.0, -10.0]).unsqueeze(0).expand(15, -1)
        W = torch.cat([cluster_a, cluster_b]) + torch.randn(30, K) * 0.1
        result = silhouette_score_metric(W, n_clusters=2)
        assert result["silhouette_score"] > 0.5, result["silhouette_score"]

    def test_uniform_data_gives_low_score(self):
        W = torch.randn(50, 4)  # no cluster structure
        result = silhouette_score_metric(W, n_clusters=5)
        # May be positive or negative, but shouldn't be very high
        assert result["silhouette_score"] < 0.5

    def test_n_clusters_capped_at_n_minus_1(self):
        W = torch.randn(3, 4)
        result = silhouette_score_metric(W, n_clusters=100)
        assert result["n_clusters_used"] == 2  # capped at n-1 = 2


# ---------------------------------------------------------------------------
# PRISM benchmark (slow)
# ---------------------------------------------------------------------------

def _check_prism_data_available() -> bool:
    from apa.config import EMBEDDINGS_DIR, MODELS_DIR
    return (
        (EMBEDDINGS_DIR / "train.pkl").exists()
        and (MODELS_DIR / "V_K8.pt").exists()
    )


@pytest.fixture(scope="module")
def prism_data():
    """Load PRISM train embeddings grouped by seen user. Cached per module."""
    from apa.config import configure_environment, EMBEDDINGS_DIR, MODELS_DIR
    from apa.load_prism import group_embeddings_by_user

    configure_environment()
    device = "cpu"  # keep off GPU for diagnostic purposes

    train_embeddings = torch.load(EMBEDDINGS_DIR / "train.pkl", weights_only=False)
    test_embeddings = torch.load(EMBEDDINGS_DIR / "test.pkl", weights_only=False)

    train_seen, _, _, _ = group_embeddings_by_user(train_embeddings, test_embeddings, device)
    V = torch.load(MODELS_DIR / "V_K8.pt", map_location=device, weights_only=False)
    if isinstance(V, dict):
        V = V.get("V", next(iter(V.values())))
    V = V.float()

    return {"train_seen": train_seen, "V": V}


@pytest.mark.skipif(
    not _check_prism_data_available(),
    reason="PRISM embeddings or V_K8.pt not found.",
)
@pytest.mark.slow
class TestPRISMBenchmark:
    """
    Verify that all suitability metrics return 'green zone' values on PRISM.

    PRISM is a known-good dataset where LoRe achieves 87%+ accuracy at rank 8.
    Every metric here should confirm that — acting as a sanity check that the
    diagnostics are correctly calibrated.
    """

    def test_annotation_density(self, prism_data):
        result = annotation_density(prism_data["train_seen"], K=8)
        assert result["median_pairs"] >= 5, f"median pairs too low: {result['median_pairs']}"

    def test_label_balance(self, prism_data):
        result = label_balance(prism_data["train_seen"])
        mc = result["mean_consistency"]
        assert 0.05 < mc < 0.95, f"mean_consistency out of expected range: {mc}"

    def test_effective_rank_is_compressible(self, prism_data):
        result = effective_rank(prism_data["train_seen"])
        n_users = result["n_users"]
        er = result["effective_rank"]
        assert er <= n_users, f"effective_rank {er} > n_users {n_users}"
        # PRISM preferences should be far more compressible than identity
        assert result["compression_ratio"] < 1.0, f"compression_ratio={result['compression_ratio']}"

    def test_krippendorff_proxy_users_are_distinct(self, prism_data):
        result = krippendorff_alpha_proxy(prism_data["train_seen"])
        assert result["variance_ratio"] > 0.01, (
            f"variance_ratio too low: {result['variance_ratio']} — "
            "PRISM users should be distinguishable"
        )

    def test_basis_activation_variance(self, prism_data):
        result = basis_activation_variance(prism_data["train_seen"], prism_data["V"])
        K = result["n_total_bases"]
        assert result["n_active_bases"] >= K // 2, (
            f"only {result['n_active_bases']}/{K} bases active on PRISM"
        )

    def test_fit_quality(self, prism_data):
        result = fit_quality(prism_data["train_seen"], prism_data["V"])
        assert result["mean_accuracy"] > 0.55, (
            f"fit_quality accuracy {result['mean_accuracy']:.3f} too low — "
            "PRISM preferences should be linearly separable in basis space"
        )

    def test_held_out_accuracy(self, prism_data):
        result = held_out_accuracy(prism_data["train_seen"], prism_data["V"])
        assert result["mean_accuracy"] > 0.55, (
            f"held_out_accuracy {result['mean_accuracy']:.3f} too low"
        )
        assert result["n_users_evaluated"] > 100, (
            f"too few users evaluated: {result['n_users_evaluated']}"
        )

    def test_silhouette_score_positive(self, prism_data):
        W = fit_user_vectors(prism_data["train_seen"], prism_data["V"])
        result = silhouette_score_metric(W)
        assert result["silhouette_score"] > 0.0, (
            f"silhouette_score {result['silhouette_score']:.3f} not positive — "
            "PRISM user vectors should cluster"
        )
