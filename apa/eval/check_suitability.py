#!/usr/bin/env python
"""
Report LoRe dataset suitability scores against green/warn/fail thresholds.

Usage (as a module):
    from apa.eval.check_suitability import report
    report("My dataset", user_pref_embeddings, V, K=8)

Each row shows the metric value, the threshold it must clear, and a status:
  PASS  — comfortably in the green zone
  WARN  — borderline; LoRe may work but results could be unreliable
  FAIL  — likely too noisy/sparse/misaligned for LoRe to learn well
  INFO  — no hard threshold; shown for context

Metrics and their null hypotheses (random data expected value):
  label_balance       norm_consistency ≈ 1.0  for random data (threshold > 1.3)
  krippendorff_proxy  corrected_ratio  ≈ 0.0  for random data (threshold > 0.03)
  nn_accuracy         accuracy         ≈ 0.5  for random data (threshold > 0.55)
  basis_coherence     corrected_ratio  ≈ 0.0  for random data (threshold > 0.03)
  population_accuracy accuracy         ≈ 0.5  for random data (threshold > 0.55)
  held_out_accuracy   accuracy         ≈ 0.5  for random data (threshold > 0.55)
"""

from __future__ import annotations
from pathlib import Path
import torch
from apa.eval.suitability import (
    annotation_density, label_balance,
    krippendorff_alpha_proxy, nearest_neighbor_accuracy,
    basis_space_coherence, population_accuracy,
    fit_user_vectors, held_out_accuracy,
    user_vector_diversity, basis_utilization_entropy,
)

_G = "\033[92m"; _Y = "\033[93m"; _R = "\033[91m"; _D = "\033[2m"; _0 = "\033[0m"


def _status(val, green, warn=None):
    """Return (colour+label, raw) for a value given (lo, hi) green/warn ranges."""
    if green[0] <= val <= green[1]:
        return f"{_G}PASS{_0}", val
    if warn and warn[0] <= val <= warn[1]:
        return f"{_Y}WARN{_0}", val
    return f"{_R}FAIL{_0}", val


def report(name: str, user_pref_embeddings: list[torch.Tensor],
           V: torch.Tensor, K: int = 8) -> dict:
    """
    Run all suitability metrics and print a formatted report.

    Args:
        name:                 Dataset label shown in the header.
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors.
        V:                    Pretrained LoRe basis [D, K].
        K:                    Rank (for annotation density threshold).

    Returns:
        Dict of raw metric results.
    """
    V = V.float()

    # --- compute ---
    ad  = annotation_density(user_pref_embeddings, K)
    lb  = label_balance(user_pref_embeddings)
    kap = krippendorff_alpha_proxy(user_pref_embeddings)
    nna = nearest_neighbor_accuracy(user_pref_embeddings)
    bsc = basis_space_coherence(user_pref_embeddings, V)
    pa  = population_accuracy(user_pref_embeddings, V)
    W   = fit_user_vectors(user_pref_embeddings, V)
    hoa = held_out_accuracy(user_pref_embeddings, V)
    uvd = user_vector_diversity(W)
    bue = basis_utilization_entropy(W)

    # --- thresholds ---
    rows = [
        # (tier, metric, display_value, threshold_label, val, green_range, warn_range)
        ("T0", "annotation density",
            f"{ad['median_pairs']:.0f} pairs/user",
            "median ≥ 5",
            ad["median_pairs"],                (5, 1e9),   (2, 4.9)),
        ("T1", "label balance",
            f"{lb['mean_normalized_consistency']:.3f} norm consistency (1.0=random)",
            "> 1.3",
            lb["mean_normalized_consistency"], (1.3, 1e9), (1.1, 1.3)),
        ("T1", "Krippendorff proxy",
            f"{kap['corrected_ratio']:.4f} corrected ratio (0=random)",
            "> 0.03",
            kap["corrected_ratio"],            (0.03, 1.0), (0.01, 0.03)),
        ("T1", "NN accuracy",
            f"{nna['mean_nn_accuracy']:.3f} mean accuracy (0.5=random)",
            "> 0.6",
            nna["mean_nn_accuracy"],           (0.6, 1.0), (0.55, 0.6)),
        ("T3", "basis coherence",
            f"{bsc['corrected_ratio']:.4f} corrected ratio (0=random)",
            "> 0.03",
            bsc["corrected_ratio"],            (0.03, 1.0), (0.01, 0.03)),
        ("T3", "population accuracy",
            f"{pa['accuracy']:.3f} held-out accuracy (0.5=random)",
            "> 0.6",
            pa["accuracy"],                    (0.6, 1.0), (0.55, 0.6)),
        ("T5", "held-out accuracy",
            f"{hoa['mean_accuracy']:.3f}  (n={hoa['n_users_evaluated']})",
            "> 0.6",
            hoa["mean_accuracy"],              (0.6, 1.0), (0.55, 0.6)),
    ]

    info_rows = [
        ("T3", "user vec mean dist",   f"{uvd['mean_pairwise_distance']:.3f}"),
        ("T3", "basis entropy (norm)", f"{bue['normalized_mean_entropy']:.3f}"),
    ]

    # --- print ---
    W2 = 54        # value column width
    sep = "─" * (6 + 22 + W2 + 16 + 8)
    print(f"\n{'─'*len(sep)}")
    print(f"  Dataset: {name}  ({len(user_pref_embeddings)} users, K={K})")
    print(sep)
    print(f"  {'Tier':<5} {'Metric':<22} {'Value':<{W2}} {'Threshold':<16} Status")
    print(sep)
    for tier, metric, display, threshold, val, green, warn in rows:
        status, _ = _status(val, green, warn)
        print(f"  {tier:<5} {metric:<22} {display:<{W2}} {threshold:<16} {status}")
    print(f"  {_D}{'─'*(len(sep)-2)}{_0}")
    for tier, metric, display in info_rows:
        print(f"  {_D}{tier:<5} {metric:<22} {display:<{W2}} {'—':<16} INFO{_0}")
    print(sep)

    return {
        "annotation_density": ad, "label_balance": lb,
        "krippendorff_proxy": kap, "nearest_neighbor_accuracy": nna,
        "basis_space_coherence": bsc, "population_accuracy": pa,
        "held_out_accuracy": hoa,
        "user_vector_diversity": uvd, "basis_utilization_entropy": bue,
    }


if __name__ == "__main__":
    # Demo: run on three datasets (small PRISM, large PRISM, random)
    import sys, random
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from apa.config import configure_environment, EMBEDDINGS_DIR, MODELS_DIR
    from apa.load_prism import group_embeddings_by_user

    configure_environment()
    print("Loading PRISM embeddings...", flush=True)
    train_emb = torch.load(EMBEDDINGS_DIR / "train.pkl", weights_only=False)
    test_emb  = torch.load(EMBEDDINGS_DIR / "test.pkl",  weights_only=False)
    train_seen, _, _, _ = group_embeddings_by_user(train_emb, test_emb, "cpu")
    V = torch.load(MODELS_DIR / "V_K8.pt", map_location="cpu", weights_only=False).float()
    D, K = V.shape

    rng = random.Random(0)
    all_idx = list(range(len(train_seen)))
    rng.shuffle(all_idx)

    small  = [train_seen[i] for i in all_idx[:50]]
    large  = [train_seen[i] for i in all_idx[:750]]
    random_data = [torch.randn(10, D) for _ in range(200)]

    report("PRISM (small, 50 users)",   small,       V, K)
    report("PRISM (large, 750 users)",  large,       V, K)
    report("Random data (200 users)",   random_data, V, K)
