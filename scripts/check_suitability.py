#!/usr/bin/env python
"""
Report LoRe dataset suitability scores against green/warn/fail thresholds.

Usage (as a module):
    from scripts.check_suitability import report
    report("My dataset", user_pref_embeddings, V, K=8)

Each row shows the metric value, the threshold it must clear, and a status:
  PASS  — comfortably in the green zone
  WARN  — borderline; LoRe may work but results could be unreliable
  FAIL  — likely too noisy/sparse/misaligned for LoRe to learn well
  INFO  — no hard threshold; shown for context
"""

from __future__ import annotations
import torch
from apa.eval_suitability import (
    annotation_density, label_balance, effective_rank,
    krippendorff_alpha_proxy, basis_activation_variance,
    fit_user_vectors, held_out_accuracy, silhouette_score_metric,
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
        Dict of raw metric results (same as evaluate_suitability).
    """
    V = V.float()

    # --- compute ---
    ad  = annotation_density(user_pref_embeddings, K)
    lb  = label_balance(user_pref_embeddings)
    er  = effective_rank(user_pref_embeddings)
    kap = krippendorff_alpha_proxy(user_pref_embeddings)
    bav = basis_activation_variance(user_pref_embeddings, V)
    W   = fit_user_vectors(user_pref_embeddings, V)
    hoa = held_out_accuracy(user_pref_embeddings, V)
    ss  = silhouette_score_metric(W)
    uvd = user_vector_diversity(W)
    bue = basis_utilization_entropy(W)

    # --- thresholds ---
    rows = [
        # (tier, metric, display_value, threshold_label, green_range, warn_range)
        ("T0", "annotation density",
            f"{ad['median_pairs']:.0f} pairs/user",
            "median ≥ 5",
            ad["median_pairs"],   (5, 1e9),  (2, 4.9)),
        ("T1", "label balance",
            f"{lb['mean_consistency']:.3f} mean consistency",
            "0.05 – 0.95",
            lb["mean_consistency"], (0.05, 0.95), (0.01, 0.99)),
        ("T1", "effective rank",
            f"{er['effective_rank']} / {er['n_users']} users  (ratio {er['compression_ratio']:.2f})",
            "ratio < 1.0",
            er["compression_ratio"], (0, 0.99), (0, 0.999)),
        ("T1", "Krippendorff proxy",
            f"{kap['variance_ratio']:.4f} between/total var",
            "> 0.01",
            kap["variance_ratio"], (0.01, 1.0), (0.001, 0.0099)),
        ("T3", "basis activation",
            f"{bav['n_active_bases']} / {bav['n_total_bases']} bases active",
            f"≥ {K // 2} active",
            bav["n_active_bases"],  (K // 2, K), (1, K // 2 - 1)),
        ("T5", "held-out accuracy",
            f"{hoa['mean_accuracy']:.3f}  (n={hoa['n_users_evaluated']})",
            "> 0.55",
            hoa["mean_accuracy"], (0.55, 1.0), (0.52, 0.5499)),
        ("T5", "silhouette score",
            f"{ss['silhouette_score']:.3f}  ({ss['n_clusters_used']} clusters)",
            "> 0.0",
            ss["silhouette_score"], (0.0, 1.0), (-0.1, -0.001)),
    ]

    info_rows = [
        ("T3", "user vec mean dist",     f"{uvd['mean_pairwise_distance']:.3f}"),
        ("T3", "basis entropy (norm)",   f"{bue['normalized_mean_entropy']:.3f}"),
    ]

    # --- print ---
    W2 = 54        # value column width (wider to fit longer strings)
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
        "annotation_density": ad, "label_balance": lb, "effective_rank": er,
        "krippendorff_proxy": kap, "basis_activation_variance": bav,
        "held_out_accuracy": hoa, "silhouette_score": ss,
        "user_vector_diversity": uvd, "basis_utilization_entropy": bue,
    }


if __name__ == "__main__":
    # Demo: run on three datasets (small PRISM, large PRISM, random)
    import sys, random
    sys.path.insert(0, __file__.replace("scripts/check_suitability.py", ""))

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
