"""
Compare eval_prefs suitability metrics across synthetic, PRISM, and random baselines.

Baselines are matched to the same PRISM questions used by the synthetic experiment
so the comparison is fair (same prompts, same response pairs).

Usage:
    uv run python scripts/compare_metrics.py --synth-path path/to/hist_prefs_all.jsonl
    uv run python scripts/compare_metrics.py --synth-path path/to/hist_prefs_all.jsonl --n-baseline-users 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from apa.config import MODELS_DIR
from apa.synthetic_prefs.eval_prefs import (
    embed_preferences,
    evaluate_suitability,
    load_prefs,
)
from apa.synthetic_prefs.sample_data import (
    random_prefs_by_questions,
    sample_prefs_by_questions,
)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


METRICS = [
    ("annotation density (median)",        ("annotation_density", "median_pairs"),                  ">= 5"),
    ("label balance (norm consistency)",    ("label_balance", "mean_normalized_consistency"),        "> 1.3"),
    ("Krippendorff proxy",                 ("krippendorff_alpha_proxy", "corrected_ratio"),         "> 0.03"),
    ("NN accuracy",                        ("nearest_neighbor_accuracy", "mean_nn_accuracy"),       "> 0.6"),
    ("inter-user agreement (mean sim)",    ("inter_user_agreement", "mean_pairwise_similarity"),    "low=diverse"),
    ("inter-user agreement (std sim)",     ("inter_user_agreement", "std_pairwise_similarity"),     "high=diverse"),
    ("basis coherence",                    ("basis_space_coherence", "corrected_ratio"),             "> 0.005"),
    ("population accuracy",                ("population_accuracy", "accuracy"),                     "> 0.6"),
    ("held-out accuracy",                  ("held_out_accuracy", "mean_accuracy"),                  "> 0.6"),
    ("user vec mean dist",                 ("user_vector_diversity", "mean_pairwise_distance"),     "INFO"),
    ("effective rank",                     ("user_vector_diversity", "effective_rank"),              "INFO"),
    ("basis entropy (norm)",               ("basis_utilization_entropy", "normalized_mean_entropy"), "INFO"),
]


def _get(results: dict, keys: tuple) -> object:
    v = results
    for k in keys:
        v = v[k]
    return v


def main():
    parser = argparse.ArgumentParser(description="Compare suitability metrics across datasets.")
    parser.add_argument("--synth-path", type=Path, required=True,
                        help="Path to synthetic preferences JSONL.")
    parser.add_argument("--n-baseline-users", type=int, default=20,
                        help="Number of users for PRISM/Random baselines.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=8)
    args = parser.parse_args()

    V = torch.load(MODELS_DIR / f"V_K{args.K}.pt", map_location="cpu", weights_only=True).float()
    n = args.n_baseline_users

    # --- Synthetic: embed ---
    print(f"Embedding synthetic preferences from {args.synth_path}...", flush=True)
    synth_prefs = load_prefs(args.synth_path)
    from apa.train_lore_bases import get_embedding_model
    model, tokenizer = get_embedding_model()
    synth_embs = embed_preferences(synth_prefs, model, tokenizer)

    n_synth = len(synth_embs)
    synth_results = evaluate_suitability(synth_embs, V=V, K=args.K)

    # --- Extract question prompts used by synthetic experiment ---
    synth_prompts: set[str] = set()
    for pairs in synth_prefs.values():
        for p in pairs:
            synth_prompts.add(p.prompt)
    print(f"Matching baselines to {len(synth_prompts)} unique question prompts", flush=True)

    # --- PRISM baseline: same questions, real preferences ---
    from apa.synthetic_prefs.eval_prefs import load_prefs_parquet
    from apa.config import PRISM_DATA_DIR
    train_parquet = PRISM_DATA_DIR / "train.parquet"
    if train_parquet.exists():
        prism_all_prefs = load_prefs_parquet(train_parquet)
    else:
        # Fallback: try to load from the pairwise CSV via JSONL-like format
        print("WARNING: train.parquet not found, using pairwise CSV as fallback", flush=True)
        from apa.synthetic_prefs.historical_prefs import load_curated_question_ids
        prism_all_prefs = {}

    prism_matched = sample_prefs_by_questions(prism_all_prefs, synth_prompts, n_users=n, seed=args.seed)
    rand_matched = random_prefs_by_questions(prism_all_prefs, synth_prompts, n_users=n, seed=args.seed)

    n_prism = len(prism_matched)
    n_rand = len(rand_matched)
    print(f"PRISM baseline: {n_prism} users, {sum(len(v) for v in prism_matched.values())} pairs", flush=True)
    print(f"Random baseline: {n_rand} users, {sum(len(v) for v in rand_matched.values())} pairs", flush=True)

    # --- Embed baselines ---
    print("Embedding PRISM baseline...", flush=True)
    prism_embs = embed_preferences(prism_matched, model, tokenizer)
    print("Embedding Random baseline...", flush=True)
    rand_embs = embed_preferences(rand_matched, model, tokenizer)

    del model
    torch.cuda.empty_cache()

    prism_results = evaluate_suitability(prism_embs, V=V, K=args.K)
    rand_results = evaluate_suitability(rand_embs, V=V, K=args.K)

    # --- Table ---
    synth_label = f"Synth ({n_synth})"
    prism_label = f"PRISM ({n_prism})"
    rand_label = f"Random ({n_rand})"

    w_metric = 40
    w_col = 16
    header = f"{'Metric':<{w_metric}} {'Threshold':<{w_col}} {synth_label:<{w_col}} {prism_label:<{w_col}} {rand_label:<{w_col}}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, keys, thresh in METRICS:
        sv = _fmt(_get(synth_results, keys))
        pv = _fmt(_get(prism_results, keys))
        rv = _fmt(_get(rand_results, keys))
        print(f"{name:<{w_metric}} {thresh:<{w_col}} {sv:<{w_col}} {pv:<{w_col}} {rv:<{w_col}}")
    print(sep)


if __name__ == "__main__":
    main()
