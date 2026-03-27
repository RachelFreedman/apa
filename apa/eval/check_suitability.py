#!/usr/bin/env python
"""
Report LoRe dataset suitability scores against green/warn/fail thresholds.

Usage:
    python -m apa.eval.check_suitability path/to/prefs.jsonl
    python -m apa.eval.check_suitability path/to/prefs.parquet

Accepts a path to raw preference data in one of two formats:

  JSONL — one JSON object per line with fields:
      {"user_id": "u1", "prompt": "...", "chosen": "...", "rejected": "..."}

  Parquet (PRISM format) — with columns including prompt (list of chat dicts)
      and extra_info containing user_id, chosen_utterance, rejected_utterance.

The script loads the reward model, embeds the preferences, loads the pretrained
basis V, and runs all suitability metrics.

Each row shows the metric value, the threshold it must clear, and a status:
  PASS  — comfortably in the green zone
  WARN  — borderline; LoRe may work but results could be unreliable
  FAIL  — likely too noisy/sparse/misaligned for LoRe to learn well
  INFO  — no hard threshold; shown for context
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

from apa.eval.suitability import (
    PreferencePair,
    annotation_density, label_balance,
    krippendorff_alpha_proxy, nearest_neighbor_accuracy,
    basis_space_coherence, population_accuracy,
    embed_preferences,
    fit_user_vectors, held_out_accuracy,
    user_vector_diversity, basis_utilization_entropy,
)

_G = "\033[92m"; _Y = "\033[93m"; _R = "\033[91m"; _D = "\033[2m"; _0 = "\033[0m"


# ---------------------------------------------------------------------------
# Load raw preferences from file
# ---------------------------------------------------------------------------

def load_prefs_jsonl(path: Path) -> dict[str, list[PreferencePair]]:
    """Load preferences from JSONL (one JSON object per line).

    Expected fields: user_id, prompt, chosen, rejected.
    """
    prefs: dict[str, list[PreferencePair]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prefs[obj["user_id"]].append(
                PreferencePair(
                    prompt=obj["prompt"],
                    chosen=obj["chosen"],
                    rejected=obj["rejected"],
                )
            )
    return dict(prefs)


def load_prefs_parquet(path: Path) -> dict[str, list[PreferencePair]]:
    """Load preferences from a PRISM-format parquet file.

    Expects columns: prompt (list of chat dicts), extra_info (dict with
    user_id, chosen_utterance, rejected_utterance).
    """
    import pandas as pd

    df = pd.read_parquet(path)
    prefs: dict[str, list[PreferencePair]] = defaultdict(list)

    for _, row in df.iterrows():
        extra = row["extra_info"]
        user_id = extra["user_id"]
        chosen = extra["chosen_utterance"]
        rejected = extra.get("rejected_utterance", "")

        # rejected_utterance may be a list/array of alternatives; take first
        if isinstance(rejected, (list, tuple)):
            if len(rejected) == 0:
                continue
            rejected = rejected[0]
        elif hasattr(rejected, '__len__') and not isinstance(rejected, str):
            # numpy array
            if len(rejected) == 0:
                continue
            rejected = str(rejected[0])

        if not rejected:
            continue

        # prompt is a list/array of chat dicts, e.g. [{"role": "user", "content": "..."}]
        prompt = row["prompt"]
        if hasattr(prompt, '__iter__') and not isinstance(prompt, str):
            # Extract the user's message text from chat turns
            prompt_text = " ".join(
                turn["content"] for turn in prompt
                if isinstance(turn, dict) and turn.get("role") == "user"
            )
        else:
            prompt_text = str(prompt)

        prefs[user_id].append(
            PreferencePair(prompt=prompt_text, chosen=chosen, rejected=rejected)
        )

    return dict(prefs)


def load_prefs(path: Path) -> dict[str, list[PreferencePair]]:
    """Load preferences from a file, auto-detecting format by extension."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_prefs_jsonl(path)
    elif suffix == ".parquet":
        return load_prefs_parquet(path)
    elif suffix == ".json":
        return load_prefs_jsonl(path)  # treat .json as JSONL
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Use .jsonl or .parquet."
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

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
        # (metric, display_value, threshold_label, val, green_range, warn_range)
        ("annotation density",
            f"{ad['median_pairs']:.0f} pairs/user",
            "median >= 5",
            ad["median_pairs"],                (5, 1e9),   (2, 4.9)),
        ("label balance",
            f"{lb['mean_normalized_consistency']:.3f} norm consistency (1.0=random)",
            "> 1.3",
            lb["mean_normalized_consistency"], (1.3, 1e9), (1.1, 1.3)),
        ("Krippendorff proxy",
            f"{kap['corrected_ratio']:.4f} corrected ratio (0=random)",
            "> 0.03",
            kap["corrected_ratio"],            (0.03, 1.0), (0.01, 0.03)),
        ("NN accuracy",
            f"{nna['mean_nn_accuracy']:.3f} mean accuracy (0.5=random)",
            "> 0.6",
            nna["mean_nn_accuracy"],           (0.6, 1.0), (0.55, 0.6)),
        ("basis coherence",
            f"{bsc['corrected_ratio']:.4f} corrected ratio (0=random)",
            "> 0.03",
            bsc["corrected_ratio"],            (0.03, 1.0), (0.01, 0.03)),
        ("population accuracy",
            f"{pa['accuracy']:.3f} held-out accuracy (0.5=random)",
            "> 0.6",
            pa["accuracy"],                    (0.6, 1.0), (0.55, 0.6)),
        ("held-out accuracy",
            f"{hoa['mean_accuracy']:.3f}  (n={hoa['n_users_evaluated']})",
            "> 0.6",
            hoa["mean_accuracy"],              (0.6, 1.0), (0.55, 0.6)),
    ]

    info_rows = [
        ("user vec mean dist",   f"{uvd['mean_pairwise_distance']:.3f}"),
        ("basis entropy (norm)", f"{bue['normalized_mean_entropy']:.3f}"),
    ]

    # --- print ---
    W2 = 54        # value column width
    sep = "─" * (24 + W2 + 16 + 8)
    print(f"\n{'─'*len(sep)}")
    print(f"  Dataset: {name}  ({len(user_pref_embeddings)} users, K={K})")
    print(sep)
    print(f"  {'Metric':<22} {'Value':<{W2}} {'Threshold':<16} Status")
    print(sep)
    for metric, display, threshold, val, green, warn in rows:
        status, _ = _status(val, green, warn)
        print(f"  {metric:<22} {display:<{W2}} {threshold:<16} {status}")
    print(f"  {_D}{'─'*(len(sep)-2)}{_0}")
    for metric, display in info_rows:
        print(f"  {_D}{metric:<22} {display:<{W2}} {'—':<16} INFO{_0}")
    print(sep)

    return {
        "annotation_density": ad, "label_balance": lb,
        "krippendorff_proxy": kap, "nearest_neighbor_accuracy": nna,
        "basis_space_coherence": bsc, "population_accuracy": pa,
        "held_out_accuracy": hoa,
        "user_vector_diversity": uvd, "basis_utilization_entropy": bue,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run LoRe suitability evaluation on a preference dataset."
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to preference data (.jsonl or .parquet).",
    )
    parser.add_argument(
        "--V", "--basis",
        type=Path,
        default=None,
        dest="basis_path",
        help="Path to pretrained basis V_K*.pt. Default: auto-detect from config.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=8,
        help="Rank of the LoRe model (default: 8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for embedding model (default: cuda if available).",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        print(f"Error: {args.data_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load raw preferences ---
    print(f"Loading preferences from {args.data_path}...", flush=True)
    user_prefs = load_prefs(args.data_path)
    n_users = len(user_prefs)
    n_pairs = sum(len(v) for v in user_prefs.values())
    print(f"  {n_users} users, {n_pairs} preference pairs", flush=True)

    # --- Load model and embed ---
    print("Loading embedding model...", flush=True)
    from apa.train_lore_bases import get_embedding_model
    model, tokenizer = get_embedding_model(device=device)

    print("Embedding preferences...", flush=True)
    user_pref_embeddings = embed_preferences(user_prefs, model, tokenizer, device=device)

    # Free GPU memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Load V ---
    if args.basis_path is not None:
        V_path = args.basis_path
    else:
        from apa.config import MODELS_DIR
        V_path = MODELS_DIR / f"V_K{args.K}.pt"

    if not V_path.exists():
        print(f"Error: basis file {V_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading basis from {V_path}...", flush=True)
    V = torch.load(V_path, map_location="cpu", weights_only=False).float()

    # --- Run report ---
    name = args.data_path.stem
    report(name, user_pref_embeddings, V, K=args.K)


if __name__ == "__main__":
    main()
