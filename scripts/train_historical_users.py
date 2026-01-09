#!/usr/bin/env python3
"""
Train historical user vectors using HistLlama preference generation.

This script:
1. Loads trained LoRe bases (V matrix) - frozen
2. Selects a subset of questions using the question selection lever
3. Generates preferences using HistLlama models
4. Trains new user vectors (W) for historical users

Usage:
    python scripts/train_historical_users.py --century C013
    python scripts/train_historical_users.py --century C017 --n_questions 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apa.config import (
    configure_environment,
    DatasetConfig,
    HistLlamaConfig,
    LoReConfig,
    CHECKPOINTS_DIR,
)
from apa.data.prism_loader import load_prism_pairwise
from apa.utils.embedding_utils import load_embeddings, embed_texts, get_embedding_model
from apa.reward.lore_model import LoReRewardModel
from apa.historical.hist_llama import load_hist_llama, century_to_name
from apa.historical.preference_gen import (
    generate_historical_preferences,
    preferences_to_labels,
)
from apa.levers import lever_select_questions


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train historical user vectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--century",
        type=str,
        required=True,
        choices=HistLlamaConfig.VALID_CENTURIES,
        help="Century to train (e.g., C013)",
    )
    parser.add_argument(
        "--n_questions",
        type=int,
        default=500,
        help="Number of questions to use for training",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of times to run each preference comparison",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="8B",
        choices=["8B", "70B"],
        help="HistLlama model size",
    )
    parser.add_argument(
        "--user_profile",
        type=str,
        default=None,
        help="Optional user profile description to condition preferences",
    )
    parser.add_argument(
        "--lore_checkpoint",
        type=str,
        default=None,
        help="Path to trained LoRe model checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs for training user vector",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for user vector training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    return parser.parse_args()


def train_user_vector(
    V: torch.Tensor,
    embeddings_1: torch.Tensor,
    embeddings_2: torch.Tensor,
    labels: torch.Tensor,
    rank: int,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Train a single user vector given frozen basis V.

    Args:
        V: Frozen basis matrix (embed_dim, rank)
        embeddings_1: Response 1 embeddings (n, embed_dim)
        embeddings_2: Response 2 embeddings (n, embed_dim)
        labels: Binary labels (n,) - 1 if response 2 preferred
        rank: Rank of the low-rank decomposition
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on

    Returns:
        Trained user vector (rank,)
    """
    # Initialize user vector (must be leaf tensor for optimizer)
    w = torch.randn(rank, device=device) * 0.01
    w = w.clone().detach().requires_grad_(True)

    # Move data to device
    V = V.to(device)
    embeddings_1 = embeddings_1.to(device)
    embeddings_2 = embeddings_2.to(device)
    labels = labels.float().to(device)

    optimizer = torch.optim.Adam([w], lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Compute rewards: r = embed @ V @ w
        r1 = embeddings_1 @ V @ w
        r2 = embeddings_2 @ V @ w

        # Bradley-Terry logits: P(2 > 1) = sigmoid(r2 - r1)
        logits = r2 - r1

        # Binary cross entropy loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction='mean'
        )

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                preds = (logits > 0).float()
                acc = (preds == labels).float().mean()
                print(f"  Epoch {epoch + 1}/{epochs}: loss={loss.item():.4f}, acc={acc.item():.4f}")

    return w.detach().cpu()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Configure environment
    configure_environment()
    dataset_config = DatasetConfig()

    print(f"\n{'='*60}")
    print(f"Training Historical User for {century_to_name(args.century)}")
    print(f"{'='*60}\n")

    # Set up paths
    output_dir = Path(args.output_dir) if args.output_dir else dataset_config.checkpoints_dir
    output_dir = output_dir / "historical"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load LoRe model
    if args.lore_checkpoint:
        lore_path = Path(args.lore_checkpoint)
    else:
        lore_path = dataset_config.checkpoints_dir / "lore_K8_best.pt"

    if not lore_path.exists():
        print(f"ERROR: LoRe checkpoint not found at {lore_path}")
        print("Please train LoRe first using: python scripts/train_lore_prism.py")
        sys.exit(1)

    print(f"Loading LoRe model from {lore_path}")
    lore_model = LoReRewardModel.load(str(lore_path), device='cpu')
    V = lore_model.V.data.clone()
    rank = lore_model.rank
    embed_dim = lore_model.embed_dim

    print(f"LoRe model: embed_dim={embed_dim}, rank={rank}")

    # Load PRISM questions
    df = load_prism_pairwise()
    print(f"Loaded {len(df)} PRISM questions")

    # Select subset using lever
    config = {'questions': 'random_subset', 'seed': 42}
    selected_df = lever_select_questions(df, args.n_questions, config)
    print(f"Selected {len(selected_df)} questions for training")

    # Convert to list of dicts
    questions = [
        {
            'question_id': row['question_id'],
            'prompt': row['prompt'],
            'response_1': row['response_1'],
            'response_2': row['response_2'],
        }
        for _, row in selected_df.iterrows()
    ]

    # Load HistLlama model
    print(f"\nLoading HistLlama {args.model_size} for {args.century}...")
    hist_model, hist_tokenizer = load_hist_llama(
        century=args.century,
        size=args.model_size,
    )

    # Generate preferences
    print(f"\nGenerating preferences with {args.n_runs} run(s) per question...")
    hist_config = HistLlamaConfig(size=args.model_size, century=args.century)

    preferences = generate_historical_preferences(
        hist_model,
        hist_tokenizer,
        questions,
        config=hist_config,
        user_profile=args.user_profile,
        n_runs=args.n_runs,
        show_progress=True,
    )

    # Convert to labels
    labels = preferences_to_labels(preferences, as_binary=True)

    # Filter out invalid labels
    valid_mask = [l != -1 for l in labels]
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    labels = [labels[i] for i in valid_indices]
    valid_questions = [questions[i] for i in valid_indices]

    print(f"\nValid preferences: {len(labels)} / {len(preferences)}")

    # Compute consistency
    consistencies = [p['consistency'] for p in preferences]
    avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 0
    print(f"Average consistency: {avg_consistency:.2%}")

    # Generate embeddings for the questions
    print("\nGenerating embeddings...")
    embedding_model = get_embedding_model()

    prompts = [q['prompt'] for q in valid_questions]
    responses_1 = [f"{q['prompt']}\n\n{q['response_1']}" for q in valid_questions]
    responses_2 = [f"{q['prompt']}\n\n{q['response_2']}" for q in valid_questions]

    embeddings_1 = embed_texts(responses_1, model=embedding_model, show_progress=False)
    embeddings_2 = embed_texts(responses_2, model=embedding_model, show_progress=False)

    embeddings_1 = torch.tensor(embeddings_1, dtype=torch.float32)
    embeddings_2 = torch.tensor(embeddings_2, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # Train user vector
    print(f"\nTraining user vector for {args.century}...")
    w = train_user_vector(
        V=V,
        embeddings_1=embeddings_1,
        embeddings_2=embeddings_2,
        labels=labels_tensor,
        rank=rank,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )

    # Save results
    user_id = f"historical_{args.century}"
    if args.user_profile:
        user_id += f"_{hash(args.user_profile) % 10000}"

    output_path = output_dir / f"W_{user_id}.pt"
    torch.save({
        'user_id': user_id,
        'century': args.century,
        'user_profile': args.user_profile,
        'w': w,
        'n_questions': len(valid_questions),
        'consistency': avg_consistency,
    }, output_path)

    print(f"\nSaved user vector to {output_path}")

    # Also save preferences for analysis
    prefs_path = output_dir / f"preferences_{user_id}.json"
    with open(prefs_path, 'w') as f:
        json.dump(preferences, f, indent=2)
    print(f"Saved preferences to {prefs_path}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
