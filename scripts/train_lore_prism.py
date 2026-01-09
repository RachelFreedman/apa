#!/usr/bin/env python3
"""
Train LoRe model on PRISM dataset.

This script trains the Low-rank Reward model on PRISM pairwise preferences,
learning both shared bases (V) and user-specific vectors (W).

Usage:
    python scripts/train_lore_prism.py
    python scripts/train_lore_prism.py --rank 8 --epochs 10
    python scripts/train_lore_prism.py --n_users 50  # Limit users for testing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apa.config import configure_environment, DatasetConfig, LoReConfig, CHECKPOINTS_DIR
from apa.data.prism_loader import load_prism_pairwise, PRISMDataset
from apa.utils.embedding_utils import load_embeddings
from apa.utils.file_utils import CheckpointManager
from apa.reward.lore_model import LoReRewardModel, LoReTrainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LoRe model on PRISM dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="Rank of low-rank decomposition (K)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10000.0,
        help="Regularization coefficient",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--n_users",
        type=int,
        default=None,
        help="Limit to first N users (for testing)",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to embeddings file (uses default if not specified)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (uses default if not specified)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Configure environment
    configure_environment()
    dataset_config = DatasetConfig()

    # Set up paths
    embeddings_path = Path(args.embeddings) if args.embeddings else dataset_config.embeddings_path
    output_dir = Path(args.output_dir) if args.output_dir else dataset_config.checkpoints_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = load_embeddings(embeddings_path)

    # Check if embeddings have user_ids
    has_user_ids = 'user_ids' in embeddings and embeddings['user_ids'] is not None

    # Filter to users if specified (using user_ids from embeddings)
    if args.n_users is not None and has_user_ids:
        import numpy as np
        user_ids = embeddings['user_ids']
        unique_users = sorted(set(user_ids))[:args.n_users]
        mask = np.isin(user_ids, unique_users)

        # Filter embeddings to match
        embeddings = {
            'response_1_embeddings': embeddings['response_1_embeddings'][mask],
            'response_2_embeddings': embeddings['response_2_embeddings'][mask],
            'labels': embeddings['labels'][mask],
            'user_ids': user_ids[mask],
        }

        print(f"Filtered to {args.n_users} users, {mask.sum()} samples")
    elif args.n_users is not None:
        print(f"Warning: --n_users ignored (no user_ids in embeddings)")

    # Create dataset
    dataset = PRISMDataset(
        embeddings=embeddings,
        labels=embeddings['labels'],
        user_ids=embeddings.get('user_ids'),
    )

    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Embedding dim: {dataset.embedding_dim}")
    print(f"  Number of users: {dataset.n_users}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Create model
    model = LoReRewardModel(
        embed_dim=dataset.embedding_dim,
        rank=args.rank,
        n_users=dataset.n_users,
        alpha=args.alpha,
    )

    print(f"\nModel configuration:")
    print(f"  Rank (K): {args.rank}")
    print(f"  Alpha: {args.alpha}")
    print(f"  V shape: {model.V.shape}")
    print(f"  W shape: {model.W.shape}")

    # Create trainer
    trainer = LoReTrainer(
        model=model,
        learning_rate=args.learning_rate,
        device=args.device,
    )

    # Training loop
    print(f"\nTraining for {args.epochs} epochs on {args.device}...")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        # Train
        train_loss = trainer.train_epoch(dataloader)

        # Evaluate
        metrics = trainer.evaluate(dataloader)

        print(f"Epoch {epoch + 1}/{args.epochs}: "
              f"train_loss={train_loss:.4f}, "
              f"eval_loss={metrics['loss']:.4f}, "
              f"accuracy={metrics['accuracy']:.4f}")

        # Save best model
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            model.save(str(output_dir / f"lore_K{args.rank}_best.pt"))

    # Save final model
    model.save(str(output_dir / f"lore_K{args.rank}_final.pt"))

    # Also save V and W separately for easier access
    torch.save(model.V.data, output_dir / f"V_lore_K_{args.rank}_alpha_{args.alpha}.pt")
    torch.save(model.W.data, output_dir / f"W_lore_seen_{args.rank}_{args.alpha}.pt")

    # Save user mapping
    if hasattr(dataset, 'user_to_idx'):
        import json
        with open(output_dir / "user_to_idx.json", 'w') as f:
            json.dump({str(k): v for k, v in dataset.user_to_idx.items()}, f)

    print(f"\nTraining complete!")
    print(f"Models saved to: {output_dir}")
    print(f"Best eval loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
