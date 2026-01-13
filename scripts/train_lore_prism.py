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
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

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
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
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
    parser.add_argument(
        "--save_curves",
        action="store_true",
        help="Save training curves to JSON file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    start_time = time.time()

    # Configure environment
    configure_environment()
    dataset_config = DatasetConfig()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Print hyperparameter summary
    print(f"\n{'='*60}")
    print("LoRe Training Configuration")
    print(f"{'='*60}")
    print(f"  Rank (K):        {args.rank}")
    print(f"  Alpha:           {args.alpha}")
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Val split:       {args.val_split:.1%}")
    print(f"  Device:          {args.device}")
    print(f"  Seed:            {args.seed}")
    print(f"{'='*60}\n")

    # Set up paths
    embeddings_path = Path(args.embeddings) if args.embeddings else dataset_config.embeddings_path
    output_dir = Path(args.output_dir) if args.output_dir else dataset_config.checkpoints_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = load_embeddings(embeddings_path)

    # Check if embeddings have user_ids
    has_user_ids = 'user_ids' in embeddings and embeddings['user_ids'] is not None

    # Filter to users if specified
    if args.n_users is not None and has_user_ids:
        user_ids = embeddings['user_ids']
        unique_users = sorted(set(user_ids))[:args.n_users]
        mask = np.isin(user_ids, unique_users)

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

    # Create train/val split
    n_samples = len(dataset)
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"\nDataset statistics:")
    print(f"  Total samples:   {n_samples}")
    print(f"  Train samples:   {n_train}")
    print(f"  Val samples:     {n_val}")
    print(f"  Embedding dim:   {dataset.embedding_dim}")
    print(f"  Number of users: {dataset.n_users}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    model = LoReRewardModel(
        embed_dim=dataset.embedding_dim,
        rank=args.rank,
        n_users=dataset.n_users,
        alpha=args.alpha,
    )

    print(f"\nModel parameters:")
    print(f"  V shape: {model.V.shape}")
    print(f"  W shape: {model.W.shape}")
    n_params = model.V.numel() + model.W.numel()
    print(f"  Total:   {n_params:,} parameters")

    # Create trainer
    trainer = LoReTrainer(
        model=model,
        learning_rate=args.learning_rate,
        device=args.device,
    )

    # Training loop
    print(f"\n{'='*60}")
    print("Training Progress")
    print(f"{'='*60}")

    best_val_loss = float('inf')
    best_epoch = 0
    training_curves = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
    }

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_loss = trainer.train_epoch(train_loader)

        # Evaluate on validation set
        val_metrics = trainer.evaluate(val_loader)
        epoch_time = time.time() - epoch_start

        # Log metrics
        training_curves['train_loss'].append(train_loss)
        training_curves['val_loss'].append(val_metrics['loss'])
        training_curves['val_accuracy'].append(val_metrics['accuracy'])

        # Print progress
        print(f"Epoch {epoch + 1:2d}/{args.epochs}: "
              f"train_loss={train_loss:.4f}, "
              f"val_loss={val_metrics['loss']:.4f}, "
              f"val_acc={val_metrics['accuracy']:.4f}, "
              f"time={epoch_time:.1f}s")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch + 1
            model.save(str(output_dir / f"lore_K{args.rank}_best.pt"))

    # Training complete
    total_time = time.time() - start_time

    # Final summary
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"  Total time:      {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best epoch:      {best_epoch}")
    print(f"  Best val loss:   {best_val_loss:.4f}")
    print(f"  Best val acc:    {training_curves['val_accuracy'][best_epoch-1]:.4f}")
    print(f"  Final train loss: {training_curves['train_loss'][-1]:.4f}")
    print(f"  Final val loss:   {training_curves['val_loss'][-1]:.4f}")

    # Overfitting detection
    final_train = training_curves['train_loss'][-1]
    final_val = training_curves['val_loss'][-1]
    if final_val > final_train * 1.2:
        print(f"\n  [Warning] Possible overfitting detected:")
        print(f"    Train-val gap: {final_val - final_train:.4f}")
        print(f"    Consider: lower learning rate, more regularization, or early stopping")

    # Save final model
    model.save(str(output_dir / f"lore_K{args.rank}_final.pt"))

    # Save V and W separately
    torch.save(model.V.data, output_dir / f"V_lore_K_{args.rank}_alpha_{args.alpha}.pt")
    torch.save(model.W.data, output_dir / f"W_lore_seen_{args.rank}_{args.alpha}.pt")

    # Save user mapping
    if hasattr(dataset, 'user_to_idx'):
        with open(output_dir / "user_to_idx.json", 'w') as f:
            json.dump({str(k): v for k, v in dataset.user_to_idx.items()}, f)

    # Save training curves
    if args.save_curves:
        curves_path = output_dir / f"training_curves_K{args.rank}.json"
        curves_data = {
            'config': {
                'rank': args.rank,
                'alpha': args.alpha,
                'learning_rate': args.learning_rate,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'val_split': args.val_split,
                'n_train': n_train,
                'n_val': n_val,
                'n_users': dataset.n_users,
            },
            'curves': training_curves,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'total_time': total_time,
        }
        with open(curves_path, 'w') as f:
            json.dump(curves_data, f, indent=2)
        print(f"\nTraining curves saved to: {curves_path}")

    print(f"\nModels saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
