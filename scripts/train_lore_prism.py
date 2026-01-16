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
    # TODO: Implement


if __name__ == "__main__":
    main()
