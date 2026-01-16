#!/usr/bin/env python3
"""
Prepare PRISM embeddings for LoRe training.

This script loads the PRISM pairwise dataset and generates embeddings
using the Skywork-Reward model (following the LoRe paper methodology).

Usage:
    python scripts/prepare_prism_embeddings.py
    python scripts/prepare_prism_embeddings.py --n_samples 1000  # For testing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apa.config import configure_environment, DatasetConfig, LoReConfig
from apa.data.prism_loader import load_prism_pairwise
from apa.utils.embedding_utils import (
    get_embedding_model,
    embed_response_pairs,
    save_embeddings,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    lore_config = LoReConfig()

    parser = argparse.ArgumentParser(
        description="Prepare PRISM embeddings for LoRe training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Limit to first N samples (for testing)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for embedding (smaller for 8B model)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=lore_config.embedding_model,
        help="Embedding model to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (uses default if not specified)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    # TODO: Implement


if __name__ == "__main__":
    main()
